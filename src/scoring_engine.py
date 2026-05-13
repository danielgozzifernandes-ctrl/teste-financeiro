"""
scoring_engine.py — Módulo 3

Motor de scoring multi-fator com normalização adaptativa Z-Score Híbrida.

Arquitetura de normalização:
  N ≥ 8 peers  → Z-Score Setorial  (clipe [-3,+3] → mapa [0,100])
  4 ≤ N < 8    → Percentil Setorial (scipy.stats.percentileofscore)
  N < 4        → Z-Score Global     (universo completo como referência)

Pilares de score e pesos:
  Fundamentalista  45%: Earnings Yield, P/VP, ROE, ROIC*, Dívida/EBITDA, DY
  Momentum         30%: Alpha vs IBOV em 3m, 6m, 12m
  Qualidade/Risco  25%: Volatilidade 180d, Beta, Volume médio 30d

  * ROIC: se NaN ou setor Financeiro, seu peso (20%) é somado ao ROE (25%→45%)

Direção dos fatores:
  "lower_is_better" → valores negados ANTES da normalização.
  Score sempre cresce com "qualidade": maior score = melhor para aquele fator.
  Raw value no NormDetail é sempre o valor ORIGINAL (não negado).

Output de score():
  DataFrame ordenado por total_score DESC, com métricas brutas, scores por pilar,
  campo 'why' e 'norm_details' (rastreabilidade para JSON de histórico).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    MAX_DIVIDA_EBITDA,
    MIN_DAILY_VOLUME_BRL,
    MIN_SECTOR_PERCENTILE,
    MIN_SECTOR_ZSCORE,
    MOMENTUM_WINDOWS,
    VOLATILITY_WINDOW,
    WEIGHTS,
)

logger = logging.getLogger(__name__)

FINANCIAL_SECTORS = {"Financeiro e Outros"}

# ── Regime-adaptive scoring weights ──────────────────────────────────────────
# 3-state market regime determined by IBOV vs MA200 and VIX level.
#   risk_on:  IBOV > MA200 and VIX < 18 — trending bull; favour momentum
#   bear:     VIX > 25 — global risk-off; favour quality/low-vol
#   mean_rev: everything else — range-bound; favour fundamental value
_REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "risk_on":  {"fundamental": 0.30, "momentum": 0.50, "quality": 0.20},
    "mean_rev": {"fundamental": 0.50, "momentum": 0.20, "quality": 0.30},
    "bear":     {"fundamental": 0.30, "momentum": 0.10, "quality": 0.60},
}

# Configuração dos fatores fundamentalistas com pesos base e direção
_FUNDAMENTAL_CFG: dict[str, dict] = {
    "earnings_yield": {"base_weight": 0.20, "direction": "higher_is_better", "label": "Earnings Yield"},
    "pvp":            {"base_weight": 0.15, "direction": "lower_is_better",  "label": "P/VP"},
    "roe":            {"base_weight": 0.25, "direction": "higher_is_better", "label": "ROE"},
    "roic":           {"base_weight": 0.20, "direction": "higher_is_better", "label": "ROIC"},
    "divida_ebitda":  {"base_weight": 0.10, "direction": "lower_is_better",  "label": "Dívida/EBITDA"},
    "dividend_yield": {"base_weight": 0.10, "direction": "higher_is_better", "label": "Dividend Yield"},
}

_MOMENTUM_CFG: dict[str, dict] = {
    "alpha_3m":      {"base_weight": 0.30, "direction": "higher_is_better", "label": "Alpha 3m"},
    "idio_alpha_6m": {"base_weight": 0.40, "direction": "higher_is_better", "label": "Momentum Idiossincr. 6m"},
    "alpha_12m":     {"base_weight": 0.30, "direction": "higher_is_better", "label": "Alpha 12m"},
}

_QUALITY_CFG: dict[str, dict] = {
    "volatility_180d": {"base_weight": 0.40, "direction": "lower_is_better",  "label": "Volatilidade 180d"},
    "beta":            {"base_weight": 0.35, "direction": "lower_is_better",  "label": "Beta vs IBOV"},
    "avg_volume_30d":  {"base_weight": 0.25, "direction": "higher_is_better", "label": "Volume Médio 30d"},
}

_FACTOR_LABELS = {f: cfg["label"] for d in [_FUNDAMENTAL_CFG, _MOMENTUM_CFG, _QUALITY_CFG] for f, cfg in d.items()}


# ---------------------------------------------------------------------------
# NormDetail — metadados de normalização por fator por ticker
# ---------------------------------------------------------------------------
@dataclass
class NormDetail:
    """
    Armazena o rastreio completo de como um fator foi normalizado para um ticker.

    Campos:
      method     — "zscore_sectoral" | "percentile_sectoral" | "zscore_global"
      n_peers    — N de tickers com dados válidos usados na referência
      raw_value  — valor bruto ORIGINAL (sem negação de direção)
      score      — score normalizado [0, 100]; None se dado inválido
      direction  — "higher_is_better" ou "lower_is_better" (para gerar texto correto)
      sector_mean/std — parâmetros do Z-Score (se aplicável)
      z_score    — z calculado APÓS negação de direção; z>0 sempre é bom
      percentile — percentil calculado APÓS negação de direção; pct>50 é bom
    """
    method:      str
    n_peers:     int
    raw_value:   Optional[float]
    score:       Optional[float]
    direction:   str = "higher_is_better"
    sector_mean: Optional[float] = None
    sector_std:  Optional[float] = None
    z_score:     Optional[float] = None
    percentile:  Optional[float] = None


# ---------------------------------------------------------------------------
# ScoringEngine
# ---------------------------------------------------------------------------
class ScoringEngine:
    """
    Calcula o score composto 0-100 para cada ticker do universo B3.

    Uso:
        engine = ScoringEngine()
        df_ranked = engine.score(df_fund, df_prices, ibov_prices)
    """

    def score(
        self,
        df_fund: pd.DataFrame,
        df_prices: pd.DataFrame,
        ibov_prices: pd.Series,
        regime: str = "mean_rev",
    ) -> pd.DataFrame:
        """
        Executa o pipeline completo de scoring.

        Args:
            df_fund:      DataFrame do DataCollector (uma linha por ticker)
            df_prices:    Wide DataFrame (date × ticker) de preços ajustados
            ibov_prices:  Série de preços IBOVESPA (index=date, values=preço)
            regime:       Regime de mercado detectado: "risk_on" | "mean_rev" | "bear"
                          Determina os pesos relativos dos pilares de score.

        Returns:
            DataFrame ordenado por total_score DESC com todas as colunas de score.
        """
        active_weights = _REGIME_WEIGHTS.get(regime, WEIGHTS)
        if regime not in _REGIME_WEIGHTS:
            logger.warning("Regime '%s' desconhecido — usando pesos padrão (mean_rev)", regime)
        else:
            logger.info(
                "Regime de mercado: %s → Fundamental=%.0f%% Momentum=%.0f%% Quality=%.0f%%",
                regime,
                active_weights["fundamental"] * 100,
                active_weights["momentum"] * 100,
                active_weights["quality"] * 100,
            )

        # Trabalhar com ticker como índice durante o scoring
        df = df_fund.copy()
        if "ticker" in df.columns:
            df = df.set_index("ticker")
        sector_map = df["setor"].astype(str)

        # ── Pré-processamento ─────────────────────────────────────────────
        df = self._derive_metrics(df)
        df = self._add_momentum_metrics(df, df_prices, ibov_prices)
        df = self._add_idiosyncratic_momentum(df, df_prices, ibov_prices, sector_map)
        df = self._add_quality_metrics(df, df_prices, ibov_prices)
        df = self._apply_hard_filters(df)
        sector_map = sector_map.reindex(df.index)  # re-alinhar após filtros

        # ── Scoring por pilar ─────────────────────────────────────────────
        fund_scores, fund_details = self._score_fundamental(df, sector_map)
        mom_scores,  mom_details  = self._score_momentum(df)
        qual_scores, qual_details = self._score_quality(df)

        # ── Score composto — regime-adaptive weights ──────────────────────
        df["fundamental_score"] = fund_scores.round(2)
        df["momentum_score"]    = mom_scores.round(2)
        df["quality_score"]     = qual_scores.round(2)
        df["market_regime"]     = regime
        df["total_score"] = (
            active_weights["fundamental"] * fund_scores
            + active_weights["momentum"]  * mom_scores
            + active_weights["quality"]   * qual_scores
        ).round(2)

        # ── Campos de rastreabilidade ─────────────────────────────────────
        df["norm_details"] = self._build_norm_details_col(
            fund_details, mom_details, qual_details, df.index
        )

        # Coluna simples: método predominante nos fatores fundamentalistas do ticker
        # Lógica: se qualquer fator fundamental usou zscore_sectoral → "zscore_sectoral"
        #         se todos usaram percentile_sectoral             → "percentile_sectoral"
        #         se algum caiu em zscore_global                  → "zscore_global"
        # (precedência decrescente: zscore_sectoral > percentile_sectoral > zscore_global)
        def _dominant_norm_method(ticker: str) -> str:
            methods = {
                d.method
                for d in fund_details.get(ticker, {}).values()
                if d is not None and d.method
            }
            if "zscore_sectoral"    in methods: return "zscore_sectoral"
            if "percentile_sectoral" in methods: return "percentile_sectoral"
            if "zscore_global"      in methods: return "zscore_global"
            return "unknown"

        df["normalization_method"] = pd.Series(
            {t: _dominant_norm_method(t) for t in df.index}, dtype=str
        )

        df["why"] = [
            self._explain(ticker, fund_details, mom_details)
            for ticker in df.index
        ]

        # ── Montar output ─────────────────────────────────────────────────
        ordered_cols = [
            "nome", "setor", "subsetor", "normalization_method", "data_source",
            # métricas brutas
            "earnings_yield", "pvp", "roe", "roic", "divida_ebitda", "dividend_yield",
            "alpha_3m", "alpha_6m", "idio_alpha_6m", "alpha_12m",
            "volatility_180d", "beta", "avg_volume_30d",
            "current_price", "market_cap",
            # pilares e total
            "fundamental_score", "momentum_score", "quality_score", "total_score",
            "market_regime",
            # explicabilidade
            "why", "norm_details",
        ]
        out_cols = [c for c in ordered_cols if c in df.columns]
        df_out = (
            df[out_cols]
            .sort_values("total_score", ascending=False)
            .reset_index()
            .rename(columns={"index": "ticker"})
        )
        if "ticker" not in df_out.columns and df_out.index.name == "ticker":
            df_out = df_out.reset_index()

        logger.info(
            "Scoring finalizado: %d tickers. Top 5: %s",
            len(df_out),
            df_out.head(5)["ticker"].tolist() if "ticker" in df_out.columns else [],
        )
        return df_out

    # ═══════════════════════════════════════════════════════════════════════
    # Pré-processamento de métricas
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _derive_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """
        Earnings Yield = 1 / P·L

        Vantagem vs P/L direto:
          P/L tem distribuição fortemente assimétrica (outliers em >30).
          EY tem distribuição mais próxima da normal → Z-Score mais estável.
          EY = 0.20 (P/L=5, barato) vs EY = 0.01 (P/L=100, caro) é intuitivo.
        """
        mask_valid = df["pl"].notna() & (df["pl"] > 0)
        df["earnings_yield"] = np.where(mask_valid, 1.0 / df["pl"], np.nan)
        logger.debug(
            "Earnings Yield: %d/%d tickers com dado válido",
            mask_valid.sum(), len(df),
        )
        return df

    def _add_momentum_metrics(
        self,
        df: pd.DataFrame,
        df_prices: pd.DataFrame,
        ibov_prices: pd.Series,
    ) -> pd.DataFrame:
        """
        Alpha = retorno da ação - retorno do IBOV no mesmo período.

        Usar retorno relativo (alpha) e não absoluto porque:
          1. Filtra o beta de mercado — captura alpha puro da ação
          2. Em períodos de alta do IBOV, a ação tem que performar ACIMA
             do índice para ter score de momentum alto
          3. Mesma escala independente do ambiente macro
        """
        for momentum_col, window in MOMENTUM_WINDOWS.items():
            label = momentum_col.replace("ret_", "alpha_")  # ret_3m → alpha_3m
            stock_returns = self._trailing_returns(df_prices, window)
            ibov_return   = self._scalar_return(ibov_prices, window)
            alpha = stock_returns - ibov_return
            df[label] = df.index.map(alpha)
            valid = df[label].notna().sum()
            logger.debug(
                "Momentum %s (window=%dd): %d/%d tickers | IBOV ref: %+.1f%%",
                label, window, valid, len(df), ibov_return * 100,
            )
        return df

    def _add_quality_metrics(
        self,
        df: pd.DataFrame,
        df_prices: pd.DataFrame,
        ibov_prices: pd.Series,
    ) -> pd.DataFrame:
        """
        Volatilidade e Beta calculados dos preços históricos.
        Sobrescreve beta do DataCollector apenas quando o campo está ausente.
        """
        vol_series  = self._volatility(df_prices, VOLATILITY_WINDOW)
        beta_series = self._betas(df_prices, ibov_prices)

        df["volatility_180d"] = df.index.map(vol_series)

        # Beta: usar da série de preços se o DataCollector não retornou
        if "beta" not in df.columns:
            df["beta"] = np.nan
        missing_beta = df["beta"].isna()
        if missing_beta.any():
            df.loc[missing_beta, "beta"] = df.loc[missing_beta].index.map(beta_series).astype(float)

        logger.debug(
            "Qualidade: vol OK=%d, beta OK=%d",
            df["volatility_180d"].notna().sum(),
            df["beta"].notna().sum(),
        )
        return df

    def _add_idiosyncratic_momentum(
        self,
        df: pd.DataFrame,
        df_prices: pd.DataFrame,
        ibov_prices: pd.Series,
        sector_map: pd.Series,
    ) -> pd.DataFrame:
        """
        Idiosyncratic momentum — cumulative OLS residual over 6 months.

        Model: R_acao = α + β1·R_ibov + β2·R_setor + ε

        ε captures stock-level alpha orthogonal to both market and sector
        risk. Stocks that consistently beat both benchmarks get high scores
        regardless of whether the market or sector was also rallying.

        Sector returns are computed leave-one-out (excluding the target
        ticker) to avoid perfect multicollinearity when a sector has only
        2 constituents.

        Uses 126-day (≈6 month) window; falls back to NaN if < 30 common
        observations are available after alignment.
        """
        _WINDOW = 126
        _MIN_OBS = 30

        ibov_daily = ibov_prices.dropna().pct_change().dropna()

        # Pre-compute full-sector daily return series {sector: pd.Series}
        full_sector_daily: dict[str, pd.Series] = {}
        for sector in sector_map.dropna().unique():
            s_tickers = sector_map[sector_map == sector].index.tolist()
            avail = [t for t in s_tickers if t in df_prices.columns]
            if len(avail) >= 2:
                full_sector_daily[sector] = df_prices[avail].pct_change().mean(axis=1)

        idio_col = pd.Series(np.nan, index=df.index, dtype=float)

        for ticker in df.index:
            if ticker not in df_prices.columns:
                continue
            series = df_prices[ticker].dropna()
            if len(series) < _WINDOW + 1:
                continue

            stock_daily = series.pct_change().dropna().tail(_WINDOW)
            sector = sector_map.get(ticker)

            # Leave-one-out sector return
            if sector and sector in full_sector_daily:
                s_tickers = sector_map[sector_map == sector].index.tolist()
                avail_loo = [t for t in s_tickers if t in df_prices.columns and t != ticker]
                if len(avail_loo) >= 1:
                    sec_ret = df_prices[avail_loo].pct_change().mean(axis=1)
                else:
                    sec_ret = full_sector_daily[sector]
            else:
                sec_ret = pd.Series(dtype=float)

            # Align all series on common dates
            common = stock_daily.index.intersection(ibov_daily.index)
            has_sector = not sec_ret.empty
            if has_sector:
                common = common.intersection(sec_ret.index)
            if len(common) < _MIN_OBS:
                continue

            y = stock_daily.loc[common].to_numpy(dtype=float)
            ibov_x = ibov_daily.loc[common].to_numpy(dtype=float)

            if has_sector:
                X = np.column_stack([np.ones(len(common)), ibov_x, sec_ret.loc[common].to_numpy(dtype=float)])
            else:
                X = np.column_stack([np.ones(len(common)), ibov_x])

            try:
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                residuals = y - X @ beta
                idio_col[ticker] = float(residuals.sum())
            except Exception:
                pass

        df["idio_alpha_6m"] = idio_col
        logger.debug(
            "Idiosyncratic momentum: %d/%d tickers com dado válido",
            idio_col.notna().sum(), len(df),
        )
        return df

    # ═══════════════════════════════════════════════════════════════════════
    # Filtros hard — aplicados antes do ranking
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _apply_hard_filters(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove tickers que violam critérios absolutos antes do ranking.

        Filtros:
          1. Liquidez < R$5M/dia — ação ilíquida impossibilita execução
          2. Dívida/EBITDA > 5x (fora do setor financeiro) — risco de crédito extremo
          3. data_source == "failed" — sem dados de nenhuma fonte
        """
        n_before = len(df)

        # Excluir tickers sem dados
        if "data_source" in df.columns:
            df = df[df["data_source"] != "failed"]

        # Filtro de liquidez (NaN mantido: não penaliza dados ausentes)
        if "avg_volume_30d" in df.columns:
            low_liq = df["avg_volume_30d"].notna() & (df["avg_volume_30d"] < MIN_DAILY_VOLUME_BRL)
            if low_liq.any():
                excluded = df[low_liq].index.tolist()
                logger.info("Hard filter liquidez: %s removidos", excluded)
                df = df[~low_liq]

        # Filtro de alavancagem (não aplica ao setor financeiro)
        if "divida_ebitda" in df.columns and "setor" in df.columns:
            is_financial = df["setor"].isin(FINANCIAL_SECTORS)
            over_levered = (
                ~is_financial
                & df["divida_ebitda"].notna()
                & (df["divida_ebitda"] > MAX_DIVIDA_EBITDA)
            )
            if over_levered.any():
                excluded = df[over_levered].index.tolist()
                logger.info("Hard filter alavancagem: %s removidos", excluded)
                df = df[~over_levered]

        n_after = len(df)
        if n_before != n_after:
            logger.info("Hard filters: %d→%d tickers (%d removidos)", n_before, n_after, n_before - n_after)
        return df

    # ═══════════════════════════════════════════════════════════════════════
    # Scoring por pilar
    # ═══════════════════════════════════════════════════════════════════════

    def _score_fundamental(
        self,
        df: pd.DataFrame,
        sector_map: pd.Series,
    ) -> tuple[pd.Series, dict[str, dict[str, NormDetail]]]:
        """
        Calcula score fundamentalista (0-100) com normalização adaptativa.

        Fluxo:
          1. Para cada fator: negar se lower_is_better → normalizar adaptativamente
          2. Guardar NormDetail com raw_value ORIGINAL (sem negação)
          3. Por ticker: calcular pesos dinâmicos → soma ponderada

        Returns:
            (composite_scores, details)
            details: {ticker: {fator: NormDetail}}
        """
        factor_scores_map: dict[str, pd.Series] = {}
        all_details: dict[str, dict] = {t: {} for t in df.index}

        for factor, cfg in _FUNDAMENTAL_CFG.items():
            if factor not in df.columns:
                logger.debug("Fator %s ausente no DataFrame — pulando", factor)
                continue

            values = df[factor].copy()
            # Salvar raw ANTES de negar (para restaurar no NormDetail)
            raw_values = values.copy()

            if cfg["direction"] == "lower_is_better":
                values = -values  # inverter: menor original → maior normalizado → maior score

            scores, details_map = self._normalize_adaptive(values, sector_map)
            factor_scores_map[factor] = scores

            for ticker, detail in details_map.items():
                if detail is not None:
                    detail.raw_value = (
                        float(raw_values[ticker]) if ticker in raw_values.index and not pd.isna(raw_values.get(ticker)) else None
                    )
                    detail.direction = cfg["direction"]
                all_details[ticker][factor] = detail

        # Soma ponderada per-ticker com pesos dinâmicos
        composite = pd.Series(np.nan, index=df.index)
        for ticker in df.index:
            row = df.loc[ticker]
            weights = self._fundamental_weights(row, factor_scores_map, ticker)
            if not weights:
                composite[ticker] = np.nan
                continue
            s = sum(
                w * factor_scores_map[f].get(ticker, np.nan)
                for f, w in weights.items()
                if f in factor_scores_map and not pd.isna(factor_scores_map[f].get(ticker, np.nan))
            )
            composite[ticker] = s

        return composite.clip(0, 100), all_details

    def _fundamental_weights(
        self,
        row: pd.Series,
        factor_scores_map: dict[str, pd.Series],
        ticker: str,
    ) -> dict[str, float]:
        """
        Calcula pesos ajustados dinamicamente para um ticker específico.

        Regras (em ordem de aplicação):
          1. ROIC ausente/inaplicável: remove ROIC e redistribui seu peso
             proporcionalmente entre os demais fatores ativos. Para o setor
             Financeiro, ROIC não é calculado (modelo bancário de alavancagem),
             então redistribuição é sempre aplicada. Redistribuição proporcional
             evita inflar artificialmente ROE (que antes recebia os 20% do ROIC).
          2. Outros fatores NaN: redistribuição proporcional entre ativos.
        """
        weights = {f: cfg["base_weight"] for f, cfg in _FUNDAMENTAL_CFG.items()}

        # Regra 1 — ROIC: redistribuição proporcional quando ausente
        roic_missing = (
            "roic" not in factor_scores_map
            or pd.isna(factor_scores_map["roic"].get(ticker, np.nan))
            or str(row.get("setor", "")) in FINANCIAL_SECTORS
        )
        if roic_missing:
            roic_weight = weights.pop("roic")  # 0.20
            remaining = {f: w for f, w in weights.items()}
            total_remaining = sum(remaining.values())
            if total_remaining > 0:
                for f in remaining:
                    weights[f] += roic_weight * (remaining[f] / total_remaining)

        # Regra 2 — NaN em demais fatores: redistribuição proporcional
        nan_factors = {
            f for f in list(weights.keys())
            if f not in factor_scores_map
            or pd.isna(factor_scores_map[f].get(ticker, np.nan))
        }
        active = {f: w for f, w in weights.items() if f not in nan_factors}
        total = sum(active.values())
        if total == 0:
            return {}
        return {f: w / total for f, w in active.items()}

    def _score_momentum(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.Series, dict[str, dict[str, NormDetail]]]:
        """
        Score de momentum (0-100) — Z-Score Global para todos os fatores.

        Momentum usa normalização global porque o alpha vs IBOV é
        comparável entre setores (é relativo ao benchmark, não ao setor).
        """
        factor_scores_map: dict[str, pd.Series] = {}
        all_details: dict[str, dict] = {t: {} for t in df.index}

        for factor, cfg in _MOMENTUM_CFG.items():
            if factor not in df.columns:
                continue
            values = df[factor].copy()
            raw_values = values.copy()
            # Momentum é sempre higher_is_better (alpha positivo = bom)
            scores, details_map = self._normalize_global(values)
            factor_scores_map[factor] = scores
            for ticker, detail in details_map.items():
                if detail is not None:
                    detail.raw_value = float(raw_values[ticker]) if ticker in raw_values.index and not pd.isna(raw_values.get(ticker)) else None
                    detail.direction = cfg["direction"]
                all_details[ticker][factor] = detail

        base_weights = {f: cfg["base_weight"] for f, cfg in _MOMENTUM_CFG.items()}
        composite = self._weighted_sum(df.index, base_weights, factor_scores_map)
        return composite.clip(0, 100), all_details

    def _score_quality(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.Series, dict[str, dict[str, NormDetail]]]:
        """
        Score de qualidade/risco (0-100) — Z-Score Global para todos os fatores.

        Risco é comparável entre setores (volatilidade, beta, liquidez
        são dimensões absolutas, não relativas ao setor).
        """
        factor_scores_map: dict[str, pd.Series] = {}
        all_details: dict[str, dict] = {t: {} for t in df.index}

        for factor, cfg in _QUALITY_CFG.items():
            if factor not in df.columns:
                continue
            values = df[factor].copy()
            raw_values = values.copy()
            if cfg["direction"] == "lower_is_better":
                values = -values
            scores, details_map = self._normalize_global(values)
            factor_scores_map[factor] = scores
            for ticker, detail in details_map.items():
                if detail is not None:
                    detail.raw_value = float(raw_values[ticker]) if ticker in raw_values.index and not pd.isna(raw_values.get(ticker)) else None
                    detail.direction = cfg["direction"]
                all_details[ticker][factor] = detail

        base_weights = {f: cfg["base_weight"] for f, cfg in _QUALITY_CFG.items()}
        composite = self._weighted_sum(df.index, base_weights, factor_scores_map)
        return composite.clip(0, 100), all_details

    # ═══════════════════════════════════════════════════════════════════════
    # Normalização — núcleo matemático
    # ═══════════════════════════════════════════════════════════════════════

    def _normalize_adaptive(
        self,
        values: pd.Series,
        sector_map: pd.Series,
    ) -> tuple[pd.Series, dict[str, Optional[NormDetail]]]:
        """
        Normalização adaptativa: escolhe método baseado em N de peers válidos por setor.

        IMPORTANTE: 'values' já deve estar com direção corrigida (lower_is_better negados).
        O N de peers é calculado sobre valores NÃO-NaN, não sobre o tamanho total do setor.
        Isso protege contra setores onde muitos tickers têm dado ausente para aquele fator.

        Mapeamento Z-Score → [0,100]:
          z clipped em [-3, +3] → score = (z + 3) / 6 * 100
          z = -3 → score = 0   (pior possível dentro do universo)
          z =  0 → score = 50  (mediana do setor)
          z = +3 → score = 100 (melhor possível dentro do universo)
        """
        scores  = pd.Series(np.nan, index=values.index, dtype=float)
        details: dict[str, Optional[NormDetail]] = {t: None for t in values.index}

        global_fallback_tickers: list[str] = []

        for sector in sector_map.dropna().unique():
            sector_idx     = sector_map[sector_map == sector].index
            sector_values  = values.reindex(sector_idx).dropna()
            n              = len(sector_values)

            if n >= MIN_SECTOR_ZSCORE:
                # ── Z-Score Setorial ────────────────────────────────────
                mu    = float(sector_values.mean())
                sigma = float(sector_values.std(ddof=1)) if n > 1 else 0.0

                for ticker in sector_idx:
                    v = values.get(ticker, np.nan)
                    if pd.isna(v):
                        details[ticker] = NormDetail(
                            method="zscore_sectoral", n_peers=n,
                            raw_value=None, score=None,
                            sector_mean=mu, sector_std=sigma,
                        )
                        continue
                    if sigma == 0.0:
                        z, score = 0.0, 50.0
                    else:
                        z     = float(np.clip((v - mu) / sigma, -3.0, 3.0))
                        score = (z + 3.0) / 6.0 * 100.0
                    scores[ticker]  = score
                    details[ticker] = NormDetail(
                        method="zscore_sectoral", n_peers=n,
                        raw_value=float(v), score=score,
                        sector_mean=mu, sector_std=sigma, z_score=z,
                    )

            elif n >= MIN_SECTOR_PERCENTILE:
                # ── Percentil Setorial ──────────────────────────────────
                # Recomendação Deutsche Bank Quant Research:
                # percentil é mais robusto a outliers quando N < 8.
                # kind='rank': empates → média dos ranks (sem viés de posição)
                peer_vals = sector_values.values.astype(float)

                for ticker in sector_idx:
                    v = values.get(ticker, np.nan)
                    if pd.isna(v):
                        details[ticker] = NormDetail(
                            method="percentile_sectoral", n_peers=n,
                            raw_value=None, score=None,
                        )
                        continue
                    pct   = float(stats.percentileofscore(peer_vals, float(v), kind="rank"))
                    scores[ticker]  = pct
                    details[ticker] = NormDetail(
                        method="percentile_sectoral", n_peers=n,
                        raw_value=float(v), score=pct, percentile=pct,
                    )

            else:
                # ── Fallback Global — setor muito pequeno (N < 4) ───────
                global_fallback_tickers.extend(sector_idx.tolist())

        # Aplicar Z-Score Global para tickers de setores pequenos
        if global_fallback_tickers:
            scores, details = self._apply_global_fallback(
                values, scores, details, global_fallback_tickers
            )

        return scores, details

    def _normalize_global(
        self,
        values: pd.Series,
    ) -> tuple[pd.Series, dict[str, Optional[NormDetail]]]:
        """
        Z-Score Global para o universo completo.
        Usado em momentum (alpha vs IBOV) e qualidade/risco.
        """
        scores  = pd.Series(np.nan, index=values.index, dtype=float)
        details: dict[str, Optional[NormDetail]] = {t: None for t in values.index}

        valid = values.dropna()
        n     = len(valid)
        if n < 2:
            logger.warning("_normalize_global: apenas %d valores válidos — scores serão NaN", n)
            return scores, details

        mu    = float(valid.mean())
        sigma = float(valid.std(ddof=1))

        for ticker in values.index:
            v = values.get(ticker, np.nan)
            if pd.isna(v):
                details[ticker] = NormDetail(
                    method="zscore_global", n_peers=n,
                    raw_value=None, score=None,
                    sector_mean=mu, sector_std=sigma,
                )
                continue
            if sigma == 0.0:
                z, score = 0.0, 50.0
            else:
                z     = float(np.clip((v - mu) / sigma, -3.0, 3.0))
                score = (z + 3.0) / 6.0 * 100.0
            scores[ticker]  = score
            details[ticker] = NormDetail(
                method="zscore_global", n_peers=n,
                raw_value=float(v), score=score,
                sector_mean=mu, sector_std=sigma, z_score=z,
            )

        return scores, details

    @staticmethod
    def _apply_global_fallback(
        values: pd.Series,
        scores: pd.Series,
        details: dict,
        fallback_tickers: list[str],
    ) -> tuple[pd.Series, dict]:
        """
        Aplica Z-Score Global para tickers de setores com N < MIN_SECTOR_PERCENTILE.
        Usa o universo COMPLETO (todos os tickers com dado válido) como referência.
        """
        all_valid = values.dropna()
        n_global  = len(all_valid)
        if n_global < 2:
            return scores, details

        mu_g    = float(all_valid.mean())
        sigma_g = float(all_valid.std(ddof=1))

        for ticker in fallback_tickers:
            v = values.get(ticker, np.nan)
            if pd.isna(v):
                details[ticker] = NormDetail(
                    method="zscore_global", n_peers=n_global,
                    raw_value=None, score=None,
                    sector_mean=mu_g, sector_std=sigma_g,
                )
                continue
            if sigma_g == 0.0:
                z, score = 0.0, 50.0
            else:
                z     = float(np.clip((v - mu_g) / sigma_g, -3.0, 3.0))
                score = (z + 3.0) / 6.0 * 100.0
            scores[ticker]  = score
            details[ticker] = NormDetail(
                method="zscore_global", n_peers=n_global,
                raw_value=float(v), score=score,
                sector_mean=mu_g, sector_std=sigma_g, z_score=z,
            )

        return scores, details

    # ═══════════════════════════════════════════════════════════════════════
    # Soma ponderada com redistribuição de NaN
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _weighted_sum(
        index: pd.Index,
        base_weights: dict[str, float],
        factor_scores_map: dict[str, pd.Series],
    ) -> pd.Series:
        """
        Soma ponderada com redistribuição proporcional de pesos para fatores NaN.

        Garante que um ticker com dados em apenas 2 dos 3 fatores de momentum
        não seja artificialmente penalizado (seu score é calculado com os 2
        fatores disponíveis, com pesos renormalizados para somar 100%).
        """
        composite = pd.Series(np.nan, index=index, dtype=float)

        for ticker in index:
            active = {
                f: w
                for f, w in base_weights.items()
                if f in factor_scores_map
                and not pd.isna(factor_scores_map[f].get(ticker, np.nan))
            }
            total_w = sum(active.values())
            if total_w == 0:
                continue
            score = sum(
                (w / total_w) * factor_scores_map[f][ticker]
                for f, w in active.items()
            )
            composite[ticker] = score

        return composite

    # ═══════════════════════════════════════════════════════════════════════
    # Cálculos de séries temporais (preços)
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _trailing_returns(df_prices: pd.DataFrame, window_days: int) -> pd.Series:
        """
        Retorno acumulado no período: (P_atual / P_window_dias_atrás) - 1.

        Usa posições -1 e -window_days para robustez com dias sem pregão.
        Retorna NaN para tickers com dados insuficientes.
        """
        if df_prices.empty or len(df_prices) < 2:
            return pd.Series(np.nan, index=df_prices.columns)

        actual_window = min(window_days, len(df_prices) - 1)
        price_now  = df_prices.iloc[-1]
        price_past = df_prices.iloc[-actual_window]

        # Evitar divisão por zero e preços inválidos
        valid = price_past > 0
        ret   = pd.Series(np.nan, index=df_prices.columns)
        ret[valid] = (price_now[valid] / price_past[valid]) - 1.0

        if actual_window < window_days:
            logger.warning(
                "Janela de %dd: apenas %dd disponíveis — retornos aproximados",
                window_days, actual_window,
            )
        return ret

    @staticmethod
    def _scalar_return(series: pd.Series, window_days: int) -> float:
        """Retorno escalar do IBOVESPA na janela especificada. Retorna 0.0 se insuficiente."""
        clean = series.dropna()
        if len(clean) < 2:
            return 0.0
        actual_window = min(window_days, len(clean) - 1)
        return float((clean.iloc[-1] / clean.iloc[-actual_window]) - 1.0)

    @staticmethod
    def _volatility(df_prices: pd.DataFrame, window: int) -> pd.Series:
        """
        Volatilidade anualizada = std(retornos diários, window dias) × √252.

        Usa tail(window) dos retornos diários para capturar vol recente.
        Retorna NaN para tickers com menos de 30 dias de dados.
        """
        if df_prices.empty:
            return pd.Series(np.nan, index=df_prices.columns)
        daily_ret = df_prices.pct_change().dropna(how="all")
        if len(daily_ret) < 30:
            return pd.Series(np.nan, index=df_prices.columns)
        recent_ret = daily_ret.tail(window)
        vol = recent_ret.std(ddof=1) * np.sqrt(252)
        return vol.rename("volatility_180d")

    @staticmethod
    def _betas(df_prices: pd.DataFrame, ibov_prices: pd.Series) -> pd.Series:
        """
        Beta = Cov(ret_ação, ret_IBOV) / Var(ret_IBOV) — janela 252 dias.

        Alinha os índices de data antes de calcular para evitar deslocamentos.
        Retorna NaN para tickers com menos de 60 observações comuns.
        """
        ibov_clean = ibov_prices.dropna()
        if ibov_clean.empty:
            return pd.Series(np.nan, index=df_prices.columns)

        common_idx = df_prices.index.intersection(ibov_clean.index)
        if len(common_idx) < 60:
            return pd.Series(np.nan, index=df_prices.columns)

        stock_ret = df_prices.loc[common_idx].pct_change().dropna(how="all").tail(252)
        ibov_ret  = ibov_clean.loc[common_idx].pct_change().dropna().tail(252)

        # Re-alinhar após pct_change (perde primeira linha)
        shared = stock_ret.index.intersection(ibov_ret.index)
        stock_ret = stock_ret.loc[shared]
        ibov_ret  = ibov_ret.loc[shared]

        ibov_var = ibov_ret.var()
        if ibov_var == 0:
            return pd.Series(np.nan, index=df_prices.columns)

        return stock_ret.apply(lambda col: col.cov(ibov_ret) / ibov_var)

    # ═══════════════════════════════════════════════════════════════════════
    # Geração de explicação ("why")
    # ═══════════════════════════════════════════════════════════════════════

    def _explain(
        self,
        ticker: str,
        fund_details: dict[str, dict],
        mom_details: dict[str, dict],
    ) -> str:
        """
        Gera texto explicativo dos TOP 2-3 fatores que mais contribuíram.

        Impacto de um fator = |score - 50|: quanto mais afastado do neutro (50),
        mais esse fator diferenciou o ticker no universo.

        Formato:
          Z-Score:    "ROE 28.0% — 2.1σ acima da média do setor (N=12)"
          Percentil:  "P/VP 0.50 — top 15% do setor (N=6)"
          Global:     "Alpha 6m +18.3% vs IBOV — 1.8σ acima do universo (N=80)"
        """
        all_details = {
            **fund_details.get(ticker, {}),
            **mom_details.get(ticker, {}),
        }

        # Calcular impacto de cada fator e selecionar top 3
        ranked: list[tuple[float, str, NormDetail]] = []
        for factor, detail in all_details.items():
            if detail is None or detail.score is None or pd.isna(detail.score):
                continue
            impact = abs(detail.score - 50.0)
            ranked.append((impact, factor, detail))

        ranked.sort(key=lambda x: -x[0])
        top3 = ranked[:3]

        if not top3:
            return "Dados insuficientes para análise detalhada."

        parts: list[str] = []
        for _, factor, detail in top3:
            text = self._format_factor_explanation(factor, detail)
            if text:
                parts.append(text)

        return "; ".join(parts) if parts else "Score baseado em múltiplos fatores."

    @staticmethod
    def _format_factor_explanation(factor: str, detail: NormDetail) -> str:
        """Formata uma linha de explicação para um fator."""
        label   = _FACTOR_LABELS.get(factor, factor)
        rv      = detail.raw_value

        if rv is None or (isinstance(rv, float) and pd.isna(rv)):
            return ""

        # Formatar valor bruto de forma legível
        if factor == "earnings_yield":
            pl_approx = 1.0 / rv if rv > 0 else None
            rv_str = f"EY={rv * 100:.1f}% (P/L≈{pl_approx:.1f})" if pl_approx else f"{rv:.3f}"
        elif factor in ("roe", "roic", "dividend_yield"):
            rv_str = f"{rv * 100:.1f}%"
        elif factor in ("alpha_3m", "alpha_6m", "alpha_12m"):
            rv_str = f"{rv * 100:+.1f}% vs IBOV"
        elif factor == "volatility_180d":
            rv_str = f"{rv * 100:.1f}%a.a."
        else:
            rv_str = f"{rv:.2f}"

        if detail.method in ("zscore_sectoral", "zscore_global"):
            scope = "do setor" if detail.method == "zscore_sectoral" else "do universo"
            z     = detail.z_score
            if z is None:
                return f"{label} {rv_str}"

            # Direção do z precisa ser interpretada com base em direction:
            #   lower_is_better: z > 0 = valor original ABAIXO da média = bom
            #   higher_is_better: z > 0 = valor original ACIMA da média = bom
            if detail.direction == "lower_is_better":
                direction_text = "abaixo da média" if z > 0 else "acima da média"
            else:
                direction_text = "acima da média" if z > 0 else "abaixo da média"

            scope_n = f"{scope} (N={detail.n_peers})"
            return f"{label} {rv_str} — {abs(z):.1f}σ {direction_text} {scope_n}"

        elif detail.method == "percentile_sectoral":
            pct = detail.percentile or detail.score
            if pct is None:
                return f"{label} {rv_str}"

            # Para lower_is_better: pct alto do valor NEGADO = original está ABAIXO da média
            # Ex: P/VP = 0.5 com pct=80 no negado → "top 20% do setor"
            if detail.direction == "lower_is_better":
                top_pct = 100.0 - pct
                if top_pct <= 50:
                    pct_text = f"top {top_pct:.0f}% do setor"
                else:
                    pct_text = f"abaixo da mediana ({100-top_pct:.0f}º percentil)"
            else:
                if pct >= 50:
                    pct_text = f"top {100 - pct:.0f}% do setor"
                else:
                    pct_text = f"abaixo da mediana ({pct:.0f}º percentil)"

            return f"{label} {rv_str} — {pct_text} (N={detail.n_peers})"

        return f"{label} {rv_str}"

    # ═══════════════════════════════════════════════════════════════════════
    # Metadata de normalização para JSON de histórico
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _build_norm_details_col(
        fund_details: dict,
        mom_details: dict,
        qual_details: dict,
        index: pd.Index,
    ) -> pd.Series:
        """
        Serializa NormDetails por ticker em um dict para armazenamento no JSON.

        Estrutura: {fator: {method, n_peers, score, z_score/pct, direction}}
        Permite rastrear, em 6 meses, se a recomendação usou zscore ou percentil.
        """
        result: dict[str, dict] = {}
        for ticker in index:
            merged = {
                **fund_details.get(ticker, {}),
                **mom_details.get(ticker, {}),
                **qual_details.get(ticker, {}),
            }
            serialized = {}
            for factor, detail in merged.items():
                if detail is None:
                    continue
                serialized[factor] = {
                    "normalization_method": detail.method,
                    "n_peers":  detail.n_peers,
                    "score":    round(detail.score, 2) if detail.score is not None else None,
                    "z_score":  round(detail.z_score, 3) if detail.z_score is not None else None,
                    "percentile": round(detail.percentile, 1) if detail.percentile is not None else None,
                    "direction": detail.direction,
                }
            result[ticker] = serialized
        return pd.Series(result, index=index)


# ---------------------------------------------------------------------------
# Função de conveniência para main.py
# ---------------------------------------------------------------------------
def compute_scores(
    df_fund: pd.DataFrame,
    df_prices: pd.DataFrame,
    ibov_prices: pd.Series,
    regime: str = "mean_rev",
) -> pd.DataFrame:
    """
    Ponto de entrada simplificado.

    Args:
        regime: Market regime detected externally ("risk_on" | "mean_rev" | "bear").
                Controls pillar weights. Default "mean_rev" matches old WEIGHTS config.

    Returns:
        DataFrame ordenado por total_score DESC, com todas as colunas de score.
    """
    return ScoringEngine().score(df_fund, df_prices, ibov_prices, regime=regime)
