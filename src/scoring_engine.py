"""
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
    ANALYST_REVISIONS_WEIGHT,
    ANALYST_TARGET_MAX_UPSIDE,
    ANALYST_TARGET_WEIGHT,
    BETA_SANITY_MAX,
    BETA_SANITY_MIN,
    BRL_CORRELATION_WINDOW,
    CONVICTION_HIGH,
    CONVICTION_MEDIUM,
    ENABLE_WINSORIZATION,
    WINSORIZATION_MAD_K,
    ENABLE_LIQUIDITY_PENALTY,
    LIQUIDITY_PENALTY_MIN_FACTOR,
    LIQUIDITY_PENALTY_THRESHOLD_BRL,
    ENABLE_ANALYST_REVISIONS,
    ENABLE_ANALYST_TARGET,
    ENABLE_BRL_FACTOR,
    ENABLE_FCF_PAYOUT_CHECK,
    ENABLE_GROWTH_FACTOR,
    ENABLE_IDIO_MOMENTUM,
    ENABLE_INVESTMENT_FACTOR,
    ENABLE_PEAD_FACTOR,
    ENABLE_SIZE_FACTOR,
    FCF_PAYOUT_UNSUSTAINABLE,
    GROWTH_WEIGHT,
    INVESTMENT_WEIGHT,
    MACRO_THEME_MAP,
    ABSOLUTE_MAX_DIVIDA_EBITDA,
    MAX_DIVIDA_EBITDA,
    SECTOR_LEVERAGE_TOLERANCE,
    MAX_PER_MACRO_THEME,
    MAX_PER_SECTOR,
    MAX_PER_SUBSECTOR,
    MIN_DAILY_VOLUME_BRL,
    MIN_FUNDAMENTALS_REQUIRED,
    MIN_SCORE_THRESHOLD,
    MIN_SECTOR_PERCENTILE,
    MIN_SECTOR_ZSCORE,
    ENABLE_REGIME_ADAPTIVE_WEIGHTS,
    MOMENTUM_SKIP_DAYS,
    MOMENTUM_WINDOWS,
    PEAD_DRIFT_ENTRY_DAYS,
    PEAD_DRIFT_HOLDING_DAYS,
    PEAD_SURPRISE_WINDOW_DAYS,
    PEAD_WEIGHT,
    SIZE_WEIGHT,
    VOLATILITY_WINDOW,
    WEIGHTS,
)

logger = logging.getLogger(__name__)

FINANCIAL_SECTORS = {"Financeiro e Outros"}

# Regime-adaptive scoring weights
# 3-state market regime determined by IBOV vs MA200 and VIX level.
#   risk_on:  IBOV > MA200 and VIX < 18 — trending bull; favour momentum
#   bear:     VIX > 25 — global risk-off; favour quality/low-vol
#   mean_rev: everything else — range-bound; favour fundamental value
_REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "risk_on":  {"fundamental": 0.30, "momentum": 0.50, "quality": 0.20},
    "mean_rev": {"fundamental": 0.50, "momentum": 0.20, "quality": 0.30},
    "bear":     {"fundamental": 0.30, "momentum": 0.10, "quality": 0.60},
}

# Configuração dos fatores fundamentalistas com pesos base e direção.
#
# Base 1.0 pré-normalização:
#   Valor/qualidade clássica:  EY+P/VP+ROE+ROIC+D/EBITDA+DY = 1.00
# Sub-fatores QMJ + Fama-French (opt-in via flags em config):
#   + log_market_cap     (Size/SMB, lower_is_better)         = 0.05
#   + revenue_growth_3y  (Growth)                             = 0.04
#   + earnings_growth_3y (Growth)                             = 0.04
#   + asset_growth_yoy   (Investment/CMA, lower_is_better)   = 0.05
# Pesos são renormalizados em _fundamental_weights conforme dados disponíveis.
_FUNDAMENTAL_CFG: dict[str, dict] = {
    "earnings_yield": {"base_weight": 0.20, "direction": "higher_is_better", "label": "Earnings Yield"},
    "pvp":            {"base_weight": 0.15, "direction": "lower_is_better",  "label": "P/VP"},
    "roe":            {"base_weight": 0.25, "direction": "higher_is_better", "label": "ROE"},
    "roic":           {"base_weight": 0.20, "direction": "higher_is_better", "label": "ROIC"},
    "divida_ebitda":  {"base_weight": 0.10, "direction": "lower_is_better",  "label": "Dívida/EBITDA"},
    "dividend_yield": {"base_weight": 0.10, "direction": "higher_is_better", "label": "Dividend Yield"},
}
if ENABLE_SIZE_FACTOR:
    _FUNDAMENTAL_CFG["log_market_cap"] = {
        "base_weight": SIZE_WEIGHT, "direction": "lower_is_better",
        "label": "Size (log Mkt Cap)",
    }
if ENABLE_GROWTH_FACTOR:
    # GROWTH_WEIGHT é dividido entre revenue e earnings growth
    _FUNDAMENTAL_CFG["revenue_growth_3y"] = {
        "base_weight": GROWTH_WEIGHT / 2, "direction": "higher_is_better",
        "label": "Crescimento Receita 3y",
    }
    _FUNDAMENTAL_CFG["earnings_growth_3y"] = {
        "base_weight": GROWTH_WEIGHT / 2, "direction": "higher_is_better",
        "label": "Crescimento Lucro 3y",
    }
if ENABLE_INVESTMENT_FACTOR:
    _FUNDAMENTAL_CFG["asset_growth_yoy"] = {
        "base_weight": INVESTMENT_WEIGHT, "direction": "lower_is_better",
        "label": "Investment (Asset Growth)",
    }

_MOMENTUM_CFG: dict[str, dict] = {
    "alpha_3m":      {"base_weight": 0.30, "direction": "higher_is_better", "label": "Alpha 3m"},
    "alpha_12m":     {"base_weight": 0.30, "direction": "higher_is_better", "label": "Alpha 12m"},
}
if ENABLE_IDIO_MOMENTUM:
    _MOMENTUM_CFG["idio_alpha_6m"] = {
        "base_weight": 0.40, "direction": "higher_is_better",
        "label": "Momentum Idiossincr. 6m",
    }
if ENABLE_ANALYST_REVISIONS:
    # Sub-fator de momentum: tendência de revisão analista (1-5, higher = better).
    # Peso baixo dentro do pilar — proxy ruidoso de dados de consenso pago.
    _MOMENTUM_CFG["analyst_rec_score"] = {
        "base_weight": ANALYST_REVISIONS_WEIGHT, "direction": "higher_is_better",
        "label": "Recomendação Analistas",
    }
if ENABLE_ANALYST_TARGET:
    # Sub-fator forward-looking: upside implícito vs preço atual.
    # Brav-Lehavy (2003): IC do nível absoluto é fraco (0.02-0.04) com viés
    # otimista EM crônico. O rank cross-sectional (que o Z-Score faz) extrai
    # o sinal. Peso baixo para não dominar pilares mais robustos.
    _MOMENTUM_CFG["analyst_target_upside"] = {
        "base_weight": ANALYST_TARGET_WEIGHT, "direction": "higher_is_better",
        "label": "Upside Implícito Analistas",
    }
if ENABLE_PEAD_FACTOR:
    # Post-Earnings Announcement Drift: surprise positiva nos últimos
    # PEAD_DRIFT_HOLDING_DAYS = 60 dias úteis → bônus de momentum.
    # Score [0, 100] computado em _compute_pead_score.
    _MOMENTUM_CFG["pead_signal"] = {
        "base_weight": PEAD_WEIGHT, "direction": "higher_is_better",
        "label": "PEAD (drift pós-resultado)",
    }

_QUALITY_CFG: dict[str, dict] = {
    "volatility_180d": {"base_weight": 0.40, "direction": "lower_is_better",  "label": "Volatilidade 180d"},
    "beta":            {"base_weight": 0.35, "direction": "lower_is_better",  "label": "Beta vs IBOV"},
    "avg_volume_30d":  {"base_weight": 0.25, "direction": "higher_is_better", "label": "Volume Médio 30d"},
}

_FACTOR_LABELS = {f: cfg["label"] for d in [_FUNDAMENTAL_CFG, _MOMENTUM_CFG, _QUALITY_CFG] for f, cfg in d.items()}


# NormDetail — metadados de normalização por fator por ticker
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


# ScoringEngine
class ScoringEngine:
    """
    Score composto 0-100 para cada ticker do universo.
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
        # Pesos por regime estão CONGELADOS por default (nunca validados —
        # IC n≈2). O regime continua sendo detectado e usado pelo allocator.
        # Ver ENABLE_REGIME_ADAPTIVE_WEIGHTS em config.py.
        if ENABLE_REGIME_ADAPTIVE_WEIGHTS:
            active_weights = _REGIME_WEIGHTS.get(regime, WEIGHTS)
            if regime not in _REGIME_WEIGHTS:
                logger.warning("Regime '%s' desconhecido — usando pesos padrão", regime)
            else:
                logger.info(
                    "Regime de mercado: %s → Fundamental=%.0f%% Momentum=%.0f%% Quality=%.0f%%",
                    regime,
                    active_weights["fundamental"] * 100,
                    active_weights["momentum"] * 100,
                    active_weights["quality"] * 100,
                )
        else:
            active_weights = WEIGHTS
            logger.info(
                "Pesos de pilar fixos (regime '%s' informativo): "
                "Fund=%.0f%% Mom=%.0f%% Qual=%.0f%%",
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

        # Pré-processamento
        df = self._derive_metrics(df)
        df = self._add_momentum_metrics(df, df_prices, ibov_prices)
        if ENABLE_IDIO_MOMENTUM:
            df = self._add_idiosyncratic_momentum(df, df_prices, ibov_prices, sector_map)
        df = self._add_quality_metrics(df, df_prices, ibov_prices)
        df = self._add_pead_signal(df, df_prices, ibov_prices)
        df = self._add_brl_exposure(df, df_prices)
        df = self._apply_hard_filters(df)
        sector_map = sector_map.reindex(df.index)  # re-alinhar após filtros

        # Scoring por pilar
        fund_scores, fund_details = self._score_fundamental(df, sector_map)
        mom_scores,  mom_details  = self._score_momentum(df)
        qual_scores, qual_details = self._score_quality(df)

        # Score composto — regime-adaptive weights
        df["fundamental_score"] = fund_scores.round(2)
        df["momentum_score"]    = mom_scores.round(2)
        df["quality_score"]     = qual_scores.round(2)
        df["market_regime"]     = regime
        total_score = (
            active_weights["fundamental"] * fund_scores
            + active_weights["momentum"]  * mom_scores
            + active_weights["quality"]   * qual_scores
        )

        # Liquidity penalty: multiplica score por sqrt(ADV / threshold).
        # Tickers acima do threshold ficam intactos (penalty=1); abaixo
        # ganham penalty < 1, com floor em LIQUIDITY_PENALTY_MIN_FACTOR.
        # Mata "alpha de papel" em small caps onde slippage > alpha esperado.
        if ENABLE_LIQUIDITY_PENALTY and "avg_volume_30d" in df.columns:
            adv = df["avg_volume_30d"].astype(float)
            penalty = np.minimum(1.0, np.sqrt(adv / LIQUIDITY_PENALTY_THRESHOLD_BRL))
            penalty = penalty.clip(lower=LIQUIDITY_PENALTY_MIN_FACTOR, upper=1.0)
            # NaN → assumir penalty 1 (não penalizar quando dado ausente)
            penalty = penalty.fillna(1.0)
            df["liquidity_penalty"] = penalty.round(3)
            total_score = total_score * penalty
            n_penalized = int((penalty < 1.0).sum())
            if n_penalized > 0:
                logger.info(
                    "Liquidity penalty aplicada em %d/%d tickers (threshold=R$%.0fM)",
                    n_penalized, len(df), LIQUIDITY_PENALTY_THRESHOLD_BRL / 1e6,
                )

        df["total_score"] = total_score.round(2)

        # Campos de rastreabilidade
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

        # Convicção: calibração de quão bem-suportada está cada pick
        conv = {
            t: self._conviction_score(
                df.at[t, "total_score"] if "total_score" in df.columns else None,
                df.at[t, "normalization_method"] if "normalization_method" in df.columns else None,
                fund_details.get(t, {}),
            )
            for t in df.index
        }
        df["conviction"]       = pd.Series(conv, dtype=float)
        df["conviction_label"] = df["conviction"].map(self._conviction_label)

        # Montar output
        ordered_cols = [
            "nome", "setor", "subsetor", "normalization_method", "data_source",
            # métricas brutas
            "earnings_yield", "pvp", "roe", "roic", "divida_ebitda", "dividend_yield",
            "alpha_3m", "alpha_6m", "idio_alpha_6m", "alpha_12m",
            "volatility_180d", "beta", "avg_volume_30d",
            "current_price", "market_cap",
            # novos fatores fundamentalistas (Size, Growth, Investment)
            "log_market_cap", "revenue_growth_3y", "earnings_growth_3y",
            "asset_growth_yoy", "net_margin_3y_avg",
            # FCF / Payout sustainability
            "free_cash_flow_ttm", "fcf_payout_ratio",
            # Earnings revisions + analyst target (momentum sub-factors)
            "analyst_rec_score", "analyst_rec_trend",
            "analyst_target_upside", "analyst_target_mean", "analyst_target_dispersion",
            # PEAD (Post-Earnings Announcement Drift)
            "pead_signal", "last_earnings_date", "days_since_earnings",
            "earnings_surprise_pct",
            # BRL exposure (risk dimension, not direct score)
            "brl_corr_90d",
            # pilares e total
            "fundamental_score", "momentum_score", "quality_score", "total_score",
            # convicção (calibração de confiança nos insumos)
            "conviction", "conviction_label",
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

    # Pré-processamento de métricas

    @staticmethod
    def _derive_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """
        Métricas derivadas dos dados brutos:

          - Earnings Yield = 1 / P·L
            (mais próximo da normal que P/L → Z-Score mais estável)
          - log_market_cap = log10(market_cap)
            (escala log para que mid-caps não fiquem espremidos pelos gigantes;
             usado como fator Size/SMB — direção lower_is_better)
        """
        mask_valid = df["pl"].notna() & (df["pl"] > 0)
        df["earnings_yield"] = np.where(mask_valid, 1.0 / df["pl"], np.nan)
        logger.debug(
            "Earnings Yield: %d/%d tickers com dado válido",
            mask_valid.sum(), len(df),
        )

        # log_market_cap: usar log10 (mesma escala estável; market_cap em BRL).
        if "market_cap" in df.columns:
            mc = df["market_cap"]
            valid_mc = mc.notna() & (mc > 0)
            df["log_market_cap"] = np.where(valid_mc, np.log10(mc.where(valid_mc, 1)), np.nan)

        # Sanidade ROE/ROIC: fora de [-100%, +150%] e' artefato de fonte
        # (ex.: patrimonio quase zero inflando o quociente). Antes apenas
        # documentado, nunca implementado — garbage de fonte ia direto ao score.
        for _col, _lo, _hi in (("roe", -1.0, 1.5), ("roic", -1.0, 1.5)):
            if _col in df.columns:
                _v = pd.to_numeric(df[_col], errors="coerce")
                df[_col] = _v.where((_v >= _lo) & (_v <= _hi))

        # Divida/EBITDA: negativo e' legitimo (caixa liquido); |x| > 50 e' erro.
        if "divida_ebitda" in df.columns:
            _de = pd.to_numeric(df["divida_ebitda"], errors="coerce")
            df["divida_ebitda"] = _de.where(_de.abs() <= 50)

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
            # Convenção skip-month (Jegadeesh-Titman): mede t-window → t-21,
            # excluindo o último mês, dominado por reversão de curto prazo.
            # Mesmo skip no IBOV para o alpha comparar períodos idênticos.
            stock_returns = self._trailing_returns(
                df_prices, window, skip_days=MOMENTUM_SKIP_DAYS)
            ibov_return   = self._scalar_return(
                ibov_prices, window, skip_days=MOMENTUM_SKIP_DAYS)
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

        Volatilidade: Yang-Zhang (via OHLC) é preferida sobre close-to-close
        — ~5× mais eficiente. Se YZ não disponível para um ticker, fallback
        para close-to-close calculado de df_prices.
        Beta: usa do DataCollector quando disponível; senão calcula.
        """
        vol_cc = self._volatility(df_prices, VOLATILITY_WINDOW)
        beta_series = self._betas(df_prices, ibov_prices)

        # Preferir Yang-Zhang vol se disponível; senão usar close-to-close
        if "volatility_yz_180d" in df.columns:
            # vol_cc é Series indexada por ticker; alinhar via reindex (vira Series)
            vol_cc_aligned = pd.Series(df.index.map(vol_cc), index=df.index)
            df["volatility_180d"] = df["volatility_yz_180d"].fillna(vol_cc_aligned)
            n_yz = df["volatility_yz_180d"].notna().sum()
            logger.debug("Vol: %d Yang-Zhang + %d close-to-close fallback",
                         n_yz, len(df) - n_yz)
        else:
            df["volatility_180d"] = df.index.map(vol_cc)

        # Beta: PREFERIR o beta calculado dos preços (janela 252d vs IBOV,
        # benchmark e janela transparentes e consistentes). O beta de fonte
        # (brapi) é opaco quanto à janela/benchmark e ocasionalmente vem
        # corrompido — ex.: PETR4 beta=-0.06, impossível para a maior
        # petroleira do IBOV. Usar a fonte só como FALLBACK quando o cálculo
        # dos preços é NaN, e ainda assim sob bounds de sanidade.
        price_beta = pd.Series(df.index.map(beta_series), index=df.index).astype(float)
        if "beta" in df.columns:
            src_beta = pd.to_numeric(df["beta"], errors="coerce")
            src_beta_sane = src_beta.where(src_beta.between(BETA_SANITY_MIN, BETA_SANITY_MAX))
            n_rejected = int((src_beta.notna() & src_beta_sane.isna()).sum())
            if n_rejected:
                logger.info(
                    "Beta de fonte rejeitado por sanidade [%g, %g] em %d ticker(s)",
                    BETA_SANITY_MIN, BETA_SANITY_MAX, n_rejected,
                )
        else:
            src_beta_sane = pd.Series(np.nan, index=df.index)
        df["beta"] = price_beta.fillna(src_beta_sane)

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

    def _add_pead_signal(
        self,
        df: pd.DataFrame,
        df_prices: pd.DataFrame,
        ibov_prices: pd.Series,
    ) -> pd.DataFrame:
        """
        Post-Earnings Announcement Drift (PEAD) signal.

        Para cada ticker com `last_earnings_date` no DataFrame, calcula:
          1. EAR = CAR(-PEAD_SURPRISE_WINDOW, +PEAD_SURPRISE_WINDOW) vs IBOV
             (proxy de surprise sem precisar de consenso de analistas)
          2. Se EAR > 0 E ticker está em janela [PEAD_DRIFT_ENTRY_DAYS,
             PEAD_DRIFT_HOLDING_DAYS] após o anúncio → signal positivo

        O valor exportado é EAR escalado pela porção de "vida útil" restante
        do drift — sinal decai linearmente após o pico.

        Tickers fora da janela ou sem earnings_date → NaN (vai para
        redistribuição de peso no scoring).

        O corte de look-ahead (só datas < hoje) já é feito no data_collector.
        """
        df["pead_signal"] = np.nan
        if not ENABLE_PEAD_FACTOR:
            return df

        if df_prices is None or df_prices.empty or ibov_prices is None or ibov_prices.empty:
            return df

        ibov_clean = ibov_prices.dropna()
        ibov_idx_naive = ibov_clean.index.tz_localize(None) if getattr(ibov_clean.index, "tz", None) is not None else ibov_clean.index

        n_signals = 0
        for ticker in df.index:
            last_date = df.at[ticker, "last_earnings_date"] if "last_earnings_date" in df.columns else None
            if last_date is None or (isinstance(last_date, float) and pd.isna(last_date)):
                continue
            try:
                event_date = pd.Timestamp(last_date)
                if event_date.tz is not None:
                    event_date = event_date.tz_localize(None)
            except Exception:
                continue

            # Janela de validade do drift (em dias úteis aproximados)
            today_ts = pd.Timestamp.now().normalize()
            days_since = (today_ts - event_date).days
            # Convertendo para úteis aproximados: × 5/7
            business_days_since = days_since * 5 / 7
            if business_days_since < PEAD_DRIFT_ENTRY_DAYS:
                continue  # ainda em janela de reversão imediata
            if business_days_since > PEAD_DRIFT_HOLDING_DAYS:
                continue  # drift já decaiu

            if ticker not in df_prices.columns:
                continue
            price_series = df_prices[ticker].dropna()
            if price_series.empty:
                continue
            price_idx_naive = price_series.index.tz_localize(None) if getattr(price_series.index, "tz", None) is not None else price_series.index

            # Encontrar a posição da data do evento no índice de preços
            # Tolerância: ±3 dias corridos (em caso de feriado)
            try:
                # idxmax/idxmin requer fechamento exato; usar searchsorted
                pos = price_idx_naive.searchsorted(event_date)
                if pos < 1 or pos >= len(price_idx_naive) - PEAD_SURPRISE_WINDOW_DAYS:
                    continue
                # Verificar tolerância
                actual_date = price_idx_naive[pos]
                if abs((actual_date - event_date).days) > 3:
                    continue
                # CAR(-w, +w) = ret_ticker - ret_ibov no período do evento
                t0_price = price_series.iloc[max(0, pos - PEAD_SURPRISE_WINDOW_DAYS)]
                t1_price = price_series.iloc[min(len(price_series) - 1, pos + PEAD_SURPRISE_WINDOW_DAYS)]
                if t0_price <= 0 or t1_price <= 0:
                    continue
                stock_ret = (t1_price / t0_price) - 1.0

                # Mesmo período para IBOV
                ibov_pos = ibov_idx_naive.searchsorted(event_date)
                if ibov_pos < 1 or ibov_pos >= len(ibov_idx_naive) - PEAD_SURPRISE_WINDOW_DAYS:
                    continue
                ibov_t0 = float(ibov_clean.iloc[max(0, ibov_pos - PEAD_SURPRISE_WINDOW_DAYS)])
                ibov_t1 = float(ibov_clean.iloc[min(len(ibov_clean) - 1, ibov_pos + PEAD_SURPRISE_WINDOW_DAYS)])
                if ibov_t0 <= 0 or ibov_t1 <= 0:
                    continue
                ibov_ret = (ibov_t1 / ibov_t0) - 1.0

                ear = stock_ret - ibov_ret

                # Decay linear: sinal cheio em DRIFT_ENTRY, zera em DRIFT_HOLDING.
                # life_remaining ∈ [0, 1]
                life_remaining = max(0.0, min(1.0,
                    (PEAD_DRIFT_HOLDING_DAYS - business_days_since)
                    / (PEAD_DRIFT_HOLDING_DAYS - PEAD_DRIFT_ENTRY_DAYS)
                ))
                # Sinal: EAR × life_remaining. Captura tanto magnitude quanto
                # quanto tempo de drift ainda resta para extrair.
                df.at[ticker, "pead_signal"] = float(ear * life_remaining)
                n_signals += 1
            except Exception:
                continue

        logger.info("PEAD signal computado para %d/%d tickers", n_signals, len(df))
        return df

    def _add_brl_exposure(
        self,
        df: pd.DataFrame,
        df_prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Exposição cambial: correlação rolling 90d de log-returns do ticker
        com USDBRL.

        Padrão B3:
          β > +0.3 → exportador (VALE, PETR, SUZB) — ganha com BRL fraco
          β ≈ 0   → neutro (bancos)
          β < -0.3 → doméstico (LREN, MGLU) — perde com BRL fraco

        Não é fator de scoring direto — é dimensão de risco para evitar
        concentração cambial no top 5 (todas as posições do mesmo lado da
        moeda = bet cambial implícita).

        Fonte: yfinance "BRL=X" (PTAX equivalente). Aceita ausência de dado.
        """
        df["brl_corr_90d"] = np.nan
        if not ENABLE_BRL_FACTOR or df_prices is None or df_prices.empty:
            return df

        try:
            import yfinance as yf
            # 1 ano de USDBRL para garantir janela de 90d com folga
            usdbrl = yf.download(
                "BRL=X",
                period="1y",
                progress=False,
                auto_adjust=True,
            )
            if usdbrl is None or usdbrl.empty:
                return df
            # Achatar MultiIndex se necessário
            if isinstance(usdbrl.columns, pd.MultiIndex):
                usdbrl.columns = usdbrl.columns.get_level_values(0)
            close_col = "Adj Close" if "Adj Close" in usdbrl.columns else "Close"
            usdbrl_close = usdbrl[close_col].dropna()
            if isinstance(usdbrl_close, pd.DataFrame):
                usdbrl_close = usdbrl_close.iloc[:, 0]
            usdbrl_ret = np.log(usdbrl_close / usdbrl_close.shift(1)).dropna()
            # Alinhar timezone
            if getattr(usdbrl_ret.index, "tz", None) is not None:
                usdbrl_ret.index = usdbrl_ret.index.tz_localize(None)

            # Pegar últimos BRL_CORRELATION_WINDOW dias
            window = BRL_CORRELATION_WINDOW
            usdbrl_window = usdbrl_ret.tail(window)
            if len(usdbrl_window) < window // 2:
                return df

            # Log returns dos tickers
            stock_log_ret = np.log(df_prices / df_prices.shift(1)).dropna(how="all")
            stock_window = stock_log_ret.tail(window)

            common_idx = stock_window.index.intersection(usdbrl_window.index)
            if len(common_idx) < 30:
                return df

            x = usdbrl_window.loc[common_idx]
            for ticker in df.index:
                if ticker not in stock_window.columns:
                    continue
                y = stock_window.loc[common_idx, ticker].dropna()
                aligned = y.index.intersection(x.index)
                if len(aligned) < 30:
                    continue
                rho = float(np.corrcoef(y.loc[aligned], x.loc[aligned])[0, 1])
                if not np.isnan(rho):
                    df.at[ticker, "brl_corr_90d"] = rho

            valid = df["brl_corr_90d"].notna().sum()
            logger.info("BRL correlation 90d: %d/%d tickers", valid, len(df))
        except Exception as exc:
            logger.debug("BRL exposure falhou (%s) — fator vai ficar NaN", exc)
        return df

    # Filtros hard — aplicados antes do ranking

    @staticmethod
    def _apply_hard_filters(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove tickers que violam critérios absolutos antes do ranking.

        Filtros:
          1. Liquidez < R$5M/dia — ação ilíquida impossibilita execução
          2. Dívida/EBITDA > 5x (fora do setor financeiro) — risco de crédito extremo
          3. data_source == "failed" — sem dados de nenhuma fonte
          4. Value trap: P/L negativo (empresa no prejuízo) + ROE < -5%
             — empresa destruindo capital; score baixo de PVP pode ser ilusório
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

        # Filtro de alavancagem SETOR-RELATIVO (não aplica ao financeiro).
        # Em vez do flat 5x — que excluía nomes legitimamente alavancados em
        # setores capital-intensivos (leasing, utilities, real estate) — um
        # não-financeiro é removido só se:
        #   (a) D/EBITDA > ABSOLUTE_MAX_DIVIDA_EBITDA (teto duro), OU
        #   (b) D/EBITDA > MAX_DIVIDA_EBITDA E acima de
        #       SECTOR_LEVERAGE_TOLERANCE × mediana do próprio setor.
        if "divida_ebitda" in df.columns and "setor" in df.columns:
            is_financial = df["setor"].isin(FINANCIAL_SECTORS)
            de = df["divida_ebitda"]
            has_de = de.notna()

            sector_median = (
                df.loc[~is_financial & has_de]
                .groupby("setor")["divida_ebitda"]
                .median()
            )
            median_for_row = df["setor"].map(sector_median)
            relative_ceiling = median_for_row * SECTOR_LEVERAGE_TOLERANCE

            absolute_kill = has_de & (de > ABSOLUTE_MAX_DIVIDA_EBITDA)
            relative_kill = (
                has_de
                & (de > MAX_DIVIDA_EBITDA)
                & median_for_row.notna()
                & (de > relative_ceiling)
            )
            over_levered = ~is_financial & (absolute_kill | relative_kill)

            if over_levered.any():
                excluded = df[over_levered].index.tolist()
                logger.info(
                    "Hard filter alavancagem (setor-relativo, teto=%gx): %s removidos",
                    ABSOLUTE_MAX_DIVIDA_EBITDA, excluded,
                )
                df = df[~over_levered]

        # Filtro value trap: prejuízo (P/L < 0) + ROE negativo (> -5%)
        # P/L negativo = empresa reportou perda no período.
        # Combinado com ROE < -5% indica destruição ativa de patrimônio,
        # não uma distorção temporária de resultado.
        if "pl" in df.columns and "roe" in df.columns:
            is_loss = df["pl"].notna() & (df["pl"] < 0)
            is_neg_roe = df["roe"].notna() & (df["roe"] < -0.05)
            value_trap = is_loss & is_neg_roe
            if value_trap.any():
                excluded = df[value_trap].index.tolist()
                logger.info("Hard filter value trap (P/L<0 & ROE<-5%%): %s removidos", excluded)
                df = df[~value_trap]

        # Filtro de cobertura mínima de fundamentos:
        # tickers com <MIN_FUNDAMENTALS_REQUIRED métricas não-NaN têm score
        # dominado pelo que está presente (geralmente momentum + qualidade)
        # e tendem a entrar artificialmente no top porque os fatores ruins
        # foram excluídos via redistribuição de peso. Excluir é mais correto
        # que penalizar.
        fund_cols = [c for c in ("pl", "pvp", "roe", "roic", "divida_ebitda", "dividend_yield")
                     if c in df.columns]
        if fund_cols:
            fund_valid_count = df[fund_cols].notna().sum(axis=1)
            insufficient = fund_valid_count < MIN_FUNDAMENTALS_REQUIRED
            # Exceção: setor financeiro não tem divida_ebitda (nem ROIC útil)
            # → exigir 1 a menos para esses casos.
            if "setor" in df.columns:
                is_financial = df["setor"].isin(FINANCIAL_SECTORS)
                insufficient = insufficient & ~(is_financial & (fund_valid_count >= MIN_FUNDAMENTALS_REQUIRED - 1))
            if insufficient.any():
                excluded = df[insufficient].index.tolist()
                logger.info(
                    "Hard filter cobertura (<%d fundamentos): %s removidos",
                    MIN_FUNDAMENTALS_REQUIRED, excluded,
                )
                df = df[~insufficient]

        n_after = len(df)
        if n_before != n_after:
            logger.info("Hard filters: %d→%d tickers (%d removidos)", n_before, n_after, n_before - n_after)
        return df

    # Scoring por pilar

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

        # Regra 2a — DY sustentabilidade via FCF (mais preciso que EY/DY)
        # Se fcf_payout_ratio disponível (yfinance cashflow) e > limite,
        # significa que dividendos pagos excedem o FCF gerado — insustentável.
        # FCF é menos manipulável que earnings reportados.
        fcf_check_applied = False
        if ENABLE_FCF_PAYOUT_CHECK:
            try:
                fcf_payout = row.get("fcf_payout_ratio")
                if fcf_payout is not None and not pd.isna(fcf_payout):
                    fcf_payout_f = float(fcf_payout)
                    if fcf_payout_f > FCF_PAYOUT_UNSUSTAINABLE:
                        weights.pop("dividend_yield", None)
                        fcf_check_applied = True
                        logger.debug(
                            "DY insustentável via FCF (%s): payout=%.1fx — DY excluído",
                            ticker, fcf_payout_f,
                        )
            except (TypeError, ValueError):
                pass

        # Regra 2b — DY sustentabilidade via EY/DY (fallback se FCF indisponível)
        # payout ≈ DY / EY. Se a empresa está pagando mais do que ganha,
        # o dividendo não é recorrente. Aplica só se FCF check não rodou.
        if not fcf_check_applied:
            try:
                ey_raw = row.get("earnings_yield")
                dy_raw = row.get("dividend_yield")
                ey_f = float(ey_raw) if ey_raw is not None else None
                dy_f = float(dy_raw) if dy_raw is not None else None
                if (ey_f and not pd.isna(ey_f) and ey_f > 0
                        and dy_f and not pd.isna(dy_f) and dy_f > 0):
                    payout = dy_f / ey_f
                    if payout > 1.5:
                        weights.pop("dividend_yield", None)
                        logger.debug(
                            "DY insustentável via EY (%s): payout≈%.1fx — DY excluído",
                            ticker, payout,
                        )
            except (TypeError, ValueError):
                pass

        # Regra 3 — NaN em demais fatores: redistribuição proporcional entre ativos
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

    # Normalização — núcleo matemático

    def _normalize_adaptive(
        self,
        values: pd.Series,
        sector_map: pd.Series,
    ) -> tuple[pd.Series, dict[str, Optional[NormDetail]]]:
        """
        Normalização adaptativa: escolhe método baseado em N de peers válidos por setor.

        'values' já vem com a direção corrigida (lower_is_better negados).
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
                # Z-Score Setorial
                # Winsorização robusta via MAD: limita os valores a
                # mediana ± k·1.4826·MAD antes de estimar μ/σ, para que um
                # outlier não infle o desvio e comprima o z-score dos demais.
                # raw_value no detail permanece o valor ORIGINAL.
                lo_w = hi_w = None
                if ENABLE_WINSORIZATION and n >= 5:
                    med = float(sector_values.median())
                    mad = float((sector_values - med).abs().median())
                    if mad > 0:
                        spread = WINSORIZATION_MAD_K * 1.4826 * mad
                        lo_w, hi_w = med - spread, med + spread
                sv = sector_values.clip(lo_w, hi_w) if lo_w is not None else sector_values
                mu    = float(sv.mean())
                sigma = float(sv.std(ddof=1)) if n > 1 else 0.0

                for ticker in sector_idx:
                    v = values.get(ticker, np.nan)
                    if pd.isna(v):
                        details[ticker] = NormDetail(
                            method="zscore_sectoral", n_peers=n,
                            raw_value=None, score=None,
                            sector_mean=mu, sector_std=sigma,
                        )
                        continue
                    v_eff = v if lo_w is None else float(np.clip(v, lo_w, hi_w))
                    if sigma == 0.0:
                        z, score = 0.0, 50.0
                    else:
                        z     = float(np.clip((v_eff - mu) / sigma, -3.0, 3.0))
                        score = (z + 3.0) / 6.0 * 100.0
                    scores[ticker]  = score
                    details[ticker] = NormDetail(
                        method="zscore_sectoral", n_peers=n,
                        raw_value=float(v), score=score,
                        sector_mean=mu, sector_std=sigma, z_score=z,
                    )

            elif n >= MIN_SECTOR_PERCENTILE:
                # Percentil Setorial
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
                # Fallback Global — setor muito pequeno (N < 4)
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

    # Soma ponderada com redistribuição de NaN

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

    # Cálculos de séries temporais (preços)

    @staticmethod
    def _trailing_returns(
        df_prices: pd.DataFrame, window_days: int, skip_days: int = 0,
    ) -> pd.Series:
        """
        Retorno acumulado no período: (P_t-skip / P_t-window) - 1.

        skip_days > 0 implementa a convenção momentum "12-1": exclui os
        últimos skip_days (reversão de curto prazo). A janela TOTAL continua
        sendo window_days contados a partir de hoje — i.e., mede de
        t-window até t-skip. Retorna NaN para tickers com dados insuficientes.
        """
        if df_prices.empty or len(df_prices) < skip_days + 2:
            return pd.Series(np.nan, index=df_prices.columns)
        if skip_days >= window_days:
            return pd.Series(np.nan, index=df_prices.columns)

        actual_window = min(window_days, len(df_prices) - 1)
        price_now  = df_prices.iloc[-1 - skip_days]
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
    def _scalar_return(
        series: pd.Series, window_days: int, skip_days: int = 0,
    ) -> float:
        """
        Retorno escalar do IBOVESPA na janela t-window → t-skip.
        Retorna 0.0 se insuficiente. skip_days espelha _trailing_returns.
        """
        clean = series.dropna()
        if len(clean) < skip_days + 2 or skip_days >= window_days:
            return 0.0
        actual_window = min(window_days, len(clean) - 1)
        return float((clean.iloc[-1 - skip_days] / clean.iloc[-actual_window]) - 1.0)

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

    # Geração de explicação ("why")

    @staticmethod
    def _conviction_score(
        total_score: Optional[float],
        norm_method: Optional[str],
        fund_details_ticker: dict,
    ) -> float:
        """
        Convicção [0,1]: quão bem-suportada está a recomendação.

        Componentes: cobertura de fatores (35%), nº de peers setoriais (25%),
        método de normalização (20%), margem de score acima de 50 (20%).

        NÃO é probabilidade de acerto — é confiança nos INSUMOS da decisão.
        Uma pick com poucos peers, muitos NaN e normalização global é frágil
        mesmo com score alto.
        """
        details = fund_details_ticker or {}
        n_total = len(details)
        n_valid = sum(
            1 for d in details.values()
            if d is not None and getattr(d, "score", None) is not None
        )
        coverage = (n_valid / n_total) if n_total else 0.0

        peers = [getattr(d, "n_peers", 0) or 0 for d in details.values() if d is not None]
        max_peers = max(peers) if peers else 0
        peer_support = 1.0 if max_peers >= 8 else 0.6 if max_peers >= 4 else 0.3

        method_score = {
            "zscore_sectoral": 1.0,
            "percentile_sectoral": 0.7,
            "zscore_global": 0.4,
        }.get(str(norm_method), 0.4)

        if total_score is None or pd.isna(total_score):
            margin = 0.0
        else:
            margin = float(np.clip((float(total_score) - 50.0) / 50.0, 0.0, 1.0))

        conviction = (
            0.35 * coverage
            + 0.25 * peer_support
            + 0.20 * method_score
            + 0.20 * margin
        )
        return round(float(np.clip(conviction, 0.0, 1.0)), 3)

    @staticmethod
    def _conviction_label(conviction: float) -> str:
        """Rótulo legível para a convicção numérica."""
        if conviction is None or pd.isna(conviction):
            return "Baixa"
        if conviction >= CONVICTION_HIGH:
            return "Alta"
        if conviction >= CONVICTION_MEDIUM:
            return "Média"
        return "Baixa"

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
        elif factor == "log_market_cap":
            # rv é log10(market_cap em BRL) — converter de volta para B/T
            mkt = 10 ** rv
            if mkt >= 1e9:
                rv_str = f"Mkt Cap R$ {mkt/1e9:.1f} B"
            else:
                rv_str = f"Mkt Cap R$ {mkt/1e6:.0f} M"
        elif factor in ("revenue_growth_3y", "earnings_growth_3y", "asset_growth_yoy"):
            rv_str = f"{rv * 100:+.1f}%a.a."
        elif factor == "analyst_rec_score":
            rv_str = f"{rv:.1f}/5"
        elif factor == "analyst_target_upside":
            rv_str = f"upside {rv * 100:+.0f}%"
        elif factor == "pead_signal":
            rv_str = f"PEAD {rv * 100:+.1f}%"
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

    # Metadata de normalização para JSON de histórico

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


# Seleção de portfólio diversificado

def select_diverse_portfolio(
    df_scored: pd.DataFrame,
    n: int = 5,
    max_per_sector: int = MAX_PER_SECTOR,
    max_per_subsector: int = MAX_PER_SUBSECTOR,
    max_per_theme: int = MAX_PER_MACRO_THEME,
    min_score: float = MIN_SCORE_THRESHOLD,
) -> pd.DataFrame:
    """
    Reordena df_scored colocando o top-n diversificado nas primeiras posições.

    Regras de diversificação (aplicadas em ordem de score DESC):
      1. Máximo 1 ticker por empresa (PETR4 e PETR3 → mesma empresa "PETR").
      2. Máximo max_per_subsector por sub-setor (evita 2 bancos, 2 mineradoras).
      3. Máximo max_per_sector por setor B3.
      4. Máximo max_per_theme por tema macroeconômico (commodity_export,
         domestic_consumer, defensive_utilities, financials, etc.). Quebra
         a concentração macro: top 5 não pode ser 4 commodities + 1 outra.
      5. Score mínimo: tickers com total_score < min_score são marcados
         como "tentativa" — só entram no top se não houver alternativa.

    Estratégia de fallback (em cascata, da mais restritiva para a mais relaxada):
      a) Tentar preencher com todas as restrições + score ≥ min_score
      b) Relaxar sub-setor se faltar (subsector cap → sector cap apenas)
      c) Aceitar tickers abaixo de min_score só se ainda faltar posição

    O restante do DataFrame mantém a ordem original de score.
    """
    if df_scored.empty:
        return df_scored

    # Helper: mapa setor → tema macro (com fallback "other")
    def _theme(sector: str) -> str:
        return MACRO_THEME_MAP.get(sector, "other")

    def _try_fill(
        positions: range,
        require_min_score: bool,
        enforce_subsector: bool,
    ) -> tuple[list[int], dict[str, int], dict[str, int], dict[str, int], set[str]]:
        """Tenta preencher n posições aplicando restrições graduadas."""
        sel: list[int] = []
        sec_cnt: dict[str, int] = {}
        sub_cnt: dict[str, int] = {}
        theme_cnt: dict[str, int] = {}
        seen_co: set[str] = set()

        for pos in positions:
            if len(sel) >= n:
                break
            row = df_scored.iloc[pos]
            ticker = str(row.get("ticker", ""))
            sector = str(row.get("setor", ""))
            subsector = str(row.get("subsetor", ""))
            theme = _theme(sector)
            company = ticker[:4]
            score = float(row.get("total_score", 0) or 0)

            if require_min_score and score < min_score:
                continue
            if company in seen_co:
                continue
            if sec_cnt.get(sector, 0) >= max_per_sector:
                continue
            if enforce_subsector and subsector and sub_cnt.get(subsector, 0) >= max_per_subsector:
                continue
            if theme_cnt.get(theme, 0) >= max_per_theme:
                continue

            sel.append(pos)
            seen_co.add(company)
            sec_cnt[sector] = sec_cnt.get(sector, 0) + 1
            if subsector:
                sub_cnt[subsector] = sub_cnt.get(subsector, 0) + 1
            theme_cnt[theme] = theme_cnt.get(theme, 0) + 1

        return sel, sec_cnt, sub_cnt, theme_cnt, seen_co

    # Cascata de fallback
    selected, *_ = _try_fill(range(len(df_scored)), require_min_score=True, enforce_subsector=True)
    if len(selected) < n:
        logger.debug("Fallback 1: relaxar sub-setor (n=%d/%d)", len(selected), n)
        selected, *_ = _try_fill(range(len(df_scored)), require_min_score=True, enforce_subsector=False)
    if len(selected) < n:
        logger.debug("Fallback 2: relaxar min_score (n=%d/%d)", len(selected), n)
        selected, *_ = _try_fill(range(len(df_scored)), require_min_score=False, enforce_subsector=False)
    if len(selected) < n:
        # Último recurso: preencher sem restrições para sempre retornar n
        logger.warning("Fallback 3: relaxar todas as restrições (n=%d/%d)", len(selected), n)
        already = set(selected)
        for pos in range(len(df_scored)):
            if len(selected) >= n:
                break
            if pos not in already:
                selected.append(pos)

    remaining = [p for p in range(len(df_scored)) if p not in set(selected)]
    result = df_scored.iloc[selected + remaining].reset_index(drop=True)

    # Anotar diagnóstico no DataFrame para o report_builder usar
    top_n = result.head(n)
    themes_in_top = [_theme(str(s)) for s in top_n.get("setor", []).tolist()]
    sectors_in_top = top_n.get("setor", []).tolist() if "setor" in top_n.columns else []
    # BRL exposure concentration: contar tickers com correlação positiva
    # (exportadores) e negativa (domésticos) significativas (|ρ| > 0.2)
    brl_signs: list[str] = []
    if "brl_corr_90d" in top_n.columns:
        for v in top_n["brl_corr_90d"]:
            if v is None or pd.isna(v):
                brl_signs.append("neutral")
            elif v > 0.2:
                brl_signs.append("exporter")    # ganha com BRL fraco
            elif v < -0.2:
                brl_signs.append("domestic")    # perde com BRL fraco
            else:
                brl_signs.append("neutral")

    result.attrs["portfolio_diagnostics"] = {
        "themes":          themes_in_top,
        "sectors":         sectors_in_top,
        "brl_signs":       brl_signs,
        "min_score":       float(top_n["total_score"].min()) if "total_score" in top_n.columns and not top_n.empty else None,
        "below_threshold": int(sum(1 for s in top_n.get("total_score", []) if s is not None and s < min_score)),
        "dominant_theme":  max(set(themes_in_top), key=themes_in_top.count) if themes_in_top else None,
        "dominant_theme_count": themes_in_top.count(max(set(themes_in_top), key=themes_in_top.count)) if themes_in_top else 0,
    }

    if "ticker" in df_scored.columns:
        original = df_scored.head(n)["ticker"].tolist()
        new = result.head(n)["ticker"].tolist()
        if original != new:
            logger.info(
                "Portfolio diversificado: %s → %s (dedupe empresa/subsetor/setor/tema, min_score=%.1f)",
                original, new, min_score,
            )

    return result


# Turnover band — anti-churn

def apply_turnover_band(
    df_scored: pd.DataFrame,
    incumbents: list[str],
    band_pts: float,
) -> pd.DataFrame:
    """
    Aplica bônus de TURNOVER_BAND_PTS no score dos incumbentes para reduzir
    rotação por flutuações pequenas (e estatisticamente não significativas).

    Lógica: incumbente só é deslocado se um candidato novo tem score > pontos
    suficientes acima. Isso reduz fricção real (corretagem, slippage, IR sobre
    ganho) e ruído de medição (delta de score < band_pts não é sinal).

    Não dá bônus a incumbentes com score raw abaixo de 50 (não rebalance para
    manter posição medíocre).

    Args:
        df_scored:  Output do scoring (DataFrame com coluna total_score).
        incumbents: Tickers atualmente na carteira (top 5 anterior).
        band_pts:   Bônus em pontos aplicado aos incumbentes.

    Returns:
        Novo DataFrame reordenado por adjusted_score.
    """
    if not incumbents or band_pts <= 0:
        return df_scored

    df = df_scored.copy()
    df["_adjusted_score"] = df["total_score"].astype(float)

    # Bônus só para incumbentes com score raw >= 50
    is_incumbent = df["ticker"].isin(incumbents) & (df["total_score"] >= 50.0)
    df.loc[is_incumbent, "_adjusted_score"] += band_pts

    n_boosted = int(is_incumbent.sum())
    logger.info(
        "Turnover band aplicada: +%.1f pts para %d incumbentes (%s)",
        band_pts, n_boosted,
        df.loc[is_incumbent, "ticker"].tolist(),
    )

    # Reordenar por adjusted_score
    df = df.sort_values("_adjusted_score", ascending=False).reset_index(drop=True)
    return df


# Função de conveniência para main.py
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
