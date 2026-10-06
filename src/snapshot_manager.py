"""
Persistência de cada execução — é daqui que o backtester tira os preços
de entrada.

Três artefatos por execução:
  1. snapshot_YYYY-MM-DD.json  — preços de fechamento de todos os tickers
                                  naquele pregão (entrada do backtester)
  2. recommendations_YYYY-MM-DD_MODE.json — Top 10 scores completos
  3. universe_YYYY-MM-DD.json  — estado do universe.csv naquele momento

Estrutura de diretórios esperada:
  data/history/
    snapshot_2025-01-06.json
    recommendations_2025-01-06_weekly.json
    universe_2025-01-06.json
    ...

O snapshot guarda preços de todo o universo (não só o top 5) para
análises post-hoc.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.b3_calendar import today_brt
from src.config import (
    ENABLE_VOLATILITY_TARGETING,
    EWMA_LAMBDA,
    HISTORY_DIR,
    HRP_COVARIANCE_METHOD,
    HRP_LOOKBACK_DAYS,
    MAX_POSITION_WEIGHT,
    MIN_POSITION_WEIGHT,
    UNIVERSE_FILE,
    USE_HRP_WEIGHTS,
    VOL_TARGET_ANNUAL,
    VOL_TARGET_LEVERAGE_MAX,
    VOL_TARGET_LEVERAGE_MIN,
    WEIGHTS,
)

logger = logging.getLogger(__name__)


# Tipos de retorno estruturados

@dataclass
class TickerRecommendation:
    """Uma linha do top-N."""
    ticker:               str
    nome:                 str
    score:                float
    score_breakdown:      dict[str, float]      # {fundamental, momentum, quality}
    normalization_method: str
    sector:               str
    sector_peers_count:   int
    why:                  str
    metrics:              dict[str, Any]        # pl, pvp, roe, roic, dy, divida_ebitda
    entry_price:          Optional[float]       # preço de fechamento na data da rec.
    norm_details:         Optional[dict] = field(default=None, repr=False)


@dataclass
class RecommendationRecord:
    """Registro completo de uma execução do sistema."""
    date:                     str
    mode:                     str                           # "weekly" | "monthly"
    universe_snapshot_path:   str
    top5:                     list[TickerRecommendation]
    top10:                    list[TickerRecommendation]
    weights:                  dict[str, float]
    entry_prices:             dict[str, float]             # todos os tickers do top-10
    execution_metadata:       dict[str, Any] = field(default_factory=dict)


# Helpers de serialização

def _safe_float(v: Any) -> Optional[float]:
    """Converte para float, retorna None em caso de NaN/inf."""
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)
    except (TypeError, ValueError):
        return None


def _to_json_safe(obj: Any) -> Any:
    """Torna qualquer objeto serializável em JSON (recursivo)."""
    if isinstance(obj, float):
        return _safe_float(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return _safe_float(float(obj))
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return str(obj)[:10]
    return obj


# SnapshotManager

class SnapshotManager:
    """
    Gerencia a persistência de snapshots e recomendações em data/history/.

    Convenção de nomes:
      snapshot_YYYY-MM-DD.json          — preços de fechamento (todos tickers)
      universe_YYYY-MM-DD.json          — estado do universe.csv
      recommendations_YYYY-MM-DD_MODE.json — scores e top-10

    Design:
      - Todos os arquivos são JSON UTF-8 (legíveis por humanos e por scripts)
      - Escritas atômicas: escreve para .tmp depois renomeia → sem arquivos corrompidos
      - IDs de data são strings ISO 8601 ("2025-01-06"), não timestamps Unix
    """

    def __init__(self, history_dir: Path = HISTORY_DIR):
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    # Escrita

    def save_price_snapshot(
        self,
        df_prices: pd.DataFrame,
        run_date: Optional[str | date] = None,
    ) -> Path:
        """
        Salva os preços de fechamento mais recentes de todos os tickers.

        Armazena apenas o último preço disponível no df_prices
        (preço de fechamento do pregão da recomendação).

        Returns:
            Path do arquivo salvo.
        """
        run_date_str = _date_str(run_date)
        path = self.history_dir / f"snapshot_{run_date_str}.json"

        if df_prices.empty:
            logger.warning("df_prices vazio — snapshot de preços não salvo.")
            return path

        # Último preço de fechamento disponível para cada ticker
        last_prices: dict[str, Optional[float]] = {}
        for ticker in df_prices.columns:
            series = df_prices[ticker].dropna()
            last_prices[ticker] = _safe_float(series.iloc[-1]) if not series.empty else None

        payload = {
            "date":        run_date_str,
            "snapshot_type": "prices",
            "pregao_date": str(df_prices.index[-1].date()) if not df_prices.empty else None,
            "tickers_count": len(last_prices),
            "prices":      last_prices,
        }

        _write_atomic(path, payload)
        logger.info("Snapshot de preços salvo: %s (%d tickers)", path.name, len(last_prices))
        return path

    def save_universe_snapshot(
        self,
        run_date: Optional[str | date] = None,
        universe_file: Path = UNIVERSE_FILE,
    ) -> Path:
        """
        Persiste o estado atual do universe.csv em JSON.

        Permite auditar se o universo mudou entre execuções
        (ex: tickers adicionados/removidos manualmente).
        """
        run_date_str = _date_str(run_date)
        path = self.history_dir / f"universe_{run_date_str}.json"

        try:
            df = pd.read_csv(universe_file)
            payload = {
                "date":          run_date_str,
                "snapshot_type": "universe",
                "tickers_count": len(df),
                "sectors":       df["setor"].value_counts().to_dict(),
                "tickers":       df.to_dict(orient="records"),
            }
            _write_atomic(path, payload)
            logger.info("Universe snapshot salvo: %s (%d tickers)", path.name, len(df))
        except Exception as exc:
            logger.error("Falha ao salvar universe snapshot: %s", exc)

        return path

    def save_recommendation(
        self,
        df_scored: pd.DataFrame,
        df_prices: pd.DataFrame,
        mode: str,
        run_date: Optional[str | date] = None,
        top_n: int = 10,
        allocation: Optional[dict] = None,
        trade_advice: Optional[dict] = None,
    ) -> Path:
        """
        Salva o output do ScoringEngine em JSON estruturado.

        Constrói obrigatoriamente:
          - top5: as 5 recomendações da semana (para Telegram + backtesting)
          - top10: ranking completo até 10 para referência
          - entry_prices: preços de entrada de todos os top-10 (para backtester)
          - universe_snapshot_path: referência cruzada com o snapshot de universo

        Args:
            df_scored:  Output do ScoringEngine.score(), ordenado por total_score DESC.
            df_prices:  DataFrame wide de preços (para capturar entry_price).
            mode:       "weekly" ou "monthly".
            run_date:   Data de referência; default = hoje.
            top_n:      Quantos tickers salvar além do top-5 (máx 10 recomendado).
            allocation: Saída do allocator (sleeves bolsa/CDI/global/inflação).
                        O gross_exposure do vol-targeting é aplicado AQUI
                        (bolsa↓ → CDI↑) para que o JSON persista a alocação
                        final executável.
            trade_advice: {ticker: {entry/stop/target/...}} do TradeAdvisor.
                        Persistido para o stop-monitor do closing diário.

        Returns:
            Path do arquivo JSON salvo.
        """
        run_date_str = _date_str(run_date)
        path = self.history_dir / f"recommendations_{run_date_str}_{mode}.json"

        universe_snap_path = str(
            self.history_dir / f"universe_{run_date_str}.json"
        )

        # Capturar preços de entrada dos top-N tickers
        entry_prices = _extract_entry_prices(df_scored, df_prices, top_n)

        # Pesos do top 5: HRP (Hierarchical Risk Parity) é robusto a correlações
        # em portfólios pequenos; fallback automático para inverse-vol se a lib
        # não estiver disponível ou se o cálculo falhar (matriz mal-condicionada).
        portfolio_weights, weights_method = _compute_portfolio_weights(
            df_scored, df_prices, n=5,
        )

        # Métricas de risco do portfólio para auditoria/relatório
        risk_metrics = _compute_portfolio_risk_metrics(
            portfolio_weights, df_prices,
        )

        # Volatility targeting: escalar gross exposure para vol target fixo.
        # Não muda os pesos RELATIVOS (HRP continua), apenas a quantidade
        # total investida (gross_exposure). Em ambiente de PF, usuário
        # interpreta como "% do capital alocado em ações vs caixa".
        vol_target_info = _apply_volatility_targeting(
            portfolio_weights, df_prices, risk_metrics,
        )

        # Allocation final: aplicar o gross do vol-target sobre o sleeve de
        # bolsa (capital liberado migra para CDI — caixa nunca fica "no ar").
        if allocation is not None:
            try:
                from src.allocator import apply_gross_exposure
                allocation = apply_gross_exposure(
                    allocation,
                    float(vol_target_info.get("gross_exposure", 1.0)),
                )
            except Exception as exc:
                logger.warning("apply_gross_exposure falhou (%s) — allocation crua", exc)

        # Validar cobertura: backtester depende de entry_prices para todo o top-N
        top_n_tickers = df_scored.head(top_n)["ticker"].tolist()
        missing_prices = [t for t in top_n_tickers if t not in entry_prices or entry_prices[t] is None]
        if missing_prices:
            logger.warning(
                "entry_prices incompleto: %d/%d tickers sem preço de entrada "
                "(%s). Backtesting desses tickers será excluído.",
                len(missing_prices), len(top_n_tickers), missing_prices,
            )
        else:
            logger.debug("entry_prices: cobertura 100%% (%d/%d tickers)", len(entry_prices), len(top_n_tickers))

        # Montar lista de recomendações
        top_recs: list[dict] = []
        for _, row in df_scored.head(top_n).iterrows():
            ticker = row.get("ticker", "")
            rec = _row_to_recommendation(row, entry_prices.get(ticker))
            top_recs.append(rec)

        top5  = top_recs[:5]
        top10 = top_recs[:top_n]

        # Scores do universo inteiro — o IC precisa deles, não só do top 10.
        # Sem isso, IC é medido apenas sobre top10 → selection bias enorme
        # (correlaciona fator com retorno SÓ entre ações que o fator já
        # selecionou). Salvar TODOS os tickers com seus scores normalizados
        # permite IC real cross-sectional sobre o universo completo.
        full_universe_scores = _extract_full_universe_scores(df_scored)

        payload = {
            "date":                   run_date_str,
            "mode":                   mode,
            "universe_snapshot":      universe_snap_path,
            "top5":                   top5,
            "top10":                  top10,
            "weights":                WEIGHTS,
            "entry_prices":           entry_prices,
            "portfolio_weights":      portfolio_weights,
            "allocation":             allocation,
            "trade_advice":           trade_advice,
            "full_universe_scores":   full_universe_scores,
            "execution_metadata": {
                "universe_size":      len(df_scored),
                "tickers_scored":     int(df_scored["total_score"].notna().sum()),
                # Cobertura declarado→coletado→pontuado (de main.py via attrs).
                "data_quality":       getattr(df_scored, "attrs", {}).get("data_quality"),
                "market_data":        getattr(df_scored, "attrs", {}).get("market_data"),
                "generated_at":       datetime.now().isoformat(),
                "portfolio_weights_method": weights_method,
                "risk_metrics":             risk_metrics,
                "vol_targeting":            vol_target_info,
                "score_range": {
                    "max": _safe_float(df_scored["total_score"].max()),
                    "min": _safe_float(df_scored["total_score"].min()),
                    "p50": _safe_float(df_scored["total_score"].median()),
                },
            },
        }

        _write_atomic(path, _to_json_safe(payload))
        logger.info(
            "Recomendação salva: %s (top5: %s)",
            path.name,
            [r["ticker"] for r in top5],
        )
        return path

    # Leitura — usada pelo backtester.py

    def load_recommendation(
        self,
        run_date: str | date,
        mode: str = "weekly",
    ) -> Optional[dict]:
        """
        Carrega o JSON de recomendação de uma data específica.

        Returns:
            dict com a estrutura completa, ou None se o arquivo não existir.
        """
        run_date_str = _date_str(run_date)
        path = self.history_dir / f"recommendations_{run_date_str}_{mode}.json"
        return _read_json(path)

    def load_latest_recommendation(
        self,
        mode: str = "weekly",
        before_date: Optional[str | date] = None,
    ) -> Optional[dict]:
        """
        Carrega o JSON de recomendação mais recente disponível.

        Itera em ordem reversa pelos arquivos recommendations_*_{mode}.json
        e retorna o primeiro encontrado.

        Args:
            mode:        "weekly" ou "monthly".
            before_date: Se fornecido, retorna a recomendação mais recente com
                         data ESTRITAMENTE anterior a `before_date`. Crítico
                         para o backtester: sem isso, ele carregaria a
                         recomendação recém-salva do próprio run (mesma data)
                         e compararia a carteira contra os preços do mesmo dia
                         (period_days=0, retorno = apenas fricção). Com o filtro,
                         o backtest compara sempre contra a carteira anterior real.

        Returns:
            dict com a recomendação correspondente, ou None se não houver nenhuma.
        """
        pattern = f"recommendations_*_{mode}.json"
        files = sorted(self.history_dir.glob(pattern), reverse=True)
        if not files:
            logger.info("Nenhuma recomendação %s encontrada em %s", mode, self.history_dir)
            return None

        cutoff = _date_str(before_date) if before_date is not None else None
        for f in files:
            rec = _read_json(f)
            if rec is None:
                continue
            if cutoff is not None and str(rec.get("date", "")) >= cutoff:
                # Pular recomendações na mesma data (ou futuras) — evita
                # backtest da recomendação contra si mesma.
                continue
            return rec

        if cutoff is not None:
            logger.info(
                "Nenhuma recomendação %s anterior a %s encontrada.", mode, cutoff,
            )
        return None

    def load_price_snapshot(self, run_date: str | date) -> Optional[dict[str, float]]:
        """
        Carrega o snapshot de preços de uma data.

        Returns:
            dict {ticker: price} ou None se o arquivo não existir.
        """
        run_date_str = _date_str(run_date)
        path = self.history_dir / f"snapshot_{run_date_str}.json"
        data = _read_json(path)
        return data.get("prices") if data else None

    def list_recommendations(self, mode: str = "weekly") -> list[dict]:
        """
        Lista todas as recomendações disponíveis em ordem cronológica.

        Returns:
            Lista de dicts com campos: date, mode, path, top5_tickers.
        """
        pattern = f"recommendations_*_{mode}.json"
        files = sorted(self.history_dir.glob(pattern))
        result = []
        for f in files:
            data = _read_json(f)
            if data:
                result.append({
                    "date":         data.get("date"),
                    "mode":         data.get("mode"),
                    "path":         str(f),
                    "top5_tickers": [r.get("ticker") for r in data.get("top5", [])],
                    "top5_scores":  [r.get("score") for r in data.get("top5", [])],
                })
        return result

    def get_entry_prices_for_backtesting(
        self,
        recommendation: dict,
    ) -> dict[str, float]:
        """
        Extrai os preços de entrada a partir de uma recomendação carregada.

        O backtester.py usa esses preços para calcular o retorno da carteira
        da semana/mês anterior comparado ao preço atual.

        Args:
            recommendation: dict retornado por load_recommendation()

        Returns:
            {ticker: entry_price} para os tickers do top-5.
        """
        entry_prices = recommendation.get("entry_prices", {})
        top5_tickers = [r["ticker"] for r in recommendation.get("top5", [])]
        return {t: entry_prices[t] for t in top5_tickers if t in entry_prices}


# Helpers privados

def _date_str(d: Optional[str | date | datetime]) -> str:
    """Normaliza para string ISO 8601 'YYYY-MM-DD'."""
    if d is None:
        return today_brt().isoformat()
    if isinstance(d, datetime):
        return d.date().isoformat()
    if isinstance(d, date):
        return d.isoformat()
    return str(d)[:10]


def _write_atomic(path: Path, payload: dict) -> None:
    """
    Escrita atômica: escreve em .tmp e renomeia.
    Previne arquivos corrompidos se o processo for interrompido.
    """
    tmp_path = path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp_path, path)   # rename atômico no mesmo filesystem
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Falha ao salvar {path}: {exc}") from exc


def _read_json(path: Path) -> Optional[dict]:
    """Lê um JSON com tratamento de erros."""
    if not path.exists():
        logger.debug("Arquivo não encontrado: %s", path)
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Falha ao ler %s: %s", path, exc)
        return None


def _extract_entry_prices(
    df_scored: pd.DataFrame,
    df_prices: pd.DataFrame,
    top_n: int,
) -> dict[str, float]:
    """Captura o último preço disponível para os top-N tickers."""
    prices: dict[str, float] = {}
    for _, row in df_scored.head(top_n).iterrows():
        ticker = row.get("ticker", "")
        # Primeiro: usar current_price do DataFrame de fundamentos
        cp = _safe_float(row.get("current_price"))
        if cp is not None:
            prices[ticker] = cp
            continue
        # Fallback: último preço no df_prices
        if ticker in df_prices.columns:
            series = df_prices[ticker].dropna()
            if not series.empty:
                prices[ticker] = round(float(series.iloc[-1]), 2)
    return prices


def _extract_full_universe_scores(df_scored: pd.DataFrame) -> dict[str, dict]:
    """
    Extrai factor scores normalizados de TODOS os tickers do df_scored,
    não só top10. Cada ticker → {fator: score [0-100]}.

    Lê norm_details que o ScoringEngine popula em build_norm_details_col.
    Filtra apenas scores válidos (não-None).
    """
    out: dict[str, dict] = {}
    if "ticker" not in df_scored.columns or "norm_details" not in df_scored.columns:
        return out

    for _, row in df_scored.iterrows():
        ticker = str(row.get("ticker", ""))
        if not ticker:
            continue
        nd = row.get("norm_details") or {}
        if isinstance(nd, str):
            try:
                import ast
                nd = ast.literal_eval(nd)
            except Exception:
                nd = {}
        if not isinstance(nd, dict):
            continue
        factor_scores: dict[str, float] = {}
        for factor, detail in nd.items():
            if isinstance(detail, dict):
                score = detail.get("score")
                if score is not None:
                    factor_scores[factor] = round(float(score), 2)
        if factor_scores:
            out[ticker] = factor_scores
    return out


def _apply_volatility_targeting(
    weights: dict[str, float],
    df_prices: pd.DataFrame,
    risk_metrics: dict,
) -> dict:
    """
    Calcula o gross exposure ideal para atingir VOL_TARGET_ANNUAL.

    Vol da carteira estimada via EWMA (RiskMetrics λ=0.94) sobre retornos
    da carteira simulada com os HRP weights. Half-life da EWMA = ln(2)/(1-λ)
    ≈ 11 dias — responsivo a mudanças de regime sem ser ruidoso.

    gross_exposure = clip(vol_target / vol_ewma, MIN, MAX)

    Em low-vol regime (vol_ewma < vol_target), exposure > 100% (mas capamos
    em 150% — PF não usa margin). Em high-vol regime, < 100%, sugerindo
    manter capital em caixa/CDI.

    Importante: NÃO altera weights relativos (HRP continua governando a
    composição). Só sinaliza quanto do capital deve estar em risco.
    """
    out = {
        "enabled":          ENABLE_VOLATILITY_TARGETING,
        "vol_target":       VOL_TARGET_ANNUAL,
        "vol_estimated":    None,
        "gross_exposure":   1.0,
        "scaled_weights":   dict(weights),
    }
    if not ENABLE_VOLATILITY_TARGETING or not weights or df_prices is None or df_prices.empty:
        return out

    avail = [t for t in weights.keys() if t in df_prices.columns]
    if not avail:
        return out

    # Renormalizar pesos sobre tickers disponíveis
    w_sum = sum(weights[t] for t in avail)
    if w_sum == 0:
        return out
    w_arr = np.array([weights[t] / w_sum for t in avail])

    # Retornos diários da carteira simulada (último ano)
    returns = df_prices[avail].tail(252).pct_change().dropna()
    if len(returns) < 30:
        return out

    portfolio_daily = returns.to_numpy() @ w_arr

    # EWMA RiskMetrics — variância recursiva com decay λ=0.94
    # σ²_t = λ · σ²_{t-1} + (1-λ) · r²_{t-1}
    lam = EWMA_LAMBDA
    ewma_var = float(portfolio_daily[0] ** 2)
    for r in portfolio_daily[1:]:
        ewma_var = lam * ewma_var + (1 - lam) * (r ** 2)

    vol_ewma_annual = float(np.sqrt(ewma_var * 252))
    if vol_ewma_annual <= 0:
        return out

    raw_lev = VOL_TARGET_ANNUAL / vol_ewma_annual
    gross_exposure = float(np.clip(raw_lev, VOL_TARGET_LEVERAGE_MIN, VOL_TARGET_LEVERAGE_MAX))

    scaled = {t: round(weights[t] * gross_exposure, 6) for t in weights}

    out.update({
        "vol_estimated":  round(vol_ewma_annual, 4),
        "gross_exposure": round(gross_exposure, 4),
        "raw_leverage":   round(raw_lev, 4),
        "scaled_weights": scaled,
        "ewma_lambda":    lam,
    })
    logger.info(
        "Vol targeting: vol_est=%.1f%% a.a. → exposure=%.0f%% (target %.0f%%)",
        vol_ewma_annual * 100, gross_exposure * 100, VOL_TARGET_ANNUAL * 100,
    )
    return out


def _compute_portfolio_risk_metrics(
    weights: dict[str, float],
    df_prices: pd.DataFrame,
    lookback_days: int = 252,
    confidence: float = 0.95,
) -> dict:
    """
    Métricas de risco do portfólio (top 5) baseadas em simulação histórica.

    VaR (Value-at-Risk) 95%: percentil 5% da distribuição de retornos diários.
      "Há 5% de chance de perder mais que VaR% num dia típico."
    CVaR (Expected Shortfall) 95%: média dos retornos abaixo do VaR.
      "Quando o cenário ruim acontece, perda média esperada."
    HHI (Herfindahl-Hirschman): Σ wᵢ². 0.20 = perfect equal-weight em 5,
      1.0 = 100% em 1 ticker.
    Effective N: 1 / HHI. Número equivalente de posições.
    Annualized vol: std × √252.
    Sharpe simplificado: ret_mean / std × √252 (sem risk-free).

    Returns dict — todos os valores são None se simulação não puder rodar.
    """
    out: dict = {
        "var_95":        None,
        "cvar_95":       None,
        "hhi":           None,
        "effective_n":   None,
        "ann_vol":       None,
        "sharpe_naive":  None,
        "n_obs":         0,
    }
    if not weights or df_prices is None or df_prices.empty:
        return out

    tickers = list(weights.keys())
    avail = [t for t in tickers if t in df_prices.columns]
    if not avail:
        return out

    # Renormalizar para os tickers disponíveis
    w_sum = sum(weights[t] for t in avail)
    if w_sum == 0:
        return out
    w = np.array([weights[t] / w_sum for t in avail])

    # Retornos diários simulados (carteira rebalanceada diariamente — proxy)
    prices_sub = df_prices[avail].tail(lookback_days)
    returns = prices_sub.pct_change().dropna(how="all").to_numpy()
    if len(returns) < 30:
        out["hhi"] = float(np.sum(w ** 2))
        out["effective_n"] = round(1 / out["hhi"], 2) if out["hhi"] > 0 else None
        return out

    # Replace NaN with column means for simulation purposes
    col_means = np.nanmean(returns, axis=0)
    nan_mask = np.isnan(returns)
    returns_clean = np.where(nan_mask, col_means, returns)

    # Retorno da carteira em cada dia
    portfolio_daily = returns_clean @ w

    var_pct = (1 - confidence) * 100  # 5%
    var_val = float(np.percentile(portfolio_daily, var_pct))
    cvar_val = float(portfolio_daily[portfolio_daily <= var_val].mean()) if (portfolio_daily <= var_val).any() else var_val

    hhi = float(np.sum(w ** 2))
    eff_n = float(1.0 / hhi) if hhi > 0 else None
    ann_vol = float(portfolio_daily.std(ddof=1) * np.sqrt(252))
    ann_ret = float(portfolio_daily.mean() * 252)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else None

    out.update({
        "var_95":        round(var_val, 5),
        "cvar_95":       round(cvar_val, 5),
        "hhi":           round(hhi, 4),
        "effective_n":   round(eff_n, 2) if eff_n is not None else None,
        "ann_vol":       round(ann_vol, 4),
        "sharpe_naive":  round(sharpe, 3) if sharpe is not None else None,
        "n_obs":         len(portfolio_daily),
    })
    return out


def _compute_portfolio_weights(
    df_scored: pd.DataFrame,
    df_prices: pd.DataFrame,
    n: int = 5,
) -> tuple[dict[str, float], str]:
    """
    Calcula pesos do top-N usando HRP (Hierarchical Risk Parity) com fallback.

    HRP (López de Prado 2016) tem três vantagens sobre inverse-volatility
    para portfólios pequenos (5-10 ativos):
      1. Respeita estrutura de correlação via clustering hierárquico
      2. Robusto a matriz de covariância mal-condicionada (comum em N pequeno)
      3. Out-of-sample bate Markowitz e iguala-ou-bate inverse-vol

    Fluxo:
      1. Se USE_HRP_WEIGHTS=False → inverse-vol direto
      2. Se Riskfolio-Lib ou retornos insuficientes → fallback inverse-vol
      3. Caso contrário → HRP com clustering single linkage, correlação Pearson

    Returns:
        (weights_dict, method_used) — método: "hrp" | "inverse_volatility" | "equal"
    """
    top_n = df_scored.head(n)
    tickers = [str(row.get("ticker", "")) for _, row in top_n.iterrows()]

    if not USE_HRP_WEIGHTS:
        return _compute_inv_vol_weights(df_scored, n), "inverse_volatility"

    # Tentar HRP
    try:
        import riskfolio as rp
    except ImportError:
        logger.info("riskfolio-lib não instalado — usando inverse-vol")
        return _compute_inv_vol_weights(df_scored, n), "inverse_volatility"

    if df_prices is None or df_prices.empty:
        logger.debug("df_prices vazio — fallback inverse-vol")
        return _compute_inv_vol_weights(df_scored, n), "inverse_volatility"

    # Selecionar apenas tickers presentes em df_prices
    avail = [t for t in tickers if t in df_prices.columns]
    if len(avail) < 2:
        logger.debug("HRP precisa >=2 tickers — fallback")
        return _compute_inv_vol_weights(df_scored, n), "inverse_volatility"

    # Retornos diários dos últimos HRP_LOOKBACK_DAYS
    prices_sub = df_prices[avail].dropna(how="all").tail(HRP_LOOKBACK_DAYS + 1)
    returns = prices_sub.pct_change().dropna(how="any")
    if len(returns) < 30:  # pelo menos ~6 semanas de dados
        logger.debug("HRP: poucos retornos (%d) — fallback", len(returns))
        return _compute_inv_vol_weights(df_scored, n), "inverse_volatility"

    try:
        port = rp.HCPortfolio(returns=returns)
        # method_cov:
        #   "ledoit" — Ledoit-Wolf shrinkage para identidade. Vantagem:
        #              reduz erro out-of-sample 15-30% quando p/N > 0.1
        #              (Ledoit-Wolf 2003). Caveat: target é identidade,
        #              não constant-correlation — perde alguma estrutura
        #              setorial. Para top-5, esse caveat é menor que o ganho.
        #   "oas"    — Oracle Approximating Shrinkage; melhor sob hipótese
        #              Gaussiana e N pequeno.
        #   "hist"   — covariância amostral (sem shrinkage).
        # δ é derivado analiticamente — NÃO setar shrinkage_constant manual.
        # linkage "ward" em vez de "single": single-linkage sofre de chaining
        # (clusters degenerados em N pequeno) e foi a causa direta da
        # concentração de 63% em NEOE3 — todos os demais ativos viraram um
        # único cluster correlacionado e o low-vol isolado levou o peso.
        # Ward produz clusters balanceados e é o padrão em implementações
        # HRP modernas para N < 20.
        w = port.optimization(
            model="HRP",
            codependence="pearson",
            method_cov=HRP_COVARIANCE_METHOD,
            rm="MV",
            rf=0,
            linkage="ward",
            max_k=10,
            leaf_order=True,
        )
        if w is None or w.empty:
            raise ValueError("HRP retornou DataFrame vazio")

        # w é DataFrame com index=tickers, coluna 'weights'
        weights_col = w.columns[0]
        weights = {t: float(w.loc[t, weights_col]) for t in avail if t in w.index}

        # Tickers do top-N que não tiveram preço → distribuir resto igualmente
        missing = [t for t in tickers if t not in weights]
        if missing:
            remaining = 1.0 - sum(weights.values())
            if remaining > 0:
                share = remaining / len(missing)
                for t in missing:
                    weights[t] = share

        # Normalizar para somar 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {t: w / total for t, w in weights.items()}

        # Cap/floor por posição — HRP cru concentra em low-vol (ver
        # _apply_weight_bounds). Aplicado por último, sobre pesos normalizados.
        weights = _apply_weight_bounds(weights)

        logger.info("Pesos HRP (com bounds): %s", weights)
        return weights, "hrp"

    except Exception as exc:
        logger.warning("HRP falhou (%s) — fallback inverse-vol", exc)
        return _compute_inv_vol_weights(df_scored, n), "inverse_volatility"


def _compute_inv_vol_weights(df_scored: pd.DataFrame, n: int = 5) -> dict[str, float]:
    """
    Inverse-volatility portfolio weights for top-N tickers.

    W_i = (1/σ_i) / Σ(1/σ_j)

    Tickers with missing volatility receive the mean inverse-vol weight of
    available tickers. Falls back to equal-weight if no volatility data at all.
    """
    top_n = df_scored.head(n)
    tickers = [str(row.get("ticker", "")) for _, row in top_n.iterrows()]

    inv_vols: dict[str, float] = {}
    for ticker, (_, row) in zip(tickers, top_n.iterrows()):
        v = _safe_float(row.get("volatility_180d"))
        if v and v > 0:
            inv_vols[ticker] = 1.0 / v

    if not inv_vols:
        eq = round(1.0 / len(tickers), 6)
        return {t: eq for t in tickers}

    total_known = sum(inv_vols.values())
    mean_iv = total_known / len(inv_vols)
    n_missing = len(tickers) - len(inv_vols)

    full_weights = {t: inv_vols.get(t, mean_iv) for t in tickers}
    total_full = total_known + mean_iv * n_missing

    return _apply_weight_bounds(
        {t: w / total_full for t, w in full_weights.items()}
    )


def _apply_weight_bounds(
    weights: dict[str, float],
    cap: float = MAX_POSITION_WEIGHT,
    floor: float = MIN_POSITION_WEIGHT,
) -> dict[str, float]:
    """
    Aplica cap/floor por posição com redistribuição proporcional iterativa.

    Sem isso, HRP em portfólios de 5 ativos concentra no ativo de menor vol
    (NEOE3 recebeu 63,4% em 08/06/2026). O excedente acima do cap é
    redistribuído proporcionalmente entre as posições não-capadas; posições
    abaixo do floor são elevadas. Se cap×N < 1 (bounds inviáveis), degrada
    para equal-weight.
    """
    n = len(weights)
    if n == 0:
        return weights
    if cap * n < 1.0 or floor * n > 1.0:
        logger.warning(
            "Bounds inviáveis (cap=%.2f, floor=%.2f, n=%d) — equal-weight",
            cap, floor, n,
        )
        return {t: round(1.0 / n, 6) for t in weights}

    # Normalizar entrada (defensivo)
    total = sum(max(v, 0.0) for v in weights.values())
    if total <= 0:
        return {t: round(1.0 / n, 6) for t in weights}
    w = {t: max(v, 0.0) / total for t, v in weights.items()}

    for _ in range(50):
        w = {t: max(v, floor) for t, v in w.items()}
        capped = {t for t, v in w.items() if v >= cap}
        free = [t for t in w if t not in capped]
        if not free:
            w = {t: 1.0 / n for t in w}
            break
        w = {t: (cap if t in capped else w[t]) for t in w}
        remaining = 1.0 - cap * len(capped)
        free_sum = sum(w[t] for t in free)
        if free_sum <= 0:
            share = remaining / len(free)
            w.update({t: share for t in free})
        else:
            w.update({t: w[t] / free_sum * remaining for t in free})
        # Convergiu se nenhuma posição livre viola bounds
        if all(floor - 1e-9 <= w[t] <= cap + 1e-9 for t in free):
            break

    total = sum(w.values())
    bounded = {t: round(v / total, 6) for t, v in w.items()}
    if bounded != {t: round(v, 6) for t, v in weights.items()}:
        logger.info("Pesos após bounds (cap=%.0f%%, floor=%.0f%%): %s",
                    cap * 100, floor * 100, bounded)
    return bounded


def _row_to_recommendation(row: pd.Series, entry_price: Optional[float]) -> dict:
    """Converte uma linha do df_scored para o schema de recomendação."""
    ticker = str(row.get("ticker", ""))

    # Capturar peers_count do norm_details se disponível
    norm_details = row.get("norm_details", {})
    if isinstance(norm_details, str):
        try:
            import ast
            norm_details = ast.literal_eval(norm_details)
        except Exception:
            norm_details = {}

    # Estimar sector_peers_count: pegar de qualquer fator fundamental
    sector_peers_count = 0
    for factor_detail in (norm_details or {}).values():
        if isinstance(factor_detail, dict) and "n_peers" in factor_detail:
            sector_peers_count = int(factor_detail["n_peers"])
            break

    metrics = {
        "pl":            _safe_float(row.get("pl")),
        "earnings_yield":_safe_float(row.get("earnings_yield")),
        "pvp":           _safe_float(row.get("pvp")),
        "roe":           _safe_float(row.get("roe")),
        "roic":          _safe_float(row.get("roic")),
        "divida_ebitda": _safe_float(row.get("divida_ebitda")),
        "dy":            _safe_float(row.get("dividend_yield")),
        "alpha_6m":      _safe_float(row.get("alpha_6m")),
        "volatility":    _safe_float(row.get("volatility_180d")),
        "beta":          _safe_float(row.get("beta")),
    }

    fund_score = _safe_float(row.get("fundamental_score")) or 0.0
    mom_score  = _safe_float(row.get("momentum_score"))    or 0.0
    qual_score = _safe_float(row.get("quality_score"))     or 0.0

    return {
        "ticker":               ticker,
        "nome":                 str(row.get("nome", "")),
        "score":                _safe_float(row.get("total_score")),
        "score_breakdown": {
            "fundamental": round(WEIGHTS["fundamental"] * fund_score, 2),
            "momentum":    round(WEIGHTS["momentum"]    * mom_score,  2),
            "quality":     round(WEIGHTS["quality"]     * qual_score, 2),
        },
        "normalization_method": str(row.get("normalization_method", "")),
        "conviction":           _safe_float(row.get("conviction")),
        "conviction_label":     str(row.get("conviction_label", "Baixa")),
        "sector":               str(row.get("setor", "")),
        "sector_peers_count":   sector_peers_count,
        "why":                  str(row.get("why", "")),
        "metrics":              metrics,
        "entry_price":          entry_price,
        "norm_details":         norm_details if norm_details else None,
    }
