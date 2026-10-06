"""
main.py — Orquestrador do B3 Stock Recommender

Executa o pipeline completo de recomendação semanal/mensal:
  1. Coleta dados (brapi.dev + yfinance fallback)
  2. Calcula scores (scoring_engine)
  3. Backtesta carteira anterior (backtester)
  4. Gera gráfico PNG (chart_generator)
  5. Constrói relatório de texto (report_builder)
  6. Envia ao Telegram (telegram_sender)
  7. Persiste snapshots e recomendações (snapshot_manager)

Flags CLI:
  --mode weekly | monthly   Período do relatório (default: weekly)
  --date YYYY-MM-DD         Data de referência (default: hoje)
  --dry-run                 Executa tudo mas não envia para o Telegram
  --send                    Envia ao Telegram (requerido explicitamente para evitar acidentes)
  --no-chart                Pula geração do gráfico
  --debug                   Logging em nível DEBUG
  --validate-token          Valida o token do Telegram e encerra

Exit codes:
  0  Sucesso
  1  Falha crítica (dados insuficientes, erro de envio)
  2  Erro de configuração (token ausente, modo inválido)
"""

import argparse
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

# Carrega .env antes de qualquer import que leia os.getenv()
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# Garante que src/ seja encontrado mesmo rodando main.py da raiz
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    OUTPUT_DIR,
    HISTORY_DIR,
    CACHE_DIR,
    ENABLE_ASSET_ALLOCATION,
    LOG_LEVEL,
    CHART_OUTPUT_PATH,
    HMM_MIN_HISTORY_DAYS,
    HMM_N_STATES,
    HMM_RANDOM_STATE,
    MIN_UNIVERSE_COVERAGE,
    TURNOVER_BAND_PTS,
    UNIVERSE_FILE,
    USE_HMM_REGIME,
)
from src.data_collector import load_data
from src.b3_calendar import today_brt
from src.scoring_engine import apply_turnover_band, compute_scores, select_diverse_portfolio
from src.backtester import run_backtest
from src.benchmark import BenchmarkManager, get_ibov_prices as _get_ibov_prices
from src.report_builder import build_report
from src.chart_generator import ChartGenerator
from src.snapshot_manager import SnapshotManager
from src.technical_analyzer import TechnicalAnalyzer
from src.trade_advisor import TradeAdvisor
from src.telegram_sender import send_report, TelegramError

# Logging

def _setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else getattr(logging, LOG_LEVEL, logging.INFO)
    fmt   = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")

logger = logging.getLogger(__name__)

# CLI

def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="B3 Stock Recommender — Relatório Semanal/Mensal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["weekly", "monthly"],
        default="weekly",
        help="Período do relatório (default: weekly)",
    )
    parser.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Data de referência (default: hoje)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Executa tudo mas não envia para o Telegram",
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="Envia ao Telegram (necessário para envio real)",
    )
    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="Pula geração do gráfico PNG",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Logging em nível DEBUG",
    )
    parser.add_argument(
        "--validate-token",
        action="store_true",
        help="Valida o token do Telegram e encerra",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=float(os.getenv("PORTFOLIO_CAPITAL_BRL", "0") or 0),
        metavar="R$",
        help="Capital total em R$ — gera folha de ordens executável "
             "(qty no fracionário, sleeves em R$, nota de IR). "
             "Também aceita env PORTFOLIO_CAPITAL_BRL.",
    )
    return parser.parse_args(argv)


# Validação de token

def _validate_token() -> int:
    from src.telegram_sender import TelegramSender
    try:
        sender = TelegramSender()
        info = sender.get_me()
        bot_name = info.get("result", {}).get("username", "?")
        print(f"Token válido. Bot: @{bot_name}")
        return 0
    except TelegramError as exc:
        print(f"Token inválido: {exc}", file=sys.stderr)
        return 2


# Preparação de diretórios

def _ensure_dirs() -> None:
    for d in (OUTPUT_DIR, HISTORY_DIR, CACHE_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)


# Parsing de data

def _resolve_date(date_arg: Optional[str]) -> str:
    if date_arg:
        try:
            datetime.strptime(date_arg, "%Y-%m-%d")
            return date_arg
        except ValueError:
            logger.error("Formato de data inválido: %s (esperado YYYY-MM-DD)", date_arg)
            sys.exit(2)
    return today_brt().isoformat()


# Regime de mercado

def _fetch_vix_prices(start_date: str):
    """Download VIX from yfinance. Returns empty Series on failure."""
    import pandas as pd
    import yfinance as yf
    try:
        raw = yf.download("^VIX", start=start_date, auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return pd.Series(dtype=float)
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close.dropna()
    except Exception as exc:
        logger.debug("VIX fetch falhou: %s", exc)
        import pandas as pd
        return pd.Series(dtype=float)


def _fetch_usdbrl_returns(n_days: int):
    """
    Busca série de log-returns diários do USDBRL para overlay macro no HMM.

    Fonte primária: BCB SGS série 1 (PTAX venda) via python-bcb — oficial.
    Fallback: yfinance "BRL=X" — disponível sempre, mesmo timezone.

    Retorna pd.Series indexada por data, ou None se ambas falharem.
    """
    import pandas as pd
    import numpy as np

    # Primary: BCB PTAX
    try:
        from bcb import sgs
        start = (today_brt() - timedelta(days=max(n_days + 30, 400))).strftime("%Y-%m-%d")
        df = sgs.get({"USDBRL": 1}, start=start)
        if df is not None and not df.empty:
            s = df["USDBRL"].dropna().astype(float)
            if len(s) > 1:
                log_ret = pd.Series(
                    data=np.log(s.values[1:] / s.values[:-1]),
                    index=s.index[1:],
                )
                if hasattr(log_ret.index, "tz") and log_ret.index.tz is not None:
                    log_ret.index = log_ret.index.tz_localize(None)
                return log_ret
    except Exception as exc:
        logger.debug("BCB USDBRL falhou (%s) — tentando yfinance", exc)

    # Fallback: yfinance
    try:
        import yfinance as yf
        df = yf.download("BRL=X", period=f"{max(n_days, 252) + 30}d",
                         auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df["Close"].dropna()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        if hasattr(close.index, "tz") and close.index.tz is not None:
            close.index = close.index.tz_localize(None)
        log_ret = np.log(close / close.shift(1)).dropna()
        return log_ret
    except Exception as exc:
        logger.debug("USDBRL fallback falhou (%s)", exc)
        return None


MIN_PRICE_ROWS_FOR_COVERAGE = 20
_FUND_COLS = ("pl", "pvp", "roe", "roic", "divida_ebitda", "dividend_yield")


def coverage_report(df_fundamentals, df_prices, df_scored, universe_file) -> dict:
    """
    Declarado → coletado → pontuado. "Coletado" = tem histórico de preço e o
    mínimo de fundamentos; o collector cria uma linha por ticker mesmo quando
    as fontes falham, então contar linhas não diz nada.
    """
    import pandas as pd
    from src.config import MIN_FUNDAMENTALS_REQUIRED

    try:
        universe = pd.read_csv(universe_file)["ticker"].astype(str).tolist()
    except Exception:
        universe = []

    with_prices = set()
    if df_prices is not None and not df_prices.empty:
        counts = df_prices.notna().sum()
        with_prices = {t for t, n in counts.items() if n >= MIN_PRICE_ROWS_FOR_COVERAGE}

    with_funds = set()
    if df_fundamentals is not None and not df_fundamentals.empty and "ticker" in df_fundamentals:
        cols = [c for c in _FUND_COLS if c in df_fundamentals.columns]
        ok = df_fundamentals[cols].notna().sum(axis=1) >= MIN_FUNDAMENTALS_REQUIRED - 1
        with_funds = set(df_fundamentals.loc[ok, "ticker"].astype(str))

    scored = set()
    if "ticker" in df_scored.columns:
        mask = df_scored["total_score"].notna() if "total_score" in df_scored.columns else True
        scored = set(df_scored.loc[mask, "ticker"].astype(str))

    base = universe or sorted(with_prices | with_funds | scored)
    collected = [t for t in base if t in with_prices and t in with_funds]
    declared = len(universe) or None
    return {
        "declared_universe": declared,
        "collected":         len(collected),
        "scored":            len(scored),
        "coverage_pct":      round(len(scored) / declared, 3) if declared else None,
        "no_price_history":  sorted(t for t in base if t not in with_prices),
        "no_fundamentals":   sorted(t for t in base if t not in with_funds),
        "filtered_out":      sorted(t for t in collected if t not in scored),
    }


def _detect_regime_binary(ibov_prices, vix_prices) -> str:
    """
    Detector binário legado (fallback). Regras:
      risk_on:  IBOV > MA200 AND VIX < 18
      bear:     VIX > 25
      mean_rev: caso contrário
    """
    import pandas as pd
    ibov_clean = ibov_prices.dropna() if not ibov_prices.empty else pd.Series(dtype=float)

    ibov_above_ma200 = False
    if len(ibov_clean) >= 200:
        ma200 = float(ibov_clean.tail(200).mean())
        ibov_above_ma200 = float(ibov_clean.iloc[-1]) > ma200

    latest_vix = None
    vix_clean = vix_prices.dropna() if not vix_prices.empty else pd.Series(dtype=float)
    if not vix_clean.empty:
        latest_vix = float(vix_clean.iloc[-1])

    if latest_vix is not None and latest_vix > 25:
        return "bear"
    if ibov_above_ma200 and (latest_vix is None or latest_vix < 18):
        return "risk_on"
    return "mean_rev"


def _detect_regime_hmm(ibov_prices, vix_prices) -> Optional[str]:
    """
    HMM 2-state (bull/bear) com features multi-fonte.

    Features:
      1. log_return IBOV — price direction
      2. vol_21d clipped — risk regime
      3. log_return USDBRL — Brazil-specific FX stress (NEW)

    Caveats:
      - GaussianHMM assume emissão Gaussiana — winsorizamos retornos
      - Label switching: re-rotular pelos means_ ordenados de μ_ibov
      - covariance_type="diag" + min_covar=1e-5 evita colapso bear→outliers

    Mapeamento:
      P(bull) > 0.6 → risk_on
      P(bear) > 0.6 → bear
      caso contrário → mean_rev

    Returns None se hmmlearn indisponível ou histórico insuficiente.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        return None

    import numpy as np
    import pandas as pd

    ibov_clean = ibov_prices.dropna() if not ibov_prices.empty else pd.Series(dtype=float)
    if len(ibov_clean) < HMM_MIN_HISTORY_DAYS:
        logger.debug("HMM: histórico curto (%d < %d)", len(ibov_clean), HMM_MIN_HISTORY_DAYS)
        return None

    # Log returns IBOV, winsorizados a ±3σ
    log_ret = np.log(ibov_clean / ibov_clean.shift(1)).dropna()
    if len(log_ret) < HMM_MIN_HISTORY_DAYS:
        return None
    sigma = log_ret.std()
    log_ret_w = log_ret.clip(lower=-3 * sigma, upper=3 * sigma)

    # Vol 21d clipada em P95
    vol_21 = log_ret_w.rolling(21).std().fillna(sigma)
    vol_clip = float(vol_21.quantile(0.95))
    vol_21 = vol_21.clip(upper=vol_clip)

    # Feature USDBRL (overlay macro BR): captura stress cambial
    # (fiscal, política, sudden stop) que IBOV demora a precificar.
    usdbrl_log_ret = _fetch_usdbrl_returns(len(ibov_clean))
    if usdbrl_log_ret is not None and len(usdbrl_log_ret) >= len(log_ret_w):
        # Alinhar pelo índice do IBOV
        usdbrl_aligned = usdbrl_log_ret.reindex(log_ret_w.index).fillna(0.0)
        usdbrl_aligned = usdbrl_aligned.clip(lower=-0.05, upper=0.05)  # ±5% diário cap
        X = np.column_stack([log_ret_w.values, vol_21.values, usdbrl_aligned.values])
        logger.debug("HMM: 3 features (ibov_ret, vol, usdbrl_ret)")
    else:
        X = np.column_stack([log_ret_w.values, vol_21.values])
        logger.debug("HMM: 2 features (USDBRL indisponível)")

    try:
        # covariance_type="diag" é mais estável com N pequeno + 2 features.
        # "full" tem 4 parâmetros por estado (matriz 2×2); "diag" tem 2.
        # min_covar=1e-5 regulariza variâncias mínimas — impede colapso a 0
        # ou inflação numérica para 1000+ que tínhamos antes.
        model = GaussianHMM(
            n_components=HMM_N_STATES,
            covariance_type="diag",
            n_iter=200,
            min_covar=1e-5,
            random_state=HMM_RANDOM_STATE,
        )
        model.fit(X)
    except Exception as exc:
        logger.debug("HMM fit falhou (%s)", exc)
        return None

    # Re-rotulação: ordenar estados por σ (volatilidade do log_return IBOV).
    # Convenção quant (Ang-Bekaert 2002, Nystrup 2018): regime de mercado é
    # determinado primariamente por VOL, não por mean. Bull = baixa vol;
    # bear = alta vol (independente do sinal de μ). Mais estável que sort
    # por μ quando μ está próximo de zero ou muda de sinal.
    if model.covars_.ndim == 2:
        vols_state = np.sqrt(model.covars_[:, 0])
    else:
        vols_state = np.sqrt(np.array([model.covars_[i][0, 0] for i in range(HMM_N_STATES)]))
    means_ret = model.means_[:, 0]
    # Estado de MENOR σ = bull (low-vol regime); MAIOR σ = bear (high-vol)
    sort_idx = np.argsort(vols_state)  # ascendente: 0 = lowest_vol = bull
    bull_idx = sort_idx[0]
    bear_idx = sort_idx[-1]

    # Probabilidades correntes (último timestep)
    probs = model.predict_proba(X)[-1]
    p_bull = float(probs[bull_idx])
    p_bear = float(probs[bear_idx])

    # Sanity check: se o estado low-vol tem retorno médio fortemente negativo,
    # é um "grind down" regime — não é bull genuíno. Inverter rótulos.
    if means_ret[bull_idx] < -0.001:  # < -0.1% diário = -25% a.a.
        bull_idx, bear_idx = bear_idx, bull_idx
        p_bull, p_bear = p_bear, p_bull

    logger.info(
        "HMM regime: P(bull)=%.2f P(bear)=%.2f | μ_bull=%.4f μ_bear=%.4f | σ_bull=%.4f σ_bear=%.4f",
        p_bull, p_bear, means_ret[bull_idx], means_ret[bear_idx],
        vols_state[bull_idx], vols_state[bear_idx],
    )

    # Mapeamento para os 3 regimes da arquitetura
    if p_bear > 0.6:
        return "bear"
    if p_bull > 0.6:
        return "risk_on"
    return "mean_rev"


def _detect_regime(ibov_prices, vix_prices) -> str:
    """
    Orchestrador: tenta HMM primeiro (se USE_HMM_REGIME), fallback binário.
    """
    if USE_HMM_REGIME:
        hmm_regime = _detect_regime_hmm(ibov_prices, vix_prices)
        if hmm_regime is not None:
            return hmm_regime
        logger.debug("HMM indisponível ou inconclusivo — usando detector binário")
    return _detect_regime_binary(ibov_prices, vix_prices)


# Pipeline principal

def run(args: argparse.Namespace) -> int:
    run_date = _resolve_date(args.date)
    mode     = args.mode
    dry_run  = args.dry_run
    do_send  = args.send and not dry_run

    logger.info("=== B3 Recommender | mode=%s | date=%s | dry_run=%s ===",
                mode, run_date, dry_run)

    _ensure_dirs()

    # 1. Coleta de dados
    logger.info("Etapa 1/7 — Coletando dados fundamentais e de preços...")
    try:
        df_fundamentals, df_prices = load_data()
    except Exception as exc:
        logger.error("Falha na coleta de dados: %s", exc, exc_info=True)
        return 1

    if df_fundamentals.empty:
        logger.error("df_fundamentals vazio — abortando.")
        return 1

    logger.info(
        "Dados coletados: %d tickers fundamentais, %d séries de preço.",
        len(df_fundamentals),
        len(df_prices.columns) if not df_prices.empty else 0,
    )

    # 2. Benchmarks
    logger.info("Etapa 2/7 — Buscando benchmarks...")
    import pandas as pd
    from datetime import timedelta
    benchmark_mgr = BenchmarkManager()
    ibov_prices = pd.Series(dtype=float)
    benchmark_returns = pd.DataFrame()
    try:
        start_bench = (today_brt() - timedelta(days=365)).strftime("%Y-%m-%d")
        benchmark_returns = benchmark_mgr.get_returns(start_bench)
        ibov_prices = _get_ibov_prices(start_bench)
        logger.info("Benchmarks obtidos: %d dias.", len(benchmark_returns))
    except Exception as exc:
        logger.warning("Benchmarks falhou (não crítico): %s", exc)

    # 2b. Regime de mercado
    vix_prices = _fetch_vix_prices(
        (today_brt() - timedelta(days=365)).strftime("%Y-%m-%d")
    )
    market_regime = _detect_regime(ibov_prices, vix_prices)
    logger.info("Regime de mercado detectado: %s", market_regime)

    # 3. Scoring
    logger.info("Etapa 3/7 — Calculando scores...")
    try:
        df_scored = compute_scores(
            df_fund=df_fundamentals,
            df_prices=df_prices,
            ibov_prices=ibov_prices,
            regime=market_regime,
        )
    except Exception as exc:
        logger.error("Falha no scoring: %s", exc, exc_info=True)
        return 1

    if df_scored.empty:
        logger.error("Nenhum ticker sobreviveu ao scoring — abortando.")
        return 1

    # Turnover band: reduz rotação ruidosa dando bônus de TURNOVER_BAND_PTS
    # aos tickers que já estavam na carteira anterior. Aplicado ANTES do
    # diversificador para que ele veja o ordenamento ajustado.
    incumbents: list = []
    try:
        prev_snap = SnapshotManager()
        prev_rec = prev_snap.load_latest_recommendation(mode=mode)
        incumbents = [r["ticker"] for r in (prev_rec or {}).get("top5", [])] if prev_rec else []
        if incumbents:
            df_scored = apply_turnover_band(df_scored, incumbents, TURNOVER_BAND_PTS)
    except Exception as exc:
        logger.debug("Turnover band não aplicada (%s)", exc)

    # Aplicar filtros de diversificação: 1 por empresa, máx 2 por setor
    df_scored = select_diverse_portfolio(df_scored, n=5, max_per_sector=2)

    top_ticker = df_scored.iloc[0]["ticker"] if "ticker" in df_scored.columns else "?"
    logger.info("Scoring concluído: %d tickers pontuados. Top: %s", len(df_scored), top_ticker)

    # 3b. Observabilidade de cobertura
    # Universo declarado (universe.csv) vs coletado (df_fundamentals) vs
    # efetivamente pontuado (sobreviventes dos hard filters). Tornar o gap
    # VISÍVEL — antes ~60% do universo sumia silenciosamente.
    data_quality = coverage_report(df_fundamentals, df_prices, df_scored, UNIVERSE_FILE)
    df_scored.attrs["data_quality"] = data_quality
    declared_universe = data_quality["declared_universe"]
    n_collected = data_quality["collected"]
    n_scored = data_quality["scored"]
    coverage = data_quality["coverage_pct"]
    # Preço de entrada = cotação do momento da execução; se o pregão está
    # aberto, não é fechamento — fica registrado na recomendação.
    from src.benchmark import is_intraday, now_brt
    _run_at = now_brt()
    df_scored.attrs["market_data"] = {
        "run_at_brt":      _run_at.isoformat(timespec="seconds"),
        "prices_intraday": is_intraday(_run_at),
        "ibovespa":        dict(benchmark_mgr.ibov_meta),
    }
    if coverage is not None and coverage < MIN_UNIVERSE_COVERAGE:
        logger.warning(
            "COBERTURA BAIXA: %d/%d tickers pontuados (%.0f%% < %.0f%% mínimo). "
            "Coletados=%d. Verifique falhas de brapi/yfinance e histórico de preços.",
            n_scored, declared_universe, coverage * 100,
            MIN_UNIVERSE_COVERAGE * 100, n_collected,
        )
    else:
        logger.info(
            "Cobertura do universo: %d declarados → %d coletados → %d pontuados.",
            declared_universe or -1, n_collected, n_scored,
        )

    # 3c. Asset allocation (camada "investidor absoluto")
    # Decide QUANTO estar em bolsa antes de QUAL ação — a decisão dominante
    # com Selic alta. Sinais: regime HMM + ERP implícito + TSMOM 12-1.
    allocation = None
    if ENABLE_ASSET_ALLOCATION:
        try:
            from src.allocator import (
                compute_allocation,
                portfolio_earnings_yield,
                selic_annual_from_daily,
            )
            selic_series = (
                benchmark_returns["selic"]
                if "selic" in benchmark_returns.columns else None
            )
            cdi_series = (
                benchmark_returns["cdi"]
                if "cdi" in benchmark_returns.columns else None
            )
            allocation = compute_allocation(
                regime=market_regime,
                portfolio_earnings_yield=portfolio_earnings_yield(df_scored),
                selic_annual=selic_annual_from_daily(selic_series),
                ibov_prices=ibov_prices,
                cdi_daily_returns=cdi_series,
            )
        except Exception as exc:
            logger.warning("Asset allocation falhou (não crítico): %s", exc, exc_info=True)

    # 4. Snapshot de preços
    # A recomendação é salva DEPOIS do trade advice (etapa 4b) para persistir
    # stops/targets no JSON — o stop-monitor do closing diário depende disso.
    logger.info("Etapa 4/7 — Salvando snapshot de preços...")
    snap = SnapshotManager()
    try:
        snap.save_price_snapshot(df_prices=df_prices, run_date=run_date)
    except Exception as exc:
        logger.warning("Snapshot de preços falhou (não crítico): %s", exc)

    # 4b. Análise técnica + trade advice para o top 5
    logger.info("Etapa 4b/7 — Análise técnica e trade advice do top 5...")
    trade_advice: dict = {}
    try:
        top5_tickers = df_scored.head(5)["ticker"].tolist() if "ticker" in df_scored.columns else []
        if top5_tickers and not df_prices.empty:
            analyzer = TechnicalAnalyzer()
            intraday = analyzer.fetch_intraday(top5_tickers)
            tech_top5 = analyzer.analyze_all(top5_tickers, df_prices, intraday)

            # Enrich tech price from intraday when available
            for ticker, day in intraday.items():
                if ticker in tech_top5 and day.get("today_close"):
                    tech_top5[ticker]["price"] = day["today_close"]

            advisor = TradeAdvisor()
            for _, row in df_scored.head(5).iterrows():
                ticker = str(row.get("ticker", ""))
                tech = tech_top5.get(ticker, {})
                result = advisor.compute(row, df_scored, tech)
                if result:
                    trade_advice[ticker] = result
                    logger.debug(
                        "Trade advice %s: entrada=%.2f alvo=%.2f stop=%.2f R/R=%.1f",
                        ticker,
                        result.get("entry_low", 0),
                        result.get("target_conservative", 0),
                        result.get("stop", 0),
                        result.get("rr", 0),
                    )
        logger.info("Trade advice calculado para %d tickers.", len(trade_advice))
    except Exception as exc:
        logger.warning("Trade advice falhou (não crítico): %s", exc, exc_info=True)

    # 4c. Salvar recomendação (com allocation + stops persistidos)
    try:
        rec_path = snap.save_recommendation(
            df_scored=df_scored, df_prices=df_prices,
            run_date=run_date, mode=mode,
            allocation=allocation, trade_advice=trade_advice or None,
        )
        # Recarregar a allocation FINAL (gross do vol-target é aplicado dentro
        # do save) para que o relatório mostre exatamente o que foi persistido.
        saved_rec: dict = {}
        try:
            import json as _json
            with open(rec_path, encoding="utf-8") as _f:
                saved_rec = _json.load(_f)
            allocation = saved_rec.get("allocation") or allocation
        except Exception:
            pass
    except Exception as exc:
        logger.warning("Snapshot de recomendação falhou (não crítico): %s", exc)
        saved_rec = {}

    # 4d. Order sheet: pesos → ordens executáveis para o capital real
    order_sheet = None
    if args.capital and args.capital > 0:
        try:
            from src.order_sheet import build_order_sheet
            last_prices = (
                df_prices.iloc[-1].to_dict() if not df_prices.empty else {}
            )
            order_sheet = build_order_sheet(
                capital_brl=args.capital,
                allocation=allocation,
                portfolio_weights=saved_rec.get("portfolio_weights") or {},
                ticker_prices=last_prices,
                previous_tickers=incumbents or None,
            )
            if order_sheet:
                logger.info(
                    "Order sheet gerada para R$ %.2f (%d ordens de bolsa)",
                    args.capital, len(order_sheet.get("equity_orders", [])),
                )
        except Exception as exc:
            logger.warning("Order sheet falhou (não crítico): %s", exc, exc_info=True)

    # 5. Backtesting
    logger.info("Etapa 5/7 — Executando backtesting...")
    backtest_result = None
    try:
        backtest_result = run_backtest(
            mode=mode,
            df_scored=df_scored,
            df_prices=df_prices,
            benchmark_manager=benchmark_mgr,
            run_date=run_date,
        )
        status = backtest_result.get("status", "unknown")
        logger.info("Backtest status: %s", status)
        if status == "success":
            port_ret = backtest_result.get("portfolio_return", 0)
            logger.info("Portfolio return: %.2f%%", (port_ret or 0) * 100)
    except Exception as exc:
        logger.warning("Backtesting falhou (não crítico): %s", exc, exc_info=True)
        backtest_result = {"status": "error", "message": str(exc)}

    # 6. Gráfico
    chart_path: Optional[Path] = None
    if not args.no_chart:
        logger.info("Etapa 6/7 — Gerando gráfico...")
        try:
            # Salvar em output/ com nome datado
            chart_filename = OUTPUT_DIR / f"chart_{mode}_{run_date}.png"
            chart_gen = ChartGenerator(output_path=chart_filename)
            chart_path = chart_gen.generate_from_scored(
                df_scored=df_scored,
                df_prices=df_prices,
                benchmark_returns=benchmark_returns,
                mode=mode,
                output_path=chart_filename,
                run_date=run_date,
            )
            logger.info("Gráfico salvo em: %s", chart_path)
        except Exception as exc:
            logger.warning("Geração do gráfico falhou (não crítico): %s", exc, exc_info=True)
    else:
        logger.info("Etapa 6/7 — Geração de gráfico desativada (--no-chart).")

    # Enriquecer df_scored com preço atual (última linha do df_prices)
    if not df_prices.empty:
        last_prices = df_prices.iloc[-1]
        df_scored = df_scored.copy()
        df_scored["current_price"] = df_scored["ticker"].map(last_prices)

    # 6b. Análises retroativas (rodam também em dry-run)
    # Factor IC, walk-forward e risk model decomposition. Estes ficam ANTES
    # do dry-run return para acumular histórico estatístico em toda execução.
    try:
        from src.factor_analysis import analyze_factors
        analyze_factors(verbose=False)
        logger.info("Factor IC atualizado em data/factor_ic.json")
    except Exception as exc:
        logger.debug("Factor IC update falhou (%s)", exc)

    try:
        from src.walk_forward import walk_forward_backtest
        walk_forward_backtest(verbose=False)
        logger.info("Walk-forward atualizado em data/walk_forward.json")
    except Exception as exc:
        logger.debug("Walk-forward falhou (%s)", exc)

    try:
        from src.risk_model import build_risk_model, portfolio_risk_decomposition
        df_fund_idx = df_fundamentals.set_index("ticker") if "ticker" in df_fundamentals.columns else df_fundamentals
        rm = build_risk_model(
            df_prices=df_prices,
            ibov_prices=ibov_prices,
            df_fundamentals=df_fund_idx,
        )
        if rm is not None:
            latest = SnapshotManager().load_latest_recommendation(mode=mode)
            if latest and latest.get("portfolio_weights"):
                decomp = portfolio_risk_decomposition(latest["portfolio_weights"], rm)
                if decomp:
                    logger.info(
                        "Risk decomposition: vol_total=%.1f%% (fatorial=%.0f%% / específico=%.0f%%)",
                        decomp.get("total_vol", 0) * 100,
                        decomp.get("factor_pct", 0) * 100,
                        decomp.get("specific_pct", 0) * 100,
                    )
                    import json
                    decomp_path = Path("data") / "portfolio_risk_decomposition.json"
                    decomp_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(decomp_path, "w", encoding="utf-8") as f:
                        json.dump(decomp, f, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        logger.debug("Risk model falhou (%s)", exc)

    # 7. Relatório de texto
    logger.info("Etapa 7/7 — Construindo relatório de texto...")
    try:
        report_text = build_report(
            df_scored=df_scored,
            backtest_result=backtest_result,
            mode=mode,
            run_date=run_date,
            trade_advice=trade_advice,
            regime=market_regime,
            allocation=allocation,
            order_sheet=order_sheet,
        )
        logger.info("Relatório construído: %d caracteres.", len(report_text))
    except Exception as exc:
        logger.error("Falha ao construir relatório: %s", exc, exc_info=True)
        return 1

    # Dry-run: imprimir e encerrar
    if dry_run:
        logger.info("=== DRY RUN — relatório não enviado ao Telegram ===")
        print("\n" + "-" * 60)
        sys.stdout.buffer.write((report_text + "\n").encode("utf-8", errors="replace"))
        sys.stdout.buffer.flush()
        print("-" * 60 + "\n")
        if chart_path:
            print(f"Gráfico gerado: {chart_path}")
        _print_summary(df_scored, backtest_result)
        return 0

    # Sem --send: encerrar sem enviar
    if not do_send:
        logger.info(
            "Envio ao Telegram DESATIVADO. Use --send para enviar "
            "ou --dry-run para visualizar sem enviar."
        )
        _print_summary(df_scored, backtest_result)
        return 0

    # Envio ao Telegram
    logger.info("Enviando ao Telegram...")
    try:
        send_report(
            text=report_text,
            chart_path=chart_path,
        )
        logger.info("Relatório enviado com sucesso ao Telegram.")
    except TelegramError as exc:
        logger.error("Falha ao enviar ao Telegram: %s", exc)
        return 1

    _print_summary(df_scored, backtest_result)
    return 0


# Resumo em stdout

def _print_summary(df_scored, backtest_result: Optional[dict]) -> None:
    print("\n=== RESUMO ===")
    if not df_scored.empty and "ticker" in df_scored.columns:
        top5 = df_scored.head(5)
        print("Top 5 recomendações:")
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            ticker = row.get("ticker", "?")
            score  = row.get("total_score", 0)
            print(f"  {i}. {ticker} — score: {score:.2f}")

    if backtest_result:
        status = backtest_result.get("status", "?")
        print(f"\nBacktest: {status}")
        if status == "success":
            port_ret = backtest_result.get("portfolio_return", 0)
            ibov_ret = backtest_result.get("benchmark_returns", {}).get("ibovespa", 0)
            print(f"  Portfolio: {(port_ret or 0)*100:.2f}%")
            print(f"  IBOV:      {(ibov_ret or 0)*100:.2f}%")

    print("==============\n")


# Entrypoint

def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    _setup_logging(debug=args.debug)

    if args.validate_token:
        return _validate_token()

    return run(args)


if __name__ == "__main__":
    sys.exit(main())
