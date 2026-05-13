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
    LOG_LEVEL,
    CHART_OUTPUT_PATH,
)
from src.data_collector import load_data
from src.scoring_engine import compute_scores
from src.backtester import run_backtest
from src.benchmark import BenchmarkManager, get_ibov_prices as _get_ibov_prices
from src.report_builder import build_report
from src.chart_generator import ChartGenerator
from src.snapshot_manager import SnapshotManager
from src.technical_analyzer import TechnicalAnalyzer
from src.trade_advisor import TradeAdvisor
from src.telegram_sender import send_report, TelegramError

# ─── Logging ─────────────────────────────────────────────────────────────────

def _setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else getattr(logging, LOG_LEVEL, logging.INFO)
    fmt   = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")

logger = logging.getLogger(__name__)

# ─── CLI ─────────────────────────────────────────────────────────────────────

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
    return parser.parse_args(argv)


# ─── Validação de token ───────────────────────────────────────────────────────

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


# ─── Preparação de diretórios ─────────────────────────────────────────────────

def _ensure_dirs() -> None:
    for d in (OUTPUT_DIR, HISTORY_DIR, CACHE_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)


# ─── Parsing de data ──────────────────────────────────────────────────────────

def _resolve_date(date_arg: Optional[str]) -> str:
    if date_arg:
        try:
            datetime.strptime(date_arg, "%Y-%m-%d")
            return date_arg
        except ValueError:
            logger.error("Formato de data inválido: %s (esperado YYYY-MM-DD)", date_arg)
            sys.exit(2)
    return date.today().strftime("%Y-%m-%d")


# ─── Regime de mercado ───────────────────────────────────────────────────────

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


def _detect_regime(ibov_prices, vix_prices) -> str:
    """
    3-state market regime:
      risk_on:  IBOV > MA200 AND VIX < 18
      bear:     VIX > 25
      mean_rev: everything else
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


# ─── Pipeline principal ───────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> int:
    run_date = _resolve_date(args.date)
    mode     = args.mode
    dry_run  = args.dry_run
    do_send  = args.send and not dry_run

    logger.info("=== B3 Recommender | mode=%s | date=%s | dry_run=%s ===",
                mode, run_date, dry_run)

    _ensure_dirs()

    # ── 1. Coleta de dados ─────────────────────────────────────────────────
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

    # ── 2. Benchmarks ─────────────────────────────────────────────────────
    logger.info("Etapa 2/7 — Buscando benchmarks...")
    import pandas as pd
    from datetime import timedelta
    benchmark_mgr = BenchmarkManager()
    ibov_prices = pd.Series(dtype=float)
    benchmark_returns = pd.DataFrame()
    try:
        start_bench = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")
        benchmark_returns = benchmark_mgr.get_returns(start_bench)
        ibov_prices = _get_ibov_prices(start_bench)
        logger.info("Benchmarks obtidos: %d dias.", len(benchmark_returns))
    except Exception as exc:
        logger.warning("Benchmarks falhou (não crítico): %s", exc)

    # ── 2b. Regime de mercado ─────────────────────────────────────────────
    vix_prices = _fetch_vix_prices(
        (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    )
    market_regime = _detect_regime(ibov_prices, vix_prices)
    logger.info("Regime de mercado detectado: %s", market_regime)

    # ── 3. Scoring ────────────────────────────────────────────────────────
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

    top_ticker = df_scored.iloc[0]["ticker"] if "ticker" in df_scored.columns else "?"
    logger.info("Scoring concluído: %d tickers pontuados. Top: %s", len(df_scored), top_ticker)

    # ── 4. Snapshot de preços + recomendação atual ────────────────────────
    logger.info("Etapa 4/7 — Salvando snapshots...")
    snap = SnapshotManager()
    try:
        snap.save_price_snapshot(df_prices=df_prices, run_date=run_date)
        snap.save_recommendation(df_scored=df_scored, df_prices=df_prices,
                                 run_date=run_date, mode=mode)
    except Exception as exc:
        logger.warning("Snapshot falhou (não crítico): %s", exc)

    # ── 4b. Análise técnica + trade advice para o top 5 ──────────────────
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

    # ── 5. Backtesting ────────────────────────────────────────────────────
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

    # ── 6. Gráfico ────────────────────────────────────────────────────────
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

    # ── 7. Relatório de texto ─────────────────────────────────────────────
    logger.info("Etapa 7/7 — Construindo relatório de texto...")
    try:
        report_text = build_report(
            df_scored=df_scored,
            backtest_result=backtest_result,
            mode=mode,
            run_date=run_date,
            trade_advice=trade_advice,
        )
        logger.info("Relatório construído: %d caracteres.", len(report_text))
    except Exception as exc:
        logger.error("Falha ao construir relatório: %s", exc, exc_info=True)
        return 1

    # ── Dry-run: imprimir e encerrar ──────────────────────────────────────
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

    # ── Sem --send: encerrar sem enviar ───────────────────────────────────
    if not do_send:
        logger.info(
            "Envio ao Telegram DESATIVADO. Use --send para enviar "
            "ou --dry-run para visualizar sem enviar."
        )
        _print_summary(df_scored, backtest_result)
        return 0

    # ── Envio ao Telegram ─────────────────────────────────────────────────
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


# ─── Resumo em stdout ─────────────────────────────────────────────────────────

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


# ─── Entrypoint ───────────────────────────────────────────────────────────────

def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    _setup_logging(debug=args.debug)

    if args.validate_token:
        return _validate_token()

    return run(args)


if __name__ == "__main__":
    sys.exit(main())
