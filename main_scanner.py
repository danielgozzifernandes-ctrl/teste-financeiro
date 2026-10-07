"""
main_scanner.py — Scanner de Oportunidades B3

Roda o pipeline completo de coleta + scoring + análise técnica no universo
inteiro e só envia mensagem ao Telegram quando uma ação passa no filtro
rigoroso de oportunidade.

Na maioria dos dias não envia nada — isso é esperado e desejado.

Filtro aplicado (todos os critérios devem ser atendidos):
  PRIMARY (todos):
    • total_score >= 72 (top fundamentalista)
    • Sinal técnico: buy ou strong_buy
    • RSI entre 28 e 48 (zona de recuperação)
    • MACD acima da linha de sinal

  SECONDARY (mínimo 2 de 3):
    • MACD cruzamento altista
    • Volume >= 1.5× a média
    • Preço no terço inferior das Bandas de Bollinger

Flags CLI:
  --date YYYY-MM-DD     Data de referência (default: hoje)
  --dry-run             Executa tudo mas não envia ao Telegram
  --send                Envia ao Telegram
  --debug               Logging DEBUG
  --force               Roda mesmo em dia sem pregão

Exit codes:
  0  Sucesso (com ou sem oportunidades)
  1  Falha crítica (dados insuficientes)
  2  Erro de configuração
"""

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from src.config import OUTPUT_DIR, HISTORY_DIR, CACHE_DIR
from src.b3_calendar import holiday_name, is_trading_day, today_brt
from src.data_collector import load_data
from src.scoring_engine import compute_scores
from src.benchmark import BenchmarkManager, get_ibov_prices as _get_ibov_prices
from src.technical_analyzer import TechnicalAnalyzer
from src.opportunity_scanner import OpportunityScanner, append_scan_record, build_scan_record
from src.opportunity_report_builder import build_opportunity_alerts
from src.telegram_sender import send_report, TelegramError

logger = logging.getLogger(__name__)


def _print_utf8(text: str) -> None:
    # Console do Windows (cp1252) não codifica emoji.
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace") + b"\n")
    sys.stdout.buffer.flush()


# CLI

def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="B3 Opportunity Scanner — dispara apenas quando há oportunidade real",
    )
    parser.add_argument("--date",    default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--send",    action="store_true")
    parser.add_argument("--debug",   action="store_true")
    parser.add_argument("--force",   action="store_true",
                        help="Roda mesmo em dia sem pregão na B3")
    return parser.parse_args(argv)


def _setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _resolve_date(date_arg: Optional[str]) -> str:
    if date_arg:
        try:
            datetime.strptime(date_arg, "%Y-%m-%d")
            return date_arg
        except ValueError:
            logger.error("Formato inválido: %s", date_arg)
            sys.exit(2)
    return today_brt().isoformat()


def _ensure_dirs() -> None:
    for d in (OUTPUT_DIR, HISTORY_DIR, CACHE_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)


# Pipeline

def run(args: argparse.Namespace) -> int:
    run_date = _resolve_date(args.date)
    do_send  = args.send and not args.dry_run

    logger.info("=== OPPORTUNITY SCANNER | %s ===", run_date)
    if not args.force and not is_trading_day(run_date):
        logger.info(
            "%s não tem pregão na B3 (%s) — nada a fazer.",
            run_date, holiday_name(run_date) or "fim de semana",
        )
        return 0
    _ensure_dirs()

    # 1. Coleta de dados
    logger.info("Etapa 1/5 — Coletando dados do universo...")
    try:
        df_fundamentals, df_prices = load_data()
    except Exception as exc:
        logger.error("Falha na coleta de dados: %s", exc, exc_info=True)
        return 1

    if df_fundamentals.empty:
        logger.error("Dados fundamentais vazios — abortando.")
        return 1

    logger.info("Dados: %d tickers fundamentais", len(df_fundamentals))

    # 2. Benchmarks
    logger.info("Etapa 2/5 — Buscando benchmarks...")
    ibov_prices = pd.Series(dtype=float)
    try:
        start_bench = (today_brt() - timedelta(days=365)).strftime("%Y-%m-%d")
        ibov_prices = _get_ibov_prices(start_bench)
    except Exception as exc:
        logger.warning("Benchmarks falhou (não crítico): %s", exc)

    # 3. Scoring
    logger.info("Etapa 3/5 — Calculando scores do universo...")
    try:
        df_scored = compute_scores(
            df_fund=df_fundamentals,
            df_prices=df_prices,
            ibov_prices=ibov_prices,
        )
    except Exception as exc:
        logger.error("Falha no scoring: %s", exc, exc_info=True)
        return 1

    if df_scored.empty:
        logger.error("Nenhum ticker sobreviveu ao scoring — abortando.")
        return 1

    # Enriquecer com preço atual
    if not df_prices.empty:
        last_prices = df_prices.iloc[-1]
        df_scored = df_scored.copy()
        df_scored["current_price"] = df_scored["ticker"].map(last_prices)

    logger.info("Scoring: %d tickers pontuados", len(df_scored))

    # 4. Análise técnica de todos os tickers
    logger.info("Etapa 4/5 — Análise técnica do universo completo...")
    tech_data: dict = {}
    try:
        all_tickers = df_scored["ticker"].tolist() if "ticker" in df_scored.columns else []
        analyzer = TechnicalAnalyzer()

        # Intraday (gap + volume) para todos de uma vez
        intraday = analyzer.fetch_intraday(all_tickers)

        # Indicadores técnicos usando df_prices (já baixado)
        tech_data = analyzer.analyze_all(all_tickers, df_prices, intraday)

        # Enriquecer tech_data com preços do intraday quando disponível
        for ticker, day in intraday.items():
            if ticker in tech_data and day.get("today_close"):
                tech_data[ticker]["price"] = day["today_close"]

        logger.info("Técnico: %d/%d tickers analisados", len(tech_data), len(all_tickers))
    except Exception as exc:
        logger.error("Análise técnica falhou: %s", exc, exc_info=True)
        return 1

    # 5. Scanner de oportunidades
    logger.info("Etapa 5/5 — Aplicando filtro de oportunidades...")
    scanner = OpportunityScanner()
    opportunities = scanner.scan(df_scored, tech_data)

    delivery = "dry_run" if args.dry_run else ("send" if do_send else "print")
    _persist_scan(run_date, scanner, df_scored, df_prices, tech_data, opportunities, delivery)

    if not opportunities:
        logger.info("Nenhuma oportunidade qualificada hoje — sem envio. ✓")
        print(f"[{run_date}] Scanner concluído: 0 oportunidades. Nenhuma mensagem enviada.")
        return 0

    logger.info("%d oportunidade(s) qualificada(s): %s",
                len(opportunities),
                [o["ticker"] for o in opportunities])

    # Construir mensagens
    messages = build_opportunity_alerts(opportunities, run_date)

    if args.dry_run:
        print(f"\n[DRY RUN] {len(opportunities)} oportunidade(s) encontrada(s):")
        for msg in messages:
            print("\n" + "-" * 60)
            _print_utf8(msg)
        print("-" * 60 + "\n")
        return 0

    if not do_send:
        print(f"\n{len(opportunities)} oportunidade(s) encontrada(s). Use --send para enviar.")
        for msg in messages:
            print("\n" + "-" * 60)
            _print_utf8(msg)
        return 0

    # Envio ao Telegram
    sent = 0
    for msg in messages:
        try:
            send_report(text=msg, chart_path=None)
            sent += 1
            logger.info("Alerta enviado (%d/%d)", sent, len(messages))
        except TelegramError as exc:
            logger.error("Falha ao enviar alerta: %s", exc)

    if sent == 0:
        return 1

    logger.info("Scanner concluído: %d alerta(s) enviado(s).", sent)
    return 0


def _persist_scan(run_date, scanner, df_scored, df_prices, tech_data,
                  opportunities, delivery) -> None:
    """Grava a execução em data/history/scanner_YYYY-MM.jsonl. Falha aqui não derruba o scanner."""
    try:
        from src.benchmark import is_intraday, now_brt

        run_at = now_brt()
        data_as_of = (
            str(pd.Timestamp(df_prices.index.max()).date())
            if not df_prices.empty else None
        )
        record = build_scan_record(
            run_date,
            scanner.evaluate(df_scored, tech_data),
            opportunities,
            meta={
                "run_at_brt":      run_at.isoformat(timespec="seconds"),
                "prices_intraday": is_intraday(run_at),
                "data_as_of":      data_as_of,
                "n_scored":        len(df_scored),
                "delivery":        delivery,
            },
        )
        path = append_scan_record(record, HISTORY_DIR)
        logger.info("Execução do scanner registrada em %s", path.name)
    except Exception as exc:
        logger.warning("Falha ao registrar execução do scanner: %s", exc, exc_info=True)


# Entrypoint

def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.debug)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
