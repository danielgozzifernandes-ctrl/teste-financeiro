"""
main_daily.py — Orquestrador do Relatório Diário B3

Executa o pipeline de relatório diário em dois modos:

  morning  (08:30 Brasília):
    1. Busca macro global (USD, S&P, petróleo, VIX...)
    2. Carrega última recomendação semanal do SnapshotManager
    3. Calcula indicadores técnicos do Top 5 (RSI, MACD, BBands, MAs)
    4. Detecta alertas (gaps, volume anormal, cruzamentos MACD)
    5. Constrói e envia relatório de texto ao Telegram

  closing  (18:00 Brasília):
    1. Carrega última recomendação semanal
    2. Busca preços do dia para Top 5 + IBOV
    3. Calcula retornos do dia
    4. Gera gráfico de barras (desempenho por ticker)
    5. Constrói e envia relatório ao Telegram com gráfico

Flags CLI:
  --mode morning | closing    Modo de execução (obrigatório)
  --date YYYY-MM-DD           Data de referência (default: hoje)
  --dry-run                   Executa tudo mas não envia ao Telegram
  --send                      Envia ao Telegram
  --debug                     Logging DEBUG
  --force                     Roda mesmo em dia sem pregão

Exit codes:
  0  Sucesso
  1  Falha (dados insuficientes, erro de envio)
  2  Erro de configuração
"""

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import OUTPUT_DIR, HISTORY_DIR, CACHE_DIR
from src.b3_calendar import holiday_name, is_trading_day, today_brt
from src.snapshot_manager import SnapshotManager
from src.macro_fetcher import MacroFetcher
from src.technical_analyzer import TechnicalAnalyzer
from src.daily_report_builder import build_daily_report
from src.closing_report_builder import build_closing_report
from src.closing_chart_generator import ClosingChartGenerator
from src.telegram_sender import send_report, TelegramError

logger = logging.getLogger(__name__)


def _print_utf8(text: str) -> None:
    # Console do Windows (cp1252) não codifica emoji.
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace") + b"\n")
    sys.stdout.buffer.flush()

# Dias corridos que cada closing recalcula com fechamento final.
_RECONCILE_DAYS = 12


# CLI

def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="B3 Daily Recommender — Relatório Matinal e de Fechamento",
    )
    parser.add_argument(
        "--mode", choices=["morning", "closing"], required=True,
        help="Modo: morning (08:30) ou closing (18:00)",
    )
    parser.add_argument(
        "--date", default=None, metavar="YYYY-MM-DD",
        help="Data de referência (default: hoje)",
    )
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
            logger.error("Formato inválido: %s (esperado YYYY-MM-DD)", date_arg)
            sys.exit(2)
    return today_brt().isoformat()


def _ensure_dirs() -> None:
    for d in (OUTPUT_DIR, HISTORY_DIR, CACHE_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)


# Fetch today's prices

def _fetch_daily_prices(
    tickers: list[str],
) -> tuple[dict[str, float], dict[str, float], float, dict]:
    """
    Fetches today's close prices and returns for a list of B3 tickers + IBOV.

    Returns:
        (ticker_returns, ticker_prices, ibov_return, ibov_meta)
        All returns are decimal (0.01 = 1%). ibov_meta = last bar date and
        whether it is an unsettled intraday print.
    """
    from src.benchmark import clean_close, now_brt

    yf_symbols = [f"{t}.SA" for t in tickers] + ["^BVSP"]
    ticker_returns: dict[str, float] = {}
    ticker_prices:  dict[str, float] = {}
    ibov_return: float = 0.0
    ibov_meta: dict = {}

    try:
        raw = yf.download(
            tickers=yf_symbols,
            period="5d",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        logger.error("_fetch_daily_prices: download falhou: %s", exc)
        return ticker_returns, ticker_prices, ibov_return, ibov_meta

    if raw is None or raw.empty:
        logger.error("_fetch_daily_prices: dados vazios")
        return ticker_returns, ticker_prices, ibov_return, ibov_meta

    def _extract(symbol: str) -> tuple[Optional[float], Optional[float], dict]:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if symbol not in raw.columns.get_level_values(1):
                    return None, None, {}
                sub = raw.xs(symbol, axis=1, level=1)
            else:
                sub = raw
            close, meta = clean_close(sub)

            if len(close) < 2:
                return None, None, meta

            today_close = float(close.iloc[-1])
            prev_close  = float(close.iloc[-2])
            ret = (today_close - prev_close) / prev_close if prev_close else 0.0
            return ret, today_close, meta
        except Exception as exc:
            logger.debug("_extract %s: %s", symbol, exc)
            return None, None, {}

    # IBOV
    ibov_ret, _, ibov_meta = _extract("^BVSP")
    if ibov_ret is not None:
        ibov_return = ibov_ret
    today = now_brt().date().isoformat()
    if ibov_meta.get("last_bar") and ibov_meta["last_bar"] != today:
        # Candle de hoje ainda não saiu: o "retorno do dia" seria o de ontem.
        ibov_meta["stale"] = True
        logger.warning(
            "IBOV: último candle é de %s, não de hoje (%s).",
            ibov_meta["last_bar"], today,
        )
    if ibov_meta.get("intraday"):
        logger.warning("IBOV: candle de hoje ainda é intradiário (não consolidado).")

    # Tickers
    for ticker in tickers:
        ret, price, _ = _extract(f"{ticker}.SA")
        if ret is not None:
            ticker_returns[ticker] = ret
        if price is not None:
            ticker_prices[ticker] = price

    logger.info(
        "Preços do dia: %d/%d tickers, IBOV %.2f%%",
        len(ticker_returns), len(tickers), ibov_return * 100,
    )
    return ticker_returns, ticker_prices, ibov_return, ibov_meta


def _portfolio_return(ticker_returns: dict[str, float]) -> float:
    """Equal-weight portfolio return (mean of all tickers)."""
    valid = [r for r in ticker_returns.values()
             if r is not None and not np.isnan(r)]
    return float(np.mean(valid)) if valid else 0.0


def _weighted_portfolio_return(
    ticker_returns: dict[str, Optional[float]],
    weights: Optional[dict[str, float]],
) -> Optional[float]:
    """
    Retorno do sleeve de bolsa com os pesos da recomendação (equal-weight se
    ela não tem pesos). Posição sem retorno no dia conta como parada (0%),
    sem redistribuir o peso dela entre as outras: renormalizar inflava a
    curva quando um papel some (NEOE3/ODPV3 deslistadas). None se nenhuma
    posição tem retorno.
    """
    if weights:
        holdings = {t: w for t, w in weights.items() if w and w > 0}
    else:
        holdings = {t: 1.0 for t in ticker_returns}
    total = sum(holdings.values())
    if not holdings or total <= 0:
        return None
    valid = {
        t: r for t, r in ticker_returns.items()
        if t in holdings and r is not None and not np.isnan(r)
    }
    if not valid:
        return None
    return float(sum(holdings[t] / total * r for t, r in valid.items()))


def _session_closes(
    symbols: list[str], start: str,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """
    Fechamentos consolidados (data × símbolo) em dias de pregão. Candle
    incompleto ou intradiário do dia fica de fora; o metadata dele vai em
    metas[símbolo].
    """
    from src.benchmark import clean_close

    raw = yf.download(tickers=symbols, start=start, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return pd.DataFrame(), {}
    closes: dict[str, pd.Series] = {}
    metas: dict[str, dict] = {}
    for sym in symbols:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if sym not in raw.columns.get_level_values(1):
                    continue
                sub = raw.xs(sym, axis=1, level=1)
            else:
                sub = raw
            sub = sub.copy()
            sub.index = pd.to_datetime(sub.index).tz_localize(None)
            close, meta = clean_close(sub)
            if meta.get("intraday"):
                close = close.iloc[:-1]
            closes[sym] = close
            metas[sym] = meta
        except Exception as exc:
            logger.debug("_session_closes %s: %s", sym, exc)
    df = pd.DataFrame(closes)
    if not df.empty:
        df = df[[is_trading_day(d) for d in df.index]]
    return df, metas


def _session_points(
    closes: pd.DataFrame,
    sessions: list,
    cdi_by_date: dict[str, float],
    snap: SnapshotManager,
) -> list[dict]:
    """Pontos da equity curve, um por pregão em `sessions`, com fechamentos finais."""
    ibov_dates = list(closes["^BVSP"].dropna().index) if "^BVSP" in closes else []

    def ret(sym: str, d, prev) -> Optional[float]:
        if sym not in closes:
            return None
        a, b = closes[sym].get(prev), closes[sym].get(d)
        if a is None or b is None or pd.isna(a) or pd.isna(b) or a <= 0:
            return None
        return float(b / a - 1.0)

    points = []
    for d in sessions:
        i = ibov_dates.index(d)
        if i == 0:
            continue
        prev = ibov_dates[i - 1]
        day = d.date().isoformat()
        # Carteira vigente no pregão: a última recomendação até ele (mesma
        # convenção de antes: a recomendação de segunda já conta na segunda).
        rec = snap.load_latest_recommendation(
            mode="weekly", before_date=(d.date() + timedelta(days=1)).isoformat(),
        )
        if not rec:
            continue
        holdings = [r["ticker"] for r in rec.get("top5", [])]
        weights = rec.get("portfolio_weights") or {t: 1.0 for t in holdings}
        weights = {t: weights.get(t, 0.0) for t in holdings}
        rets = {t: ret(f"{t}.SA", d, prev) for t in holdings}
        port = _weighted_portfolio_return(rets, weights)
        cdi = cdi_by_date.get(day)

        blended = None
        sleeves = (rec.get("allocation") or {}).get("sleeves") or {}
        missing_sleeves = []
        if sleeves and port is not None:
            sleeve_rets = {
                "equities_br": port,
                "cdi":         cdi,
                "global_usd":  ret("IVVB11.SA", d, prev),
                "inflation":   ret("IMAB11.SA", d, prev),
            }
            missing_sleeves = [k for k in sleeves if sleeve_rets.get(k) is None]
            blended = float(sum(
                w * (sleeve_rets.get(k) or 0.0) for k, w in sleeves.items()
            ))

        point = {
            "date":        day,
            "port_ret":    port,
            "ibov_ret":    ret("^BVSP", d, prev),
            "cdi_ret":     cdi,
            "blended_ret": blended,
            "rec_date":    rec.get("date"),
        }
        missing = [t for t, r in rets.items() if r is None]
        if missing:
            point["missing"] = missing
        if missing_sleeves:
            point["missing_sleeves"] = missing_sleeves
        points.append(point)
    return points


def _equity_curve_points(
    run_date: str, snap: SnapshotManager,
) -> tuple[list[dict], str]:
    """
    Pontos a gravar na equity curve neste closing e o início da janela que
    eles cobrem: os pregões recentes com fechamento final (reconcilia ponto
    provisório, preliminar ou que faltou) e, se o pregão de hoje ainda não
    consolidou, um ponto provisório nulo.
    """
    from src.benchmark import BenchmarkManager

    run_day = date.fromisoformat(run_date)
    start = (run_day - timedelta(days=_RECONCILE_DAYS)).isoformat()

    tickers: set[str] = set()
    for f in Path(snap.history_dir).glob("recommendations_*_weekly.json"):
        rec_day = f.name.split("_")[1]
        if rec_day >= (run_day - timedelta(days=_RECONCILE_DAYS + 10)).isoformat():
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
                tickers.update(r["ticker"] for r in rec.get("top5", []))
            except Exception:
                continue
    current = snap.load_latest_recommendation(mode="weekly")
    tickers.update(r["ticker"] for r in (current or {}).get("top5", []))

    symbols = [f"{t}.SA" for t in sorted(tickers)] + ["^BVSP", "IVVB11.SA", "IMAB11.SA"]
    closes, metas = _session_closes(symbols, start)
    if closes.empty or "^BVSP" not in closes:
        logger.warning("Equity curve: sem fechamentos do IBOV — nada a gravar.")
        return [], start

    cdi_by_date: dict[str, float] = {}
    try:
        cdi = BenchmarkManager().get_returns(start, run_date)["cdi"].dropna()
        cdi_by_date = {d.date().isoformat(): float(v) for d, v in cdi.items()}
    except Exception as exc:
        logger.warning("Equity curve: CDI indisponível (%s)", exc)

    ts = pd.Timestamp(run_day)
    sessions = [d for d in closes["^BVSP"].dropna().index if d <= ts]
    points = _session_points(closes, sessions, cdi_by_date, snap)

    if is_trading_day(run_day) and ts not in sessions:
        points.append({
            "date":        run_date,
            "port_ret":    None,
            "ibov_ret":    None,
            "cdi_ret":     None,
            "blended_ret": None,
            "provisional": True,
            "market_data": {"ibovespa": metas.get("^BVSP", {})},
        })
        logger.warning(
            "Pregão de %s ainda sem fechamento consolidado: ponto provisório, "
            "o próximo closing reconcilia.", run_date,
        )
    return points, start


def _calc_cumulative_returns(
    ticker_prices: dict[str, float],
    entry_prices: dict[str, float],
) -> dict[str, float]:
    """
    Cumulative return since recommendation for each ticker.
    Formula: (close_today / entry_price) - 1
    """
    result: dict[str, float] = {}
    for ticker, current in ticker_prices.items():
        entry = entry_prices.get(ticker)
        if entry and entry > 0 and current and current > 0:
            result[ticker] = (current / entry) - 1
    return result


def _fetch_ibov_cumulative(rec_date: str) -> Optional[float]:
    """
    Fetches IBOV cumulative return from rec_date (first available close) to today.
    Returns None on failure.
    """
    try:
        raw = yf.download(
            tickers=["^BVSP"],
            start=rec_date,
            auto_adjust=True,
            progress=False,
        )
        if raw is None or raw.empty:
            return None

        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]["^BVSP"].dropna()
        else:
            close = raw["Close"].dropna()

        if len(close) < 2:
            return None

        return float(close.iloc[-1] / close.iloc[0]) - 1
    except Exception as exc:
        logger.warning("_fetch_ibov_cumulative: %s", exc)
        return None


# Fetch historical prices for technical analysis

def _fetch_hist_prices(tickers: list[str], lookback_days: int = 300) -> pd.DataFrame:
    """
    Fetches historical adjusted-close prices for technical indicator computation.
    Returns wide DataFrame (date × ticker).
    """
    start = (today_brt() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    yf_symbols = [f"{t}.SA" for t in tickers]

    try:
        raw = yf.download(
            tickers=yf_symbols,
            start=start,
            auto_adjust=True,
            progress=False,
        )
        if raw is None or raw.empty:
            return pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw[["Close"]]

        close.index = pd.to_datetime(close.index).tz_localize(None)

        # Rename columns from "PETR4.SA" → "PETR4"
        rename = {f"{t}.SA": t for t in tickers}
        close = close.rename(columns=rename)

        # Keep only requested tickers
        available = [t for t in tickers if t in close.columns]
        return close[available].dropna(how="all")

    except Exception as exc:
        logger.error("_fetch_hist_prices: %s", exc)
        return pd.DataFrame()


# Morning pipeline

def run_morning(run_date: str, do_send: bool, dry_run: bool) -> int:
    logger.info("=== MORNING REPORT | %s ===", run_date)

    # Load latest weekly recommendation for top 5 tickers
    snap = SnapshotManager()
    recommendation = snap.load_latest_recommendation(mode="weekly")

    top5_tickers: list[str] = []
    if recommendation:
        top5_tickers = [r["ticker"] for r in recommendation.get("top5", [])]
        logger.info("Top 5 carregado: %s", top5_tickers)
    else:
        logger.warning("Nenhuma recomendação semanal encontrada — relatório sem análise técnica")

    # Step 1: Macro
    logger.info("Etapa 1/3 — Buscando macro...")
    macro_snapshot: dict = {}
    try:
        fetcher = MacroFetcher()
        macro_snapshot = fetcher.get_snapshot()
        sentiment = fetcher.risk_sentiment(macro_snapshot)
        logger.info("Macro: %d indicadores, sentimento=%s", len(macro_snapshot), sentiment)
    except Exception as exc:
        logger.warning("Macro falhou (não crítico): %s", exc)
        sentiment = "neutral"

    # Step 2: Technical analysis
    logger.info("Etapa 2/3 — Análise técnica do Top 5...")
    tech_data: dict = {}
    alerts: list = []
    if top5_tickers:
        try:
            analyzer = TechnicalAnalyzer()
            df_hist = _fetch_hist_prices(top5_tickers)
            intraday = analyzer.fetch_intraday(top5_tickers)
            tech_data = analyzer.analyze_all(top5_tickers, df_hist, intraday)
            alerts = analyzer.detect_alerts(tech_data, intraday)
            logger.info(
                "Técnico: %d tickers analisados, %d alertas",
                len(tech_data), len(alerts),
            )
        except Exception as exc:
            logger.warning("Análise técnica falhou (não crítico): %s", exc, exc_info=True)

    # Step 3: Build report
    logger.info("Etapa 3/3 — Construindo relatório matinal...")
    try:
        report_text = build_daily_report(
            macro_snapshot=macro_snapshot,
            tech_data=tech_data,
            alerts=alerts,
            run_date=run_date,
            sentiment=sentiment,
            recommendation=recommendation,
        )
        logger.info("Relatório matinal: %d caracteres", len(report_text))
    except Exception as exc:
        logger.error("Falha ao construir relatório: %s", exc, exc_info=True)
        return 1

    if dry_run:
        print("\n" + "-" * 60)
        _print_utf8(report_text)
        print("-" * 60 + "\n")
        return 0

    if not do_send:
        logger.info("Envio desativado. Use --send para enviar ao Telegram.")
        print("\n" + "-" * 60)
        _print_utf8(report_text)
        print("-" * 60 + "\n")
        return 0

    try:
        send_report(text=report_text, chart_path=None)
        logger.info("Relatório matinal enviado ao Telegram.")
    except TelegramError as exc:
        logger.error("Falha ao enviar ao Telegram: %s", exc)
        return 1

    return 0


# Closing pipeline

def run_closing(run_date: str, do_send: bool, dry_run: bool) -> int:
    logger.info("=== CLOSING REPORT | %s ===", run_date)

    # Load latest weekly recommendation
    snap = SnapshotManager()
    recommendation = snap.load_latest_recommendation(mode="weekly")

    top5_tickers: list[str] = []
    if recommendation:
        top5_tickers = [r["ticker"] for r in recommendation.get("top5", [])]
        logger.info("Top 5 carregado: %s", top5_tickers)
    else:
        logger.error("Nenhuma recomendação semanal encontrada — não é possível gerar o relatório de fechamento")
        return 1

    # Step 1: Fetch today's prices
    logger.info("Etapa 1/3 — Buscando preços do dia...")
    try:
        ticker_returns, ticker_prices, ibov_return, ibov_meta = _fetch_daily_prices(top5_tickers)
    except Exception as exc:
        logger.error("Falha ao buscar preços do dia: %s", exc, exc_info=True)
        return 1

    if not ticker_returns:
        logger.error("Sem dados de preço para hoje — mercado fechado ou dados indisponíveis")
        return 1

    # Retorno do sleeve de bolsa com os pesos REAIS da recomendação (HRP
    # com bounds) — medir equal-weight enquanto se recomenda HRP é medir
    # outra carteira.
    port_return = _weighted_portfolio_return(
        {t: ticker_returns.get(t) for t in top5_tickers},
        recommendation.get("portfolio_weights") or {t: 1.0 for t in top5_tickers},
    ) or 0.0

    # Cumulative returns since recommendation
    entry_prices: dict[str, float] = recommendation.get("entry_prices", {})
    rec_date: str = recommendation.get("date", "")
    cumulative_returns = _calc_cumulative_returns(ticker_prices, entry_prices)
    cumulative_portfolio = _portfolio_return(cumulative_returns) if cumulative_returns else None
    ibov_cumulative = _fetch_ibov_cumulative(rec_date) if rec_date else None
    logger.info(
        "Retorno acumulado desde %s: carteira=%.2f%% IBOV=%.2f%%",
        rec_date,
        (cumulative_portfolio or 0) * 100,
        (ibov_cumulative or 0) * 100,
    )

    # Step 1b: Stop monitor — stops persistidos na recomendação viram alertas
    # acionáveis. Sem isso, stop é decoração (PETR4 caiu −10% em mai/2026 sem
    # nenhum aviso).
    stop_alerts: list = []
    try:
        from src.stop_monitor import check_levels
        stop_alerts = check_levels(recommendation, ticker_prices)
    except Exception as exc:
        logger.warning("Stop monitor falhou (não crítico): %s", exc)

    # Step 1c: Equity curve — NAV encadeado vs IBOV e CDI (a régua absoluta).
    equity_summary: Optional[dict] = None
    noise_band_pp: Optional[float] = None
    try:
        from src.equity_curve import alpha_noise_band_pp, upsert_points
        points, window_start = _equity_curve_points(run_date, snap)
        equity_summary = upsert_points(points, window_start=window_start)
        noise_band_pp = alpha_noise_band_pp()
    except Exception as exc:
        logger.warning("Equity curve falhou (não crítico): %s", exc, exc_info=True)

    # Step 2: Volume ratios (from intraday)
    volume_ratios: dict[str, float] = {}
    try:
        analyzer = TechnicalAnalyzer()
        intraday = analyzer.fetch_intraday(top5_tickers)
        volume_ratios = {
            t: d.get("volume_ratio", float("nan"))
            for t, d in intraday.items()
            if d.get("volume_ratio") is not None
        }
    except Exception as exc:
        logger.warning("Volume ratios: %s", exc)

    # Step 3: Chart + report
    logger.info("Etapa 2/3 — Gerando gráfico de fechamento...")
    chart_path: Optional[Path] = None
    try:
        chart_filename = OUTPUT_DIR / f"closing_chart_{run_date}.png"
        chart_gen = ClosingChartGenerator()
        chart_path = chart_gen.generate(
            ticker_returns=ticker_returns,
            ibov_return=ibov_return,
            portfolio_return=port_return,
            run_date=run_date,
            ticker_prices=ticker_prices,
            output_path=chart_filename,
        )
        logger.info("Gráfico de fechamento: %s", chart_path)
    except Exception as exc:
        logger.warning("Gráfico de fechamento falhou (não crítico): %s", exc, exc_info=True)

    logger.info("Etapa 3/3 — Construindo relatório de fechamento...")
    try:
        report_text = build_closing_report(
            ticker_returns=ticker_returns,
            ticker_prices=ticker_prices,
            ibov_return=ibov_return,
            portfolio_return=port_return,
            run_date=run_date,
            recommendation=recommendation,
            volume_ratios=volume_ratios,
            cumulative_returns=cumulative_returns,
            cumulative_portfolio_return=cumulative_portfolio,
            ibov_cumulative_return=ibov_cumulative,
            recommendation_date=rec_date,
            stop_alerts=stop_alerts,
            equity_summary=equity_summary,
            noise_band_pp=noise_band_pp,
        )
        logger.info("Relatório de fechamento: %d caracteres", len(report_text))
    except Exception as exc:
        logger.error("Falha ao construir relatório: %s", exc, exc_info=True)
        return 1

    if dry_run:
        print("\n" + "-" * 60)
        _print_utf8(report_text)
        print("-" * 60 + "\n")
        if chart_path:
            print(f"Gráfico: {chart_path}")
        return 0

    if not do_send:
        logger.info("Envio desativado. Use --send para enviar ao Telegram.")
        print("\n" + "-" * 60)
        _print_utf8(report_text)
        print("-" * 60 + "\n")
        return 0

    try:
        send_report(text=report_text, chart_path=chart_path)
        logger.info("Relatório de fechamento enviado ao Telegram.")
    except TelegramError as exc:
        logger.error("Falha ao enviar ao Telegram: %s", exc)
        return 1

    return 0


# Entrypoint

def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.debug)

    run_date = _resolve_date(args.date)
    do_send  = args.send and not args.dry_run

    if not args.force and not is_trading_day(run_date):
        logger.info(
            "%s não tem pregão na B3 (%s) — nada a fazer.",
            run_date, holiday_name(run_date) or "fim de semana",
        )
        return 0

    _ensure_dirs()

    if args.mode == "morning":
        return run_morning(run_date, do_send, args.dry_run)
    else:
        return run_closing(run_date, do_send, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
