"""
Monta a base histórica de preços da B3 a partir do COTAHIST.

    python tools/build_price_history.py                 # tudo
    python tools/build_price_history.py --steps series  # só refaz as séries

Etapas: download → parse → renames → universe → corp → series.
Dados em B3_DATA_DIR (padrão C:\\Users\\danie\\b3_data), fora do repositório.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.history import corporate, cotahist  # noqa: E402
from src.history.adjust import BIG_MOVE, build_series, patch_big_moves  # noqa: E402
from src.history.paths import CORP_DIR, DERIVED_DIR, RAW_DIR  # noqa: E402
from src.history.renames import detect_renames, security_ids  # noqa: E402
from src.history.universe import universe_history  # noqa: E402

STEPS = ["download", "parse", "renames", "universe", "corp", "series", "patch"]
log = logging.getLogger("build_price_history")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default=",".join(STEPS))
    ap.add_argument("--start-year", type=int, default=2010)
    args = ap.parse_args()
    steps = set(args.steps.split(","))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)

    if "download" in steps:
        cotahist.download_annual(range(args.start_year, date.today().year + 1))
    if "parse" in steps:
        cotahist.build_parquet(sorted(RAW_DIR.glob("COTAHIST_A*.ZIP"))
                               + sorted((RAW_DIR / "daily").glob("COTAHIST_D*.ZIP")))

    df = cotahist.load_all()
    if "renames" in steps:
        detect_renames(df).to_csv(DERIVED_DIR / "renames.csv", index=False)
    renames = pd.read_csv(DERIVED_DIR / "renames.csv")
    sid_of = security_ids(df, renames)
    df["sid"] = df["ticker"].map(sid_of)

    if "universe" in steps:
        universe_history(df).to_parquet(DERIVED_DIR / "universe_history.parquet", index=False)
    universe = pd.read_parquet(DERIVED_DIR / "universe_history.parquet")
    sids = set(universe["sid"])
    members = [t for t, s in sid_of.items() if s in sids]
    roots = sorted({t[:4] for t in members})

    if "corp" in steps:
        session = requests.Session()
        for root in roots:
            corporate.fetch_issuer(root, session)

    if "series" in steps:
        issuers = [corporate.fetch_issuer(r) for r in roots if (CORP_DIR / f"{r}.json").exists()]
        cash = pd.concat([corporate.cash_events(i) for i in issuers], ignore_index=True)
        stock = pd.concat([corporate.stock_events(i) for i in issuers], ignore_index=True)
        series, jumps = build_series(df, sid_of, cash, stock, sids)
        series.to_parquet(DERIVED_DIR / "prices_daily.parquet", index=False)
        jumps.to_csv(DERIVED_DIR / "price_jumps.csv", index=False)
        log.info("séries: %d papéis, %d linhas; saltos sem evento: %d (%d aceitos)",
                 series["sid"].nunique(), len(series), len(jumps),
                 int(jumps["accepted"].sum()) if len(jumps) else 0)
    if "patch" in steps:
        import yfinance as yf

        from src.benchmark import clean_close

        series = pd.read_parquet(DERIVED_DIR / "prices_daily.parquet")
        for c in ("ret_total", "ret_price"):
            if f"{c}_raw" not in series.columns:
                series[f"{c}_raw"] = series[c]
            else:
                series[c] = series[f"{c}_raw"]
        flagged = sorted(series.loc[series["ret_total"].abs() > BIG_MOVE, "ticker"].unique())
        reference = {}
        for t in flagged:
            try:
                raw = yf.download(f"{t}.SA", start="2009-12-01", progress=False, auto_adjust=True)
            except Exception as exc:
                log.warning("%s: yfinance falhou (%s)", t, exc)
                continue
            if raw is None or raw.empty:
                continue
            raw.index = pd.to_datetime(raw.index).tz_localize(None)
            reference[t] = clean_close(raw)[0].pct_change()
        series, plog = patch_big_moves(series, reference)
        series.to_parquet(DERIVED_DIR / "prices_daily.parquet", index=False)
        plog.to_csv(DERIVED_DIR / "big_moves_review.csv", index=False)
        log.info("dias > %.0f%%: %s", BIG_MOVE * 100, plog["action"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
