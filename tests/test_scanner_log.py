"""
tests/test_scanner_log.py

Registro de cada execução do scanner em data/history/scanner_YYYY-MM.jsonl.
"""

import json

import numpy as np
import pandas as pd
import pytest

import main_scanner
from src.opportunity_scanner import (
    OpportunityScanner,
    append_scan_record,
    build_scan_record,
)


def _tech(signal="buy", rsi=40.0, macd_above=True, cross="bullish", vol=1.0, bb=0.2):
    return {
        "price": 10.0, "signal": signal, "rsi": rsi, "macd_above_signal": macd_above,
        "macd_crossover": cross, "volume_ratio": vol, "bb_position": bb,
        "ma20": 11.0, "ma50": 12.0, "ma200": 13.0, "high_52w": 14.0,
    }


@pytest.fixture
def universe():
    df = pd.DataFrame({
        "ticker":      ["AAAA3", "BBBB3", "CCCC3", "DDDD3", "EEEE3", "FFFF3"],
        "total_score": [80.0,    75.0,    60.0,    73.0,    90.0,    85.0],
    })
    tech = {
        "AAAA3": _tech(),                                  # qualifica
        "BBBB3": _tech(cross=None, bb=0.9),                # primário ok, 0 secundários
        "CCCC3": _tech(),                                  # só falha o score
        "DDDD3": _tech(signal="neutral", rsi=60.0),        # falha 2 primários
        "EEEE3": _tech(rsi=np.nan, vol=np.float64(2.0)),   # falha RSI (nan)
        # FFFF3 sem dado técnico → fora da avaliação, como no scan()
    }
    return df, tech


def test_evaluate_qualifies_same_tickers_as_scan(universe):
    df, tech = universe
    sc = OpportunityScanner()
    qualified = {e["ticker"] for e in sc.evaluate(df, tech) if e["qualified"]}
    assert qualified == {o["ticker"] for o in sc.scan(df, tech)} == {"AAAA3"}
    assert {e["ticker"] for e in sc.evaluate(df, tech)} == set(tech)


def test_record_counts_and_near_misses(universe):
    df, tech = universe
    sc = OpportunityScanner()
    rec = build_scan_record("2026-10-13", sc.evaluate(df, tech), sc.scan(df, tech),
                            meta={"prices_intraday": True})
    assert rec["counts"]["qualified"] == 1
    assert rec["counts"]["score"] == 4
    assert rec["counts"]["primary"] == 2
    assert {n["ticker"] for n in rec["near_misses"]} == {"BBBB3", "CCCC3", "EEEE3"}
    assert [a["ticker"] for a in rec["alerts"]] == ["AAAA3"]
    assert rec["prices_intraday"] is True
    assert len(rec["universe"]["rows"]) == 5
    assert rec["universe"]["columns"][0] == "ticker"


def test_append_writes_one_line_per_run(universe, tmp_path):
    df, tech = universe
    sc = OpportunityScanner()
    for d in ("2026-10-13", "2026-10-14"):
        path = append_scan_record(build_scan_record(d, sc.evaluate(df, tech), []), tmp_path)
    assert path.name == "scanner_2026-10.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(l)["date"] for l in lines] == ["2026-10-13", "2026-10-14"]
    append_scan_record(build_scan_record("2026-11-03", [], []), tmp_path)
    assert (tmp_path / "scanner_2026-11.jsonl").exists()


def test_scanner_run_logs_day_without_alerts(universe, tmp_path, monkeypatch):
    df, tech = universe
    tech = {t: {**v, "rsi": 70.0} for t, v in tech.items()}   # nada qualifica
    prices = pd.DataFrame({"AAAA3": [10.0, 10.1]},
                          index=pd.to_datetime(["2026-10-09", "2026-10-13"]))

    class FakeAnalyzer:
        def fetch_intraday(self, tickers):
            return {}

        def analyze_all(self, tickers, df_prices, intraday):
            return tech

    monkeypatch.setattr(main_scanner, "HISTORY_DIR", tmp_path)
    monkeypatch.setattr(main_scanner, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(main_scanner, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(main_scanner, "load_data", lambda: (df, prices))
    monkeypatch.setattr(main_scanner, "_get_ibov_prices", lambda s: pd.Series(dtype=float))
    monkeypatch.setattr(main_scanner, "compute_scores", lambda **k: df)
    monkeypatch.setattr(main_scanner, "TechnicalAnalyzer", FakeAnalyzer)

    assert main_scanner.main(["--date", "2026-10-13", "--dry-run"]) == 0
    rec = json.loads((tmp_path / "scanner_2026-10.jsonl").read_text(encoding="utf-8"))
    assert rec["date"] == "2026-10-13"
    assert rec["alerts"] == []
    assert rec["counts"]["qualified"] == 0
    assert rec["data_as_of"] == "2026-10-13"
    assert rec["delivery"] == "dry_run"
