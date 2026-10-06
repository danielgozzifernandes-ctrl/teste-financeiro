"""Cobertura declarado → coletado → pontuado."""

import numpy as np
import pandas as pd

from main import coverage_report


def test_collected_counts_usable_tickers_not_rows(tmp_path):
    uni = tmp_path / "universe.csv"
    uni.write_text("ticker,nome\nAAAA3,a\nBBBB4,b\nCCCC3,c\nDDDD3,d\n", encoding="utf-8")
    idx = pd.bdate_range("2026-01-01", periods=30)
    prices = pd.DataFrame({"AAAA3": 1.0, "BBBB4": 1.0, "CCCC3": np.nan}, index=idx)
    funds = pd.DataFrame({
        "ticker": ["AAAA3", "BBBB4", "CCCC3", "DDDD3"],
        "pl": [5, 6, 7, None], "pvp": [1, 1, 1, None], "roe": [0.1, None, 0.1, None],
    })
    scored = pd.DataFrame({"ticker": ["AAAA3"], "total_score": [70.0]})

    dq = coverage_report(funds, prices, scored, uni)
    assert dq["declared_universe"] == 4
    assert dq["collected"] == 2                 # A e B: preço + fundamentos
    assert dq["scored"] == 1
    assert dq["no_price_history"] == ["CCCC3", "DDDD3"]
    assert dq["no_fundamentals"] == ["DDDD3"]
    assert dq["filtered_out"] == ["BBBB4"]
