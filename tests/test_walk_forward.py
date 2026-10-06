"""Walk-forward: mesma janela para carteira e IBOV, sobreposição, ticker sem preço."""

import json

import numpy as np
import pandas as pd
import pytest

from src.walk_forward import walk_forward_backtest

DATES = pd.bdate_range("2026-01-05", periods=80)


def _write_recs(tmp_path, n=8, tickers=("AAAA3", "BBBB4")):
    for k in range(n):
        day = DATES[k * 5]
        rec = {
            "date": day.strftime("%Y-%m-%d"), "mode": "weekly",
            "execution_metadata": {"market_data": {"run_at_brt": f"{day.date()}T11:00:00"}},
            "top5": [{"ticker": t} for t in tickers],
            "portfolio_weights": {t: 1 / len(tickers) for t in tickers},
        }
        (tmp_path / f"recommendations_{rec['date']}_weekly.json").write_text(
            json.dumps(rec), encoding="utf-8")


def _bench(daily_ibov=0.001, daily_cdi=0.0005):
    return lambda s, e: pd.DataFrame({"ibovespa": daily_ibov, "cdi": daily_cdi}, index=DATES)


def test_portfolio_and_ibov_share_the_window(tmp_path):
    _write_recs(tmp_path)
    prices = pd.DataFrame({"AAAA3": 1.002 ** np.arange(80), "BBBB4": 1.002 ** np.arange(80)},
                          index=DATES) * 10
    r = walk_forward_backtest(history_dir=tmp_path, output_path=None, verbose=False,
                              price_loader=lambda t, s, e: prices[t], bench_loader=_bench())
    p = next(x for x in r["per_period"] if x["window"] == "1w")
    assert p["port_ret"] == pytest.approx(1.002 ** 5 - 1, abs=1e-6)
    assert p["ibov_ret"] == pytest.approx(1.001 ** 5 - 1, abs=1e-6)        # 5 pregões, não 4
    assert p["alpha"] == pytest.approx(p["port_ret"] - p["ibov_ret"], abs=1e-6)


def test_overlapping_windows_are_not_tested_as_independent(tmp_path):
    _write_recs(tmp_path, n=10)
    prices = pd.DataFrame({"AAAA3": 1.003 ** np.arange(80), "BBBB4": 1.001 ** np.arange(80)},
                          index=DATES) * 10
    r = walk_forward_backtest(history_dir=tmp_path, output_path=None, verbose=False,
                              price_loader=lambda t, s, e: prices[t], bench_loader=_bench())
    w4 = r["per_window"]["4w"]
    assert w4["n_eff"] == pytest.approx(w4["n_periods"] / 4)
    assert w4["t_nw_alpha"] is None and w4["significant"] is False


def test_unpriced_name_keeps_weight_at_zero(tmp_path):
    _write_recs(tmp_path, n=2, tickers=("AAAA3", "GONE3"))
    prices = pd.DataFrame({"AAAA3": 1.01 ** np.arange(80)}, index=DATES) * 10
    r = walk_forward_backtest(history_dir=tmp_path, output_path=None, verbose=False,
                              price_loader=lambda t, s, e: prices.reindex(columns=t),
                              bench_loader=_bench())
    p = next(x for x in r["per_period"] if x["window"] == "1w")
    assert p["port_ret"] == pytest.approx(0.5 * (1.01 ** 5 - 1), abs=1e-6)
    assert p["missing"] == ["GONE3"]
