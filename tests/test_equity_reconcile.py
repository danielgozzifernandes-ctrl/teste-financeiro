"""
tests/test_equity_reconcile.py

Equity curve gravada a partir de fechamento final: dado ausente vira null,
pregão não consolidado fica provisório e o closing seguinte reconcilia.
"""

import json
from datetime import date

import pandas as pd
import pytest

import main_daily
from main_daily import _session_points, _weighted_portfolio_return
from src.equity_curve import _load, upsert_points
from src.snapshot_manager import SnapshotManager


def _pt(d, port, ibov, cdi=0.0005, blended=None, **extra):
    return {"date": d, "port_ret": port, "ibov_ret": ibov, "cdi_ret": cdi,
            "blended_ret": blended, **extra}


# ---------------------------------------------------------------------------
# upsert_points
# ---------------------------------------------------------------------------

def test_provisional_point_is_reconciled_and_nav_rechained(tmp_path):
    path = tmp_path / "eq.json"
    upsert_points([_pt("2026-10-01", 0.01, 0.02)], path=path)
    upsert_points([_pt("2026-10-02", None, None, None, provisional=True)], path=path)
    s = _load(path)
    assert s[-1]["provisional"] is True
    assert s[-1]["nav"] == s[0]["nav"]            # null não move o NAV
    assert s[-1]["cdi_nav"] == s[0]["cdi_nav"]    # nem acumula CDI

    summary = upsert_points(
        [_pt("2026-10-02", 0.03, 0.01), _pt("2026-10-05", -0.01, 0.0)], path=path,
    )
    s = _load(path)
    assert [p["date"] for p in s] == ["2026-10-01", "2026-10-02", "2026-10-05"]
    assert "provisional" not in s[1]
    assert s[1]["nav"] == pytest.approx(100 * 1.01 * 1.03)
    assert s[2]["nav"] == pytest.approx(100 * 1.01 * 1.03 * 0.99)
    assert s[2]["ibov_nav"] == pytest.approx(100 * 1.02 * 1.01)
    assert summary["nav"] == pytest.approx(s[2]["nav"])


def test_final_close_replaces_preliminary_value(tmp_path):
    path = tmp_path / "eq.json"
    upsert_points([_pt("2026-10-01", 0.01, 0.02)], path=path)
    upsert_points([_pt("2026-10-01", 0.05, 0.05)], path=path)
    assert _load(path)[0]["port_ret"] == 0.05


def test_window_drops_points_on_non_trading_days(tmp_path):
    # Versão antiga rotulava o pregão de 02/10 como 03/10 (sábado) e o de
    # 05/10 como 06/10: dentro da janela, o fechamento final corrige.
    path = tmp_path / "eq.json"
    upsert_points([
        _pt("2026-09-30", 0.0, 0.0),
        _pt("2026-10-02", 0.004, 0.0046),
        _pt("2026-10-03", 0.019, 0.0263),
        _pt("2026-10-06", 0.038, 0.077),
    ], path=path)
    upsert_points([
        _pt("2026-10-02", 0.019, 0.0263),
        _pt("2026-10-05", 0.038, 0.077),
        _pt("2026-10-06", None, None, None, provisional=True),
    ], path=path, window_start="2026-09-24")
    s = _load(path)
    assert [p["date"] for p in s] == ["2026-09-30", "2026-10-02", "2026-10-05", "2026-10-06"]
    assert s[-1]["provisional"] is True
    assert s[-1]["ibov_nav"] == pytest.approx(100 * 1.0263 * 1.077)


def test_null_cdi_on_settled_day_uses_last_rate(tmp_path):
    path = tmp_path / "eq.json"
    upsert_points([_pt("2026-10-01", 0.0, 0.0, 0.0006)], path=path)
    upsert_points([_pt("2026-10-02", 0.0, 0.0, None)], path=path)
    s = _load(path)
    assert s[1]["cdi_ret"] is None
    assert s[1]["cdi_nav"] == pytest.approx(100 * 1.0006 ** 2)


# ---------------------------------------------------------------------------
# Retorno do sleeve de bolsa
# ---------------------------------------------------------------------------

def test_missing_holding_counts_as_flat_not_renormalized():
    w = {"A": 0.5, "B": 0.5}
    assert _weighted_portfolio_return({"A": 0.10, "B": None}, w) == pytest.approx(0.05)
    assert _weighted_portfolio_return({"A": None, "B": None}, w) is None


# ---------------------------------------------------------------------------
# Pontos por pregão a partir de fechamentos
# ---------------------------------------------------------------------------

def _write_rec(d, weights, history):
    rec = {"date": d, "mode": "weekly",
           "top5": [{"ticker": t} for t in weights],
           "portfolio_weights": weights,
           "allocation": {"sleeves": {"equities_br": 0.5, "cdi": 0.5}}}
    (history / f"recommendations_{d}_weekly.json").write_text(json.dumps(rec), encoding="utf-8")


@pytest.fixture
def history(tmp_path):
    _write_rec("2026-09-28", {"AAAA3": 0.6, "BBBB3": 0.4}, tmp_path)
    _write_rec("2026-10-05", {"CCCC3": 1.0}, tmp_path)
    return tmp_path


def test_session_points_use_the_portfolio_held_that_day(history):
    idx = pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-05"])
    closes = pd.DataFrame({
        "^BVSP":    [100.0, 102.0, 101.0],
        "AAAA3.SA": [10.0, 11.0, 11.0],
        "BBBB3.SA": [20.0, float("nan"), 20.0],   # sem candle em 02/10
        "CCCC3.SA": [5.0, 5.0, 5.5],
    }, index=idx)
    pts = _session_points(closes, list(idx), {"2026-10-02": 0.0005},
                          SnapshotManager(history_dir=history))
    by = {p["date"]: p for p in pts}
    assert set(by) == {"2026-10-02", "2026-10-05"}   # 01/10 não tem pregão anterior
    assert by["2026-10-02"]["rec_date"] == "2026-09-28"
    assert by["2026-10-02"]["port_ret"] == pytest.approx(0.6 * 0.10)
    assert by["2026-10-02"]["missing"] == ["BBBB3"]
    assert by["2026-10-02"]["ibov_ret"] == pytest.approx(0.02)
    assert by["2026-10-02"]["blended_ret"] == pytest.approx(0.5 * 0.06 + 0.5 * 0.0005)
    assert by["2026-10-05"]["rec_date"] == "2026-10-05"
    assert by["2026-10-05"]["port_ret"] == pytest.approx(0.10)


def test_unsettled_session_becomes_provisional_point(history, monkeypatch):
    idx = pd.to_datetime(["2026-10-01", "2026-10-02"])
    closes = pd.DataFrame({"^BVSP": [100.0, 102.0], "AAAA3.SA": [10.0, 11.0],
                           "BBBB3.SA": [20.0, 20.0]}, index=idx)
    meta = {"^BVSP": {"last_bar": "2026-10-05", "intraday": True}}
    monkeypatch.setattr(main_daily, "_session_closes", lambda s, start: (closes, meta))

    class NoCDI:
        def get_returns(self, *a):
            raise RuntimeError("offline")

    monkeypatch.setattr("src.benchmark.BenchmarkManager", NoCDI)
    pts, start = main_daily._equity_curve_points("2026-10-05", SnapshotManager(history_dir=history))
    assert start == "2026-09-23"
    assert pts[-1]["date"] == "2026-10-05"
    assert pts[-1]["provisional"] is True
    assert pts[-1]["port_ret"] is None
    assert pts[-1]["market_data"]["ibovespa"]["intraday"] is True
    assert [p["date"] for p in pts[:-1]] == ["2026-10-02"]
