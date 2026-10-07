"""Retorno total, custo pelo giro e ticker sem preço no backtester."""

import json
from datetime import date

import pytest

from src import backtester as bt_mod
from src.backtester import Backtester, BacktestStatus, transaction_cost
from src.snapshot_manager import SnapshotManager


def _rec(day, weights, entry, adv=None):
    return {
        "date": day, "mode": "weekly",
        "top5": [
            {"ticker": t, "entry_price": entry[t], "sector": "S",
             "metrics": {"adv_brl": (adv or {}).get(t)}}
            for t in weights
        ],
        "entry_prices": dict(entry),
        "portfolio_weights": dict(weights),
    }


def _write(tmp_path, rec):
    (tmp_path / f"recommendations_{rec['date']}_weekly.json").write_text(
        json.dumps(rec), encoding="utf-8")


@pytest.fixture
def bt(tmp_path):
    return Backtester(snapshot_manager=SnapshotManager(history_dir=tmp_path),
                      history_dir=tmp_path)


def test_dividends_enter_the_return(tmp_path, bt):
    _write(tmp_path, _rec("2026-09-28", {"AAAA3": 0.5, "BBBB4": 0.5},
                          {"AAAA3": 10.0, "BBBB4": 20.0}))
    prices = {"AAAA3": 10.0, "BBBB4": 20.0}
    res = bt.run("weekly", prices, {"ibovespa": 0.0, "cdi": 0.003}, run_date="2026-10-05",
                 dividends={"AAAA3": {"amount": 1.0, "jcp": 0.4, "source": "b3"},
                            "BBBB4": {"amount": 0.0, "jcp": 0.0, "source": "b3"}})
    assert res["return_basis"] == "total"
    assert res["gross_return"] == pytest.approx(0.05)
    a = next(h for h in res["holdings"] if h["ticker"] == "AAAA3")
    assert a["return"] == pytest.approx(0.10) and a["jcp"] == 0.4
    assert res["dividends_missing"] == []

    res = bt.run("weekly", prices, {"ibovespa": 0.0}, run_date="2026-10-05")
    assert res["return_basis"] == "price"
    assert res["gross_return"] == pytest.approx(0.0)


def test_missing_price_keeps_weight_at_zero_return(tmp_path, bt):
    _write(tmp_path, _rec("2026-09-28", {"AAAA3": 0.6, "BBBB4": 0.4},
                          {"AAAA3": 10.0, "BBBB4": 10.0}))
    res = bt.run("weekly", {"BBBB4": 11.0}, {"ibovespa": 0.0, "cdi": 0.01},
                 run_date="2026-10-05")
    assert res["status"] == BacktestStatus.PARTIAL_DATA
    # sem renormalizar: 0,4 × 10% (antes daria 10%)
    assert res["gross_return"] == pytest.approx(0.04)


def test_cost_is_charged_on_turnover_only(tmp_path, bt):
    w = {"AAAA3": 0.5, "BBBB4": 0.5}
    entry = {"AAAA3": 10.0, "BBBB4": 10.0}
    _write(tmp_path, _rec("2026-09-21", w, entry))
    _write(tmp_path, _rec("2026-09-28", w, entry))
    res = bt.run("weekly", entry, {"ibovespa": 0.0}, run_date="2026-10-05")
    assert res["transaction_cost"] == 0.0
    assert res["portfolio_return"] == pytest.approx(0.0)


def test_transaction_cost_first_buy_and_illiquid_names():
    rec = _rec("2026-09-28", {"AAAA3": 0.5, "BBBB4": 0.5}, {"AAAA3": 1, "BBBB4": 1},
               adv={"AAAA3": 500e6, "BBBB4": 5e6})
    cost, turnover = transaction_cost(rec, None)
    base = bt_mod._FRICTION
    assert turnover == pytest.approx(0.5)
    # BBBB4 tem ADV 10× menor que a referência → fricção √10 × base
    assert cost == pytest.approx(0.5 * base + 0.5 * base * 10 ** 0.5)

    prior = _rec("2026-09-21", {"AAAA3": 1.0}, {"AAAA3": 1})
    cost, turnover = transaction_cost(rec, prior)
    assert turnover == pytest.approx(0.5)


def test_fetch_dividends_uses_b3_com_date_and_class(monkeypatch):
    rows = [
        {"isinCode": "BRPETRACNPR6", "lastDatePrior": "28/09/2026", "rate": "1,00000",
         "label": "DIVIDENDO"},
        {"isinCode": "BRPETRACNPR6", "lastDatePrior": "01/10/2026", "rate": "0,50000",
         "label": "JRS CAP PROPRIO"},
        {"isinCode": "BRPETRACNPR6", "lastDatePrior": "01/10/2026", "rate": "0,50000",
         "label": "JRS CAP PROPRIO"},                       # parcela repetida conta
        {"isinCode": "BRPETRACNPR6", "lastDatePrior": "05/10/2026", "rate": "9,00000",
         "label": "DIVIDENDO"},                             # data-com = dia do backtest: fora
        {"isinCode": "BRPETRACNOR9", "lastDatePrior": "01/10/2026", "rate": "7,00000",
         "label": "DIVIDENDO"},                             # ON, não PN
    ]
    monkeypatch.setattr(bt_mod, "_b3_cash_dividends", lambda issuer: rows)
    monkeypatch.setattr(bt_mod, "fetch_dividends_yf",
                        lambda tickers, s, e: {t: 0.25 for t in tickers})
    out = bt_mod.fetch_dividends(["PETR4", "XPTO99"], date(2026, 9, 28), date(2026, 10, 5))
    assert out["PETR4"] == {"amount": pytest.approx(2.0), "jcp": pytest.approx(1.0),
                            "source": "b3"}
    assert out["XPTO99"] == {"amount": 0.25, "jcp": None, "source": "yfinance"}
