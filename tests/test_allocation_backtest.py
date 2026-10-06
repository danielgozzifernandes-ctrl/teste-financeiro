"""Simulador do backtest de alocação: lag de rebalanceamento, drift, CDI, sleeves sem dado."""

import numpy as np
import pandas as pd
import pytest

from src import allocation_backtest as ab


def _prices(n=60, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {"equities_br": 100.0, "global_usd": 100.0, "inflation": 100.0}, index=idx,
    )


def test_new_weights_apply_from_next_day():
    p = _prices()
    p.loc[p.index[21]:, "equities_br"] = 110.0      # +10% no dia do rebalance
    p.loc[p.index[22]:, "equities_br"] = 121.0      # +10% no dia seguinte
    cdi = pd.Series(0.0, index=p.index)

    def target(i):
        return {"equities_br": 1.0} if i < 21 else {"cdi": 1.0}

    curve, _, _ = ab._simulate(p, cdi, 21, target)
    cost = ab.COST_PER_SIDE_BPS / 10_000
    # o +10% de i=21 ainda é da carteira antiga; o de i=22 já não
    assert curve.iloc[21] == pytest.approx(100 * (1 - cost) * 1.10)
    assert curve.iloc[22] == pytest.approx(curve.iloc[21] * (1 - 2 * cost))


def test_weights_drift_between_rebalances():
    p = _prices()
    p["equities_br"] = np.linspace(100, 200, len(p))
    cdi = pd.Series(0.0, index=p.index)
    _, turnover, _ = ab._simulate(p, cdi, 21, lambda i: dict(ab.STATIC_MIX))
    assert turnover > 0.01   # a bolsa subiu, rebalancear de volta custa giro


def test_sleeve_without_prices_sits_in_cdi():
    p = _prices()
    p.loc[: p.index[30], "inflation"] = np.nan
    cdi = pd.Series(0.0004, index=p.index)
    _, _, log = ab._simulate(p, cdi, 21, lambda i: dict(ab.STATIC_MIX))
    assert "inflation" not in log[0]
    assert log[0]["cdi"] == pytest.approx(0.45)
    assert log[-1]["inflation"] == pytest.approx(0.15)


def test_cdi_compounds_over_dates_missing_from_price_index():
    idx = pd.bdate_range("2020-01-01", periods=5)
    cdi = pd.Series(0.001, index=idx)
    out = ab._cdi_on_index(cdi, idx[[0, 2, 4]])
    assert out.iloc[1] == pytest.approx(1.001 ** 2 - 1)


def test_cdi_strategy_has_no_sharpe_vs_itself():
    idx = pd.bdate_range("2020-01-01", periods=300)
    cdi = pd.Series(0.0004, index=idx)
    curve = (1 + cdi).cumprod() * 100
    m = ab._metrics(curve, cdi)
    assert m["sharpe_vs_cdi"] is None
    assert "dsr" not in m


def test_metrics_report_psr_and_dsr_ranges():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    cdi = pd.Series(0.0004, index=idx)
    rets = pd.Series(rng.normal(0.0008, 0.01, len(idx)), index=idx)
    curve = (1 + rets).cumprod() * 100
    m = ab._metrics(curve, cdi)
    assert 0 <= m["psr_vs_0"] <= 1
    dsr = [m["dsr"][f"N={n}"] for n in ab.DSR_TRIALS]
    assert dsr == sorted(dsr, reverse=True)      # mais tentativas, barra mais alta
