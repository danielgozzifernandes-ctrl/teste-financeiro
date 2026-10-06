"""
tests/test_selection.py

Mudanças de seleção: idio_alpha_6m desligado, filtro de preço congelado,
exclusion_date do universo.
"""

import numpy as np
import pandas as pd
import pytest

from src import scoring_engine as se
from src.scoring_engine import ScoringEngine


def _prices(n_days: int = 200, tickers=("AAA3", "BBB3", "CCC3", "DDD3"), seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2026-01-02", periods=n_days)
    rets = rng.normal(0.0005, 0.02, size=(n_days, len(tickers)))
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=list(tickers))


def test_old_idio_momentum_is_zero_by_construction():
    prices = _prices()
    ibov = prices.mean(axis=1) * 1000
    df = pd.DataFrame(index=prices.columns)
    sector_map = pd.Series({"AAA3": "X", "BBB3": "X", "CCC3": "Y", "DDD3": "Y"})
    out = ScoringEngine()._add_idiosyncratic_momentum(df, prices, ibov, sector_map)
    assert out["idio_alpha_6m"].notna().all()
    assert out["idio_alpha_6m"].abs().max() < 1e-10


def test_idio_momentum_is_off_by_default():
    assert se.ENABLE_IDIO_MOMENTUM is False
    assert "idio_alpha_6m" not in se._MOMENTUM_CFG


def test_idio_column_has_no_effect_on_momentum_pillar():
    tickers = [f"T{i}" for i in range(8)]
    df = pd.DataFrame({
        "alpha_3m":  np.linspace(-0.1, 0.1, 8),
        "alpha_12m": np.linspace(0.2, -0.2, 8),
    }, index=tickers)
    eng = ScoringEngine()
    base, _ = eng._score_momentum(df.copy())
    df["idio_alpha_6m"] = np.linspace(1e-16, -1e-16, 8)
    with_idio, _ = eng._score_momentum(df)
    pd.testing.assert_series_equal(base, with_idio)


# ---------------------------------------------------------------------------
# Preço congelado
# ---------------------------------------------------------------------------

from src.scoring_engine import detect_stale_prices
from src.snapshot_manager import _compute_inv_vol_weights, _compute_portfolio_weights


def _frozen_tail(prices: pd.DataFrame, col: str, n: int, value: float) -> pd.DataFrame:
    p = prices.copy()
    p.iloc[-n:, p.columns.get_loc(col)] = value
    return p


def test_detects_frozen_series_like_neoe3():
    p = _frozen_tail(_prices(), "AAA3", 5, 33.799999)
    assert set(detect_stale_prices(p)) == {"AAA3"}


def test_low_price_tick_noise_is_not_stale():
    # RAIZ4 a R$0,42 teve 5 fechamentos iguais negociando normalmente
    p = _frozen_tail(_prices(), "AAA3", 5, 0.42)
    assert detect_stale_prices(p) == {}
    p = _frozen_tail(_prices(), "AAA3", 10, 0.42)
    assert "AAA3" in detect_stale_prices(p)


def test_series_that_stopped_updating_is_stale():
    p = _prices()
    p.iloc[-4:, p.columns.get_loc("BBB3")] = np.nan
    assert "BBB3" in detect_stale_prices(p)
    p = _prices()
    p.iloc[-3:, p.columns.get_loc("BBB3")] = np.nan
    assert detect_stale_prices(p) == {}


def test_live_series_are_not_stale():
    assert detect_stale_prices(_prices()) == {}


def _scored(tickers, vols):
    return pd.DataFrame({"ticker": tickers, "volatility_180d": vols,
                         "total_score": np.linspace(90, 70, len(tickers))})


def test_frozen_ticker_gets_zero_weight():
    tickers = ["AAA3", "BBB3", "CCC3", "DDD3"]
    p = _frozen_tail(_prices(tickers=tickers), "AAA3", 30, 33.8)
    w, _ = _compute_portfolio_weights(_scored(tickers, [0.01, 0.3, 0.3, 0.3]), p, n=4)
    assert w["AAA3"] == 0.0
    assert sum(v for t, v in w.items() if t != "AAA3") == pytest.approx(1.0, abs=1e-4)


def test_inverse_vol_floor_limits_near_zero_vol():
    w = _compute_inv_vol_weights(_scored(["A", "B", "C", "D", "E"], [0.001, 0.15, 0.15, 0.15, 0.15]))
    # sem piso, A teria ~97% e pararia no cap de 30%; com piso de 10% fica em 10/(10+4×6,67)
    assert w["A"] == pytest.approx(10 / (10 + 4 / 0.15), abs=0.005)
    assert w["A"] < 0.30
