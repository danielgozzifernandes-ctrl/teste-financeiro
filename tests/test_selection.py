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
