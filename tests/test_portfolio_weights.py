"""
Testes de bounds de peso por posição (_apply_weight_bounds) e do skip-month
no momentum.

Contexto: em 08/06/2026 o HRP cru alocou 63,4% em NEOE3 (HHI 0,44,
N efetivo 2,28). Os bounds cap=30%/floor=5% garantem diversificação mínima
real independente do método de pesos.
"""

import numpy as np
import pandas as pd
import pytest

from src.snapshot_manager import _apply_weight_bounds
from src.scoring_engine import ScoringEngine


# ─── _apply_weight_bounds ────────────────────────────────────────────────────

def test_cap_reduces_concentrated_position():
    # Caso real de 08/06/2026
    w = {"PETR4": 0.133174, "ABEV3": 0.103688, "USIM5": 0.048916,
         "NEOE3": 0.633633, "INTB3": 0.080589}
    bounded = _apply_weight_bounds(w, cap=0.30, floor=0.05)

    assert abs(sum(bounded.values()) - 1.0) < 1e-6
    assert all(v <= 0.30 + 1e-6 for v in bounded.values())
    assert all(v >= 0.05 - 1e-6 for v in bounded.values())
    assert bounded["NEOE3"] == pytest.approx(0.30, abs=1e-6)


def test_no_change_when_within_bounds():
    w = {"A": 0.25, "B": 0.25, "C": 0.20, "D": 0.15, "E": 0.15}
    bounded = _apply_weight_bounds(w, cap=0.30, floor=0.05)
    for t, v in w.items():
        assert bounded[t] == pytest.approx(v, abs=1e-6)


def test_floor_raises_token_position():
    w = {"A": 0.30, "B": 0.30, "C": 0.30, "D": 0.09, "E": 0.01}
    bounded = _apply_weight_bounds(w, cap=0.30, floor=0.05)
    assert bounded["E"] >= 0.05 - 1e-6
    assert abs(sum(bounded.values()) - 1.0) < 1e-6
    assert all(v <= 0.30 + 1e-6 for v in bounded.values())


def test_infeasible_cap_falls_back_to_equal_weight():
    w = {"A": 0.6, "B": 0.4}
    bounded = _apply_weight_bounds(w, cap=0.30, floor=0.05)  # 2×0.30 < 1
    assert bounded == {"A": 0.5, "B": 0.5}


def test_empty_weights_passthrough():
    assert _apply_weight_bounds({}) == {}


def test_zero_total_falls_back_to_equal_weight():
    bounded = _apply_weight_bounds({"A": 0.0, "B": 0.0})
    assert bounded == {"A": 0.5, "B": 0.5}


def test_effective_n_improves():
    """HHI deve cair (N efetivo subir) após bounds no caso real."""
    w = {"PETR4": 0.133174, "ABEV3": 0.103688, "USIM5": 0.048916,
         "NEOE3": 0.633633, "INTB3": 0.080589}
    hhi_before = sum(v ** 2 for v in w.values())
    bounded = _apply_weight_bounds(w, cap=0.30, floor=0.05)
    hhi_after = sum(v ** 2 for v in bounded.values())
    assert hhi_after < hhi_before
    assert 1.0 / hhi_after > 3.0  # N efetivo > 3 com cap 30%


# ─── Momentum skip-month ─────────────────────────────────────────────────────

def _price_frame(n_days: int, final_jump: float) -> pd.DataFrame:
    """Série flat com salto de final_jump nos últimos 21 dias."""
    base = np.full(n_days, 100.0)
    base[-21:] *= (1.0 + final_jump)
    idx = pd.date_range("2025-01-01", periods=n_days, freq="B")
    return pd.DataFrame({"TICK3": base}, index=idx)


def test_trailing_returns_skip_ignores_last_month():
    """Com skip=21, um rally só no último mês NÃO gera momentum."""
    df = _price_frame(300, final_jump=0.50)
    no_skip = ScoringEngine._trailing_returns(df, 252)["TICK3"]
    skipped = ScoringEngine._trailing_returns(df, 252, skip_days=21)["TICK3"]
    assert no_skip == pytest.approx(0.50, abs=1e-9)
    assert skipped == pytest.approx(0.0, abs=1e-9)


def test_trailing_returns_skip_insufficient_data_nan():
    df = _price_frame(15, final_jump=0.0)
    out = ScoringEngine._trailing_returns(df, 252, skip_days=21)
    assert out["TICK3"] != out["TICK3"]  # NaN


def test_scalar_return_skip_consistency():
    df = _price_frame(300, final_jump=0.50)
    series = df["TICK3"]
    assert ScoringEngine._scalar_return(series, 252, skip_days=21) == \
        pytest.approx(0.0, abs=1e-9)
    assert ScoringEngine._scalar_return(series, 252) == \
        pytest.approx(0.50, abs=1e-9)
