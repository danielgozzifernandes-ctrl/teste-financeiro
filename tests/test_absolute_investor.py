"""
Testes da camada "investidor absoluto":
  - allocator (sleeves, tilts, bounds, gross exposure, dado faltante)
  - stop_monitor (stop_hit/near/target, recomendações antigas sem trade_advice)
  - equity_curve (encadeamento de NAV, dedup por data, banda de ruído)
  - order_sheet (quantidades fracionárias, sobra → CDI, nota de IR)
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.allocator import (
    apply_gross_exposure,
    compute_allocation,
    portfolio_earnings_yield,
    selic_annual_from_daily,
)
from src.config import (
    ALLOCATION_BASE,
    ALLOCATION_TILT_PP,
    EQUITIES_SLEEVE_MAX,
    EQUITIES_SLEEVE_MIN,
)
from src.equity_curve import (
    alpha_noise_band_pp,
    summarize,
    update_equity_curve,
    DEFAULT_ALPHA_NOISE_PP,
)
from src.order_sheet import build_order_sheet
from src.stop_monitor import check_levels


# Fixtures

def _ibov_series(n: int = 300, trend: float = 0.001) -> pd.Series:
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.Series(100.0 * np.cumprod(np.full(n, 1 + trend)), index=idx)


def _cdi_series(n: int = 300, daily: float = 0.00055) -> pd.Series:
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.Series(np.full(n, daily), index=idx)


# Allocator

def test_allocation_sums_to_one_all_regimes():
    for regime in ("risk_on", "mean_rev", "bear"):
        out = compute_allocation(
            regime=regime, portfolio_earnings_yield=0.12, selic_annual=0.15,
            ibov_prices=_ibov_series(), cdi_daily_returns=_cdi_series(),
        )
        assert abs(sum(out["sleeves"].values()) - 1.0) < 1e-3
        assert EQUITIES_SLEEVE_MIN - 1e-9 <= out["sleeves"]["equities_br"] \
            <= EQUITIES_SLEEVE_MAX + 1e-9


def test_bear_allocates_less_equity_than_risk_on():
    kwargs = dict(
        portfolio_earnings_yield=0.12, selic_annual=0.15,
        ibov_prices=_ibov_series(), cdi_daily_returns=_cdi_series(),
    )
    bear = compute_allocation(regime="bear", **kwargs)
    bull = compute_allocation(regime="risk_on", **kwargs)
    assert bear["sleeves"]["equities_br"] < bull["sleeves"]["equities_br"]


def test_low_erp_reduces_equity():
    """EY 10% com Selic 15% → ERP −5pp < threshold → tilt negativo."""
    base = ALLOCATION_BASE["mean_rev"]["equities_br"]
    out = compute_allocation(
        regime="mean_rev", portfolio_earnings_yield=0.10, selic_annual=0.15,
        ibov_prices=_ibov_series(trend=0.002),  # TSMOM positivo (+tilt)
        cdi_daily_returns=_cdi_series(),
    )
    # tilts: ERP −0.10, TSMOM +0.10 → líquido = base
    assert out["signals"]["erp"] == pytest.approx(-0.05, abs=1e-6)
    assert out["signals"]["erp_tilt"] == -ALLOCATION_TILT_PP
    assert out["sleeves"]["equities_br"] == pytest.approx(base, abs=0.02)


def test_negative_tsmom_reduces_equity():
    out = compute_allocation(
        regime="mean_rev", portfolio_earnings_yield=None, selic_annual=None,
        ibov_prices=_ibov_series(trend=-0.002),  # bear market 12m
        cdi_daily_returns=_cdi_series(),
    )
    assert out["signals"]["tsmom_tilt"] == -ALLOCATION_TILT_PP
    assert out["signals"]["erp_tilt"] == 0.0  # sem dado → tilt 0


def test_missing_data_degrades_to_base():
    out = compute_allocation(
        regime="mean_rev", portfolio_earnings_yield=None, selic_annual=None,
        ibov_prices=None, cdi_daily_returns=None,
    )
    base = ALLOCATION_BASE["mean_rev"]
    assert out["sleeves"]["equities_br"] == pytest.approx(base["equities_br"], abs=1e-3)
    assert out["signals"]["erp"] is None
    assert out["signals"]["tsmom"] is None


def test_apply_gross_exposure_moves_equity_to_cdi():
    out = compute_allocation(
        regime="mean_rev", portfolio_earnings_yield=None, selic_annual=None,
        ibov_prices=None, cdi_daily_returns=None,
    )
    eq_before, cdi_before = out["sleeves"]["equities_br"], out["sleeves"]["cdi"]
    scaled = apply_gross_exposure(out, 0.5)
    assert scaled["sleeves"]["equities_br"] == pytest.approx(eq_before * 0.5, abs=1e-3)
    assert scaled["sleeves"]["cdi"] == pytest.approx(
        cdi_before + eq_before * 0.5, abs=1e-3)
    assert abs(sum(scaled["sleeves"].values()) - 1.0) < 1e-3
    # Idempotente em gross >= 1
    assert apply_gross_exposure(out, 1.0)["sleeves"] == out["sleeves"]


def test_selic_annualization():
    s = pd.Series([0.00055] * 10)  # ~15% a.a.
    annual = selic_annual_from_daily(s)
    assert 0.14 < annual < 0.16
    assert selic_annual_from_daily(pd.Series(dtype=float)) is None


def test_portfolio_earnings_yield_requires_majority():
    df = pd.DataFrame({"earnings_yield": [0.20, 0.10, np.nan, np.nan, np.nan]})
    assert portfolio_earnings_yield(df) == pytest.approx(0.15)
    df_sparse = pd.DataFrame({"earnings_yield": [0.20, np.nan, np.nan, np.nan, np.nan]})
    assert portfolio_earnings_yield(df_sparse) is None


# Stop monitor

_REC = {
    "trade_advice": {
        "PETR4": {"stop": 40.0, "target_conservative": 50.0},
        "VALE3": {"stop": 70.0, "target_conservative": 90.0},
        "ABEV3": {"stop": 14.0, "target_conservative": 18.0},
    }
}


def test_stop_hit_detected():
    alerts = check_levels(_REC, {"PETR4": 39.50, "VALE3": 75.0, "ABEV3": 16.0})
    kinds = {a["ticker"]: a["kind"] for a in alerts}
    assert kinds.get("PETR4") == "stop_hit"
    assert "VALE3" not in kinds and "ABEV3" not in kinds


def test_stop_near_and_target_hit():
    alerts = check_levels(_REC, {"VALE3": 70.5, "ABEV3": 18.2})
    kinds = {a["ticker"]: a["kind"] for a in alerts}
    assert kinds.get("VALE3") == "stop_near"
    assert kinds.get("ABEV3") == "target_hit"
    # severidade: stop_near vem antes de target_hit
    assert alerts[0]["kind"] == "stop_near"


def test_old_recommendation_without_trade_advice():
    assert check_levels({"top5": []}, {"PETR4": 10.0}) == []
    assert check_levels(None, {}) == []


# Equity curve

def test_equity_curve_chains_nav(tmp_path):
    path = tmp_path / "curve.json"
    update_equity_curve("2026-06-01", 0.01, 0.005, 0.0005, path=path)
    s = update_equity_curve("2026-06-02", -0.02, -0.01, 0.0005, path=path)
    assert s["n_obs"] == 2
    assert s["nav"] == pytest.approx(100 * 1.01 * 0.98, abs=1e-4)
    assert s["ibov_nav"] == pytest.approx(100 * 1.005 * 0.99, abs=1e-4)
    assert s["cdi_nav"] == pytest.approx(100 * 1.0005 ** 2, abs=1e-4)
    assert s["alpha_vs_cdi_pp"] < 0  # perdeu do CDI — a régua absoluta


def test_equity_curve_dedup_same_date(tmp_path):
    path = tmp_path / "curve.json"
    update_equity_curve("2026-06-01", 0.01, 0.005, 0.0005, path=path)
    s = update_equity_curve("2026-06-01", 0.02, 0.005, 0.0005, path=path)  # re-run
    assert s["n_obs"] == 1
    assert s["nav"] == pytest.approx(102.0, abs=1e-4)


def test_equity_curve_cdi_fallback_uses_last_rate(tmp_path):
    path = tmp_path / "curve.json"
    update_equity_curve("2026-06-01", 0.0, 0.0, 0.0006, path=path)
    s = update_equity_curve("2026-06-02", 0.0, 0.0, None, path=path)  # BCB falhou
    assert s["cdi_nav"] == pytest.approx(100 * 1.0006 ** 2, abs=1e-4)
    # lacuna auditável: cdi_ret null no JSON
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["series"][-1]["cdi_ret"] is None


def test_noise_band_default_with_small_sample(tmp_path):
    path = tmp_path / "curve.json"
    update_equity_curve("2026-06-01", 0.01, 0.005, None, path=path)
    assert alpha_noise_band_pp(path=path) == DEFAULT_ALPHA_NOISE_PP


def test_summarize_empty():
    s = summarize(series=[])
    assert s["n_obs"] == 0 and s["nav"] == 100.0
    assert s["blended_nav"] is None


def test_blended_nav_tracks_full_portfolio(tmp_path):
    """A carteira completa (blended) é encadeada separada do sleeve bolsa."""
    path = tmp_path / "curve.json"
    # Dia 1: bolsa -2%, mas carteira completa (40% bolsa) só -0.5%
    update_equity_curve("2026-06-01", -0.02, -0.01, 0.0005,
                        blended_daily_return=-0.005, path=path)
    s = update_equity_curve("2026-06-02", 0.01, 0.005, 0.0005,
                            blended_daily_return=0.004, path=path)
    assert s["nav"] == pytest.approx(100 * 0.98 * 1.01, abs=1e-4)
    assert s["blended_nav"] == pytest.approx(100 * 0.995 * 1.004, abs=1e-4)
    # blended deve ter caído menos que o sleeve bolsa (diversificação)
    assert s["blended_nav"] > s["nav"]
    assert s["blended_alpha_vs_cdi_pp"] is not None


def test_blended_absent_stays_none(tmp_path):
    """Recs antigas sem allocation: blended fica None, não vira curva de zeros."""
    path = tmp_path / "curve.json"
    s = update_equity_curve("2026-06-01", 0.01, 0.0, 0.0005,
                            blended_daily_return=None, path=path)
    assert s["blended_nav"] is None
    assert s["cum_blended"] is None


def test_weighted_portfolio_return_uses_real_weights():
    from main_daily import _weighted_portfolio_return
    rets = {"A": 0.10, "B": -0.10}
    # equal-weight daria 0; pesos 80/20 dão +6%
    assert _weighted_portfolio_return(rets, {"A": 0.8, "B": 0.2}) == \
        pytest.approx(0.06)
    # sem pesos → fallback equal-weight
    assert _weighted_portfolio_return(rets, None) == pytest.approx(0.0)
    # nenhuma posição da carteira tem retorno → sem dado
    assert _weighted_portfolio_return(rets, {"C": 1.0}) is None


# Order sheet

_ALLOC = {"sleeves": {"equities_br": 0.40, "cdi": 0.30,
                      "global_usd": 0.15, "inflation": 0.15}}
_WEIGHTS = {"PETR4": 0.30, "VALE3": 0.30, "ABEV3": 0.40}
_PRICES = {"PETR4": 40.0, "VALE3": 80.0, "ABEV3": 16.0}


def test_order_sheet_quantities_and_residual():
    sheet = build_order_sheet(50_000, _ALLOC, _WEIGHTS, _PRICES)
    assert sheet["capital_brl"] == 50_000
    # sleeve bolsa = 20k; PETR4 30% = 6000/40 = 150x
    petr = next(o for o in sheet["equity_orders"] if o["ticker"] == "PETR4")
    assert petr["qty"] == 150
    assert petr["value_brl"] == pytest.approx(6_000.0)
    # valor executado + CDI ajustado: nada fica "no ar"
    executed = sum(o["value_brl"] for o in sheet["equity_orders"])
    assert executed + sheet["sleeve_values"]["cdi"] == pytest.approx(
        50_000 * (0.40 + 0.30) + 0, abs=1.0)


def test_order_sheet_rotation_and_tax_note():
    sheet = build_order_sheet(
        500_000, _ALLOC, _WEIGHTS, _PRICES,
        previous_tickers=["PETR4", "VALE3", "USIM5", "NEOE3", "INTB3"],
    )
    assert set(sheet["rotation"]["exits"]) == {"USIM5", "NEOE3", "INTB3"}
    assert sheet["rotation"]["entries"] == ["ABEV3"]
    # 3 saídas × (200k/5) = 120k > isenção de 20k → nota menciona IR
    assert sheet["estimated_sales_brl"] > 20_000
    assert "ACIMA" in sheet["tax_note"]


def test_order_sheet_small_capital_exempt():
    sheet = build_order_sheet(
        10_000, _ALLOC, _WEIGHTS, _PRICES,
        previous_tickers=["USIM5", "PETR4", "VALE3"],
    )
    assert sheet["estimated_sales_brl"] < 20_000
    assert "isenção" in sheet["tax_note"]


def test_order_sheet_invalid_capital():
    assert build_order_sheet(0, _ALLOC, _WEIGHTS, _PRICES) is None
    assert build_order_sheet(-1, _ALLOC, _WEIGHTS, _PRICES) is None


def test_order_sheet_without_allocation_is_full_equity():
    sheet = build_order_sheet(10_000, None, _WEIGHTS, _PRICES)
    assert sheet["sleeve_values"]["equities_br"] == 10_000
