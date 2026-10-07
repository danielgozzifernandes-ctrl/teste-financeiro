import numpy as np
import pandas as pd
import pytest

from src.history import research as rs


def test_newey_west_matches_plain_t_without_autocorrelation():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0.01, 0.05, 2000))
    plain = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert rs.newey_west_t(x, 0) == pytest.approx(plain, rel=1e-6)
    # séries com autocorrelação positiva têm t menor com lags
    ar = pd.Series(np.convolve(rng.normal(0.01, 0.05, 2010), np.ones(10) / 10, "valid"))
    assert abs(rs.newey_west_t(ar, 9)) < abs(rs.newey_west_t(ar, 0))


def test_rank_ic_and_quintiles():
    s = pd.Series(np.arange(50, dtype=float))
    assert rs.rank_ic(s, s * 2) == pytest.approx(1.0)
    assert rs.rank_ic(s, -s) == pytest.approx(-1.0)
    q = rs.quintile(s)
    assert q.value_counts().tolist() == [10] * 5
    assert q.iloc[-1] == 5 and q.iloc[0] == 1


def test_benjamini_hochberg():
    p = pd.Series({"a": 0.001, "b": 0.01, "c": 0.03, "d": 0.5})
    out = rs.benjamini_hochberg(p, 0.05)
    assert out.to_dict() == {"a": True, "b": True, "c": True, "d": False}
    assert not rs.benjamini_hochberg(pd.Series({"a": 0.2, "b": 0.6})).any()


def test_deflated_sharpe_penalizes_many_trials():
    rng = np.random.default_rng(1)
    e = pd.Series(rng.normal(0.01, 0.04, 120))
    psr = rs.probabilistic_sharpe(e)
    dsr = rs.deflated_sharpe(e, n_trials=100, sr_var=0.01)
    assert 0 < dsr < psr < 1
    assert rs.expected_max_sharpe(1, 0.01) == 0.0


def test_sector_adaptive_z_uses_sector_when_large_enough():
    x = pd.Series(list(range(10)) + [100, 101, 102], dtype=float)
    sec = pd.Series(["A"] * 10 + ["B"] * 3)
    z = rs.sector_adaptive_z(x, sec)
    # setor A (N=10) é padronizado dentro do setor: média ~0
    assert z.iloc[:10].mean() == pytest.approx(0, abs=1e-9)
    # setor B (N=3) cai no z global: muito acima da média
    assert (z.iloc[10:] > 1).all()
    assert z.abs().max() <= 3


def test_inverse_vol_weights_respect_bounds():
    w = rs.inverse_vol_weights(pd.Series([0.05, 0.2, 0.3, 0.4, 0.6]))
    assert w.sum() == pytest.approx(1)
    assert w.max() <= 0.30 + 1e-9 and w.min() >= 0.05 - 1e-9


def test_friction_scales_with_illiquidity():
    f = rs.friction(pd.Series([1e9, 50e6, 12.5e6, 1e3]))
    assert f.iloc[0] == pytest.approx(0.0013)
    assert f.iloc[1] == pytest.approx(0.0013)
    assert f.iloc[2] == pytest.approx(0.0026)
    assert f.iloc[3] == pytest.approx(0.0065)  # teto de 5×


def test_portfolio_returns_charges_cost_on_turnover_only():
    rows = []
    for t in pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31"]):
        for i in range(20):
            rows.append({"t": t, "sid": f"S{i}", "score": float(i), "fwd": 0.0,
                         "adv30": 1e9, "vol126": 0.3})
    panel = pd.DataFrame(rows)
    out = rs.portfolio_returns(panel, lambda g: g["score"], "quintile")
    # mesma carteira todo mês e retorno zero: só o 1º mês paga entrada
    assert out["Q5_net"].iloc[0] == pytest.approx(-0.0013)
    assert out["Q5_net"].iloc[1:].abs().max() == pytest.approx(0)
    assert out["Q5_turnover"].iloc[1:].max() == pytest.approx(0)


def test_mad_winsor_limits_outliers_only():
    x = pd.Series([1.0, 2, 3, 4, 5, 1000])
    w = rs.mad_winsor(x)
    assert w.iloc[:5].tolist() == [1, 2, 3, 4, 5]
    assert w.iloc[-1] < 20


def test_prepare_drops_month_with_unpriced_event():
    t = pd.to_datetime(["2021-02-26", "2021-03-31"])
    panel = pd.DataFrame({"t": t, "sid": ["PCAR3", "PCAR3"], "fwd": [-0.7, 0.01],
                          "is_fin": [False, False]})
    for c in ("mcap", "ni", "equity", "debt", "cash", "ebit", "gp", "ocf", "assets", "dna"):
        panel[c] = 1.0
    panel["sector"] = "x"
    out = rs.prepare(panel)
    assert np.isnan(out.loc[0, "fwd"]) and out.loc[1, "fwd"] == 0.01
