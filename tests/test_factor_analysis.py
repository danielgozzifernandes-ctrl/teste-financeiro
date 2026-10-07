"""IC no universo inteiro, mesma série de preço nas duas pontas, NW + BH."""

import json

import numpy as np
import pandas as pd
import pytest

from src import factor_analysis as fa


def test_newey_west_lag0_matches_plain_t():
    x = np.array([0.1, -0.05, 0.2, 0.05, 0.0, 0.15])
    t0 = fa.newey_west_t(x, 0)
    plain = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert t0 == pytest.approx(plain)


def test_benjamini_hochberg():
    q = fa.benjamini_hochberg([0.01, 0.04, 0.03, 0.5])
    assert q == pytest.approx([0.04, 0.0533333, 0.0533333, 0.5], rel=1e-4)


def test_entry_is_first_close_after_run():
    dates = pd.bdate_range("2026-09-28", periods=5)
    assert fa._entry_index(dates, pd.Timestamp("2026-09-28 11:00")) == 0
    assert fa._entry_index(dates, pd.Timestamp("2026-09-28 19:30")) == 1
    assert fa._entry_index(dates, pd.Timestamp("2026-09-27 08:00")) == 0


def _universe_history(tmp_path, n_recs=10, n_tickers=15, top10_only_first=True):
    """Fator 'good' ordena perfeitamente o retorno da semana seguinte."""
    dates = pd.bdate_range("2026-01-05", periods=n_recs * 5 + 70)
    tickers = [f"T{i:02d}3" for i in range(n_tickers)]
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(rng.normal(0, 0.01, (len(dates), n_tickers)), index=dates, columns=tickers)
    for k in range(n_recs):
        i0 = k * 5
        for j, t in enumerate(tickers):           # semana seguinte: retorno cresce com j
            rets.iloc[i0 + 1:i0 + 6, j] = 0.001 * j
    prices = (1 + rets).cumprod() * 10

    for k in range(n_recs):
        day = dates[k * 5]
        rec = {
            "date": day.strftime("%Y-%m-%d"), "mode": "weekly",
            "execution_metadata": {"market_data": {"run_at_brt": f"{day.date()}T11:00:00-03:00"}},
            "full_universe_scores": {
                t: {"good": float(j), "noise": float(rng.normal())}
                for j, t in enumerate(tickers)
            },
        }
        if top10_only_first and k == 0:
            rec.pop("full_universe_scores")
            rec["top10"] = [{"ticker": t, "norm_details": {"good": {"score": 1.0}}} for t in tickers]
        (tmp_path / f"recommendations_{rec['date']}_weekly.json").write_text(
            json.dumps(rec), encoding="utf-8")
    return lambda tk, s, e: prices[tk]


def test_ic_uses_full_universe_and_flags_real_signal(tmp_path):
    loader = _universe_history(tmp_path, n_recs=12)
    r = fa.analyze_factors(history_dir=tmp_path, output_path=tmp_path / "ic.json",
                           verbose=False, price_loader=loader)
    good = r["factors"]["good"]["1w"]
    assert r["n_recommendations_full_universe"] == 11      # a só-top10 fica fora
    assert good["mean_ic"] == pytest.approx(1.0)
    assert good["mean_n_tickers"] == 15
    assert r["usage"] == "monitor"
    # 4w/12w: poucas janelas independentes → sem teste
    assert r["factors"]["good"]["4w"].get("testable") in (False, None)
    assert json.loads((tmp_path / "ic.json").read_text(encoding="utf-8"))["factors"]


def test_exclude_from_drops_windows_ending_late(tmp_path):
    loader = _universe_history(tmp_path, n_recs=12)
    full = fa.analyze_factors(history_dir=tmp_path, output_path=None, verbose=False,
                              price_loader=loader)
    cut = fa.analyze_factors(history_dir=tmp_path, output_path=None, verbose=False,
                             price_loader=loader, exclude_from="2026-02-16")
    assert cut["factors"]["good"]["1w"]["n_obs"] < full["factors"]["good"]["1w"]["n_obs"]
