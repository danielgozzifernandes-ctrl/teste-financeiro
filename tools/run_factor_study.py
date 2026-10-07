"""Roda o estudo pré-registrado sobre o painel e grava tabelas em DERIVED_DIR/study/."""
import itertools
import json
import warnings

import numpy as np
import pandas as pd

from src.history import research as rs
from src.history.paths import DERIVED_DIR

warnings.filterwarnings("ignore")
OUT = DERIVED_DIR / "study"
OUT.mkdir(exist_ok=True)

panel = rs.prepare(pd.read_parquet(DERIVED_DIR / "research_panel.parquet"))
ibov = pd.read_parquet(DERIVED_DIR / "bench_ibov.parquet")["ibov"]
cdi_d = pd.read_parquet(DERIVED_DIR / "bench_cdi.parquet")["cdi"]
prices = pd.read_parquet(DERIVED_DIR / "prices_daily.parquet", columns=["date", "sid", "tr_close"])
tr = prices.pivot(index="date", columns="sid", values="tr_close").sort_index()

dates = sorted(panel["t"].unique())
nxt = dict(zip(dates[:-1], dates[1:]))
nxt[dates[-1]] = pd.Timestamp("2026-09-30")


def period_ret(series, a, b):
    s = series.sort_index()
    va, vb = s.asof(a), s.asof(b)
    return vb / va - 1


cdi_idx = (1 + cdi_d).cumprod()
bench = pd.DataFrame({
    "cdi": [period_ret(cdi_idx, t, nxt[t]) for t in dates],
    "ibov": [period_ret(ibov, t, nxt[t]) for t in dates],
}, index=pd.Index(dates, name="t"))

TRAIN = bench.index <= rs.TRAIN_END
TEST = ~TRAIN
SUBS = {"2011-14": ("2011", "2014"), "2015-18": ("2015", "2018"),
        "2019-22": ("2019", "2022"), "2023-26": ("2023", "2026")}

results = {}

# --- cobertura do vínculo ----------------------------------------------------
cov = panel.groupby("t").agg(
    members=("sid", "size"),
    linked=("cd_cvm", lambda x: x.notna().mean()),
    fundamentals=("ni", lambda x: x.notna().mean()),
    mcap=("mcap", lambda x: x.notna().mean()),
)
results["coverage"] = {
    "members_median": float(cov["members"].median()),
    "linked_mean": float(cov["linked"].mean()),
    "fundamentals_mean": float(cov["fundamentals"].mean()),
    "mcap_mean": float(cov["mcap"].mean()),
    "mcap_first_year": float(cov.loc[:"2011-12-31", "mcap"].mean()),
    "mcap_last_year": float(cov.loc["2026-01-01":, "mcap"].mean()),
}

# --- IC por fator ---------------------------------------------------------------
def fwd_h(t, h):
    i = dates.index(t)
    end = dates[i + h] if i + h < len(dates) else None
    if end is None:
        return None
    a, b = tr.loc[:t].iloc[-1], tr.loc[:end].ffill().iloc[-1]
    return b / a - 1


fwd_cache = {h: {t: fwd_h(t, h) for t in dates} for h in (3, 6, 12)}


def signal(g, f, neutral):
    x = g[f].astype(float) * rs.FACTORS[f]
    if f == "LIQ":
        x = np.log(g[f].clip(lower=1)) * rs.FACTORS[f]
    return rs.sector_neutral(x, g["sector"]) if neutral else x


ic_rows, ic_series = [], {}
for f, neutral in itertools.product(rs.FACTORS, (False, True)):
    name = f + ("_SN" if neutral else "")
    s = {}
    dec = {h: {} for h in (3, 6, 12)}
    for t, g in panel.groupby("t"):
        g = g.set_index("sid")
        x = signal(g, f, neutral)
        s[t] = rs.rank_ic(x, g["fwd"])
        for h in (3, 6, 12):
            fw = fwd_cache[h][t]
            if fw is not None:
                dec[h][t] = rs.rank_ic(x, fw.reindex(g.index))
    s = pd.Series(s)
    ic_series[name] = s
    row = {"factor": name}
    for lab, mask in (("train", s.index <= rs.TRAIN_END), ("test", s.index > rs.TRAIN_END)):
        v = s[mask].dropna()
        row[f"{lab}_ic"] = v.mean()
        row[f"{lab}_t"] = rs.newey_west_t(v, 3)
        row[f"{lab}_p"] = rs.two_sided_p(row[f"{lab}_t"])
        row[f"{lab}_n"] = len(v)
    for h in (3, 6, 12):
        v = pd.Series(dec[h]).dropna()
        row[f"ic_{h}m"] = v.mean()
        row[f"t_{h}m"] = rs.newey_west_t(v, h)
    for sub, (a, b) in SUBS.items():
        row[f"ic_{sub}"] = s[a:b].mean()
    ic_rows.append(row)
ic = pd.DataFrame(ic_rows).set_index("factor")
ic["train_bh"] = rs.benjamini_hochberg(ic["train_p"])
ic["test_bh"] = rs.benjamini_hochberg(ic["test_p"])
ic.to_csv(OUT / "ic.csv")


# --- quintis por fator ------------------------------------------------------------
def stats(ret: pd.Series, ref: pd.Series, mask) -> dict:
    e = (ret - ref)[mask].dropna()
    return {"ann": float((1 + ret[mask].dropna()).prod() ** (12 / max(len(ret[mask].dropna()), 1)) - 1),
            "excess_ann": float(e.mean() * 12), "sharpe": rs.sharpe(e), "t": rs.newey_west_t(e, 3),
            "n": int(len(e))}


q_rows = []
quint = {}
for f, neutral in itertools.product(rs.FACTORS, (False, True)):
    name = f + ("_SN" if neutral else "")
    pr = rs.portfolio_returns(panel, lambda g, f=f, n=neutral: signal(g, f, n), "quintile")
    pr = pr.join(bench, how="left")
    quint[name] = pr
    row = {"factor": name, "turnover_Q5": pr["Q5_turnover"].mean()}
    for lab, m in (("train", pr.index <= rs.TRAIN_END), ("test", pr.index > rs.TRAIN_END)):
        for k, ref in (("LS_net", 0), ("Q5_net", pr["ew_univ"]), ("Q5_net_ibov", pr["ibov"])):
            col = "Q5_net" if k == "Q5_net_ibov" else k
            r = stats(pr[col], ref if isinstance(ref, pd.Series) else pr[col] * 0, m)
            row[f"{lab}_{k}_excess"] = r["excess_ann"]
            row[f"{lab}_{k}_sharpe"] = r["sharpe"]
            row[f"{lab}_{k}_t"] = r["t"]
        row[f"{lab}_LS_gross"] = float(pr.loc[m, "LS_gross"].mean() * 12)
    q_rows.append(row)
qt = pd.DataFrame(q_rows).set_index("factor")
qt.to_csv(OUT / "quintiles.csv")

# --- combinações -------------------------------------------------------------------
grid = [(a / 4, b / 4, (4 - a - b) / 4) for a in range(5) for b in range(5 - a)]
combo_ret = {}
for w in grid:
    combo_ret[f"G{w}"] = rs.portfolio_returns(panel, lambda g, w=w: rs.composite(g, w), "quintile").join(bench)
combo_ret["C1_repo"] = combo_ret[f"G{(0.45, 0.30, 0.25)}"] if f"G{(0.45, 0.30, 0.25)}" in combo_ret else \
    rs.portfolio_returns(panel, rs.composite, "quintile").join(bench)
combo_ret["C2_momval"] = rs.portfolio_returns(panel, rs.mom_value, "quintile").join(bench)

train_sr = {}
for k, pr in combo_ret.items():
    e = (pr["Q5_net"] - pr["cdi"])[pr.index <= rs.TRAIN_END]
    train_sr[k] = rs.sharpe(e)
grid_keys = [f"G{w}" for w in grid]
best = max(grid_keys, key=lambda k: train_sr[k])
best_w = grid[grid_keys.index(best)]
combo_ret["C3_trainbest"] = combo_ret[best]

# Sharpe por período de todas as estratégias de quintil (para a variância do DSR)
all_sr = []
for name, pr in quint.items():
    e = (pr["Q5_net"] - pr["cdi"])[pr.index <= rs.TRAIN_END].dropna()
    all_sr.append(e.mean() / e.std())
for k in grid_keys + ["C1_repo", "C2_momval"]:
    pr = combo_ret[k]
    e = (pr["Q5_net"] - pr["cdi"])[pr.index <= rs.TRAIN_END].dropna()
    all_sr.append(e.mean() / e.std())
sr_var = float(np.nanvar(all_sr))
N_TRIALS = 82

# --- top-5 do repo ----------------------------------------------------------------
top = {
    "C1_repo": rs.portfolio_returns(panel, rs.composite, "top", filters=True).join(bench),
    "C2_momval": rs.portfolio_returns(panel, rs.mom_value, "top", filters=True).join(bench),
    "C3_trainbest": rs.portfolio_returns(panel, lambda g: rs.composite(g, best_w), "top", filters=True).join(bench),
}
ew = combo_ret["C1_repo"][["ew_univ", "cdi", "ibov"]]


def block_boot_diff(a: pd.Series, b: pd.Series, block=6, n=2000, seed=0):
    d = pd.concat([a, b], axis=1).dropna()
    rng = np.random.default_rng(seed)
    m = len(d)
    out = []
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + block) % m for s in rng.integers(0, m, m // block + 1)])[:m]
        x = d.iloc[idx]
        out.append(rs.sharpe(x.iloc[:, 0]) - rs.sharpe(x.iloc[:, 1]))
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def _cut(x, period):
    a, b = period
    return x[(x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b))]


def summary(ret: pd.Series, b: pd.DataFrame, period) -> dict:
    r = _cut(ret, period).dropna()
    e = _cut(ret - b["cdi"].reindex(ret.index), period).dropna()
    ei = _cut(ret - b["ibov"].reindex(ret.index), period).dropna()
    nav = (1 + r).cumprod()
    return {"ann": float(nav.iloc[-1] ** (12 / len(r)) - 1), "vol": float(r.std() * np.sqrt(12)),
            "sharpe_cdi": rs.sharpe(e), "excess_ibov_ann": float(ei.mean() * 12),
            "t_ibov": rs.newey_west_t(ei, 3), "maxdd": float((nav / nav.cummax() - 1).min()),
            "psr": rs.probabilistic_sharpe(e),
            "dsr": rs.deflated_sharpe(e, N_TRIALS, sr_var), "n": int(len(r))}


PERIODS = {"train": ("2011-01-01", "2018-12-31"), "test": ("2019-01-01", "2026-08-31")}
comp = {}
for lab, per in PERIODS.items():
    comp[lab] = {
        "EW_univ": summary(ew["ew_univ"], bench, per),
        "IBOV": summary(ew["ibov"], bench, per),
        **{f"Q5_{k}": summary(combo_ret[k]["Q5_net"], bench, per) for k in ("C1_repo", "C2_momval", "C3_trainbest")},
        **{f"TOP5_{k}": summary(top[k]["TOP_net"], bench, per) for k in top},
    }

subs = {}
for sub, (a, b) in SUBS.items():
    per = (a + "-01-01", b + "-12-31")
    subs[sub] = {k: summary(v, bench, per)["sharpe_cdi"] for k, v in {
        "EW_univ": ew["ew_univ"], "IBOV": ew["ibov"],
        "Q5_C1": combo_ret["C1_repo"]["Q5_net"], "Q5_C2": combo_ret["C2_momval"]["Q5_net"],
        "Q5_C3": combo_ret["C3_trainbest"]["Q5_net"],
        "TOP5_C1": top["C1_repo"]["TOP_net"], "TOP5_C2": top["C2_momval"]["TOP_net"],
        "TOP5_C3": top["C3_trainbest"]["TOP_net"]}.items()}

def ex_cdi_test(ret):
    return _cut(ret - bench["cdi"].reindex(ret.index), PERIODS["test"])


boot = {
    "Q5_C3_minus_C1": block_boot_diff(ex_cdi_test(combo_ret["C3_trainbest"]["Q5_net"]), ex_cdi_test(combo_ret["C1_repo"]["Q5_net"])),
    "Q5_C2_minus_C1": block_boot_diff(ex_cdi_test(combo_ret["C2_momval"]["Q5_net"]), ex_cdi_test(combo_ret["C1_repo"]["Q5_net"])),
    "TOP5_C2_minus_C1": block_boot_diff(ex_cdi_test(top["C2_momval"]["TOP_net"]), ex_cdi_test(top["C1_repo"]["TOP_net"])),
    "TOP5_C3_minus_C1": block_boot_diff(ex_cdi_test(top["C3_trainbest"]["TOP_net"]), ex_cdi_test(top["C1_repo"]["TOP_net"])),
}


# --- robustez: sem PETR/VALE, sem financeiras ---------------------------------------
def robust(filter_fn, label):
    p2 = panel[filter_fn(panel)]
    out = {}
    for k, fn in (("C1", rs.composite), ("C2", rs.mom_value), ("C3", lambda g: rs.composite(g, best_w))):
        q = rs.portfolio_returns(p2, fn, "quintile").join(bench)
        t5 = rs.portfolio_returns(p2, fn, "top", filters=True).join(bench)
        m = q.index > rs.TRAIN_END
        out[f"Q5_{k}"] = rs.sharpe((q["Q5_net"] - q["cdi"])[m])
        out[f"TOP5_{k}"] = rs.sharpe((t5["TOP_net"] - t5["cdi"])[t5.index > rs.TRAIN_END])
    for f in ("MOM", "EY", "BP", "VOL", "ROE"):
        s = {}
        for t, g in p2.groupby("t"):
            g = g.set_index("sid")
            s[t] = rs.rank_ic(signal(g, f, False), g["fwd"])
        s = pd.Series(s)
        out[f"IC_{f}_test"] = float(s[s.index > rs.TRAIN_END].mean())
        out[f"IC_{f}_t_test"] = rs.newey_west_t(s[s.index > rs.TRAIN_END], 3)
    results[f"robust_{label}"] = out


robust(lambda p: ~p["sid"].str[:4].isin(["PETR", "VALE"]), "ex_petr_vale")
robust(lambda p: ~p["is_fin"].astype(bool), "ex_financials")

import pickle
with open(OUT / "returns.pkl", "wb") as fh:
    pickle.dump({"quint": quint, "combo": combo_ret, "top": top, "bench": bench}, fh)

results.update({
    "best_grid_weights": best_w, "train_sharpe": {k: train_sr[k] for k in ["C1_repo", "C2_momval", best]},
    "sr_var_per_period": sr_var, "n_trials": N_TRIALS, "comparison": comp, "subperiods": subs,
    "bootstrap_sharpe_diff_test": boot,
    "grid_train_sharpe": {str(grid[i]): train_sr[k] for i, k in enumerate(grid_keys)},
})
with open(OUT / "results.json", "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=1, default=float)
print(json.dumps({k: results[k] for k in ("coverage", "best_grid_weights", "train_sharpe")}, indent=1, default=float))
print(ic[["train_ic", "train_t", "test_ic", "test_t", "train_bh", "test_bh", "ic_3m", "ic_12m"]].round(3).to_string())
