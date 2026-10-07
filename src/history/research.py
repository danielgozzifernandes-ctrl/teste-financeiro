"""
Estudo de fatores point-in-time sobre a base histórica (COTAHIST + CVM).

Fluxo: build_panel() monta, para cada fim de mês t, os membros do universo
vigente com sinais calculados só com dados até o pregão anterior (t−1) e o
retorno total de t a t+1. As funções de métrica (IC, quintis, NW, DSR) são
puras e testadas em tests/test_research.py.
"""

from __future__ import annotations

import glob
import logging
from math import erf, sqrt
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.history.paths import DATA_DIR

logger = logging.getLogger(__name__)

# Emissores renomeados ou cancelados que não estão no cadastro atual da B3 nem
# no FCA (2018+). Conferido no cadastro CVM pelo nome.
MANUAL_ISSUERS = {
    "EMBR": "020087", "BRPR": "019925", "CTIP": "021792", "KROT": "017973",
    "GETI": "018350", "TBLE": "017329", "CRUZ": "004057", "AMBV": "018112",
    "PETZ": "025089", "BBTG": "022616", "BISA": "020265", "SMLE": "023140",
    "LLXL": "021482", "PRML": "021482", "ALLL": "017450", "MPXE": "021237",
    "AEDU": "018961", "AMIL": "021172", "TNLP": "017655", "RDCD": "020893",
    "TAMM": "016390", "TCSL": "017639", "CNFB": "004650", "BRTO": "011312",
    "TMAR": "011320", "ENAT": "022365", "NETC": "014621", "VIVO": "017710",
    "DROG": "005258", "OHLB": "019771", "TERI": "022136",
}
# Mesmo emissor com código novo: herda o código CVM do sucessor.
SUCCESSOR_PREFIX = {"HRTP": "PRIO", "BPNM": "BPAN", "TLPP": "VIVT"}

FRICTION_BPS = 13
ADV_REF = 50_000_000
TRAIN_END = pd.Timestamp("2018-12-31")
TEST_LAST_FORMATION = pd.Timestamp("2026-08-31")

# Eventos societários sem preço na base (cisão, reorganização, redução de
# capital): o retorno do mês que os contém sai do estudo.
UNPRICED_EVENTS = [
    ("PCAR3", "2021-03-01"),   # cisão do Assaí
    ("LAME3", "2021-07-19"),   # reorganização Americanas (ações AMER)
    ("LAME4", "2021-07-19"),
    ("BRPR3", "2023-10-04"),   # redução de capital em dinheiro
]

# direção a priori: +1 maior é melhor, −1 menor é melhor
FACTORS = {
    "EY": 1, "BP": 1, "EBIT_EV": 1, "MOM": 1, "REV": -1, "VOL": -1,
    "BETA": -1, "ROE": 1, "GP": 1, "ACC": -1, "SIZE": -1, "LIQ": -1,
}


# ---------------------------------------------------------------------------
# Métricas puras
# ---------------------------------------------------------------------------

def newey_west_t(x: pd.Series, lags: int) -> float:
    x = pd.Series(x).dropna().to_numpy(float)
    n = len(x)
    if n < 3:
        return np.nan
    e = x - x.mean()
    var = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1 - k / (lags + 1)
        var += 2 * w * (e[k:] @ e[:-k]) / n
    if var <= 0:
        return np.nan
    return float(x.mean() / sqrt(var / n))


def rank_ic(signal: pd.Series, fwd: pd.Series, min_n: int = 20) -> float:
    d = pd.concat([signal, fwd], axis=1).dropna()
    if len(d) < min_n:
        return np.nan
    return float(d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank()))


def quintile(signal: pd.Series, q: int = 5) -> pd.Series:
    s = signal.dropna()
    if len(s) < q * 3:
        return pd.Series(dtype=float)
    return pd.qcut(s.rank(method="first"), q, labels=False) + 1


def sharpe(excess: pd.Series, periods: int = 12) -> float:
    e = pd.Series(excess).dropna()
    if len(e) < 6 or e.std() == 0:
        return np.nan
    return float(e.mean() / e.std() * sqrt(periods))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def _norm_ppf(p: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(p))


def probabilistic_sharpe(excess: pd.Series, sr_benchmark: float = 0.0) -> float:
    """PSR de Bailey & López de Prado, com Sharpe por período (não anualizado)."""
    e = pd.Series(excess).dropna()
    n = len(e)
    if n < 6 or e.std() == 0:
        return np.nan
    sr = e.mean() / e.std()
    skew = float(((e - e.mean()) ** 3).mean() / e.std() ** 3)
    kurt = float(((e - e.mean()) ** 4).mean() / e.std() ** 4)
    denom = sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return _norm_cdf((sr - sr_benchmark) * sqrt(n - 1) / denom)


def expected_max_sharpe(n_trials: int, sr_var: float) -> float:
    """Sharpe máximo esperado (por período) entre N tentativas sem habilidade."""
    if n_trials < 2:
        return 0.0
    g = 0.5772156649
    return sqrt(sr_var) * ((1 - g) * _norm_ppf(1 - 1 / n_trials)
                           + g * _norm_ppf(1 - 1 / (n_trials * np.e)))


def deflated_sharpe(excess: pd.Series, n_trials: int, sr_var: float) -> float:
    return probabilistic_sharpe(excess, expected_max_sharpe(n_trials, sr_var))


def benjamini_hochberg(pvals: pd.Series, q: float = 0.05) -> pd.Series:
    p = pd.Series(pvals).dropna().sort_values()
    m = len(p)
    if m == 0:
        return pd.Series(dtype=bool)
    thresh = q * np.arange(1, m + 1) / m
    passed = p.to_numpy() <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    out = pd.Series(False, index=p.index)
    out.iloc[:k] = True
    return out.reindex(pvals.index).fillna(False)


def two_sided_p(t: float) -> float:
    if not np.isfinite(t):
        return np.nan
    return 2 * (1 - _norm_cdf(abs(t)))


def friction(adv: pd.Series) -> pd.Series:
    scale = np.clip(np.sqrt(ADV_REF / adv.clip(lower=1.0)), 1.0, 5.0)
    return FRICTION_BPS / 10_000 * scale


def sector_adaptive_z(x: pd.Series, sector: pd.Series, direction: int = 1) -> pd.Series:
    """N ≥ 8 z setorial; 4–7 percentil setorial → z; < 4 z global. Winsor ±3."""
    x = mad_winsor(x.replace([np.inf, -np.inf], np.nan)) * direction
    out = pd.Series(np.nan, index=x.index)
    g_mean, g_std = x.mean(), x.std()
    for sec, idx in x.groupby(sector.reindex(x.index).fillna("?")).groups.items():
        v = x.loc[idx].dropna()
        if len(v) >= 8 and v.std() > 0:
            out.loc[v.index] = (v - v.mean()) / v.std()
        elif len(v) >= 4:
            pct = (v.rank() - 0.5) / len(v)
            out.loc[v.index] = [_norm_ppf(p) for p in pct]
        elif len(v) and g_std > 0:
            out.loc[v.index] = (v - g_mean) / g_std
    return out.clip(-3, 3)


def mad_winsor(x: pd.Series, k: float = 5.0) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    if not np.isfinite(mad) or mad == 0:
        return x
    return x.clip(med - k * mad, med + k * mad)


def global_z(x: pd.Series, direction: int = 1) -> pd.Series:
    x = mad_winsor(x.replace([np.inf, -np.inf], np.nan)) * direction
    if x.std() == 0 or x.dropna().empty:
        return x * np.nan
    return ((x - x.mean()) / x.std()).clip(-3, 3)


def sector_neutral(x: pd.Series, sector: pd.Series) -> pd.Series:
    r = x.rank(pct=True)
    return r - r.groupby(sector.reindex(r.index).fillna("?")).transform("mean")


def inverse_vol_weights(vol: pd.Series, lo: float = 0.05, hi: float = 0.30) -> pd.Series:
    w = 1 / vol.clip(lower=0.10)
    w = w / w.sum()
    for _ in range(20):
        w = w.clip(lo, hi)
        w = w / w.sum()
        if w.max() <= hi + 1e-9 and w.min() >= lo - 1e-9:
            break
    return w


# ---------------------------------------------------------------------------
# Insumos
# ---------------------------------------------------------------------------

def _month_end_pairs(dates: pd.DatetimeIndex) -> pd.DataFrame:
    d = pd.Series(dates.sort_values().unique())
    me = d.groupby(d.dt.to_period("M")).max()
    prev = {x: d[d < x].max() for x in me}
    return pd.DataFrame({"t": me.values, "t_prev": [prev[x] for x in me]})


def issuer_map(base: Path) -> dict[str, str]:
    cvm = base / "cvm" / "processed"
    iss = pd.read_parquet(cvm / "b3_issuers.parquet")
    tk = pd.read_parquet(cvm / "tickers.parquet")
    m = {t[:4]: c for t, c in zip(tk["ticker"], tk["cd_cvm"])}
    m.update(dict(zip(iss["issuer_code"], iss["cd_cvm"])))
    m.update(MANUAL_ISSUERS)
    for old, new in SUCCESSOR_PREFIX.items():
        if new in m:
            m[old] = m[new]
    return m


def _fetch_benchmarks(start: str = "2010-01-01") -> tuple[pd.Series, pd.Series]:
    import requests
    import yfinance as yf

    raw = yf.download("^BVSP", start=start, progress=False, auto_adjust=True)
    ibov = raw["Close"]
    if isinstance(ibov, pd.DataFrame):
        ibov = ibov.iloc[:, 0]
    ibov.index = pd.to_datetime(ibov.index).tz_localize(None)
    ibov = ibov[(raw["Volume"].iloc[:, 0] if isinstance(raw["Volume"], pd.DataFrame) else raw["Volume"]) > 0]

    parts = []
    for y0 in range(2010, pd.Timestamp.today().year + 1, 9):
        y1 = min(y0 + 8, pd.Timestamp.today().year)
        url = ("https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados?formato=json"
               f"&dataInicial=01/01/{y0}&dataFinal=31/12/{y1}")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        parts.append(pd.DataFrame(r.json()))
    cdi = pd.concat(parts)
    cdi["data"] = pd.to_datetime(cdi["data"], dayfirst=True)
    cdi = cdi.drop_duplicates("data").set_index("data")["valor"].astype(float) / 100
    return ibov.sort_index(), cdi.sort_index()


def _class_closes(base: Path, dates: set[pd.Timestamp]) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(base / "cotahist" / "parquet" / "*.parquet"))):
        d = pd.read_parquet(f, columns=["date", "bdi", "ticker", "close", "volume"])
        d = d[(d["bdi"] == "02") & d["date"].isin(dates)]
        frames.append(d[["date", "ticker", "close"]])
    out = pd.concat(frames, ignore_index=True)
    out["ticker"] = out["ticker"].str.strip()
    return out


# ---------------------------------------------------------------------------
# Painel
# ---------------------------------------------------------------------------

def build_panel(base: Optional[Path] = None) -> pd.DataFrame:
    from src.history.cvm_fundamentals import fundamentals_as_of, load_docs
    from src.history.cvm_reference import shares_as_of

    base = Path(base or DATA_DIR)
    prices = pd.read_parquet(base / "derived" / "prices_daily.parquet")
    univ = pd.read_parquet(base / "derived" / "universe_history.parquet")
    docs = load_docs()
    fre = pd.read_parquet(base / "cvm" / "processed" / "fre_shares.parquet")
    sectors = pd.read_parquet(base / "cvm" / "processed" / "sectors.parquet")
    imap = issuer_map(base)
    ibov, cdi = _fetch_benchmarks()

    tr = prices.pivot(index="date", columns="sid", values="tr_close").sort_index()
    vol_brl = prices.pivot(index="date", columns="sid", values="volume").sort_index()
    ticker_at = prices.pivot(index="date", columns="sid", values="ticker").sort_index()
    split = prices.pivot(index="date", columns="sid", values="split_mult").sort_index().fillna(1.0)
    rets = tr.pct_change(fill_method=None)
    ibov_r = ibov.reindex(tr.index).ffill().pct_change(fill_method=None)

    pairs = _month_end_pairs(tr.index)
    pairs = pairs[(pairs["t"] >= "2011-01-31") & (pairs["t"] <= TEST_LAST_FORMATION)]
    pairs["t_next"] = pairs["t"].shift(-1)
    last_t = pairs["t"].max()
    pairs.loc[pairs["t"] == last_t, "t_next"] = tr.index[tr.index.to_period("M") == (last_t.to_period("M") + 1)].max()

    closes = _class_closes(base, set(pairs["t_prev"]))
    closes = closes.set_index(["date", "ticker"])["close"]

    rebal = sorted(univ["rebalance_date"].unique())
    rows = []
    for _, p in pairs.iterrows():
        t, tp, tn = p["t"], p["t_prev"], p["t_next"]
        rb = max([r for r in rebal if r <= t], default=None)
        if rb is None or pd.isna(tn):
            continue
        members = univ.loc[univ["rebalance_date"] == rb, "sid"].tolist()
        members = [s for s in members if s in tr.columns]
        if not members:
            continue

        hist = tr.loc[:tp, members]
        i = len(hist) - 1
        if i < 260:
            continue
        r = rets.loc[:tp, members]
        px = hist.iloc[-1]
        mom = hist.iloc[i - 21] / hist.iloc[i - 252] - 1
        rev = px / hist.iloc[i - 21] - 1
        ret3 = hist.iloc[i - 21] / hist.iloc[i - 63] - 1
        ret6 = hist.iloc[i - 21] / hist.iloc[i - 126] - 1
        vol252 = r.iloc[-252:].std() * sqrt(252)
        vol180 = r.iloc[-180:].std() * sqrt(252)
        vol126 = r.iloc[-126:].std() * sqrt(252)
        ib = ibov_r.loc[r.index[-252:]]
        rr = r.iloc[-252:]
        beta = rr.apply(lambda c: c.cov(ib) / ib.var() if c.notna().sum() > 150 else np.nan)
        adv63 = vol_brl.loc[:tp, members].iloc[-63:].mean()
        adv30 = vol_brl.loc[:tp, members].iloc[-30:].mean()
        dy = prices[(prices["date"] > tp - pd.Timedelta(days=365)) & (prices["date"] <= tp)
                     & prices["sid"].isin(members)].groupby("sid")["dividend"].sum()

        # retorno t → t+1; sem preço em t+1 = último preço disponível no mês
        seg = tr.loc[t:tn, members]
        start_px = seg.iloc[0]
        end_px = seg.ffill().iloc[-1]
        fwd = end_px / start_px - 1
        traded_next = seg.iloc[1:].notna().any()
        fwd[~traded_next] = 0.0

        fa = fundamentals_as_of(tp, docs)
        fa = fa.set_index("cd_cvm")
        sh = shares_as_of(tp, docs, fre).drop_duplicates("cd_cvm").set_index("cd_cvm")
        sec_year = sectors[sectors["year"] == min(tp.year, sectors["year"].max())]
        sec = sec_year.drop_duplicates("cd_cvm").set_index("cd_cvm")["sector"]

        for sid in members:
            tick = ticker_at.loc[:tp, sid].dropna()
            tick = tick.iloc[-1] if len(tick) else sid
            pre = str(tick)[:4]
            cd = imap.get(pre)
            row = {
                "t": t, "sid": sid, "ticker": tick, "cd_cvm": cd, "fwd": fwd.get(sid),
                "MOM": mom.get(sid), "REV": rev.get(sid), "VOL": vol252.get(sid),
                "BETA": beta.get(sid), "LIQ": adv63.get(sid), "ret3": ret3.get(sid),
                "ret6": ret6.get(sid), "vol180": vol180.get(sid), "vol126": vol126.get(sid),
                "adv30": adv30.get(sid), "dy": (dy.get(sid, 0.0) / px.get(sid)) if px.get(sid) else np.nan,
                "price_raw": closes.get((tp, tick), np.nan),
            }
            if cd is not None and cd in fa.index:
                f = fa.loc[cd]
                if isinstance(f, pd.DataFrame):
                    f = f.iloc[-1]
                row.update({
                    "is_fin": bool(f["is_bank"] or f["is_insurer"]),
                    "sector": sec.get(cd, f.get("sector")),
                    "ni": f["net_income_controlling_ttm"] if pd.notna(f["net_income_controlling_ttm"]) else f["net_income_ttm"],
                    "equity": f["equity_controlling"] if pd.notna(f["equity_controlling"]) else f["equity"],
                    "ebit": f["ebit_ttm"], "gp": f["gross_profit_ttm"], "ocf": f["operating_cash_flow_ttm"],
                    "assets": f["total_assets"], "debt": f["gross_debt"],
                    "cash": (f["cash"] or 0) + (f["short_term_investments"] if pd.notna(f["short_term_investments"]) else 0),
                    "dna": f["dna_ttm"], "rev_ttm": f["revenue_ttm"], "doc_ref": f["dt_refer"],
                })
                if cd in sh.index:
                    s = sh.loc[cd]
                    if isinstance(s, pd.DataFrame):
                        s = s.iloc[-1]
                    row.update({"sh_on": s.get("outstanding_on"), "sh_pn": s.get("outstanding_pn"),
                                "sh_total": s.get("outstanding_total"), "sh_ref": s.get("dt_refer")})
            rows.append(row)
    panel = pd.DataFrame(rows)
    panel = _add_market_cap(panel, closes, split)
    panel = _add_growth(panel, docs, imap)
    panel = _derive(panel)
    panel.attrs["ibov"] = ibov
    panel.attrs["cdi"] = cdi
    return panel


def _add_market_cap(panel: pd.DataFrame, closes: pd.Series, split: pd.DataFrame) -> pd.DataFrame:
    """ON × preço ON + PN × preço PN, ações ajustadas por desdobramentos após o documento."""
    caps = []
    close_df = closes.reset_index()
    close_df["pre"] = close_df["ticker"].str[:4]
    close_df["cls"] = close_df["ticker"].str[4:]
    by_key = close_df.set_index(["date", "pre", "cls"])["close"]
    dates = sorted(close_df["date"].unique())
    for _, r in panel.iterrows():
        tp = max([d for d in dates if d < r["t"]], default=None)
        if tp is None or pd.isna(r.get("sh_total")):
            caps.append(np.nan)
            continue
        pre = str(r["ticker"])[:4]
        mult = 1.0
        if r["sid"] in split.columns and pd.notna(r.get("sh_ref")):
            s = split[r["sid"]]
            mult = float(s[(s.index > pd.Timestamp(r["sh_ref"])) & (s.index <= tp)].prod())

        def px(cls):
            for c in cls:
                v = by_key.get((tp, pre, c))
                if v is not None and pd.notna(v):
                    return float(v)
            return None

        p_on, p_pn = px(["3"]), px(["4", "5", "6"])
        on, pn = r.get("sh_on"), r.get("sh_pn")
        cap = np.nan
        if p_on and pd.notna(on) and on > 0:
            cap = on * p_on + (pn * (p_pn or p_on) if pd.notna(pn) and pn > 0 else 0)
        elif p_pn and pd.notna(r["sh_total"]):
            cap = r["sh_total"] * p_pn
        caps.append(cap * mult if pd.notna(cap) else np.nan)
    panel["mcap"] = caps
    return panel


def _add_growth(panel: pd.DataFrame, docs: pd.DataFrame, imap: dict) -> pd.DataFrame:
    """Crescimento da receita TTM e do ativo em 1 ano, com o documento de ~12 meses antes."""
    d = docs[["cd_cvm", "dt_refer", "available_from", "revenue_ttm", "total_assets"]].copy()
    d["dt_refer"] = pd.to_datetime(d["dt_refer"])
    d = d.sort_values(["cd_cvm", "dt_refer", "available_from"]).drop_duplicates(["cd_cvm", "dt_refer"], keep="last")
    key = d.set_index(["cd_cvm", "dt_refer"])
    rg, ag = [], []
    for cd, ref, rev, assets in zip(panel["cd_cvm"], panel.get("doc_ref"), panel.get("rev_ttm"), panel.get("assets")):
        if cd is None or pd.isna(ref):
            rg.append(np.nan); ag.append(np.nan); continue
        prev = pd.Timestamp(ref) - pd.DateOffset(years=1)
        try:
            p = key.loc[(cd, prev)]
        except KeyError:
            rg.append(np.nan); ag.append(np.nan); continue
        rg.append(rev / p["revenue_ttm"] - 1 if p["revenue_ttm"] and p["revenue_ttm"] > 0 else np.nan)
        ag.append(assets / p["total_assets"] - 1 if p["total_assets"] and p["total_assets"] > 0 else np.nan)
    panel["rev_growth"] = rg
    panel["asset_growth"] = ag
    return panel


def _derive(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    p["is_fin"] = p["is_fin"].astype("boolean").fillna(False).astype(bool)
    pos_cap = p["mcap"].where(p["mcap"] > 0)
    p["EY"] = p["ni"] / pos_cap
    p["BP"] = p["equity"] / pos_cap
    ev = pos_cap + p["debt"].fillna(0) - p["cash"].fillna(0)
    p["EBIT_EV"] = (p["ebit"] / ev.where(ev > 0)).where(~p["is_fin"])
    p["ROE"] = (p["ni"] / p["equity"].where(p["equity"] > 0))
    assets = p["assets"].where(p["assets"] > 0)
    p["GP"] = (p["gp"] / assets).where(~p["is_fin"])
    p["ACC"] = ((p["ni"] - p["ocf"]) / assets).where(~p["is_fin"])
    p["SIZE"] = np.log(pos_cap)
    ic = p["equity"].fillna(0) + p["debt"].fillna(0) - p["cash"].fillna(0)
    p["ROIC"] = (p["ebit"] * 0.66 / ic.where(ic > 0)).where(~p["is_fin"])
    ebitda = p["ebit"] + p["dna"].fillna(0)
    p["ND_EBITDA"] = ((p["debt"].fillna(0) - p["cash"].fillna(0)) / ebitda.where(ebitda > 0)).where(~p["is_fin"])
    p["PE"] = pos_cap / p["ni"].where(p["ni"] > 0)
    p["sector"] = p["sector"].fillna("?")
    return p


def prepare(panel: pd.DataFrame) -> pd.DataFrame:
    """Recalcula os derivados e tira os meses com evento societário sem preço."""
    p = _derive(panel)
    dates = sorted(p["t"].unique())
    nxt = dict(zip(dates[:-1], dates[1:]))
    drop = pd.Series(False, index=p.index)
    for sid, d in UNPRICED_EVENTS:
        d = pd.Timestamp(d)
        hit = (p["sid"] == sid) & (p["t"] < d) & (p["t"].map(nxt).fillna(pd.Timestamp.max) >= d)
        drop |= hit
    p.loc[drop, "fwd"] = np.nan
    return p


# ---------------------------------------------------------------------------
# Scores e carteiras
# ---------------------------------------------------------------------------

FUND_W = {"EY": 0.20, "BP": 0.15, "ROE": 0.25, "ROIC": 0.20, "ND_EBITDA": 0.10, "dy": 0.10,
          "SIZE": 0.05, "rev_growth": 0.08, "asset_growth": 0.05}
FUND_DIR = {"EY": 1, "BP": 1, "ROE": 1, "ROIC": 1, "ND_EBITDA": -1, "dy": 1,
            "SIZE": -1, "rev_growth": 1, "asset_growth": -1}
MOM_W = {"ret3": 0.30, "ret6": 0.40, "MOM": 0.30}
QUAL_W = {"vol180": 0.40, "BETA": 0.35, "adv30": 0.25}
QUAL_DIR = {"vol180": -1, "BETA": -1, "adv30": 1}


def _pillar(g: pd.DataFrame, weights: dict, dirs: dict, sectoral: bool) -> pd.Series:
    zs = {}
    for f, w in weights.items():
        x = g[f].astype(float)
        if f == "adv30":
            x = np.log(x.clip(lower=1))
        zs[f] = sector_adaptive_z(x, g["sector"], dirs.get(f, 1)) if sectoral else global_z(x, dirs.get(f, 1))
    z = pd.DataFrame(zs)
    w = pd.Series(weights)
    avail = z.notna().mul(w, axis=1)
    return (z.fillna(0).mul(w, axis=1).sum(axis=1) / avail.sum(axis=1)).where(avail.sum(axis=1) > 0)


def pillar_scores(g: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "fund": _pillar(g, FUND_W, FUND_DIR, sectoral=True),
        "mom": _pillar(g, MOM_W, {}, sectoral=False),
        "qual": _pillar(g, QUAL_W, QUAL_DIR, sectoral=False),
    })


def repo_filters(g: pd.DataFrame) -> pd.Series:
    ok = g["adv30"] >= 5_000_000
    ok &= ~((~g["is_fin"]) & (g["ND_EBITDA"] > 5))
    ok &= ~((g["ni"] < 0) & (g["ROE"] < -0.05))
    ok &= ~(g["PE"] > 80)
    ok &= ~(g["ROE"] < -0.50)
    return ok.fillna(True)


def composite(g: pd.DataFrame, w: tuple[float, float, float] = (0.45, 0.30, 0.25)) -> pd.Series:
    ps = pillar_scores(g)
    ww = pd.Series({"fund": w[0], "mom": w[1], "qual": w[2]})
    avail = ps.notna().mul(ww, axis=1).sum(axis=1)
    return (ps.fillna(0).mul(ww, axis=1).sum(axis=1) / avail).where(avail > 0)


def mom_value(g: pd.DataFrame) -> pd.Series:
    v = pd.concat([global_z(g["EY"]), global_z(g["BP"])], axis=1).mean(axis=1)
    return 0.5 * global_z(g["MOM"]) + 0.5 * v


def portfolio_returns(panel: pd.DataFrame, score_fn, construction: str = "quintile",
                      filters: bool = False, n_top: int = 5) -> pd.DataFrame:
    """Retornos mensais brutos e líquidos de Q5, Q1, L/S e top-N com custo sobre o giro."""
    out = []
    prev_w: dict[str, pd.Series] = {}
    for t, g in panel.groupby("t"):
        g = g.set_index("sid")
        s = score_fn(g)
        if filters:
            s = s[repo_filters(g).reindex(s.index).fillna(True)]
        s = s.dropna()
        fr = friction(g["adv30"].fillna(ADV_REF))
        rec = {"t": t, "ew_univ": g["fwd"].mean(), "n": len(s)}
        if construction == "quintile":
            q = quintile(s)
            if q.empty:
                continue
            legs = {"Q5": q[q == 5].index, "Q1": q[q == 1].index}
        else:
            top = s.sort_values(ascending=False).head(n_top).index
            legs = {"TOP": top}
        for leg, idx in legs.items():
            if construction == "quintile":
                w = pd.Series(1 / len(idx), index=idx)
            else:
                w = inverse_vol_weights(g.loc[idx, "vol126"].fillna(0.3))
            gross = float((w * g.loc[idx, "fwd"].fillna(0)).sum())
            pw = prev_w.get(leg, pd.Series(dtype=float))
            allidx = w.index.union(pw.index)
            dw = (w.reindex(allidx).fillna(0) - pw.reindex(allidx).fillna(0)).abs()
            cost = float((dw * fr.reindex(allidx).fillna(FRICTION_BPS / 10_000)).sum())
            rec[f"{leg}_gross"] = gross
            rec[f"{leg}_net"] = gross - cost
            rec[f"{leg}_turnover"] = float(dw.sum() / 2)
            # deriva até o próximo rebalanceamento
            grown = w * (1 + g.loc[idx, "fwd"].fillna(0))
            prev_w[leg] = grown / grown.sum() if grown.sum() > 0 else w
        out.append(rec)
    df = pd.DataFrame(out).set_index("t")
    if construction == "quintile":
        df["LS_gross"] = df["Q5_gross"] - df["Q1_gross"]
        df["LS_net"] = df["Q5_net"] - df["Q1_net"]
    return df
