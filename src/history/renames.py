"""
Ligação de tickers renomeados.

O ISIN da B3 embute o código do papel, então muda junto com o ticker
(ELET3 → AXIA3, EMBR3 → EMBJ3...). Considera renomeação quando um papel para
de negociar num pregão e outro do mesmo tipo começa no pregão seguinte com
abertura até 20% do último fechamento e volume da mesma ordem.
"""

from __future__ import annotations

import pandas as pd

from src.history.universe import equities_only, share_type

MAX_PRICE_GAP = 0.20
MAX_VOLUME_RATIO = 10.0


def detect_renames(df: pd.DataFrame, min_avg_volume: float = 1e6) -> pd.DataFrame:
    eq = equities_only(df).copy()
    eq["type"] = share_type(eq["spec"])
    sessions = pd.DatetimeIndex(sorted(df["date"].unique()))
    next_session = dict(zip(sessions[:-1], sessions[1:]))
    g = eq.sort_values("date").groupby("ticker")
    first = g.first()
    last = g.last()
    avg_vol = g["volume"].apply(lambda v: v.tail(20).mean())
    head_vol = g["volume"].apply(lambda v: v.head(20).mean())
    ended = last[last["date"] < sessions[-1]]
    rows = []
    for old, r in ended.iterrows():
        if avg_vol[old] < min_avg_volume:
            continue
        nxt = next_session.get(r["date"])
        cands = first[(first["date"] == nxt) & (first["type"] == r["type"])]
        for new, c in cands.iterrows():
            gap = c["open"] / r["close"] - 1
            vr = head_vol[new] / max(avg_vol[old], 1.0)
            if abs(gap) <= MAX_PRICE_GAP and 1 / MAX_VOLUME_RATIO <= vr <= MAX_VOLUME_RATIO:
                rows.append({"old": old, "new": new, "last_date": r["date"],
                             "first_date": c["date"], "price_gap": round(gap, 4),
                             "volume_ratio": round(vr, 2)})
    out = pd.DataFrame(rows, columns=["old", "new", "last_date", "first_date",
                                      "price_gap", "volume_ratio"])
    # um antigo com mais de um candidato é ambíguo: descarta. Um novo com mais
    # de um antigo é fusão (BRFS3 + MRFG3 → MBRF3): fica o de preço mais
    # próximo, o outro é tratado como encerrado.
    out = out[~out["old"].duplicated(keep=False)]
    out = out.assign(_gap=out["price_gap"].abs()).sort_values("_gap")
    return (out.drop_duplicates("new", keep="first").drop(columns="_gap")
               .sort_values("last_date").reset_index(drop=True))


def security_ids(df: pd.DataFrame, renames: pd.DataFrame) -> dict[str, str]:
    """ticker → id estável (o primeiro ticker da cadeia de renomeações)."""
    parent = dict(zip(renames["new"], renames["old"]))
    out = {}
    for t in df["ticker"].unique():
        root = t
        seen = set()
        while root in parent and root not in seen:
            seen.add(root)
            root = parent[root]
        out[t] = root
    return out
