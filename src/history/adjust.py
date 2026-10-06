"""
Séries ajustadas por papel (id estável através de renomeações).

Retorno diário entre dois pregões consecutivos do papel:

    r_preço = close_t * m_t / close_{t-1} - 1
    r_total = (close_t * m_t + D_t) / close_{t-1} - 1

m_t = ações novas por ação antiga nos eventos com data-ex em t (desdobramento,
grupamento, bonificação); D_t = proventos brutos por ação antiga com data-ex
em t (dividendo + JCP; JCP bruto, antes dos 15% de IR). Data-ex = primeiro
pregão do papel depois da data-com informada pela B3.

`adj_close` e `tr_close` são reescalados para terminar no último fechamento
bruto (mesma convenção do auto_adjust do yfinance).

Saltos de preço sem evento da B3 por perto são testados contra razões de
desdobramento comuns; os aceitos e os suspeitos vão para um log de revisão.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.history.universe import share_type

logger = logging.getLogger(__name__)

COMMON_RATIOS = [2, 3, 4, 5, 8, 10, 20, 25, 50, 100, 1000,
                 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 8, 1 / 10, 1 / 20, 1 / 25,
                 1 / 50, 1 / 100, 1 / 1000, 1.25, 1.5, 0.8]
JUMP_LOW, JUMP_HIGH = 0.45, 2.2
RATIO_TOL = 0.06          # |log(ratio_obs * mult)| aceito como desdobramento
EVENT_WINDOW = 3          # pregões em torno de um evento B3 já conhecido


def _ex_dates(dates: pd.DatetimeIndex, last_prior: pd.Series) -> pd.Series:
    """Primeiro pregão estritamente depois da data-com."""
    pos = dates.searchsorted(last_prior.values, side="right")
    ok = pos < len(dates)
    out = pd.Series(pd.NaT, index=last_prior.index)
    out[ok] = dates[pos[ok]]
    return out


def build_series(px: pd.DataFrame, sid_of: dict[str, str],
                 cash: pd.DataFrame, stock: pd.DataFrame,
                 sids: Optional[set[str]] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    px: linhas do COTAHIST (date, ticker, isin, spec, close, volume, trades, quantity).
    cash: root, type, last_date_prior, value.   stock: isin, last_date_prior, multiplier.
    Retorna (séries diárias, log de saltos).
    """
    px = px.copy()
    px["sid"] = px["ticker"].map(sid_of)
    if sids is not None:
        px = px[px["sid"].isin(sids)]
    px["type"] = share_type(px["spec"])
    px = px.sort_values(["sid", "date"])

    isin_to_sid = px.drop_duplicates("isin").set_index("isin")["sid"].to_dict()
    roots_types = (px.assign(root=px["ticker"].str[:4])
                     .drop_duplicates(["root", "type", "sid"])[["root", "type", "sid"]])

    cash = cash.merge(roots_types, on=["root", "type"], how="inner")
    cash = cash.drop_duplicates(["sid", "last_date_prior", "value", "label"])
    stock = stock.assign(sid=stock["isin"].map(isin_to_sid)).dropna(subset=["sid"])
    stock = stock.drop_duplicates(["sid", "last_date_prior", "multiplier"])

    frames, jumps = [], []
    for sid, g in px.groupby("sid", sort=False):
        g = g.drop_duplicates("date", keep="last").set_index("date")
        dates = g.index
        mult = pd.Series(1.0, index=dates)
        div = pd.Series(0.0, index=dates)

        se = stock[stock["sid"] == sid]
        if len(se):
            ex = _ex_dates(dates, se["last_date_prior"])
            for d, m in zip(ex, se["multiplier"]):
                if pd.notna(d):
                    mult[d] *= m
        ce = cash[cash["sid"] == sid]
        if len(ce):
            ex = _ex_dates(dates, ce["last_date_prior"])
            for d, v in zip(ex, ce["value"]):
                if pd.notna(d):
                    div[d] += v

        close = g["close"]
        prev = close.shift(1)
        raw = close / prev
        known = mult.ne(1.0)
        near_known = known.rolling(2 * EVENT_WINDOW + 1, center=True, min_periods=1).max().astype(bool)
        for d in raw.index[((raw < JUMP_LOW) | (raw > JUMP_HIGH)) & ~near_known]:
            obs = raw[d]
            best = min(COMMON_RATIOS, key=lambda k: abs(np.log(obs * k)))
            accepted = abs(np.log(obs * best)) <= RATIO_TOL
            if accepted:
                mult[d] *= best
            jumps.append({"sid": sid, "ticker": g.loc[d, "ticker"], "date": d,
                          "raw_ratio": round(float(obs), 5), "multiplier": best,
                          "accepted": accepted})

        r_price = close * mult / prev - 1
        r_total = (close * mult + div) / prev - 1
        out = pd.DataFrame({
            "sid": sid, "ticker": g["ticker"], "close": close,
            "volume": g["volume"], "trades": g["trades"],
            "split_mult": mult, "dividend": div,
            "ret_price": r_price, "ret_total": r_total,
        })
        pi = (1 + r_price.fillna(0)).cumprod()
        ti = (1 + r_total.fillna(0)).cumprod()
        out["adj_close"] = pi / pi.iloc[-1] * close.iloc[-1]
        out["tr_close"] = ti / ti.iloc[-1] * close.iloc[-1]
        frames.append(out.reset_index())

    series = pd.concat(frames, ignore_index=True)
    return series, pd.DataFrame(jumps)
