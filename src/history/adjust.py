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

A lista de eventos em ações da B3 é incompleta (falta, por exemplo, a
bonificação de 10% do Itaú em 07/2015) e tem registros que não batem com o
preço (grupamento 100:1 do ITUB em 2011 sem salto). Por isso:
  - cada evento da B3 só é aceito se o preço do dia-ex confirmar o fator;
  - no primeiro pregão com marca ex de bonificação/grupamento/desdobramento
    no campo especificação do COTAHIST (EB, EG, EX e combinações) sem evento
    aceito por perto, o fator é inferido da razão de preço;
  - saltos grandes sem evento algum são testados contra razões comuns.
Tudo que foi inferido ou rejeitado vai para o log de revisão.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

from src.history.universe import share_type

logger = logging.getLogger(__name__)

COMMON_RATIOS = [2, 3, 4, 5, 8, 10, 20, 25, 50, 100, 1000,
                 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 8, 1 / 10, 1 / 20, 1 / 25,
                 1 / 50, 1 / 100, 1 / 1000, 1.25, 1.5, 0.8]
# bonificações típicas (10%, 5%...) e seus inversos
MARKER_RATIOS = sorted(set(COMMON_RATIOS + [1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.6,
                                            1 / 1.05, 1 / 1.1, 1 / 1.2]))
STOCK_MARKERS = set("BGX")
JUMP_LOW, JUMP_HIGH = 0.6, 1.6
EVENT_CONFIRM_TOL = 0.25  # |log(razão * fator)| para aceitar evento da B3
RATIO_TOL = 0.10          # |log(razão * fator)| aceito como desdobramento
EVENT_WINDOW = 3          # pregões em torno de um evento B3 já conhecido


def _marker(spec: str) -> str:
    for tok in str(spec).split()[1:]:
        if re.fullmatch(r"E[A-Z]{1,3}", tok):
            return tok
    return ""


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

        ce = cash[cash["sid"] == sid]
        if len(ce):
            ex = _ex_dates(dates, ce["last_date_prior"])
            for d, v in zip(ex, ce["value"]):
                if pd.notna(d):
                    div[d] += v

        close = g["close"]
        prev = close.shift(1)
        raw = (close + div) / prev

        def log_event(d, m, source, accepted):
            jumps.append({"sid": sid, "ticker": g.loc[d, "ticker"], "date": d,
                          "raw_ratio": round(float(raw[d]), 5) if pd.notna(raw[d]) else None,
                          "multiplier": m, "source": source, "accepted": accepted})

        se = stock[stock["sid"] == sid]
        if len(se):
            # A B3 costuma registrar uma troca como par de eventos no mesmo dia
            # (desdobramento 20x + grupamento 0,1 = 2x): valida o fator líquido.
            ex = _ex_dates(dates, se["last_date_prior"])
            net = (pd.DataFrame({"d": ex.values, "m": se["multiplier"].values})
                     .dropna().groupby("d")["m"].prod())
            for d, m in net.items():
                if pd.isna(raw[d]) or abs(m - 1) < 1e-9:
                    continue
                if abs(np.log(raw[d] * m)) <= EVENT_CONFIRM_TOL:
                    mult[d] *= m
                else:
                    log_event(d, m, "b3_rejected", False)

        def near_event():
            known = mult.ne(1.0)
            return known.rolling(2 * EVENT_WINDOW + 1, center=True,
                                 min_periods=1).max().astype(bool)

        marker = g["spec"].map(_marker)
        new_marker = (marker != marker.shift(1)) & marker.map(
            lambda m: bool(STOCK_MARKERS & set(m[1:])))
        near = near_event()
        for d in marker.index[new_marker & ~near]:
            obs = raw[d]
            if pd.isna(obs):
                continue
            best = min(MARKER_RATIOS, key=lambda k: abs(np.log(obs * k)))
            accepted = (abs(np.log(obs * best)) < abs(np.log(obs)) - 0.02
                        and abs(np.log(obs * best)) <= 0.05)
            if accepted:
                mult[d] *= best
            log_event(d, best, "marker_" + marker[d], accepted)

        near = near_event()
        for d in raw.index[((raw < JUMP_LOW) | (raw > JUMP_HIGH)) & ~near]:
            obs = raw[d]
            best = min(COMMON_RATIOS, key=lambda k: abs(np.log(obs * k)))
            accepted = abs(np.log(obs * best)) <= RATIO_TOL
            if accepted:
                mult[d] *= best
            log_event(d, best, "jump", accepted)

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


BIG_MOVE = 0.25
PATCH_DISAGREE = 0.15


def patch_big_moves(series: pd.DataFrame, reference: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Dias com |retorno total| > 25% conferidos contra uma segunda fonte
    (retorno diário ajustado do yfinance, por ticker). Se as fontes divergem
    mais de 15pp e a referência é a menor em módulo, vale a referência —
    típico de cisão (PCAR3/Assaí) ou desdobramento que a B3 não lista.
    Dias sem referência ficam como estão e entram no log.
    """
    s = series.copy()
    log = []
    for idx in s.index[s["ret_total"].abs() > BIG_MOVE]:
        t, d, mine = s.at[idx, "ticker"], s.at[idx, "date"], s.at[idx, "ret_total"]
        ref = reference.get(t)
        r = ref.get(d) if ref is not None else None
        action = "no_reference"
        if r is not None and pd.notna(r):
            if abs(mine - r) > PATCH_DISAGREE and abs(r) < abs(mine):
                s.at[idx, "ret_total"] = r
                s.at[idx, "ret_price"] = r
                action = "patched"
            else:
                action = "confirmed"
        log.append({"sid": s.at[idx, "sid"], "ticker": t, "date": d,
                    "ret_cotahist": round(float(mine), 5),
                    "ret_reference": None if r is None or pd.isna(r) else round(float(r), 5),
                    "action": action})
    for sid, g in s.groupby("sid"):
        last = g["close"].iloc[-1]
        pi = (1 + g["ret_price"].fillna(0)).cumprod()
        ti = (1 + g["ret_total"].fillna(0)).cumprod()
        s.loc[g.index, "adj_close"] = pi / pi.iloc[-1] * last
        s.loc[g.index, "tr_close"] = ti / ti.iloc[-1] * last
    return s, pd.DataFrame(log)
