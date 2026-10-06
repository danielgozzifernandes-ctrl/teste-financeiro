"""
Universo líquido sem viés de sobrevivência, no espírito do IBrX-100.

Aproximação da metodologia da B3 (carteira quadrimestral, jan/mai/set):
  - janela: 12 meses anteriores à data de rebalanceamento;
  - elegível: ação ou unit (exclui BDR, fundos, índices), presença em
    >= 95% dos pregões da janela e último preço >= R$ 1;
  - ranking pelo índice de negociabilidade IN = sqrt(n_i/N * v_i/V), com
    n = nº de negócios e v = volume financeiro na janela;
  - os 100 primeiros entram.
Diferenças conhecidas: a B3 calcula o IN pregão a pregão e aplica limites por
emissor e por free float; aqui é sobre totais da janela e sem free float.
"""

from __future__ import annotations

import pandas as pd

EQUITY_TYPES = {"ON", "PN", "PNA", "PNB", "PNC", "PND", "PNE", "PNF", "UNT"}
REBALANCE_MONTHS = (1, 5, 9)
LOOKBACK_DAYS = 365
MIN_PRESENCE = 0.95
MIN_PRICE = 1.0
SIZE = 100


def share_type(spec: pd.Series) -> pd.Series:
    return spec.str.split().str[0].str.rstrip("*")


def equities_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[share_type(df["spec"]).isin(EQUITY_TYPES)]


def rebalance_dates(sessions: pd.DatetimeIndex, start_year: int, end_year: int) -> list[pd.Timestamp]:
    out = []
    for y in range(start_year, end_year + 1):
        for m in REBALANCE_MONTHS:
            month = sessions[(sessions.year == y) & (sessions.month == m)]
            if len(month):
                out.append(month[0])
    return out


def liquid_universe(df: pd.DataFrame, asof: pd.Timestamp, size: int = SIZE,
                    min_presence: float = MIN_PRESENCE) -> pd.DataFrame:
    """Membros na data `asof` usando só pregões estritamente anteriores."""
    win = df[(df["date"] < asof) & (df["date"] >= asof - pd.Timedelta(days=LOOKBACK_DAYS))]
    win = equities_only(win)
    n_sessions = win["date"].nunique()
    if n_sessions == 0:
        return pd.DataFrame(columns=["ticker", "isin", "in_index", "presence", "rank"])
    key = "sid" if "sid" in win.columns else "ticker"
    g = win.sort_values("date").groupby(key)
    stats = pd.DataFrame({
        "ticker": g["ticker"].last(),
        "isin": g["isin"].last(),
        "trades": g["trades"].sum(),
        "volume": g["volume"].sum(),
        "sessions": g["date"].nunique(),
        "last_close": g["close"].last(),
        "last_date": g["date"].max(),
    })
    stats["presence"] = stats["sessions"] / n_sessions
    last_session = win["date"].max()
    stats = stats[
        (stats["presence"] >= min_presence)
        & (stats["last_close"] >= MIN_PRICE)
        & (stats["last_date"] == last_session)
    ]
    if stats.empty:
        return pd.DataFrame(columns=["ticker", "isin", "in_index", "presence", "rank"])
    stats["in_index"] = ((stats["trades"] / stats["trades"].sum())
                         * (stats["volume"] / stats["volume"].sum())) ** 0.5
    stats = stats.sort_values("in_index", ascending=False).head(size)
    stats["rank"] = range(1, len(stats) + 1)
    stats.index.name = "sid"
    stats = stats.reset_index()
    return stats[["sid", "ticker", "isin", "in_index", "presence", "rank",
                  "volume", "last_close"]]


def universe_history(df: pd.DataFrame, start_year: int = 2011,
                     end_year: int = 2026) -> pd.DataFrame:
    sessions = pd.DatetimeIndex(sorted(df["date"].unique()))
    frames = []
    for d in rebalance_dates(sessions, start_year, end_year):
        u = liquid_universe(df, d)
        u.insert(0, "rebalance_date", d)
        frames.append(u)
    return pd.concat(frames, ignore_index=True)
