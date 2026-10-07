"""
Calendário de pregões da B3 e relógio de Brasília.

Feriados nacionais em que a B3 não abre, mais 24/12 e 31/12 (sem pregão).
Desde 2022 a B3 abre nos feriados municipal (25/1) e estadual (9/7) de São
Paulo; 20/11 virou feriado nacional em 2024 (Lei 14.759/2023).
Quarta de Cinzas é pregão (abre às 13h).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Optional

# Brasil sem horário de verão desde 2019 → offset fixo.
BRT = timezone(timedelta(hours=-3))


def now_brt() -> datetime:
    return datetime.now(BRT)


def today_brt() -> date:
    return now_brt().date()


def _easter(year: int) -> date:
    """Domingo de Páscoa (algoritmo de Meeus/Jones/Butcher, calendário gregoriano)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


@lru_cache(maxsize=64)
def holidays(year: int) -> dict[date, str]:
    """Dias úteis (seg–sex ou não) sem pregão na B3 no ano."""
    easter = _easter(year)
    out = {
        date(year, 1, 1):          "Confraternização Universal",
        easter - timedelta(48):    "Carnaval",
        easter - timedelta(47):    "Carnaval",
        easter - timedelta(2):     "Sexta-feira Santa",
        date(year, 4, 21):         "Tiradentes",
        date(year, 5, 1):          "Dia do Trabalho",
        easter + timedelta(60):    "Corpus Christi",
        date(year, 9, 7):          "Independência",
        date(year, 10, 12):        "Nossa Senhora Aparecida",
        date(year, 11, 2):         "Finados",
        date(year, 11, 15):        "Proclamação da República",
        date(year, 12, 24):        "Véspera de Natal (sem pregão)",
        date(year, 12, 25):        "Natal",
        date(year, 12, 31):        "Último dia do ano (sem pregão)",
    }
    if year >= 2024:
        out[date(year, 11, 20)] = "Consciência Negra"
    return out


def _as_date(d: date | datetime | str) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d)[:10])


def holiday_name(d: date | datetime | str) -> Optional[str]:
    d = _as_date(d)
    return holidays(d.year).get(d)


def is_trading_day(d: date | datetime | str) -> bool:
    d = _as_date(d)
    return d.weekday() < 5 and d not in holidays(d.year)


def previous_trading_day(d: date | datetime | str) -> date:
    """Último pregão estritamente anterior a `d`."""
    d = _as_date(d) - timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def last_trading_day(d: date | datetime | str) -> date:
    """Último pregão em ou antes de `d`."""
    d = _as_date(d)
    return d if is_trading_day(d) else previous_trading_day(d)


def trading_days(start: date | str, end: date | str) -> list[date]:
    """Pregões no intervalo fechado [start, end]."""
    s, e = _as_date(start), _as_date(end)
    out = []
    while s <= e:
        if is_trading_day(s):
            out.append(s)
        s += timedelta(days=1)
    return out
