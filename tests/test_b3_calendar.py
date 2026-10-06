"""
tests/test_b3_calendar.py

Calendário B3, datas em horário de Brasília e entrypoints em dia sem pregão.
"""

from datetime import date, datetime

import pytest

import main_daily
import main_scanner
from src import b3_calendar as cal
from src.b3_calendar import (
    BRT,
    is_trading_day,
    last_trading_day,
    previous_trading_day,
    trading_days,
)
from src.benchmark import is_intraday

# Feriados em dia útil, conferidos contra o calendário publicado pela B3.
WEEKDAY_HOLIDAYS = {
    2024: ["01-01", "02-12", "02-13", "03-29", "05-01", "05-30",
           "11-15", "11-20", "12-24", "12-25", "12-31"],
    2025: ["01-01", "03-03", "03-04", "04-18", "04-21", "05-01",
           "06-19", "11-20", "12-24", "12-25", "12-31"],
    2026: ["01-01", "02-16", "02-17", "04-03", "04-21", "05-01", "06-04",
           "09-07", "10-12", "11-02", "11-20", "12-24", "12-25", "12-31"],
    2027: ["01-01", "02-08", "02-09", "03-26", "04-21", "05-27", "09-07",
           "10-12", "11-02", "11-15", "12-24", "12-31"],
}


@pytest.mark.parametrize("year", sorted(WEEKDAY_HOLIDAYS))
def test_weekday_holidays_match_b3(year):
    got = sorted(
        d.strftime("%m-%d") for d in cal.holidays(year) if d.weekday() < 5
    )
    assert got == WEEKDAY_HOLIDAYS[year]


def test_easter_dates():
    assert cal._easter(2024) == date(2024, 3, 31)
    assert cal._easter(2025) == date(2025, 4, 20)
    assert cal._easter(2026) == date(2026, 4, 5)
    assert cal._easter(2027) == date(2027, 3, 28)


def test_days_b3_opens_despite_local_holidays():
    assert is_trading_day("2026-02-18")       # quarta de cinzas
    assert is_trading_day("2027-01-25")       # aniversário de SP
    assert is_trading_day("2026-07-09")       # data magna de SP
    assert is_trading_day("2023-11-20")       # antes de virar feriado nacional
    assert not is_trading_day("2024-11-20")


def test_navigation():
    assert previous_trading_day("2026-10-13") == date(2026, 10, 9)
    assert last_trading_day("2026-10-12") == date(2026, 10, 9)
    assert last_trading_day("2026-10-13") == date(2026, 10, 13)
    assert trading_days("2026-10-09", "2026-10-14") == [
        date(2026, 10, 9), date(2026, 10, 13), date(2026, 10, 14),
    ]


def test_intraday_is_false_on_holiday():
    assert is_intraday(datetime(2026, 10, 12, 14, 0, tzinfo=BRT)) is False
    assert is_intraday(datetime(2026, 10, 13, 14, 0, tzinfo=BRT)) is True


def test_run_date_uses_brasilia_clock(monkeypatch):
    # 22:30 BRT de 05/10 já é 06/10 em UTC; a data do run tem de ser 05/10.
    monkeypatch.setattr(main_daily, "today_brt", lambda: date(2026, 10, 5))
    monkeypatch.setattr(main_scanner, "today_brt", lambda: date(2026, 10, 5))
    assert main_daily._resolve_date(None) == "2026-10-05"
    assert main_scanner._resolve_date(None) == "2026-10-05"


def test_snapshot_default_date_uses_brasilia_clock(monkeypatch):
    from src import snapshot_manager

    monkeypatch.setattr(snapshot_manager, "today_brt", lambda: date(2026, 10, 5))
    assert snapshot_manager._date_str(None) == "2026-10-05"


@pytest.mark.parametrize("mode", ["morning", "closing"])
def test_daily_skips_holiday(mode, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("não deveria rodar em feriado")

    monkeypatch.setattr(main_daily, "run_morning", boom)
    monkeypatch.setattr(main_daily, "run_closing", boom)
    assert main_daily.main(["--mode", mode, "--date", "2026-10-12", "--dry-run"]) == 0


def test_daily_force_runs_on_holiday(monkeypatch):
    called = []
    monkeypatch.setattr(main_daily, "run_closing", lambda *a: called.append(a) or 0)
    main_daily.main(["--mode", "closing", "--date", "2026-10-12", "--dry-run", "--force"])
    assert called


def test_scanner_skips_holiday(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("não deveria coletar em feriado")

    monkeypatch.setattr(main_scanner, "load_data", boom)
    assert main_scanner.main(["--date", "2026-10-12", "--dry-run"]) == 0
