"""
tests/test_benchmark.py

Candle incompleto do yfinance e consistência IBOV x BOVA11.
"""

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.backtester import Backtester
from src.benchmark import (
    BRT,
    IBOV_ETF_TICKER,
    IBOV_ETF_TOLERANCE,
    BenchmarkManager,
    clean_close,
    is_intraday,
    period_return_from_close,
)

HISTORY_DIR = Path(__file__).resolve().parent.parent / "data" / "history"


def _bars(rows: dict[str, tuple]) -> pd.DataFrame:
    """rows: {date: (open, high, low, close, volume)}"""
    df = pd.DataFrame.from_dict(
        rows, orient="index", columns=["Open", "High", "Low", "Close", "Volume"],
    )
    df.index = pd.to_datetime(df.index)
    return df


AFTER_CLOSE = datetime(2026, 10, 5, 20, 0, tzinfo=BRT)
MID_SESSION = datetime(2026, 10, 5, 14, 0, tzinfo=BRT)


# ---------------------------------------------------------------------------
# clean_close
# ---------------------------------------------------------------------------

def test_trailing_zeroed_bar_is_kept_and_flagged():
    df = _bars({
        "2026-10-01": (186353, 187437, 184768, 187198, 11_848_800),
        "2026-10-02": (187437, 192119, 186144, 192115, 13_208_900),
        "2026-10-05": (0, 0, 0, 206911, 0),
    })
    close, meta = clean_close(df, now=AFTER_CLOSE)
    assert close.index[-1] == pd.Timestamp("2026-10-05")
    assert meta == {"last_bar": "2026-10-05", "intraday": True}


def test_interior_zeroed_bar_is_dropped():
    df = _bars({
        "2026-09-30": (183880, 187835, 183880, 186340, 11_768_700),
        "2026-10-01": (0, 0, 0, 999_999, 0),
        "2026-10-02": (187437, 192119, 186144, 192115, 13_208_900),
    })
    close, meta = clean_close(df, now=AFTER_CLOSE)
    assert list(close.index.strftime("%Y-%m-%d")) == ["2026-09-30", "2026-10-02"]
    assert meta["intraday"] is False
    assert meta["dropped_bars"] == ["2026-10-01"]


def test_todays_full_bar_is_intraday_while_session_is_open():
    df = _bars({
        "2026-10-02": (187437, 192119, 186144, 192115, 13_208_900),
        "2026-10-05": (192115, 207000, 192000, 206000, 9_000_000),
    })
    assert clean_close(df, now=MID_SESSION)[1]["intraday"] is True
    assert clean_close(df, now=AFTER_CLOSE)[1]["intraday"] is False


def test_is_intraday_window():
    assert is_intraday(datetime(2026, 10, 5, 9, 59, tzinfo=BRT)) is False
    assert is_intraday(datetime(2026, 10, 5, 10, 0, tzinfo=BRT)) is True
    assert is_intraday(datetime(2026, 10, 5, 18, 30, tzinfo=BRT)) is False
    assert is_intraday(datetime(2026, 10, 3, 14, 0, tzinfo=BRT)) is False  # sábado


def test_period_return_matches_get_period_return_window():
    close = pd.Series(
        [100.0, 102.0, 101.0, 105.0],
        index=pd.to_datetime(["2026-09-25", "2026-09-28", "2026-09-29", "2026-10-02"]),
    )
    # retornos com data >= 28/09 → base é o fechamento de 25/09
    r = period_return_from_close(close, date(2026, 9, 28), date(2026, 10, 2))
    assert r == pytest.approx(0.05)
    assert period_return_from_close(close, date(2026, 9, 1), date(2026, 9, 2)) is None


# ---------------------------------------------------------------------------
# check_against_etf
# ---------------------------------------------------------------------------

_ETF = pd.Series(
    [180.82, 180.18, 190.00],
    index=pd.to_datetime(["2026-09-25", "2026-09-28", "2026-10-02"]),
)


@pytest.fixture
def manager(monkeypatch, tmp_path):
    from src.data_collector import CacheManager

    monkeypatch.setattr(
        BenchmarkManager, "_download_close",
        staticmethod(lambda t, s, e: (_ETF[_ETF.index.date <= e],
                                      {"last_bar": "2026-10-02", "intraday": False})),
    )
    return BenchmarkManager(cache=CacheManager(cache_dir=tmp_path))


def test_etf_check_passes_for_consistent_return(manager):
    etf_ret = 190.00 / 180.82 - 1
    out = manager.check_against_etf("2026-09-28", "2026-10-02", etf_ret + 0.002)
    assert out["consistent"] is True
    assert out["gap_pp"] == pytest.approx(0.2, abs=1e-3)
    assert out["etf"] == IBOV_ETF_TICKER


def test_etf_check_flags_wrong_period(manager):
    # Retorno de um período errado (ex.: duas semanas somadas) não passa
    out = manager.check_against_etf("2026-09-28", "2026-10-02", 0.09)
    assert out["consistent"] is False
    assert abs(out["gap_pp"]) > IBOV_ETF_TOLERANCE * 100


def test_etf_check_handles_missing_ibov(manager):
    assert "error" in manager.check_against_etf("2026-09-28", "2026-10-02", None)


# ---------------------------------------------------------------------------
# Metadata no backtest
# ---------------------------------------------------------------------------

def test_backtest_result_carries_intraday_flag(tmp_path):
    rec = {
        "date": "2026-09-28", "mode": "weekly",
        "top5": [{"ticker": "PETR4", "entry_price": 48.84, "metrics": {}, "sector": "X"}],
        "entry_prices": {"PETR4": 48.84},
    }
    (tmp_path / "recommendations_2026-09-28_weekly.json").write_text(
        json.dumps(rec), encoding="utf-8",
    )
    from src.snapshot_manager import SnapshotManager

    bt = Backtester(snapshot_manager=SnapshotManager(history_dir=tmp_path), history_dir=tmp_path)
    md = {"prices_intraday": False, "ibovespa": {"last_bar": "2026-10-05", "intraday": True}}
    res = bt.run("weekly", {"PETR4": 56.35}, {"ibovespa": 0.14}, run_date="2026-10-05",
                 market_data=md)
    assert res["intraday"] is True
    assert res["market_data"]["ibovespa"]["last_bar"] == "2026-10-05"

    res = bt.run("weekly", {"PETR4": 56.35}, {"ibovespa": 0.14}, run_date="2026-10-05")
    assert res["intraday"] is False


# ---------------------------------------------------------------------------
# Histórico real: IBOV gravado nos backtests x BOVA11 (precisa de rede)
# ---------------------------------------------------------------------------

def _stored_ibov_windows() -> list[tuple[str, str, str, float]]:
    out = []
    for f in sorted(HISTORY_DIR.glob("backtest_*.json")):
        b = json.loads(f.read_text(encoding="utf-8"))
        ibov = (b.get("benchmark_returns") or {}).get("ibovespa")
        if b.get("status") in ("success", "partial_data") and ibov is not None:
            out.append((f.name, b["recommendation_date"], b["backtest_date"], ibov))
    return out


def test_stored_ibov_returns_agree_with_bova11():
    windows = _stored_ibov_windows()
    if not windows:
        pytest.skip("sem backtests em data/history")
    yf = pytest.importorskip("yfinance")
    start = min(w[1] for w in windows)
    try:
        raw = yf.download(IBOV_ETF_TICKER, start=str(pd.Timestamp(start) - pd.Timedelta(days=10))[:10],
                          progress=False, auto_adjust=True)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"yfinance indisponível: {exc}")
    if raw is None or raw.empty:
        pytest.skip("yfinance sem dados do BOVA11")
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    etf, _ = clean_close(raw)

    # Arquivos antigos não registram o horário dos dados: uma execução antes
    # da abertura termina no fechamento anterior, uma no pregão fica perto do
    # fechamento do dia. Aceita a janela que estiver mais perto.
    bad = []
    for name, s, e, ibov in windows:
        start, end = date.fromisoformat(s), date.fromisoformat(e)
        candidates = [
            r for r in (
                period_return_from_close(etf, start, end),
                period_return_from_close(etf, start, end - timedelta(days=1)),
            ) if r is not None
        ]
        if not candidates:
            continue
        etf_ret = min(candidates, key=lambda r: abs(ibov - r))
        if abs(ibov - etf_ret) > IBOV_ETF_TOLERANCE:
            bad.append(f"{name}: IBOV {ibov:+.4f} vs BOVA11 {etf_ret:+.4f}")
    assert not bad, "IBOV inconsistente com BOVA11:\n" + "\n".join(bad)
