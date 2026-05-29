"""
tests/test_backtester.py

Testes do bug crítico corrigido: o backtester NÃO pode comparar uma
recomendação contra os preços do próprio dia em que foi gerada.

Cobertura:
  1. load_latest_recommendation(before_date) ignora recs na mesma data/futuras
  2. Backtester.run() usa a recomendação ANTERIOR real (period_days > 0)
  3. Primeira execução (sem rec estritamente anterior) → NO_HISTORY
  4. compute_track_record exclui backtests degenerados (period_days == 0)
  5. Gating estatístico: factor_analysis e walk_forward marcam amostra
     insuficiente em vez de exibir métricas como se fossem sinal.
"""

import json
from pathlib import Path

import pytest

from src.backtester import Backtester, BacktestStatus
from src.snapshot_manager import SnapshotManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_rec(history_dir: Path, date_str: str, entry: dict, mode: str = "weekly") -> Path:
    """Escreve um JSON de recomendação mínimo porém válido para o backtester."""
    tickers = list(entry.keys())
    rec = {
        "date": date_str,
        "mode": mode,
        "top5": [
            {"ticker": t, "entry_price": entry[t], "metrics": {}, "sector": "Setor"}
            for t in tickers
        ],
        "entry_prices": dict(entry),
        "portfolio_weights": {t: 1.0 / len(tickers) for t in tickers},
    }
    path = history_dir / f"recommendations_{date_str}_{mode}.json"
    path.write_text(json.dumps(rec), encoding="utf-8")
    return path


def _write_backtest(history_dir: Path, date_str: str, period_days: int,
                    port_ret: float, alpha: float, mode: str = "weekly") -> None:
    path = history_dir / f"backtest_{date_str}_{mode}.json"
    path.write_text(json.dumps({
        "status": "success",
        "period_days": period_days,
        "portfolio_return": port_ret,
        "benchmark_returns": {"ibovespa": port_ret - alpha, "cdi": 0.002},
        "alpha_vs_ibov": alpha,
    }), encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. load_latest_recommendation(before_date)
# ---------------------------------------------------------------------------

def test_load_latest_before_date(tmp_path):
    snap = SnapshotManager(history_dir=tmp_path)
    _write_rec(tmp_path, "2026-01-05", {"PETR4": 40.0})
    _write_rec(tmp_path, "2026-01-12", {"VALE3": 60.0})

    # Sem before_date → mais recente
    assert snap.load_latest_recommendation("weekly")["date"] == "2026-01-12"

    # before_date estritamente anterior → pula a rec da mesma data
    prev = snap.load_latest_recommendation("weekly", before_date="2026-01-12")
    assert prev["date"] == "2026-01-05"

    # Nada antes da mais antiga
    assert snap.load_latest_recommendation("weekly", before_date="2026-01-05") is None


# ---------------------------------------------------------------------------
# 2. Backtester usa a recomendação anterior real — NÃO a do próprio dia
# ---------------------------------------------------------------------------

def test_backtest_uses_previous_not_self(tmp_path):
    snap = SnapshotManager(history_dir=tmp_path)
    # Recomendação anterior (a que deve ser avaliada)
    _write_rec(tmp_path, "2026-01-05", {"PETR4": 40.0})
    # Recomendação "de hoje" (recém-salva no mesmo run) — NÃO deve ser usada
    _write_rec(tmp_path, "2026-01-12", {"PETR4": 44.0})

    bt = Backtester(snapshot_manager=snap, history_dir=tmp_path)
    result = bt.run(
        mode="weekly",
        current_prices={"PETR4": 44.0},
        benchmark_period_returns={"ibovespa": 0.05, "cdi": 0.01},
        run_date="2026-01-12",
    )

    assert result["status"] == BacktestStatus.SUCCESS
    # O ponto central do bug: deve avaliar a rec de 05/01, não a de 12/01
    assert result["recommendation_date"] == "2026-01-05"
    assert result["period_days"] == 7
    # PETR4 40 → 44 (~+10% antes da fricção): retorno claramente positivo
    assert result["portfolio_return"] > 0.05


# ---------------------------------------------------------------------------
# 3. Sem recomendação estritamente anterior → NO_HISTORY (não backtesta a si)
# ---------------------------------------------------------------------------

def test_backtest_first_run_has_no_prior(tmp_path):
    snap = SnapshotManager(history_dir=tmp_path)
    _write_rec(tmp_path, "2026-01-12", {"PETR4": 44.0})  # só a de hoje

    bt = Backtester(snapshot_manager=snap, history_dir=tmp_path)
    result = bt.run(
        mode="weekly",
        current_prices={"PETR4": 44.0},
        benchmark_period_returns={},
        run_date="2026-01-12",
    )
    assert result["status"] == BacktestStatus.NO_HISTORY


# ---------------------------------------------------------------------------
# 4. compute_track_record exclui backtests degenerados (period_days == 0)
# ---------------------------------------------------------------------------

def test_track_record_excludes_zero_period(tmp_path):
    bt = Backtester(history_dir=tmp_path)
    # Degenerado: mesmo dia, retorno = só fricção
    _write_backtest(tmp_path, "2026-01-05", period_days=0, port_ret=-0.0026, alpha=-0.0026)
    # Real: 7 dias, alpha positivo
    _write_backtest(tmp_path, "2026-01-12", period_days=7, port_ret=0.03, alpha=0.02)

    tr = bt.compute_track_record("weekly")
    assert tr["n_periods"] == 1                       # degenerado excluído
    assert tr["avg_alpha_ibov"] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# 5. Gating estatístico — amostra insuficiente é sinalizada, não mascarada
# ---------------------------------------------------------------------------

def test_factor_analysis_flags_insufficient_data(tmp_path):
    from src.factor_analysis import analyze_factors
    result = analyze_factors(
        history_dir=tmp_path,                 # vazio → insuficiente
        output_path=tmp_path / "factor_ic.json",
        verbose=False,
    )
    ds = result.get("data_sufficiency")
    assert ds is not None
    assert ds["is_significant"] is False
    assert ds["warning"]                      # mensagem presente e não vazia


def test_walk_forward_flags_insufficient_data(tmp_path):
    from src.walk_forward import walk_forward_backtest
    result = walk_forward_backtest(
        history_dir=tmp_path,                 # vazio → insuficiente
        output_path=tmp_path / "walk_forward.json",
        verbose=False,
    )
    ds = result.get("data_sufficiency")
    assert ds is not None
    assert ds["is_significant"] is False
    assert ds["warning"]
