"""
tests/test_pipeline.py

Smoke test de integração: scoring → diversificação → persistência → backtest.
Pega erro de orquestração que os testes unitários não veem (ex.: backtest
comparando a recomendação contra ela mesma). Sem rede: dados sintéticos e
history dir temporário.
"""

import numpy as np
import pandas as pd
import pytest

from src.scoring_engine import compute_scores, select_diverse_portfolio
from src.snapshot_manager import SnapshotManager
from src.backtester import Backtester, BacktestStatus


def _synthetic_universe(seed: int = 0, n_per_sector: int = 7):
    """Universo sintético líquido em 3 setores / 3 temas macro distintos."""
    np.random.seed(seed)
    sectors = ["Bancos", "Energia Elétrica", "Materiais Básicos"]
    tickers, setores, subsetores = [], [], []
    for s in sectors:
        for i in range(n_per_sector):
            tickers.append(f"{s[:3].upper()}{i}")
            setores.append(s)
            subsetores.append(f"{s}_{i}")  # subsetor único (não bloquear cap=1)
    n = len(tickers)
    df_fund = pd.DataFrame({
        "ticker": tickers, "nome": tickers, "setor": setores, "subsetor": subsetores,
        "pl": np.random.uniform(5, 15, n),
        "pvp": np.random.uniform(0.8, 2.5, n),
        "roe": np.random.uniform(0.08, 0.30, n),
        "roic": np.random.uniform(0.05, 0.20, n),
        "divida_ebitda": np.random.uniform(0.5, 3.0, n),
        "dividend_yield": np.random.uniform(0.02, 0.12, n),
        "beta": np.random.uniform(0.5, 1.5, n),
        "avg_volume_30d": np.random.uniform(2e7, 5e8, n),  # todos líquidos (R$)
        "current_price": np.random.uniform(10, 50, n),
        "market_cap": np.random.uniform(1e9, 1e11, n),
        "data_source": ["brapi+yfinance"] * n,
    })
    dates = pd.date_range("2025-01-01", periods=300, freq="B")
    prices = pd.DataFrame(
        {t: 20 * np.cumprod(1 + np.random.normal(0.0004, 0.02, 300)) for t in tickers},
        index=dates,
    )
    ibov = pd.Series(
        120000 * np.cumprod(1 + np.random.normal(0.0002, 0.015, 300)), index=dates
    )
    return df_fund, prices, ibov


def test_pipeline_scoring_to_backtest(tmp_path):
    df_fund, prices, ibov = _synthetic_universe()

    # 1. Scoring + diversificação ─────────────────────────────────────────
    scored = compute_scores(df_fund=df_fund, df_prices=prices, ibov_prices=ibov, regime="mean_rev")
    scored = select_diverse_portfolio(scored, n=5, max_per_sector=2)

    assert len(scored) >= 5
    assert "conviction" in scored.columns and "conviction_label" in scored.columns
    assert scored["conviction"].between(0, 1).all()
    # diversificação: top5 com no máximo 2 do mesmo setor
    top5_sectors = scored.head(5)["setor"].tolist()
    assert max(top5_sectors.count(s) for s in set(top5_sectors)) <= 2

    # data_quality é populado pelo main.py via attrs — simular para testar flow
    scored.attrs["data_quality"] = {
        "declared_universe": 21, "collected": 21, "scored": len(scored),
        "coverage_pct": 1.0,
    }

    # 2. Persistência da rec da semana 1 ──────────────────────────────────
    snap = SnapshotManager(history_dir=tmp_path)
    snap.save_price_snapshot(df_prices=prices, run_date="2026-01-05")
    snap.save_recommendation(df_scored=scored, df_prices=prices,
                             run_date="2026-01-05", mode="weekly")

    rec = snap.load_recommendation("2026-01-05", "weekly")
    assert rec is not None
    assert len(rec["top5"]) == 5
    assert all("conviction_label" in r for r in rec["top5"])   # schema convicção
    assert rec["entry_prices"]                                 # backtester depende disso
    assert rec["full_universe_scores"]                         # IC sem selection bias
    assert rec["execution_metadata"]["data_quality"] is not None  # cobertura persistida

    # 3. Semana 2: preços +3% → nova rec ─────────────────────────────────
    prices2 = prices * 1.03
    snap.save_price_snapshot(df_prices=prices2, run_date="2026-01-12")
    scored2 = compute_scores(df_fund=df_fund, df_prices=prices2, ibov_prices=ibov, regime="mean_rev")
    scored2 = select_diverse_portfolio(scored2, n=5, max_per_sector=2)
    snap.save_recommendation(df_scored=scored2, df_prices=prices2,
                             run_date="2026-01-12", mode="weekly")

    # 4. Backtest da semana 2 avalia a rec da SEMANA 1 (não a de hoje) ────
    current_prices = {t: float(prices2[t].iloc[-1]) for t in prices2.columns}
    bt = Backtester(snapshot_manager=snap, history_dir=tmp_path)
    result = bt.run(
        mode="weekly",
        current_prices=current_prices,
        benchmark_period_returns={"ibovespa": 0.01, "cdi": 0.002},
        run_date="2026-01-12",
    )
    assert result["status"] in (BacktestStatus.SUCCESS, BacktestStatus.PARTIAL_DATA)
    assert result["recommendation_date"] == "2026-01-05"   # rec anterior, NÃO a de hoje
    assert result["period_days"] == 7
    # Preços +3% → retorno positivo após fricção
    assert result["portfolio_return"] > 0.0


def test_sector_relative_leverage_filter():
    """Alavancagem alta porém normal no setor sobrevive; outlier absoluto morre."""
    from src.scoring_engine import ScoringEngine
    # Setor 'Locação' inteiro alavancado (~6x): nenhum deve ser excluído pelo
    # relativo. 'Varejo' com mediana baixa: o nome a 9x deve ser excluído.
    df = pd.DataFrame({
        "avg_volume_30d": [5e7] * 6,
        "pvp": [1.5] * 6, "roe": [0.15] * 6, "roic": [0.10] * 6,
        "divida_ebitda": [6.0, 6.2, 5.8,   1.0, 1.2, 9.0],
        "setor": ["Locação", "Locação", "Locação", "Varejo", "Varejo", "Varejo"],
        "data_source": ["brapi+yfinance"] * 6,
    }, index=["LOC1", "LOC2", "LOC3", "VAR1", "VAR2", "VAR3"])

    res = ScoringEngine._apply_hard_filters(df)

    # Setor de locação (mediana ~6x): todos sobrevivem apesar de > 5x flat
    assert "LOC1" in res.index and "LOC2" in res.index and "LOC3" in res.index
    # Varejo: VAR3 (9x, muito acima da mediana ~1.2x) é excluído
    assert "VAR3" not in res.index
    assert "VAR1" in res.index and "VAR2" in res.index


def test_absolute_leverage_ceiling():
    """Acima do teto absoluto (10x) exclui mesmo se for a norma do setor."""
    from src.scoring_engine import ScoringEngine
    df = pd.DataFrame({
        "avg_volume_30d": [5e7] * 3,
        "pvp": [1.5] * 3, "roe": [0.15] * 3, "roic": [0.10] * 3,
        "divida_ebitda": [12.0, 13.0, 11.0],   # setor todo > 10x → teto duro
        "setor": ["Alavancado"] * 3,
        "data_source": ["brapi+yfinance"] * 3,
    }, index=["A1", "A2", "A3"])

    res = ScoringEngine._apply_hard_filters(df)
    # Teto absoluto de 10x mata todos, independentemente da mediana setorial
    assert len(res) == 0
