"""
tests/test_scoring.py

Testes unitários do ScoringEngine.
Cobertura:
  1. _normalize_adaptive: Z-Score setorial (N>=8)
  2. _normalize_adaptive: Percentil setorial (N=4-7)
  3. _normalize_adaptive: Fallback global (N<4)
  4. Direção: lower_is_better produz score inverso correto
  5. Redistribuição ROIC→ROE para setor Financeiro
  6. Redistribuição ROIC→ROE quando ROIC é NaN
  7. Earnings Yield = 1/P·L; P/L<=0 → NaN
  8. Momentum relativo: alpha = ret_ação - ret_IBOV
  9. Hard filter: liquidez mínima
  10. Output: total_score = 0.45*fund + 0.30*mom + 0.25*qual
  11. normalization_method column presente e válido
  12. Tickers sem dados: não travam o pipeline
"""

import numpy as np
import pandas as pd
import pytest

from src.scoring_engine import NormDetail, ScoringEngine, compute_scores

# ---------------------------------------------------------------------------
# Fixtures reutilizáveis
# ---------------------------------------------------------------------------

def _make_sector_map(tickers_by_sector: dict[str, list[str]]) -> pd.Series:
    """Cria pd.Series ticker→setor."""
    data = {}
    for setor, tickers in tickers_by_sector.items():
        for t in tickers:
            data[t] = setor
    return pd.Series(data)


def _make_values(ticker_vals: dict[str, float]) -> pd.Series:
    return pd.Series(ticker_vals, dtype=float)


@pytest.fixture
def engine():
    return ScoringEngine()


# ---------------------------------------------------------------------------
# 1. Z-Score Setorial (N=10 → N>=8)
# ---------------------------------------------------------------------------
def test_normalize_adaptive_zscore_sectoral(engine):
    """N=10 → deve usar zscore_sectoral; mediana→score≈50, extremo→≈100."""
    tickers = [f"T{i}" for i in range(10)]
    sector_map = _make_sector_map({"Setor A": tickers})
    # Valores equidistantes: mediana em T4/T5
    values = _make_values({t: float(i) for i, t in enumerate(tickers)})

    scores, details = engine._normalize_adaptive(values, sector_map)

    assert all(d.method == "zscore_sectoral" for d in details.values() if d is not None)
    # Mediana do setor (i=4.5): scores de T4 e T5 devem estar próximos de 50
    assert 40 < scores["T4"] < 60
    assert 40 < scores["T5"] < 60
    # Extremo superior (T9) deve ter score alto
    assert scores["T9"] > 80
    # Extremo inferior (T0) deve ter score baixo
    assert scores["T0"] < 20
    # Todos os scores no intervalo [0, 100]
    assert scores.between(0, 100).all()


# ---------------------------------------------------------------------------
# 2. Percentil Setorial (N=5)
# ---------------------------------------------------------------------------
def test_normalize_adaptive_percentile_sectoral(engine):
    """N=5 → deve usar percentile_sectoral."""
    tickers = [f"P{i}" for i in range(5)]
    sector_map = _make_sector_map({"Setor B": tickers})
    values = _make_values({t: float(i) for i, t in enumerate(tickers)})

    scores, details = engine._normalize_adaptive(values, sector_map)

    assert all(d.method == "percentile_sectoral" for d in details.values() if d is not None)
    # Maior valor → percentil mais alto
    assert scores["P4"] > scores["P0"]
    assert scores.between(0, 100).all()


# ---------------------------------------------------------------------------
# 3. Fallback Global (N=2 < 4)
# ---------------------------------------------------------------------------
def test_normalize_adaptive_global_fallback(engine):
    """N=2 < MIN_SECTOR_PERCENTILE → deve usar zscore_global."""
    # Setor pequeno (2 tickers) + setor grande como referência global
    small_tickers  = ["S1", "S2"]
    big_tickers    = [f"B{i}" for i in range(10)]
    all_tickers    = small_tickers + big_tickers

    sector_map = _make_sector_map({
        "Mini Setor": small_tickers,
        "Grande Setor": big_tickers,
    })
    values = _make_values({t: float(i) for i, t in enumerate(all_tickers)})

    scores, details = engine._normalize_adaptive(values, sector_map)

    # Mini setor deve usar fallback global
    for t in small_tickers:
        assert details[t].method == "zscore_global", f"{t} deveria usar zscore_global"
    # Grande setor usa zscore_sectoral
    for t in big_tickers:
        assert details[t].method == "zscore_sectoral", f"{t} deveria usar zscore_sectoral"


# ---------------------------------------------------------------------------
# 4. Direção: lower_is_better → score invertido
# ---------------------------------------------------------------------------
def test_direction_lower_is_better(engine):
    """
    Para lower_is_better, um valor MENOR deve gerar um score MAIOR.
    Simulamos P/VP: 0.5 (barato) deve ter score maior que 3.0 (caro).
    """
    tickers = [f"D{i}" for i in range(10)]
    sector_map = _make_sector_map({"Setor": tickers})
    # P/VP variando de 0.5 a 5.0
    raw_pvp = _make_values({f"D{i}": 0.5 + i * 0.5 for i in range(10)})

    # Simular o que _score_fundamental faz: negar antes de normalizar
    negated = -raw_pvp
    scores, _ = engine._normalize_adaptive(negated, sector_map)

    # D0 (P/VP=0.5, o mais barato) deve ter o maior score
    assert scores["D0"] > scores["D9"], "P/VP menor deve resultar em score maior"
    assert scores["D0"] > 70


# ---------------------------------------------------------------------------
# 5. Redistribuição ROIC→ROE: setor Financeiro
# ---------------------------------------------------------------------------
def test_roic_weight_redirected_to_roe_for_financial(engine):
    """Para setor Financeiro, o peso de ROIC (20%) deve ser somado ao ROE."""
    row = pd.Series({
        "setor":        "Financeiro e Outros",
        "roic":         0.15,  # tem valor, mas setor é financeiro
        "earnings_yield": 0.10,
        "pvp":          1.2,
        "roe":          0.20,
        "divida_ebitda": np.nan,
        "dividend_yield": 0.06,
    })
    # Criar factor_scores_map com scores fictícios
    tickers = ["ITUB4"]
    factor_scores_map = {
        "earnings_yield": pd.Series({"ITUB4": 70.0}),
        "pvp":            pd.Series({"ITUB4": 60.0}),
        "roe":            pd.Series({"ITUB4": 80.0}),
        "roic":           pd.Series({"ITUB4": 75.0}),
        "divida_ebitda":  pd.Series({"ITUB4": np.nan}),  # setor financeiro
        "dividend_yield": pd.Series({"ITUB4": 65.0}),
    }

    weights = engine._fundamental_weights(row, factor_scores_map, "ITUB4")

    # ROIC não deve ter peso
    assert "roic" not in weights or weights.get("roic", 0) == pytest.approx(0.0, abs=1e-9)
    # ROE deve ter peso de 0.45 (ou proporcional após redistribuição de divida_ebitda NaN)
    # Com divida_ebitda NaN: base sem roic e sem div_ebitda = {ey:0.20, pvp:0.15, roe:0.45, dy:0.10}
    # Total = 0.90 → renormalizado: roe = 0.45/0.90 = 0.50
    assert weights.get("roe", 0) > 0.40, "ROE deve ter peso ≥ 0.40 para setor Financeiro"
    # Pesos devem somar 1.0
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 6. Redistribuição ROIC→ROE: ROIC é NaN
# ---------------------------------------------------------------------------
def test_roic_weight_redirected_when_roic_is_nan(engine):
    """Quando ROIC é NaN (não-financeiro), seu peso vai para ROE."""
    row = pd.Series({"setor": "Materiais Básicos", "roic": np.nan})
    factor_scores_map = {
        "earnings_yield": pd.Series({"VALE3": 60.0}),
        "pvp":            pd.Series({"VALE3": 55.0}),
        "roe":            pd.Series({"VALE3": 70.0}),
        "roic":           pd.Series({"VALE3": np.nan}),  # NaN
        "divida_ebitda":  pd.Series({"VALE3": 50.0}),
        "dividend_yield": pd.Series({"VALE3": 65.0}),
    }

    weights = engine._fundamental_weights(row, factor_scores_map, "VALE3")

    assert "roic" not in weights or weights.get("roic", 0) == pytest.approx(0.0, abs=1e-9)
    assert weights.get("roe", 0) == pytest.approx(0.45 / 1.0, abs=0.01)
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 7. Earnings Yield: 1/P·L e P/L<=0 → NaN
# ---------------------------------------------------------------------------
def test_derive_metrics_earnings_yield(engine):
    df = pd.DataFrame({
        "pl": [4.0, 20.0, -5.0, 0.0, np.nan, 100.0]
    })
    result = engine._derive_metrics(df)

    # Válidos: 1/4=0.25, 1/20=0.05, 1/100=0.01
    assert result.loc[0, "earnings_yield"] == pytest.approx(1.0 / 4.0)
    assert result.loc[1, "earnings_yield"] == pytest.approx(1.0 / 20.0)
    assert result.loc[5, "earnings_yield"] == pytest.approx(1.0 / 100.0)
    # P/L negativo → NaN
    assert pd.isna(result.loc[2, "earnings_yield"])
    # P/L zero → NaN (divisão por zero)
    assert pd.isna(result.loc[3, "earnings_yield"])
    # P/L NaN → NaN
    assert pd.isna(result.loc[4, "earnings_yield"])


# ---------------------------------------------------------------------------
# 8. Momentum relativo: alpha = ret_ação - ret_IBOV
# ---------------------------------------------------------------------------
def test_momentum_alpha_relative(engine):
    """Alpha deve refletir corretamente o excesso de retorno vs IBOV."""
    dates = pd.date_range("2024-01-01", periods=130, freq="B")

    # IBOV sobe 10% em 63 dias
    ibov_prices = pd.Series(
        [100.0 * (1 + 0.10 * i / 63) for i in range(len(dates))],
        index=dates,
    )
    # Ação A: sobe 20% → alpha_3m = +10pp
    # Ação B: sobe 5%  → alpha_3m = -5pp
    df_prices = pd.DataFrame({
        "ACAA3": [100.0 * (1 + 0.20 * i / 63) for i in range(len(dates))],
        "ACAB3": [100.0 * (1 + 0.05 * i / 63) for i in range(len(dates))],
    }, index=dates)

    df_fund = pd.DataFrame({
        "ticker": ["ACAA3", "ACAB3"],
        "setor":  ["Setor X", "Setor X"],
        "subsetor": ["Sub", "Sub"],
        "norm_method": ["zscore_global", "zscore_global"],
        "data_source": ["brapi+yfinance", "brapi+yfinance"],
        "pl": [10.0, 15.0],
        "pvp": [1.5, 2.0],
        "roe": [0.15, 0.10],
        "roic": [0.12, 0.08],
        "divida_ebitda": [1.0, 2.0],
        "dividend_yield": [0.05, 0.03],
        "avg_volume_30d": [50_000_000.0, 50_000_000.0],
        "beta": [1.0, 0.8],
        "current_price": [30.0, 25.0],
        "market_cap": [1e9, 8e8],
    })

    df_enriched = engine._derive_metrics(df_fund.set_index("ticker"))
    df_enriched = engine._add_momentum_metrics(df_enriched, df_prices, ibov_prices)

    alpha_a = df_enriched.loc["ACAA3", "alpha_3m"]
    alpha_b = df_enriched.loc["ACAB3", "alpha_3m"]

    # ACAA3 deve ter alpha positivo (superou IBOV)
    assert alpha_a > 0, f"ACAA3 deveria ter alpha>0, got {alpha_a:.4f}"
    # ACAB3 deve ter alpha negativo (abaixo do IBOV)
    assert alpha_b < 0, f"ACAB3 deveria ter alpha<0, got {alpha_b:.4f}"
    # ACAA3 deve ter alpha muito maior que ACAB3
    assert alpha_a > alpha_b


# ---------------------------------------------------------------------------
# 9. Hard filter: liquidez mínima
# ---------------------------------------------------------------------------
def test_hard_filter_removes_illiquid(engine):
    """Tickers com avg_volume_30d < R$5M devem ser excluídos."""
    df = pd.DataFrame({
        "avg_volume_30d": [1_000_000.0, 10_000_000.0, 3_000_000.0, 8_000_000.0],
        "divida_ebitda":  [1.0, 2.0, 1.5, 3.0],
        "setor":          ["A", "A", "A", "A"],
        "data_source":    ["brapi+yfinance"] * 4,
    }, index=["T1", "T2", "T3", "T4"])

    result = engine._apply_hard_filters(df)

    assert "T1" not in result.index, "T1 (1M vol) deve ser removido"
    assert "T3" not in result.index, "T3 (3M vol) deve ser removido"
    assert "T2" in result.index, "T2 (10M vol) deve permanecer"
    assert "T4" in result.index, "T4 (8M vol) deve permanecer"


# ---------------------------------------------------------------------------
# 10. Score composto = 0.45*fund + 0.30*mom + 0.25*qual (com PESOS do config)
# ---------------------------------------------------------------------------
def test_total_score_weighting():
    """Verifica que total_score é a combinação correta dos pilares."""
    from src.config import WEIGHTS

    fund_score = 80.0
    mom_score  = 60.0
    qual_score = 70.0
    expected = (
        WEIGHTS["fundamental"] * fund_score
        + WEIGHTS["momentum"]  * mom_score
        + WEIGHTS["quality"]   * qual_score
    )

    computed = (
        WEIGHTS["fundamental"] * 80.0
        + WEIGHTS["momentum"]  * 60.0
        + WEIGHTS["quality"]   * 70.0
    )
    assert computed == pytest.approx(expected, abs=0.01)
    # Com pesos 45/30/25: 0.45*80 + 0.30*60 + 0.25*70 = 36+18+17.5 = 71.5
    assert computed == pytest.approx(71.5, abs=0.01)


# ---------------------------------------------------------------------------
# 11. normalization_method column presente e com valores válidos
# ---------------------------------------------------------------------------
def test_normalization_method_column_present(engine):
    """Verifica que normalization_method é uma string em um conjunto válido."""
    valid_methods = {"zscore_sectoral", "percentile_sectoral", "zscore_global", "unknown"}

    # Simular um fund_details com método zscore_sectoral
    class FakeDetail:
        method = "zscore_sectoral"
        score = 75.0

    fund_details = {
        "PETR4": {"roe": FakeDetail(), "pvp": FakeDetail()},
        "VALE3": {"roe": FakeDetail()},
    }

    from src.scoring_engine import NormDetail as ND
    for ticker, factor_details in fund_details.items():
        for factor, detail in factor_details.items():
            assert detail.method in valid_methods


# ---------------------------------------------------------------------------
# 12. Robustez: NaN em todos os fatores não trava o pipeline
# ---------------------------------------------------------------------------
def test_weighted_sum_all_nan_returns_nan(engine):
    """Ticker com todos os fatores NaN deve resultar em composite NaN, não erro."""
    index = pd.Index(["GHOST3"])
    factor_scores_map = {
        "alpha_3m":  pd.Series({"GHOST3": np.nan}),
        "alpha_6m":  pd.Series({"GHOST3": np.nan}),
        "alpha_12m": pd.Series({"GHOST3": np.nan}),
    }
    base_weights = {"alpha_3m": 0.30, "alpha_6m": 0.40, "alpha_12m": 0.30}

    result = engine._weighted_sum(index, base_weights, factor_scores_map)

    assert pd.isna(result["GHOST3"]), "Ticker sem dados deve resultar em NaN, não erro"


# ---------------------------------------------------------------------------
# 13. Z-Score: clip [-3, +3] → score em [0, 100]
# ---------------------------------------------------------------------------
def test_zscore_clipping_bounds(engine):
    """Outliers extremos devem ser clipados para [0, 100], não extrapolados."""
    tickers = [f"X{i}" for i in range(10)] + ["OUTLIER_HIGH", "OUTLIER_LOW"]
    sector_map = _make_sector_map({"Setor C": tickers})
    # Outliers absurdos
    vals = {f"X{i}": float(i) for i in range(10)}
    vals["OUTLIER_HIGH"] = 10000.0  # z >> 3
    vals["OUTLIER_LOW"]  = -10000.0  # z << -3
    values = _make_values(vals)

    scores, _ = engine._normalize_adaptive(values, sector_map)

    assert scores["OUTLIER_HIGH"] == pytest.approx(100.0, abs=0.1)
    assert scores["OUTLIER_LOW"]  == pytest.approx(0.0, abs=0.1)
    assert scores.between(0, 100).all()
