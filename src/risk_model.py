"""
risk_model.py — Multi-factor risk model estilo Barra (simplificado)

Decompõe os retornos cross-sectionais em fatores observáveis:
  R_i = α + β_market · MKT + β_size · SMB + β_value · HML + β_mom · UMD
        + Σ β_sector_k · Sector_k + ε_i

Onde:
  MKT (Market):   retorno do IBOV
  SMB (Size):     small-minus-big, baseado em market cap
  HML (Value):    high-minus-low, baseado em EY ou P/VP
  UMD (Momentum): up-minus-down, baseado em alpha 6m
  Sector_k:       dummies por setor B3 (Petróleo, Materiais, Energia, etc.)

Output:
  - factor_returns: time series diária dos prêmios de cada fator
  - exposures: para cada ticker, beta a cada fator (coefientes da regressão)
  - specific_risk: vol dos resíduos por ticker (risco idiossincrático)

Uso primário:
  Detectar concentração de risco fatorial — top 5 carregado demais em
  um fator (ex.: betas altos em Size = bet em small caps). Compara com
  HRP que olha covariância empírica direta.

Limitações honestas (vs Barra real):
  - Fatores observáveis (não estimados via PCA); Barra usa estimação iterativa
  - Cross-section por dia em vez de painel completo
  - Sem shrinkage Bayesiano em exposições
  - Sem decomposição setor + style ortogonalizados
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskModelOutput:
    """Resultado completo do ajuste do risk model."""
    factor_returns:    pd.DataFrame   # data × fator
    exposures:         pd.DataFrame   # ticker × fator (betas)
    specific_risk:     pd.Series      # ticker → vol anualizada do resíduo
    factor_cov:        pd.DataFrame   # cov entre fatores (anualizada)
    r_squared:         pd.Series      # ticker → R² da regressão de exposição
    n_obs:             int


def build_risk_model(
    df_prices: pd.DataFrame,
    ibov_prices: pd.Series,
    df_fundamentals: pd.DataFrame,
    sector_col: str = "setor",
    lookback_days: int = 252,
) -> Optional[RiskModelOutput]:
    """
    Ajusta um risk model multi-fator sobre janela de `lookback_days`.

    Args:
        df_prices:        DataFrame wide (date × ticker) preços ajustados
        ibov_prices:      Série de preços IBOV alinhada
        df_fundamentals:  DataFrame indexado por ticker com {market_cap,
                          earnings_yield ou pvp, alpha_6m, setor}
        sector_col:       Nome da coluna de setor B3
        lookback_days:    Janela de retornos diários para estimar exposições

    Returns:
        RiskModelOutput ou None se dados insuficientes.

    Pegadinhas mitigadas:
      - Look-ahead: usa apenas janela passada, não vê retorno futuro
      - Multicolinearidade: setor dummies excluem 1 categoria (Petróleo
        como baseline) — evita matriz singular
      - Outliers: winsoriza retornos a ±3σ antes da regressão
    """
    if df_prices.empty or ibov_prices.empty:
        logger.debug("risk_model: dados de preço vazios")
        return None

    # ── 1. Retornos diários log
    stock_log_ret = np.log(df_prices / df_prices.shift(1)).dropna(how="all").tail(lookback_days)
    ibov_log_ret  = np.log(ibov_prices / ibov_prices.shift(1)).dropna().tail(lookback_days)

    common_idx = stock_log_ret.index.intersection(ibov_log_ret.index)
    if len(common_idx) < 30:
        logger.debug("risk_model: < 30 dias comuns (%d)", len(common_idx))
        return None

    stock_log_ret = stock_log_ret.loc[common_idx]
    ibov_log_ret  = ibov_log_ret.loc[common_idx]

    # Winsorize ±3σ
    sigma = stock_log_ret.std(skipna=True)
    upper = sigma * 3
    lower = -sigma * 3
    stock_log_ret = stock_log_ret.clip(lower=lower, upper=upper, axis=1)

    # ── 2. Construir fatores estilísticos por dia (Fama-French style)
    factors_daily = _build_style_factors(
        stock_log_ret, df_fundamentals, ibov_log_ret,
    )
    if factors_daily is None or factors_daily.empty:
        return None

    # ── 3. Setor dummies (baseline = primeiro setor encontrado)
    sector_factors = _build_sector_returns(
        stock_log_ret, df_fundamentals[sector_col],
    )

    # Concatenar todos os fatores
    factor_returns = pd.concat([factors_daily, sector_factors], axis=1).dropna()
    if len(factor_returns) < 30:
        return None

    # ── 4. Estimar exposições por ticker via OLS individual
    exposures, specific_risk, r_squared = _estimate_exposures(
        stock_log_ret, factor_returns,
    )

    # ── 5. Covariância dos fatores (anualizada)
    factor_cov = factor_returns.cov() * 252

    logger.info(
        "Risk model: %d fatores × %d tickers × %d dias | R² médio: %.2f",
        len(factor_returns.columns), len(exposures), len(factor_returns),
        float(r_squared.mean()) if not r_squared.empty else 0,
    )

    return RiskModelOutput(
        factor_returns=factor_returns,
        exposures=exposures,
        specific_risk=specific_risk,
        factor_cov=factor_cov,
        r_squared=r_squared,
        n_obs=len(factor_returns),
    )


def _build_style_factors(
    stock_log_ret: pd.DataFrame,
    df_fund: pd.DataFrame,
    ibov_log_ret: pd.Series,
) -> Optional[pd.DataFrame]:
    """
    Constrói fatores estilísticos diários via dollar-neutral portfolios.

    SMB: long bottom-30% market_cap, short top-30% (média setorialmente neutra)
    HML: long top-30% EY, short bottom-30%
    UMD: long top-30% alpha_6m, short bottom-30%

    Retorna DataFrame com colunas [MKT, SMB, HML, UMD].
    """
    factors_dict: dict[str, pd.Series] = {}

    # MKT: retorno do IBOV
    factors_dict["MKT"] = ibov_log_ret.copy()

    # Para SMB/HML/UMD: precisamos identificar long e short legs
    common_tickers = [t for t in df_fund.index if t in stock_log_ret.columns]

    # SMB via log_market_cap (já capturado no scoring; senão usar market_cap)
    if "market_cap" in df_fund.columns:
        mc = df_fund.loc[common_tickers, "market_cap"].dropna()
        if len(mc) >= 6:
            short_size, long_size = _terciles(mc, low_is_long=True)
            factors_dict["SMB"] = _dollar_neutral_return(stock_log_ret, long_size, short_size)

    # HML via earnings_yield (high EY = cheap = long)
    if "earnings_yield" in df_fund.columns:
        ey = df_fund.loc[common_tickers, "earnings_yield"].dropna()
        if len(ey) >= 6:
            short_v, long_v = _terciles(ey, low_is_long=False)
            factors_dict["HML"] = _dollar_neutral_return(stock_log_ret, long_v, short_v)

    # UMD via alpha_6m
    if "alpha_6m" in df_fund.columns:
        mom = df_fund.loc[common_tickers, "alpha_6m"].dropna()
        if len(mom) >= 6:
            short_m, long_m = _terciles(mom, low_is_long=False)
            factors_dict["UMD"] = _dollar_neutral_return(stock_log_ret, long_m, short_m)

    if not factors_dict:
        return None

    return pd.DataFrame(factors_dict).dropna(how="all")


def _terciles(series: pd.Series, low_is_long: bool) -> tuple[list[str], list[str]]:
    """Separa em tercil inferior / superior. Retorna (long_tickers, short_tickers)."""
    sorted_series = series.sort_values()
    n = len(sorted_series)
    cut = n // 3
    if cut == 0:
        return [], []
    bottom = sorted_series.head(cut).index.tolist()
    top = sorted_series.tail(cut).index.tolist()
    return (bottom, top) if low_is_long else (top, bottom)


def _dollar_neutral_return(
    stock_log_ret: pd.DataFrame,
    longs: list[str],
    shorts: list[str],
) -> pd.Series:
    """Retorno de uma carteira long-short dollar-neutral equal-weight."""
    long_legs = [t for t in longs if t in stock_log_ret.columns]
    short_legs = [t for t in shorts if t in stock_log_ret.columns]
    if not long_legs or not short_legs:
        return pd.Series(dtype=float)
    long_ret  = stock_log_ret[long_legs].mean(axis=1, skipna=True)
    short_ret = stock_log_ret[short_legs].mean(axis=1, skipna=True)
    return (long_ret - short_ret).dropna()


def _build_sector_returns(
    stock_log_ret: pd.DataFrame,
    sector_map: pd.Series,
) -> pd.DataFrame:
    """
    Retornos médios por setor B3 menos retorno médio do universo (sector spread).
    Baseline: primeiro setor (alfabético) é DROPADO para evitar singularidade.
    """
    universe_avg = stock_log_ret.mean(axis=1, skipna=True)

    sector_series: dict[str, pd.Series] = {}
    for sector in sector_map.dropna().unique():
        tickers = sector_map[sector_map == sector].index.tolist()
        avail = [t for t in tickers if t in stock_log_ret.columns]
        if len(avail) >= 2:
            sec_avg = stock_log_ret[avail].mean(axis=1, skipna=True)
            sector_series[f"SEC_{sector}"] = sec_avg - universe_avg

    if not sector_series:
        return pd.DataFrame()

    out = pd.DataFrame(sector_series)
    # Dropar baseline (alfabeticamente primeiro)
    if len(out.columns) > 1:
        baseline = sorted(out.columns)[0]
        out = out.drop(columns=[baseline])
    return out


def _estimate_exposures(
    stock_log_ret: pd.DataFrame,
    factor_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Para cada ticker, regredir retorno em todos os fatores via OLS.
    Retorna (exposures, specific_risk, r_squared).
    """
    common_idx = stock_log_ret.index.intersection(factor_returns.index)
    if len(common_idx) < 20:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)

    X = factor_returns.loc[common_idx].to_numpy()
    X_with_intercept = np.column_stack([np.ones(len(X)), X])

    exposures: dict[str, dict] = {}
    specific_risk: dict[str, float] = {}
    r_squared: dict[str, float] = {}

    for ticker in stock_log_ret.columns:
        y = stock_log_ret.loc[common_idx, ticker].dropna()
        if len(y) < 20:
            continue
        # Re-alinhar
        idx = y.index.intersection(common_idx)
        if len(idx) < 20:
            continue
        y_arr = y.loc[idx].to_numpy()
        X_arr = factor_returns.loc[idx].to_numpy()
        Xi = np.column_stack([np.ones(len(X_arr)), X_arr])

        try:
            beta, _, _, _ = np.linalg.lstsq(Xi, y_arr, rcond=None)
            residuals = y_arr - Xi @ beta

            exposures[ticker] = {
                "alpha": float(beta[0]),
                **{f: float(beta[i + 1]) for i, f in enumerate(factor_returns.columns)},
            }
            specific_risk[ticker] = float(np.std(residuals, ddof=1) * np.sqrt(252))
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_arr - y_arr.mean()) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            r_squared[ticker] = float(max(0.0, r2))
        except np.linalg.LinAlgError:
            continue

    return (
        pd.DataFrame.from_dict(exposures, orient="index"),
        pd.Series(specific_risk, name="specific_risk"),
        pd.Series(r_squared, name="r_squared"),
    )


def portfolio_risk_decomposition(
    portfolio_weights: dict[str, float],
    risk_model: RiskModelOutput,
) -> dict:
    """
    Decompõe o risco da carteira em contribuições fatoriais.

    Risco total² = β·Σ_f·β + Σ_i w_i² · σ²_specific_i

    Onde:
      β é o vetor de exposições do portfólio (média ponderada das exposições)
      Σ_f é a covariância dos fatores
      σ²_specific é o risco idiossincrático de cada ticker

    Returns dict com:
      total_vol:        vol anualizada estimada do portfólio
      factor_vol:       vol vinda de exposição a fatores
      specific_vol:     vol vinda do risco idiossincrático
      factor_contributions: contribuição de cada fator individual ao risco total
      effective_factor_exposures: β_portfolio por fator
    """
    if not portfolio_weights or risk_model is None:
        return {}

    valid_tickers = [t for t in portfolio_weights.keys() if t in risk_model.exposures.index]
    if not valid_tickers:
        return {}

    # Renormalizar pesos sobre os tickers que temos exposição
    weights = {t: portfolio_weights[t] for t in valid_tickers}
    total_w = sum(weights.values())
    if total_w == 0:
        return {}
    weights = {t: w / total_w for t, w in weights.items()}

    # Exposições do portfólio (média ponderada)
    factor_cols = [c for c in risk_model.exposures.columns if c != "alpha"]
    portfolio_exposures = pd.Series(0.0, index=factor_cols)
    for t, w in weights.items():
        portfolio_exposures += w * risk_model.exposures.loc[t, factor_cols]

    # Risco fatorial
    cov_f = risk_model.factor_cov.loc[factor_cols, factor_cols]
    factor_var = float(portfolio_exposures.values @ cov_f.values @ portfolio_exposures.values)
    factor_vol = float(np.sqrt(max(0.0, factor_var)))

    # Risco específico
    specific_var = 0.0
    for t, w in weights.items():
        if t in risk_model.specific_risk.index:
            specific_var += (w ** 2) * (risk_model.specific_risk[t] ** 2)
    specific_vol = float(np.sqrt(specific_var))

    total_vol = float(np.sqrt(factor_var + specific_var))

    # Contribuição de cada fator (β·cov_f·β decomposto)
    factor_contributions: dict[str, float] = {}
    for f in factor_cols:
        contrib = float(portfolio_exposures[f] * (cov_f.loc[f].values @ portfolio_exposures.values))
        factor_contributions[f] = contrib

    return {
        "total_vol":                    round(total_vol, 4),
        "factor_vol":                   round(factor_vol, 4),
        "specific_vol":                 round(specific_vol, 4),
        "factor_pct":                   round(factor_var / (factor_var + specific_var), 4) if (factor_var + specific_var) > 0 else 0,
        "specific_pct":                 round(specific_var / (factor_var + specific_var), 4) if (factor_var + specific_var) > 0 else 0,
        "effective_factor_exposures":   {f: round(float(portfolio_exposures[f]), 4) for f in factor_cols},
        "factor_contributions":         {f: round(v, 6) for f, v in factor_contributions.items()},
    }
