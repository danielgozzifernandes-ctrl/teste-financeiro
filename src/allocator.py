"""
Asset Allocator — a camada "investidor absoluto".

Decide o split de capital entre 4 sleeves ANTES do stock-picking:

    equities_br  → carteira Top-5 B3 (o sistema existente)
    cdi          → Tesouro Selic / CDB 100% CDI (risk-free a ~15% a.a.)
    global_usd   → IVVB11 (S&P 500 sem hedge — protege contra risco-Brasil)
    inflation    → IMAB11 (NTN-B — carrega juro real)

Sinais simples, tirados da literatura e não calibrados nos nossos dados:

Com ALLOCATION_MODE = "static" (padrão) o split é o mix fixo e os sinais
abaixo só entram no relatório. No modo "dynamic":

    1. Regime HMM (bull/bear/range) — já detectado pelo pipeline semanal.
       Define a alocação-base (ALLOCATION_BASE).
    2. ERP implícito = earnings yield da carteira − Selic. Se a bolsa não
       paga prêmio sobre o risk-free, não há razão para overweight.
    3. Time-series momentum 12-1 do IBOV em excesso do CDI
       (Moskowitz-Ooi-Pedersen 2012). Sinal binário: tilt ±ALLOCATION_TILT_PP.

Integração com vol-targeting: o gross_exposure (≤1.0) do vol-target escala o
sleeve de bolsa; o capital liberado vai para o CDI — caixa nunca fica "no ar".

A saída é serializável (vai para o recommendation JSON) e inclui rationale
humano-legível para o relatório Telegram.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.config import (
    ALLOCATION_BASE,
    ALLOCATION_INSTRUMENTS,
    ALLOCATION_MODE,
    ALLOCATION_STATIC_MIX,
    ALLOCATION_TILT_PP,
    EQUITIES_SLEEVE_MAX,
    EQUITIES_SLEEVE_MIN,
    ERP_HIGH_THRESHOLD,
    ERP_LOW_THRESHOLD,
    TSMOM_SKIP_DAYS,
    TSMOM_WINDOW_DAYS,
)

logger = logging.getLogger(__name__)


def compute_allocation(
    regime: str,
    portfolio_earnings_yield: Optional[float],
    selic_annual: Optional[float],
    ibov_prices: Optional[pd.Series],
    cdi_daily_returns: Optional[pd.Series],
    gross_exposure: float = 1.0,
    mode: Optional[str] = None,
) -> dict[str, Any]:
    """
    Calcula a alocação entre sleeves.

    Args:
        regime:                    "risk_on" | "mean_rev" | "bear"
        portfolio_earnings_yield:  EY médio da carteira top-5 (decimal, ex. 0.12)
        selic_annual:              Selic anualizada (decimal, ex. 0.15)
        ibov_prices:               série de preços do IBOV (>= ~273 obs p/ TSMOM)
        cdi_daily_returns:         série de retornos diários do CDI
        gross_exposure:            saída do vol-targeting (0.5–1.0); escala o
                                   sleeve de bolsa, liberando caixa para o CDI.

    Returns:
        dict serializável:
        {
          "sleeves":     {sleeve: peso},          # soma 1.0
          "signals":     {regime, erp, erp_tilt, tsmom, tsmom_tilt, ...},
          "instruments": {sleeve: descrição},
          "rationale":   [linhas humano-legíveis],
        }
        Sinais indisponíveis (dado faltante) entram com tilt 0 e são
        aparecem no rationale.
    """
    base = ALLOCATION_BASE.get(regime)
    rationale: list[str] = []
    if base is None:
        logger.warning("Regime '%s' desconhecido — base mean_rev", regime)
        base = ALLOCATION_BASE["mean_rev"]
        rationale.append(f"Regime '{regime}' desconhecido → base neutra (mean_rev)")
    else:
        rationale.append(
            f"Regime {regime}: base bolsa {base['equities_br']:.0%}"
        )

    # Sinal 1: ERP implícito
    erp: Optional[float] = None
    erp_tilt = 0.0
    if portfolio_earnings_yield is not None and selic_annual is not None:
        erp = float(portfolio_earnings_yield) - float(selic_annual)
        if erp < ERP_LOW_THRESHOLD:
            erp_tilt = -ALLOCATION_TILT_PP
            rationale.append(
                f"ERP implícito {erp:+.1%} < {ERP_LOW_THRESHOLD:.0%}: "
                f"bolsa não paga o risco vs Selic → −{ALLOCATION_TILT_PP:.0%} bolsa"
            )
        elif erp > ERP_HIGH_THRESHOLD:
            erp_tilt = +ALLOCATION_TILT_PP
            rationale.append(
                f"ERP implícito {erp:+.1%} > {ERP_HIGH_THRESHOLD:.0%}: "
                f"prêmio gordo → +{ALLOCATION_TILT_PP:.0%} bolsa"
            )
        else:
            rationale.append(f"ERP implícito {erp:+.1%}: neutro")
    else:
        rationale.append("ERP indisponível (EY ou Selic faltando) → tilt 0")

    # Sinal 2: TS momentum 12-1 do IBOV vs CDI
    tsmom: Optional[float] = None
    tsmom_tilt = 0.0
    ibov_excess = _tsmom_excess_return(ibov_prices, cdi_daily_returns)
    if ibov_excess is not None:
        tsmom = ibov_excess
        tsmom_tilt = ALLOCATION_TILT_PP if tsmom > 0 else -ALLOCATION_TILT_PP
        rationale.append(
            f"TSMOM 12-1 IBOV vs CDI: {tsmom:+.1%} → "
            f"{'+' if tsmom_tilt > 0 else '−'}{ALLOCATION_TILT_PP:.0%} bolsa"
        )
    else:
        rationale.append("TSMOM indisponível (histórico IBOV/CDI curto) → tilt 0")

    mode = mode or ALLOCATION_MODE
    if mode == "static":
        sleeves = dict(ALLOCATION_STATIC_MIX)
        rationale.append(
            "Mix estático 40/30/15/15: regime, ERP e TSMOM acima são "
            "informativos (não venceram o mix fixo no backtest)"
        )
        logger.info("Allocation [static]: %s", sleeves)
        return {
            "sleeves": sleeves,
            "signals": {
                "mode":         "static",
                "regime":       regime,
                "erp":          round(erp, 4) if erp is not None else None,
                "erp_tilt":     0.0,
                "tsmom":        round(tsmom, 4) if tsmom is not None else None,
                "tsmom_tilt":   0.0,
                "portfolio_ey": (round(float(portfolio_earnings_yield), 4)
                                 if portfolio_earnings_yield is not None else None),
                "selic_annual": (round(float(selic_annual), 4)
                                 if selic_annual is not None else None),
                "gross_exposure": 1.0,
            },
            "instruments": dict(ALLOCATION_INSTRUMENTS),
            "rationale":   rationale,
        }

    # Compor: tilts movem bolsa ↔ CDI, bounds duros
    equities = float(np.clip(
        base["equities_br"] + erp_tilt + tsmom_tilt,
        EQUITIES_SLEEVE_MIN, EQUITIES_SLEEVE_MAX,
    ))
    delta = equities - base["equities_br"]
    cdi_sleeve = max(base["cdi"] - delta, 0.0)

    sleeves = {
        "equities_br": equities,
        "cdi":         cdi_sleeve,
        "global_usd":  base["global_usd"],
        "inflation":   base["inflation"],
    }

    # Vol-targeting: gross < 1 reduz bolsa, caixa vai pro CDI
    gross = float(np.clip(gross_exposure, 0.0, 1.0))
    if gross < 1.0:
        freed = sleeves["equities_br"] * (1.0 - gross)
        sleeves["equities_br"] *= gross
        sleeves["cdi"] += freed
        rationale.append(
            f"Vol-targeting: exposure {gross:.0%} → "
            f"{freed:.0%} do capital migra de bolsa para CDI"
        )

    # Normalizar (bounds podem ter quebrado a soma)
    total = sum(sleeves.values())
    sleeves = {k: round(v / total, 4) for k, v in sleeves.items()}

    logger.info(
        "Allocation [%s]: bolsa %.0f%% | CDI %.0f%% | global %.0f%% | inflação %.0f%%",
        regime, sleeves["equities_br"] * 100, sleeves["cdi"] * 100,
        sleeves["global_usd"] * 100, sleeves["inflation"] * 100,
    )

    return {
        "sleeves":     sleeves,
        "signals": {
            "mode":        "dynamic",
            "regime":      regime,
            "erp":         round(erp, 4) if erp is not None else None,
            "erp_tilt":    round(erp_tilt, 4),
            "tsmom":       round(tsmom, 4) if tsmom is not None else None,
            "tsmom_tilt":  round(tsmom_tilt, 4),
            "portfolio_ey": (round(float(portfolio_earnings_yield), 4)
                             if portfolio_earnings_yield is not None else None),
            "selic_annual": (round(float(selic_annual), 4)
                             if selic_annual is not None else None),
            "gross_exposure": round(gross, 4),
        },
        "instruments": dict(ALLOCATION_INSTRUMENTS),
        "rationale":   rationale,
    }


def apply_gross_exposure(allocation: dict[str, Any], gross_exposure: float) -> dict[str, Any]:
    """
    Aplica o gross do vol-targeting sobre uma allocation já computada:
    bolsa × gross; capital liberado migra para o sleeve CDI.

    Idempotente quando gross >= 1.0 (retorna a allocation intacta). Usado
    por save_recommendation, que conhece o gross só depois do vol-target.
    """
    gross = float(np.clip(gross_exposure, 0.0, 1.0))
    if gross >= 1.0 or not allocation or "sleeves" not in allocation:
        return allocation
    if (allocation.get("signals") or {}).get("mode") == "static":
        # O mix fixo foi testado sem overlay de vol: o vol target só reduzia
        # drawdown, com Sharpe menor (0,39 x 0,43 com a mesma exposição média).
        out = dict(allocation)
        out["rationale"] = list(allocation.get("rationale", [])) + [
            f"Vol-targeting sugeria exposure {gross:.0%} — não aplicado no mix estático"
        ]
        return out

    sleeves = dict(allocation["sleeves"])
    freed = sleeves.get("equities_br", 0.0) * (1.0 - gross)
    sleeves["equities_br"] = sleeves.get("equities_br", 0.0) * gross
    sleeves["cdi"] = sleeves.get("cdi", 0.0) + freed

    total = sum(sleeves.values()) or 1.0
    sleeves = {k: round(v / total, 4) for k, v in sleeves.items()}

    out = dict(allocation)
    out["sleeves"] = sleeves
    out["signals"] = {**allocation.get("signals", {}), "gross_exposure": round(gross, 4)}
    out["rationale"] = list(allocation.get("rationale", [])) + [
        f"Vol-targeting: exposure {gross:.0%} → "
        f"{freed:.0%} do capital migra de bolsa para CDI"
    ]
    return out


def _tsmom_excess_return(
    ibov_prices: Optional[pd.Series],
    cdi_daily_returns: Optional[pd.Series],
) -> Optional[float]:
    """
    Retorno 12-1 do IBOV menos CDI acumulado na mesma janela.

    Janela: t-TSMOM_WINDOW_DAYS → t-TSMOM_SKIP_DAYS (consistente com o
    momentum de ações pós-rodada-5). None se histórico insuficiente.
    """
    if ibov_prices is None:
        return None
    clean = ibov_prices.dropna()
    min_obs = TSMOM_SKIP_DAYS + 30  # precisa de algo além do skip
    if len(clean) < min_obs:
        return None

    window = min(TSMOM_WINDOW_DAYS, len(clean) - 1)
    if window <= TSMOM_SKIP_DAYS:
        return None
    ibov_ret = float(clean.iloc[-1 - TSMOM_SKIP_DAYS] / clean.iloc[-window] - 1.0)

    cdi_ret = 0.0
    if cdi_daily_returns is not None:
        cdi_clean = cdi_daily_returns.dropna()
        if len(cdi_clean) > 0:
            # mesma janela em nº de observações (aproximação de calendário)
            n = min(window - TSMOM_SKIP_DAYS, len(cdi_clean))
            cdi_ret = float((1.0 + cdi_clean.tail(n)).prod() - 1.0)

    return ibov_ret - cdi_ret


def selic_annual_from_daily(cdi_or_selic_daily: Optional[pd.Series]) -> Optional[float]:
    """
    Anualiza a última taxa diária válida: (1+r)^252 − 1.

    Usa a ÚLTIMA observação (não média) — Selic é taxa administrada, o nível
    corrente é o que importa para custo de oportunidade.
    """
    if cdi_or_selic_daily is None:
        return None
    clean = cdi_or_selic_daily.dropna()
    if clean.empty:
        return None
    last = float(clean.iloc[-1])
    if last <= -1.0:
        return None
    return float((1.0 + last) ** 252 - 1.0)


def portfolio_earnings_yield(df_scored: pd.DataFrame, n: int = 5) -> Optional[float]:
    """
    EY médio simples do top-N (sinal de valuation agregado da carteira).

    Média simples, não ponderada por peso: o sinal pergunta "a cesta que o
    sistema quer comprar está cara ou barata?" — ponderar por peso HRP
    misturaria sinal de valuation com decisão de risco.
    """
    if df_scored is None or df_scored.empty or "earnings_yield" not in df_scored.columns:
        return None
    top = df_scored.head(n)["earnings_yield"].dropna()
    # Exigir maioria com EY válido — 1 EY de 5 não representa a cesta
    if len(top) < max(2, n // 2):
        return None
    return float(top.mean())
