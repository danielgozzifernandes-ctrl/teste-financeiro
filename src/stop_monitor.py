"""
Stop Monitor — torna os stops do TradeAdvisor EXECUTÁVEIS.

O trade_advisor calcula entry/alvo/stop por ticker, mas até esta versão
ninguém monitorava os níveis — um stop que ninguém verifica é decoração
que dá falsa sensação de controle de risco (PETR4 caiu −10% dentro da
carteira de 16/05/2026 sem nenhum alerta).

Este módulo roda no closing diário: compara o preço de fechamento contra os
níveis persistidos no recommendation JSON (`trade_advice`) e gera alertas:

    stop_hit     — fechou NO/ABAIXO do stop  → ação: SAIR na abertura
    stop_near    — a <2% do stop             → ação: atenção, não mediana
    target_hit   — fechou NO/ACIMA do alvo conservador → realizar/reavaliar

Sem side-effects: retorna uma lista de alertas; o report builder decide a
apresentação e o main_daily decide o envio.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Margem do alerta preventivo: preço a menos de 2% do stop.
STOP_NEAR_PCT = 0.02


def check_levels(
    recommendation: Optional[dict],
    ticker_prices: dict[str, float],
) -> list[dict[str, Any]]:
    """
    Compara preços de fechamento contra stops/alvos persistidos.

    Args:
        recommendation: recommendation JSON (precisa de `trade_advice`;
                        recomendações antigas sem o campo retornam []).
        ticker_prices:  {ticker: preço de fechamento de hoje}

    Returns:
        Lista de alertas ordenada por severidade (stop_hit primeiro):
        [{ticker, kind, price, level, distance_pct}]
    """
    if not recommendation:
        return []
    advice: dict = recommendation.get("trade_advice") or {}
    if not advice:
        logger.debug("Recomendação sem trade_advice — stop monitor inativo")
        return []

    alerts: list[dict[str, Any]] = []
    for ticker, levels in advice.items():
        price = ticker_prices.get(ticker)
        if price is None or not isinstance(levels, dict):
            continue
        price = float(price)
        if price <= 0:
            continue

        stop = _to_float(levels.get("stop"))
        target = _to_float(levels.get("target_conservative"))

        if stop is not None and stop > 0:
            dist = (price - stop) / price
            if price <= stop:
                alerts.append({
                    "ticker": ticker, "kind": "stop_hit",
                    "price": price, "level": stop,
                    "distance_pct": round(dist, 4),
                })
                continue  # stop_hit domina; não emitir near/target junto
            if dist < STOP_NEAR_PCT:
                alerts.append({
                    "ticker": ticker, "kind": "stop_near",
                    "price": price, "level": stop,
                    "distance_pct": round(dist, 4),
                })

        if target is not None and target > 0 and price >= target:
            alerts.append({
                "ticker": ticker, "kind": "target_hit",
                "price": price, "level": target,
                "distance_pct": round((price - target) / price, 4),
            })

    severity = {"stop_hit": 0, "stop_near": 1, "target_hit": 2}
    alerts.sort(key=lambda a: (severity[a["kind"]], a["ticker"]))
    if alerts:
        logger.info(
            "Stop monitor: %d alerta(s) — %s",
            len(alerts), [(a["ticker"], a["kind"]) for a in alerts],
        )
    return alerts


def _to_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return f if f == f else None  # NaN check
    except (TypeError, ValueError):
        return None
