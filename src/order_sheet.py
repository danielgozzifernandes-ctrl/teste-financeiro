"""
Order Sheet — converte pesos-alvo em ordens executáveis para um capital real.

Pesos percentuais são abstração; investidor PF executa em quantidades e
reais. Dado `--capital R$X`, este módulo gera:

  1. Valor-alvo por sleeve (bolsa / CDI / IVVB11 / IMAB11)
  2. Quantidades por ticker do top-5 no mercado FRACIONÁRIO (qty inteira,
     sem lote-padrão de 100 — PF pequeno deve usar o fracionário e pagar
     o spread, que é menor que o erro de arredondar para lotes de 100)
  3. Rotação vs carteira anterior (entradas/saídas)
  4. Nota fiscal honesta: NÃO conhecemos o preço médio de compra do usuário,
     então não dá para calcular o IR exato — informamos o valor estimado de
     vendas do rebalanceamento e a regra (isenção swing-trade até R$20k de
     VENDAS/mês; acima, 15% sobre o GANHO, DARF até último dia útil do mês
     seguinte). Estimar imposto sem a base de custo seria número inventado.

Sem side-effects: retorna dict serializável; report_builder formata.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Isenção de IR para vendas em swing-trade no mês (PF, ações à vista).
IR_EXEMPTION_MONTHLY_SALES_BRL = 20_000.0
IR_SWING_RATE = 0.15


def build_order_sheet(
    capital_brl: float,
    allocation: Optional[dict],
    portfolio_weights: dict[str, float],
    ticker_prices: dict[str, float],
    previous_tickers: Optional[list[str]] = None,
) -> Optional[dict[str, Any]]:
    """
    Monta a folha de ordens para um capital em R$.

    Args:
        capital_brl:       capital total a alocar (R$)
        allocation:        saída do allocator ({"sleeves": {...}}) — se None,
                           assume 100% no sleeve de bolsa (modo legado)
        portfolio_weights: pesos do top-5 dentro do sleeve de bolsa
        ticker_prices:     {ticker: preço atual} para converter R$ → qty
        previous_tickers:  top-5 da recomendação anterior (para rotação)

    Returns:
        dict serializável ou None se capital inválido.
    """
    if not capital_brl or capital_brl <= 0:
        return None

    sleeves = (allocation or {}).get("sleeves") or {"equities_br": 1.0}
    sleeve_values = {
        s: round(capital_brl * w, 2) for s, w in sleeves.items() if w > 0
    }

    # ── Ordens de bolsa: ticker → qty no fracionário ──────────────────────
    equity_value = sleeve_values.get("equities_br", 0.0)
    orders: list[dict[str, Any]] = []
    unallocated = 0.0
    for ticker, w in sorted(portfolio_weights.items(), key=lambda kv: -kv[1]):
        target = equity_value * w
        price = ticker_prices.get(ticker)
        if price is None or price <= 0:
            orders.append({
                "ticker": ticker, "target_brl": round(target, 2),
                "qty": None, "note": "sem preço — calcular manualmente",
            })
            unallocated += target
            continue
        qty = math.floor(target / price)
        executed = qty * price
        unallocated += target - executed
        orders.append({
            "ticker":     ticker,
            "weight":     round(w, 4),
            "price":      round(price, 2),
            "qty":        qty,
            "value_brl":  round(executed, 2),
        })

    # Sobra de arredondamento vai para o sleeve CDI (nunca fica "no ar")
    if unallocated > 0 and "cdi" in sleeve_values:
        sleeve_values["cdi"] = round(sleeve_values["cdi"] + unallocated, 2)

    # ── Rotação vs carteira anterior ─────────────────────────────────────
    current = [o["ticker"] for o in orders]
    rotation = None
    estimated_sales = 0.0
    if previous_tickers:
        exits = [t for t in previous_tickers if t not in current]
        entries = [t for t in current if t not in previous_tickers]
        if exits or entries:
            # Valor estimado de venda: posição equal-share do capital de bolsa
            # anterior (aproximação declarada — não conhecemos o capital nem
            # os preços da época de compra do usuário).
            per_position = equity_value / max(len(previous_tickers), 1)
            estimated_sales = per_position * len(exits)
            rotation = {"exits": exits, "entries": entries}

    # ── Nota fiscal (honesta: sem base de custo, sem número inventado) ───
    tax_note = _tax_note(estimated_sales)

    return {
        "capital_brl":     round(capital_brl, 2),
        "sleeve_values":   sleeve_values,
        "equity_orders":   orders,
        "rotation":        rotation,
        "estimated_sales_brl": round(estimated_sales, 2) if estimated_sales else 0.0,
        "tax_note":        tax_note,
        "cash_residual_brl": round(unallocated, 2),
    }


def _tax_note(estimated_sales_brl: float) -> str:
    if estimated_sales_brl <= 0:
        return "Sem vendas estimadas neste rebalanceamento — sem evento de IR."
    if estimated_sales_brl < IR_EXEMPTION_MONTHLY_SALES_BRL:
        return (
            f"Vendas estimadas ≈ R$ {estimated_sales_brl:,.0f}: abaixo da "
            f"isenção de R$ {IR_EXEMPTION_MONTHLY_SALES_BRL:,.0f}/mês "
            "(swing-trade PF) — sem IR se não houver outras vendas no mês."
        )
    return (
        f"Vendas estimadas ≈ R$ {estimated_sales_brl:,.0f}: ACIMA da isenção "
        f"de R$ {IR_EXEMPTION_MONTHLY_SALES_BRL:,.0f}/mês — "
        f"{IR_SWING_RATE:.0%} de IR sobre o GANHO (depende do seu preço "
        "médio; DARF até o último dia útil do mês seguinte)."
    )
