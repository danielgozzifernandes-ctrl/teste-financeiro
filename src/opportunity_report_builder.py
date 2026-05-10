"""
opportunity_report_builder.py

Builds the special opportunity alert message for Telegram (MarkdownV2).

Format per opportunity:
  🚨 ALERTA DE OPORTUNIDADE — TICKER

  💰 Preço atual: R$ XX.XX
  🎯 Entrada sugerida: R$ XX.XX – R$ XX.XX
  🏹 Alvo 1: R$ XX.XX (MA50) · +X.X%
  🏹 Alvo 2: R$ XX.XX (Máx 52s) · +X.X%
  🛑 Stop: R$ XX.XX · -7.0%
  ⏱ Horizonte: 2–6 semanas
  ⚠️ Risco: Moderado

  Por que agora:
  • ROE 28.0% — 2.1σ acima do setor
  • RSI 34 — zona sobrevendida com potencial de recuperação
  • MACD cruzamento altista — momentum revertendo agora
  • Volume 2.1× acima da média — interesse institucional

  📊 Score: 78/100  |  Setor: Petróleo

Python 3.11 compatible: no backslashes inside f-string expressions.
"""

import logging
from datetime import date
from typing import Optional

import numpy as np

from src.report_builder import (
    bold, bold_pre, escape, fmt_float, fmt_pct, italic,
)

logger = logging.getLogger(__name__)

_SEP = escape("━" * 22)
_SEP_THIN = escape("─" * 22)

_RISK_EMOJI = {
    "Alto":     "🔴",
    "Moderado": "🟡",
    "Baixo":    "🟢",
}

_SIGNAL_LABEL = {
    "strong_buy": "Forte Compra",
    "buy":        "Compra",
}


class OpportunityReportBuilder:
    """
    Builds the opportunity alert Telegram message.

    Each opportunity gets its own message (sent separately if multiple).
    """

    def build_alert(self, opportunity: dict, run_date: Optional[str] = None) -> str:
        """
        Builds a complete MarkdownV2 alert for a single opportunity.

        Args:
            opportunity: Dict from OpportunityScanner._build_opportunity()
            run_date:    Reference date string (YYYY-MM-DD)

        Returns:
            MarkdownV2 string ready for Telegram.
        """
        ticker  = str(opportunity.get("ticker", ""))
        nome    = str(opportunity.get("nome", ""))
        setor   = str(opportunity.get("setor", ""))
        price   = opportunity.get("price")
        targets = opportunity.get("targets", [])
        stop    = opportunity.get("stop_loss")
        risk    = opportunity.get("risk", "Moderado")
        horizon = str(opportunity.get("horizon", ""))
        reasons = opportunity.get("reasons", [])
        score   = opportunity.get("score")
        signal  = opportunity.get("signal", "buy")

        e_low  = opportunity.get("entry_low")
        e_high = opportunity.get("entry_high")

        lines = []

        # ── Header ────────────────────────────────────────────────────────────
        lines.append(_SEP)
        sig_label = escape(_SIGNAL_LABEL.get(signal, "Compra"))
        lines.append(f"🚨 {bold('ALERTA DE OPORTUNIDADE')}")
        lines.append(f"{bold(ticker)}  •  {italic_safe(sig_label)}")
        if nome and nome != ticker:
            lines.append(italic(nome))

        # ── Price & entry ─────────────────────────────────────────────────────
        lines.append("")
        if price is not None:
            price_str = fmt_float(price, 2)
            lines.append(f"💰 Preço atual: {bold_pre('R$ ' + price_str)}")

        if e_low is not None and e_high is not None:
            lo_str = fmt_float(e_low, 2)
            hi_str = fmt_float(e_high, 2)
            lines.append(f"🎯 Entrada sugerida: R\\$ {lo_str} – R\\$ {hi_str}")

        # ── Targets ───────────────────────────────────────────────────────────
        for i, tgt in enumerate(targets[:2], 1):
            level    = tgt.get("level")
            lbl      = escape(str(tgt.get("label", "")))
            upside   = tgt.get("upside", 0)
            if level is not None:
                lvl_str = fmt_float(level, 2)
                ups_str = fmt_pct(upside, 1, sign=True)
                lines.append(f"🏹 Alvo {i}: R\\$ {lvl_str} \\({lbl}\\)  {bold_pre(ups_str)}")

        # ── Stop-loss ─────────────────────────────────────────────────────────
        if stop is not None and price is not None:
            stop_str  = fmt_float(stop, 2)
            stop_down = fmt_pct((stop - price) / price, 1, sign=True)
            lines.append(f"🛑 Stop\\-loss: R\\$ {stop_str}  \\({stop_down}\\)")

        # ── Horizon + risk ────────────────────────────────────────────────────
        if horizon:
            lines.append(f"⏱ Horizonte: {escape(horizon)}")

        risk_e     = escape(risk)
        risk_emoji = _RISK_EMOJI.get(risk, "🟡")
        lines.append(f"⚡ Risco: {risk_emoji} {risk_e}")

        # ── Reasons ───────────────────────────────────────────────────────────
        if reasons:
            lines.append("")
            lines.append(bold("Por que agora:"))
            for reason in reasons:
                lines.append(f"• {escape(reason)}")

        # ── Score + sector footer ─────────────────────────────────────────────
        lines.append("")
        footer_parts = []
        if score is not None:
            footer_parts.append(f"Score: {bold_pre(fmt_float(score, 1))}/100")
        if setor:
            footer_parts.append(f"Setor: {escape(setor)}")

        rsi = opportunity.get("rsi")
        if rsi is not None and not np.isnan(rsi):
            footer_parts.append(f"RSI: {escape(str(int(rsi)))}")

        if footer_parts:
            lines.append("📊 " + escape("  |  ").join(footer_parts))

        lines.append(_SEP_THIN)
        lines.append(f"⚠️ {italic('Não é recomendação de investimento.')}")

        return "\n".join(lines)

    def build_summary_header(
        self,
        count: int,
        run_date: Optional[str] = None,
    ) -> str:
        """
        Header message sent before individual alerts when count > 1.
        """
        date_str = _format_date(run_date)
        count_str = escape(str(count))
        return (
            f"🔔 {bold('Oportunidades Identificadas')}\n"
            f"{escape(date_str)}  •  {count_str} ação\\(ões\\) passaram no filtro rigoroso"
        )

    def build_no_opportunity_message(self) -> Optional[str]:
        """Returns None — when nothing qualifies, we send nothing (silent)."""
        return None


# ─── Helpers ──────────────────────────────────────────────────────────────────

def italic_safe(text: str) -> str:
    """Italic without re-escaping — use when text is already escaped."""
    return f"_{text}_"


def _format_date(run_date: Optional[str]) -> str:
    if run_date and len(run_date) == 10 and "-" in run_date:
        parts = run_date.split("-")
        return f"{parts[2]}/{parts[1]}/{parts[0]}"
    return run_date or date.today().strftime("%d/%m/%Y")


# ─── Convenience ─────────────────────────────────────────────────────────────

def build_opportunity_alerts(
    opportunities: list[dict],
    run_date: Optional[str] = None,
) -> list[str]:
    """
    Returns list of MarkdownV2 strings — one per opportunity.
    Returns empty list if no opportunities.
    """
    if not opportunities:
        return []

    builder = OpportunityReportBuilder()
    messages = []

    if len(opportunities) > 1:
        header = builder.build_summary_header(len(opportunities), run_date)
        messages.append(header)

    for opp in opportunities:
        messages.append(builder.build_alert(opp, run_date))

    return messages
