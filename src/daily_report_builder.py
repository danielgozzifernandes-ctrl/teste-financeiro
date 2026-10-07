"""
daily_report_builder.py

Builds the morning daily report for Telegram (MarkdownV2).

Sections:
  1. Header       — date, day of week, market opens in X min
  2. Macro        — USD/BRL, S&P 500, petróleo, VIX, risk sentiment
  3. Top 5 Técnico — RSI, MACD, Bollinger, MAs, trend, signal for each ticker
  4. Alertas      — gaps, volume spikes, MACD crossovers, 52w extremes
  5. Disclaimer

Python 3.11 compatible: no backslashes inside f-string expressions.
"""

import logging
from datetime import date, datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.report_builder import (
    bold, bold_pre, escape, fmt_float, fmt_pct, italic, italic_pre,
)

logger = logging.getLogger(__name__)

_SIGNAL_EMOJI = {
    "strong_buy":  "🟢🟢",
    "buy":         "🟢",
    "neutral":     "⚪",
    "sell":        "🔴",
    "strong_sell": "🔴🔴",
}

_SIGNAL_LABEL = {
    "strong_buy":  "Forte compra",
    "buy":         "Compra",
    "neutral":     "Neutro",
    "sell":        "Venda",
    "strong_sell": "Forte venda",
}

_TREND_EMOJI = {
    "uptrend":   "📈",
    "downtrend": "📉",
    "sideways":  "➡️",
}

_SENTIMENT_LABEL = {
    "risk_on":  "Risk-on 🟢",
    "risk_off": "Risk-off 🔴",
    "neutral":  "Neutro ⚪",
}

_WEEKDAY_PT = {
    0: "Segunda", 1: "Terça", 2: "Quarta",
    3: "Quinta",  4: "Sexta", 5: "Sábado", 6: "Domingo",
}

_SEP = escape("─" * 20)


class DailyReportBuilder:
    """
    Builds the morning Telegram report.

    Usage:
        builder = DailyReportBuilder()
        text = builder.build(
            macro_snapshot=macro,
            tech_data=tech,
            alerts=alerts,
            run_date="2026-05-12",
            sentiment="risk_on",
        )
    """

    def build(
        self,
        macro_snapshot: dict[str, dict],
        tech_data: dict[str, dict],
        alerts: list[dict],
        run_date: Optional[str] = None,
        sentiment: str = "neutral",
        recommendation: Optional[dict] = None,
    ) -> str:
        """
        Assembles the full morning report.

        Args:
            macro_snapshot:  Output of MacroFetcher.get_snapshot()
            tech_data:       Output of TechnicalAnalyzer.analyze_all()
            alerts:          Output of TechnicalAnalyzer.detect_alerts()
            run_date:        Reference date (YYYY-MM-DD)
            sentiment:       "risk_on" | "risk_off" | "neutral"
            recommendation:  Latest weekly recommendation dict (from SnapshotManager)

        Returns:
            MarkdownV2 string ready for Telegram.
        """
        sections = []

        sections.append(self._header(run_date, sentiment))

        macro_sec = self._macro_section(macro_snapshot)
        if macro_sec:
            sections.append(macro_sec)

        if tech_data:
            sections.append(self._technical_section(tech_data, recommendation))

        alert_sec = self._alerts_section(alerts)
        if alert_sec:
            sections.append(alert_sec)

        sections.append(self._disclaimer())

        return "\n\n".join(filter(None, sections))

    # Sections

    @staticmethod
    def _header(run_date: Optional[str], sentiment: str) -> str:
        today = date.today()
        if run_date and len(run_date) == 10 and "-" in run_date:
            parts = run_date.split("-")
            date_str = f"{parts[2]}/{parts[1]}/{parts[0]}"
            try:
                parsed = datetime.strptime(run_date, "%Y-%m-%d").date()
                weekday = _WEEKDAY_PT.get(parsed.weekday(), "")
            except ValueError:
                weekday = _WEEKDAY_PT.get(today.weekday(), "")
        else:
            date_str = today.strftime("%d/%m/%Y")
            weekday = _WEEKDAY_PT.get(today.weekday(), "")

        sentiment_txt = _SENTIMENT_LABEL.get(sentiment, "Neutro ⚪")
        sentiment_esc = escape(sentiment_txt)

        header = (
            "📊 " + bold("Bom dia! Relatório Diário") + "\n"
            + escape(f"{weekday}, {date_str}") + "  •  " + sentiment_esc
        )
        return header

    @staticmethod
    def _macro_section(snapshot: dict[str, dict]) -> str:
        if not snapshot:
            return ""

        lines = [_SEP, "🌍 " + bold("Macro")]

        order = ["usdbrl", "ibov", "sp500", "nasdaq", "oil_wti", "brent", "gold", "vix", "dxy"]
        shown = 0
        for key in order:
            item = snapshot.get(key)
            if not item:
                continue

            label     = escape(item.get("label", key))
            value     = item.get("value", 0)
            prefix    = item.get("prefix", "")
            dec       = item.get("decimals", 2)
            chg       = item.get("change_pct", 0) * 100
            direction = item.get("direction", "flat")

            arrow = "▲" if direction == "up" else ("▼" if direction == "down" else "─")
            sign  = "+" if chg >= 0 else ""

            if prefix:
                val_str = escape(f"{prefix} {value:,.{dec}f}")
            elif dec == 0:
                val_str = escape(f"{value:,.0f}")
            else:
                val_str = escape(f"{value:,.{dec}f}")

            chg_str = escape(f"{sign}{chg:.2f}%")
            arrow_e = escape(arrow)

            lines.append(f"  {arrow_e} {label}: {bold_pre(val_str)}  {chg_str}")
            shown += 1

        return "\n".join(lines) if shown > 0 else ""

    @staticmethod
    def _technical_section(
        tech_data: dict[str, dict],
        recommendation: Optional[dict],
    ) -> str:
        lines = [_SEP, "📈 " + bold("Top 5 — Análise Técnica")]

        rec_tickers: dict[str, dict] = {}
        if recommendation:
            for item in recommendation.get("top5", []):
                t = item.get("ticker", "")
                rec_tickers[t] = item

        for rank, (ticker, data) in enumerate(tech_data.items(), 1):
            if not data or "price" not in data:
                continue

            lines.append("")

            # Line 1: rank, ticker, price, signal
            price     = data.get("price", 0)
            price_str = fmt_float(price, 2)
            signal    = data.get("signal", "neutral")
            sig_emoji = _SIGNAL_EMOJI.get(signal, "⚪")
            sig_label = escape(_SIGNAL_LABEL.get(signal, "Neutro"))
            rank_str  = escape(f"{rank}.")

            change_1d = data.get("change_1d")
            if change_1d is not None and not np.isnan(change_1d):
                chg_str = fmt_pct(change_1d, 2, sign=True)
                lines.append(
                    f"{rank_str} {bold(ticker)} — R\\$ {price_str} "
                    f"\\({chg_str}\\)  {sig_emoji} {italic_pre(sig_label)}"
                )
            else:
                lines.append(
                    f"{rank_str} {bold(ticker)} — R\\$ {price_str}  "
                    f"{sig_emoji} {italic_pre(sig_label)}"
                )

            # Line 2: RSI + trend + MA position
            rsi   = data.get("rsi")
            trend = data.get("trend", "sideways")
            trend_e = _TREND_EMOJI.get(trend, "➡️")
            ma20  = data.get("ma20")
            price_f = data.get("price", 0)

            indicators = []
            if rsi is not None and not np.isnan(rsi):
                rsi_str = escape(f"{rsi:.0f}")
                rsi_label = "sobrevendido" if rsi < 30 else ("sobrecomprado" if rsi > 70 else "neutro")
                rsi_l_esc = escape(rsi_label)
                indicators.append(f"RSI {rsi_str} \\({rsi_l_esc}\\)")

            if ma20 and price_f:
                ma_label = "acima MA20" if price_f > ma20 else "abaixo MA20"
                indicators.append(escape(ma_label))

            crossover = data.get("macd_crossover")
            if crossover == "bullish":
                indicators.append(escape("MACD ↑"))
            elif crossover == "bearish":
                indicators.append(escape("MACD ↓"))

            if indicators:
                ind_str = escape(" | ").join(indicators)
                lines.append(f"   {trend_e} {ind_str}")

            # Line 3: volume + gap
            extra = []
            vol_ratio = data.get("volume_ratio")
            if vol_ratio is not None and not np.isnan(vol_ratio):
                vr_str = escape(f"Vol {vol_ratio:.1f}x média")
                extra.append(vr_str)

            gap = data.get("gap_pct")
            if gap is not None and not np.isnan(gap) and abs(gap) >= 0.005:
                gap_pct = gap * 100
                sign = "+" if gap_pct >= 0 else ""
                gap_str = escape(f"Gap {sign}{gap_pct:.1f}%")
                extra.append(gap_str)

            if extra:
                lines.append("   " + "  ".join(extra))

        return "\n".join(lines)

    @staticmethod
    def _alerts_section(alerts: list[dict]) -> str:
        if not alerts:
            return ""

        lines = [_SEP, "⚡ " + bold("Alertas do Dia")]

        for alert in alerts[:8]:
            ticker  = escape(str(alert.get("ticker", "")))
            label   = escape(str(alert.get("label", "")))
            emoji   = alert.get("emoji", "•")
            value   = alert.get("value")
            a_type  = alert.get("type", "")

            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                if a_type in ("gap_up", "gap_down"):
                    pct = value * 100
                    sign = "+" if pct >= 0 else ""
                    val_str = escape(f"{sign}{pct:.1f}%")
                    lines.append(f"  {emoji} {bold_pre(ticker)}: {label} {val_str}")
                elif a_type == "volume_spike":
                    val_str = escape(f"{value:.1f}x")
                    lines.append(f"  {emoji} {bold_pre(ticker)}: {label} {val_str}")
                elif a_type in ("near_52w_high", "near_52w_low"):
                    pct = abs(value) * 100
                    val_str = escape(f"{pct:.1f}% de distância")
                    lines.append(f"  {emoji} {bold_pre(ticker)}: {label} \\({val_str}\\)")
                elif a_type in ("oversold", "overbought"):
                    val_str = escape(f"RSI {value:.0f}")
                    lines.append(f"  {emoji} {bold_pre(ticker)}: {label} \\({val_str}\\)")
                else:
                    lines.append(f"  {emoji} {bold_pre(ticker)}: {label}")
            else:
                lines.append(f"  {emoji} {bold_pre(ticker)}: {label}")

        if not alerts:
            lines.append(italic("Nenhum alerta relevante hoje."))

        return "\n".join(lines)

    @staticmethod
    def _disclaimer() -> str:
        sep = escape("─" * 20)
        return (
            f"{sep}\n"
            f"⚠️ {italic('Análise técnica automatizada. '  )}"
            f"{italic('Não é recomendação de investimento. ')}"
            f"{italic('Faça sua própria análise.')}"
        )


# Convenience function

def build_daily_report(
    macro_snapshot: dict,
    tech_data: dict,
    alerts: list,
    run_date: Optional[str] = None,
    sentiment: str = "neutral",
    recommendation: Optional[dict] = None,
) -> str:
    return DailyReportBuilder().build(
        macro_snapshot=macro_snapshot,
        tech_data=tech_data,
        alerts=alerts,
        run_date=run_date,
        sentiment=sentiment,
        recommendation=recommendation,
    )
