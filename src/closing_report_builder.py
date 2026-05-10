"""
closing_report_builder.py

Builds the end-of-day Telegram report (MarkdownV2).

Sections:
  1. Header       — date + "Fechamento do Mercado"
  2. Per-ticker   — name, close price, daily %, volume
  3. Summary      — portfolio return vs IBOV + daily alpha
  4. Best/worst   — highlight best and worst performer of the day
  5. Disclaimer

Python 3.11 compatible: no backslashes inside f-string expressions.
"""

import logging
from datetime import date
from typing import Optional

import numpy as np

from src.report_builder import (
    bold, bold_pre, escape, fmt_float, fmt_pct, italic, italic_pre,
)

logger = logging.getLogger(__name__)

_SEP = escape("─" * 20)

_SECTOR_EMOJI: dict[str, str] = {
    "Petróleo Gás e Biocombustíveis": "🛢",
    "Energia Elétrica":               "⚡",
    "Materiais Básicos":              "⛏",
    "Consumo Cíclico":                "🛍",
    "Consumo Não Cíclico":           "🛒",
    "Saúde":                          "🏥",
    "Tecnologia da Informação":       "💻",
    "Comunicações":                   "📡",
    "Utilidade Pública":              "🚰",
    "Bens Industriais":               "🏭",
    "Financeiro e Outros":            "🏦",
}


class ClosingReportBuilder:
    """
    Builds the end-of-day Telegram text report.

    Usage:
        builder = ClosingReportBuilder()
        text = builder.build(
            ticker_returns={"PETR4": 0.032, ...},
            ticker_prices={"PETR4": 47.12, ...},
            ibov_return=0.006,
            portfolio_return=0.015,
            run_date="2026-05-12",
            recommendation=rec_dict,
        )
    """

    def build(
        self,
        ticker_returns: dict[str, float],
        ticker_prices: dict[str, float],
        ibov_return: float,
        portfolio_return: float,
        run_date: Optional[str] = None,
        recommendation: Optional[dict] = None,
        volume_ratios: Optional[dict[str, float]] = None,
    ) -> str:
        """
        Assembles the full closing report.

        Args:
            ticker_returns:   {ticker: daily_return_decimal}
            ticker_prices:    {ticker: close_price}
            ibov_return:      IBOV daily return decimal
            portfolio_return: Equal-weight portfolio return for the day
            run_date:         Reference date (YYYY-MM-DD)
            recommendation:   Latest weekly rec dict from SnapshotManager
            volume_ratios:    Optional {ticker: volume_ratio}

        Returns:
            MarkdownV2 string.
        """
        sections = []
        sections.append(self._header(run_date))
        sections.append(
            self._tickers_section(
                ticker_returns, ticker_prices,
                recommendation, volume_ratios or {},
            )
        )
        sections.append(
            self._summary_section(
                ticker_returns, ibov_return, portfolio_return,
            )
        )
        sections.append(self._disclaimer())
        return "\n\n".join(filter(None, sections))

    # ─── Sections ─────────────────────────────────────────────────────────────

    @staticmethod
    def _header(run_date: Optional[str]) -> str:
        date_display = _format_date(run_date)
        return (
            "📊 " + bold("Fechamento do Mercado") + "\n"
            + escape(date_display) + "  •  " + escape("17h30 · B3")
        )

    @staticmethod
    def _tickers_section(
        ticker_returns: dict[str, float],
        ticker_prices: dict[str, float],
        recommendation: Optional[dict],
        volume_ratios: dict[str, float],
    ) -> str:
        lines = [_SEP, "📋 " + bold("Carteira da Semana — Resultado do Dia")]

        # Build metadata map from recommendation
        meta: dict[str, dict] = {}
        if recommendation:
            for item in recommendation.get("top5", []):
                t = item.get("ticker", "")
                meta[t] = item

        # Sort by return descending
        sorted_tickers = sorted(
            ticker_returns.keys(),
            key=lambda t: ticker_returns.get(t, 0),
            reverse=True,
        )

        for ticker in sorted_tickers:
            ret   = ticker_returns.get(ticker)
            price = ticker_prices.get(ticker)

            if ret is None or (isinstance(ret, float) and np.isnan(ret)):
                continue

            pct_str   = fmt_pct(ret, 2, sign=True)
            arrow     = "▲" if ret > 0 else ("▼" if ret < 0 else "─")
            arrow_esc = escape(arrow)

            # Price
            if price and not np.isnan(price):
                price_str = fmt_float(price, 2)
                tick_line = f"  {arrow_esc} {bold(ticker)} R\\$ {price_str}  {bold_pre(pct_str)}"
            else:
                tick_line = f"  {arrow_esc} {bold(ticker)}  {bold_pre(pct_str)}"

            lines.append(tick_line)

            # Sub-line: sector + volume ratio
            sub = []
            ticker_meta = meta.get(ticker, {})
            sector = ticker_meta.get("sector", "")
            if sector:
                sector_e = _SECTOR_EMOJI.get(sector, "📌")
                sub.append(f"{sector_e} {italic(sector)}")

            vol_r = volume_ratios.get(ticker)
            if vol_r and not np.isnan(vol_r):
                vol_str = escape(f"Vol {vol_r:.1f}x")
                sub.append(vol_str)

            if sub:
                lines.append("     " + "  ".join(sub))

        return "\n".join(lines)

    @staticmethod
    def _summary_section(
        ticker_returns: dict[str, float],
        ibov_return: float,
        portfolio_return: float,
    ) -> str:
        alpha_day  = portfolio_return - ibov_return
        alpha_sign = "🟢" if alpha_day >= 0 else "🔴"

        port_str  = fmt_pct(portfolio_return, 2, sign=True)
        ibov_str  = fmt_pct(ibov_return, 2, sign=True)
        alpha_str = fmt_pct(alpha_day, 2, sign=True)

        lines = [
            _SEP,
            "📊 " + bold("Resumo do Dia"),
            "",
            f"   Carteira:  {bold_pre(port_str)}",
            f"   IBOVESPA:  {ibov_str}",
            f"   {alpha_sign} Alpha:    {bold_pre(alpha_str)}",
        ]

        # Best + worst of the day
        valid = {t: r for t, r in ticker_returns.items()
                 if r is not None and not np.isnan(r)}
        if valid:
            best_t  = max(valid, key=lambda t: valid[t])
            worst_t = min(valid, key=lambda t: valid[t])
            best_r  = valid[best_t]
            worst_r = valid[worst_t]

            lines.append("")
            lines.append(
                f"   🏆 Melhor: {bold(best_t)} {bold_pre(fmt_pct(best_r, 2, sign=True))}"
            )
            lines.append(
                f"   💤 Pior:   {bold(worst_t)} {bold_pre(fmt_pct(worst_r, 2, sign=True))}"
            )

        return "\n".join(lines)

    @staticmethod
    def _disclaimer() -> str:
        sep = escape("─" * 20)
        return (
            f"{sep}\n"
            f"⚠️ {italic('Não é recomendação de investimento. ')}"
            f"{italic('Análise quantitativa automatizada.')}"
        )


# ─── Convenience function ─────────────────────────────────────────────────────

def build_closing_report(
    ticker_returns: dict[str, float],
    ticker_prices: dict[str, float],
    ibov_return: float,
    portfolio_return: float,
    run_date: Optional[str] = None,
    recommendation: Optional[dict] = None,
    volume_ratios: Optional[dict[str, float]] = None,
) -> str:
    return ClosingReportBuilder().build(
        ticker_returns=ticker_returns,
        ticker_prices=ticker_prices,
        ibov_return=ibov_return,
        portfolio_return=portfolio_return,
        run_date=run_date,
        recommendation=recommendation,
        volume_ratios=volume_ratios,
    )


# ─── Helper ───────────────────────────────────────────────────────────────────

def _format_date(run_date: Optional[str]) -> str:
    if run_date and len(run_date) == 10 and "-" in run_date:
        parts = run_date.split("-")
        return f"{parts[2]}/{parts[1]}/{parts[0]}"
    return run_date or date.today().strftime("%d/%m/%Y")
