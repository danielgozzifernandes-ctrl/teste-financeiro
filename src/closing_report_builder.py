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
    bold, bold_pre, escape, fmt_float, fmt_pct, fmt_pp, italic, italic_pre,
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
        cumulative_returns: Optional[dict[str, float]] = None,
        cumulative_portfolio_return: Optional[float] = None,
        ibov_cumulative_return: Optional[float] = None,
        recommendation_date: Optional[str] = None,
        stop_alerts: Optional[list[dict]] = None,
        equity_summary: Optional[dict] = None,
        noise_band_pp: Optional[float] = None,
    ) -> str:
        """
        Assembles the full closing report.

        Args:
            ticker_returns:              {ticker: daily_return_decimal}
            ticker_prices:               {ticker: close_price}
            ibov_return:                 IBOV daily return decimal
            portfolio_return:            Equal-weight portfolio daily return
            run_date:                    Reference date (YYYY-MM-DD)
            recommendation:              Latest weekly rec dict from SnapshotManager
            volume_ratios:               Optional {ticker: volume_ratio}
            cumulative_returns:          {ticker: cumulative_return since recommendation}
            cumulative_portfolio_return: Equal-weight cumulative portfolio return since rec
            ibov_cumulative_return:      IBOV cumulative return since recommendation date
            recommendation_date:         Date the recommendation was made (YYYY-MM-DD)
            stop_alerts:                 Saída do stop_monitor.check_levels()
            equity_summary:              Saída do equity_curve.summarize()
            noise_band_pp:               1σ do alpha diário em pp (banda de ruído)

        Returns:
            MarkdownV2 string.
        """
        sections = []
        sections.append(self._header(run_date))

        # Alertas de stop PRIMEIRO — é a única parte acionável do relatório.
        alerts_section = self._stop_alerts_section(stop_alerts or [])
        if alerts_section:
            sections.append(alerts_section)

        sections.append(
            self._tickers_section(
                ticker_returns, ticker_prices,
                recommendation, volume_ratios or {},
                cumulative_returns or {},
            )
        )
        sections.append(
            self._summary_section(
                ticker_returns, ibov_return, portfolio_return,
                cumulative_portfolio_return, ibov_cumulative_return,
                recommendation_date, noise_band_pp,
            )
        )
        equity_section = self._equity_section(equity_summary)
        if equity_section:
            sections.append(equity_section)
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
    def _stop_alerts_section(stop_alerts: list[dict]) -> str:
        """
        Alertas de stop/alvo — a parte ACIONÁVEL do relatório.

        stop_hit:    fechou no/abaixo do stop → sair na abertura
        stop_near:   a <2% do stop → atenção
        target_hit:  alvo conservador atingido → realizar/reavaliar
        """
        if not stop_alerts:
            return ""

        lines = [_SEP, "🚨 " + bold("Alertas de Nível")]
        templates = {
            "stop_hit": (
                "🔴 {t}: fechou em R$ {p:.2f}, ABAIXO do stop R$ {l:.2f} "
                "— regra do sistema: SAIR na abertura"
            ),
            "stop_near": (
                "🟠 {t}: R$ {p:.2f} a {d:.1f}% do stop R$ {l:.2f} — atenção"
            ),
            "target_hit": (
                "🎯 {t}: R$ {p:.2f} atingiu o alvo R$ {l:.2f} "
                "— considerar realizar/reavaliar"
            ),
        }
        for a in stop_alerts:
            tmpl = templates.get(a.get("kind"))
            if not tmpl:
                continue
            txt = tmpl.format(
                t=a["ticker"], p=a["price"], l=a["level"],
                d=abs(a.get("distance_pct", 0)) * 100,
            )
            lines.append("  " + escape(txt))
        return "\n".join(lines)

    @staticmethod
    def _equity_section(equity_summary: Optional[dict]) -> str:
        """
        Desde o início (NAV base 100) — a régua do investidor absoluto.

        CDI primeiro: é o custo de oportunidade real. Bater o Ibov caindo
        menos é consolo relativo; a pergunta absoluta é "paga mais que o CDI?"
        """
        if not equity_summary or equity_summary.get("n_obs", 0) < 2:
            return ""

        cum   = equity_summary["cum_return"]
        cum_c = equity_summary["cum_cdi"]
        cum_i = equity_summary["cum_ibov"]
        a_cdi  = equity_summary["alpha_vs_cdi_pp"]
        a_ibov = equity_summary["alpha_vs_ibov_pp"]
        cum_b = equity_summary.get("cum_blended")
        a_b_cdi = equity_summary.get("blended_alpha_vs_cdi_pp")
        since = equity_summary.get("since") or ""
        since_disp = _format_date(since) if since else "início"

        sign_cdi  = "🟢" if a_cdi >= 0 else "🔴"
        sign_ibov = "🟢" if a_ibov >= 0 else "🔴"

        lines = [
            _SEP,
            "🧭 " + bold("Desde o início") + " " + italic(f"({since_disp}, NAV base 100)"),
        ]

        # Carteira COMPLETA primeiro (bolsa+CDI+IVVB11+IMAB11): é o que o
        # sistema mandou fazer — a régua do investidor absoluto.
        if cum_b is not None and a_b_cdi is not None:
            sign_b = "🟢" if a_b_cdi >= 0 else "🔴"
            lines.append(
                f"   Carteira completa: {bold_pre(fmt_pct(cum_b, 2, sign=True))}"
                f"   {sign_b} vs CDI {bold_pre(fmt_pp(a_b_cdi / 100))}"
            )

        lines += [
            f"   Sleeve bolsa: {bold_pre(fmt_pct(cum, 2, sign=True))}",
            f"   CDI:        {fmt_pct(cum_c, 2, sign=True)}"
            f"   {sign_cdi} {bold_pre(fmt_pp(a_cdi / 100))}",
            f"   IBOVESPA:   {fmt_pct(cum_i, 2, sign=True)}"
            f"   {sign_ibov} {bold_pre(fmt_pp(a_ibov / 100))}",
            italic(f"n={equity_summary['n_obs']} pregões"),
        ]
        return "\n".join(lines)

    @staticmethod
    def _tickers_section(
        ticker_returns: dict[str, float],
        ticker_prices: dict[str, float],
        recommendation: Optional[dict],
        volume_ratios: dict[str, float],
        cumulative_returns: dict[str, float],
    ) -> str:
        lines = [_SEP, "📋 " + bold("Carteira da Semana — Resultado do Dia")]

        # Build metadata map from recommendation
        meta: dict[str, dict] = {}
        if recommendation:
            for item in recommendation.get("top5", []):
                t = item.get("ticker", "")
                meta[t] = item

        # Sort by daily return descending
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

            # Price + daily return
            if price and not np.isnan(price):
                price_str = fmt_float(price, 2)
                tick_line = f"  {arrow_esc} {bold(ticker)} R\\$ {price_str}  {bold_pre(pct_str)}"
            else:
                tick_line = f"  {arrow_esc} {bold(ticker)}  {bold_pre(pct_str)}"

            lines.append(tick_line)

            # Sub-line: sector, volume, cumulative since recommendation
            sub = []
            ticker_meta = meta.get(ticker, {})
            sector = ticker_meta.get("sector", "")
            if sector:
                sector_e = _SECTOR_EMOJI.get(sector, "📌")
                sub.append(f"{sector_e} {italic(sector)}")

            vol_r = volume_ratios.get(ticker)
            if vol_r and not np.isnan(vol_r):
                sub.append(escape(f"Vol {vol_r:.1f}x"))

            cum_r = cumulative_returns.get(ticker)
            if cum_r is not None and not np.isnan(cum_r):
                cum_str = fmt_pct(cum_r, 2, sign=True)
                sub.append(escape("desde rec.: ") + bold_pre(cum_str))

            if sub:
                lines.append("     " + "  ".join(sub))

        return "\n".join(lines)

    @staticmethod
    def _summary_section(
        ticker_returns: dict[str, float],
        ibov_return: float,
        portfolio_return: float,
        cumulative_portfolio_return: Optional[float] = None,
        ibov_cumulative_return: Optional[float] = None,
        recommendation_date: Optional[str] = None,
        noise_band_pp: Optional[float] = None,
    ) -> str:
        alpha_day  = portfolio_return - ibov_return
        alpha_sign = "🟢" if alpha_day >= 0 else "🔴"

        port_str  = fmt_pct(portfolio_return, 2, sign=True)
        ibov_str  = fmt_pct(ibov_return, 2, sign=True)
        alpha_str = fmt_pct(alpha_day, 2, sign=True)

        lines = [
            _SEP,
            "📊 " + bold("Resumo"),
            "",
            italic("Hoje"),
            f"   Portfólio:  {bold_pre(port_str)}",
            f"   IBOVESPA:   {ibov_str}",
            f"   {alpha_sign} Alpha:     {bold_pre(alpha_str)}",
        ]

        # Banda de ruído: alpha de 1 dia de uma estratégia SEMANAL é quase
        # sempre ruído estatístico. Dizer isso explicitamente protege o
        # operador de reagir a flutuação diária.
        if noise_band_pp is not None and noise_band_pp > 0:
            alpha_pp = abs(alpha_day) * 100
            if alpha_pp < noise_band_pp:
                verdict = f"dentro do ruído (1σ = {noise_band_pp:.1f}pp) — ignorar"
            elif alpha_pp < 2 * noise_band_pp:
                verdict = f"entre 1σ e 2σ ({noise_band_pp:.1f}pp) — observar"
            else:
                verdict = f"acima de 2σ ({noise_band_pp:.1f}pp) — atípico"
            lines.append("   " + italic(f"Alpha do dia: {verdict}"))

        # Cumulative block — only when data is available
        if cumulative_portfolio_return is not None and ibov_cumulative_return is not None:
            cum_alpha = cumulative_portfolio_return - ibov_cumulative_return
            cum_sign  = "🟢" if cum_alpha >= 0 else "🔴"

            cum_port_str  = fmt_pct(cumulative_portfolio_return, 2, sign=True)
            cum_ibov_str  = fmt_pct(ibov_cumulative_return, 2, sign=True)
            cum_alpha_str = fmt_pct(cum_alpha, 2, sign=True)

            # Label: "Desde DD/MM" when date is known, else "Acumulado"
            if recommendation_date and len(recommendation_date) == 10:
                parts = recommendation_date.split("-")
                since_label = italic(f"Desde {parts[2]}/{parts[1]}")
            else:
                since_label = italic("Acumulado")

            lines += [
                "",
                since_label,
                f"   Portfólio:  {bold_pre(cum_port_str)}",
                f"   IBOVESPA:   {cum_ibov_str}",
                f"   {cum_sign} Alpha:     {bold_pre(cum_alpha_str)}",
            ]

        # Best + worst of the day
        valid = {t: r for t, r in ticker_returns.items()
                 if r is not None and not np.isnan(r)}
        if valid:
            best_t  = max(valid, key=lambda t: valid[t])
            worst_t = min(valid, key=lambda t: valid[t])

            lines.append("")
            lines.append(
                f"   🏆 Melhor: {bold(best_t)} {bold_pre(fmt_pct(valid[best_t], 2, sign=True))}"
            )
            lines.append(
                f"   💤 Pior:   {bold(worst_t)} {bold_pre(fmt_pct(valid[worst_t], 2, sign=True))}"
            )

        return "\n".join(lines)

    @staticmethod
    def _disclaimer() -> str:
        sep = escape("─" * 20)
        return (
            f"{sep}\n"
            f"⚠️ {italic('Não é recomendação de investimento. Análise quantitativa automatizada.')}"
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
    cumulative_returns: Optional[dict[str, float]] = None,
    cumulative_portfolio_return: Optional[float] = None,
    ibov_cumulative_return: Optional[float] = None,
    recommendation_date: Optional[str] = None,
    stop_alerts: Optional[list[dict]] = None,
    equity_summary: Optional[dict] = None,
    noise_band_pp: Optional[float] = None,
) -> str:
    return ClosingReportBuilder().build(
        ticker_returns=ticker_returns,
        ticker_prices=ticker_prices,
        ibov_return=ibov_return,
        portfolio_return=portfolio_return,
        run_date=run_date,
        recommendation=recommendation,
        volume_ratios=volume_ratios,
        cumulative_returns=cumulative_returns,
        cumulative_portfolio_return=cumulative_portfolio_return,
        ibov_cumulative_return=ibov_cumulative_return,
        recommendation_date=recommendation_date,
        stop_alerts=stop_alerts,
        equity_summary=equity_summary,
        noise_band_pp=noise_band_pp,
    )


# ─── Helper ───────────────────────────────────────────────────────────────────

def _format_date(run_date: Optional[str]) -> str:
    if run_date and len(run_date) == 10 and "-" in run_date:
        parts = run_date.split("-")
        return f"{parts[2]}/{parts[1]}/{parts[0]}"
    return run_date or date.today().strftime("%d/%m/%Y")
