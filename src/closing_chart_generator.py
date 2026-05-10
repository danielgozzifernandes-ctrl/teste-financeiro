"""
closing_chart_generator.py

Generates the end-of-day horizontal bar chart showing daily returns
for each portfolio ticker vs IBOV and portfolio average.

Design:
  - Dark theme consistent with chart_generator.py (same PALETTE)
  - Horizontal bars: green for positive, red for negative
  - Portfolio weighted-average line
  - IBOV reference line
  - Per-bar price annotation (close price + daily %)
  - Output: 1000×600px PNG optimized for Telegram
"""

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

logger = logging.getLogger(__name__)

# Same dark palette as chart_generator.py
PALETTE = {
    "bg":           "#141416",
    "surface":      "#1E1E22",
    "text_primary": "#E8E8EA",
    "text_muted":   "#8E8E93",
    "grid":         "#2C2C30",
    "positive":     "#2ECC71",   # green
    "negative":     "#E74C3C",   # red
    "neutral":      "#8E8E93",
    "portfolio":    "#B5593A",   # terracota (consistent with main chart)
    "ibov":         "#C8C8C8",
}


class ClosingChartGenerator:
    """
    Generates the EOD horizontal bar chart.

    Usage:
        gen = ClosingChartGenerator()
        path = gen.generate(
            ticker_returns={"PETR4": 0.032, "VALE3": 0.021, ...},
            ibov_return=0.006,
            portfolio_return=0.015,
            run_date="2026-05-12",
            ticker_prices={"PETR4": 47.12, ...},
            output_path=Path("output/closing_chart_2026-05-12.png"),
        )
    """

    def generate(
        self,
        ticker_returns: dict[str, float],
        ibov_return: float,
        portfolio_return: float,
        run_date: Optional[str] = None,
        ticker_prices: Optional[dict[str, float]] = None,
        output_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Generates and saves the closing bar chart.

        Args:
            ticker_returns:   {ticker: daily_return_decimal}  e.g. {"PETR4": 0.032}
            ibov_return:      IBOV daily return decimal
            portfolio_return: Equal-weight portfolio daily return
            run_date:         Reference date string (YYYY-MM-DD or DD/MM/YYYY)
            ticker_prices:    Optional {ticker: close_price} for annotations
            output_path:      PNG output path

        Returns:
            Path to saved PNG, or None on failure.
        """
        if not ticker_returns:
            logger.warning("ClosingChartGenerator: ticker_returns vazio")
            return None

        if output_path is None:
            date_str = run_date or date.today().strftime("%Y-%m-%d")
            output_path = Path("output") / f"closing_chart_{date_str}.png"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._draw(
                ticker_returns=ticker_returns,
                ibov_return=ibov_return,
                portfolio_return=portfolio_return,
                run_date=run_date,
                ticker_prices=ticker_prices or {},
                output_path=output_path,
            )
            size_kb = output_path.stat().st_size / 1024
            logger.info("Closing chart salvo: %s (%.0f KB)", output_path.name, size_kb)
            return output_path
        except Exception as exc:
            logger.error("ClosingChartGenerator: falha ao gerar gráfico: %s", exc, exc_info=True)
            return None

    def _draw(
        self,
        ticker_returns: dict[str, float],
        ibov_return: float,
        portfolio_return: float,
        run_date: Optional[str],
        ticker_prices: dict[str, float],
        output_path: Path,
    ) -> None:
        # Sort tickers by return (descending)
        sorted_items = sorted(ticker_returns.items(), key=lambda x: x[1], reverse=True)
        tickers = [t for t, _ in sorted_items]
        returns = [r for _, r in sorted_items]

        n = len(tickers)
        fig_height = max(5.0, 1.2 * (n + 2))
        fig, ax = plt.subplots(figsize=(10, fig_height), dpi=100)
        fig.patch.set_facecolor(PALETTE["bg"])
        ax.set_facecolor(PALETTE["surface"])

        # ── Horizontal bars ──────────────────────────────────────────────────
        y_pos = np.arange(n)
        colors = [PALETTE["positive"] if r >= 0 else PALETTE["negative"] for r in returns]

        bars = ax.barh(
            y_pos, [r * 100 for r in returns],
            color=colors, height=0.55,
            alpha=0.88, zorder=3,
            edgecolor="none",
        )

        # ── Portfolio + IBOV reference lines ─────────────────────────────────
        ax.axvline(
            portfolio_return * 100,
            color=PALETTE["portfolio"], linewidth=2.0,
            linestyle="--", zorder=5, alpha=0.9,
            label=f"Carteira ({portfolio_return*100:+.2f}%)",
        )
        ax.axvline(
            ibov_return * 100,
            color=PALETTE["ibov"], linewidth=1.5,
            linestyle=":", zorder=4, alpha=0.8,
            label=f"IBOV ({ibov_return*100:+.2f}%)",
        )
        ax.axvline(0, color=PALETTE["grid"], linewidth=0.8, zorder=2)

        # ── Bar annotations ───────────────────────────────────────────────────
        for i, (ticker, ret) in enumerate(zip(tickers, returns)):
            pct_str = f"{ret*100:+.2f}%"
            price   = ticker_prices.get(ticker)
            label   = f"{pct_str}  R${price:.2f}" if price else pct_str

            x_offset = 0.15
            ha = "left" if ret >= 0 else "right"
            x_pos = ret * 100 + (x_offset if ret >= 0 else -x_offset)

            ax.text(
                x_pos, i, label,
                va="center", ha=ha,
                fontsize=9, color=PALETTE["text_primary"],
                fontfamily="monospace", zorder=10,
            )

        # ── Y-axis tickers ────────────────────────────────────────────────────
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tickers, fontsize=10.5, color=PALETTE["text_primary"], fontweight="bold")
        ax.tick_params(axis="y", length=0)

        # ── X-axis ────────────────────────────────────────────────────────────
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.1f}%"))
        ax.tick_params(axis="x", colors=PALETTE["text_muted"], labelsize=8.5, length=3)

        # X range: symmetric around 0, with margin
        max_abs = max(abs(r) * 100 for r in returns + [ibov_return, portfolio_return])
        margin = max(1.0, max_abs * 0.4)
        ax.set_xlim(-max_abs - margin, max_abs + margin)

        # ── Grid ──────────────────────────────────────────────────────────────
        ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.6, alpha=0.7, zorder=1)
        ax.grid(axis="y", visible=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # ── Title ────────────────────────────────────────────────────────────
        date_display = _format_date(run_date)
        ax.set_title(
            f"Desempenho do Dia  •  {date_display}",
            color=PALETTE["text_primary"],
            fontsize=12, fontweight="bold",
            pad=12, loc="left",
        )

        # ── Performance box (portfolio vs IBOV alpha) ────────────────────────
        alpha_day = portfolio_return - ibov_return
        alpha_sign = "+" if alpha_day >= 0 else ""
        box_lines = [
            f"Carteira:  {portfolio_return*100:+.2f}%",
            f"IBOV:      {ibov_return*100:+.2f}%",
            f"Alpha dia: {alpha_sign}{alpha_day*100:.2f}pp",
        ]
        ax.text(
            0.98, 0.97, "\n".join(box_lines),
            transform=ax.transAxes, fontsize=8.5,
            color=PALETTE["text_primary"], va="top", ha="right",
            fontfamily="monospace", linespacing=1.55,
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor=PALETTE["bg"],
                edgecolor=PALETTE["grid"],
                alpha=0.88, linewidth=0.8,
            ),
            zorder=20,
        )

        # ── Legend ────────────────────────────────────────────────────────────
        ax.legend(
            loc="lower right", fontsize=8.5,
            framealpha=0.85, facecolor=PALETTE["bg"],
            edgecolor=PALETTE["grid"],
            labelcolor=PALETTE["text_primary"],
        )

        # ── Disclaimer ────────────────────────────────────────────────────────
        fig.text(
            0.5, 0.01,
            "⚠  Não é recomendação de investimento. Análise quantitativa automatizada.",
            ha="center", fontsize=7.5,
            color=PALETTE["text_muted"], style="italic",
        )

        fig.tight_layout(rect=[0, 0.04, 1, 1])
        fig.savefig(
            str(output_path),
            dpi=100, bbox_inches="tight",
            facecolor=PALETTE["bg"], edgecolor="none", format="png",
        )
        plt.close(fig)


def _format_date(run_date: Optional[str]) -> str:
    if run_date and len(run_date) == 10 and "-" in run_date:
        parts = run_date.split("-")
        return f"{parts[2]}/{parts[1]}/{parts[0]}"
    return run_date or date.today().strftime("%d/%m/%Y")
