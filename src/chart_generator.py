"""
chart_generator.py — Módulo 5

Gera o gráfico comparativo de retorno acumulado (base-100) da carteira
recomendada vs IBOVESPA, SELIC e CDI.

Design visual:
  - Fundo escuro (dark mode — compatível com Telegram)
  - Carteira: Terracota (#B5593A) — cor obrigatória, linha mais espessa
  - IBOVESPA:  Cinza claro (#B8B8B8) — linha sólida
  - SELIC:     Âmbar/Dourado (#D4A017) — linha tracejada
  - CDI:       Cinza médio (#888888) — linha pontilhada
  - fill_between entre Carteira e IBOV:
      Alpha positivo (Carteira > IBOV): terracota semi-transparente
      Alpha negativo (Carteira < IBOV): vermelho semi-transparente
  - Output: 1200×800px PNG (figsize=12×8 @ dpi=100), otimizado para Telegram

Uso:
    gen = ChartGenerator()
    chart_path = gen.generate(
        portfolio_returns=port_series,    # pd.Series, retornos decimais diários
        benchmark_returns=df_bench,       # DataFrame ibovespa/selic/cdi
        mode="weekly",
        output_path=Path("output_chart.png"),
        top5_tickers=["PETR4", "VALE3", "ITUB4", "WEGE3", "EGIE3"],
    )
"""

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # backend sem GUI — essencial para GitHub Actions

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.config import CHART_OUTPUT_PATH, TOP_N_RECOMMENDATIONS

logger = logging.getLogger(__name__)

# ─── Paleta de cores ─────────────────────────────────────────────────────────

PALETTE = {
    # Fundos
    "bg":           "#141416",   # quase preto — melhor contraste no Telegram
    "surface":      "#1E1E22",   # superfície do painel

    # Linhas
    "portfolio":    "#B5593A",   # Terracota (obrigatório)
    "ibov":         "#C8C8C8",   # cinza claro
    "selic":        "#D4A017",   # dourado âmbar
    "cdi":          "#808080",   # cinza médio

    # Alpha fill
    "alpha_pos":    "#B5593A",   # Terracota (carteira > IBOV)
    "alpha_neg":    "#C0392B",   # vermelho escuro (carteira < IBOV)

    # Texto e grade
    "text_primary": "#E8E8EA",   # quase branco
    "text_muted":   "#8E8E93",   # cinza suave
    "grid":         "#2C2C30",   # grade muito sutil
    "baseline":     "#3A3A40",   # linha de base (100)
    "accent":       "#B5593A",   # destaque = terracota
}

# Estilos de linha por série
LINE_STYLES: dict[str, dict] = {
    "portfolio": {
        "color":     PALETTE["portfolio"],
        "linewidth": 2.8,
        "linestyle": "-",
        "zorder":    10,
        "label":     "Carteira (Top 5)",
        "alpha":     1.0,
    },
    "ibovespa": {
        "color":     PALETTE["ibov"],
        "linewidth": 1.8,
        "linestyle": "-",
        "zorder":    9,
        "label":     "IBOVESPA",
        "alpha":     0.9,
    },
    "selic": {
        "color":     PALETTE["selic"],
        "linewidth": 1.3,
        "linestyle": "--",
        "zorder":    8,
        "label":     "SELIC",
        "alpha":     0.85,
        "dashes":    (6, 3),
    },
    "cdi": {
        "color":     PALETTE["cdi"],
        "linewidth": 1.1,
        "linestyle": ":",
        "zorder":    7,
        "label":     "CDI",
        "alpha":     0.75,
    },
}

# Formatação de datas por modo
DATE_FORMATS = {
    "weekly":  mdates.DateFormatter("%d/%m"),
    "monthly": mdates.DateFormatter("%b/%y"),
}
DATE_LOCATORS = {
    "weekly":  mdates.WeekdayLocator(byweekday=mdates.MO),
    "monthly": mdates.MonthLocator(),
}


class ChartGenerator:
    """
    Gera gráficos profissionais de retorno acumulado para o relatório.

    Uso:
        gen = ChartGenerator()
        path = gen.generate(portfolio_returns, benchmark_returns, "weekly")
    """

    def __init__(self, output_path: Path = CHART_OUTPUT_PATH):
        self.default_output = Path(output_path)

    # ═══════════════════════════════════════════════════════════════════════
    # API pública
    # ═══════════════════════════════════════════════════════════════════════

    def generate(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.DataFrame,
        mode: str = "weekly",
        output_path: Optional[Path] = None,
        top5_tickers: Optional[list[str]] = None,
        run_date: Optional[str] = None,
    ) -> Path:
        """
        Gera e salva o gráfico de retorno acumulado.

        Args:
            portfolio_returns:  pd.Series de retornos decimais diários da carteira.
                                Deve ter DatetimeIndex alinhado com benchmark_returns.
            benchmark_returns:  DataFrame do BenchmarkManager (ibovespa, selic, cdi).
            mode:               "weekly" ou "monthly" — afeta título e formatação do eixo X.
            output_path:        Caminho para salvar o PNG. Default: config.CHART_OUTPUT_PATH.
            top5_tickers:       Lista dos tickers da carteira (para legenda).
            run_date:           Data de referência para o título (YYYY-MM-DD ou DD/MM/YYYY).

        Returns:
            Path do arquivo PNG gerado.
        """
        out_path = Path(output_path) if output_path else self.default_output
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Montar DataFrame unificado de retornos e converter para base-100
        df_cum = self._build_cumulative(portfolio_returns, benchmark_returns)
        if df_cum.empty:
            logger.error("DataFrame de retornos vazio — gráfico não gerado.")
            return out_path

        # Criar figura
        fig, ax = self._create_figure()

        # Plotar séries
        self._plot_lines(ax, df_cum)

        # fill_between: área de alpha (Carteira vs IBOV)
        if "portfolio" in df_cum.columns and "ibovespa" in df_cum.columns:
            self._plot_alpha_fill(ax, df_cum["portfolio"], df_cum["ibovespa"])

        # Linha de base (100)
        ax.axhline(100, color=PALETTE["baseline"], linewidth=0.8, linestyle="-", zorder=1)

        # Formatação dos eixos e estilo
        self._format_xaxis(ax, df_cum, mode)
        self._format_yaxis(ax, df_cum)
        self._add_title(ax, mode, run_date)
        self._add_performance_box(ax, df_cum, top5_tickers)
        self._add_legend(ax)
        self._add_disclaimer(fig)
        self._apply_dark_style(fig, ax)

        # Salvar
        fig.savefig(
            str(out_path),
            dpi=100,             # 12×8 @ 100dpi = 1200×800px
            bbox_inches="tight",
            facecolor=PALETTE["bg"],
            edgecolor="none",
            format="png",
        )
        plt.close(fig)

        size_kb = out_path.stat().st_size / 1024
        logger.info("Gráfico salvo: %s (%.0f KB, %s)", out_path.name, size_kb, mode)
        return out_path

    @staticmethod
    def compute_portfolio_returns(
        top5_tickers: list[str],
        df_prices: pd.DataFrame,
    ) -> pd.Series:
        """
        Calcula os retornos diários da carteira Top-5 com pesos iguais (20% cada).

        Equal-weight: r_carteira = média simples dos retornos dos 5 tickers.
        Equivalente a rebalancear diariamente (simplificação conservadora).

        Args:
            top5_tickers: lista dos tickers da carteira.
            df_prices:    DataFrame wide de preços ajustados (date × ticker).

        Returns:
            pd.Series de retornos decimais diários com DatetimeIndex.
        """
        valid_tickers = [t for t in top5_tickers if t in df_prices.columns]
        if not valid_tickers:
            logger.warning("Nenhum dos top-5 tickers encontrado em df_prices.")
            return pd.Series(dtype=float)

        if len(valid_tickers) < len(top5_tickers):
            missing = set(top5_tickers) - set(valid_tickers)
            logger.warning("Tickers ausentes em df_prices: %s", missing)

        daily_returns = df_prices[valid_tickers].pct_change().dropna(how="all")

        # Média simples (pesos iguais, rebalanceamento diário)
        portfolio = daily_returns.mean(axis=1).rename("portfolio")
        return portfolio

    # ═══════════════════════════════════════════════════════════════════════
    # Construção dos dados
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _build_cumulative(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Alinha retornos e converte para base-100.

        Fórmula: cumulative_t = 100 × ∏(1 + r_τ) para τ ∈ [0, t]
        O ponto de partida (t=0) é sempre 100 para todas as séries.
        """
        # Unir portfolio + benchmarks em um único DataFrame
        all_series: dict[str, pd.Series] = {}

        port = portfolio_returns.dropna()
        if not port.empty:
            all_series["portfolio"] = port

        for col in ["ibovespa", "selic", "cdi"]:
            if col in benchmark_returns.columns:
                s = benchmark_returns[col].dropna()
                if not s.empty:
                    all_series[col] = s

        if not all_series:
            return pd.DataFrame()

        df = pd.DataFrame(all_series)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().dropna(how="all")

        # Base-100: produto acumulado de (1 + r)
        # NaN em uma série não afeta as outras (fillna(0) apenas para o cumprod)
        df_filled = df.fillna(0.0)
        df_cum = 100.0 * (1.0 + df_filled).cumprod()

        # Restaurar NaN onde havia NaN original (não queremos plotar dados inventados)
        df_cum[df.isna()] = np.nan

        logger.debug(
            "Base-100 construída: %d dias, colunas: %s, range: [%.1f, %.1f]",
            len(df_cum),
            list(df_cum.columns),
            df_cum.min().min(),
            df_cum.max().max(),
        )
        return df_cum

    # ═══════════════════════════════════════════════════════════════════════
    # Plotagem
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _create_figure() -> tuple[plt.Figure, plt.Axes]:
        """Cria figura 1200×800px com fundo escuro."""
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
        fig.patch.set_facecolor(PALETTE["bg"])
        ax.set_facecolor(PALETTE["surface"])
        return fig, ax

    @staticmethod
    def _plot_lines(ax: plt.Axes, df_cum: pd.DataFrame) -> None:
        """Plota cada série com o estilo definido em LINE_STYLES."""
        order = ["cdi", "selic", "ibovespa", "portfolio"]  # portfolio sempre por cima
        for key in order:
            if key not in df_cum.columns:
                continue
            style = LINE_STYLES.get(key, {})
            series = df_cum[key].dropna()
            if series.empty:
                continue

            ax.plot(
                series.index,
                series.values,
                color=style.get("color", "#FFFFFF"),
                linewidth=style.get("linewidth", 1.5),
                linestyle=style.get("linestyle", "-"),
                alpha=style.get("alpha", 1.0),
                zorder=style.get("zorder", 5),
                label=style.get("label", key),
                solid_capstyle="round",
                solid_joinstyle="round",
            )

            # Anotação do valor final à direita da linha
            final_val = series.dropna().iloc[-1]
            ax.annotate(
                f"{final_val:.1f}",
                xy=(series.index[-1], final_val),
                xytext=(6, 0),
                textcoords="offset points",
                color=style.get("color", "#FFFFFF"),
                fontsize=8.5,
                fontweight="bold" if key == "portfolio" else "normal",
                va="center",
                alpha=0.9,
            )

    @staticmethod
    def _plot_alpha_fill(
        ax: plt.Axes,
        portfolio: pd.Series,
        ibov: pd.Series,
    ) -> None:
        """
        fill_between entre carteira e IBOV para destacar o alpha.

        Área positiva (Carteira > IBOV): Terracota semi-transparente
        Área negativa (Carteira < IBOV): vermelho semi-transparente

        Por que dois fill_between?
          Um único fill_between com where= cria artefatos nas cruzadas de linha.
          Usar dois (positivo e negativo) garante transições limpas.
        """
        # Alinhar índices
        common_idx = portfolio.dropna().index.intersection(ibov.dropna().index)
        if len(common_idx) < 2:
            return

        p = portfolio.reindex(common_idx)
        b = ibov.reindex(common_idx)

        # Alpha positivo: carteira acima do IBOV
        ax.fill_between(
            common_idx, p, b,
            where=(p >= b),
            alpha=0.18,
            color=PALETTE["alpha_pos"],
            zorder=3,
            interpolate=True,
            label="_nolegend_",
        )

        # Alpha negativo: carteira abaixo do IBOV
        ax.fill_between(
            common_idx, p, b,
            where=(p < b),
            alpha=0.15,
            color=PALETTE["alpha_neg"],
            zorder=3,
            interpolate=True,
            label="_nolegend_",
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Formatação e estilo
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _format_xaxis(ax: plt.Axes, df_cum: pd.DataFrame, mode: str) -> None:
        """Formata eixo X com localizador e formato de data por modo."""
        locator  = DATE_LOCATORS.get(mode, mdates.WeekdayLocator(byweekday=mdates.MO))
        fmt      = DATE_FORMATS.get(mode, mdates.DateFormatter("%d/%m"))

        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(fmt)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

        # Adicionar margem à direita para as anotações de valor final
        if not df_cum.empty:
            x_end = df_cum.index[-1]
            x_start = df_cum.index[0]
            margin = (x_end - x_start) * 0.05
            ax.set_xlim(x_start, x_end + margin)

    @staticmethod
    def _format_yaxis(ax: plt.Axes, df_cum: pd.DataFrame) -> None:
        """Formata eixo Y com range adequado e grid horizontal."""
        if df_cum.empty:
            return

        y_min = df_cum.min().min()
        y_max = df_cum.max().max()
        padding = (y_max - y_min) * 0.08
        ax.set_ylim(y_min - padding, y_max + padding)

        # Grid horizontal apenas (menos poluído)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.6, alpha=0.8)
        ax.grid(axis="x", visible=False)

        # Desabilitar spines (bordas desnecessárias)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(PALETTE["grid"])

    @staticmethod
    def _add_title(ax: plt.Axes, mode: str, run_date: Optional[str]) -> None:
        """Adiciona título e subtítulo ao gráfico."""
        mode_label = "Semanal" if mode == "weekly" else "Mensal"
        date_str   = run_date or date.today().strftime("%d/%m/%Y")
        # Garantir formato DD/MM/YYYY para exibição
        if date_str and len(date_str) == 10 and "-" in date_str:
            parts = date_str.split("-")
            date_str = f"{parts[2]}/{parts[1]}/{parts[0]}"

        ax.set_title(
            f"Retorno Acumulado — {mode_label}  •  {date_str}",
            color=PALETTE["text_primary"],
            fontsize=13,
            fontweight="bold",
            pad=14,
            loc="left",
        )
        ax.set_xlabel("", labelpad=8)
        ax.set_ylabel("Base 100", color=PALETTE["text_muted"], fontsize=9, labelpad=8)

    @staticmethod
    def _add_performance_box(
        ax: plt.Axes,
        df_cum: pd.DataFrame,
        top5_tickers: Optional[list[str]],
    ) -> None:
        """
        Adiciona caixa de texto com performance do período no canto superior esquerdo.

        Calcula automaticamente o retorno de pico a vale de cada série.
        """
        if df_cum.empty:
            return

        lines: list[str] = []

        period_returns: dict[str, float] = {}
        for col in ["portfolio", "ibovespa", "selic", "cdi"]:
            if col not in df_cum.columns:
                continue
            s = df_cum[col].dropna()
            if len(s) < 2:
                continue
            # Retorno = (valor_final / valor_inicial) - 1
            ret = (s.iloc[-1] / s.iloc[0]) - 1
            period_returns[col] = ret

        # Linha principal: performance da carteira
        if "portfolio" in period_returns:
            p_ret = period_returns["portfolio"]
            sign  = "+" if p_ret >= 0 else ""
            lines.append(f"Carteira:  {sign}{p_ret * 100:.2f}%")

        # Alpha vs IBOV
        if "portfolio" in period_returns and "ibovespa" in period_returns:
            alpha = period_returns["portfolio"] - period_returns["ibovespa"]
            sign  = "+" if alpha >= 0 else ""
            lines.append(f"IBOV:      {'+' if period_returns['ibovespa'] >= 0 else ''}{period_returns['ibovespa'] * 100:.2f}%")
            lines.append(f"Alpha:     {sign}{alpha * 100:.2f}pp")

        if "selic" in period_returns:
            s_ret = period_returns["selic"]
            lines.append(f"SELIC:     +{s_ret * 100:.2f}%")

        # Tickers da carteira (se fornecidos)
        if top5_tickers:
            tickers_str = " · ".join(top5_tickers[:5])
            lines.append(f"\n{tickers_str}")

        if not lines:
            return

        text_content = "\n".join(lines)
        ax.text(
            0.015, 0.985,
            text_content,
            transform=ax.transAxes,
            fontsize=8.5,
            color=PALETTE["text_primary"],
            va="top", ha="left",
            linespacing=1.55,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=PALETTE["bg"],
                edgecolor=PALETTE["grid"],
                alpha=0.85,
                linewidth=0.8,
            ),
            fontfamily="monospace",
            zorder=20,
        )

    @staticmethod
    def _add_legend(ax: plt.Axes) -> None:
        """Legenda compacta no canto superior direito."""
        leg = ax.legend(
            loc="upper right",
            fontsize=9,
            framealpha=0.85,
            facecolor=PALETTE["bg"],
            edgecolor=PALETTE["grid"],
            labelcolor=PALETTE["text_primary"],
            handlelength=1.8,
            handleheight=0.8,
            borderpad=0.6,
            labelspacing=0.45,
        )
        leg.get_frame().set_linewidth(0.8)

    @staticmethod
    def _add_disclaimer(fig: plt.Figure) -> None:
        """Disclaimer financeiro no rodapé."""
        fig.text(
            0.5, 0.01,
            "⚠  Não é recomendação de investimento. Análise quantitativa automatizada. Faça sua própria análise.",
            ha="center",
            fontsize=7.5,
            color=PALETTE["text_muted"],
            style="italic",
        )

    @staticmethod
    def _apply_dark_style(fig: plt.Figure, ax: plt.Axes) -> None:
        """Aplica estilo dark mode consistente em todos os elementos de texto."""
        ax.tick_params(
            axis="both",
            colors=PALETTE["text_muted"],
            labelsize=8.5,
            length=4,
            width=0.5,
        )
        # Remover ticks superiores e direitos
        ax.tick_params(top=False, right=False)

        # Garantir que os ticks do eixo X também usem a cor correta
        for label in ax.get_xticklabels():
            label.set_color(PALETTE["text_muted"])
        for label in ax.get_yticklabels():
            label.set_color(PALETTE["text_muted"])

        # Padding interno
        ax.margins(x=0.02)
        fig.tight_layout(rect=[0, 0.04, 1, 1])  # reservar espaço para disclaimer

    # ═══════════════════════════════════════════════════════════════════════
    # Utilitários
    # ═══════════════════════════════════════════════════════════════════════

    def generate_from_scored(
        self,
        df_scored: pd.DataFrame,
        df_prices: pd.DataFrame,
        benchmark_returns: pd.DataFrame,
        mode: str = "weekly",
        output_path: Optional[Path] = None,
        run_date: Optional[str] = None,
    ) -> Path:
        """
        Conveniência: gera o gráfico diretamente do output do ScoringEngine.

        Extrai os top-5 tickers, calcula retornos da carteira e gera o PNG.

        Args:
            df_scored:          Output do ScoringEngine, ordenado por total_score.
            df_prices:          DataFrame wide de preços ajustados.
            benchmark_returns:  DataFrame do BenchmarkManager.
            mode:               "weekly" ou "monthly".
            output_path:        Destino do PNG.
            run_date:           Data de referência para o título.

        Returns:
            Path do arquivo PNG gerado.
        """
        top5 = (
            df_scored.head(TOP_N_RECOMMENDATIONS)["ticker"].tolist()
            if "ticker" in df_scored.columns
            else []
        )
        portfolio_returns = self.compute_portfolio_returns(top5, df_prices)

        if portfolio_returns.empty:
            logger.warning("Retornos da carteira vazios — gráfico pode estar incompleto.")

        return self.generate(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            mode=mode,
            output_path=output_path,
            top5_tickers=top5,
            run_date=run_date,
        )
