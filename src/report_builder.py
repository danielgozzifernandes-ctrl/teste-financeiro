"""
report_builder.py — Módulo 6

Constrói a mensagem de texto para o Telegram em formato MarkdownV2.

Por que MarkdownV2 e não HTML?
  MarkdownV2 é mais legível como texto plano (fallback),
  suporta negrito/itálico/código sem tags verbosas e é o formato
  recomendado pela Telegram Bot API para mensagens ricas.

Regras críticas do MarkdownV2:
  Os caracteres a seguir DEVEM ser escapados com \\ fora de entidades:
  _ * [ ] ( ) ~ ` > # + - = | { } . !
  E também a própria barra invertida \\.
  O helper escape() cuida disso. NUNCA passe texto raw sem passar por escape().

Estrutura da mensagem:
  📊 Cabeçalho com data e modo
  🏆 Top 5 com score, setor, métricas e why
  📉 Performance da carteira anterior (backtesting)
  ⚠️  Disclaimer legal

Limite do Telegram: 4096 caracteres por mensagem.
  A classe monitora o tamanho e trunca seções se necessário.
"""

import logging
from datetime import date
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Telegram MarkdownV2: todos os caracteres especiais que precisam de escape
_MDV2_SPECIAL_CHARS = r"_*[]()~`>#+-=|{}.!"

# Limite de caracteres do Telegram por mensagem
_TELEGRAM_MAX_CHARS = 4096

# Tamanho máximo do campo "why" por ticker (para caber no limite)
_WHY_MAX_CHARS = 140

# Emojis por posição no ranking
_RANK_EMOJI = {1: "🥇", 2: "🥈", 3: "🥉", 4: "4️⃣", 5: "5️⃣"}

# Emojis por setor (B3)
_SECTOR_EMOJI: dict[str, str] = {
    "Petróleo Gás e Biocombustíveis":    "🛢",
    "Energia Elétrica":                  "⚡",
    "Materiais Básicos":                 "⛏",
    "Consumo Cíclico":                   "🛍",
    "Consumo Não Cíclico":              "🛒",
    "Saúde":                             "🏥",
    "Tecnologia da Informação":          "💻",
    "Comunicações":                      "📡",
    "Utilidade Pública":                 "🚰",
    "Bens Industriais":                  "🏭",
    "Financeiro e Outros":               "🏦",
}


# ─── Helpers de escape e formatação MarkdownV2 ───────────────────────────────

def escape(text: Any) -> str:
    """
    Escapa todos os caracteres especiais do MarkdownV2.

    DEVE ser chamado em todo texto raw antes de inserir na mensagem.
    Não chamar em formatadores (*bold*, _italic_) — apenas no conteúdo.
    """
    s = str(text) if text is not None else ""
    # Barra invertida primeiro (para não re-escapar os próximos)
    s = s.replace("\\", "\\\\")
    for char in _MDV2_SPECIAL_CHARS:
        s = s.replace(char, f"\\{char}")
    return s


def bold(text: Any) -> str:
    """*texto em negrito* para MarkdownV2. Escapa o texto internamente."""
    return f"*{escape(text)}*"


def bold_pre(text: str) -> str:
    """*negrito* sem re-escape — use quando o texto já foi escapado por fmt_*."""
    return f"*{text}*"


def italic(text: Any) -> str:
    """_texto em itálico_ para MarkdownV2. Escapa o texto internamente."""
    return f"_{escape(text)}_"


def italic_pre(text: str) -> str:
    """_itálico_ sem re-escape — use quando o texto já foi escapado por fmt_*."""
    return f"_{text}_"


def code(text: Any) -> str:
    """`código` para MarkdownV2."""
    return f"`{escape(text)}`"


def fmt_pct(value: Optional[float], decimals: int = 2, sign: bool = True) -> str:
    """
    Formata um decimal como percentagem já escapada.
    Ex: 0.0831 -> "+8.31%" (com sinal) ou "8.31%" (sem sinal) — já escapado.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return escape("N/D")
    pct = value * 100
    prefix = "+" if sign and pct >= 0 else ""
    return escape(f"{prefix}{pct:.{decimals}f}%")


def fmt_float(value: Optional[float], decimals: int = 2) -> str:
    """Formata um float já escapado para MarkdownV2. Ex: 4.2 -> "4.2"."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return escape("N/D")
    return escape(f"{value:.{decimals}f}")


def fmt_pp(value: Optional[float], decimals: int = 2) -> str:
    """Formata alpha em pontos percentuais. Ex: 0.032 -> "+3.20 pp" — já escapado."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return escape("N/D")
    pct = value * 100
    prefix = "+" if pct >= 0 else ""
    return escape(f"{prefix}{pct:.{decimals}f} pp")


# ─── ReportBuilder ────────────────────────────────────────────────────────────

class ReportBuilder:
    """
    Constrói a mensagem do Telegram para relatórios semanais e mensais.

    Uso:
        builder = ReportBuilder()
        text = builder.build(
            df_scored=df_scored,
            backtest_result=backtest_result,
            mode="weekly",
            run_date="2025-01-06",
        )
    """

    def build(
        self,
        df_scored: pd.DataFrame,
        backtest_result: Optional[dict] = None,
        mode: str = "weekly",
        run_date: Optional[str] = None,
    ) -> str:
        """
        Monta a mensagem completa em MarkdownV2.

        Args:
            df_scored:       Output do ScoringEngine (ordenado por total_score DESC).
            backtest_result: Resultado do Backtester ou None (primeira execução).
            mode:            "weekly" ou "monthly".
            run_date:        Data de referência no formato "YYYY-MM-DD" ou "DD/MM/YYYY".

        Returns:
            String pronta para envio via Telegram Bot API (parse_mode=MarkdownV2).
        """
        sections: list[str] = []

        sections.append(self._header(mode, run_date))
        sections.append(self._top5_section(df_scored))

        if backtest_result:
            sections.append(self._performance_section(backtest_result))

        sections.append(self._disclaimer())

        message = "\n\n".join(sections)

        # Monitorar comprimento — Telegram limita a 4096 chars
        if len(message) > _TELEGRAM_MAX_CHARS:
            logger.warning(
                "Mensagem excede %d chars (%d). Truncando seção de why.",
                _TELEGRAM_MAX_CHARS, len(message),
            )
            message = self._truncate(message)

        logger.debug("Mensagem montada: %d caracteres", len(message))
        return message

    # ═══════════════════════════════════════════════════════════════════════
    # Seções da mensagem
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _header(mode: str, run_date: Optional[str]) -> str:
        """Cabeçalho com data e modo do relatório."""
        mode_label = "Semana" if mode == "weekly" else "Mês"

        # Normalizar data para DD/MM/AAAA
        if run_date:
            if "-" in run_date and len(run_date) == 10:
                parts = run_date.split("-")
                date_display = f"{parts[2]}/{parts[1]}/{parts[0]}"
            else:
                date_display = run_date
        else:
            date_display = date.today().strftime("%d/%m/%Y")

        return (
            f"📊 {bold(f'Recomendações da {mode_label} — {date_display}')}\n"
            f"{italic('Análise quantitativa automatizada · B3')}"
        )

    def _top5_section(self, df_scored: pd.DataFrame) -> str:
        """Seção Top 5 com detalhes de cada recomendação."""
        if df_scored.empty:
            return f"🏆 {bold('Top 5 Ações')}\n\n{italic('Sem dados disponíveis.')}"

        lines = [f"🏆 {bold('Top 5 Ações')}"]
        for rank, (_, row) in enumerate(df_scored.head(5).iterrows(), start=1):
            lines.append(self._format_recommendation(row, rank))

        return "\n".join(lines)

    def _format_recommendation(self, row: pd.Series, rank: int) -> str:
        """Formata um único ticker com todas as informações."""
        ticker  = str(row.get("ticker", ""))
        setor   = str(row.get("setor", ""))
        score   = row.get("total_score")
        why_raw = str(row.get("why", ""))
        price   = row.get("current_price")

        setor_emoji = _SECTOR_EMOJI.get(setor, "📌")
        rank_emoji  = _RANK_EMOJI.get(rank, f"{rank}\\.")

        # Métricas fundamentalistas
        roe = row.get("roe")
        dy  = row.get("dividend_yield")
        pvp = row.get("pvp")
        ey  = row.get("earnings_yield")
        a6m = row.get("alpha_6m")

        # Linha 1: rank, ticker, preço atual, score
        score_str = fmt_float(score, decimals=1) if score is not None else escape("N/D")
        if price is not None and not pd.isna(price):
            price_str = fmt_float(price, decimals=2)
            line1 = f"\n{rank_emoji} {bold(ticker)} — R\\$ {price_str} — Score: {bold_pre(score_str)}"
        else:
            line1 = f"\n{rank_emoji} {bold(ticker)} — Score: {bold_pre(score_str)}"

        # Linha 2: setor
        line2 = f"   {setor_emoji} {italic(setor)}"

        # Linha 3: métricas — "=" deve ser escapado no MarkdownV2
        metrics_parts: list[str] = []
        if ey is not None and not pd.isna(ey):
            pl_equiv = 1 / ey if ey > 0 else None
            if pl_equiv:
                metrics_parts.append(f"P/L≈{fmt_float(pl_equiv, 1)}")
        if pvp is not None and not pd.isna(pvp):
            metrics_parts.append(f"P/VP\\={fmt_float(pvp, 2)}")
        if roe is not None and not pd.isna(roe):
            metrics_parts.append(f"ROE\\={fmt_pct(roe, 1, sign=False)}")
        if dy is not None and not pd.isna(dy):
            metrics_parts.append(f"DY\\={fmt_pct(dy, 1, sign=False)}")

        line3 = ""
        if metrics_parts:
            metrics_str = escape(" | ").join(metrics_parts)
            line3 = f"   📈 {metrics_str}"

        # Linha 4: momentum relativo
        line4 = ""
        if a6m is not None and not pd.isna(a6m):
            a6m_str = fmt_pct(a6m, 1, sign=True)
            line4 = f"   ⚡ Alpha 6m vs IBOV: {a6m_str}"

        # Linha 5: explicação (why) — truncada e em itálico
        line5 = ""
        why_short = self._truncate_why(why_raw)
        if why_short:
            line5 = f"   💡 {italic(why_short)}"

        return "\n".join(filter(None, [line1, line2, line3, line4, line5]))

    @staticmethod
    def _performance_section(backtest_result: dict) -> str:
        """Seção de performance da carteira anterior."""
        sep = escape("━" * 16)

        status = backtest_result.get("status", "")

        if status == "no_history":
            no_hist_msg = italic('Primeira execução — sem histórico para comparação.')
            return (
                f"{sep}\n"
                f"📉 {bold('Performance da Carteira Anterior')}\n\n"
                f"{no_hist_msg}"
            )

        if status in ("error", "partial_data"):
            msg = escape(backtest_result.get("message", "Dados insuficientes."))
            return (
                f"{sep}\n"
                f"📉 {bold('Performance da Carteira Anterior')}\n\n"
                f"{italic(msg)}"
            )

        # Resultado completo
        port_ret  = backtest_result.get("portfolio_return")
        ibov_ret  = backtest_result.get("benchmark_returns", {}).get("ibovespa")
        cdi_ret   = backtest_result.get("benchmark_returns", {}).get("cdi")
        selic_ret = backtest_result.get("benchmark_returns", {}).get("selic")
        alpha_ibov = backtest_result.get("alpha_vs_ibov")
        alpha_cdi  = backtest_result.get("alpha_vs_cdi")
        period_days = backtest_result.get("period_days", "")

        rec_date = backtest_result.get("recommendation_date", "")
        if rec_date and "-" in rec_date:
            parts = rec_date.split("-")
            rec_date = f"{parts[2]}/{parts[1]}/{parts[0]}"

        period_str = ""
        if rec_date:
            period_str = f" {italic(f'desde {escape(rec_date)}')}"

        lines = [
            f"{sep}",
            f"📉 {bold('Performance da Carteira Anterior')}{period_str}",
            "",
            f"   Carteira:  {bold_pre(fmt_pct(port_ret))}",
            f"   IBOVESPA:  {fmt_pct(ibov_ret)}",
            f"   CDI:       {fmt_pct(cdi_ret)}",
        ]

        if selic_ret is not None:
            lines.append(f"   SELIC:     {fmt_pct(selic_ret)}")

        lines.append("")

        # Alpha com indicador visual
        if alpha_ibov is not None:
            alpha_sign = "🟢" if alpha_ibov >= 0 else "🔴"
            lines.append(f"   {alpha_sign} Alpha vs IBOV: {bold_pre(fmt_pp(alpha_ibov))}")

        if alpha_cdi is not None:
            alpha_sign_cdi = "🟢" if alpha_cdi >= 0 else "🔴"
            lines.append(f"   {alpha_sign_cdi} Alpha vs CDI:  {bold_pre(fmt_pp(alpha_cdi))}")

        # Holdings individuais (compacto)
        holdings = backtest_result.get("holdings", [])
        if holdings:
            lines.append("")
            lines.append(f"   {italic('Composição anterior:')}")
            for h in holdings[:5]:
                t    = escape(str(h.get("ticker", "")))
                ret  = h.get("return")
                sign = "+" if (ret or 0) >= 0 else ""
                ret_str = f"{sign}{(ret or 0)*100:.1f}%"
                lines.append(f"   · {t}: {escape(ret_str)}")

        return "\n".join(lines)

    @staticmethod
    def _disclaimer() -> str:
        """Aviso legal obrigatório."""
        sep = escape("━" * 16)
        return (
            f"{sep}\n"
            f"⚠️ {italic('Não é recomendação de investimento. '  )}"
            f"{italic('Análise quantitativa automatizada. '      )}"
            f"{italic('Faça sua própria análise antes de investir.')}"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Utilitários
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _truncate_why(why_text: str, max_chars: int = _WHY_MAX_CHARS) -> str:
        """
        Trunca o texto 'why' para caber no limite da mensagem.

        Estratégia: manter apenas o primeiro driver se o texto for longo,
        pois é o mais impactante para o leitor.
        """
        if not why_text or why_text in ("Score baseado em múltiplos fatores.", "Dados insuficientes para análise detalhada."):
            return ""
        # Pegar apenas o primeiro fator (separado por ";")
        first_driver = why_text.split(";")[0].strip()
        if len(first_driver) > max_chars:
            first_driver = first_driver[:max_chars - 3] + "..."
        return first_driver

    @staticmethod
    def _truncate(message: str) -> str:
        """
        Trunca a mensagem para o limite do Telegram preservando o disclaimer.

        Estratégia conservadora: cortar a seção de holdings detalhados
        e encerrar com disclaimer.
        """
        disclaimer_marker = "⚠️"
        idx = message.rfind(disclaimer_marker)
        if idx == -1:
            return message[:_TELEGRAM_MAX_CHARS - 10] + escape("...")

        # Manter o início + disclaimer
        body = message[:idx].strip()
        disclaimer = message[idx:]
        available = _TELEGRAM_MAX_CHARS - len(disclaimer) - 5
        return body[:available] + "\n\n" + disclaimer


# ─── Função de conveniência ───────────────────────────────────────────────────

def build_report(
    df_scored: pd.DataFrame,
    backtest_result: Optional[dict] = None,
    mode: str = "weekly",
    run_date: Optional[str] = None,
) -> str:
    """Ponto de entrada simplificado para main.py."""
    return ReportBuilder().build(df_scored, backtest_result, mode, run_date)
