"""
Benchmarks: IBOV (yfinance ^BVSP), SELIC e CDI (BCB SGS 11 e 12).

get_returns() devolve retornos diários decimais (não base-100), colunas
ibovespa/selic/cdi, indexados pelos pregões do IBOV. BCB entra por left
join com forward-fill curto para lacunas do SGS.

SGS 11 e 12 vêm em % a.d. (ex.: 0,0551). _convert_to_daily_return decide
o formato pela mediana, então também aceitaria % a.a. se o BCB mudar.
"""

import logging
import time
from datetime import date, datetime, time as dtime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import (
    BCB_CDI_SERIES,
    BCB_SELIC_SERIES,
    BENCHMARKS,
    CACHE_DIR_PATH,
    CACHE_TTL_HOURS,
)
from src.b3_calendar import BRT, is_trading_day, now_brt  # noqa: F401 (reexport)
from src.data_collector import CacheManager

logger = logging.getLogger(__name__)

# Dias úteis por ano — convenção DU/252 (padrão ANBIMA/B3)
_DU_YEAR = 252

# Parâmetros de retry para o SGS/BCB
_BCB_MAX_RETRIES = 3
_BCB_BACKOFF_BASE = 2  # espera = 2**tentativa → 1s, 2s

# Máximo de dias de forward-fill para lacunas do BCB
# 5 cobre Carnaval (quarta a sexta = 3 pregões) + margem para feriados estaduais
_BCB_FFILL_LIMIT = 5

# Intervalo plausível para taxa diária brasileira (decimal, não %)
# SELIC mín histórico ~2% a.a. → (1.02)^(1/252)-1 ≈ 0.0000788
# SELIC máx histórico ~45% a.a. (1999) → (1.45)^(1/252)-1 ≈ 0.00148
_BCB_DAILY_RATE_MIN = 0.00005   # 0.005% a.d.  (~1.3% a.a.) — limite inferior seguro
_BCB_DAILY_RATE_MAX = 0.00200   # 0.200% a.d. (~65% a.a.)  — limite superior seguro

_SESSION_OPEN_BRT = dtime(10, 0)
# Pregão fecha 17:00 (18:00 no horário de verão americano); o yfinance
# costuma consolidar o candle do dia só depois disso.
_SESSION_SETTLED_BRT = dtime(18, 30)

# BOVA11 replica o Ibovespa (taxa 0,10% a.a., dividendos reinvestidos),
# então o retorno no mesmo período deve bater com o do índice.
# Tolerância de 1,5pp: o tracking semanal do ETF é < 0,1pp; o resto é
# horário — execução no meio do pregão grava um print intradiário. Nos 25
# backtests de mai–out/2026 o maior desvio contra o BOVA11 foi 0,99pp
# (semana da eleição, rodada às 16h28). Erro de período, candle quebrado
# ou unidade dá desvio de vários pp (σ semanal do IBOV ~2,5pp), bem acima.
IBOV_ETF_TICKER = "BOVA11.SA"
IBOV_ETF_TOLERANCE = 0.015


def is_intraday(now: Optional[datetime] = None) -> bool:
    """True se `now` cai dentro de um pregão ainda não consolidado."""
    now = (now or now_brt()).astimezone(BRT)
    if not is_trading_day(now.date()):
        return False
    return _SESSION_OPEN_BRT <= now.time() < _SESSION_SETTLED_BRT


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    s = df[name]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s


def clean_close(
    df: pd.DataFrame, now: Optional[datetime] = None,
) -> tuple[pd.Series, dict]:
    """
    Fechamentos de um download do yfinance, separando candles incompletos.

    O yfinance publica o pregão corrente com OHL zerado e/ou volume 0 até
    consolidar; o Close desse candle é um print ao vivo, não fechamento.
    Se for o último candle, fica na série mas é marcado como intradiário;
    em qualquer outra posição é descartado. Candle de hoje antes do pregão
    consolidar também conta como intradiário, mesmo com OHLV preenchido.
    """
    close = _col(df, "Close").dropna()
    if close.empty:
        return close, {"last_bar": None, "intraday": False}

    bad = pd.Series(False, index=close.index)
    for c in ("Open", "High", "Low", "Volume"):
        if c in df.columns.get_level_values(0):
            bad |= _col(df, c).reindex(close.index).fillna(0) <= 0

    last = close.index[-1]
    dropped = [d for d in close.index[bad] if d != last]
    if dropped:
        logger.warning(
            "%d candle(s) incompleto(s) descartado(s): %s",
            len(dropped), ", ".join(str(d.date()) for d in dropped),
        )
        close = close.drop(dropped)

    now = (now or now_brt()).astimezone(BRT)
    today_open = last.date() == now.date() and is_intraday(now)
    meta = {
        "last_bar": str(last.date()),
        "intraday": bool(bad.loc[last]) or today_open,
    }
    if dropped:
        meta["dropped_bars"] = [str(d.date()) for d in dropped]
    return close, meta


def period_return_from_close(
    close: pd.Series, start: date, end: date,
) -> Optional[float]:
    """
    Retorno do último fechamento antes de `start` até o último candle <= `end`.
    Mesma janela de get_period_return (retornos diários com data >= start).
    """
    dates = pd.Index(close.index.date)
    base = close[dates < start]
    tail = close[dates <= end]
    if base.empty or tail.empty:
        return None
    return float(tail.iloc[-1] / base.iloc[-1] - 1)


def last_bar_on_or_before(close: pd.Series, d: date) -> Optional[date]:
    dates = [x for x in close.index.date if x <= d]
    return max(dates) if dates else None


class BenchmarkError(Exception):
    """Falha irrecuperável na coleta de dados de benchmark."""


class BenchmarkManager:
    """Coleta e alinha IBOV, SELIC e CDI (retornos diários decimais)."""

    def __init__(self, cache: Optional[CacheManager] = None):
        self.cache = cache or CacheManager(
            cache_dir=CACHE_DIR_PATH, ttl_hours=CACHE_TTL_HOURS
        )
        # Estado do último candle do IBOV usado (preenchido em get_returns).
        self.ibov_meta: dict = {}

    def get_returns(
        self,
        start_date: str | date | datetime,
        end_date: Optional[str | date | datetime] = None,
    ) -> pd.DataFrame:
        """
        Retornos diários de start a end (inclusive; end default = hoje).
        NaN onde a fonte não tem dado. Levanta BenchmarkError se o IBOV
        falhar, porque ele define o calendário.
        """
        start = _parse_date(start_date)
        end   = _parse_date(end_date) if end_date else date.today()

        # margem para o pct_change do primeiro dia
        fetch_start = start - timedelta(days=7)

        cache_key = f"benchmark_returns_{start}_{end}"
        cached = self.cache.get(cache_key)
        if cached:
            df = pd.DataFrame(cached)
            df.index = pd.to_datetime(df.index)
            logger.info("Benchmarks: cache hit (%d dias)", len(df))
            # Só candles consolidados vão para o cache.
            self.ibov_meta = {
                "last_bar": str(df.index[-1].date()) if len(df) else None,
                "intraday": False,
                "from_cache": True,
            }
            return df

        ibov_returns = self._fetch_ibov(fetch_start, end)
        if ibov_returns is None or ibov_returns.empty:
            raise BenchmarkError(
                "Falha ao obter IBOVESPA (^BVSP). "
                "Verifique conexão ou disponibilidade do yfinance."
            )

        bcb_df = self._fetch_bcb_with_retry(fetch_start, end)

        # Alinhamento de calendário
        df = self._align_and_merge(ibov_returns, bcb_df)

        # Cortar para a janela solicitada (após pct_change que consome 1 linha)
        df = df.loc[df.index.date >= start]

        # Garantir tipos float64
        df = df.astype("float64")

        if not self.ibov_meta.get("intraday"):
            df_cache = df.copy()
            df_cache.index = df_cache.index.strftime("%Y-%m-%d")
            self.cache.set(cache_key, df_cache.to_dict())
        logger.info(
            "Benchmarks coletados: %d dias | %s → %s",
            len(df), df.index[0].date() if len(df) else "?", df.index[-1].date() if len(df) else "?",
        )
        return df

    def get_cumulative(
        self,
        start_date: str | date | datetime,
        end_date: Optional[str | date | datetime] = None,
        base: float = 100.0,
    ) -> pd.DataFrame:
        """base × ∏(1 + r_t), com o primeiro dia da janela já aplicado."""
        daily = self.get_returns(start_date, end_date)
        # Inserir linha inicial com retorno zero (t=0, base=100)
        zero_row = pd.DataFrame(
            [[0.0] * len(daily.columns)],
            index=[daily.index[0] - pd.tseries.offsets.BDay(1)],
            columns=daily.columns,
        )
        with_zero = pd.concat([zero_row, daily])
        cumulative = base * (1 + with_zero).cumprod()
        return cumulative.iloc[1:]  # remover linha auxiliar t=0

    def get_period_return(
        self,
        start_date: str | date | datetime,
        end_date: Optional[str | date | datetime] = None,
    ) -> pd.Series:
        """∏(1 + r_t) - 1 por benchmark, com retornos de data >= start."""
        daily = self.get_returns(start_date, end_date)
        return (1 + daily).prod() - 1

    # Coleta IBOVESPA

    def _fetch_ibov(
        self,
        start: date,
        end: date,
    ) -> Optional[pd.Series]:
        """
        Baixa preços do IBOVESPA via yfinance e retorna retornos diários decimais.

        Usa auto_adjust=True para preços já ajustados por dividendos/splits.
        O pct_change() produz r_t = (P_t / P_{t-1}) - 1.
        """
        ibov_ticker = BENCHMARKS["ibovespa"]["ticker"]  # "^BVSP"
        cache_key   = f"ibov_raw_{start}_{end}"

        cached = self.cache.get(cache_key)
        if cached:
            s = pd.Series(cached["returns"], index=pd.to_datetime(cached["dates"]))
            logger.debug("IBOV: cache hit (%d dias)", len(s))
            self.ibov_meta = {
                "last_bar": str(s.index[-1].date()) if len(s) else None,
                "intraday": False,
                "from_cache": True,
            }
            return s

        try:
            logger.info("Baixando IBOVESPA (%s → %s)...", start, end)
            close, meta = self._download_close(ibov_ticker, start, end)
            if close is None or close.empty:
                logger.error("yfinance retornou DataFrame vazio para ^BVSP")
                return None

            returns = close.pct_change().dropna()
            returns.name = "ibovespa"
            self.ibov_meta = meta

            if meta["intraday"]:
                logger.warning(
                    "IBOV: candle de %s é intradiário (pregão não consolidado) — "
                    "retorno do período usa print ao vivo; não vai para o cache.",
                    meta["last_bar"],
                )
            else:
                self.cache.set(cache_key, {
                    "dates":   [str(d.date()) for d in returns.index],
                    "returns": [round(float(v), 8) for v in returns.values],
                })
            logger.debug("IBOV: %d dias de retornos coletados", len(returns))
            return returns

        except Exception as exc:
            logger.error("Falha ao baixar IBOVESPA: %s", exc)
            return None

    @staticmethod
    def _download_close(
        ticker: str, start: date, end: date,
    ) -> tuple[Optional[pd.Series], dict]:
        df = yf.download(
            ticker,
            start=str(start),
            end=str(end + timedelta(days=1)),  # yfinance: end é exclusivo
            progress=False,
            auto_adjust=True,
        )
        if df is None or df.empty:
            return None, {}
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return clean_close(df)

    def check_against_etf(
        self,
        start_date: str | date | datetime,
        end_date: Optional[str | date | datetime],
        ibov_return: Optional[float],
    ) -> dict:
        """
        Confere o retorno do IBOV no período contra o BOVA11 na mesma janela.

        Não altera o número usado no backtest — só registra o desvio e loga
        warning quando passa de IBOV_ETF_TOLERANCE.
        """
        start = _parse_date(start_date)
        end = _parse_date(end_date) if end_date else date.today()
        out: dict = {
            "etf": IBOV_ETF_TICKER,
            "tolerance_pp": IBOV_ETF_TOLERANCE * 100,
        }
        if ibov_return is None:
            out["error"] = "sem retorno do IBOV"
            return out
        try:
            close, meta = self._download_close(
                IBOV_ETF_TICKER, start - timedelta(days=10), end,
            )
        except Exception as exc:
            out["error"] = str(exc)
            return out
        etf_ret = period_return_from_close(close, start, end) if close is not None else None
        if etf_ret is None:
            out["error"] = "sem preços do ETF na janela"
            return out
        # yfinance às vezes fica sem o candle do ETF em um dia que o índice
        # tem; aí a janela do ETF termina antes e a comparação não vale.
        etf_last = last_bar_on_or_before(close, end)
        ibov_last = self.ibov_meta.get("last_bar")
        if ibov_last and str(etf_last) != ibov_last:
            out["error"] = f"ETF termina em {etf_last}, IBOV em {ibov_last}"
            return out

        gap = float(ibov_return) - etf_ret
        out.update({
            "etf_return": round(etf_ret, 6),
            "gap_pp": round(gap * 100, 4),
            "consistent": bool(abs(gap) <= IBOV_ETF_TOLERANCE),
            "etf_intraday": bool(meta.get("intraday", False)),
        })
        if not out["consistent"]:
            logger.warning(
                "IBOV %+.2f%% vs %s %+.2f%% em %s→%s: desvio de %.2fpp acima "
                "da tolerância (%.1fpp). Verificar dado/período do benchmark.",
                ibov_return * 100, IBOV_ETF_TICKER, etf_ret * 100, start, end,
                gap * 100, IBOV_ETF_TOLERANCE * 100,
            )
        return out

    # Coleta BCB — SELIC e CDI com retry

    def _fetch_bcb_with_retry(
        self,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """
        SGS 11/12 com até 3 tentativas (esperas de 1s e 2s). Se tudo falhar,
        devolve DataFrame vazio e SELIC/CDI ficam NaN — o IBOV segue.
        """
        cache_key = f"bcb_sgs_{start}_{end}"
        cached = self.cache.get(cache_key)
        if cached:
            df = pd.DataFrame(cached)
            df.index = pd.to_datetime(df.index)
            logger.debug("BCB SGS: cache hit (%d dias)", len(df))
            return df

        last_exc: Optional[Exception] = None
        for attempt in range(_BCB_MAX_RETRIES):
            try:
                df = self._fetch_bcb_raw(start, end)
                df_cache = df.copy()
                df_cache.index = df_cache.index.strftime("%Y-%m-%d")
                self.cache.set(cache_key, df_cache.to_dict())
                logger.info(
                    "BCB SGS: %d dias coletados (tentativa %d)", len(df), attempt + 1
                )
                return df

            except Exception as exc:
                last_exc = exc
                wait = _BCB_BACKOFF_BASE ** attempt
                logger.warning(
                    "BCB SGS falhou (tentativa %d/%d): %s. Aguardando %ds...",
                    attempt + 1, _BCB_MAX_RETRIES, exc, wait,
                )
                if attempt < _BCB_MAX_RETRIES - 1:
                    time.sleep(wait)

        logger.error(
            "BCB SGS indisponível após %d tentativas: %s. "
            "SELIC/CDI serão NaN nesta execução.",
            _BCB_MAX_RETRIES, last_exc,
        )
        return pd.DataFrame(columns=["selic", "cdi"])

    def _fetch_bcb_raw(self, start: date, end: date) -> pd.DataFrame:
        """SGS via python-bcb, convertido para retorno decimal diário."""
        try:
            from bcb import sgs as bcb_sgs  # import lazy: evita crash se não instalado
        except ImportError:
            raise BenchmarkError(
                "python-bcb não instalado. Execute: pip install python-bcb"
            )

        logger.debug("Buscando BCB SGS séries 11 e 12 (%s → %s)", start, end)
        # python-bcb sgs.get() espera {nome: series_id}
        raw_df: pd.DataFrame = bcb_sgs.get(
            {
                "selic": BCB_SELIC_SERIES,
                "cdi":   BCB_CDI_SERIES,
            },
            start=str(start),
            end=str(end),
        )

        if raw_df is None or raw_df.empty:
            raise BenchmarkError("BCB SGS retornou DataFrame vazio.")

        raw_df.index = pd.to_datetime(raw_df.index).tz_localize(None)
        raw_df = raw_df.sort_index()

        for col in ["selic", "cdi"]:
            if col not in raw_df.columns:
                raw_df[col] = np.nan
                continue
            raw_df[col] = self._convert_to_daily_return(raw_df[col], col_name=col)

        return raw_df

    @staticmethod
    def _convert_to_daily_return(series: pd.Series, col_name: str = "") -> pd.Series:
        """
        Taxa do BCB → retorno decimal diário. SGS 11/12 hoje vêm em % a.d.;
        o formato é inferido pela mediana:
          mediana > 1 → % a.a. → (1 + taxa/100)^(1/252) - 1   (DU/252, composto)
          mediana ≤ 1 → % a.d. → taxa/100
        Se o resultado sair da faixa plausível, tenta o outro formato.
        """
        valid = series.dropna()
        if valid.empty:
            return series

        # mediana: um outlier não inverte o formato
        median_val = float(valid.median())
        is_annual = median_val > 1.0

        def _apply_conversion(s: pd.Series, annual: bool) -> pd.Series:
            if annual:
                return (1.0 + s / 100.0) ** (1.0 / _DU_YEAR) - 1.0
            return s / 100.0

        daily = _apply_conversion(series, is_annual)

        daily_median = float(daily.dropna().median())
        is_plausible = _BCB_DAILY_RATE_MIN <= daily_median <= _BCB_DAILY_RATE_MAX

        if not is_plausible:
            logger.error(
                "BCB %s: taxa diária suspeita após conversão "
                "(mediana=%.6f = %.4f%% a.d., formato='%s'). "
                "Esperado entre %.5f e %.5f. Tentando formato alternativo.",
                col_name, daily_median, daily_median * 100,
                "anual" if is_annual else "diário",
                _BCB_DAILY_RATE_MIN, _BCB_DAILY_RATE_MAX,
            )
            alt_daily = _apply_conversion(series, not is_annual)
            alt_median = float(alt_daily.dropna().median())
            if _BCB_DAILY_RATE_MIN <= alt_median <= _BCB_DAILY_RATE_MAX:
                logger.warning(
                    "BCB %s: formato alternativo ('%s') é plausível "
                    "(mediana=%.6f = %.4f%% a.d.). Usando-o.",
                    col_name, "anual" if not is_annual else "diário",
                    alt_median, alt_median * 100,
                )
                daily = alt_daily
                is_annual = not is_annual
            else:
                logger.error(
                    "BCB %s: nenhum formato produz taxa plausível. "
                    "Dados podem estar corrompidos. Mantendo conversão original.",
                    col_name,
                )

        logger.debug(
            "BCB %s: formato '%s' (mediana_entrada=%.4f), "
            "retorno diário médio=%.6f%%",
            col_name, "anual" if is_annual else "diário",
            median_val, float(daily.dropna().mean()) * 100,
        )
        return daily

    def _align_and_merge(
        self,
        ibov_returns: pd.Series,
        bcb_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Left join no calendário do IBOV; lacunas do BCB (pregão sem dado no
        SGS) com ffill de até _BCB_FFILL_LIMIT dias. Inner join perderia
        pregões reais.
        """
        ibov_idx = pd.to_datetime(ibov_returns.index).normalize()
        ibov_series = ibov_returns.copy()
        ibov_series.index = ibov_idx

        df = pd.DataFrame({"ibovespa": ibov_series})

        if bcb_df.empty:
            df["selic"] = np.nan
            df["cdi"]   = np.nan
            logger.warning("BCB DataFrame vazio — SELIC/CDI serão NaN.")
            return df

        bcb_idx = pd.to_datetime(bcb_df.index).normalize()
        bcb_aligned = bcb_df.copy()
        bcb_aligned.index = bcb_idx

        for col in ["selic", "cdi"]:
            if col in bcb_aligned.columns:
                df[col] = bcb_aligned[col].reindex(df.index)
            else:
                df[col] = np.nan

        # Forward-fill para lacunas curtas (feriados municipais/estaduais)
        # O retorno do dia seguinte carrega a taxa do dia em que o BCB não reportou
        n_gaps_before = df[["selic", "cdi"]].isna().sum().sum()
        df[["selic", "cdi"]] = df[["selic", "cdi"]].ffill(limit=_BCB_FFILL_LIMIT)
        n_gaps_after = df[["selic", "cdi"]].isna().sum().sum()

        if n_gaps_before > n_gaps_after:
            logger.debug(
                "BCB: %d lacunas preenchidas por forward-fill (%d restantes)",
                n_gaps_before - n_gaps_after, n_gaps_after,
            )
        if n_gaps_after > 0:
            logger.warning(
                "BCB: %d lacunas NaN remanescentes após forward-fill "
                "(possível início de série ou feriado longo)",
                n_gaps_after,
            )

        # Remover a primeira linha (sempre NaN por causa do pct_change do IBOV)
        df = df.dropna(subset=["ibovespa"])

        self._log_alignment_summary(df)
        return df

    @staticmethod
    def _log_alignment_summary(df: pd.DataFrame) -> None:
        if df.empty:
            return
        logger.info(
            "Benchmarks alinhados: %d pregões | "
            "IBOV: %.1f%% dados | SELIC: %.1f%% dados | CDI: %.1f%% dados",
            len(df),
            df["ibovespa"].notna().mean() * 100,
            df["selic"].notna().mean() * 100 if "selic" in df.columns else 0,
            df["cdi"].notna().mean() * 100 if "cdi" in df.columns else 0,
        )


def get_benchmark_returns(
    start_date: str | date | datetime,
    end_date: Optional[str | date | datetime] = None,
    cache: Optional[CacheManager] = None,
) -> pd.DataFrame:
    """Atalho para BenchmarkManager().get_returns()."""
    return BenchmarkManager(cache=cache).get_returns(start_date, end_date)


def get_ibov_prices(
    start_date: str | date | datetime,
    end_date: Optional[str | date | datetime] = None,
    cache: Optional[CacheManager] = None,
) -> pd.Series:
    """Fechamentos do ^BVSP (preços, não retornos)."""
    start = _parse_date(start_date)
    end   = _parse_date(end_date) if end_date else date.today()
    cache = cache or CacheManager()

    cache_key = f"ibov_prices_{start}_{end}"
    cached = cache.get(cache_key)
    if cached:
        s = pd.Series(cached["prices"], index=pd.to_datetime(cached["dates"]))
        return s

    ibov_ticker = BENCHMARKS["ibovespa"]["ticker"]
    try:
        prices, meta = BenchmarkManager._download_close(ibov_ticker, start, end)
        if prices is None or prices.empty:
            logger.error("Sem preços do IBOV para o período solicitado.")
            return pd.Series(dtype=float)
        prices.name = "ibovespa"

        if not meta.get("intraday"):
            cache.set(cache_key, {
                "dates":  [str(d.date()) for d in prices.index],
                "prices": [round(float(v), 2) for v in prices.values],
            })
        return prices

    except Exception as exc:
        logger.error("Falha ao baixar preços do IBOV: %s", exc)
        return pd.Series(dtype=float)


def _parse_date(d: str | date | datetime) -> date:
    """Converte str, date ou datetime para date."""
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d), "%Y-%m-%d").date()
