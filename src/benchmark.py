"""
benchmark.py — Módulo 4

Busca e alinha dados de referência para comparação de performance e cálculo de alpha.

Fontes:
  IBOVESPA: yfinance (^BVSP)   — calendar anchor (dias de pregão da B3)
  SELIC:    BCB SGS Série 11   — taxa anualizada % a.a. → retorno decimal diário
  CDI:      BCB SGS Série 12   — taxa anualizada % a.a. → retorno decimal diário

Nota técnica sobre as séries BCB:
  Séries 11 e 12 retornam a taxa ANUALIZADA em % (ex: 13.75 = 13,75% a.a.).
  Para converter para retorno diário composto (convenção DU/252 da B3):
    r_diário = (1 + taxa_anual/100)^(1/252) - 1
  Essa convenção é a padrão do mercado brasileiro (ANBIMA, B3).

Output de get_returns():
  pd.DataFrame — index=DatetimeIndex (dias de pregão), columns=['ibovespa','selic','cdi']
  Valores em retorno decimal diário (0.01 = 1%). NÃO base-100.
  A normalização base-100 é feita pelo chart_generator.py conforme a janela.

Calendário:
  O IBOVESPA é usado como referência de dias úteis (pregão B3).
  Merge LEFT: todas as datas do IBOV são mantidas; BCB é forward-filled
  para cobrir dias em que o SGS tem lacunas (feriados municipais, etc.).
"""

import logging
import time
from datetime import date, datetime, timedelta
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
from src.data_collector import CacheManager

logger = logging.getLogger(__name__)

# Dias úteis por ano — convenção DU/252 (padrão ANBIMA/B3)
_DU_YEAR = 252

# Parâmetros de retry para o SGS/BCB
_BCB_MAX_RETRIES = 3
_BCB_BACKOFF_BASE = 2  # segundos (exponencial: 2s, 4s, 8s)

# Máximo de dias de forward-fill para lacunas do BCB
_BCB_FFILL_LIMIT = 3


class BenchmarkError(Exception):
    """Falha irrecuperável na coleta de dados de benchmark."""


class BenchmarkManager:
    """
    Centraliza a coleta e alinhamento dos benchmarks de mercado.

    Uso:
        bm = BenchmarkManager()
        df_returns = bm.get_returns(start_date="2024-01-01", end_date="2024-12-31")

        # Retorna DataFrame com retornos diários decimais:
        #         ibovespa     selic       cdi
        # date
        # 2024-01-02  0.00823  0.000476  0.000476
        # 2024-01-03 -0.00412  0.000476  0.000476
        # ...

        # Para base-100 (chart_generator):
        cumulative = (1 + df_returns).cumprod() * 100
    """

    def __init__(self, cache: Optional[CacheManager] = None):
        self.cache = cache or CacheManager(
            cache_dir=CACHE_DIR_PATH, ttl_hours=CACHE_TTL_HOURS
        )

    # ═══════════════════════════════════════════════════════════════════════
    # API pública
    # ═══════════════════════════════════════════════════════════════════════

    def get_returns(
        self,
        start_date: str | date | datetime,
        end_date: Optional[str | date | datetime] = None,
    ) -> pd.DataFrame:
        """
        Retorna DataFrame de retornos diários decimais para o período solicitado.

        Args:
            start_date: data de início (inclusive). Aceita str "YYYY-MM-DD", date ou datetime.
            end_date:   data de fim (inclusive). Default: hoje.

        Returns:
            pd.DataFrame com:
              - index:   DatetimeIndex (dias de pregão da B3, sem fins de semana)
              - columns: ['ibovespa', 'selic', 'cdi']
              - valores: float64, retorno decimal diário (0.01 = 1%)
              - NaN:     onde não há dado para aquela data/benchmark

        Raises:
            BenchmarkError: se IBOVESPA falhar (é o anchor do calendário).
        """
        start = _parse_date(start_date)
        end   = _parse_date(end_date) if end_date else date.today()

        # Adiciona margem de 5 dias úteis para garantir lookback completo após pct_change
        fetch_start = start - timedelta(days=7)

        cache_key = f"benchmark_returns_{start}_{end}"
        cached = self.cache.get(cache_key)
        if cached:
            df = pd.DataFrame(cached)
            df.index = pd.to_datetime(df.index)
            logger.info("Benchmarks: cache hit (%d dias)", len(df))
            return df

        # ── Coleta paralela (independente por fonte) ──────────────────────
        ibov_returns = self._fetch_ibov(fetch_start, end)
        if ibov_returns is None or ibov_returns.empty:
            raise BenchmarkError(
                "Falha ao obter IBOVESPA (^BVSP). "
                "Verifique conexão ou disponibilidade do yfinance."
            )

        bcb_df = self._fetch_bcb_with_retry(fetch_start, end)

        # ── Alinhamento de calendário ─────────────────────────────────────
        df = self._align_and_merge(ibov_returns, bcb_df)

        # Cortar para a janela solicitada (após pct_change que consome 1 linha)
        df = df.loc[df.index.date >= start]

        # Garantir tipos float64
        df = df.astype("float64")

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
        """
        Conveniência: retorna retornos acumulados em base `base` (padrão 100).

        Fórmula: base × ∏(1 + r_t) para cada benchmark.
        O primeiro dia sempre começa em `base`.

        Usado pelo backtester e pelo chart_generator.
        """
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
        """
        Retorno total (decimal) de cada benchmark no período.

        Fórmula: ∏(1 + r_t) - 1.

        Exemplo:
            {'ibovespa': 0.083, 'selic': 0.064, 'cdi': 0.063}
            → IBOV +8.3%, SELIC +6.4%, CDI +6.3% no período
        """
        daily = self.get_returns(start_date, end_date)
        return (1 + daily).prod() - 1

    # ═══════════════════════════════════════════════════════════════════════
    # Coleta IBOVESPA
    # ═══════════════════════════════════════════════════════════════════════

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
            return s

        try:
            logger.info("Baixando IBOVESPA (%s → %s)...", start, end)
            df = yf.download(
                ibov_ticker,
                start=str(start),
                end=str(end + timedelta(days=1)),  # yfinance: end é exclusivo
                progress=False,
                auto_adjust=True,
            )
            if df is None or df.empty:
                logger.error("yfinance retornou DataFrame vazio para ^BVSP")
                return None

            # Garantir índice DatetimeIndex sem timezone
            df.index = pd.to_datetime(df.index).tz_localize(None)

            # Retornos diários: (Close_t / Close_{t-1}) - 1
            # yfinance >=0.2.x retorna MultiIndex — achatar para Series
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            returns = close.pct_change().dropna()
            returns.name = "ibovespa"

            self.cache.set(cache_key, {
                "dates":   [str(d.date()) for d in returns.index],
                "returns": [round(float(v), 8) for v in returns.values],
            })
            logger.debug("IBOV: %d dias de retornos coletados", len(returns))
            return returns

        except Exception as exc:
            logger.error("Falha ao baixar IBOVESPA: %s", exc)
            return None

    # ═══════════════════════════════════════════════════════════════════════
    # Coleta BCB — SELIC e CDI com retry
    # ═══════════════════════════════════════════════════════════════════════

    def _fetch_bcb_with_retry(
        self,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """
        Busca as Séries 11 (SELIC) e 12 (CDI) do SGS/BCB com retry exponencial.

        Retry policy:
          Tentativa 1: imediata
          Tentativa 2: aguarda 2s
          Tentativa 3: aguarda 4s
          Falha final: retorna DataFrame vazio (scores de SELIC/CDI serão NaN)

        Prefere não travar toda a execução por falha do BCB — IBOV continua.
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
                wait = _BCB_BACKOFF_BASE ** attempt  # 1s, 2s, 4s
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
        """
        Chamada à API do SGS via python-bcb e conversão para retornos diários.

        Série 11 e 12 retornam a taxa ANUALIZADA em % (ex: 13.75 = 13,75% a.a.).
        Conversão DU/252: r_diário = (1 + taxa_anual/100)^(1/252) - 1

        Por que DU/252 e não linear?
          O mercado brasileiro usa capitalização composta (convenção ANBIMA).
          Divisão simples por 252 subestima o juro composto de longo prazo.
        """
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

        # ── Detecção automática do formato da taxa ────────────────────────
        # Heurística robusta: taxa anual do SELIC raramente é < 1% ou > 50%
        for col in ["selic", "cdi"]:
            if col not in raw_df.columns:
                raw_df[col] = np.nan
                continue
            raw_df[col] = self._convert_to_daily_return(raw_df[col], col_name=col)

        return raw_df

    @staticmethod
    def _convert_to_daily_return(series: pd.Series, col_name: str = "") -> pd.Series:
        """
        Converte taxa do BCB para retorno decimal diário.

        Lógica de detecção de formato:
          median > 1.0  → taxa anualizada em % (ex: 13.75)
                          r = (1 + taxa/100)^(1/252) - 1
          median ≤ 1.0  → taxa já diária em % (ex: 0.0487)
                          r = taxa/100

        O threshold 1.0 funciona porque:
          - Taxa anual SELIC/CDI: histórico BR entre 2% e 45% → sempre > 1
          - Taxa diária: 13.75% a.a. → 0.0487% a.d. → sempre < 1
        """
        valid = series.dropna()
        if valid.empty:
            return series

        median_val = float(valid.median())

        if median_val > 1.0:
            # Formato: % ao ano (ex: 13.75) → DU/252
            annual = series / 100.0
            daily  = (1.0 + annual) ** (1.0 / _DU_YEAR) - 1.0
            logger.debug(
                "%s: detectado formato anual (mediana=%.2f%% a.a.). "
                "Retorno diário médio: %.5f%%",
                col_name, median_val, daily.mean() * 100,
            )
        else:
            # Formato: % ao dia (ex: 0.0487) → dividir por 100
            daily = series / 100.0
            logger.debug(
                "%s: detectado formato diário (mediana=%.5f%% a.d.).",
                col_name, median_val,
            )

        return daily

    # ═══════════════════════════════════════════════════════════════════════
    # Alinhamento de calendário
    # ═══════════════════════════════════════════════════════════════════════

    def _align_and_merge(
        self,
        ibov_returns: pd.Series,
        bcb_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Alinha IBOV (anchor) com SELIC/CDI usando LEFT JOIN.

        Estratégia:
          1. IBOV define o índice de referência (dias de pregão da B3).
          2. BCB é reindexado para os dias do IBOV.
          3. NaN do BCB após reindex → forward-fill (≤ 3 dias).
             Cobertura de: feriados municipais que o BCB não registrou
             mas a B3 operou normalmente.
          4. NaN remanescentes (início de série, feriados longos) → mantidos.

        Por que LEFT e não INNER?
          INNER descartaria dias de pregão onde o BCB tem pequenas lacunas,
          perdendo retornos reais do mercado. LEFT preserva todos os pregões.
        """
        # Garantir que o índice do IBOV seja DatetimeIndex normalizado
        ibov_idx = pd.to_datetime(ibov_returns.index).normalize()
        ibov_series = ibov_returns.copy()
        ibov_series.index = ibov_idx

        # Criar DataFrame base com o calendário do IBOV
        df = pd.DataFrame({"ibovespa": ibov_series})

        if bcb_df.empty:
            df["selic"] = np.nan
            df["cdi"]   = np.nan
            logger.warning("BCB DataFrame vazio — SELIC/CDI serão NaN.")
            return df

        # Normalizar índice do BCB
        bcb_idx = pd.to_datetime(bcb_df.index).normalize()
        bcb_aligned = bcb_df.copy()
        bcb_aligned.index = bcb_idx

        # LEFT JOIN: mantém todas as datas do IBOV
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


# ═══════════════════════════════════════════════════════════════════════════
# Funções de conveniência para uso em outros módulos
# ═══════════════════════════════════════════════════════════════════════════

def get_benchmark_returns(
    start_date: str | date | datetime,
    end_date: Optional[str | date | datetime] = None,
    cache: Optional[CacheManager] = None,
) -> pd.DataFrame:
    """
    Função de conveniência para main.py e backtester.py.

    Returns:
        pd.DataFrame: retornos diários decimais (ibovespa, selic, cdi).
    """
    return BenchmarkManager(cache=cache).get_returns(start_date, end_date)


def get_ibov_prices(
    start_date: str | date | datetime,
    end_date: Optional[str | date | datetime] = None,
    cache: Optional[CacheManager] = None,
) -> pd.Series:
    """
    Retorna série de PREÇOS (não retornos) do IBOVESPA.
    Necessário para o scoring_engine.py calcular beta e alpha.

    Returns:
        pd.Series: preços de fechamento ajustados (^BVSP).
    """
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
        df = yf.download(
            ibov_ticker,
            start=str(start),
            end=str(end + timedelta(days=1)),
            progress=False,
            auto_adjust=True,
        )
        if df is None or df.empty:
            logger.error("Sem preços do IBOV para o período solicitado.")
            return pd.Series(dtype=float)

        df.index = pd.to_datetime(df.index).tz_localize(None)
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        prices = close.dropna()
        prices.name = "ibovespa"

        cache.set(cache_key, {
            "dates":  [str(d.date()) for d in prices.index],
            "prices": [round(float(v), 2) for v in prices.values],
        })
        return prices

    except Exception as exc:
        logger.error("Falha ao baixar preços do IBOV: %s", exc)
        return pd.Series(dtype=float)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers internos
# ═══════════════════════════════════════════════════════════════════════════

def _parse_date(d: str | date | datetime) -> date:
    """Converte str, date ou datetime para date."""
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d), "%Y-%m-%d").date()
