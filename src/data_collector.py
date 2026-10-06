"""
Coleta:
  1. brapi.dev — quotes em batch de 10: preço, P/L, DY, volume
  2. yfinance  — o que o plano free da brapi não dá (P/VP, ROE, ROIC,
                 Dívida/EBITDA) e o histórico OHLCV de 1 ano
  3. cache JSON local, TTL 24h por chave

Output público:
  load_data() → (df_fundamentals: pd.DataFrame, df_prices: pd.DataFrame)

  df_fundamentals colunas:
    ticker, nome, setor, subsetor, liquidez_minima_MM, norm_method,
    pl, pvp, roe, roic, divida_ebitda, dividend_yield,
    beta, avg_volume_30d, current_price, market_cap,
    week52_high, week52_low, data_source

  df_prices:
    index = date (DatetimeIndex), columns = tickers (preço de fechamento ajustado)
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from src.config import (
    BRAPI_BASE_URL,
    BRAPI_RATE_LIMIT,
    BRAPI_TIMEOUT,
    CACHE_DIR_PATH,
    CACHE_TTL_HOURS,
    MAX_PLAUSIBLE_DY,
    UNIVERSE_FILE,
    YFINANCE_TIMEOUT,
)

logger = logging.getLogger(__name__)

# Brapi suporta /quote/T1,T2,...,TN em um único request.
# Batch de 10 reduz chamadas de ~350 para ~35, mantendo-se dentro do rate limit.
BATCH_SIZE = 10

# Se mais de 20% dos tickers falharem em ambas as fontes, abortamos
MAX_FAILURES_RATIO = 0.20

# Setores onde Dívida/EBITDA é conceitualmente inaplicável (modelo bancário)
FINANCIAL_SECTORS = {"Financeiro e Outros"}


class DataCollectionError(Exception):
    """Falha irrecuperável na coleta de dados."""


# CacheManager — leitura/escrita de JSON com TTL
class CacheManager:
    """
    Cache de arquivos JSON em disco com TTL configurável.

    Cada chave vira um arquivo {key}.json no cache_dir.
    Validade verificada pelo mtime do arquivo vs. TTL.
    """

    def __init__(
        self,
        cache_dir: Path = CACHE_DIR_PATH,
        ttl_hours: int = CACHE_TTL_HOURS,
    ):
        self.cache_dir = Path(cache_dir)
        self.ttl = timedelta(hours=ttl_hours)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace(":", "_").replace(",", "-")
        return self.cache_dir / f"{safe}.json"

    def is_valid(self, key: str) -> bool:
        p = self._path(key)
        if not p.exists():
            return False
        age = datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)
        return age < self.ttl

    def get(self, key: str) -> Optional[dict | list]:
        if not self.is_valid(key):
            return None
        try:
            with open(self._path(key), encoding="utf-8") as f:
                data = json.load(f)
            logger.debug("Cache HIT  → %s", key)
            return data
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, key: str, data: dict | list) -> None:
        try:
            with open(self._path(key), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, default=str)
            logger.debug("Cache WRITE → %s", key)
        except OSError as e:
            logger.warning("Falha ao escrever cache %s: %s", key, e)

    def invalidate(self, key: str) -> None:
        p = self._path(key)
        if p.exists():
            p.unlink(missing_ok=True)


# BrapiClient — wrapper HTTP com retry e batch
class BrapiClient:
    """
    Acessa brapi.dev para quotes em lote e histórico de preços.

    Rate limiting: sleep(rate_limit) entre cada request de batch.
    Estratégia de falha:
      1. Tenta batch completo de BATCH_SIZE tickers.
      2. Se o batch falha, faz retry ticker-a-ticker.
      3. Registra falhos mas não interrompe a coleta.
    """

    def __init__(
        self,
        base_url: str = BRAPI_BASE_URL,
        timeout: int = BRAPI_TIMEOUT,
        rate_limit: float = BRAPI_RATE_LIMIT,
        token: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json", "User-Agent": "recomendador-b3/1.0"})
        self._params = {"token": token} if token else {}

    def _get(self, path: str, extra_params: Optional[dict] = None) -> dict:
        params = {**self._params, **(extra_params or {})}
        url = f"{self.base_url}{path}"
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def _fetch_batch(self, tickers: list[str]) -> dict[str, dict]:
        """Um único request para até BATCH_SIZE tickers. Retorna dict ticker→raw."""
        raw = self._get(f"/quote/{','.join(tickers)}")
        return {
            item["symbol"]: item
            for item in raw.get("results", [])
            if item.get("symbol")
        }

    def fetch_quotes_all(
        self,
        tickers: list[str],
        cache: CacheManager,
    ) -> tuple[dict[str, dict], list[str]]:
        """
        Busca quotes para todos os tickers em batches.

        Falha parcial:
          - Batch falha → retry individual com sleep entre cada um
          - Individual falha → ticker vai para lista `failed`
          - Tickers não retornados pelo brapi (sem erro HTTP) → também `failed`

        Returns:
          all_data: dict[ticker, raw_brapi_dict] para tickers bem-sucedidos
          failed:   list[ticker] sem dados
        """
        all_data: dict[str, dict] = {}
        failed: list[str] = []
        batches = [tickers[i : i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]

        for batch_idx, batch in enumerate(batches):
            cache_key = f"brapi_batch_{'_'.join(sorted(batch))}"
            cached = cache.get(cache_key)
            if cached:
                all_data.update(cached)
                logger.debug("Batch %d/%d: cache hit (%d tickers)", batch_idx + 1, len(batches), len(cached))
                continue

            logger.info("Batch %d/%d: requisitando %s...", batch_idx + 1, len(batches), batch)
            try:
                batch_data = self._fetch_batch(batch)
                all_data.update(batch_data)
                cache.set(cache_key, batch_data)

                missing = [t for t in batch if t not in batch_data]
                if missing:
                    logger.warning("Brapi não retornou dados para %s (sem erro HTTP)", missing)
                    failed.extend(missing)

            except requests.HTTPError as e:
                logger.warning("Batch %s → HTTP %s — retry individual", batch, e.response.status_code)
                self._retry_individual(batch, all_data, failed, cache)

            except (requests.ConnectionError, requests.Timeout) as e:
                logger.warning("Batch %s → timeout/conexão (%s) — retry individual", batch, e)
                self._retry_individual(batch, all_data, failed, cache)

            time.sleep(self.rate_limit)

        logger.info("Brapi: %d coletados, %d falhos", len(all_data), len(failed))
        return all_data, failed

    def _retry_individual(
        self,
        tickers: list[str],
        all_data: dict,
        failed: list,
        cache: CacheManager,
    ) -> None:
        for ticker in tickers:
            cache_key = f"brapi_single_{ticker}"
            cached = cache.get(cache_key)
            if cached:
                all_data[ticker] = cached
                continue
            try:
                time.sleep(self.rate_limit * 2)  # back-off maior no retry
                data = self._fetch_batch([ticker])
                if ticker in data:
                    all_data[ticker] = data[ticker]
                    cache.set(cache_key, data[ticker])
                    logger.debug("Retry individual OK: %s", ticker)
                else:
                    logger.error("Retry individual sem dados: %s", ticker)
                    failed.append(ticker)
            except Exception as exc:
                logger.error("Retry individual falhou para %s: %s", ticker, exc)
                failed.append(ticker)

    def fetch_historical(
        self,
        ticker: str,
        range_: str = "1y",
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """
        Retorna DataFrame OHLCV para o ticker.
        Brapi retorna historicalDataPrice como lista de dicts com timestamp Unix.
        """
        try:
            raw = self._get(
                f"/quote/{ticker}",
                extra_params={"range": range_, "interval": interval, "fundamental": "false"},
            )
            results = raw.get("results", [])
            if not results:
                return None
            hist = results[0].get("historicalDataPrice", [])
            if not hist:
                return None

            df = pd.DataFrame(hist)
            # brapi retorna timestamp Unix em segundos
            df["date"] = pd.to_datetime(df["date"], unit="s", utc=True).dt.tz_localize(None)
            df = df.set_index("date").sort_index()
            df = df.rename(columns={
                "open":          "Open",
                "high":          "High",
                "low":           "Low",
                "close":         "Close",
                "volume":        "Volume",
                "adjustedClose": "Adj Close",
            })
            close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
            df = df[df[close_col].notna() & (df[close_col] > 0)]
            return df[[c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]]

        except Exception as exc:
            logger.warning("Brapi histórico %s falhou: %s", ticker, exc)
            return None


# YFinanceClient — fallback para fundamentais e histórico
class YFinanceClient:
    """
    Usa yfinance como fonte secundária.
    Tickers brasileiros precisam de sufixo .SA (ex: PETR4.SA).

    Retorna fundamentais que o brapi free tier não oferece:
      priceToBook (P/VP), returnOnEquity (ROE),
      returnOnCapitalEmployed (ROIC), totalDebt + ebitda → Dívida/EBITDA
    """

    # Mapeamento yfinance key → nossa key interna
    _FIELD_MAP = {
        "trailingPE":               "pl",
        "forwardPE":                "_forward_pe",  # fallback quando trailing é NaN
        "priceToBook":              "pvp",
        "returnOnEquity":           "roe",
        "returnOnCapitalEmployed":  "roic",
        "dividendYield":            "dividend_yield",
        "beta":                     "beta",
        "averageVolume":            "avg_volume_30d",
        "marketCap":                "market_cap",
        "currentPrice":             "current_price",
        "fiftyTwoWeekHigh":         "week52_high",
        "fiftyTwoWeekLow":          "week52_low",
        "totalDebt":                "_total_debt",
        "ebitda":                   "_ebitda",
        "enterpriseToEbitda":       "ev_ebitda",  # múltiplo de valor alternativo a P/L
        "returnOnAssets":           "_roa",  # proxy ROIC quando ROCE indisponível
    }

    def fetch_info(self, ticker: str) -> dict:
        """
        Retorna dict com fundamentais ou {"_source": "failed"} em caso de erro.

        Sobre ROIC: yfinance expõe returnOnCapitalEmployed quando disponível.
        Caso indisponível, usamos returnOnAssets como proxy (conservador mas razoável).
        """
        try:
            info = yf.Ticker(f"{ticker}.SA").info or {}
            if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
                # yfinance retornou dict vazio ou ticker inexistente
                return {"_source": "failed", "ticker": ticker}

            result: dict = {"_source": "yfinance", "ticker": ticker}
            for yf_key, our_key in self._FIELD_MAP.items():
                val = info.get(yf_key)
                result[our_key] = val if val not in (None, "None", "N/A", 0.0) else None

            # Calcular Dívida/EBITDA a partir dos campos brutos
            debt  = result.pop("_total_debt", None)
            ebitda = result.pop("_ebitda", None)
            roa   = result.pop("_roa", None)
            fwd_pe = result.pop("_forward_pe", None)

            if debt is not None and ebitda and ebitda != 0:
                result["divida_ebitda"] = debt / ebitda
            else:
                result["divida_ebitda"] = None

            # ROIC: preferir ROCE; se None, usar ROA como proxy
            if result.get("roic") is None and roa is not None:
                result["roic"] = roa

            # P/L: fallback para forward P/L quando trailing está NaN.
            # Trailing P/L falta com frequência em ações que mudaram de regime
            # de lucro (ex.: PETR4 com lucro recente após anos de prejuízo) ou
            # quando o último ano fiscal teve evento não-recorrente.
            if result.get("pl") is None and fwd_pe is not None:
                try:
                    fwd_pe_f = float(fwd_pe)
                    if 0 < fwd_pe_f < 80:  # mesma faixa de plausibilidade
                        result["pl"] = fwd_pe_f
                except (TypeError, ValueError):
                    pass

            return result

        except Exception as exc:
            logger.error("yfinance info %s: %s", ticker, exc)
            return {"_source": "failed", "ticker": ticker}

    def fetch_history(self, ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Retorna DataFrame OHLCV (auto-adjusted) ou None."""
        try:
            df = yf.download(
                f"{ticker}.SA",
                period=period,
                progress=False,
                auto_adjust=True,
                timeout=YFINANCE_TIMEOUT,
            )
            if df is None or df.empty:
                return None
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as exc:
            logger.warning("yfinance histórico %s: %s", ticker, exc)
            return None

    def fetch_advanced_fundamentals(self, ticker: str) -> dict:
        """
        Coleta fundamentos derivados de demonstrações 3y + analyst recs.

        Returns dict com (todos opcionais):
          - net_margin_3y_avg     — média 3y de (Net Income / Total Revenue)
          - revenue_growth_3y     — (revenue_now / revenue_3y_ago)^(1/3) - 1
          - earnings_growth_3y    — análogo para net income (capped a [-1, +5])
          - asset_growth_yoy      — total_assets YoY mais recente (fator Investment)
          - free_cash_flow_ttm    — FCF anual mais recente
          - fcf_payout_ratio      — dividendos pagos / FCF (lower_is_better)
          - analyst_rec_score     — 1.0 (strong sell) a 5.0 (strong buy) média ponderada
          - analyst_rec_trend     — variação score recente (90d) vs anterior (180d)

        Falha silenciosa: qualquer chave que não puder ser computada vira None.
        yfinance bloqueia se chamado muito rápido — rate-limit é externo.
        """
        result: dict = {}
        try:
            yticker = yf.Ticker(f"{ticker}.SA")

            # Income statement (annual, last ~4 years)
            inc = getattr(yticker, "income_stmt", None)
            if inc is not None and not inc.empty:
                # yfinance retorna colunas mais novas → mais antigas (DataFrames TTM-first)
                years = list(inc.columns)
                if years:
                    # Net Income / Revenue series por ano
                    ni_row = _find_row(inc, ["Net Income", "NetIncome", "Net Income Common Stockholders"])
                    rev_row = _find_row(inc, ["Total Revenue", "TotalRevenue", "Operating Revenue", "OperatingRevenue"])

                    if ni_row is not None and rev_row is not None:
                        margins = []
                        for col in years[:4]:  # até 4 anos
                            ni = _to_float(ni_row.get(col))
                            rev = _to_float(rev_row.get(col))
                            if ni is not None and rev and rev > 0:
                                margins.append(ni / rev)
                        if margins:
                            result["net_margin_3y_avg"] = float(np.mean(margins))

                        # Earnings growth 3y CAGR
                        if ni_row is not None and len(years) >= 4:
                            ni_recent = _to_float(ni_row.get(years[0]))
                            ni_old    = _to_float(ni_row.get(years[3]))
                            if ni_recent is not None and ni_old is not None and ni_old > 0 and ni_recent > 0:
                                cagr = (ni_recent / ni_old) ** (1/3) - 1
                                # cap em [-100%, +500%] para outliers
                                result["earnings_growth_3y"] = float(max(-1.0, min(5.0, cagr)))

                        # Revenue growth 3y CAGR
                        if rev_row is not None and len(years) >= 4:
                            r_recent = _to_float(rev_row.get(years[0]))
                            r_old    = _to_float(rev_row.get(years[3]))
                            if r_recent is not None and r_old is not None and r_old > 0:
                                cagr = (r_recent / r_old) ** (1/3) - 1
                                result["revenue_growth_3y"] = float(max(-1.0, min(5.0, cagr)))

            # Balance sheet (asset growth YoY)
            bs = getattr(yticker, "balance_sheet", None)
            if bs is not None and not bs.empty:
                ta_row = _find_row(bs, ["Total Assets", "TotalAssets"])
                if ta_row is not None and len(bs.columns) >= 2:
                    ta_recent = _to_float(ta_row.get(bs.columns[0]))
                    ta_prev   = _to_float(ta_row.get(bs.columns[1]))
                    if ta_recent is not None and ta_prev is not None and ta_prev > 0:
                        result["asset_growth_yoy"] = float((ta_recent / ta_prev) - 1.0)

            # Cash flow (FCF + dividends paid)
            cf = getattr(yticker, "cashflow", None)
            if cf is not None and not cf.empty:
                fcf_row = _find_row(cf, ["Free Cash Flow", "FreeCashFlow"])
                if fcf_row is not None:
                    fcf_recent = _to_float(fcf_row.get(cf.columns[0]))
                    if fcf_recent is not None and fcf_recent != 0:
                        result["free_cash_flow_ttm"] = fcf_recent
                        # Dividendos pagos (sempre negativo no cashflow yfinance)
                        div_row = _find_row(cf, ["Cash Dividends Paid", "CashDividendsPaid", "Common Stock Dividend Paid"])
                        if div_row is not None:
                            div_paid = _to_float(div_row.get(cf.columns[0]))
                            if div_paid is not None and fcf_recent > 0:
                                # div_paid é negativo (saída) — usar abs
                                payout = abs(div_paid) / fcf_recent
                                # cap razoável; se > 5 provavelmente dado quebrado
                                if payout < 5.0:
                                    result["fcf_payout_ratio"] = float(payout)

            # Analyst recommendations
            try:
                rec = getattr(yticker, "recommendations", None)
                if rec is not None and not rec.empty:
                    # yfinance recommendations: colunas strongBuy/buy/hold/sell/strongSell,
                    # uma linha por mês (4 meses recentes tipicamente)
                    score = _analyst_score(rec.head(2))   # média dos 2 meses mais recentes
                    score_prev = _analyst_score(rec.iloc[2:5]) if len(rec) >= 3 else None
                    if score is not None:
                        result["analyst_rec_score"] = score
                    if score is not None and score_prev is not None:
                        result["analyst_rec_trend"] = float(score - score_prev)
            except Exception as exc:
                logger.debug("Analyst rec fetch %s: %s", ticker, exc)

            # Analyst price targets (forward-looking sinal)
            # Upside absoluto tem IC fraco (~0.02-0.04) e viés otimista em EM
            # (~+25%) (Brav-Lehavy 2003, Da-Schaumburg 2011); o que se usa é o
            # rank cross-sectional, feito no scoring_engine. Aqui só o bruto.
            try:
                apt = getattr(yticker, "analyst_price_targets", None)
                if isinstance(apt, dict) and apt:
                    mean_t = _to_float(apt.get("mean"))
                    median_t = _to_float(apt.get("median"))
                    current = _to_float(apt.get("current"))
                    high_t = _to_float(apt.get("high"))
                    low_t = _to_float(apt.get("low"))
                    # Usar median (mais robusto a outliers) com fallback para mean
                    target = median_t if median_t is not None else mean_t
                    if target is not None and current is not None and current > 0:
                        upside = (target / current) - 1.0
                        result["analyst_target_upside"] = float(upside)
                        if mean_t is not None:
                            result["analyst_target_mean"] = mean_t
                        # Dispersão como qualidade do sinal (alta dispersão = ruído)
                        if high_t is not None and low_t is not None and target > 0:
                            result["analyst_target_dispersion"] = float((high_t - low_t) / target)
            except Exception as exc:
                logger.debug("Analyst price target %s: %s", ticker, exc)

            # PEAD: Earnings dates + EAR proxy
            # Sinal: tickers em janela 5-60 dias úteis APÓS resultado com
            # surpresa positiva (CAR -1/+1 vs IBOV) tendem a continuar subindo.
            # Filtramos earnings_dates com cuidado para evitar look-ahead
            # (yfinance retorna datas FUTURAS misturadas com passadas).
            try:
                ed = getattr(yticker, "earnings_dates", None)
                if ed is not None and not ed.empty:
                    today_ts = pd.Timestamp.now().normalize()
                    # Filtrar SÓ datas passadas (eventos já reportados)
                    ed_idx = pd.to_datetime(ed.index).tz_localize(None) if getattr(ed.index, "tz", None) is not None else pd.to_datetime(ed.index)
                    past_mask = ed_idx < today_ts
                    past_events = ed[past_mask] if past_mask.any() else None
                    if past_events is not None and not past_events.empty:
                        # Pegar o mais recente
                        last_event_date = pd.to_datetime(past_events.index[0])
                        if last_event_date.tz is not None:
                            last_event_date = last_event_date.tz_localize(None)
                        days_since = (today_ts - last_event_date).days
                        result["last_earnings_date"] = str(last_event_date.date())
                        result["days_since_earnings"] = int(days_since)
                        # Surprise se yfinance fornecer (colunas variam: 'Surprise(%)', 'Reported EPS', etc.)
                        for col in past_events.columns:
                            cl = str(col).lower()
                            if "surprise" in cl and "%" in cl:
                                surprise_val = _to_float(past_events.iloc[0][col])
                                if surprise_val is not None:
                                    result["earnings_surprise_pct"] = surprise_val / 100.0 if abs(surprise_val) > 1 else surprise_val
                                break
            except Exception as exc:
                logger.debug("Earnings dates %s: %s", ticker, exc)

        except Exception as exc:
            logger.debug("yfinance advanced %s: %s", ticker, exc)

        return result


# Helpers de parsing e normalização de dados brapi
def _compute_yz_vol(df_ohlc: pd.DataFrame) -> Optional[float]:
    """
    Yang-Zhang volatility anualizada estimada de OHLC.

    σ²_YZ = σ²_overnight + k·σ²_open-to-close + (1-k)·σ²_RS

    Onde:
      σ²_overnight    = Var(log(O_t / C_{t-1}))   — gap noturno
      σ²_open-to-close = Var(log(C_t / O_t))      — movimento intraday
      σ²_RS (Rogers-Satchell) = mean(log(H/C)·log(H/O) + log(L/C)·log(L/O))
      k = 0.34 / (1.34 + (n+1)/(n-1))

    YZ é não-tendencioso (unbiased), eficiente sob drift não-nulo, e ~5×
    menor RMSE que close-to-close (Yang & Zhang 2000).

    Retorna vol anualizada (×√252). None se cálculo falhar.
    """
    if df_ohlc is None or df_ohlc.empty or len(df_ohlc) < 20:
        return None

    needed = {"Open", "High", "Low", "Close"}
    cols = set(df_ohlc.columns)
    if not needed.issubset(cols):
        return None

    try:
        o = df_ohlc["Open"].astype(float)
        h = df_ohlc["High"].astype(float)
        l = df_ohlc["Low"].astype(float)
        c = df_ohlc["Close"].astype(float)

        # Filtrar zeros/inválidos
        valid = (o > 0) & (h > 0) & (l > 0) & (c > 0)
        o, h, l, c = o[valid], h[valid], l[valid], c[valid]
        if len(c) < 20:
            return None

        # Log price relationships
        log_ho = np.log(h / o)
        log_lo = np.log(l / o)
        log_co = np.log(c / o)
        log_oc_prev = np.log(o / c.shift(1)).dropna()  # overnight
        log_cc = np.log(c / c.shift(1)).dropna()       # close-to-close (não usado direto)

        # Rogers-Satchell por dia
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        rs = rs.dropna()

        # Overnight
        overnight_var = log_oc_prev.var(ddof=1) if len(log_oc_prev) > 1 else 0.0
        # Open-to-close
        oc_var = log_co.var(ddof=1) if len(log_co) > 1 else 0.0
        # Rogers-Satchell média
        rs_mean = float(rs.mean()) if len(rs) > 0 else 0.0

        n = len(rs)
        if n < 2:
            return None
        k = 0.34 / (1.34 + (n + 1) / (n - 1))

        yz_var_daily = float(overnight_var + k * oc_var + (1 - k) * rs_mean)
        if yz_var_daily <= 0 or np.isnan(yz_var_daily):
            return None
        yz_vol_annual = float(np.sqrt(yz_var_daily * 252))
        return yz_vol_annual
    except Exception:
        return None


def _find_row(df: pd.DataFrame, candidates: list[str]):
    """Procura nas linhas do DataFrame o primeiro nome em `candidates` (case-insensitive)."""
    if df is None or df.empty:
        return None
    idx_lower = {str(i).lower().strip(): i for i in df.index}
    for cand in candidates:
        key = cand.lower().strip()
        if key in idx_lower:
            return df.loc[idx_lower[key]]
    return None


def _to_float(v) -> Optional[float]:
    """Conversão segura para float, tratando NaN/None/strings."""
    if v is None:
        return None
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _analyst_score(rec_df) -> Optional[float]:
    """
    Converte um sub-DataFrame de recommendations do yfinance em score 1-5.

    Mapeamento: strongSell=1, sell=2, hold=3, buy=4, strongBuy=5.
    Retorna média ponderada pelo número de analistas.
    """
    if rec_df is None or rec_df.empty:
        return None
    mapping = {"strongBuy": 5, "buy": 4, "hold": 3, "sell": 2, "strongSell": 1}
    total_weight = 0.0
    weighted_sum = 0.0
    for col, val in mapping.items():
        if col in rec_df.columns:
            count = rec_df[col].sum()
            if count > 0:
                weighted_sum += val * count
                total_weight += count
    return float(weighted_sum / total_weight) if total_weight > 0 else None


def _parse_brapi_quote(raw: dict) -> dict:
    """
    Extrai e renomeia campos de um item do endpoint /quote da brapi.

    Atenção ao DY: brapi às vezes retorna 9.12 (porcentagem) e às vezes 0.0912.
    Regra: se valor > 1.0, assumir que está em % e dividir por 100.
    """
    dy = raw.get("dividendYield")
    if dy is not None and dy > 1.0:
        dy = dy / 100.0

    return {
        "current_price":  raw.get("regularMarketPrice"),
        "pl":             raw.get("priceEarningsRatio"),
        "dividend_yield": dy,
        "beta":           raw.get("beta"),
        "avg_volume_30d": (
            raw.get("averageDailyVolume3Month")
            or raw.get("regularMarketVolume")
        ),
        "market_cap":    raw.get("marketCap"),
        "week52_high":   raw.get("fiftyTwoWeekHigh"),
        "week52_low":    raw.get("fiftyTwoWeekLow"),
        "_source":       "brapi",
    }


def _merge_sources(brapi_parsed: dict, yf_info: dict) -> dict:
    """
    Mescla dados brapi (P/L, DY, volume) com yfinance (P/VP, ROE, ROIC, Dívida/EBITDA).
    Regra: brapi tem prioridade para campos que ele retorna; yfinance preenche gaps.
    """
    merged = {**yf_info}  # base: yfinance
    for key, value in brapi_parsed.items():
        if key.startswith("_"):
            continue
        if value is not None:
            merged[key] = value  # brapi sobrescreve se não-nulo

    # Fonte composta
    sources = set()
    if brapi_parsed.get("_source") == "brapi":
        sources.add("brapi")
    if yf_info.get("_source") == "yfinance":
        sources.add("yfinance")
    merged["data_source"] = "+".join(sorted(sources)) if sources else "failed"
    return merged


# DataCollector — orquestrador principal
class DataCollector:
    """
    Coleta fundamentos e preços do universo inteiro.
    """

    def __init__(
        self,
        universe_file: Path = UNIVERSE_FILE,
        cache: Optional[CacheManager] = None,
        brapi: Optional[BrapiClient] = None,
        yf_client: Optional[YFinanceClient] = None,
    ):
        self.universe_file = Path(universe_file)
        self.cache  = cache     or CacheManager()
        self.brapi  = brapi     or BrapiClient(token=os.getenv("BRAPI_TOKEN") or None)
        self.yf     = yf_client or YFinanceClient()
        self._universe: Optional[pd.DataFrame] = None

    @property
    def universe(self) -> pd.DataFrame:
        """
        Universe filtrado para a data atual (PIT-aware).

        Aplica:
          - inclusion_date ≤ today: ticker já era investável na data
          - exclusion_date > today (ou vazia): ticker ainda não foi excluído

        Para backtests históricos, use universe_at(date) em vez desta property.
        """
        if self._universe is None:
            self._universe = self._load_universe_at(datetime.now())
        return self._universe

    def universe_at(self, as_of_date) -> pd.DataFrame:
        """
        Retorna universo investível em uma data específica (sem cache).

        Crítico para walk-forward backtest evitar survivorship bias:
        tickers delistados/trocados de mercado entre datas devem reaparecer
        nas datas anteriores ao seu exclusion_date.
        """
        return self._load_universe_at(as_of_date)

    def _load_universe_at(self, as_of):
        """Lê universe.csv e filtra por inclusion/exclusion dates."""
        df = pd.read_csv(self.universe_file)
        as_of_ts = pd.Timestamp(as_of)

        # Parse datas; default permissivo se ausentes
        if "inclusion_date" in df.columns:
            inc = pd.to_datetime(df["inclusion_date"], errors="coerce")
            inc = inc.fillna(pd.Timestamp("1900-01-01"))
            df = df[inc <= as_of_ts]
        if "exclusion_date" in df.columns:
            exc = pd.to_datetime(df["exclusion_date"], errors="coerce")
            # NaT (vazia) = ativo; senão, comparar
            active_mask = exc.isna() | (exc > as_of_ts)
            df = df[active_mask]

        return df.reset_index(drop=True)

    # Ponto de entrada público
    def collect(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executa a coleta completa com cache.

        Returns:
            df_fundamentals: uma linha por ticker, colunas tipadas float64
            df_prices:       wide DataFrame (date × ticker, preço ajustado)
        """
        today = datetime.now().strftime("%Y-%m-%d")
        fund_key  = f"fundamentals_{today}"
        price_key = f"prices_wide_{today}"

        # --- Fundamentais ---
        cached_fund = self.cache.get(fund_key)
        if cached_fund:
            df_fund = pd.DataFrame(cached_fund)
            logger.info("Fundamentais: cache hit (%d tickers)", len(df_fund))
        else:
            df_fund = self._collect_fundamentals()
            self.cache.set(fund_key, df_fund.to_dict(orient="records"))
            logger.info("Fundamentais coletados e cacheados (%d tickers)", len(df_fund))

        # --- Preços históricos ---
        cached_prices = self.cache.get(price_key)
        adv_key = f"adv_brl_{today}"
        if cached_prices:
            df_prices = pd.DataFrame.from_dict(cached_prices)
            df_prices.index = pd.to_datetime(df_prices.index)
            logger.info("Preços: cache hit (%d dias × %d tickers)", *df_prices.shape)
            # Carregar YZ vols do cache per-ticker
            self._yz_vols = self._load_yz_from_cache(df_prices.columns.tolist(), today)
            # ADV R$ do cache (calculado no primeiro run do dia)
            self._adv_brl = self.cache.get(adv_key) or {}
        else:
            valid_tickers = df_fund.loc[
                df_fund["data_source"] != "failed", "ticker"
            ].tolist()
            df_prices = self._collect_prices(valid_tickers)
            df_cache = df_prices.copy()
            df_cache.index = df_cache.index.strftime("%Y-%m-%d")
            self.cache.set(price_key, df_cache.to_dict())
            self.cache.set(adv_key, getattr(self, "_adv_brl", {}))
            logger.info("Preços coletados e cacheados (%d dias × %d tickers)", *df_prices.shape)

        # Merge YZ vol em df_fund (campo volatility_yz_180d)
        yz_map = getattr(self, "_yz_vols", {})
        if yz_map:
            df_fund["volatility_yz_180d"] = df_fund["ticker"].map(yz_map)
            n_yz = df_fund["volatility_yz_180d"].notna().sum()
            logger.info("Yang-Zhang vol: %d/%d tickers", n_yz, len(df_fund))

        # ADV em R$ calculado do OHLCV SUBSTITUI avg_volume_30d (que vinha da
        # brapi em escala inconsistente — causava exclusão indevida de ações
        # líquidas no filtro de liquidez, derrubando o universo a ~40%).
        adv_map = getattr(self, "_adv_brl", {})
        if adv_map:
            calc_adv = df_fund["ticker"].map(adv_map)
            n_adv = int(calc_adv.notna().sum())
            # Onde o ADV calculado existe, usar; senão manter NaN (o filtro de
            # liquidez trata NaN como "não penalizar", evitando exclusão por
            # dado ausente — mais seguro que confiar no campo bruto da brapi).
            df_fund["avg_volume_30d"] = calc_adv
            logger.info("ADV R$ (OHLCV close×volume): %d/%d tickers", n_adv, len(df_fund))

        return df_fund, df_prices

    def _load_yz_from_cache(self, tickers: list[str], today: str) -> dict[str, float]:
        """Carrega YZ vol já computada no cache de cada ticker."""
        out: dict[str, float] = {}
        for t in tickers:
            cache_key = f"prices_{t}_{today}"
            cached = self.cache.get(cache_key)
            if cached and cached.get("yz_vol_180d") is not None:
                out[t] = float(cached["yz_vol_180d"])
        return out

    def _load_adv_from_cache(self, tickers: list[str], today: str) -> dict[str, float]:
        """Carrega ADV (R$) já computado no cache per-ticker (cache hit de preços)."""
        out: dict[str, float] = {}
        for t in tickers:
            cached = self.cache.get(f"prices_{t}_{today}")
            if cached and cached.get("adv_brl_21d") is not None:
                out[t] = float(cached["adv_brl_21d"])
        return out

    @staticmethod
    def _adv_brl_from_ohlc(df_ohlc: pd.DataFrame, window: int = 21) -> Optional[float]:
        """
        Average Daily Volume em R$ = mediana de (Close × Volume) nos últimos
        `window` pregões. Mediana (não média) para robustez a dias de pico.

        Não usar averageDailyVolume3Month da brapi: vem em escala inconsistente
        entre tickers e derrubava ações líquidas (VIVT3, EGIE3, TAEE11...) no
        filtro de liquidez. Close×Volume dá R$ comparáveis ao threshold.
        """
        if df_ohlc is None or df_ohlc.empty:
            return None
        if "Close" not in df_ohlc.columns or "Volume" not in df_ohlc.columns:
            return None
        try:
            tail = df_ohlc.tail(window)
            dollar_vol = (tail["Close"] * tail["Volume"]).dropna()
            if dollar_vol.empty:
                return None
            adv = float(dollar_vol.median())
            return adv if adv > 0 else None
        except Exception as exc:
            logger.debug("ADV R$ falhou: %s", exc)
            return None

    # Coleta de Fundamentais
    def _collect_fundamentals(self) -> pd.DataFrame:
        tickers = self.universe["ticker"].tolist()
        logger.info("Coletando fundamentais para %d tickers", len(tickers))

        # Passo 1: batch brapi para quotes (já paraleliza por batch)
        brapi_data, brapi_failed = self.brapi.fetch_quotes_all(tickers, self.cache)

        # Passo 2: yfinance — fetch paralelo de info + advanced fundamentals.
        # ThreadPoolExecutor: chamadas yfinance são I/O-bound (network), GIL libera
        # durante o request. max_workers=6 é seguro vs rate limit do Yahoo.
        # Sequencial seria ~10 min para 98 tickers × 5 chamadas. Paralelo: ~2 min.
        yf_results: dict[str, tuple[dict, dict]] = {}  # ticker → (info, adv)
        yf_failed: list[str] = []

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {
                pool.submit(self._fetch_yf_for_ticker, t): t
                for t in tickers
            }
            for fut in as_completed(futures):
                ticker = futures[fut]
                try:
                    info, adv = fut.result()
                    yf_results[ticker] = (info, adv)
                    if info.get("_source") == "failed":
                        yf_failed.append(ticker)
                except Exception as exc:
                    logger.warning("yf fetch %s falhou: %s", ticker, exc)
                    yf_results[ticker] = ({"_source": "failed", "ticker": ticker}, {})
                    yf_failed.append(ticker)

        logger.info("yfinance paralelo: %d ok, %d falhos", len(yf_results) - len(yf_failed), len(yf_failed))

        # Passo 3: merge sequencial (sem I/O — rápido)
        records: list[dict] = []
        for _, row in self.universe.iterrows():
            ticker = row["ticker"]
            setor  = row["setor"]

            record: dict = {
                "ticker":             ticker,
                "nome":               row["nome"],
                "setor":              setor,
                "subsetor":           row["subsetor"],
                "liquidez_minima_MM": float(row["liquidez_minima_MM"]),
                "norm_method":        row["norm_method"],
            }

            brapi_parsed = (
                _parse_brapi_quote(brapi_data[ticker])
                if ticker in brapi_data
                else {"_source": "missing"}
            )

            yf_info, adv = yf_results.get(ticker, ({"_source": "failed", "ticker": ticker}, {}))

            merged = _merge_sources(brapi_parsed, yf_info)
            record.update({k: v for k, v in merged.items() if not k.startswith("_")})

            # Advanced fundamentals: adiciona apenas chaves não-None
            if adv:
                record.update({k: v for k, v in adv.items() if v is not None})

            # Setor financeiro: Dívida/EBITDA não se aplica (modelo bancário)
            if setor in FINANCIAL_SECTORS:
                record["divida_ebitda"] = float("nan")

            # Se ambas as fontes falharam, marcar explicitamente
            if brapi_parsed.get("_source") == "missing" and yf_info.get("_source") == "failed":
                record["data_source"] = "failed"
                logger.error("Sem dados de nenhuma fonte para %s", ticker)

            records.append(record)

        # Passo 3: avaliar taxa de falha global
        total_failed = len(
            set(t for t, r in zip(tickers, records) if r.get("data_source") == "failed")
        )
        fail_ratio = total_failed / len(tickers)
        if fail_ratio > MAX_FAILURES_RATIO:
            raise DataCollectionError(
                f"{total_failed}/{len(tickers)} tickers sem dados ({fail_ratio:.0%}). "
                "Verifique conexão, rate limits ou disponibilidade das APIs."
            )
        if total_failed:
            failed_tickers = [r["ticker"] for r in records if r.get("data_source") == "failed"]
            logger.warning("%d tickers excluídos por falha total de dados: %s", total_failed, failed_tickers)

        df = pd.DataFrame(records)
        df = self._cast_and_clean(df)
        self._log_coverage_report(df)
        return df

    def _fetch_yf_for_ticker(self, ticker: str) -> tuple[dict, dict]:
        """
        Coleta info + advanced fundamentals de um único ticker via yfinance.

        Cache de 24h por ticker. Designed para chamada em pool paralelo
        (não usa state compartilhado além do cache, que é thread-safe via
        atomic-write em disco — múltiplos workers escrevendo cache simultâneo
        no mesmo arquivo é improvável já que tickers diferentes geram chaves
        diferentes).

        Returns:
            (info, advanced) — dois dicts; info pode ter "_source": "failed"
        """
        today = datetime.now().strftime("%Y-%m-%d")

        # Info (básico)
        info_key = f"yf_info_{ticker}_{today}"
        info = self.cache.get(info_key)
        if info is None:
            info = self.yf.fetch_info(ticker)
            self.cache.set(info_key, info)

        # Skip advanced se info falhou
        if info.get("_source") == "failed":
            return info, {}

        # Advanced (3y growth, FCF, analyst recs)
        adv_key = f"yf_advanced_{ticker}_{today}"
        adv = self.cache.get(adv_key)
        if adv is None:
            adv = self.yf.fetch_advanced_fundamentals(ticker)
            self.cache.set(adv_key, adv)

        return info, adv

    # Coleta de Preços Históricos (12 meses)
    def _collect_prices(self, tickers: list[str]) -> pd.DataFrame:
        """
        Retorna DataFrame wide: index=date (DatetimeIndex), columns=tickers.

        Estratégia: yfinance history (1y) por ticker em paralelo (6 workers).
        Cache de 24h por ticker — inclui OHLC para cálculo de Yang-Zhang vol.

        Yang-Zhang vol é ~5x mais eficiente que close-to-close (Yang-Zhang
        2000), captura overnight gap + intraday range. Armazenada como
        atributo `yz_vols` na instância para merge posterior em df_fund.
        """
        logger.info("Coletando preços históricos (1y) para %d tickers", len(tickers))
        series: dict[str, pd.Series] = {}
        yz_vols: dict[str, float] = {}
        adv_brl: dict[str, float] = {}
        today = datetime.now().strftime("%Y-%m-%d")

        def _fetch_one(ticker: str) -> tuple[Optional[pd.Series], Optional[float], Optional[float]]:
            cache_key = f"prices_{ticker}_{today}"
            cached = self.cache.get(cache_key)
            if cached:
                s = pd.Series(
                    data=cached["close"],
                    index=pd.to_datetime(cached["dates"]),
                    name=ticker,
                )
                yz = cached.get("yz_vol_180d")
                adv = cached.get("adv_brl_21d")
                return s, yz, adv

            df_hist = self.yf.fetch_history(ticker)
            if df_hist is None or df_hist.empty:
                logger.warning("Sem histórico de preços para %s", ticker)
                return None, None, None

            if isinstance(df_hist.columns, pd.MultiIndex):
                df_hist.columns = df_hist.columns.get_level_values(0)

            close_col = "Adj Close" if "Adj Close" in df_hist.columns else "Close"
            raw = df_hist[close_col].dropna()
            if isinstance(raw, pd.DataFrame):
                raw = raw.iloc[:, 0]
            s = raw.astype(float)
            s.name = ticker

            # Yang-Zhang vol nos últimos 180 dias (com OHLC do mesmo df_hist)
            yz = _compute_yz_vol(df_hist.tail(180))

            # ADV em R$ = mediana(Close × Volume, 21d). Tem que ser calculado
            # neste caminho de coleta; senão avg_volume_30d fica com o valor da
            # brapi, em escala errada.
            adv = self._adv_brl_from_ohlc(df_hist)

            self.cache.set(cache_key, {
                "ticker":     ticker,
                "source":     "yfinance",
                "dates":      [str(d.date()) for d in s.index],
                "close":      [round(float(v), 4) for v in s.to_numpy()],
                "yz_vol_180d": yz,
                "adv_brl_21d": adv,
            })
            return s, yz, adv

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(_fetch_one, t): t for t in tickers}
            for fut in as_completed(futures):
                ticker = futures[fut]
                try:
                    s, yz, adv = fut.result()
                    if s is not None:
                        series[ticker] = s
                    if yz is not None:
                        yz_vols[ticker] = yz
                    if adv is not None:
                        adv_brl[ticker] = adv
                except Exception as exc:
                    logger.warning("Histórico %s falhou: %s", ticker, exc)
        # Disponibilizar para merge em df_fundamentals
        self._yz_vols = yz_vols
        self._adv_brl = adv_brl

        if not series:
            raise DataCollectionError("Nenhum histórico de preços coletado.")

        df_prices = pd.DataFrame(series)
        df_prices.index = pd.to_datetime(df_prices.index)
        df_prices = df_prices.sort_index()

        # Manter apenas o último 1 ano
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=1)
        df_prices = df_prices.loc[df_prices.index >= cutoff]

        # Forward fill de lacunas curtas (feriados, circuit breakers)
        df_prices = df_prices.ffill(limit=3)

        logger.info(
            "df_prices final: %d dias × %d tickers (%.1f%% cobertura)",
            len(df_prices),
            len(df_prices.columns),
            df_prices.notna().mean().mean() * 100,
        )
        return df_prices

    # Limpeza e tipagem do DataFrame de fundamentais
    @staticmethod
    def _cast_and_clean(df: pd.DataFrame) -> pd.DataFrame:
        """
        Garante tipagem float64 e trata anomalias nos dados fundamentalistas.

        Anomalias tratadas:
          - P/L negativo: empresa com prejuízo → NaN (não ranking-ável)
          - ROE em porcentagem (> 5.0): dividir por 100 para normalizar em fração
          - DY em porcentagem (> 1.0): dividir por 100
          - Volume em unidades de ações: converter para R$ multiplicando pelo preço
          - ROIC negativo extremo (< -1.0): provável erro de dados → NaN
        """
        float_cols = [
            "pl", "pvp", "roe", "roic", "divida_ebitda",
            "dividend_yield", "beta", "avg_volume_30d",
            "current_price", "market_cap", "liquidez_minima_MM",
            "week52_high", "week52_low", "ev_ebitda",
            # Fundamentos avançados (yfinance financials)
            "net_margin_3y_avg", "revenue_growth_3y", "earnings_growth_3y",
            "asset_growth_yoy", "free_cash_flow_ttm", "fcf_payout_ratio",
            "analyst_rec_score", "analyst_rec_trend",
        ]
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        # P/L: negativo = empresa em prejuízo; P/L zero = divisão impossível
        if "pl" in df.columns:
            df.loc[df["pl"] <= 0, "pl"] = np.nan

        # ROE: yfinance retorna em fração (0.28 = 28%); brapi às vezes em % (28.0)
        if "roe" in df.columns:
            high_mask = df["roe"].abs() > 5.0
            df.loc[high_mask, "roe"] = df.loc[high_mask, "roe"] / 100.0
            df.loc[df["roe"] < -1.0, "roe"] = np.nan  # > -100% = dado inválido

        # ROIC: mesma lógica do ROE
        if "roic" in df.columns:
            high_mask = df["roic"].abs() > 5.0
            df.loc[high_mask, "roic"] = df.loc[high_mask, "roic"] / 100.0
            df.loc[df["roic"] < -1.0, "roic"] = np.nan

        # DY: normalizar para fração
        if "dividend_yield" in df.columns:
            high_mask = df["dividend_yield"] > 1.0
            df.loc[high_mask, "dividend_yield"] = df.loc[high_mask, "dividend_yield"] / 100.0
            # Filtrar outliers extremos: DY > MAX_PLAUSIBLE_DY (default 20%)
            # indica provento extraordinário, amortização de capital ou erro
            # de dados. Ex.: SBSP3 retornou DY=55% após dividendo especial 2024.
            implausible = df["dividend_yield"].notna() & (df["dividend_yield"] > MAX_PLAUSIBLE_DY)
            if implausible.any():
                outliers = df.loc[implausible, ["ticker", "dividend_yield"]].to_dict("records")
                logger.warning(
                    "DY implausível (>%.0f%%): %s — setando NaN",
                    MAX_PLAUSIBLE_DY * 100,
                    [(o["ticker"], f"{o['dividend_yield']*100:.1f}%") for o in outliers],
                )
                df.loc[implausible, "dividend_yield"] = np.nan

        # Volume: se avg_volume_30d parece ser número de ações (< 1M para ações do IBrX),
        # converter para BRL multiplicando pelo preço atual
        if "avg_volume_30d" in df.columns and "current_price" in df.columns:
            vol = df["avg_volume_30d"]
            price = df["current_price"]
            # Heurística: volume financeiro de ação do IBrX raramente abaixo de R$ 1M
            needs_convert = (vol < 1_000_000) & (vol > 0) & (price > 0)
            df.loc[needs_convert, "avg_volume_30d"] = (
                df.loc[needs_convert, "avg_volume_30d"]
                * df.loc[needs_convert, "current_price"]
            )

        str_cols = ["ticker", "nome", "setor", "subsetor", "norm_method", "data_source"]
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        return df

    # Relatório de cobertura de dados (log apenas)
    @staticmethod
    def _log_coverage_report(df: pd.DataFrame) -> None:
        metrics = ["pl", "pvp", "roe", "roic", "divida_ebitda", "dividend_yield", "beta"]
        logger.info("=== Cobertura de dados por métrica ===")
        for m in metrics:
            if m in df.columns:
                coverage = df[m].notna().sum()
                total = len(df)
                logger.info("  %-20s %3d/%d tickers (%.0f%%)", m, coverage, total, 100 * coverage / total)
        failed = (df["data_source"] == "failed").sum()
        if failed:
            logger.warning("  FAILED: %d tickers sem dados de nenhuma fonte", failed)


# Função de conveniência para uso em main.py e outros módulos
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ponto de entrada público do módulo.

    Returns:
        df_fundamentals: pd.DataFrame com métricas fundamentalistas por ticker
        df_prices:       pd.DataFrame wide (date × ticker) de preços ajustados
    """
    collector = DataCollector()
    return collector.collect()
