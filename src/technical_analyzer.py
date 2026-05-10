"""
technical_analyzer.py

Computes technical indicators for a list of tickers using historical price data.
All indicators are implemented manually with pandas/numpy — no extra dependencies.

Indicators:
  RSI(14)              — momentum oscillator (Wilder's smoothing)
  MACD(12,26,9)        — trend/momentum, crossover detection
  Bollinger Bands(20,2) — volatility bands, normalized price position
  SMA 20 / 50 / 200   — short, medium, long-term trend reference
  Volume ratio         — today's volume vs 20-day average
  Intraday gap         — (open − prev_close) / prev_close
  52-week high/low     — proximity as % distance

Output signal composite:
  Scores each factor +/- then maps to:
  strong_buy | buy | neutral | sell | strong_sell
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ─── Indicator functions ──────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> float:
    """Wilder's RSI using EWM smoothing (equivalent to Wilder's MA)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    last_loss = avg_loss.iloc[-1]
    if last_loss == 0:
        return 100.0
    rs = avg_gain.iloc[-1] / last_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """MACD line, signal line, histogram, and crossover direction."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    crossover = None
    if len(histogram) >= 2:
        prev_h = histogram.iloc[-2]
        curr_h = histogram.iloc[-1]
        if prev_h < 0 and curr_h >= 0:
            crossover = "bullish"
        elif prev_h > 0 and curr_h <= 0:
            crossover = "bearish"

    return {
        "macd":         float(macd_line.iloc[-1]),
        "signal_line":  float(signal_line.iloc[-1]),
        "histogram":    float(histogram.iloc[-1]),
        "crossover":    crossover,
        "above_signal": bool(macd_line.iloc[-1] > signal_line.iloc[-1]),
    }


def _bollinger(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> dict:
    """
    Bollinger Bands with normalized position.

    position=0 → price at lower band
    position=1 → price at upper band
    position=0.5 → price at middle (SMA)
    """
    sma = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0)
    upper = sma + std_dev * std
    lower = sma - std_dev * std

    price = float(series.iloc[-1])
    up = float(upper.iloc[-1])
    lo = float(lower.iloc[-1])
    mid = float(sma.iloc[-1])

    band_width = up - lo
    position = (price - lo) / band_width if band_width > 0 else 0.5
    bandwidth = band_width / mid if mid != 0 else float("nan")

    return {
        "upper":     up,
        "middle":    mid,
        "lower":     lo,
        "position":  float(np.clip(position, -0.2, 1.2)),
        "bandwidth": float(bandwidth),
    }


def _sma(series: pd.Series, period: int) -> Optional[float]:
    if len(series) < period:
        return None
    return float(series.rolling(period).mean().iloc[-1])


def _trend_label(price: float, ma20: Optional[float], ma50: Optional[float], ma200: Optional[float]) -> str:
    if ma20 and ma50 and ma200:
        if price > ma20 > ma50 > ma200:
            return "uptrend"
        if price < ma20 < ma50:
            return "downtrend"
    if ma20 and ma50:
        if price > ma20 and price > ma50:
            return "uptrend"
        if price < ma20 and price < ma50:
            return "downtrend"
    return "sideways"


def _signal(data: dict) -> str:
    """
    Composite technical signal score.

    Each condition adds/subtracts from a float score.
    Maps: > +1.5 → strong_buy, > +0.5 → buy, < -1.5 → strong_sell, < -0.5 → sell.
    """
    score = 0.0

    rsi = data.get("rsi")
    if rsi is not None and not np.isnan(rsi):
        if rsi < 30:
            score += 1.0
        elif rsi < 45:
            score += 0.4
        elif rsi > 70:
            score -= 1.0
        elif rsi > 55:
            score -= 0.4

    price = data.get("price")
    ma20 = data.get("ma20")
    ma50 = data.get("ma50")
    ma200 = data.get("ma200")
    if price and ma20:
        score += 0.5 if price > ma20 else -0.5
    if price and ma50:
        score += 0.5 if price > ma50 else -0.5
    if price and ma200:
        score += 0.25 if price > ma200 else -0.25

    if data.get("macd_above_signal"):
        score += 0.4
    else:
        score -= 0.4

    crossover = data.get("macd_crossover")
    if crossover == "bullish":
        score += 0.5
    elif crossover == "bearish":
        score -= 0.5

    bb_pos = data.get("bb_position")
    if bb_pos is not None and not np.isnan(bb_pos):
        if bb_pos < 0.15:
            score += 0.5
        elif bb_pos > 0.85:
            score -= 0.5

    if score > 1.5:
        return "strong_buy"
    if score > 0.5:
        return "buy"
    if score < -1.5:
        return "strong_sell"
    if score < -0.5:
        return "sell"
    return "neutral"


# ─── TechnicalAnalyzer ────────────────────────────────────────────────────────

class TechnicalAnalyzer:
    """
    Computes technical indicators for B3 tickers.

    Usage:
        analyzer = TechnicalAnalyzer()
        intraday = analyzer.fetch_intraday(["PETR4", "VALE3"])
        results  = analyzer.analyze_all(tickers, df_prices, intraday)
    """

    def analyze_ticker(
        self,
        ticker: str,
        df_prices: pd.DataFrame,
        intraday: Optional[dict] = None,
    ) -> dict:
        """
        Compute all technical indicators for a single ticker.

        Args:
            ticker:    Ticker symbol (e.g. "PETR4")
            df_prices: Wide adjusted-close DataFrame (date × ticker)
            intraday:  Optional dict from fetch_intraday() for this ticker

        Returns:
            Dict of all computed indicators. Missing values are None.
        """
        result: dict = {"ticker": ticker}

        if ticker not in df_prices.columns:
            logger.warning("TechnicalAnalyzer: %s não encontrado em df_prices", ticker)
            return result

        series = df_prices[ticker].dropna()
        if len(series) < 30:
            logger.warning("TechnicalAnalyzer: %s com apenas %d dias — insuficiente", ticker, len(series))
            return result

        price = float(series.iloc[-1])
        result["price"] = price

        if len(series) >= 2:
            prev = float(series.iloc[-2])
            result["price_prev"] = prev
            result["change_1d"] = (price - prev) / prev if prev != 0 else 0.0

        # ── Indicadores de momentum ─────────────────────────────────────────
        if len(series) >= 20:
            result["rsi"] = _rsi(series)

        if len(series) >= 35:
            macd_data = _macd(series)
            result["macd_line"]      = macd_data["macd"]
            result["macd_signal_line"] = macd_data["signal_line"]
            result["macd_histogram"] = macd_data["histogram"]
            result["macd_crossover"] = macd_data["crossover"]
            result["macd_above_signal"] = macd_data["above_signal"]

        # ── Bandas de Bollinger ─────────────────────────────────────────────
        if len(series) >= 20:
            bb = _bollinger(series)
            result["bb_upper"]     = bb["upper"]
            result["bb_middle"]    = bb["middle"]
            result["bb_lower"]     = bb["lower"]
            result["bb_position"]  = bb["position"]
            result["bb_bandwidth"] = bb["bandwidth"]

        # ── Médias móveis ───────────────────────────────────────────────────
        result["ma20"]  = _sma(series, 20)
        result["ma50"]  = _sma(series, 50)
        result["ma200"] = _sma(series, 200)

        result["trend"] = _trend_label(
            price,
            result.get("ma20"),
            result.get("ma50"),
            result.get("ma200"),
        )

        # ── 52 semanas ──────────────────────────────────────────────────────
        w52 = series.tail(252) if len(series) >= 252 else series
        high52 = float(w52.max())
        low52  = float(w52.min())
        result["high_52w"]          = high52
        result["low_52w"]           = low52
        result["pct_from_52w_high"] = (price - high52) / high52
        result["pct_from_52w_low"]  = (price - low52)  / low52  if low52 != 0 else float("nan")
        result["near_52w_high"]     = abs(result["pct_from_52w_high"]) < 0.03
        result["near_52w_low"]      = result["pct_from_52w_low"] < 0.05

        # ── Dados intraday (gap, volume) ────────────────────────────────────
        if intraday and ticker in intraday:
            day = intraday[ticker]
            result["gap_pct"]      = day.get("gap_pct")
            result["volume_ratio"] = day.get("volume_ratio")
            result["today_volume"] = day.get("today_volume")
            # Override 1d change with intraday data (more accurate)
            if day.get("change_1d") is not None:
                result["change_1d"] = day["change_1d"]

        # ── Sinal composto ──────────────────────────────────────────────────
        result["signal"] = _signal(result)

        return result

    def analyze_all(
        self,
        tickers: list[str],
        df_prices: pd.DataFrame,
        intraday: Optional[dict] = None,
    ) -> dict[str, dict]:
        """Analyze multiple tickers. Returns {ticker: indicators_dict}."""
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.analyze_ticker(ticker, df_prices, intraday)
            except Exception as exc:
                logger.warning("TechnicalAnalyzer: erro em %s — %s", ticker, exc)
                results[ticker] = {"ticker": ticker}
        return results

    def fetch_intraday(self, tickers: list[str]) -> dict[str, dict]:
        """
        Fetches last 5 days of daily OHLCV for gap and volume ratio computation.

        Uses period=5d/interval=1d for reliability.
        Returns {ticker: {gap_pct, change_1d, volume_ratio, today_volume}}.
        """
        if not tickers:
            return {}

        yf_symbols = [f"{t}.SA" for t in tickers]
        result: dict[str, dict] = {}

        try:
            raw = yf.download(
                tickers=yf_symbols,
                period="5d",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if raw is None or raw.empty:
                return result

            for ticker in tickers:
                yf_sym = f"{ticker}.SA"
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        level1 = raw.columns.get_level_values(1)
                        if yf_sym not in level1:
                            continue
                        close  = raw["Close"][yf_sym].dropna()
                        volume = raw["Volume"][yf_sym].dropna()
                        open_p = raw["Open"][yf_sym].dropna()
                    else:
                        close  = raw["Close"].dropna()
                        volume = raw["Volume"].dropna()
                        open_p = raw["Open"].dropna()

                    if len(close) < 2:
                        continue

                    today_close  = float(close.iloc[-1])
                    today_open   = float(open_p.iloc[-1]) if not open_p.empty else today_close
                    prev_close   = float(close.iloc[-2])
                    today_vol    = float(volume.iloc[-1]) if not volume.empty else float("nan")
                    avg_vol      = float(volume.iloc[:-1].mean()) if len(volume) > 1 else float("nan")

                    result[ticker] = {
                        "today_open":   today_open,
                        "today_close":  today_close,
                        "prev_close":   prev_close,
                        "gap_pct":      (today_open - prev_close) / prev_close if prev_close else 0.0,
                        "change_1d":    (today_close - prev_close) / prev_close if prev_close else 0.0,
                        "today_volume": today_vol,
                        "avg_volume":   avg_vol,
                        "volume_ratio": today_vol / avg_vol if avg_vol > 0 and not np.isnan(avg_vol) else float("nan"),
                    }
                except Exception as exc:
                    logger.debug("fetch_intraday %s: %s", ticker, exc)

        except Exception as exc:
            logger.warning("TechnicalAnalyzer.fetch_intraday falhou: %s", exc)

        return result

    @staticmethod
    def detect_alerts(
        tech_data: dict[str, dict],
        intraday: dict[str, dict],
    ) -> list[dict]:
        """
        Detects notable technical events across all tickers.

        Alert types:
          gap_up / gap_down    — abertura >2% vs fechamento anterior
          volume_spike         — volume >2× a média de 20 dias
          near_52w_high        — ação a <3% da máxima de 52 semanas
          near_52w_low         — ação a <5% da mínima de 52 semanas
          macd_crossover       — cruzamento de sinal (bullish/bearish)
          oversold / overbought — RSI < 30 ou > 70

        Returns list of {ticker, type, label, value, emoji}.
        """
        alerts = []

        for ticker, data in tech_data.items():
            # Gap alerts
            gap = data.get("gap_pct")
            if gap is not None and not np.isnan(gap):
                if gap > 0.02:
                    alerts.append({
                        "ticker": ticker, "type": "gap_up",
                        "label": "Gap de abertura",
                        "value": gap, "emoji": "🚀",
                    })
                elif gap < -0.02:
                    alerts.append({
                        "ticker": ticker, "type": "gap_down",
                        "label": "Gap de abertura",
                        "value": gap, "emoji": "🔻",
                    })

            # Volume spike
            vol_ratio = data.get("volume_ratio")
            if vol_ratio is not None and not np.isnan(vol_ratio) and vol_ratio >= 2.0:
                alerts.append({
                    "ticker": ticker, "type": "volume_spike",
                    "label": "Volume anormal",
                    "value": vol_ratio, "emoji": "🔥",
                })

            # 52-week proximity
            if data.get("near_52w_high"):
                pct = data.get("pct_from_52w_high", 0)
                alerts.append({
                    "ticker": ticker, "type": "near_52w_high",
                    "label": "Próximo da máxima 52s",
                    "value": pct, "emoji": "📈",
                })
            if data.get("near_52w_low"):
                pct = data.get("pct_from_52w_low", 0)
                alerts.append({
                    "ticker": ticker, "type": "near_52w_low",
                    "label": "Próximo da mínima 52s",
                    "value": pct, "emoji": "📉",
                })

            # MACD crossover
            crossover = data.get("macd_crossover")
            if crossover == "bullish":
                alerts.append({
                    "ticker": ticker, "type": "macd_bullish",
                    "label": "MACD cruzamento altista",
                    "value": None, "emoji": "⚡",
                })
            elif crossover == "bearish":
                alerts.append({
                    "ticker": ticker, "type": "macd_bearish",
                    "label": "MACD cruzamento baixista",
                    "value": None, "emoji": "⚠️",
                })

            # RSI extremes
            rsi = data.get("rsi")
            if rsi is not None and not np.isnan(rsi):
                if rsi < 30:
                    alerts.append({
                        "ticker": ticker, "type": "oversold",
                        "label": "RSI sobrevendido",
                        "value": rsi, "emoji": "🟢",
                    })
                elif rsi > 70:
                    alerts.append({
                        "ticker": ticker, "type": "overbought",
                        "label": "RSI sobrecomprado",
                        "value": rsi, "emoji": "🔴",
                    })

        return alerts
