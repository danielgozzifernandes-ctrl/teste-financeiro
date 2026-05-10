"""
macro_fetcher.py

Fetches global macro indicators for the daily morning report.
All data sourced from yfinance (free tier) — no API keys required.

Indicators:
  USD/BRL   — exchange rate
  S&P 500   — US equity benchmark
  Nasdaq    — US tech benchmark
  VIX       — implied volatility / "fear index"
  WTI Oil   — crude oil (impacts PETR3/4)
  Brent     — crude oil (European reference)
  Gold      — safe-haven / risk sentiment
  DXY       — US dollar index (impacts all emerging markets)
  IBOV      — IBOVESPA (today's local reference)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# (yf_symbol, display_label, currency_prefix, decimal_places)
_MACRO_TICKERS: dict[str, tuple[str, str, str, int]] = {
    "usdbrl":  ("BRL=X",     "USD/BRL",   "R$",  2),
    "sp500":   ("^GSPC",     "S&P 500",   "",    0),
    "nasdaq":  ("^IXIC",     "Nasdaq",    "",    0),
    "vix":     ("^VIX",      "VIX",       "",    1),
    "oil_wti": ("CL=F",      "WTI",       "US$", 2),
    "brent":   ("BZ=F",      "Brent",     "US$", 2),
    "gold":    ("GC=F",      "Ouro",      "US$", 0),
    "dxy":     ("DX-Y.NYB",  "DXY",       "",    2),
    "ibov":    ("^BVSP",     "IBOVESPA",  "",    0),
}

_RISK_OFF_THRESHOLD_VIX = 25.0   # VIX acima disso = risk-off
_LARGE_MOVE_PCT = 0.015           # variação >1.5% = movimento relevante


class MacroFetcher:
    """
    Fetches and structures global macro data for the daily report.

    Usage:
        fetcher = MacroFetcher()
        snapshot = fetcher.get_snapshot()
        # snapshot["usdbrl"] = {"label": "USD/BRL", "value": 5.82, "change_pct": 0.003, ...}
    """

    def get_snapshot(self) -> dict[str, dict]:
        """
        Returns current values and daily % changes for all macro indicators.

        Returns:
            Dict keyed by indicator name. Each value contains:
              label       — display name
              value       — latest price/value
              prev_value  — previous session value
              change_pct  — (value - prev_value) / prev_value
              prefix      — currency prefix (e.g. "R$")
              decimals    — decimal places for display
              direction   — "up" | "down" | "flat"
        """
        all_symbols = [v[0] for v in _MACRO_TICKERS.values()]
        result: dict[str, dict] = {}

        try:
            raw = yf.download(
                tickers=all_symbols,
                period="5d",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:
            logger.error("MacroFetcher: falha no download batch: %s", exc)
            raw = None

        if raw is None or raw.empty:
            logger.warning("MacroFetcher: dados vazios — tentando download individual")
            return self._get_snapshot_individual()

        for key, (symbol, label, prefix, decimals) in _MACRO_TICKERS.items():
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    level1 = raw.columns.get_level_values(1)
                    if symbol not in level1:
                        continue
                    close = raw["Close"][symbol].dropna()
                else:
                    close = raw["Close"].dropna()

                if len(close) < 2:
                    continue

                value      = float(close.iloc[-1])
                prev_value = float(close.iloc[-2])
                change_pct = (value - prev_value) / prev_value if prev_value != 0 else 0.0

                result[key] = {
                    "label":      label,
                    "value":      value,
                    "prev_value": prev_value,
                    "change_pct": change_pct,
                    "prefix":     prefix,
                    "decimals":   decimals,
                    "symbol":     symbol,
                    "direction":  "up" if change_pct > 0.001 else ("down" if change_pct < -0.001 else "flat"),
                }
            except Exception as exc:
                logger.debug("MacroFetcher %s (%s): %s", key, symbol, exc)

        logger.info("Macro: %d/%d indicadores obtidos", len(result), len(_MACRO_TICKERS))
        return result

    def _get_snapshot_individual(self) -> dict[str, dict]:
        """Fallback: downloads each ticker individually when batch fails."""
        result: dict[str, dict] = {}
        for key, (symbol, label, prefix, decimals) in _MACRO_TICKERS.items():
            try:
                df = yf.download(symbol, period="5d", interval="1d",
                                 auto_adjust=True, progress=False)
                if df is None or df.empty or len(df) < 2:
                    continue

                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close = close.dropna()
                if len(close) < 2:
                    continue

                value      = float(close.iloc[-1])
                prev_value = float(close.iloc[-2])
                change_pct = (value - prev_value) / prev_value if prev_value != 0 else 0.0

                result[key] = {
                    "label":      label,
                    "value":      value,
                    "prev_value": prev_value,
                    "change_pct": change_pct,
                    "prefix":     prefix,
                    "decimals":   decimals,
                    "symbol":     symbol,
                    "direction":  "up" if change_pct > 0.001 else ("down" if change_pct < -0.001 else "flat"),
                }
            except Exception as exc:
                logger.debug("MacroFetcher individual %s: %s", key, exc)
        return result

    @staticmethod
    def risk_sentiment(snapshot: dict[str, dict]) -> str:
        """
        Classifies macro risk sentiment as 'risk_on', 'risk_off', or 'neutral'.

        Heuristic:
          risk_off if VIX > 25, or USD rising strongly + S&P falling > 1.5%
          risk_on  if VIX < 18 and S&P rising > 0.5%
          neutral  otherwise
        """
        vix = snapshot.get("vix", {}).get("value")
        sp_chg = snapshot.get("sp500", {}).get("change_pct", 0)
        dxy_chg = snapshot.get("dxy", {}).get("change_pct", 0)

        if vix and vix > _RISK_OFF_THRESHOLD_VIX:
            return "risk_off"
        if sp_chg < -_LARGE_MOVE_PCT and dxy_chg > 0.005:
            return "risk_off"
        if vix and vix < 18 and sp_chg > 0.005:
            return "risk_on"
        return "neutral"

    @staticmethod
    def format_value(item: dict) -> str:
        """Formats a macro item value as a display string."""
        value   = item.get("value", 0)
        prefix  = item.get("prefix", "")
        dec     = item.get("decimals", 2)
        if prefix:
            return f"{prefix} {value:,.{dec}f}"
        if dec == 0:
            return f"{value:,.0f}"
        return f"{value:,.{dec}f}"

    @staticmethod
    def format_change(item: dict) -> str:
        """Formats daily % change with sign and emoji arrow."""
        chg = item.get("change_pct", 0) * 100
        sign = "+" if chg >= 0 else ""
        return f"{sign}{chg:.2f}%"
