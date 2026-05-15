"""
trade_advisor.py

Calculates actionable trade parameters for each top-5 ticker:
  - Entry zone (±2% around current price)
  - Price targets via 4 independent methods:
      A. Sector P/L reversion  — EPS × sector median P/L
      B. Sector P/VP reversion — BVS × sector median P/VP
      C. Graham Number         — sqrt(22.5 × EPS × BVS)
      D. Technical resistance  — MA20 / MA50 / MA200 / 52w high
  - Conservative target = nearest above entry (lowest of all valid)
  - Extended target      = highest of all valid targets
  - Stop-loss: MA50 × 0.99 if MA50 is below price and > -12%; else -7% flat; capped at -10%
  - Risk/Reward = (conservative_target - entry_mid) / (entry_mid - stop)
  - Opportunity type: Valor / Momentum / Qualidade / Misto
  - Horizon: 1-4w / 2-6w / 3-8w

Python 3.11 compatible.
"""

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.config import GRAHAM_CONSTANT_FALLBACK, GRAHAM_RF_BASE

logger = logging.getLogger(__name__)

# Cache de instância para evitar re-fetch do Graham constant na mesma execução
_GRAHAM_CONSTANT_CACHE: Optional[float] = None


def _graham_constant() -> float:
    """
    Ajusta a constante de Graham (22.5) pela taxa livre de risco brasileira.

    Graham calibrou 22.5 assumindo rf ≈ 4% a.a. Com SELIC mais alta,
    o múltiplo justo é menor: constante = 22.5 × (4% / selic_atual).

    Exemplo: SELIC 13.75% → 22.5 × (0.04 / 0.1375) ≈ 6.5
             Um P/L de 8× e P/VP de 0.8× → Graham = sqrt(6.5 × EPS × BVS)

    Busca SELIC do cache do BenchmarkManager (sem rede se cache válido).
    Fallback para GRAHAM_CONSTANT_FALLBACK se indisponível.
    """
    global _GRAHAM_CONSTANT_CACHE
    if _GRAHAM_CONSTANT_CACHE is not None:
        return _GRAHAM_CONSTANT_CACHE

    try:
        from src.benchmark import BenchmarkManager
        bm = BenchmarkManager()
        recent = bm.get_returns(
            start_date=date.today() - timedelta(days=30),
            end_date=date.today(),
        )
        if "selic" in recent.columns:
            daily_vals = recent["selic"].dropna()
            if len(daily_vals) >= 5:
                daily_mean = float(daily_vals.mean())
                annual_selic = float((1 + daily_mean) ** 252 - 1)
                if 0.02 <= annual_selic <= 0.50:  # faixa histórica plausível BR
                    constant = round(22.5 * (GRAHAM_RF_BASE / annual_selic), 4)
                    logger.info(
                        "Graham constant dinâmica: %.4f "
                        "(SELIC atual %.2f%% a.a., base %.0f%%)",
                        constant, annual_selic * 100, GRAHAM_RF_BASE * 100,
                    )
                    _GRAHAM_CONSTANT_CACHE = constant
                    return constant
    except Exception as exc:
        logger.debug("Graham constant: fallback para config (%s)", exc)

    _GRAHAM_CONSTANT_CACHE = GRAHAM_CONSTANT_FALLBACK
    logger.debug("Graham constant: usando fallback %.4f", GRAHAM_CONSTANT_FALLBACK)
    return GRAHAM_CONSTANT_FALLBACK

_MIN_TARGET_GAP = 0.025   # target must be ≥2.5% above entry to count
_STOP_TECH_FLOOR = 0.88   # MA50 must be within -12% of price to use as stop
_STOP_DEFAULT_PCT = 0.07  # 7% default stop
_STOP_MAX_PCT = 0.10      # never risk more than 10%


class TradeAdvisor:
    """
    Computes entry, targets, stop-loss, R/R, type, and horizon for a single ticker.

    Usage:
        advisor = TradeAdvisor()
        trade = advisor.compute(row, df_scored, tech)
        # Returns dict or {} if price unavailable
    """

    def compute(
        self,
        row: pd.Series,
        df_scored: pd.DataFrame,
        tech: dict,
    ) -> dict:
        price = tech.get("price") or _safe_float(row.get("current_price"))
        if not price or price <= 0:
            return {}

        setor = str(row.get("setor", ""))
        ey = _safe_float(row.get("earnings_yield"))   # = E/P  (1/P·L)
        pvp = _safe_float(row.get("pvp"))             # = P/BV

        # Sector medians from other tickers in same sector
        sector_ey_med, sector_pvp_med = self._sector_medians(df_scored, setor)

        # ── Target methods ────────────────────────────────────────────────────
        target_pl = self._target_from_pl(price, ey, sector_ey_med)
        target_pvp = self._target_from_pvp(price, pvp, sector_pvp_med)
        target_graham = self._target_graham(price, ey, pvp)
        tech_levels = self._technical_targets(price, tech)

        # All valid targets above entry (price * (1 + gap))
        min_level = price * (1 + _MIN_TARGET_GAP)
        fund_targets = [t for t in [target_pl, target_pvp, target_graham]
                        if t is not None and t > min_level]
        valid_tech = [t for t in tech_levels if t > min_level]
        all_targets = sorted(fund_targets + valid_tech)

        if all_targets:
            target_conservative = all_targets[0]
            target_extended = all_targets[-1]
        else:
            target_conservative = round(price * 1.10, 2)
            target_extended = round(price * 1.20, 2)

        if target_extended <= target_conservative:
            target_extended = round(target_conservative * 1.10, 2)

        # ── Entry zone ────────────────────────────────────────────────────────
        entry_low = round(price * 0.98, 2)
        entry_high = round(price * 1.02, 2)
        entry_mid = price  # symmetric, ≈ current price

        # ── Stop-loss ─────────────────────────────────────────────────────────
        # Priority: ATR(14) × 2.0 → MA50 × 0.99 → -7% flat; always capped at -10%
        #
        # Salvaguardas adicionais:
        #   - ATR > 10% do preço (volatilidade anômala, ex: dia de crash) é
        #     ignorado — usar default 7% para não posicionar stop absurdo
        #   - Stop nunca abaixo do cap (max risco -10%)
        #   - Stop nunca a menos de 3% abaixo do preço (sob-risco/ruído)
        ma50 = tech.get("ma50")
        atr  = tech.get("atr")
        cap  = round(price * (1 - _STOP_MAX_PCT), 2)
        floor_close = round(price * 0.97, 2)  # stop não pode ficar dentro do ruído de 3%

        atr_usable = (atr is not None and not np.isnan(atr)
                      and 0 < atr < price * 0.10)

        if atr_usable:
            stop = max(round(price - 2.0 * atr, 2), cap)
        elif (ma50 and not np.isnan(ma50)
                and price * _STOP_TECH_FLOOR < ma50 < price):
            stop = max(round(ma50 * 0.99, 2), cap)
        else:
            stop = max(round(price * (1 - _STOP_DEFAULT_PCT), 2), cap)

        # Garantir separação mínima do preço (anti-ruído)
        if stop > floor_close:
            stop = floor_close

        # ── Risk / Reward ─────────────────────────────────────────────────────
        downside = max(entry_mid - stop, 0.01)
        upside = target_conservative - entry_mid
        rr = round(upside / downside, 2) if upside > 0 else 0.0

        # ── Type & horizon ────────────────────────────────────────────────────
        opp_type = self._opportunity_type(row, tech)
        horizon = self._horizon(row, tech, opp_type)

        return {
            "entry_low":           entry_low,
            "entry_high":          entry_high,
            "target_conservative": round(target_conservative, 2),
            "target_extended":     round(target_extended, 2),
            "stop":                stop,
            "rr":                  rr,
            "opportunity_type":    opp_type,
            "horizon":             horizon,
            "target_pl":           round(target_pl, 2) if target_pl else None,
            "target_pvp":          round(target_pvp, 2) if target_pvp else None,
            "target_graham":       round(target_graham, 2) if target_graham else None,
        }

    # ── Target methods ────────────────────────────────────────────────────────

    @staticmethod
    def _target_from_pl(
        price: float,
        ey: Optional[float],
        sector_ey_med: Optional[float],
    ) -> Optional[float]:
        """EPS × sector-median P/L — mean-reversion of earnings multiple."""
        if not ey or not sector_ey_med or sector_ey_med <= 0:
            return None
        eps = price * ey
        sector_pl_med = 1.0 / sector_ey_med
        if sector_pl_med <= 0 or sector_pl_med > 60:
            return None
        return round(eps * sector_pl_med, 2)

    @staticmethod
    def _target_from_pvp(
        price: float,
        pvp: Optional[float],
        sector_pvp_med: Optional[float],
    ) -> Optional[float]:
        """Book value × sector-median P/VP — book-value mean-reversion."""
        if not pvp or pvp <= 0 or not sector_pvp_med or sector_pvp_med <= 0:
            return None
        bvs = price / pvp
        target = bvs * sector_pvp_med
        if target > price * 3:
            return None
        return round(target, 2)

    @staticmethod
    def _target_graham(
        price: float,
        ey: Optional[float],
        pvp: Optional[float],
    ) -> Optional[float]:
        """
        Graham Number = sqrt(K × EPS × BVS).

        K é ajustado pela SELIC atual vs taxa base de Graham (~4%).
        Com SELIC 13.75%: K ≈ 6.5 (vs 22.5 original).
        Isso reflete que ações brasileiras devem negociar a múltiplos menores
        num ambiente de juros mais altos.
        """
        if not ey or ey <= 0 or not pvp or pvp <= 0:
            return None
        eps = price * ey
        bvs = price / pvp
        k = _graham_constant()
        product = k * eps * bvs
        if product <= 0:
            return None
        return round(np.sqrt(product), 2)

    @staticmethod
    def _technical_targets(price: float, tech: dict) -> list[float]:
        """Resistance levels from moving averages and 52-week high."""
        candidates = []
        for key in ("ma20", "ma50", "ma200", "high_52w"):
            val = tech.get(key)
            if val and not np.isnan(val) and val > price:
                candidates.append(round(val, 2))
        candidates.sort()
        return candidates

    # ── Sector medians ────────────────────────────────────────────────────────

    @staticmethod
    def _sector_medians(
        df: pd.DataFrame,
        setor: str,
    ) -> tuple[Optional[float], Optional[float]]:
        if df.empty or not setor:
            return None, None
        mask = df["setor"] == setor
        sector_df = df[mask]
        if len(sector_df) < 2:
            return None, None

        ey_vals = pd.to_numeric(sector_df.get("earnings_yield", pd.Series()), errors="coerce").dropna()
        ey_vals = ey_vals[ey_vals > 0]
        ey_med = float(ey_vals.median()) if len(ey_vals) >= 2 else None

        pvp_vals = pd.to_numeric(sector_df.get("pvp", pd.Series()), errors="coerce").dropna()
        pvp_vals = pvp_vals[pvp_vals > 0]
        pvp_med = float(pvp_vals.median()) if len(pvp_vals) >= 2 else None

        return ey_med, pvp_med

    # ── Type & horizon ────────────────────────────────────────────────────────

    @staticmethod
    def _opportunity_type(row: pd.Series, tech: dict) -> str:
        """Classifies the opportunity: Valor / Momentum / Qualidade / Misto."""
        pvp = _safe_float(row.get("pvp"))
        dy = _safe_float(row.get("dividend_yield"))
        roe = _safe_float(row.get("roe"))
        alpha = _safe_float(row.get("alpha_6m"))
        signal = tech.get("signal", "neutral")
        macd_cross = tech.get("macd_crossover")

        is_valor = (pvp is not None and pvp < 1.3) or (dy is not None and dy > 0.05)
        is_momentum = (alpha is not None and alpha > 0.06) or (signal == "strong_buy" and macd_cross == "bullish")
        is_qualidade = (roe is not None and roe > 0.18)

        hits = sum([is_valor, is_momentum, is_qualidade])
        if hits >= 2:
            return "Misto"
        if is_valor:
            return "Valor"
        if is_momentum:
            return "Momentum"
        if is_qualidade:
            return "Qualidade"
        return "Misto"

    @staticmethod
    def _horizon(row: pd.Series, tech: dict, opp_type: str) -> str:
        signal = tech.get("signal", "neutral")
        macd_cross = tech.get("macd_crossover")
        rsi = tech.get("rsi") or 50
        score = float(row.get("total_score") or 0)

        if signal == "strong_buy" and macd_cross == "bullish" and rsi < 38:
            return "1–4 semanas"
        if score >= 80 and signal in ("buy", "strong_buy"):
            return "2–6 semanas"
        return "3–8 semanas"


# ── Helper ────────────────────────────────────────────────────────────────────

def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None
