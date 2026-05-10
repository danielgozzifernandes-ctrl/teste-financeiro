"""
opportunity_scanner.py

Scans the full scored universe for high-conviction opportunities.
Only fires when a ticker passes ALL primary criteria AND at least 2 of 3
secondary criteria simultaneously — intentionally rare.

Filter logic:

  PRIMARY (all required):
    1. total_score >= 72        Strong fundamentals (top ~15% of universe)
    2. signal in (buy, strong_buy)  Positive technical composite
    3. 28 <= RSI <= 48          Oversold-to-neutral zone — recovery potential
    4. MACD above signal line   Momentum direction confirmed

  SECONDARY (at least 2 of 3):
    A. MACD bullish crossover   Recent direction change — high timing value
    B. volume_ratio >= 1.5      Elevated volume — institutional interest
    C. bb_position <= 0.35      Price in lower Bollinger Band — mean-reversion setup

Price targets:
  Nearest significant resistance levels above current price:
    MA20 → MA50 → MA200 → 52-week high
  If none found above current price, uses fixed +10%/+20% targets.

Stop-loss:
  7% below entry price — standard individual equity risk management.

Horizon estimate:
  Based on signal strength and fundamental score.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Thresholds ───────────────────────────────────────────────────────────────

_MIN_SCORE       = 72.0    # total_score mínimo
_RSI_MIN         = 28.0    # RSI mínimo (não em colapso)
_RSI_MAX         = 48.0    # RSI máximo (ainda em zona favorável)
_VOL_THRESHOLD   = 1.5     # volume ratio mínimo para critério secundário
_BB_THRESHOLD    = 0.35    # posição BB máxima (0=banda inferior, 1=banda superior)
_MIN_SECONDARY   = 2       # critérios secundários mínimos para aprovação
_STOP_LOSS_PCT   = 0.07    # stop-loss 7% abaixo da entrada
_MIN_TARGET_GAP  = 0.015   # resistência deve estar ≥1.5% acima do preço atual


class OpportunityScanner:
    """
    Scans a scored DataFrame + technical indicators for high-conviction setups.

    Usage:
        scanner = OpportunityScanner()
        opportunities = scanner.scan(df_scored, tech_data)
        # Returns list of opportunity dicts — empty list = nothing qualifies today
    """

    def scan(
        self,
        df_scored: pd.DataFrame,
        tech_data: dict[str, dict],
    ) -> list[dict]:
        """
        Applies the rigorous multi-factor filter across all scored tickers.

        Args:
            df_scored:  Output of ScoringEngine, ordered by total_score DESC.
            tech_data:  Output of TechnicalAnalyzer.analyze_all() for the same tickers.

        Returns:
            List of opportunity dicts, sorted by composite conviction score.
            Empty list when nothing qualifies.
        """
        opportunities = []

        for _, row in df_scored.iterrows():
            ticker = str(row.get("ticker", ""))
            tech = tech_data.get(ticker, {})

            if not tech or "price" not in tech:
                continue

            if not self._passes_primary(row, tech):
                continue

            secondary_hits = self._count_secondary(tech)
            if secondary_hits < _MIN_SECONDARY:
                logger.debug(
                    "%s: passou primary mas falhou secondary (%d/%d)",
                    ticker, secondary_hits, _MIN_SECONDARY,
                )
                continue

            opp = self._build_opportunity(row, tech, secondary_hits)
            opportunities.append(opp)
            logger.info(
                "OPORTUNIDADE: %s | score=%.1f | signal=%s | RSI=%.0f | secondary=%d/3",
                ticker,
                row.get("total_score", 0),
                tech.get("signal"),
                tech.get("rsi", 0),
                secondary_hits,
            )

        # Sort by conviction: secondary_hits DESC, then total_score DESC
        opportunities.sort(key=lambda x: (x["secondary_hits"], x["score"]), reverse=True)

        logger.info(
            "Scanner concluído: %d/%d tickers qualificados",
            len(opportunities), len(df_scored),
        )
        return opportunities

    # ─── Filter criteria ──────────────────────────────────────────────────────

    @staticmethod
    def _passes_primary(row: pd.Series, tech: dict) -> bool:
        """All 4 primary criteria must pass."""

        # 1. Fundamental score
        score = row.get("total_score")
        if score is None or pd.isna(score) or float(score) < _MIN_SCORE:
            return False

        # 2. Technical composite signal
        signal = tech.get("signal", "neutral")
        if signal not in ("buy", "strong_buy"):
            return False

        # 3. RSI in recovery zone
        rsi = tech.get("rsi")
        if rsi is None or np.isnan(rsi) or not (_RSI_MIN <= rsi <= _RSI_MAX):
            return False

        # 4. MACD above signal line (momentum direction)
        if not tech.get("macd_above_signal", False):
            return False

        return True

    @staticmethod
    def _count_secondary(tech: dict) -> int:
        """Counts how many secondary criteria are met."""
        hits = 0

        # A. MACD bullish crossover (recent, high timing value)
        if tech.get("macd_crossover") == "bullish":
            hits += 1

        # B. Elevated volume (institutional interest)
        vol_ratio = tech.get("volume_ratio")
        if vol_ratio is not None and not np.isnan(vol_ratio) and vol_ratio >= _VOL_THRESHOLD:
            hits += 1

        # C. Price in lower Bollinger Band (mean-reversion setup)
        bb_pos = tech.get("bb_position")
        if bb_pos is not None and not np.isnan(bb_pos) and bb_pos <= _BB_THRESHOLD:
            hits += 1

        return hits

    # ─── Opportunity builder ──────────────────────────────────────────────────

    def _build_opportunity(
        self,
        row: pd.Series,
        tech: dict,
        secondary_hits: int,
    ) -> dict:
        ticker = str(row.get("ticker", ""))
        price  = tech.get("price") or _safe_float(row.get("current_price"))

        targets   = self._compute_targets(price, tech)
        stop_loss = round(price * (1 - _STOP_LOSS_PCT), 2) if price else None
        risk      = self._risk_level(
            _safe_float(row.get("volatility_180d")),
            _safe_float(row.get("beta")),
        )
        reasons  = self._build_reasons(row, tech)
        horizon  = self._estimate_horizon(row, tech, secondary_hits)

        # Entry range: current price ±2%
        entry_low  = round(price * 0.98, 2) if price else None
        entry_high = round(price * 1.02, 2) if price else None

        return {
            "ticker":            ticker,
            "nome":              str(row.get("nome", "")),
            "setor":             str(row.get("setor", "")),
            "price":             price,
            "entry_low":         entry_low,
            "entry_high":        entry_high,
            "stop_loss":         stop_loss,
            "targets":           targets,
            "risk":              risk,
            "horizon":           horizon,
            "reasons":           reasons,
            "score":             _safe_float(row.get("total_score")),
            "fundamental_score": _safe_float(row.get("fundamental_score")),
            "momentum_score":    _safe_float(row.get("momentum_score")),
            "quality_score":     _safe_float(row.get("quality_score")),
            "signal":            tech.get("signal"),
            "rsi":               tech.get("rsi"),
            "macd_crossover":    tech.get("macd_crossover"),
            "volume_ratio":      tech.get("volume_ratio"),
            "bb_position":       tech.get("bb_position"),
            "alpha_6m":          _safe_float(row.get("alpha_6m")),
            "secondary_hits":    secondary_hits,
            "roe":               _safe_float(row.get("roe")),
            "dy":                _safe_float(row.get("dividend_yield")),
            "pvp":               _safe_float(row.get("pvp")),
        }

    # ─── Price targets ────────────────────────────────────────────────────────

    @staticmethod
    def _compute_targets(price: Optional[float], tech: dict) -> list[dict]:
        """
        Finds the nearest significant resistance levels above current price.

        Priority: MA20 → MA50 → MA200 → 52w high
        Falls back to fixed +10%/+20% if no resistance is found above price.
        """
        if not price or price <= 0:
            return []

        candidates = []
        checks = [
            (tech.get("ma20"),      "MA20"),
            (tech.get("ma50"),      "MA50"),
            (tech.get("ma200"),     "MA200"),
            (tech.get("high_52w"),  "Máx 52 semanas"),
        ]

        for level, label in checks:
            if level and not np.isnan(level) and level > price * (1 + _MIN_TARGET_GAP):
                upside = (level - price) / price
                candidates.append({
                    "level":  round(level, 2),
                    "label":  label,
                    "upside": upside,
                })

        candidates.sort(key=lambda x: x["upside"])

        if not candidates:
            candidates = [
                {"level": round(price * 1.10, 2), "label": "Alvo conservador", "upside": 0.10},
                {"level": round(price * 1.20, 2), "label": "Alvo otimista",    "upside": 0.20},
            ]
        elif len(candidates) == 1:
            first = candidates[0]
            candidates.append({
                "level":  round(first["level"] * 1.08, 2),
                "label":  "Alvo estendido",
                "upside": (first["level"] * 1.08 - price) / price,
            })

        return candidates[:2]

    # ─── Supporting helpers ───────────────────────────────────────────────────

    @staticmethod
    def _risk_level(volatility: Optional[float], beta: Optional[float]) -> str:
        vol = volatility or 0.0
        b   = beta or 1.0
        if vol > 0.40 or b > 1.6:
            return "Alto"
        if vol < 0.22 and b < 0.85:
            return "Baixo"
        return "Moderado"

    @staticmethod
    def _estimate_horizon(row: pd.Series, tech: dict, secondary_hits: int) -> str:
        score  = float(row.get("total_score") or 0)
        signal = tech.get("signal", "neutral")

        if signal == "strong_buy" and secondary_hits == 3:
            return "1–4 semanas"
        if score >= 80 and secondary_hits >= 2:
            return "2–6 semanas"
        return "3–8 semanas"

    @staticmethod
    def _build_reasons(row: pd.Series, tech: dict) -> list[str]:
        """
        Builds 3–5 human-readable reasons that explain why this ticker qualifies.
        Combines scoring engine's 'why' field with real-time technical triggers.
        """
        reasons: list[str] = []

        # Fundamental drivers (from scoring engine explanation)
        why = str(row.get("why", ""))
        skip = {"Dados insuficientes para análise detalhada.", "Score baseado em múltiplos fatores."}
        if why and why not in skip:
            for factor in [f.strip() for f in why.split(";")][:2]:
                if factor:
                    reasons.append(factor)

        # RSI context
        rsi = tech.get("rsi")
        if rsi is not None and not np.isnan(rsi) and _RSI_MIN <= rsi <= 40:
            reasons.append(f"RSI {rsi:.0f} — zona sobrevendida com potencial de recuperação")

        # MACD crossover (high-value timing signal)
        if tech.get("macd_crossover") == "bullish":
            reasons.append("MACD cruzamento altista — momentum revertendo agora")

        # Volume
        vol_ratio = tech.get("volume_ratio")
        if vol_ratio and not np.isnan(vol_ratio) and vol_ratio >= _VOL_THRESHOLD:
            reasons.append(f"Volume {vol_ratio:.1f}× acima da média — interesse institucional")

        # Bollinger Band
        bb_pos = tech.get("bb_position")
        if bb_pos is not None and not np.isnan(bb_pos) and bb_pos <= 0.20:
            reasons.append("Preço na banda inferior de Bollinger — extremo de volatilidade negativa")

        # Alpha momentum (if present)
        alpha = _safe_float(row.get("alpha_6m"))
        if alpha and alpha > 0.08:
            sign = "+" if alpha >= 0 else ""
            reasons.append(f"Alpha 6m {sign}{alpha*100:.1f}% vs IBOV — momentum estrutural positivo")

        return reasons[:5]


# ─── Helper ───────────────────────────────────────────────────────────────────

def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None
