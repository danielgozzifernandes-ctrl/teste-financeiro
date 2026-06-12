"""
Equity Curve — a régua do investidor absoluto.

Mantém em `data/equity_curve.json` a série diária encadeada do NAV do
sistema (base 100) contra os DOIS benchmarks que importam:

    nav      — carteira top-5, retorno diário ponderado encadeado entre
               rebalanceamentos (a curva sobrevive à troca de carteira)
    ibov_nav — Ibovespa (benchmark relativo)
    cdi_nav  — CDI (benchmark ABSOLUTO — o custo de oportunidade real de
               um PF com Selic alta; é a régua que decide se o sistema
               merece existir)

Alimentada pelo closing diário (que já roda Seg–Sex no CI e commita
data/*.json de volta ao repo). Append-only com dedup por data — re-runs do
mesmo dia sobrescrevem a própria linha, nunca duplicam.

Também fornece a banda de ruído do alpha diário: com a série acumulada,
o relatório pode dizer se o alpha de HOJE é sinal ou ruído (|alpha| < 1σ),
combatendo a leitura ansiosa de resultados diários de uma estratégia
semanal.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

EQUITY_CURVE_PATH = DATA_DIR / "equity_curve.json"

# Banda de ruído default (pp) usada enquanto a série tem <10 observações.
# Estimativa conservadora: carteira 5 ações B3 vs IBOV, tracking error
# diário típico ~1.0-1.5pp.
DEFAULT_ALPHA_NOISE_PP = 1.2
MIN_OBS_FOR_NOISE_BAND = 10


def update_equity_curve(
    run_date: str,
    portfolio_daily_return: Optional[float],
    ibov_daily_return: Optional[float],
    cdi_daily_return: Optional[float] = None,
    blended_daily_return: Optional[float] = None,
    path: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Acrescenta (ou sobrescreve) o ponto do dia e persiste.

    Duas curvas de sistema:
      nav         — sleeve de bolsa (top-5, pesos reais da recomendação)
      blended_nav — CARTEIRA COMPLETA recomendada (todos os sleeves).
                    É a régua do investidor absoluto: mede exatamente o
                    que o sistema mandou fazer, não só a parte de bolsa.

    Retornos None são tratados como 0 no encadeamento mas registrados como
    null no JSON — a lacuna fica auditável, não escondida.

    Returns:
        Resumo do estado: {n_obs, nav, ibov_nav, cdi_nav, blended_nav,
                           cum_return, cum_ibov, cum_cdi, cum_blended,
                           alpha_vs_ibov_pp, alpha_vs_cdi_pp,
                           blended_alpha_vs_cdi_pp}
    """
    path = path or EQUITY_CURVE_PATH
    series = _load(path)

    # Dedup: re-run do mesmo dia substitui a própria linha
    series = [p for p in series if p.get("date") != run_date]

    prev = series[-1] if series else None
    prev_nav = float(prev["nav"]) if prev else 100.0
    prev_ibov = float(prev["ibov_nav"]) if prev else 100.0
    prev_cdi = float(prev["cdi_nav"]) if prev else 100.0
    prev_blend = float(prev.get("blended_nav") or prev["nav"]) if prev else 100.0

    # CDI fallback: última taxa diária conhecida da própria série — taxa
    # administrada muda raramente; melhor que zerar o benchmark absoluto.
    if cdi_daily_return is None and prev is not None:
        cdi_daily_return_eff = _implied_daily(prev, series)
    else:
        cdi_daily_return_eff = cdi_daily_return

    point = {
        "date":        run_date,
        "port_ret":    _round(portfolio_daily_return),
        "ibov_ret":    _round(ibov_daily_return),
        "cdi_ret":     _round(cdi_daily_return),  # null se não veio — auditável
        "blended_ret": _round(blended_daily_return),
        "nav":         round(prev_nav * (1.0 + (portfolio_daily_return or 0.0)), 6),
        "ibov_nav":    round(prev_ibov * (1.0 + (ibov_daily_return or 0.0)), 6),
        "cdi_nav":     round(prev_cdi * (1.0 + (cdi_daily_return_eff or 0.0)), 6),
        "blended_nav": round(prev_blend * (1.0 + (blended_daily_return or 0.0)), 6),
    }
    series.append(point)
    series.sort(key=lambda p: p["date"])

    _save(path, series)
    summary = summarize(series)
    logger.info(
        "Equity curve: n=%d | NAV %.2f | IBOV %.2f | CDI %.2f | "
        "alpha vs CDI %+.2fpp",
        summary["n_obs"], summary["nav"], summary["ibov_nav"],
        summary["cdi_nav"], summary["alpha_vs_cdi_pp"],
    )
    return summary


def summarize(series: Optional[list[dict]] = None,
              path: Optional[Path] = None) -> dict[str, Any]:
    """Resumo acumulado da curva (carrega do disco se series=None)."""
    if series is None:
        series = _load(path or EQUITY_CURVE_PATH)
    if not series:
        return {
            "n_obs": 0, "nav": 100.0, "ibov_nav": 100.0, "cdi_nav": 100.0,
            "blended_nav": None,
            "cum_return": 0.0, "cum_ibov": 0.0, "cum_cdi": 0.0,
            "cum_blended": None,
            "alpha_vs_ibov_pp": 0.0, "alpha_vs_cdi_pp": 0.0,
            "blended_alpha_vs_cdi_pp": None,
            "since": None,
        }
    last = series[-1]
    nav, ibov, cdi = float(last["nav"]), float(last["ibov_nav"]), float(last["cdi_nav"])
    cum, cum_i, cum_c = nav / 100 - 1, ibov / 100 - 1, cdi / 100 - 1

    # Blended só é reportado se ALGUM ponto teve retorno blended real —
    # senão a curva seria só o encadeamento de zeros (mentira silenciosa).
    has_blended = any(p.get("blended_ret") is not None for p in series)
    blended = float(last.get("blended_nav") or 0.0) if has_blended else None
    cum_b = (blended / 100 - 1) if blended is not None else None

    return {
        "n_obs":            len(series),
        "nav":              nav,
        "ibov_nav":         ibov,
        "cdi_nav":          cdi,
        "blended_nav":      blended,
        "cum_return":       round(cum, 6),
        "cum_ibov":         round(cum_i, 6),
        "cum_cdi":          round(cum_c, 6),
        "cum_blended":      round(cum_b, 6) if cum_b is not None else None,
        "alpha_vs_ibov_pp": round((cum - cum_i) * 100, 4),
        "alpha_vs_cdi_pp":  round((cum - cum_c) * 100, 4),
        "blended_alpha_vs_cdi_pp": (round((cum_b - cum_c) * 100, 4)
                                    if cum_b is not None else None),
        "since":            series[0]["date"],
    }


def alpha_noise_band_pp(series: Optional[list[dict]] = None,
                        path: Optional[Path] = None) -> float:
    """
    1σ do alpha diário (port − ibov) em pp, da série histórica real.

    Com <MIN_OBS_FOR_NOISE_BAND observações usa DEFAULT_ALPHA_NOISE_PP —
    melhor um prior conservador declarado que fingir precisão com n=3.
    """
    if series is None:
        series = _load(path or EQUITY_CURVE_PATH)
    alphas = [
        (p["port_ret"] - p["ibov_ret"]) * 100
        for p in series
        if p.get("port_ret") is not None and p.get("ibov_ret") is not None
    ]
    if len(alphas) < MIN_OBS_FOR_NOISE_BAND:
        return DEFAULT_ALPHA_NOISE_PP
    return float(np.std(alphas, ddof=1))


# ─── IO ──────────────────────────────────────────────────────────────────────

def _load(path: Path) -> list[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("series", []) if isinstance(data, dict) else []
    except FileNotFoundError:
        return []
    except Exception as exc:
        logger.warning("equity_curve corrompida (%s) — recomeçando", exc)
        return []


def _save(path: Path, series: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"series": series}, f, ensure_ascii=False, indent=1)
    tmp.replace(path)


def _implied_daily(prev: dict, series: list[dict]) -> Optional[float]:
    """Última taxa diária de CDI não-nula registrada na série."""
    for p in reversed(series):
        if p.get("cdi_ret") is not None:
            return float(p["cdi_ret"])
    return None


def _round(v: Optional[float], nd: int = 6) -> Optional[float]:
    return round(float(v), nd) if v is not None else None
