"""
factor_analysis.py — Information Coefficient (IC) framework

Mede o poder preditivo dos fatores do sistema sobre o histórico de
recomendações salvas em data/history/.

Conceitos:
  IC (Information Coefficient): correlação de Spearman entre o score de
    um fator em T0 e o retorno realizado de T0 a T0+N. IC ≈ 0.05 já é
    considerado bom em quant; IC > 0.10 é excelente. IC negativo significa
    que o fator está prevendo ao contrário (problema).

  IR (Information Ratio): mean(IC) / std(IC) ao longo de múltiplos períodos.
    Mede consistência da previsão. IR > 0.5 é geralmente investível.

  Hit rate: % de períodos em que IC > 0 — indica robustez.

Uso:
  CLI:  python -m src.factor_analysis
  Lib:  from src.factor_analysis import analyze_factors
        result = analyze_factors()

Output:
  data/factor_ic.json (tabela completa)
  stdout (tabela formatada)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None

from src.config import HISTORY_DIR, IC_FORWARD_WINDOWS, IC_OUTPUT_PATH

logger = logging.getLogger(__name__)


# ─── Carregamento de histórico ────────────────────────────────────────────────

def _list_recommendations(
    history_dir: Path,
    mode: str = "weekly",
) -> list[dict]:
    """Carrega recomendações em ordem cronológica."""
    files = sorted(history_dir.glob(f"recommendations_*_{mode}.json"))
    out: list[dict] = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                out.append(json.load(fh))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Falha ao ler %s: %s", f, exc)
    return out


def _list_snapshots(history_dir: Path) -> dict[str, dict[str, float]]:
    """Carrega snapshots de preço por data: {date_str: {ticker: price}}."""
    files = sorted(history_dir.glob("snapshot_*.json"))
    out: dict[str, dict[str, float]] = {}
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
                d = data.get("date") or f.stem.replace("snapshot_", "")
                prices = data.get("prices") or {}
                # Filtrar preços nulos
                out[d] = {k: float(v) for k, v in prices.items() if v is not None}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Falha ao ler %s: %s", f, exc)
    return out


# ─── Cálculo de retornos forward ──────────────────────────────────────────────

def _find_forward_snapshot(
    snapshots: dict[str, dict[str, float]],
    start_date: str,
    target_days: int,
) -> Optional[tuple[str, dict[str, float]]]:
    """
    Acha o snapshot mais próximo de start_date + target_days úteis.

    Como os snapshots não são diários, busca o que está mais próximo dentro
    de uma janela de tolerância de ±5 dias corridos.
    """
    if not snapshots:
        return None
    start_ts = pd.Timestamp(start_date)
    target_ts = start_ts + pd.Timedelta(days=int(target_days * 1.45))  # úteis→corridos aprox

    best: Optional[tuple[str, dict[str, float]]] = None
    best_diff = float("inf")
    for d, prices in snapshots.items():
        d_ts = pd.Timestamp(d)
        if d_ts <= start_ts:
            continue
        diff = abs((d_ts - target_ts).days)
        # Janela de tolerância: ±5d corridos
        if diff <= 5 and diff < best_diff:
            best = (d, prices)
            best_diff = diff

    return best


def _forward_returns(
    start_prices: dict[str, float],
    end_prices: dict[str, float],
) -> dict[str, float]:
    """Retorna {ticker: return} para tickers presentes em ambos."""
    out: dict[str, float] = {}
    for t, p0 in start_prices.items():
        p1 = end_prices.get(t)
        if p0 and p1 and p0 > 0:
            out[t] = (p1 / p0) - 1.0
    return out


# ─── Cálculo de IC ────────────────────────────────────────────────────────────

def _spearman_ic(
    factor_scores: dict[str, float],
    forward_returns: dict[str, float],
) -> Optional[tuple[float, int]]:
    """
    Computa Spearman IC entre fator e retorno forward (cross-sectional).

    Returns:
        (ic, n) — IC e tamanho efetivo. None se N < 5 ou variância zero.
    """
    if spearmanr is None:
        return None
    common = sorted(set(factor_scores.keys()) & set(forward_returns.keys()))
    if len(common) < 5:
        return None
    f = np.array([factor_scores[t] for t in common])
    r = np.array([forward_returns[t] for t in common])
    if np.std(f) == 0 or np.std(r) == 0:
        return None
    rho, _ = spearmanr(f, r)
    if np.isnan(rho):
        return None
    return float(rho), len(common)


def _extract_factor_scores(
    rec: dict,
    factor: str,
) -> dict[str, float]:
    """
    Extrai {ticker: factor_score} de uma recomendação.

    O score é o valor normalizado [0,100] guardado em norm_details.
    Considera todos os tickers do top10 (rec só guarda top10 — não o universo
    completo, então IC é estimado sobre o universo recomendado, não global).
    """
    out: dict[str, float] = {}
    for r in rec.get("top10", []):
        ticker = r.get("ticker")
        nd = (r.get("norm_details") or {})
        if factor in nd:
            score = nd[factor].get("score")
            if score is not None:
                out[ticker] = float(score)
    return out


# ─── Pipeline principal ───────────────────────────────────────────────────────

def analyze_factors(
    history_dir: Path = HISTORY_DIR,
    mode: str = "weekly",
    output_path: Path = IC_OUTPUT_PATH,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Analisa IC e IR de todos os fatores sobre o histórico disponível.

    Para cada (recomendação T0, janela forward), busca o snapshot futuro e
    computa Spearman IC entre o score do fator e o retorno realizado.

    Returns:
        dict com estrutura:
          {
            "analysis_date": "YYYY-MM-DD",
            "n_snapshots": int,
            "n_recommendations": int,
            "factors": {
              "earnings_yield": {
                "1w": {"mean_ic": 0.05, "ir": 0.5, "hit_rate": 0.6, "n_obs": 12},
                "4w": {...}, "12w": {...}
              }, ...
            }
          }
    """
    if spearmanr is None:
        logger.error("scipy.stats.spearmanr indisponível — abortando")
        return {}

    history_dir = Path(history_dir)
    recs = _list_recommendations(history_dir, mode)
    snaps = _list_snapshots(history_dir)

    if len(recs) < 1 or len(snaps) < 2:
        logger.warning(
            "Histórico insuficiente: %d recomendações, %d snapshots "
            "(mínimo 1 + 2 para qualquer cálculo)", len(recs), len(snaps)
        )
        result = {
            "analysis_date": datetime.now().isoformat(),
            "n_snapshots": len(snaps),
            "n_recommendations": len(recs),
            "factors": {},
            "note": "Histórico insuficiente — precisa >=2 snapshots e >=1 recomendação",
        }
        _save_result(result, output_path)
        return result

    # Descobrir todos os fatores presentes
    all_factors: set[str] = set()
    for rec in recs:
        for r in rec.get("top10", []):
            for f in (r.get("norm_details") or {}).keys():
                all_factors.add(f)

    # Para cada (rec, factor, window): coletar IC
    # Estrutura intermediária: ic_buckets[factor][window] = [(ic, n, date), ...]
    ic_buckets: dict[str, dict[str, list[tuple[float, int, str]]]] = {
        f: {w: [] for w in IC_FORWARD_WINDOWS} for f in all_factors
    }

    for rec in recs:
        rec_date = rec.get("date")
        if not rec_date:
            continue
        # Preço inicial: usar entry_prices da recomendação (capturado no momento)
        start_prices = rec.get("entry_prices") or {}
        if not start_prices:
            continue

        for window_name, window_days in IC_FORWARD_WINDOWS.items():
            forward = _find_forward_snapshot(snaps, rec_date, window_days)
            if forward is None:
                continue
            end_date, end_prices = forward
            fwd_returns = _forward_returns(start_prices, end_prices)
            if len(fwd_returns) < 5:
                continue

            for factor in all_factors:
                fs = _extract_factor_scores(rec, factor)
                ic_result = _spearman_ic(fs, fwd_returns)
                if ic_result is not None:
                    ic, n = ic_result
                    ic_buckets[factor][window_name].append((ic, n, rec_date))

    # Agregar
    factors_summary: dict[str, dict[str, dict[str, Any]]] = {}
    for factor, by_window in ic_buckets.items():
        factors_summary[factor] = {}
        for window_name, observations in by_window.items():
            if not observations:
                factors_summary[factor][window_name] = {
                    "mean_ic": None, "ir": None, "hit_rate": None,
                    "n_obs": 0,
                }
                continue
            ics = np.array([o[0] for o in observations])
            mean_ic = float(np.mean(ics))
            std_ic = float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0
            ir = float(mean_ic / std_ic) if std_ic > 0 else None
            hit_rate = float((ics > 0).sum() / len(ics))
            factors_summary[factor][window_name] = {
                "mean_ic":  round(mean_ic, 4),
                "ir":       round(ir, 4) if ir is not None else None,
                "hit_rate": round(hit_rate, 4),
                "n_obs":    len(observations),
            }

    # ── Decay analysis: ajustar curva exponencial IC(τ) = IC₀ × exp(-λτ)
    # Half-life = ln(2) / λ — em dias úteis. Fatores com half-life longa
    # (>40d) são duráveis; <10d são noise-driven.
    decay_summary = _compute_decay(factors_summary, IC_FORWARD_WINDOWS)

    result = {
        "analysis_date":      datetime.now().isoformat(),
        "n_snapshots":        len(snaps),
        "n_recommendations":  len(recs),
        "forward_windows_days": IC_FORWARD_WINDOWS,
        "factors":            factors_summary,
        "decay":              decay_summary,
        "interpretation": {
            "mean_ic":  "Correlação de Spearman média entre score do fator e retorno forward. "
                        "IC > 0.05 é bom; > 0.10 excelente. IC < 0 é vermelho — fator prevê ao contrário.",
            "ir":       "Information Ratio = mean(IC)/std(IC). > 0.5 sugere fator robusto.",
            "hit_rate": "% de períodos com IC > 0. > 0.55 é desejável.",
            "half_life": "Dias úteis para IC cair pela metade. > 40d = durável; < 10d = ruidoso.",
        },
    }

    _save_result(result, output_path)
    if verbose:
        _print_table(result)
    return result


def _compute_decay(
    factors_summary: dict,
    forward_windows: dict,
) -> dict:
    """
    Ajusta IC(τ) = IC₀ × exp(-λτ) e calcula half-life.

    Usa pelo menos 2 pontos (janelas com n_obs > 0 e mean_ic > 0). Se < 2
    pontos válidos, retorna half_life=None.

    OLS em log-space: log|IC| = log|IC₀| - λτ
    λ = -slope, half_life = ln(2) / λ.

    Aceita só IC positivo (interpretação de decay só faz sentido se fator
    está predizendo). Fatores com IC negativo recebem half_life=None.
    """
    out: dict[str, dict] = {}
    for factor, by_window in factors_summary.items():
        points: list[tuple[float, float]] = []  # (τ, IC)
        for window_name, days in forward_windows.items():
            entry = by_window.get(window_name, {})
            ic = entry.get("mean_ic")
            n = entry.get("n_obs", 0)
            if ic is None or n == 0 or ic <= 0:
                continue
            points.append((float(days), float(ic)))

        if len(points) < 2:
            out[factor] = {"half_life_days": None, "n_points": len(points)}
            continue

        # OLS em log-space
        tau = np.array([p[0] for p in points])
        log_ic = np.log(np.array([p[1] for p in points]))
        try:
            slope, intercept = np.polyfit(tau, log_ic, 1)
            if slope >= 0:  # sem decaimento — IC constante ou crescendo
                out[factor] = {"half_life_days": None, "n_points": len(points), "trend": "stable_or_growing"}
                continue
            lambda_ = -slope
            half_life = float(np.log(2) / lambda_)
            # IC inicial extrapolado
            ic_zero = float(np.exp(intercept))
            out[factor] = {
                "half_life_days":   round(half_life, 1),
                "ic_zero":          round(ic_zero, 4),
                "decay_rate":       round(lambda_, 5),
                "n_points":         len(points),
            }
        except (np.linalg.LinAlgError, ValueError):
            out[factor] = {"half_life_days": None, "n_points": len(points)}

    return out


def _save_result(result: dict, output_path: Path) -> None:
    """Salva o resultado em JSON com escrita atômica."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        tmp.replace(output_path)
        logger.info("Factor IC salvo em %s", output_path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        logger.error("Falha ao salvar IC: %s", exc)


def _print_table(result: dict) -> None:
    """Imprime tabela de IC formatada no stdout."""
    print()
    print("=" * 72)
    _safe_print(f"  Factor IC Analysis — {result['analysis_date'][:10]}")
    _safe_print(f"  Snapshots: {result['n_snapshots']} | Recomendacoes: {result['n_recommendations']}")
    print("=" * 72)

    factors = result.get("factors", {})
    if not factors:
        note = result.get("note", "Sem dados.")
        print(f"  {note}")
        print("=" * 72)
        return

    windows = list(IC_FORWARD_WINDOWS.keys())
    header = f"  {'Factor':<24}" + "".join(f"  {w:>16}" for w in windows)
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Ordenar por mean_ic do primeiro window disponível (descrescente)
    def _sort_key(item):
        f, by_w = item
        for w in windows:
            mi = by_w.get(w, {}).get("mean_ic")
            if mi is not None:
                return -mi
        return 0

    for factor, by_window in sorted(factors.items(), key=_sort_key):
        row = f"  {factor:<24}"
        for w in windows:
            entry = by_window.get(w, {})
            ic = entry.get("mean_ic")
            n = entry.get("n_obs", 0)
            if ic is None or n == 0:
                row += f"  {'-':>16}"
            else:
                marker = "++" if ic > 0.05 else (" +" if ic > 0 else "--")
                row += f"  {marker} {ic:+.3f} (n={n:>2})"
        _safe_print(row)

    print("=" * 72)
    _safe_print("  ++ IC > 0.05 (bom)   + IC > 0 (marginal)   -- IC < 0 (prevê ao contrário)")
    print("=" * 72)
    print()


def _safe_print(text: str) -> None:
    """Print resiliente a cp1252 (Windows): substitui chars não encodáveis."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback: usa stdout.buffer com utf-8 e replace
        import sys
        sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))
        sys.stdout.buffer.flush()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    """Entry point CLI: python -m src.factor_analysis"""
    _setup_logging()
    analyze_factors()


if __name__ == "__main__":
    main()
