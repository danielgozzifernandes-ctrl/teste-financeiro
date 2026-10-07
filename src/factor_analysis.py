"""
IC (Information Coefficient) dos fatores sobre as recomendações salvas.

IC = Spearman entre o score do fator em T0 e o retorno forward, no universo
pontuado inteiro (full_universe_scores). Medir só no top-10 correlaciona o
fator com ações que ele mesmo escolheu.

Retorno forward: fechamentos ajustados (mesma série nas duas pontas). Entrada
no primeiro fechamento depois da execução; saída h pregões depois.

Significância: t de Newey-West sobre a série de IC (lag = sobreposição das
janelas, recomendações semanais) e Benjamini-Hochberg sobre todos os pares
fator × horizonte. Com a amostra ao vivo o IC é monitor, não gatilho para
mudar pesos — a análise de poder vai no JSON.

  CLI:  python -m src.factor_analysis
  Lib:  from src.factor_analysis import analyze_factors

Saída: data/factor_ic.json + tabela no stdout.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

try:
    from scipy.stats import norm, spearmanr, t as student_t
except ImportError:
    spearmanr = None

from src.config import (
    HISTORY_DIR,
    IC_FORWARD_WINDOWS,
    IC_OUTPUT_PATH,
    MIN_OBS_FOR_SIGNIFICANCE,
)

logger = logging.getLogger(__name__)

MIN_TICKERS_PER_IC = 10
FDR_LEVEL = 0.05
SESSION_CLOSE_BRT_HOUR = 17
RECS_PER_WEEK = 1   # recomendações semanais → sobreposição = ceil(h/5) - 1

PriceLoader = Callable[[list[str], str, str], pd.DataFrame]


# Histórico

def _list_recommendations(history_dir: Path, mode: str = "weekly") -> list[dict]:
    """Recomendações em ordem cronológica, sem duplicatas de fim de semana."""
    out: list[dict] = []
    for f in sorted(history_dir.glob(f"recommendations_*_{mode}.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                rec = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Falha ao ler %s: %s", f, exc)
            continue
        # Duas recomendações a 2 dias uma da outra (ex.: sexta e sábado)
        # são a mesma carteira; contar as duas infla a amostra.
        if out and rec.get("date") and out[-1].get("date"):
            gap = (pd.Timestamp(rec["date"]) - pd.Timestamp(out[-1]["date"])).days
            if gap <= 2:
                continue
        out.append(rec)
    return out


def _run_time_brt(rec: dict) -> pd.Timestamp:
    """Horário da execução em BRT. Arquivos antigos só têm generated_at do runner (UTC)."""
    md = (rec.get("execution_metadata") or {}).get("market_data") or {}
    if md.get("run_at_brt"):
        return pd.Timestamp(md["run_at_brt"]).tz_localize(None)
    gen = (rec.get("execution_metadata") or {}).get("generated_at")
    if gen:
        return pd.Timestamp(gen) - pd.Timedelta(hours=3)
    return pd.Timestamp(rec["date"]) + pd.Timedelta(hours=12)


def _entry_index(dates: pd.DatetimeIndex, run_at: pd.Timestamp) -> Optional[int]:
    """Primeiro fechamento posterior à execução."""
    day = run_at.normalize()
    pos = int(dates.searchsorted(day))
    if pos >= len(dates):
        return None
    if dates[pos] == day and run_at.hour >= SESSION_CLOSE_BRT_HOUR:
        pos += 1
    return pos if pos < len(dates) else None


def _load_prices_yf(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Fechamentos ajustados (.SA), sem candle incompleto nem pregão em aberto."""
    import yfinance as yf

    from src.benchmark import is_intraday, now_brt

    raw = yf.download([f"{t}.SA" for t in tickers], start=start, end=end,
                      auto_adjust=True, progress=False, threads=True)
    if raw is None or raw.empty:
        return pd.DataFrame()
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    close = raw["Close"].copy()
    for c in ("Open", "Volume"):
        if c in raw.columns.get_level_values(0):
            close = close.where(raw[c].reindex(columns=close.columns).fillna(0) > 0)
    now = now_brt()
    if is_intraday(now):
        close = close[close.index.date < now.date()]
    close.columns = [c.replace(".SA", "") for c in close.columns]
    return close.dropna(how="all")


# Estatística

def _spearman_ic(factor_scores: dict[str, float], forward_returns: dict[str, float]) -> Optional[tuple[float, int]]:
    common = sorted(set(factor_scores) & set(forward_returns))
    if len(common) < MIN_TICKERS_PER_IC:
        return None
    f = np.array([factor_scores[t] for t in common])
    r = np.array([forward_returns[t] for t in common])
    if np.std(f) == 0 or np.std(r) == 0:
        return None
    rho, _ = spearmanr(f, r)
    if np.isnan(rho):
        return None
    return float(rho), len(common)


def newey_west_t(x: np.ndarray, lag: int) -> Optional[float]:
    """t da média com erro-padrão Newey-West (kernel de Bartlett)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return None
    d = x - x.mean()
    var = float(d @ d) / n
    for k in range(1, min(lag, n - 1) + 1):
        var += 2 * (1 - k / (lag + 1)) * float(d[k:] @ d[:-k]) / n
    if var <= 0:
        return None
    return float(x.mean() / math.sqrt(var / n))


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """q-values de Benjamini-Hochberg na ordem de entrada."""
    m = len(pvalues)
    if m == 0:
        return []
    order = np.argsort(pvalues)
    q = np.empty(m)
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, pvalues[i] * m / rank)
        q[i] = prev
    return [float(v) for v in q]


def _extract_factor_scores(rec: dict, factor: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for ticker, scores in (rec.get("full_universe_scores") or {}).items():
        v = scores.get(factor)
        if v is not None:
            out[ticker] = float(v)
    return out


# Pipeline

def analyze_factors(
    history_dir: Path = HISTORY_DIR,
    mode: str = "weekly",
    output_path: Optional[Path] = IC_OUTPUT_PATH,
    verbose: bool = True,
    price_loader: Optional[PriceLoader] = None,
    exclude_from: Optional[str] = None,
) -> dict[str, Any]:
    """
    IC por fator e horizonte.

    exclude_from: descarta observações cuja janela termina nessa data ou depois
    (para separar uma janela atípica).
    """
    if spearmanr is None:
        logger.error("scipy indisponível — abortando")
        return {}

    history_dir = Path(history_dir)
    recs_all = _list_recommendations(history_dir, mode)
    recs = [r for r in recs_all if r.get("full_universe_scores") and r.get("date")]

    result: dict[str, Any] = {
        "analysis_date": datetime.now().isoformat(),
        "n_recommendations": len(recs_all),
        "n_recommendations_full_universe": len(recs),
        "forward_windows_days": IC_FORWARD_WINDOWS,
        "usage": "monitor",
        "usage_note": (
            "IC ao vivo é monitor de saúde do modelo, não gatilho para mudar "
            "pesos: a amostra não tem poder para detectar IC realista (ver "
            "power). Decisão de pesos só com backtest histórico point-in-time."
        ),
        "factors": {},
    }

    if not recs:
        result["data_sufficiency"] = {
            "min_obs_required": MIN_OBS_FOR_SIGNIFICANCE,
            "max_obs_available": 0,
            "is_significant": False,
            "warning": "Sem recomendações com full_universe_scores para medir IC.",
        }
        result["note"] = "Histórico insuficiente"
        _finish(result, output_path, verbose)
        return result

    tickers = sorted({t for r in recs for t in r["full_universe_scores"]})
    start = (pd.Timestamp(recs[0]["date"]) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    loader = price_loader or _load_prices_yf
    try:
        prices = loader(tickers, start, end)
    except Exception as exc:
        logger.warning("Preços para o IC indisponíveis: %s", exc)
        prices = pd.DataFrame()
    if prices.empty:
        result["data_sufficiency"] = {
            "min_obs_required": MIN_OBS_FOR_SIGNIFICANCE,
            "max_obs_available": 0,
            "is_significant": False,
            "warning": "Sem preços para calcular retornos forward.",
        }
        _finish(result, output_path, verbose)
        return result
    prices = prices.sort_index()
    dates = prices.index
    cutoff = pd.Timestamp(exclude_from) if exclude_from else None

    all_factors = sorted({f for r in recs for s in r["full_universe_scores"].values() for f in s})
    obs: dict[str, dict[str, list[tuple[float, int, str]]]] = {
        f: {w: [] for w in IC_FORWARD_WINDOWS} for f in all_factors
    }

    for rec in recs:
        i0 = _entry_index(dates, _run_time_brt(rec))
        if i0 is None:
            continue
        p0 = prices.iloc[i0]
        for w, h in IC_FORWARD_WINDOWS.items():
            i1 = i0 + h
            if i1 >= len(dates):
                continue
            if cutoff is not None and dates[i1] >= cutoff:
                continue
            fwd = (prices.iloc[i1] / p0 - 1).dropna()
            fwd = {t: float(v) for t, v in fwd.items() if np.isfinite(v)}
            for f in all_factors:
                ic = _spearman_ic(_extract_factor_scores(rec, f), fwd)
                if ic is not None:
                    obs[f][w].append((ic[0], ic[1], rec["date"]))

    # Agregação + testes
    tests: list[tuple[str, str, float]] = []
    summary: dict[str, dict[str, dict]] = {}
    for f, by_w in obs.items():
        summary[f] = {}
        for w, o in by_w.items():
            h = IC_FORWARD_WINDOWS[w]
            if not o:
                summary[f][w] = {"mean_ic": None, "ir": None, "hit_rate": None,
                                 "n_obs": 0, "significant": False}
                continue
            ics = np.array([x[0] for x in o])
            lag = max(0, math.ceil(h / 5) * RECS_PER_WEEK - 1)
            std = float(np.std(ics, ddof=1)) if len(ics) > 1 else None
            n_eff = len(ics) / (lag + 1)
            # Com poucas janelas independentes o Newey-West degenera (janelas
            # quase idênticas → variância ~0 → t absurdo). Aí não há teste.
            t_nw = newey_west_t(ics, lag) if n_eff >= MIN_OBS_FOR_SIGNIFICANCE else None
            p = (float(2 * student_t.sf(abs(t_nw), df=max(n_eff - 1, 1)))
                 if t_nw is not None else None)
            summary[f][w] = {
                "mean_ic":   round(float(ics.mean()), 4),
                "ir":        round(float(ics.mean() / std), 4) if std else None,
                "hit_rate":  round(float((ics > 0).mean()), 4),
                "n_obs":     len(ics),
                "n_eff":     round(n_eff, 1),
                "testable":  t_nw is not None,
                "nw_lag":    lag,
                "t_nw":      round(t_nw, 3) if t_nw is not None else None,
                "p_value":   round(p, 4) if p is not None else None,
                "mean_n_tickers": round(float(np.mean([x[1] for x in o])), 1),
                "significant": False,
            }
            if p is not None:
                tests.append((f, w, p))

    qs = benjamini_hochberg([t[2] for t in tests])
    for (f, w, _), q in zip(tests, qs):
        e = summary[f][w]
        e["q_value"] = round(q, 4)
        e["significant"] = bool(q <= FDR_LEVEL and e["n_obs"] >= MIN_OBS_FOR_SIGNIFICANCE)

    n_sig = sum(e.get("significant", False) for by_w in summary.values() for e in by_w.values())
    max_obs = max((e.get("n_obs", 0) for by_w in summary.values() for e in by_w.values()), default=0)
    result["factors"] = summary
    result["decay"] = _compute_decay(summary, IC_FORWARD_WINDOWS)
    result["n_tests"] = len(tests)
    result["power"] = _power(obs)
    result["data_sufficiency"] = {
        "min_obs_required": MIN_OBS_FOR_SIGNIFICANCE,
        "max_obs_available": max_obs,
        "test": f"Newey-West + Benjamini-Hochberg (FDR {FDR_LEVEL:.0%}) sobre {len(tests)} testes",
        "n_significant": n_sig,
        "is_significant": n_sig > 0,
        "warning": None if n_sig else (
            f"Nenhum fator significativo após correção para {len(tests)} testes "
            f"(máx. {max_obs} observações). IC/IR/hit-rate abaixo são ruído."
        ),
    }
    if exclude_from:
        result["excluded_windows_ending_from"] = exclude_from
    result["price_source"] = "yfinance auto_adjust" if price_loader is None else "custom"

    _finish(result, output_path, verbose)
    return result


def _power(obs: dict) -> dict:
    """Semanas necessárias para detectar IC de 0,03/0,05 (80% de poder, 5% bilateral)."""
    sds = [np.std([x[0] for x in by_w["1w"]], ddof=1)
           for by_w in obs.values() if len(by_w.get("1w", [])) > 2]
    if not sds:
        return {}
    sd = float(np.median(sds))
    z = norm.ppf(0.975) + norm.ppf(0.8)
    return {
        "ic_std_1w_median": round(sd, 4),
        "weeks_needed_ic_0.05": int(math.ceil((z * sd / 0.05) ** 2)),
        "weeks_needed_ic_0.03": int(math.ceil((z * sd / 0.03) ** 2)),
    }


def _compute_decay(factors_summary: dict, forward_windows: dict) -> dict:
    """IC(τ) = IC₀·exp(−λτ) em log-space; só com IC médio positivo em ≥ 2 horizontes."""
    out: dict[str, dict] = {}
    for factor, by_window in factors_summary.items():
        points = [
            (float(days), float(by_window[w]["mean_ic"]))
            for w, days in forward_windows.items()
            if by_window.get(w, {}).get("mean_ic") is not None
            and by_window[w].get("n_obs", 0) > 0 and by_window[w]["mean_ic"] > 0
        ]
        if len(points) < 2:
            out[factor] = {"half_life_days": None, "n_points": len(points)}
            continue
        tau = np.array([p[0] for p in points])
        log_ic = np.log(np.array([p[1] for p in points]))
        try:
            slope, intercept = np.polyfit(tau, log_ic, 1)
        except (np.linalg.LinAlgError, ValueError):
            out[factor] = {"half_life_days": None, "n_points": len(points)}
            continue
        if slope >= 0:
            out[factor] = {"half_life_days": None, "n_points": len(points),
                           "trend": "stable_or_growing"}
            continue
        out[factor] = {
            "half_life_days": round(float(np.log(2) / -slope), 1),
            "ic_zero":        round(float(np.exp(intercept)), 4),
            "decay_rate":     round(float(-slope), 5),
            "n_points":       len(points),
        }
    return out


def _finish(result: dict, output_path: Optional[Path], verbose: bool) -> None:
    if output_path is not None:
        _save_result(result, output_path)
    if verbose:
        _print_table(result)


def _save_result(result: dict, output_path: Path) -> None:
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
    print()
    print("=" * 84)
    _safe_print(f"  Factor IC — {result['analysis_date'][:10]} | "
                f"recomendações com universo completo: {result.get('n_recommendations_full_universe', 0)}")
    print("=" * 84)
    ds = result.get("data_sufficiency", {})
    if ds and not ds.get("is_significant", True):
        _safe_print(f"  [!] {ds.get('warning', 'Amostra insuficiente.')}")
        print("=" * 84)

    factors = result.get("factors", {})
    if not factors:
        print(f"  {result.get('note', 'Sem dados.')}")
        print("=" * 84)
        return

    windows = list(IC_FORWARD_WINDOWS.keys())
    print(f"  {'Factor':<24}" + "".join(f"  {w:>18}" for w in windows))
    for factor, by_window in sorted(factors.items()):
        row = f"  {factor:<24}"
        for w in windows:
            e = by_window.get(w, {})
            if not e.get("n_obs"):
                row += f"  {'-':>18}"
            else:
                mark = "*" if e.get("significant") else " "
                t = e.get("t_nw")
                row += f"  {e['mean_ic']:+.3f} t={t if t is not None else float('nan'):+.1f}{mark}(n={e['n_obs']:>2})"
        _safe_print(row)
    print("=" * 84)
    _safe_print("  * significativo após Newey-West + Benjamini-Hochberg")
    print()


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        import sys
        sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))
        sys.stdout.buffer.flush()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    analyze_factors()


if __name__ == "__main__":
    main()
