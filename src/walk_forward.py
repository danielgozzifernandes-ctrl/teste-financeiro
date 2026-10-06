"""
Walk-forward fora da amostra das recomendações salvas.

Para cada recomendação e horizonte h (pregões): compra o top-5 com os pesos
gravados no primeiro fechamento depois da execução e vende h pregões depois.
Carteira, IBOV e CDI medidos na mesma janela; preços ajustados por proventos
(o IBOV também é de retorno total). Ticker sem preço fica com retorno 0.

Janelas de 4w/12w começam toda semana e se sobrepõem: o t do alpha usa
Newey-West, e Sharpe/drawdown/acumulado usam só janelas não sobrepostas.
Sharpe em excesso ao CDI.

CLI:  python -m src.walk_forward
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

from src.config import DATA_DIR, HISTORY_DIR, MIN_PERIODS_WALK_FORWARD
from src.factor_analysis import (
    _entry_index,
    _list_recommendations,
    _load_prices_yf,
    _run_time_brt,
    newey_west_t,
)

logger = logging.getLogger(__name__)

WALK_FORWARD_OUTPUT_PATH = DATA_DIR / "walk_forward.json"
DEFAULT_WINDOWS = {"1w": 5, "4w": 20, "12w": 60}

PriceLoader = Callable[[list[str], str, str], pd.DataFrame]
BenchLoader = Callable[[str, str], pd.DataFrame]   # retornos diários: ibovespa, cdi


def _load_bench(start: str, end: str) -> pd.DataFrame:
    from src.benchmark import BenchmarkManager

    return BenchmarkManager().get_returns(start, end)


def _weights(rec: dict) -> dict[str, float]:
    tickers = [r.get("ticker") for r in rec.get("top5", []) if r.get("ticker")]
    stored = rec.get("portfolio_weights") or {}
    raw = {t: float(stored.get(t) or 0) for t in tickers}
    if sum(raw.values()) <= 0:
        raw = {t: 1.0 for t in tickers}
    total = sum(raw.values())
    return {t: w / total for t, w in raw.items()}


def _compound(daily: pd.Series, d0: pd.Timestamp, d1: pd.Timestamp) -> Optional[float]:
    """Retorno composto dos dias em (d0, d1]."""
    s = daily[(daily.index > d0) & (daily.index <= d1)].dropna()
    if s.empty:
        return None
    return float((1 + s).prod() - 1)


def walk_forward_backtest(
    history_dir: Path = HISTORY_DIR,
    mode: str = "weekly",
    forward_windows: Optional[dict[str, int]] = None,
    output_path: Optional[Path] = WALK_FORWARD_OUTPUT_PATH,
    verbose: bool = True,
    price_loader: Optional[PriceLoader] = None,
    bench_loader: Optional[BenchLoader] = None,
    exclude_from: Optional[str] = None,
) -> dict[str, Any]:
    forward_windows = forward_windows or DEFAULT_WINDOWS
    recs = [r for r in _list_recommendations(Path(history_dir), mode) if r.get("top5")]

    result: dict[str, Any] = {
        "analysis_date": datetime.now().isoformat(),
        "n_recommendations": len(recs),
        "per_window": {},
        "per_period": [],
    }
    if not recs:
        result["data_sufficiency"] = {
            "min_periods_required": MIN_PERIODS_WALK_FORWARD,
            "max_periods_available": 0,
            "is_significant": False,
            "warning": "Histórico insuficiente — nenhuma recomendação com top5.",
        }
        result["note"] = "Histórico insuficiente"
        _finish(result, output_path, verbose)
        return result

    tickers = sorted({t for r in recs for t in _weights(r)})
    start = (pd.Timestamp(recs[0]["date"]) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        prices = (price_loader or _load_prices_yf)(tickers, start, end).sort_index()
        bench = (bench_loader or _load_bench)(start, end)
    except Exception as exc:
        logger.warning("Walk-forward sem dados: %s", exc)
        prices, bench = pd.DataFrame(), pd.DataFrame()
    if prices.empty or bench.empty:
        result["data_sufficiency"] = {
            "min_periods_required": MIN_PERIODS_WALK_FORWARD,
            "max_periods_available": 0,
            "is_significant": False,
            "warning": "Sem preços ou benchmark para o período.",
        }
        _finish(result, output_path, verbose)
        return result

    bench.index = pd.to_datetime(bench.index)
    dates = prices.index
    filled = prices.ffill()   # deslistado/OPA: último preço, retorno 0 dali em diante
    cutoff = pd.Timestamp(exclude_from) if exclude_from else None
    per_window: dict[str, dict] = {}
    detail: list[dict] = []

    for w, h in forward_windows.items():
        periods: list[dict] = []
        for rec in recs:
            i0 = _entry_index(dates, _run_time_brt(rec))
            if i0 is None or i0 + h >= len(dates):
                continue
            d0, d1 = dates[i0], dates[i0 + h]
            if cutoff is not None and d1 >= cutoff:
                continue
            port, missing = 0.0, []
            for t, wt in _weights(rec).items():
                p0 = prices[t].iloc[i0] if t in prices else np.nan
                p1 = filled[t].iloc[i0 + h] if t in filled else np.nan
                if not (np.isfinite(p0) and np.isfinite(p1)) or p0 <= 0:
                    missing.append(t)
                    continue
                port += wt * (p1 / p0 - 1)
            ibov = _compound(bench.get("ibovespa", pd.Series(dtype=float)), d0, d1)
            cdi = _compound(bench.get("cdi", pd.Series(dtype=float)), d0, d1)
            periods.append({
                "rec_date":  rec["date"],
                "entry_date": d0.strftime("%Y-%m-%d"),
                "exit_date": d1.strftime("%Y-%m-%d"),
                "window":    w,
                "port_ret":  round(port, 6),
                "ibov_ret":  round(ibov, 6) if ibov is not None else None,
                "cdi_ret":   round(cdi, 6) if cdi is not None else None,
                "alpha":     round(port - ibov, 6) if ibov is not None else None,
                "excess_cdi": round(port - cdi, 6) if cdi is not None else None,
                "missing":   missing,
            })
        per_window[w] = _summarize(periods, h)
        detail.extend(periods)

    max_eff = max((m.get("n_eff", 0) for m in per_window.values()), default=0)
    n_sig = sum(bool(m.get("significant")) for m in per_window.values())
    result.update({
        "per_window": per_window,
        "per_period": detail,
        "decay_alpha": {w: m["mean_alpha_ibov"] for w, m in per_window.items()
                        if m.get("mean_alpha_ibov") is not None},
        "data_sufficiency": {
            "min_periods_required": MIN_PERIODS_WALK_FORWARD,
            "max_periods_available": int(max_eff),
            "test": "t de Newey-West do alpha vs IBOV, só com n_eff >= mínimo",
            "is_significant": n_sig > 0,
            "warning": None if n_sig else (
                f"Nenhum horizonte com alpha significativo (máx. {max_eff:.0f} janelas "
                f"independentes; mínimo {MIN_PERIODS_WALK_FORWARD}). Números abaixo são ruído."
            ),
        },
        "interpretation": {
            "sharpe_excess_cdi": "Sharpe anualizado do excesso sobre o CDI, janelas não sobrepostas",
            "hit_rate": "% de janelas com alpha > IBOV",
            "n_eff": "janelas independentes = n / sobreposição",
        },
    })
    if exclude_from:
        result["excluded_windows_ending_from"] = exclude_from
    _finish(result, output_path, verbose)
    return result


def _summarize(periods: list[dict], h: int) -> dict:
    if not periods:
        return {"n_periods": 0}
    step = max(1, math.ceil(h / 5))
    alphas = np.array([p["alpha"] for p in periods if p["alpha"] is not None])
    excess = np.array([p["excess_cdi"] for p in periods if p["excess_cdi"] is not None])
    rets = np.array([p["port_ret"] for p in periods])
    n_eff = len(periods) / step

    t_nw = (newey_west_t(alphas, step - 1)
            if len(alphas) and n_eff >= MIN_PERIODS_WALK_FORWARD else None)
    p_val = None
    if t_nw is not None:
        from scipy.stats import t as student_t
        p_val = float(2 * student_t.sf(abs(t_nw), df=max(n_eff - 1, 1)))

    chain = rets[::step]
    ex_chain = excess[::step] if len(excess) else np.array([])
    sharpe = None
    if len(ex_chain) >= 3 and ex_chain.std(ddof=1) > 0:
        sharpe = float(ex_chain.mean() / ex_chain.std(ddof=1) * np.sqrt(252 / h))
    cum = np.cumprod(1 + chain)
    dd = float((cum / np.maximum.accumulate(cum) - 1).min())

    return {
        "n_periods":          len(periods),
        "n_eff":              round(n_eff, 1),
        "mean_return":        round(float(rets.mean()), 4),
        "median_return":      round(float(np.median(rets)), 4),
        "mean_alpha_ibov":    round(float(alphas.mean()), 4) if len(alphas) else None,
        "t_nw_alpha":         round(t_nw, 3) if t_nw is not None else None,
        "p_value":            round(p_val, 4) if p_val is not None else None,
        "hit_rate":           round(float((alphas > 0).mean()), 3) if len(alphas) else None,
        "mean_excess_cdi":    round(float(excess.mean()), 4) if len(excess) else None,
        "sharpe_excess_cdi":  round(sharpe, 3) if sharpe is not None else None,
        "sharpe_annualized":  round(sharpe, 3) if sharpe is not None else None,
        "max_drawdown":       round(dd, 4),
        "cumulative_return":  round(float(cum[-1] - 1), 4),
        "n_nonoverlap":       len(chain),
        "significant":        bool(p_val is not None and p_val < 0.05),
    }


def _finish(result: dict, output_path: Optional[Path], verbose: bool) -> None:
    if output_path is not None:
        _save(result, output_path)
    if verbose:
        _print_table(result)


def _save(result: dict, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        tmp.replace(output_path)
        logger.info("Walk-forward salvo em %s", output_path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        logger.error("Falha ao salvar walk-forward: %s", exc)


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        import sys
        sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))


def _print_table(result: dict) -> None:
    print()
    print("=" * 84)
    _safe_print(f"  Walk-Forward — {result['analysis_date'][:10]} | "
                f"recomendações: {result['n_recommendations']}")
    print("=" * 84)
    ds = result.get("data_sufficiency", {})
    if ds and not ds.get("is_significant", True):
        _safe_print(f"  [!] {ds.get('warning', 'Amostra insuficiente.')}")
        print("=" * 84)
    per_window = result.get("per_window", {})
    if not per_window:
        _safe_print(f"  {result.get('note', 'Sem dados.')}")
        return
    _safe_print(f"  {'Janela':<7}{'N':>4}{'Neff':>6}{'Ret':>9}{'Alpha':>9}{'t NW':>7}"
                f"{'Hit':>7}{'Sharpe-CDI':>12}{'MaxDD':>8}")
    for w, m in per_window.items():
        if not m.get("n_periods"):
            _safe_print(f"  {w:<7}{0:>4}")
            continue
        t = m.get("t_nw_alpha")
        sh = m.get("sharpe_excess_cdi")
        _safe_print(
            f"  {w:<7}{m['n_periods']:>4}{m['n_eff']:>6.1f}{m['mean_return'] * 100:>8.2f}%"
            f"{(m.get('mean_alpha_ibov') or 0) * 100:>8.2f}%"
            f"{'—' if t is None else f'{t:.2f}':>7}"
            f"{(m.get('hit_rate') or 0) * 100:>6.0f}%"
            f"{'—' if sh is None else f'{sh:.2f}':>12}{m['max_drawdown'] * 100:>7.1f}%"
        )
    print("=" * 84)
    print()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    walk_forward_backtest()


if __name__ == "__main__":
    main()
