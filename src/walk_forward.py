"""
walk_forward.py — Rolling out-of-sample backtest

Para cada par (snapshot histórico T0, snapshot futuro T1), simula:
  1. Compra do top-N de T0 com pesos salvos
  2. Aguarda até T1
  3. Mede retorno realizado vs IBOV (entry-to-entry)

Output:
  - Métricas agregadas: retorno cumulativo, Sharpe, alpha vs IBOV, max DD
  - Per-period: retorno por janela, hit rate (% janelas com alpha > 0)
  - Decay: como o alpha varia com lookahead (1w vs 4w vs 12w)

Uso primário:
  Detectar overfit do modelo. Se Sharpe in-sample (não computamos) >> Sharpe
  out-of-sample (este módulo), o sistema está sobreparametrizado.

Limites com dados disponíveis:
  - Walk-forward verdadeiro requer histórico de recomendações
  - Sem histórico de fundamentais PIT, só podemos simular a partir do dia
    em que o sistema começou a rodar
  - Statistical significance precisa de ~30+ janelas (≈ 6+ meses semanais)

CLI:  python -m src.walk_forward
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.config import HISTORY_DIR, DATA_DIR

logger = logging.getLogger(__name__)

WALK_FORWARD_OUTPUT_PATH = DATA_DIR / "walk_forward.json"


def _load_recommendations(history_dir: Path, mode: str = "weekly") -> list[dict]:
    """Carrega recomendações em ordem cronológica."""
    files = sorted(history_dir.glob(f"recommendations_*_{mode}.json"))
    out: list[dict] = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                out.append(json.load(fh))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def _load_snapshots(history_dir: Path) -> dict[str, dict[str, float]]:
    """Carrega snapshots de preço por data."""
    files = sorted(history_dir.glob("snapshot_*.json"))
    out: dict[str, dict[str, float]] = {}
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
                d = data.get("date") or f.stem.replace("snapshot_", "")
                prices = data.get("prices") or {}
                out[d] = {k: float(v) for k, v in prices.items() if v is not None}
        except (json.JSONDecodeError, OSError):
            continue
    return out


def _find_forward_snapshot(
    snapshots: dict[str, dict[str, float]],
    start_date: str,
    target_days: int,
    tolerance_days: int = 5,
) -> Optional[tuple[str, dict[str, float]]]:
    """Acha o snapshot mais próximo de start_date + target_days."""
    start_ts = pd.Timestamp(start_date)
    target_ts = start_ts + pd.Timedelta(days=int(target_days * 1.45))

    best: Optional[tuple[str, dict[str, float]]] = None
    best_diff = float("inf")
    for d, prices in snapshots.items():
        d_ts = pd.Timestamp(d)
        if d_ts <= start_ts:
            continue
        diff = abs((d_ts - target_ts).days)
        if diff <= tolerance_days and diff < best_diff:
            best = (d, prices)
            best_diff = diff
    return best


def _portfolio_return(
    weights: dict[str, float],
    entry_prices: dict[str, float],
    exit_prices: dict[str, float],
) -> Optional[float]:
    """
    Retorno ponderado do portfólio entry-to-entry.

    Tickers ausentes em exit_prices têm peso redistribuído proporcionalmente
    aos demais (fica com retorno de quem permaneceu).
    """
    valid = {t: w for t, w in weights.items() if t in entry_prices and t in exit_prices and entry_prices[t] > 0}
    if not valid:
        return None
    total_w = sum(valid.values())
    if total_w == 0:
        return None
    # Renormalizar
    valid_w = {t: w / total_w for t, w in valid.items()}
    ret = sum(valid_w[t] * ((exit_prices[t] / entry_prices[t]) - 1.0) for t in valid_w)
    return float(ret)


def _ibov_return(
    snapshots: dict[str, dict[str, float]],
    start_date: str,
    end_date: str,
    ibov_proxy: str = "^BVSP",
) -> Optional[float]:
    """
    Tenta retornar o IBOV no período. Se snapshots não tiverem IBOV (não tem),
    busca yfinance. Caching: idempotente; chamadas repetidas são OK.
    """
    try:
        import yfinance as yf
        df = yf.download(
            ibov_proxy,
            start=start_date, end=end_date,
            progress=False, auto_adjust=True,
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        s = df[col].dropna()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        if len(s) < 2:
            return None
        return float((s.iloc[-1] / s.iloc[0]) - 1.0)
    except Exception:
        return None


def walk_forward_backtest(
    history_dir: Path = HISTORY_DIR,
    mode: str = "weekly",
    forward_windows: dict[str, int] = None,
    output_path: Path = WALK_FORWARD_OUTPUT_PATH,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Roda walk-forward sobre histórico acumulado.

    Para cada (rec_T0, janela_forward), procura snapshot futuro e mede
    retorno do top-5 vs IBOV no período.

    Returns dict com:
      - n_recommendations / n_snapshots
      - per_window: {1w: {n_periods, mean_return, mean_alpha_ibov, hit_rate,
                          sharpe_annualized, max_drawdown}, 4w: ..., 12w: ...}
      - per_period: detalhes de cada janela computada
      - decay: alpha médio por horizonte (proxy de half-life)
    """
    if forward_windows is None:
        forward_windows = {"1w": 5, "4w": 20, "12w": 60}

    history_dir = Path(history_dir)
    recs = _load_recommendations(history_dir, mode)
    snaps = _load_snapshots(history_dir)

    if len(recs) < 1 or len(snaps) < 2:
        result = {
            "analysis_date": datetime.now().isoformat(),
            "n_recommendations": len(recs),
            "n_snapshots": len(snaps),
            "per_window": {},
            "note": "Histórico insuficiente — precisa >= 1 recomendação e >= 2 snapshots",
        }
        _save(result, output_path)
        if verbose:
            _print_table(result)
        return result

    per_window_results: dict[str, dict] = {}
    per_period_detail: list[dict] = []

    for window_name, window_days in forward_windows.items():
        periods: list[dict] = []

        for rec in recs:
            rec_date = rec.get("date")
            if not rec_date:
                continue
            entry_prices = rec.get("entry_prices") or {}
            weights = rec.get("portfolio_weights") or {}
            if not entry_prices or not weights:
                continue

            forward = _find_forward_snapshot(snaps, rec_date, window_days)
            if forward is None:
                continue
            end_date, end_prices = forward

            port_ret = _portfolio_return(weights, entry_prices, end_prices)
            if port_ret is None:
                continue

            ibov_ret = _ibov_return(snaps, rec_date, end_date)
            alpha = (port_ret - ibov_ret) if ibov_ret is not None else None

            periods.append({
                "rec_date":   rec_date,
                "exit_date":  end_date,
                "port_ret":   round(port_ret, 4),
                "ibov_ret":   round(ibov_ret, 4) if ibov_ret is not None else None,
                "alpha":      round(alpha, 4) if alpha is not None else None,
                "window":     window_name,
            })

        if not periods:
            per_window_results[window_name] = {"n_periods": 0}
            continue

        # Agregar métricas
        returns = np.array([p["port_ret"] for p in periods], dtype=float)
        alphas = np.array(
            [p["alpha"] for p in periods if p["alpha"] is not None],
            dtype=float,
        )

        mean_ret  = float(np.mean(returns))
        std_ret   = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        # Sharpe ANUALIZADO: assume rebalance cada `window_days` úteis
        periods_per_year = 252.0 / window_days
        sharpe = float(mean_ret / std_ret * np.sqrt(periods_per_year)) if std_ret > 0 else None
        # Max drawdown sobre série cumulativa
        cum = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum)
        dd = (cum - running_max) / running_max
        max_dd = float(dd.min())

        per_window_results[window_name] = {
            "n_periods":            len(periods),
            "mean_return":          round(mean_ret, 4),
            "median_return":        round(float(np.median(returns)), 4),
            "std_return":           round(std_ret, 4),
            "sharpe_annualized":    round(sharpe, 3) if sharpe is not None else None,
            "max_drawdown":         round(max_dd, 4),
            "mean_alpha_ibov":      round(float(np.mean(alphas)), 4) if len(alphas) > 0 else None,
            "hit_rate":             round(float((alphas > 0).sum() / len(alphas)), 3) if len(alphas) > 0 else None,
            "cumulative_return":    round(float(cum[-1] - 1), 4),
        }
        per_period_detail.extend(periods)

    # Decay analysis: alpha médio por horizonte
    decay = {}
    for w, m in per_window_results.items():
        if m.get("mean_alpha_ibov") is not None:
            decay[w] = m["mean_alpha_ibov"]

    result = {
        "analysis_date":        datetime.now().isoformat(),
        "n_recommendations":    len(recs),
        "n_snapshots":          len(snaps),
        "per_window":           per_window_results,
        "per_period":           per_period_detail,
        "decay_alpha":          decay,
        "interpretation": {
            "sharpe": "Sharpe anualizado > 1.0 é bom; > 1.5 é excelente; < 0.5 questionável",
            "hit_rate": "% de janelas com alpha > IBOV. > 0.55 é desejável",
            "max_drawdown": "Pior perda cumulativa de pico a vale na série de retornos",
        },
    }
    _save(result, output_path)
    if verbose:
        _print_table(result)
    return result


def _save(result: dict, output_path: Path) -> None:
    """Salva o resultado em JSON."""
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
    """Print resiliente a cp1252 (Windows)."""
    try:
        print(text)
    except UnicodeEncodeError:
        import sys
        sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))


def _print_table(result: dict) -> None:
    """Imprime tabela formatada no stdout."""
    print()
    print("=" * 78)
    _safe_print(f"  Walk-Forward Backtest — {result['analysis_date'][:10]}")
    _safe_print(
        f"  Snapshots: {result['n_snapshots']} | Recomendacoes: {result['n_recommendations']}"
    )
    print("=" * 78)

    per_window = result.get("per_window", {})
    if not per_window:
        print("  ", result.get("note", "Sem dados."))
        print("=" * 78)
        return

    header = f"  {'Window':<8} {'N':>4} {'MeanRet':>10} {'Sharpe':>8} {'MaxDD':>8} {'Alpha vs IBOV':>16} {'Hit Rate':>10} {'CumRet':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for w, m in per_window.items():
        if not m.get("n_periods"):
            row = f"  {w:<8} {'0':>4} {'-':>10} {'-':>8} {'-':>8} {'-':>16} {'-':>10} {'-':>10}"
        else:
            row = (
                f"  {w:<8} {m['n_periods']:>4d} "
                f"{m['mean_return']*100:>+9.2f}% "
                f"{m.get('sharpe_annualized') or 0:>8.2f} "
                f"{m['max_drawdown']*100:>+7.1f}% "
                f"{(m.get('mean_alpha_ibov') or 0)*100:>+15.2f}% "
                f"{(m.get('hit_rate') or 0)*100:>9.1f}% "
                f"{m['cumulative_return']*100:>+9.2f}%"
            )
        _safe_print(row)

    print("=" * 78)
    decay = result.get("decay_alpha", {})
    if decay:
        _safe_print(
            "  Alpha por horizonte: " + " | ".join(
                f"{k}={v*100:+.2f}%" for k, v in decay.items()
            )
        )
        print("=" * 78)
    print()


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    _setup_logging()
    walk_forward_backtest()


if __name__ == "__main__":
    main()
