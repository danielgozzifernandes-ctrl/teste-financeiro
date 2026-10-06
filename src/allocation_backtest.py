"""
Backtest standalone da camada de Asset Allocation.

    python -m src.allocation_backtest [--years 10] [--rebalance 21] [--end YYYY-MM-DD]

Roda sobre ETFs (BOVA11/IVVB11/IMAB11) + CDI. Regras:
  - Sinais com dados até o fechamento de t; pesos novos valem a partir de t+1.
  - Entre rebalanceamentos os pesos derivam com os preços.
  - Sleeve sem preço ainda (IMAB11 só negocia desde 2019-05-17) fica em CDI.
  - Custo de 10 bps por lado sobre o giro de cada rebalanceamento.
  - Regime por proxy de vol realizada (o HMM do pipeline depende de séries
    que não existem no histórico todo). ERP não entra (sem EY histórico).

Duas janelas: "primary" começa quando todos os ETFs têm preço; "full" usa
`--years` com o sleeve sem dado em CDI. Sharpe anualizado sobre o excesso ao
CDI; PSR e Deflated Sharpe (Bailey & López de Prado) para alguns N de
estratégias testadas, porque o N real não é conhecido.

Saída: data/allocation_backtest.json (ou --output) + resumo no stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.config import (
    ALLOCATION_BASE,
    ALLOCATION_BACKTEST_TICKERS,
    ALLOCATION_TILT_PP,
    DATA_DIR,
    EQUITIES_SLEEVE_MAX,
    EQUITIES_SLEEVE_MIN,
    TSMOM_SKIP_DAYS,
    TSMOM_WINDOW_DAYS,
)

logger = logging.getLogger(__name__)

OUTPUT_PATH = DATA_DIR / "allocation_backtest.json"
COST_PER_SIDE_BPS = 10
VOL_REGIME_WINDOW = 21
STATIC_MIX = {"equities_br": 0.40, "cdi": 0.30, "global_usd": 0.15, "inflation": 0.15}
SLEEVES = ("equities_br", "cdi", "global_usd", "inflation")
DSR_TRIALS = (4, 10, 20)
SENSITIVITY_OFFSETS = range(0, 21)
_SGS_CHUNK_YEARS = 9   # SGS recusa janelas diárias > 10 anos


# Dados

def _fetch_etf_prices(start: str, end: Optional[str] = None) -> pd.DataFrame:
    """Fechamentos ajustados dos ETFs (colunas = sleeves), sem candle incompleto."""
    import yfinance as yf

    symbols = list(ALLOCATION_BACKTEST_TICKERS.values())
    kw = {"end": str(pd.Timestamp(end) + pd.Timedelta(days=1))[:10]} if end else {}
    raw = yf.download(tickers=symbols, start=start, auto_adjust=True, progress=False, **kw)
    if raw is None or raw.empty:
        raise RuntimeError("yfinance não retornou preços de ETF")
    raw.index = pd.to_datetime(raw.index).tz_localize(None)

    close = raw["Close"].copy()
    # yfinance publica o pregão corrente com OHL/volume zerados até consolidar.
    for c in ("Open", "Volume"):
        if c in raw.columns.get_level_values(0):
            close = close.where(raw[c].reindex(columns=close.columns).fillna(0) > 0)

    inv = {v: k for k, v in ALLOCATION_BACKTEST_TICKERS.items()}
    close = close.rename(columns=inv)
    sleeves = [s for s in inv.values() if s in close.columns]
    return close[sleeves].dropna(how="all")


def _fetch_cdi_daily(start: str, end: str) -> pd.Series:
    """CDI diário (decimal) do SGS série 12, em blocos para caber no limite da API."""
    from bcb import sgs

    from src.benchmark import BenchmarkManager

    parts = []
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    while s <= e:
        chunk_end = min(s + pd.DateOffset(years=_SGS_CHUNK_YEARS), e)
        df = sgs.get({"cdi": 12}, start=s.strftime("%Y-%m-%d"), end=chunk_end.strftime("%Y-%m-%d"))
        if df is not None and not df.empty:
            parts.append(df["cdi"])
        s = chunk_end + pd.Timedelta(days=1)
    if not parts:
        return pd.Series(dtype=float)
    raw = pd.concat(parts).sort_index()
    raw = raw[~raw.index.duplicated()]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    return BenchmarkManager._convert_to_daily_return(raw, col_name="cdi").dropna()


def _cdi_on_index(cdi: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """CDI composto entre datas consecutivas do índice (não perde dia do SGS fora dele)."""
    if cdi.empty:
        return pd.Series(0.0, index=index)
    level = (1 + cdi).cumprod()
    level = level.reindex(level.index.union(index)).ffill().reindex(index)
    return level.pct_change().fillna(0.0)


# Sinais (só passado)

def _regime_proxy(equity_prices: pd.Series, asof_idx: int) -> str:
    """Vol realizada 21d vs mediana expansiva: < mediana risk_on, > 1,5× bear."""
    rets = equity_prices.iloc[:asof_idx + 1].dropna().pct_change(fill_method=None).dropna()
    if len(rets) < 100:
        return "mean_rev"
    vol_now = float(rets.tail(VOL_REGIME_WINDOW).std() * np.sqrt(252))
    vol_hist = rets.rolling(VOL_REGIME_WINDOW).std().dropna() * np.sqrt(252)
    med = float(vol_hist.median())
    if med <= 0:
        return "mean_rev"
    if vol_now > 1.5 * med:
        return "bear"
    if vol_now < med:
        return "risk_on"
    return "mean_rev"


def _tsmom_tilt(equity_prices: pd.Series, cdi: pd.Series, asof_idx: int) -> float:
    """TSMOM 12-1 vs CDI até asof (mesma regra do allocator). 0 se sem dados."""
    hist = equity_prices.iloc[:asof_idx + 1].dropna()
    if len(hist) < TSMOM_SKIP_DAYS + 60:
        return 0.0
    window = min(TSMOM_WINDOW_DAYS, len(hist) - 1)
    if window <= TSMOM_SKIP_DAYS:
        return 0.0
    eq_ret = float(hist.iloc[-1 - TSMOM_SKIP_DAYS] / hist.iloc[-window] - 1.0)

    cdi_ret = 0.0
    if not cdi.empty:
        cdi_window = cdi[cdi.index <= hist.index[-1]].tail(window - TSMOM_SKIP_DAYS)
        if len(cdi_window) > 0:
            cdi_ret = float((1 + cdi_window).prod() - 1)

    return ALLOCATION_TILT_PP if eq_ret - cdi_ret > 0 else -ALLOCATION_TILT_PP


def _weights_at(equity_prices: pd.Series, cdi: pd.Series, asof_idx: int) -> dict[str, float]:
    regime = _regime_proxy(equity_prices, asof_idx)
    base = ALLOCATION_BASE[regime]
    tilt = _tsmom_tilt(equity_prices, cdi, asof_idx)

    eq = float(np.clip(base["equities_br"] + tilt, EQUITIES_SLEEVE_MIN, EQUITIES_SLEEVE_MAX))
    delta = eq - base["equities_br"]
    w = {
        "equities_br": eq,
        "cdi":         max(base["cdi"] - delta, 0.0),
        "global_usd":  base["global_usd"],
        "inflation":   base["inflation"],
    }
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def _available(prices: pd.DataFrame, i: int, target: dict[str, float]) -> dict[str, float]:
    """Sleeve sem preço até a data i vai para o CDI."""
    out = dict(target)
    for s in ("equities_br", "global_usd", "inflation"):
        if s in out and (s not in prices.columns or prices[s].iloc[:i + 1].dropna().empty):
            out["cdi"] = out.get("cdi", 0.0) + out.pop(s)
    return out


# Simulação

def _simulate(
    prices: pd.DataFrame,
    cdi: pd.Series,
    rebalance_days: int,
    target_fn: Callable[[int], dict[str, float]],
) -> tuple[pd.Series, float, list[dict]]:
    """
    Pesos-alvo decididos no fechamento de i (target_fn(i)) valem a partir de i+1.
    Entre rebalanceamentos as posições derivam; custo sobre o giro.

    Returns: (NAV, giro médio one-way por rebalance, log de alocações)
    """
    rets = prices.pct_change(fill_method=None)
    cdi_aligned = _cdi_on_index(cdi, prices.index)

    values: dict[str, float] = {}
    nav = 100.0
    curve: dict[pd.Timestamp, float] = {}
    turnovers: list[float] = []
    alloc_log: list[dict] = []

    for i, dt in enumerate(prices.index):
        if values:
            for s in values:
                r = cdi_aligned.iloc[i] if s == "cdi" else rets[s].iloc[i]
                values[s] *= 1.0 + (0.0 if pd.isna(r) else float(r))
            nav = sum(values.values())
        curve[dt] = nav

        if i % rebalance_days == 0 and i < len(prices.index) - 1:
            target = _available(prices, i, target_fn(i))
            current = {s: v / nav for s, v in values.items()} if values else {}
            turnover = sum(abs(target.get(s, 0.0) - current.get(s, 0.0))
                           for s in set(target) | set(current))
            if values:
                turnovers.append(turnover / 2)
            nav *= 1.0 - turnover * COST_PER_SIDE_BPS / 10_000
            values = {s: w * nav for s, w in target.items()}
            alloc_log.append({"date": dt.strftime("%Y-%m-%d"),
                              **{k: round(v, 4) for k, v in target.items()}})

    return pd.Series(curve), float(np.mean(turnovers)) if turnovers else 0.0, alloc_log


# Métricas

def _psr(sr: float, n: int, skew: float, kurt: float, sr0: float = 0.0) -> Optional[float]:
    """Probabilistic Sharpe Ratio (SR por período, não anualizado)."""
    from scipy.stats import norm

    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if n < 2 or denom <= 0:
        return None
    return float(norm.cdf((sr - sr0) * math.sqrt(n - 1) / math.sqrt(denom)))


def _expected_max_sr(n_trials: int, sr_var: float) -> float:
    """SR máximo esperado entre N tentativas sem habilidade (Bailey & López de Prado)."""
    from scipy.stats import norm

    gamma = 0.5772156649
    return math.sqrt(sr_var) * (
        (1 - gamma) * norm.ppf(1 - 1.0 / n_trials)
        + gamma * norm.ppf(1 - 1.0 / (n_trials * math.e))
    )


def _metrics(curve: pd.Series, cdi: pd.Series) -> dict:
    rets = curve.pct_change().dropna()
    if len(rets) < 60:
        return {"error": "amostra insuficiente"}
    n_years = len(rets) / 252
    ann_ret = float((curve.iloc[-1] / curve.iloc[0]) ** (1 / n_years) - 1)
    ann_vol = float(rets.std() * np.sqrt(252))
    cdi_aligned = _cdi_on_index(cdi, curve.index).reindex(rets.index)
    excess = rets - cdi_aligned
    out = {
        "ann_return":    round(ann_ret, 4),
        "ann_vol":       round(ann_vol, 4),
        "sharpe_vs_cdi": None,
        "max_drawdown":  round(float((curve / curve.cummax() - 1).min()), 4),
        "total_return":  round(float(curve.iloc[-1] / curve.iloc[0] - 1), 4),
        "n_days":        len(rets),
    }
    # Excesso ~0 (a própria estratégia CDI) dá Sharpe de ruído de ponto flutuante.
    if excess.std() < 1e-8:
        return out
    sr_d = float(excess.mean() / excess.std())
    out["sharpe_vs_cdi"] = round(sr_d * math.sqrt(252), 3)
    skew, kurt = float(excess.skew()), float(excess.kurt()) + 3.0
    psr = _psr(sr_d, len(excess), skew, kurt)
    out["psr_vs_0"] = round(psr, 3) if psr is not None else None
    sr_var = 1.0 / (len(excess) - 1)
    out["dsr"] = {}
    for n in DSR_TRIALS:
        d = _psr(sr_d, len(excess), skew, kurt, sr0=_expected_max_sr(n, sr_var))
        out["dsr"][f"N={n}"] = round(d, 3) if d is not None else None
    return out


def _run_window(prices: pd.DataFrame, cdi: pd.Series, rebalance_days: int) -> dict:
    eq = prices["equities_br"]
    dyn_curve, dyn_turnover, alloc_log = _simulate(
        prices, cdi, rebalance_days, lambda i: _weights_at(eq, cdi, i))
    static_curve, static_turnover, _ = _simulate(
        prices, cdi, rebalance_days, lambda i: dict(STATIC_MIX))
    bova = (1 + eq.pct_change(fill_method=None).fillna(0)).cumprod() * 100
    cdi_curve = (1 + _cdi_on_index(cdi, prices.index)).cumprod() * 100

    return {
        "period": {"start": prices.index[0].strftime("%Y-%m-%d"),
                   "end": prices.index[-1].strftime("%Y-%m-%d")},
        "avg_turnover_per_rebalance": {"dynamic": round(dyn_turnover, 4),
                                       "static": round(static_turnover, 4)},
        "strategies": {
            "dynamic_allocation": _metrics(dyn_curve, cdi),
            "static_40_30_15_15": _metrics(static_curve, cdi),
            "buy_hold_bova11":    _metrics(bova, cdi),
            "cdi_100pct":         _metrics(cdi_curve, cdi) if not cdi.empty
                                  else {"error": "CDI indisponível"},
        },
        "allocation_log_tail": alloc_log[-12:],
    }


def _start_sensitivity(prices: pd.DataFrame, cdi: pd.Series, rebalance_days: int) -> dict:
    """Faixa do Sharpe deslocando o início em 0–20 pregões (dependência de caminho)."""
    out: dict[str, list[float]] = {"dynamic_allocation": [], "static_40_30_15_15": []}
    for k in SENSITIVITY_OFFSETS:
        p = prices.iloc[k:]
        eq = p["equities_br"]
        for name, fn in (("dynamic_allocation", lambda i: _weights_at(eq, cdi, i)),
                         ("static_40_30_15_15", lambda i: dict(STATIC_MIX))):
            curve, _, _ = _simulate(p, cdi, rebalance_days, fn)
            sr = _metrics(curve, cdi).get("sharpe_vs_cdi")
            if sr is not None:
                out[name].append(sr)
    return {
        name: {"min": round(min(v), 3), "median": round(float(np.median(v)), 3),
               "max": round(max(v), 3), "n_offsets": len(v)}
        for name, v in out.items() if v
    }


# Entrypoint

def run_allocation_backtest(
    years: int = 10,
    rebalance_days: int = 21,
    end: Optional[str] = None,
    output_path: Path = OUTPUT_PATH,
    verbose: bool = True,
    prices: Optional[pd.DataFrame] = None,
    cdi: Optional[pd.Series] = None,
) -> dict:
    if prices is None:
        start = (date.today() - timedelta(days=int(years * 365.25) + 30)).strftime("%Y-%m-%d")
        prices = _fetch_etf_prices(start, end)
    if end:
        prices = prices.loc[:end]
    if len(prices) < 252:
        raise RuntimeError(f"Histórico insuficiente: {len(prices)} pregões")
    if cdi is None:
        cdi = _fetch_cdi_daily(prices.index[0].strftime("%Y-%m-%d"),
                               prices.index[-1].strftime("%Y-%m-%d"))
    if cdi.empty:
        raise RuntimeError("CDI indisponível — sem ele o Sharpe vs CDI não existe")

    first_all = prices.dropna().index.min()
    primary = _run_window(prices.loc[first_all:], cdi, rebalance_days)
    full = _run_window(prices, cdi, rebalance_days)

    result = {
        "generated_at": date.today().isoformat(),
        "rebalance_days": rebalance_days,
        "cost_per_side_bps": COST_PER_SIDE_BPS,
        "cdi_source": "bcb_sgs_12",
        "dsr_note": (
            "DSR para N estratégias testadas; o N real não é conhecido (várias "
            "versões da regra ao longo do histórico), por isso a faixa N=4/10/20. "
            "Variância do SR sob H0 aproximada por 1/(T-1)."
        ),
        "limitations": [
            "Regime via proxy de vol (não o HMM do pipeline)",
            "Sleeve de bolsa = BOVA11, não a carteira top-5 (testa só a camada de allocation)",
            "Sem imposto; custo fixo 10bps/lado",
            "ERP tilt não testado (sem EY histórico aqui)",
            f"Janela 'full': sleeves sem preço (IMAB11 antes de {first_all.date()}) ficam em CDI",
        ],
        # Compatibilidade: chaves de topo = janela principal (todos os ETFs com preço).
        **primary,
        "full": full,
        "start_sensitivity": {
            "primary": _start_sensitivity(prices.loc[first_all:], cdi, rebalance_days),
            "full": _start_sensitivity(prices, cdi, rebalance_days),
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if verbose:
        _print_summary(result, output_path)
    return result


def _print_summary(result: dict, output_path: Path) -> None:
    out = [""]
    for label, block in (("principal", result), ("completa", result["full"])):
        out += ["=" * 64,
                f" ALLOCATION ({label}) {block['period']['start']} -> {block['period']['end']}"
                f" | rebalance {result['rebalance_days']}d | {result['cost_per_side_bps']}bps/lado",
                "=" * 64,
                f" {'estratégia':<24}{'ret a.a.':>9}{'vol':>8}{'Sharpe':>8}{'maxDD':>8}"]
        for name, m in block["strategies"].items():
            if "error" in m:
                out.append(f" {name:<24}{m['error']}")
                continue
            sr = m["sharpe_vs_cdi"]
            out.append(
                f" {name:<24}{m['ann_return'] * 100:>8.1f}%{m['ann_vol'] * 100:>7.1f}%"
                f"{'—' if sr is None else f'{sr:.2f}':>8}{m['max_drawdown'] * 100:>7.1f}%"
            )
    out += ["-" * 64, " Limitações: " + "; ".join(result["limitations"]),
            f" JSON: {output_path}", ""]
    sys.stdout.buffer.write(("\n".join(out) + "\n").encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest da camada de asset allocation")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--rebalance", type=int, default=21, metavar="DIAS")
    parser.add_argument("--end", default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    try:
        run_allocation_backtest(years=args.years, rebalance_days=args.rebalance,
                                end=args.end, output_path=Path(args.output))
        return 0
    except Exception as exc:
        logger.error("Backtest falhou: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
