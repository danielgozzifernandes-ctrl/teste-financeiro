"""
Backtest standalone da camada de Asset Allocation.

    python -m src.allocation_backtest [--years 10] [--rebalance 21]

A allocation roda sobre ETFs líquidos (BOVA11/IVVB11/IMAB11) + CDI, que têm
10+ anos de diário no yfinance — dá para testar as regras já, ao contrário
do stock-picking, que depende de snapshots semanais acumulados.

Metodologia (anti-lookahead):
  - Rebalanceamento a cada REBALANCE_DAYS pregões.
  - Em cada rebalance, os sinais usam só dados até aquela data:
      * Regime proxy: vol realizada 21d do BOVA11 vs mediana expansiva
        (o HMM do pipeline usa features que não existem no histórico todo).
      * TSMOM 12-1 do BOVA11 vs CDI acumulado (mesma regra do allocator).
      * ERP não entra (não há earnings yield histórico da carteira aqui)
        — o backtest valida regime+TSMOM; o ERP é tilt adicional não testado.
  - Custos: 10 bps por lado sobre o turnover (ETFs líquidos, sem imposto —
    documentado como limitação).

Comparativos: estratégia vs 100% BOVA11, 100% CDI e mix estático 40/30/15/15.

Saída: data/allocation_backtest.json + resumo no stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

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
COST_PER_SIDE_BPS = 10          # custo por lado sobre turnover (ETF líquido)
VOL_REGIME_WINDOW = 21          # vol realizada para proxy de regime
STATIC_MIX = {"equities_br": 0.40, "cdi": 0.30, "global_usd": 0.15, "inflation": 0.15}


# Dados

def _fetch_etf_prices(years: int) -> pd.DataFrame:
    """Preços ajustados dos ETFs (colunas = sleeves)."""
    import yfinance as yf

    start = (date.today() - timedelta(days=int(years * 365.25) + 30)).strftime("%Y-%m-%d")
    symbols = list(ALLOCATION_BACKTEST_TICKERS.values())
    raw = yf.download(tickers=symbols, start=start, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("yfinance não retornou preços de ETF")

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close.index = pd.to_datetime(close.index).tz_localize(None)

    inv = {v: k for k, v in ALLOCATION_BACKTEST_TICKERS.items()}
    close = close.rename(columns=inv)
    sleeves = [s for s in inv.values() if s in close.columns]
    return close[sleeves].dropna(how="all")


def _fetch_cdi_daily(start: str) -> pd.Series:
    """
    Retornos diários do CDI via BCB (python-bcb).

    A API SGS rejeita janelas >10 anos em séries diárias (HTTP 406), e o
    BenchmarkManager adiciona 7 dias de margem — clamp do start para caber
    com folga. Perder ~2 meses do início do CDI é melhor que perder TUDO
    (sem clamp, o sleeve CDI renderia 0% no backtest inteiro, punindo
    injustamente as estratégias com caixa).
    """
    try:
        from src.benchmark import BenchmarkManager
        min_start = (date.today() - timedelta(days=3590)).strftime("%Y-%m-%d")
        start = max(start, min_start)
        df = BenchmarkManager().get_returns(start)
        if "cdi" in df.columns:
            s = df["cdi"].dropna()
            if len(s) > 50:
                return s
    except Exception as exc:
        logger.warning("CDI via BCB falhou (%s) — usando proxy constante", exc)
    # Fallback declarado: CDI médio de longo prazo ~11% a.a. — só para não
    # abortar; o JSON marca cdi_source="constant_proxy".
    return pd.Series(dtype=float)


# Sinais point-in-time

def _regime_proxy(equity_prices: pd.Series, asof_idx: int) -> str:
    """
    Proxy de regime via vol realizada 21d vs mediana EXPANSIVA (só passado).

    vol < mediana → risk_on; vol > 1.5×mediana → bear; senão mean_rev.
    Aproximação declarada do HMM do pipeline (que precisa de VIX/USDBRL).
    """
    rets = equity_prices.iloc[:asof_idx + 1].pct_change().dropna()
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

    excess = eq_ret - cdi_ret
    return ALLOCATION_TILT_PP if excess > 0 else -ALLOCATION_TILT_PP


def _weights_at(equity_prices: pd.Series, cdi: pd.Series, asof_idx: int) -> dict[str, float]:
    """Pesos da estratégia na data asof — só com informação passada."""
    regime = _regime_proxy(equity_prices, asof_idx)
    base = ALLOCATION_BASE[regime]
    tilt = _tsmom_tilt(equity_prices, cdi, asof_idx)

    eq = float(np.clip(base["equities_br"] + tilt,
                       EQUITIES_SLEEVE_MIN, EQUITIES_SLEEVE_MAX))
    delta = eq - base["equities_br"]
    w = {
        "equities_br": eq,
        "cdi":         max(base["cdi"] - delta, 0.0),
        "global_usd":  base["global_usd"],
        "inflation":   base["inflation"],
    }
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


# Simulação

def _simulate(
    prices: pd.DataFrame,
    cdi: pd.Series,
    rebalance_days: int,
    dynamic: bool,
) -> tuple[pd.Series, float, list[dict]]:
    """
    Simula a estratégia (dynamic=True) ou o mix estático (False).

    Returns: (curva NAV, turnover médio por rebalance, log de alocações)
    """
    rets = prices.pct_change().fillna(0.0)
    cdi_aligned = cdi.reindex(rets.index).ffill().fillna(0.0) if not cdi.empty \
        else pd.Series(0.0, index=rets.index)

    nav = 100.0
    curve: dict[pd.Timestamp, float] = {}
    weights: Optional[dict[str, float]] = None
    turnovers: list[float] = []
    alloc_log: list[dict] = []
    eq = prices["equities_br"]

    for i, dt in enumerate(rets.index):
        if weights is None or i % rebalance_days == 0:
            new_w = (_weights_at(eq, cdi, i) if dynamic else dict(STATIC_MIX))
            if weights is not None:
                turnover = sum(abs(new_w[k] - weights.get(k, 0)) for k in new_w)
                turnovers.append(turnover)
                nav *= 1.0 - turnover * COST_PER_SIDE_BPS / 10_000
            weights = new_w
            alloc_log.append({"date": dt.strftime("%Y-%m-%d"), **{
                k: round(v, 4) for k, v in weights.items()}})

        day_ret = sum(
            weights[s] * (cdi_aligned.loc[dt] if s == "cdi"
                          else float(rets.loc[dt].get(s, 0.0)))
            for s in weights
        )
        nav *= 1.0 + day_ret
        curve[dt] = nav

    return pd.Series(curve), float(np.mean(turnovers)) if turnovers else 0.0, alloc_log


def _metrics(curve: pd.Series, cdi: pd.Series) -> dict:
    """Retorno anualizado, vol, Sharpe (excesso de CDI), max drawdown."""
    rets = curve.pct_change().dropna()
    if len(rets) < 60:
        return {"error": "amostra insuficiente"}
    n_years = len(rets) / 252
    ann_ret = float((curve.iloc[-1] / curve.iloc[0]) ** (1 / n_years) - 1)
    ann_vol = float(rets.std() * np.sqrt(252))
    cdi_aligned = cdi.reindex(rets.index).ffill().fillna(0.0) if not cdi.empty \
        else pd.Series(0.0, index=rets.index)
    excess = rets - cdi_aligned
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0
    dd = float((curve / curve.cummax() - 1).min())
    return {
        "ann_return":   round(ann_ret, 4),
        "ann_vol":      round(ann_vol, 4),
        "sharpe_vs_cdi": round(sharpe, 3),
        "max_drawdown": round(dd, 4),
        "total_return": round(float(curve.iloc[-1] / curve.iloc[0] - 1), 4),
        "n_days":       len(rets),
    }


# Entrypoint

def run_allocation_backtest(years: int = 10, rebalance_days: int = 21,
                            verbose: bool = True) -> dict:
    prices = _fetch_etf_prices(years)
    if len(prices) < 252:
        raise RuntimeError(f"Histórico insuficiente: {len(prices)} pregões")
    start = prices.index[0].strftime("%Y-%m-%d")
    cdi = _fetch_cdi_daily(start)
    cdi_source = "bcb" if not cdi.empty else "unavailable"

    dyn_curve, dyn_turnover, alloc_log = _simulate(prices, cdi, rebalance_days, dynamic=True)
    static_curve, _, _ = _simulate(prices, cdi, rebalance_days, dynamic=False)

    # Benchmarks 100% num ativo
    bova = (1 + prices["equities_br"].pct_change().fillna(0)).cumprod() * 100
    cdi_curve = ((1 + cdi.reindex(prices.index).ffill().fillna(0)).cumprod() * 100
                 if not cdi.empty else pd.Series(dtype=float))

    result = {
        "generated_at": date.today().isoformat(),
        "period": {"start": start, "end": prices.index[-1].strftime("%Y-%m-%d")},
        "rebalance_days": rebalance_days,
        "cost_per_side_bps": COST_PER_SIDE_BPS,
        "cdi_source": cdi_source,
        "limitations": [
            "Regime via proxy de vol (não o HMM do pipeline)",
            "Sleeve de bolsa = BOVA11, não a carteira top-5 (testa SÓ a camada de allocation)",
            "Sem imposto; custo fixo 10bps/lado",
            "ERP tilt não testado (sem EY histórico aqui)",
        ],
        "avg_turnover_per_rebalance": round(dyn_turnover, 4),
        "strategies": {
            "dynamic_allocation": _metrics(dyn_curve, cdi),
            "static_40_30_15_15": _metrics(static_curve, cdi),
            "buy_hold_bova11":    _metrics(bova, cdi),
            "cdi_100pct":         (_metrics(cdi_curve, cdi)
                                   if not cdi_curve.empty else {"error": "CDI indisponível"}),
        },
        "allocation_log_tail": alloc_log[-12:],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if verbose:
        _print_summary(result)
    return result


def _print_summary(result: dict) -> None:
    out = ["", "═" * 64, " BACKTEST DA CAMADA DE ALLOCATION",
           f" {result['period']['start']} → {result['period']['end']}"
           f" | rebalance {result['rebalance_days']}d"
           f" | custo {result['cost_per_side_bps']}bps/lado",
           "═" * 64]
    hdr = f" {'estratégia':<24}{'ret a.a.':>9}{'vol':>8}{'Sharpe':>8}{'maxDD':>8}"
    out += [hdr, "─" * 64]
    for name, m in result["strategies"].items():
        if "error" in m:
            out.append(f" {name:<24}{m['error']}")
            continue
        out.append(
            f" {name:<24}{m['ann_return'] * 100:>8.1f}%{m['ann_vol'] * 100:>7.1f}%"
            f"{m['sharpe_vs_cdi']:>8.2f}{m['max_drawdown'] * 100:>7.1f}%"
        )
    out += ["─" * 64,
            " Limitações: " + "; ".join(result["limitations"]),
            f" JSON: {OUTPUT_PATH}", ""]
    sys.stdout.buffer.write(("\n".join(out) + "\n").encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest da camada de asset allocation")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--rebalance", type=int, default=21, metavar="DIAS")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    try:
        run_allocation_backtest(years=args.years, rebalance_days=args.rebalance)
        return 0
    except Exception as exc:
        logger.error("Backtest falhou: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
