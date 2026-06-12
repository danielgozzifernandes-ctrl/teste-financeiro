"""
bloomberg_download.py — rodar NO LAB DO INSPER com Bloomberg Terminal logado.

    pip install xbbg pandas
    python tools/bloomberg_download.py

Extrai os datasets prioritários para `bloomberg_data/` (criada na raiz do
repo). Cada bloco é INDEPENDENTE: se um campo falhar (permissão de licença,
cota diária), os demais continuam — o script nunca aborta no meio.

⚠️ NÃO TESTADO em terminal real (escrito fora do lab). Os nomes de campos
seguem a documentação BBG padrão; se algum der erro, consulte FLDS <GO> no
Terminal para o nome exato e ajuste a constante correspondente.

Saída esperada (ver docs/BLOOMBERG_DOWNLOAD.md):
    bloomberg_data/ibx_composition.csv
    bloomberg_data/consensus_eps.csv
    bloomberg_data/analyst_targets.csv
    bloomberg_data/short_interest.csv
"""

from __future__ import annotations

import csv
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "bloomberg_data"
UNIVERSE_CSV = ROOT / "data" / "universe.csv"
ASOF = date.today().isoformat()

# Datas trimestrais para composição histórica do IBX (desde 2016)
HIST_START_YEAR = 2016


def _load_universe_tickers() -> list[str]:
    """Tickers do universe.csv no formato Bloomberg ('PETR4 BZ Equity')."""
    tickers: list[str] = []
    with open(UNIVERSE_CSV, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            t = (row.get("ticker") or "").strip()
            if t:
                tickers.append(f"{t} BZ Equity")
    return tickers


def _save(df, name: str) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / name
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"  OK -> {path} ({len(df)} linhas)")


def download_ibx_composition(blp) -> None:
    """PRIORIDADE #1: composição histórica do IBX (mata survivorship bias)."""
    print("[1/4] Composição histórica IBX...")
    import pandas as pd

    frames = []
    quarters = [
        f"{y}{m:02d}01"
        for y in range(HIST_START_YEAR, date.today().year + 1)
        for m in (1, 4, 7, 10)
        if date(y, m, 1) <= date.today()
    ]
    for d in quarters:
        try:
            df = blp.bds("IBX Index", "INDX_MWEIGHT_HIST",
                         END_DATE_OVERRIDE=d)
            if df is not None and not df.empty:
                df = df.reset_index()
                df["date"] = f"{d[:4]}-{d[4:6]}-{d[6:]}"
                frames.append(df)
                print(f"    {d}: {len(df)} membros")
        except Exception as exc:
            print(f"    {d}: FALHOU ({exc}) — seguindo")
    if frames:
        out = pd.concat(frames, ignore_index=True)
        out["asof_date"] = ASOF
        _save(out, "ibx_composition.csv")
    else:
        print("  NADA baixado — verificar permissão INDX_MWEIGHT_HIST (FLDS <GO>)")


def download_consensus_eps(blp, tickers: list[str]) -> None:
    """PRIORIDADE #2: consenso de EPS (SUE real para o PEAD)."""
    print("[2/4] Consenso de EPS...")
    try:
        df = blp.bdp(tickers, ["BEST_EPS", "IS_EPS", "BEST_EPS_NUMEST"])
        df = df.reset_index().rename(columns={"index": "ticker"})
        df["asof_date"] = ASOF
        _save(df, "consensus_eps.csv")
    except Exception as exc:
        print(f"  FALHOU ({exc}) — seguindo")


def download_analyst_targets(blp, tickers: list[str]) -> None:
    """PRIORIDADE #3: preço-alvo com dispersão + nº de analistas."""
    print("[3/4] Preços-alvo de analistas...")
    try:
        df = blp.bdp(tickers, [
            "BEST_TARGET_PRICE", "BEST_TARGET_HI", "BEST_TARGET_LO",
            "TOT_ANALYST_REC",
        ])
        df = df.reset_index().rename(columns={"index": "ticker"})
        df["asof_date"] = ASOF
        _save(df, "analyst_targets.csv")
    except Exception as exc:
        print(f"  FALHOU ({exc}) — seguindo")


def download_short_interest(blp, tickers: list[str]) -> None:
    """Bônus: short interest (sinal contrarian/squeeze)."""
    print("[4/4] Short interest...")
    try:
        df = blp.bdp(tickers, ["SI_TOT_EQY", "SHORT_INT_RATIO"])
        df = df.reset_index().rename(columns={"index": "ticker"})
        df["asof_date"] = ASOF
        _save(df, "short_interest.csv")
    except Exception as exc:
        print(f"  FALHOU ({exc}) — seguindo")


def main() -> int:
    try:
        from xbbg import blp
    except ImportError:
        print("xbbg não instalado. Rode:  pip install xbbg pandas")
        return 1

    tickers = _load_universe_tickers()
    print(f"Universo: {len(tickers)} tickers | saída: {OUT_DIR}\n")

    download_ibx_composition(blp)
    download_consensus_eps(blp, tickers)
    download_analyst_targets(blp, tickers)
    download_short_interest(blp, tickers)

    print(
        "\nPronto. Agora:\n"
        "  git add bloomberg_data/ && "
        f"git commit -m 'data: bloomberg download {ASOF}' && git push\n"
        "(sem git no lab: copie bloomberg_data/ para pendrive e commite de casa)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
