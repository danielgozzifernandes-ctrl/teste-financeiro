"""
Download de dados Bloomberg para rodar no lab, com o Terminal logado na máquina.

    pip install xbbg pandas
    python tools/bloomberg_download.py --pilot          # teste rápido: 5 tickers
    python tools/bloomberg_download.py                  # tudo
    python tools/bloomberg_download.py --steps comp,earn

Etapas (na ordem de prioridade, por causa da cota da licença acadêmica):
  comp   composição mensal de IBOV/IBX/IBrA (membros + pesos)
  earn   histórico de divulgações de resultado com EPS reportado e esperado
  cons   consenso semanal (EPS próximo exercício, dispersão, preço-alvo, recomendações)
  px     preços diários, retorno total e ações em circulação de TODOS os membros
         históricos (inclui deslistadas — é o que tira o viés de sobrevivência)
  si     short interest semanal

Cada etapa grava o próprio CSV e segue mesmo se outra falhar. Nomes de campo
seguem a documentação padrão; nada foi testado num Terminal real — se um campo
der erro, conferir com FLDS <GO>.

Dado bruto da Bloomberg não pode ser redistribuído: a pasta de saída está no
.gitignore e não deve ir para o repositório público.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNIVERSE_CSV = ROOT / "data" / "universe.csv"
ASOF = date.today().isoformat()

INDICES = ["IBOV Index", "IBX Index", "IBRA Index"]
HIST_START = "2010-01-01"
# Carteiras do IBOV/IBX rebalanceiam na 1ª segunda de jan/mai/set; amostrar
# todo dia 15 pega cada carteira nova e as mudanças extraordinárias no meio.
COMP_DAY = 15

CONSENSUS_FIELDS = [
    "BEST_EPS", "BEST_EPS_NUMEST", "BEST_EPS_STDDEV",
    "BEST_TARGET_PRICE", "BEST_ANALYST_RATING",
    "TOT_BUY_REC", "TOT_HOLD_REC", "TOT_SELL_REC",
]
PRICE_FIELDS = ["PX_LAST", "TOT_RETURN_INDEX_GROSS_DVDS", "EQY_SH_OUT", "TURNOVER"]
SI_FIELDS = ["SI_TOT_EQY", "SHORT_INT_RATIO"]

STEPS = ["comp", "earn", "cons", "px", "si"]


def _universe_tickers() -> list[str]:
    with open(UNIVERSE_CSV, encoding="utf-8", errors="replace") as f:
        return [f"{r['ticker'].strip()} BZ Equity" for r in csv.DictReader(f)
                if (r.get("ticker") or "").strip()]


def _month_dates(start: str) -> list[str]:
    y, m = int(start[:4]), int(start[5:7])
    today = date.today()
    out = []
    while date(y, m, COMP_DAY) <= today:
        out.append(f"{y}{m:02d}{COMP_DAY:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _save(df, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df["asof_date"] = ASOF
    df.to_csv(out_dir / name, index=False, encoding="utf-8")
    print(f"  ok -> {name} ({len(df)} linhas)")


def _member_tickers(out_dir: Path) -> list[str]:
    """Todos os membros históricos já baixados na etapa comp, no formato BBG."""
    import pandas as pd

    path = out_dir / "index_composition.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    col = next((c for c in df.columns if c.lower().replace(" ", "_") in
                ("index_member", "member_ticker_and_exchange_code")), None)
    if col is None:
        return []
    return sorted({f"{str(t).strip()} Equity" for t in df[col].dropna()})


def step_comp(blp, out_dir: Path, indices: list[str], dates: list[str]) -> None:
    import pandas as pd

    print(f"[comp] {len(indices)} índice(s) x {len(dates)} datas")
    frames = []
    for idx in indices:
        for d in dates:
            try:
                df = blp.bds(idx, "INDX_MWEIGHT_HIST", END_DATE_OVERRIDE=d)
            except Exception as exc:
                print(f"    {idx} {d}: falhou ({exc})")
                continue
            if df is None or df.empty:
                print(f"    {idx} {d}: vazio")
                continue
            df = df.reset_index(drop=True)
            df["index"] = idx
            df["date"] = f"{d[:4]}-{d[4:6]}-{d[6:]}"
            frames.append(df)
        print(f"    {idx}: ok")
    if frames:
        _save(pd.concat(frames, ignore_index=True), out_dir, "index_composition.csv")
    else:
        print("  nada baixado — conferir permissão de INDX_MWEIGHT_HIST")


def step_earn(blp, out_dir: Path, tickers: list[str]) -> None:
    import pandas as pd

    print(f"[earn] {len(tickers)} tickers")
    frames = []
    for t in tickers:
        try:
            df = blp.bds(t, "EARN_ANN_DT_TIME_HIST_WITH_EPS")
        except Exception as exc:
            print(f"    {t}: falhou ({exc})")
            continue
        if df is not None and not df.empty:
            df = df.reset_index(drop=True)
            df["ticker"] = t
            frames.append(df)
    if frames:
        _save(pd.concat(frames, ignore_index=True), out_dir, "earnings_history.csv")


def step_bdh(blp, out_dir: Path, tickers: list[str], fields: list[str],
             name: str, per: str, **overrides) -> None:
    import pandas as pd

    print(f"[{name}] {len(tickers)} tickers, {len(fields)} campos, Per={per}")
    frames = []
    for i in range(0, len(tickers), 20):
        chunk = tickers[i:i + 20]
        try:
            df = blp.bdh(chunk, fields, start_date=HIST_START, end_date=ASOF,
                         Per=per, **overrides)
        except Exception as exc:
            print(f"    lote {i // 20}: falhou ({exc})")
            continue
        if df is None or df.empty:
            continue
        long = df.stack(level=0).reset_index()
        long.columns = ["date", "ticker", *long.columns[2:]]
        frames.append(long)
    if frames:
        _save(pd.concat(frames, ignore_index=True), out_dir, f"{name}.csv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default=",".join(STEPS),
                    help=f"subconjunto de {STEPS}, separado por vírgula")
    ap.add_argument("--pilot", action="store_true",
                    help="5 tickers, só IBOV, 2 datas — para checar campos e cota")
    ap.add_argument("--out", default=str(ROOT / "bloomberg_data"))
    args = ap.parse_args()

    try:
        from xbbg import blp
    except ImportError:
        print("xbbg não instalado: pip install xbbg pandas")
        return 1

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    out_dir = Path(args.out)
    tickers = _universe_tickers()
    indices, dates = INDICES, _month_dates(HIST_START)
    if args.pilot:
        tickers, indices, dates = tickers[:5], INDICES[:1], dates[-2:]
        out_dir = out_dir / "pilot"
    print(f"saída: {out_dir} | etapas: {steps}\n")

    if "comp" in steps:
        step_comp(blp, out_dir, indices, dates)

    # Ex-membros entram nas etapas seguintes; sem eles o histórico continua
    # só com sobreviventes.
    members = _member_tickers(out_dir)
    full = sorted(set(tickers) | set(members)) if not args.pilot else tickers
    if members:
        print(f"\n{len(full)} tickers (universo atual + {len(members)} membros históricos)\n")

    if "earn" in steps:
        step_earn(blp, out_dir, full)
    if "cons" in steps:
        step_bdh(blp, out_dir, full, CONSENSUS_FIELDS, "consensus_weekly", "W",
                 BEST_FPERIOD_OVERRIDE="1BF")
    if "px" in steps:
        step_bdh(blp, out_dir, full, PRICE_FIELDS, "prices_daily", "D")
    if "si" in steps:
        step_bdh(blp, out_dir, full, SI_FIELDS, "short_interest_weekly", "W")

    print(f"\nPronto. Leve a pasta {out_dir} num pendrive/OneDrive; não commitar no repositório.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
