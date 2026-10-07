"""
COTAHIST da B3: download e parser do arquivo posicional de cotações.

Layout (registro tipo 01, 245 colunas, latin-1; posições 1-based, inclusivas):

    1-2     TIPREG   "01"
    3-10    DATA     AAAAMMDD
    11-12   CODBDI   02 = lote padrão, 96 = fracionário, 12 = FII ...
    13-24   CODNEG   ticker
    25-27   TPMERC   010 = à vista
    28-39   NOMRES   nome resumido
    40-49   ESPECI   ON / PN / UNT / ... + segmento (NM, N1, N2, ED, EJ ...)
    57-69   PREABE   (11)V99
    70-82   PREMAX
    83-95   PREMIN
    96-108  PREMED
    109-121 PREULT
    148-152 TOTNEG   nº de negócios
    153-170 QUATOT   quantidade
    171-188 VOLTOT   (16)V99, R$
    211-217 FATCOT   fator de cotação (1 = por ação, 1000 = por lote de mil)
    231-242 CODISI   ISIN

Preços divididos por 100 e por FATCOT ficam em R$ por ação.
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

from src.history.paths import PARQUET_DIR, RAW_DIR

logger = logging.getLogger(__name__)

BASE_URL = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist"
SPOT_MARKET = "010"
LOT_BDI = {"02"}

# (nome, início 1-based, fim 1-based inclusivo)
FIELDS = [
    ("date", 3, 10),
    ("bdi", 11, 12),
    ("ticker", 13, 24),
    ("market", 25, 27),
    ("name", 28, 39),
    ("spec", 40, 49),
    ("open", 57, 69),
    ("high", 70, 82),
    ("low", 83, 95),
    ("avg", 96, 108),
    ("close", 109, 121),
    ("trades", 148, 152),
    ("quantity", 153, 170),
    ("volume", 171, 188),
    ("fatcot", 211, 217),
    ("isin", 231, 242),
]
PRICE_COLS = ["open", "high", "low", "avg", "close"]


# ── download ────────────────────────────────────────────────────────────────

def annual_url(year: int) -> str:
    return f"{BASE_URL}/COTAHIST_A{year}.ZIP"


def daily_url(d: date) -> str:
    return f"{BASE_URL}/COTAHIST_D{d:%d%m%Y}.ZIP"


def _download(url: str, dest: Path, session: requests.Session,
              force: bool = False) -> Optional[Path]:
    """Baixa se ausente ou se o tamanho remoto mudou; valida o ZIP."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    head = session.head(url, timeout=60, allow_redirects=True)
    if head.status_code == 404:
        return None
    head.raise_for_status()
    remote_size = int(head.headers.get("content-length") or 0)
    if dest.exists() and not force and remote_size and dest.stat().st_size == remote_size:
        return dest

    tmp = dest.with_suffix(".part")
    with session.get(url, timeout=600, stream=True) as r:
        if r.status_code == 404:
            return None
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if remote_size and tmp.stat().st_size != remote_size:
        tmp.unlink(missing_ok=True)
        raise IOError(f"{url}: tamanho {tmp.stat().st_size} != {remote_size}")
    with zipfile.ZipFile(tmp) as z:
        bad = z.testzip()
        if bad:
            raise IOError(f"{url}: ZIP corrompido ({bad})")
    tmp.replace(dest)
    logger.info("baixado %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def download_annual(years: Iterable[int], force_current: bool = True,
                    session: Optional[requests.Session] = None) -> list[Path]:
    session = session or requests.Session()
    out = []
    this_year = date.today().year
    for y in years:
        p = _download(annual_url(y), RAW_DIR / f"COTAHIST_A{y}.ZIP", session,
                      force=False)
        if p is None:
            logger.warning("ano %s indisponível", y)
            continue
        out.append(p)
        time.sleep(1.0)
    return out


def download_daily(start: date, end: date,
                   session: Optional[requests.Session] = None) -> list[Path]:
    """Arquivos diários (para dias após o fim do anual corrente)."""
    session = session or requests.Session()
    out = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            p = _download(daily_url(d), RAW_DIR / "daily" / f"COTAHIST_D{d:%d%m%Y}.ZIP",
                          session)
            if p:
                out.append(p)
            time.sleep(0.5)
        d += timedelta(days=1)
    return out


# ── parser ──────────────────────────────────────────────────────────────────

def parse_lines(lines: Iterable[str], bdi: set[str] = LOT_BDI) -> pd.DataFrame:
    rows = []
    for line in lines:
        if not line.startswith("01"):
            continue
        if line[24:27] != SPOT_MARKET or line[10:12] not in bdi:
            continue
        rows.append({name: line[a - 1:b] for name, a, b in FIELDS})
    if not rows:
        return pd.DataFrame(columns=[f[0] for f in FIELDS])
    df = pd.DataFrame(rows)
    for c in ("ticker", "name", "spec", "isin", "bdi", "market"):
        df[c] = df[c].str.strip()
    df["spec"] = df["spec"].str.split().str.join(" ")
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    fatcot = df["fatcot"].astype("int64").clip(lower=1)
    for c in PRICE_COLS:
        df[c] = df[c].astype("int64") / 100.0 / fatcot
    df["volume"] = df["volume"].astype("int64") / 100.0
    df["trades"] = df["trades"].astype("int64")
    df["quantity"] = df["quantity"].astype("int64")
    df["fatcot"] = fatcot
    return df.drop(columns=["market"])


def parse_zip(path: Path, bdi: set[str] = LOT_BDI) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            text = io.TextIOWrapper(f, encoding="latin-1")
            return parse_lines(text, bdi)


def build_parquet(paths: Iterable[Path]) -> list[Path]:
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in paths:
        df = parse_zip(p)
        dest = PARQUET_DIR / (p.stem + ".parquet")
        df.to_parquet(dest, index=False)
        logger.info("%s: %d linhas, %s → %s", p.name, len(df),
                    df["date"].min().date() if len(df) else "-",
                    df["date"].max().date() if len(df) else "-")
        out.append(dest)
    return out


def load_all() -> pd.DataFrame:
    """Todas as linhas de lote padrão, sem duplicata (anual + diários)."""
    files = sorted(PARQUET_DIR.glob("COTAHIST_*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return (df.drop_duplicates(["date", "ticker"], keep="last")
              .sort_values(["ticker", "date"]).reset_index(drop=True))
