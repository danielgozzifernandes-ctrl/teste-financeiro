"""
Download dos dados abertos da CVM (companhias abertas): DFP, ITR, FCA e FRE.

Arquivos anuais em <B3_DATA_DIR>/cvm/raw. Anos fechados são baixados uma vez;
o ano corrente é rebaixado se o arquivo local tiver mais de `max_age_hours`
(a CVM atualiza os ZIPs conforme chegam entregas e reapresentações).
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional

import requests

from src.history import data_dir

logger = logging.getLogger(__name__)

BASE = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC"
FORMS = ("DFP", "ITR", "FCA", "FRE")
FIRST_YEAR = {"DFP": 2010, "ITR": 2011, "FCA": 2010, "FRE": 2010}


def raw_dir() -> Path:
    return data_dir() / "cvm" / "raw"


def zip_path(form: str, year: int) -> Path:
    return raw_dir() / f"{form.lower()}_cia_aberta_{year}.zip"


def url(form: str, year: int) -> str:
    return f"{BASE}/{form}/DADOS/{form.lower()}_cia_aberta_{year}.zip"


def _is_fresh(path: Path, year: int, max_age_hours: float) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    if year < date.today().year:
        return True
    age = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds()
    return age < max_age_hours * 3600


def download(form: str, year: int, max_age_hours: float = 24.0,
             session: Optional[requests.Session] = None, retries: int = 3) -> Optional[Path]:
    path = zip_path(form, year)
    if _is_fresh(path, year, max_age_hours):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    session = session or requests.Session()
    tmp = path.with_suffix(".part")
    for attempt in range(1, retries + 1):
        try:
            with session.get(url(form, year), stream=True, timeout=120) as r:
                if r.status_code == 404:
                    logger.info("%s %d: não publicado", form, year)
                    return None
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            tmp.replace(path)
            logger.info("%s %d: %.1f MB", form, year, path.stat().st_size / 1e6)
            return path
        except requests.RequestException as exc:
            logger.warning("%s %d: tentativa %d falhou (%s)", form, year, attempt, exc)
            time.sleep(2 * attempt)
    return path if path.exists() else None


CAD_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/CAD/DADOS/cad_cia_aberta.csv"


def download_registry(session: Optional[requests.Session] = None) -> Optional[Path]:
    """Cadastro geral (inclui canceladas, com data e motivo); sempre rebaixado."""
    path = raw_dir() / "cad_cia_aberta.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = (session or requests).get(CAD_URL, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    except requests.RequestException as exc:
        logger.warning("cadastro CVM: falhou (%s)", exc)
    return path if path.exists() else None


def download_all(forms: Iterable[str] = FORMS, last_year: Optional[int] = None) -> list[Path]:
    last_year = last_year or date.today().year
    out = []
    session = requests.Session()
    download_registry(session)
    for form in forms:
        for year in range(FIRST_YEAR[form], last_year + 1):
            p = download(form, year, session=session)
            if p:
                out.append(p)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    download_all()
