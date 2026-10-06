"""
Arquiva a carteira teórica do dia e a prévia de IBOV, IBrX-100 e IBrA (B3).

A B3 só publica a carteira vigente e a próxima; não há histórico gratuito.
Rodando todo dia, isso vira composição point-in-time daqui pra frente.

Grava um arquivo só quando membros ou quantidades teóricas mudam (o peso do
dia sai de quantidade × preço), em data/history/index_portfolios/.
Falha de rede não derruba o job: loga e sai com 0.
"""

from __future__ import annotations

import base64
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "history" / "index_portfolios"
BASE_URL = "https://sistemaswebb3-listados.b3.com.br/indexProxy/indexCall"
INDICES = ["IBOV", "IBXX", "IBRA"]
CALLS = {"day": "GetPortfolioDay", "preview": "GetQuartelyPreview"}
BRT = timezone(timedelta(hours=-3))

logger = logging.getLogger("archive_index_portfolios")


def _num(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    try:
        return float(str(s).replace(".", "").replace(",", "."))
    except ValueError:
        return None


def fetch(index: str, kind: str, session: requests.Session) -> dict:
    payload = {"language": "pt-br", "pageNumber": 1, "pageSize": 500,
               "index": index, "segment": "1"}
    token = base64.b64encode(json.dumps(payload).encode()).decode()
    r = session.get(f"{BASE_URL}/{CALLS[kind]}/{token}", timeout=30)
    r.raise_for_status()
    return r.json()


def normalize(raw: dict, index: str, kind: str, today: str) -> dict:
    header = raw.get("header") or {}
    members = sorted(
        (
            {
                "ticker": (m.get("cod") or "").strip(),
                "name": (m.get("asset") or "").strip(),
                "type": " ".join((m.get("type") or "").split()),
                "theoretical_qty": _num(m.get("theoricalQty")),
                "weight_pct": _num(m.get("part")),
            }
            for m in raw.get("results") or []
        ),
        key=lambda m: m["ticker"],
    )
    return {
        "index": index,
        "kind": kind,
        "fetched_on": today,
        "b3_date": header.get("date"),
        "reductor": _num(header.get("reductor")),
        "theoretical_qty_total": _num(header.get("theoricalQty")),
        "members": members,
    }


def _signature(snap: dict) -> list:
    return [(m["ticker"], m["theoretical_qty"]) for m in snap["members"]]


def _last_saved(index: str, kind: str) -> Optional[dict]:
    files = sorted(OUT_DIR.glob(f"{index}_{kind}_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def archive(session: Optional[requests.Session] = None,
            today: Optional[str] = None) -> list[Path]:
    session = session or requests.Session()
    today = today or datetime.now(BRT).date().isoformat()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for index in INDICES:
        for kind in CALLS:
            try:
                snap = normalize(fetch(index, kind, session), index, kind, today)
            except Exception as exc:
                logger.warning("%s %s: falhou (%s)", index, kind, exc)
                continue
            if not snap["members"]:
                logger.warning("%s %s: sem membros", index, kind)
                continue
            last = _last_saved(index, kind)
            if last and _signature(last) == _signature(snap):
                continue
            path = OUT_DIR / f"{index}_{kind}_{today}.json"
            path.write_text(json.dumps(snap, ensure_ascii=False, indent=1),
                            encoding="utf-8")
            written.append(path)
            logger.info("%s %s: %d membros -> %s", index, kind,
                        len(snap["members"]), path.name)
    return written


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    archive()
    sys.exit(0)
