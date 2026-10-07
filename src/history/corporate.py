"""
Proventos e eventos societários da B3 (listedCompaniesProxy), por emissor.

- GetListedSupplementCompany: nome de pregão e eventos em ações
  (desdobramento, grupamento, bonificação) com data-com e ISIN.
- GetListedCashDividends: histórico completo de dividendos e JCP desde ~2007,
  por espécie (ON/PN/UNT...), valor bruto por ação na época (não ajustado).

Respostas brutas ficam em CORP_DIR/<raiz>.json para não refazer chamadas.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

from src.history.paths import CORP_DIR

logger = logging.getLogger(__name__)

BASE = "https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall"
SLEEP = 0.4
# Emissores cujo nome de pregão atual não acha o histórico de proventos.
NAME_ALIASES = {"JBSS": ["JBS"], "EMBR": ["EMBRAER"]}


def _call(session: requests.Session, fn: str, payload: dict):
    token = base64.b64encode(json.dumps(payload).encode()).decode()
    for attempt in range(3):
        try:
            r = session.get(f"{BASE}/{fn}/{token}", timeout=60)
            if r.status_code == 200:
                return r.json() if r.text.strip() else None
        except (requests.RequestException, ValueError) as exc:
            logger.debug("%s falhou (%s)", fn, exc)
        time.sleep(2 ** attempt)
    return None


def fetch_issuer(root: str, session: Optional[requests.Session] = None,
                 refresh: bool = False) -> dict:
    session = session or requests.Session()
    CORP_DIR.mkdir(parents=True, exist_ok=True)
    path = CORP_DIR / f"{root}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))

    sup = _call(session, "GetListedSupplementCompany",
                {"issuingCompany": root, "language": "pt-br"})
    time.sleep(SLEEP)
    info = sup[0] if isinstance(sup, list) and sup else {}
    trading_name = (info.get("tradingName") or "").strip()
    cash = []
    # A busca por nome de pregão falha com "/" (AMBEV S/A); sem pontuação funciona.
    names = []
    for n in (trading_name, re.sub(r"[^A-Z0-9 ]", "", trading_name),
              re.sub(r"[^A-Z0-9]", "", trading_name)):
        if n and n not in names:
            names.append(n)
    names += NAME_ALIASES.get(root, [])
    for name in names:
        page = 1
        while True:
            res = _call(session, "GetListedCashDividends",
                        {"language": "pt-br", "pageNumber": page,
                         "pageSize": 120, "tradingName": name})
            time.sleep(SLEEP)
            if not res or not res.get("results"):
                break
            cash.extend(res["results"])
            if page >= (res.get("page") or {}).get("totalPages", 1):
                break
            page += 1
        if cash:
            break
    out = {"root": root, "trading_name": trading_name,
           "stock_events": info.get("stockDividends") or [],
           "cash": cash, "fetched_at": datetime.now().isoformat(timespec="seconds")}
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def _num(s) -> Optional[float]:
    if s in (None, ""):
        return None
    try:
        return float(str(s).replace(".", "").replace(",", "."))
    except ValueError:
        return None


def _date(s) -> Optional[pd.Timestamp]:
    try:
        return pd.Timestamp(datetime.strptime(s, "%d/%m/%Y"))
    except (TypeError, ValueError):
        return None


# fator = ações novas por ação antiga
def share_multiplier(label: str, factor: float) -> Optional[float]:
    label = label.upper()
    if label.startswith("DESDOBRAMENTO") or label.startswith("BONIFICACAO"):
        return 1.0 + factor / 100.0
    if label.startswith("GRUPAMENTO"):
        return factor
    return None


def cash_events(issuer: dict) -> pd.DataFrame:
    """Uma linha por provento: espécie, data-com, valor bruto, tipo."""
    rows = []
    for c in issuer.get("cash", []):
        rows.append({
            "root": issuer["root"],
            "type": (c.get("typeStock") or "").strip().upper(),
            "last_date_prior": _date(c.get("lastDatePriorEx")),
            "value": _num(c.get("valueCash")),
            "quoted_per_shares": _num(c.get("quotedPerShares")) or 1.0,
            "label": (c.get("corporateAction") or "").strip(),
        })
    df = pd.DataFrame(rows, columns=["root", "type", "last_date_prior", "value",
                                     "quoted_per_shares", "label"])
    df = df.dropna(subset=["last_date_prior", "value"])
    df["value"] = df["value"] / df["quoted_per_shares"].clip(lower=1)
    return df[df["value"] > 0]


def stock_events(issuer: dict) -> pd.DataFrame:
    """Eventos em ações com multiplicador de quantidade, por ISIN."""
    rows = []
    for e in issuer.get("stock_events", []):
        mult = share_multiplier(e.get("label") or "", _num(e.get("factor")) or 0.0)
        if mult is None or mult <= 0 or abs(mult - 1) < 1e-9:
            continue
        rows.append({
            "root": issuer["root"],
            "isin": (e.get("isinCode") or "").strip(),
            "last_date_prior": _date(e.get("lastDatePrior")),
            "multiplier": mult,
            "label": (e.get("label") or "").strip(),
        })
    df = pd.DataFrame(rows, columns=["root", "isin", "last_date_prior", "multiplier", "label"])
    return df.dropna(subset=["last_date_prior"]).drop_duplicates()
