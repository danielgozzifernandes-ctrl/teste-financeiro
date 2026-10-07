"""
Cadastro CVM: ticker ↔ companhia, setor e quantidade de ações.

- Tickers: FCA `valor_mobiliario` de todos os anos (inclui códigos antigos e
  de companhias que saíram da bolsa). Junta com COTAHIST por ticker.
- Setor: FCA `geral` (Setor_Atividade), último valor por ano.
- Ações:
  * 2020+: composição do capital do ITR/DFP (trimestral). Parte das
    companhias declara em milhares sem indicar; a escala é corrigida pelo
    total do FRE do mesmo ano (razão ~1000) e, sem FRE, pelo valor
    patrimonial por ação (> R$ 1.000/ação ⇒ milhares).
  * antes de 2020: FRE, "Capital Integralizado". O arquivo traz só a última
    versão do ano; fica disponível na DT_RECEB dessa versão (conservador:
    pode atrasar um desdobramento, mas não antecipa).
"""

from __future__ import annotations

import base64
import json
import logging
import zipfile
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import requests

from src.history.cvm_download import FIRST_YEAR, raw_dir, zip_path
from src.history.cvm_fundamentals import _norm, _read, processed_dir

logger = logging.getLogger(__name__)


def _years(form: str, years: Optional[Iterable[int]]) -> list[int]:
    if years is not None:
        return [y for y in years if y >= FIRST_YEAR[form]]
    return list(range(FIRST_YEAR[form], pd.Timestamp.today().year + 1))


def _kind(valor_mobiliario: str, ticker: str) -> str:
    v = _norm(valor_mobiliario)
    if "unit" in v or "certificado" in v:
        return "UNIT"
    if "preferenciais" in v:
        return "PN"
    if "ordinarias" in v:
        return "ON"
    t = str(ticker)
    return {"3": "ON", "4": "PN", "5": "PN", "6": "PN", "11": "UNIT"}.get(t[4:], "OTHER")


def ticker_map(years: Optional[Iterable[int]] = None) -> pd.DataFrame:
    frames = []
    for y in _years("FCA", years):
        p = zip_path("FCA", y)
        if not p.exists():
            continue
        z = zipfile.ZipFile(p)
        vm = _read(z, f"fca_cia_aberta_valor_mobiliario_{y}.csv")
        idx = _read(z, f"fca_cia_aberta_{y}.csv", usecols=["CNPJ_CIA", "CD_CVM"]).drop_duplicates()
        if vm.empty:
            continue
        vm = vm[vm["Codigo_Negociacao"].notna() & (vm["Mercado"].map(_norm) == "bolsa")]
        vm = vm.merge(idx, left_on="CNPJ_Companhia", right_on="CNPJ_CIA", how="left")
        vm["year"] = y
        frames.append(vm)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["ticker"] = df["Codigo_Negociacao"].str.strip().str.upper()
    df["kind"] = [_kind(v, t) for v, t in zip(df["Valor_Mobiliario"], df["ticker"])]
    for c in ("Data_Inicio_Negociacao", "Data_Fim_Negociacao", "Data_Fim_Listagem"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    out = df.groupby(["CD_CVM", "ticker"]).agg(
        cnpj=("CNPJ_Companhia", "last"), name=("Nome_Empresarial", "last"), kind=("kind", "last"),
        first_year=("year", "min"), last_year=("year", "max"),
        trading_start=("Data_Inicio_Negociacao", "min"), trading_end=("Data_Fim_Negociacao", "max"),
        listing_end=("Data_Fim_Listagem", "max"),
    ).reset_index().rename(columns={"CD_CVM": "cd_cvm"})
    return out


def sectors(years: Optional[Iterable[int]] = None) -> pd.DataFrame:
    frames = []
    for y in _years("FCA", years):
        p = zip_path("FCA", y)
        if not p.exists():
            continue
        g = _read(zipfile.ZipFile(p), f"fca_cia_aberta_geral_{y}.csv",
                  usecols=["Codigo_CVM", "Setor_Atividade", "Situacao_Registro_CVM", "Especie_Controle_Acionario"])
        if not g.empty:
            g["year"] = y
            frames.append(g)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).rename(columns={
        "Codigo_CVM": "cd_cvm", "Setor_Atividade": "sector",
        "Situacao_Registro_CVM": "registration_status", "Especie_Controle_Acionario": "control"})
    return df.sort_values("year").drop_duplicates(["cd_cvm", "year"], keep="last")


def fre_shares(years: Optional[Iterable[int]] = None) -> pd.DataFrame:
    frames = []
    for y in _years("FRE", years):
        p = zip_path("FRE", y)
        if not p.exists():
            continue
        z = zipfile.ZipFile(p)
        cs = _read(z, f"fre_cia_aberta_capital_social_{y}.csv")
        idx = _read(z, f"fre_cia_aberta_{y}.csv", usecols=["CNPJ_CIA", "CD_CVM", "VERSAO", "DT_RECEB"])
        if cs.empty or idx.empty:
            continue
        cs = cs[cs["Tipo_Capital"].map(_norm) == "capital integralizado"]
        cs = cs.merge(idx, left_on=["CNPJ_Companhia", "Versao"], right_on=["CNPJ_CIA", "VERSAO"], how="left")
        frames.append(cs)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({
        "cd_cvm": df["CD_CVM"], "cnpj": df["CNPJ_Companhia"],
        "dt_refer": df["Data_Referencia"], "approved_on": df["Data_Autorizacao_Aprovacao"],
        "available_from": df["DT_RECEB"],
        "shares_on": pd.to_numeric(df["Quantidade_Acoes_Ordinarias"], errors="coerce"),
        "shares_pn": pd.to_numeric(df["Quantidade_Acoes_Preferenciais"], errors="coerce"),
        "shares_total": pd.to_numeric(df["Quantidade_Total_Acoes"], errors="coerce"),
        "source": "FRE",
    })
    out = out.dropna(subset=["cd_cvm", "available_from", "shares_total"])
    # Mais de uma linha integralizada por documento (aumentos no ano): fica a última aprovação.
    out = out.sort_values(["cd_cvm", "dt_refer", "approved_on"]).drop_duplicates(
        ["cd_cvm", "dt_refer"], keep="last")
    return out.reset_index(drop=True)


def fix_share_scale(docs: pd.DataFrame, fre: pd.DataFrame) -> pd.DataFrame:
    """Corrige composição de capital declarada em milhares (ver docstring do módulo)."""
    docs = docs.copy()
    if "shares_total" not in docs.columns:
        docs["shares_scale"] = np.nan
        return docs
    fre_y = fre.assign(year=pd.to_datetime(fre["dt_refer"]).dt.year)[["cd_cvm", "year", "shares_total"]]
    fre_y = fre_y.rename(columns={"shares_total": "fre_total"}).drop_duplicates(["cd_cvm", "year"], keep="last")
    docs["year"] = pd.to_datetime(docs["dt_refer"]).dt.year
    docs = docs.merge(fre_y, on=["cd_cvm", "year"], how="left")
    ratio = docs["fre_total"] / docs["shares_total"]
    by_fre = ratio.between(300, 3000)
    bvps = docs["equity_controlling"] / docs["shares_total"]
    by_bvps = docs["fre_total"].isna() & (bvps > 1000)
    scale = np.where(by_fre | by_bvps, 1000.0, 1.0)
    docs["shares_scale"] = np.where(docs["shares_total"].notna(), scale, np.nan)
    docs["shares_scale_source"] = np.where(by_fre, "fre", np.where(by_bvps, "bvps", None))
    for c in ("shares_on", "shares_pn", "shares_total", "treasury_on", "treasury_pn", "treasury_total"):
        if c in docs.columns:
            docs[c] = docs[c] * docs["shares_scale"].fillna(1.0)
    return docs.drop(columns=["year", "fre_total"])


def shares_as_of(as_of, docs: pd.DataFrame, fre: pd.DataFrame) -> pd.DataFrame:
    """Ações em circulação (total − tesouraria quando houver) disponíveis em `as_of`."""
    as_of = pd.Timestamp(as_of)
    comp = docs.dropna(subset=["shares_total"])
    comp = comp[pd.to_datetime(comp["available_from"]) <= as_of]
    comp = comp.sort_values(["cd_cvm", "dt_refer", "available_from"]).groupby("cd_cvm").tail(1)
    comp = comp.assign(source="ITR/DFP")[["cd_cvm", "dt_refer", "available_from", "shares_on", "shares_pn",
                                         "shares_total", "treasury_on", "treasury_pn", "treasury_total", "source"]]
    f = fre[pd.to_datetime(fre["available_from"]) <= as_of]
    f = f.sort_values(["cd_cvm", "dt_refer", "available_from"]).groupby("cd_cvm").tail(1)
    f = f[~f["cd_cvm"].isin(comp["cd_cvm"])]
    out = pd.concat([comp, f.drop(columns=["cnpj", "approved_on"])], ignore_index=True)
    for k in ("on", "pn", "total"):
        out[f"outstanding_{k}"] = out[f"shares_{k}"] - out.get(f"treasury_{k}", 0).fillna(0)
    return out


B3_COMPANIES = ("https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/"
                "CompanyCall/GetInitialCompanies/")


def b3_issuers(session: Optional[requests.Session] = None) -> pd.DataFrame:
    """
    Emissores registrados na B3: código CVM ↔ código de emissor (prefixo de
    4 letras do ticker, o que liga ao COTAHIST). Só traz quem ainda tem
    registro; companhias canceladas não aparecem.
    """
    s = session or requests.Session()
    rows, page = [], 1
    while True:
        token = base64.b64encode(json.dumps(
            {"language": "pt-br", "pageNumber": page, "pageSize": 120}).encode()).decode()
        d = s.get(B3_COMPANIES + token, timeout=30).json()
        rows += d.get("results") or []
        if page >= (d.get("page") or {}).get("totalPages", 0):
            break
        page += 1
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return pd.DataFrame({
        "cd_cvm": df["codeCVM"].str.zfill(6), "issuer_code": df["issuingCompany"].str.strip(),
        "cnpj": df["cnpj"], "trading_name": df["tradingName"], "segment": df["segment"],
    }).drop_duplicates()


def cvm_registry() -> pd.DataFrame:
    """Cadastro CVM com situação e data/motivo de cancelamento."""
    path = raw_dir() / "cad_cia_aberta.csv"
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_csv(path, sep=";", encoding="latin1", dtype=str)
    out = pd.DataFrame({
        "cd_cvm": d["CD_CVM"].str.zfill(6), "cnpj": d["CNPJ_CIA"], "name": d["DENOM_SOCIAL"],
        "trade_name": d["DENOM_COMERC"], "status": d["SIT"], "registered_on": d["DT_REG"],
        "canceled_on": d["DT_CANCEL"], "cancel_reason": d["MOTIVO_CANCEL"], "sector": d["SETOR_ATIV"],
        "control": d["CONTROLE_ACIONARIO"],
    })
    return out.sort_values("registered_on").drop_duplicates("cd_cvm", keep="last")


def build_reference(save: bool = True) -> dict[str, pd.DataFrame]:
    ref = {"tickers": ticker_map(), "sectors": sectors(), "fre_shares": fre_shares(),
           "registry": cvm_registry()}
    try:
        ref["b3_issuers"] = b3_issuers()
    except Exception as exc:  # API não documentada da B3
        logger.warning("cadastro B3 indisponível: %s", exc)
    if save:
        out = processed_dir()
        out.mkdir(parents=True, exist_ok=True)
        for k, v in ref.items():
            v.to_parquet(out / f"{k}.parquet", index=False)
    return ref
