"""
Fundamentos point-in-time a partir de DFP/ITR da CVM.

Pontos que definem o desenho:
- O índice de cada ZIP (`*_cia_aberta_YYYY.csv`) lista todas as versões de
  cada documento com DT_RECEB; os arquivos de demonstração trazem só a
  versão mais recente. Então: disponível a partir da DT_RECEB da 1ª versão,
  com os valores da última. Reapresentações vazam valor revisto para trás —
  `n_versions` > 1 marca esses casos.
- DRE e DFC do ITR vêm acumulados no ano (e a DRE também no trimestre). O
  acumulado é a linha ÚLTIMO de DT_INI mais antigo; o comparativo do ano
  anterior é a PENÚLTIMO equivalente. TTM = acumulado + anual anterior −
  comparativo acumulado anterior. No DFP, TTM = anual.
- Consolidado quando o documento tem; senão individual.
- Contas escolhidas por descrição das contas fixas de 1º/2º nível, que é
  estável entre os modelos (industrial, banco, seguradora). Capex e D&A vêm
  de linhas livres da DFC: melhor esforço, por padrão de texto.
"""

from __future__ import annotations

import logging
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.history import data_dir
from src.history.cvm_download import FIRST_YEAR, zip_path

logger = logging.getLogger(__name__)

FLOW_FIELDS = ["revenue", "gross_profit", "ebit", "pretax_income", "net_income",
               "net_income_controlling", "operating_cash_flow", "capex", "dna"]
STOCK_FIELDS = ["total_assets", "equity", "equity_controlling", "cash",
                "short_term_investments", "gross_debt"]
SHARE_FIELDS = ["shares_on", "shares_pn", "shares_total",
                "treasury_on", "treasury_pn", "treasury_total"]

_SCALE = {"MIL": 1_000.0, "UNIDADE": 1.0}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip().lower()


def processed_dir() -> Path:
    return data_dir() / "cvm" / "processed"


# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------

def _read(z: zipfile.ZipFile, name: str, usecols: Optional[list[str]] = None) -> pd.DataFrame:
    if name not in z.namelist():
        return pd.DataFrame()
    df = pd.read_csv(z.open(name), sep=";", encoding="latin1", dtype=str, usecols=usecols)
    df.columns = [c.strip() for c in df.columns]
    return df


_STMT_COLS = ["CD_CVM", "DT_REFER", "VERSAO", "ESCALA_MOEDA", "ORDEM_EXERC",
              "DT_INI_EXERC", "DT_FIM_EXERC", "CD_CONTA", "DS_CONTA", "VL_CONTA"]


def _statement(z: zipfile.ZipFile, form: str, year: int, sheet: str, scope: str,
               prefixes: Optional[tuple[str, ...]] = None, max_depth: Optional[int] = None) -> pd.DataFrame:
    name = f"{form.lower()}_cia_aberta_{sheet}_{scope}_{year}.csv"
    cols = _STMT_COLS if sheet not in ("BPA", "BPP") else [c for c in _STMT_COLS if c != "DT_INI_EXERC"]
    df = _read(z, name, usecols=cols)
    if df.empty:
        return df
    if "DT_INI_EXERC" not in df.columns:
        df["DT_INI_EXERC"] = None
    df["ordem"] = df["ORDEM_EXERC"].map(lambda s: "prev" if _norm(s).startswith("pen") else "cur")
    if prefixes:
        df = df[df["CD_CONTA"].str.startswith(prefixes)]
    if max_depth is not None:
        df = df[df["CD_CONTA"].str.count(r"\.") <= max_depth]
    df["value"] = pd.to_numeric(df["VL_CONTA"], errors="coerce") * df["ESCALA_MOEDA"].map(_SCALE).fillna(1.0)
    df["ds"] = df["DS_CONTA"].map(_norm)
    df["scope"] = scope
    return df.drop(columns=["VL_CONTA", "ORDEM_EXERC", "ESCALA_MOEDA", "DS_CONTA"])


def _pick_scope(con: pd.DataFrame, ind: pd.DataFrame) -> pd.DataFrame:
    """Consolidado por documento quando existe; senão individual."""
    if con.empty:
        return ind
    if ind.empty:
        return con
    have = set(zip(con["CD_CVM"], con["DT_REFER"]))
    keep_ind = ind[[k not in have for k in zip(ind["CD_CVM"], ind["DT_REFER"])]]
    return pd.concat([con, keep_ind], ignore_index=True)


# ---------------------------------------------------------------------------
# Seleção de contas
# ---------------------------------------------------------------------------

def _first(rows: pd.DataFrame, pattern: str, depth: Optional[int] = None,
           code: Optional[str] = None) -> float:
    r = rows
    if code is not None:
        r = r[r["CD_CONTA"] == code]
    if depth is not None:
        r = r[r["CD_CONTA"].str.count(r"\.") == depth]
    r = r[r["ds"].str.contains(pattern, regex=True)]
    if r.empty:
        return np.nan
    return float(r.sort_values("CD_CONTA").iloc[0]["value"])


def _child(rows: pd.DataFrame, parent_pattern: str, child_pattern: str) -> float:
    parents = rows[rows["ds"].str.contains(parent_pattern, regex=True)]
    if parents.empty:
        return np.nan
    p = parents.sort_values("CD_CONTA").iloc[0]["CD_CONTA"]
    kids = rows[rows["CD_CONTA"].str.startswith(p + ".") & rows["ds"].str.contains(child_pattern, regex=True)]
    if kids.empty:
        return np.nan
    return float(kids.sort_values("CD_CONTA").iloc[0]["value"])


_NI = r"^lucro.*(?:consolidado|liquido|prejuizo).*do periodo|^lucro/prejuizo do periodo|^lucro ou prejuizo liquido"


def _flows(dre: pd.DataFrame, dfc: pd.DataFrame) -> dict[str, float]:
    out = {k: np.nan for k in FLOW_FIELDS}
    if not dre.empty:
        out["revenue"] = _first(dre, r".", code="3.01")
        out["gross_profit"] = _first(dre, r"resultado bruto", depth=1)
        out["ebit"] = _first(dre, r"^resultado antes do resultado financeiro", depth=1)
        out["pretax_income"] = _first(dre, r"^resultado antes dos tributos", depth=1)
        out["net_income"] = _first(dre, _NI, depth=1)
        ctrl = _child(dre[dre["CD_CONTA"].str.count(r"\.") <= 2], _NI, r"controladora")
        out["net_income_controlling"] = ctrl if not np.isnan(ctrl) else out["net_income"]
    if not dfc.empty:
        out["operating_cash_flow"] = _first(dfc, r"operaciona", code="6.01")
        inv = dfc[dfc["CD_CONTA"].str.startswith("6.02.")]
        capex = inv[inv["ds"].str.contains(r"imobilizado|intangivel|ativo fixo|investimentos em ativos")
                    & ~inv["ds"].str.contains(r"venda|alienacao|baixa|recebimento")
                    & (inv["value"] < 0)]
        # só folhas, para não somar pai e filho
        capex = capex[[not any(o.startswith(c + ".") for o in capex["CD_CONTA"]) for c in capex["CD_CONTA"]]]
        out["capex"] = float(capex["value"].sum()) if not capex.empty else np.nan
        adj = dfc[dfc["CD_CONTA"].str.startswith("6.01.01.")]
        dna = adj[adj["ds"].str.contains(r"deprecia|amortiza|exaust")]
        dna = dna[[not any(o.startswith(c + ".") for o in dna["CD_CONTA"]) for c in dna["CD_CONTA"]]]
        out["dna"] = float(dna["value"].sum()) if not dna.empty else np.nan
    return out


def _stocks(bpa: pd.DataFrame, bpp: pd.DataFrame) -> dict[str, float]:
    out = {k: np.nan for k in STOCK_FIELDS}
    if not bpa.empty:
        out["total_assets"] = _first(bpa, r".", code="1")
        out["cash"] = _first(bpa, r"^caixa e equivalentes")
        out["short_term_investments"] = _first(bpa[bpa["CD_CONTA"].str.startswith("1.01.")],
                                               r"^aplicacoes financeiras", depth=2)
    if not bpp.empty:
        eq = _first(bpp, r"patrimonio liquido", depth=1)
        out["equity"] = eq
        nci = _child(bpp[bpp["CD_CONTA"].str.count(r"\.") <= 2], r"patrimonio liquido", r"nao controlador")
        out["equity_controlling"] = eq - nci if not np.isnan(nci) else eq
        debt = bpp[bpp["CD_CONTA"].str.match(r"^2\.0[12]\.\d\d$")
                   & bpp["ds"].str.contains(r"^emprestimos e financiamentos")]
        out["gross_debt"] = float(debt["value"].sum()) if not debt.empty else np.nan
    return out


def _is_bank(dre: pd.DataFrame) -> bool:
    r = dre[dre["CD_CONTA"] == "3.01"]
    return bool(not r.empty and r["ds"].str.contains("intermediacao financeira").any())


def _is_insurer(dre: pd.DataFrame) -> bool:
    r = dre[dre["CD_CONTA"].str.count(r"\.") <= 1]
    return bool(r["ds"].str.contains(r"premios|operacoes de seguro|resseguro").any())


# ---------------------------------------------------------------------------
# Extração por documento
# ---------------------------------------------------------------------------

def _ytd_rows(df: pd.DataFrame, ordem: str) -> tuple[pd.DataFrame, Optional[str], Optional[str]]:
    """Linhas do acumulado no ano (DT_INI mais antigo) da ordem pedida."""
    r = df[df["ordem"] == ordem]
    if r.empty:
        return r, None, None
    ini = r["DT_INI_EXERC"].dropna().min()
    if ini is None or (isinstance(ini, float) and np.isnan(ini)):
        return r, None, None
    r = r[r["DT_INI_EXERC"] == ini]
    return r, ini, r["DT_FIM_EXERC"].max()


def extract_year(form: str, year: int) -> pd.DataFrame:
    """Uma linha por documento (cd_cvm, dt_refer) com acumulados e saldos."""
    path = zip_path(form, year)
    if not path.exists():
        return pd.DataFrame()
    z = zipfile.ZipFile(path)
    dre = _pick_scope(_statement(z, form, year, "DRE", "con", max_depth=2),
                      _statement(z, form, year, "DRE", "ind", max_depth=2))
    dfc = pd.concat([
        _pick_scope(_statement(z, form, year, f"DFC_{m}", "con", prefixes=("6.01", "6.02")),
                    _statement(z, form, year, f"DFC_{m}", "ind", prefixes=("6.01", "6.02")))
        for m in ("MI", "MD")
    ], ignore_index=True)
    bpa = _pick_scope(_statement(z, form, year, "BPA", "con", max_depth=2),
                      _statement(z, form, year, "BPA", "ind", max_depth=2))
    bpp = _pick_scope(_statement(z, form, year, "BPP", "con", max_depth=2),
                      _statement(z, form, year, "BPP", "ind", max_depth=2))
    bpa, bpp = bpa[bpa["ordem"] == "cur"], bpp[bpp["ordem"] == "cur"]

    def groups(df):
        return {k: g for k, g in df.groupby(["CD_CVM", "DT_REFER"])} if not df.empty else {}

    G = {n: groups(d) for n, d in (("dre", dre), ("dfc", dfc), ("bpa", bpa), ("bpp", bpp))}
    keys = set().union(*[set(g) for g in G.values()])
    empty = pd.DataFrame(columns=["CD_CONTA", "ds", "value", "ordem", "DT_INI_EXERC", "DT_FIM_EXERC", "scope"])
    rows = []
    for key in sorted(keys):
        d = {n: G[n].get(key, empty) for n in G}
        dre_cur, ini, fim = _ytd_rows(d["dre"], "cur")
        dre_prev, _, _ = _ytd_rows(d["dre"], "prev")
        dfc_cur, dini, dfim = _ytd_rows(d["dfc"], "cur")
        dfc_prev, _, _ = _ytd_rows(d["dfc"], "prev")
        ini, fim = ini or dini, fim or dfim
        cur = _flows(dre_cur, dfc_cur)
        prev = _flows(dre_prev, dfc_prev)
        row = {
            "cd_cvm": key[0], "dt_refer": key[1], "form": form,
            "scope": (d["dre"]["scope"].iloc[0] if not d["dre"].empty
                      else d["bpp"]["scope"].iloc[0] if not d["bpp"].empty else None),
            "fy_start": ini, "ytd_end": fim,
            "ytd_months": _months(ini, fim),
            "is_bank": _is_bank(d["dre"]), "is_insurer": _is_insurer(d["dre"]),
            **{f"{k}_ytd": v for k, v in cur.items()},
            **{f"{k}_ytd_prev": v for k, v in prev.items()},
            **_stocks(d["bpa"], d["bpp"]),
        }
        if row["is_bank"]:
            for k in ("capex", "dna"):
                row[f"{k}_ytd"] = row[f"{k}_ytd_prev"] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    shares = extract_shares(z, form, year)
    if not out.empty and not shares.empty:
        out = out.merge(shares, on=["cd_cvm", "dt_refer"], how="left")
    return out


def _months(ini: Optional[str], fim: Optional[str]) -> Optional[int]:
    if not ini or not fim:
        return None
    a, b = pd.Timestamp(ini), pd.Timestamp(fim)
    return int(round((b - a).days / 30.44))


def extract_shares(z: zipfile.ZipFile, form: str, year: int) -> pd.DataFrame:
    df = _read(z, f"{form.lower()}_cia_aberta_composicao_capital_{year}.csv")
    if df.empty:
        return df
    df = df.sort_values("VERSAO", key=lambda s: pd.to_numeric(s, errors="coerce")).drop_duplicates(
        ["CNPJ_CIA", "DT_REFER"], keep="last")
    m = {
        "QT_ACAO_ORDIN_CAP_INTEGR": "shares_on", "QT_ACAO_PREF_CAP_INTEGR": "shares_pn",
        "QT_ACAO_TOTAL_CAP_INTEGR": "shares_total", "QT_ACAO_ORDIN_TESOURO": "treasury_on",
        "QT_ACAO_PREF_TESOURO": "treasury_pn", "QT_ACAO_TOTAL_TESOURO": "treasury_total",
    }
    out = df[["CNPJ_CIA", "DT_REFER", *[c for c in m if c in df.columns]]].rename(columns=m)
    for c in m.values():
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    idx = _read(z, f"{form.lower()}_cia_aberta_{year}.csv", usecols=["CNPJ_CIA", "CD_CVM"]).drop_duplicates()
    out = out.merge(idx, on="CNPJ_CIA", how="left").rename(columns={"CD_CVM": "cd_cvm", "DT_REFER": "dt_refer"})
    return out.drop(columns=["CNPJ_CIA"]).dropna(subset=["cd_cvm"]).drop_duplicates(["cd_cvm", "dt_refer"])


def extract_index(form: str, year: int) -> pd.DataFrame:
    """Todas as versões de cada documento, com a data de entrega."""
    path = zip_path(form, year)
    if not path.exists():
        return pd.DataFrame()
    z = zipfile.ZipFile(path)
    df = _read(z, f"{form.lower()}_cia_aberta_{year}.csv")
    if df.empty:
        return df
    df = df.rename(columns={"CD_CVM": "cd_cvm", "DT_REFER": "dt_refer", "CNPJ_CIA": "cnpj",
                            "DENOM_CIA": "name", "VERSAO": "versao", "DT_RECEB": "dt_receb"})
    df["form"] = form
    return df[["cd_cvm", "cnpj", "name", "dt_refer", "versao", "dt_receb", "form"]]


# ---------------------------------------------------------------------------
# Montagem da base PIT
# ---------------------------------------------------------------------------

def add_ttm(docs: pd.DataFrame) -> pd.DataFrame:
    """TTM das contas de fluxo: DFP = anual; ITR = acumulado + anual anterior − comparativo."""
    docs = docs.copy()
    fy_start = pd.to_datetime(docs["fy_start"])
    docs["_prev_fy_end"] = (fy_start - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    annual = docs[docs["form"] == "DFP"].sort_values("available_from").drop_duplicates(
        ["cd_cvm", "dt_refer"], keep="last")
    annual = annual[["cd_cvm", "dt_refer", *[f"{f}_ytd" for f in FLOW_FIELDS]]].rename(
        columns={"dt_refer": "_prev_fy_end", **{f"{f}_ytd": f"{f}_prev_annual" for f in FLOW_FIELDS}})
    docs = docs.merge(annual, on=["cd_cvm", "_prev_fy_end"], how="left")
    full_year = (docs["form"] == "DFP") | (docs["ytd_months"] == 12)
    for f in FLOW_FIELDS:
        partial = docs[f"{f}_ytd"] + docs[f"{f}_prev_annual"] - docs[f"{f}_ytd_prev"]
        docs[f"{f}_ttm"] = np.where(full_year, docs[f"{f}_ytd"], partial)
    return docs.drop(columns=["_prev_fy_end", *[f"{f}_prev_annual" for f in FLOW_FIELDS]])


def build(years: Optional[Iterable[int]] = None, forms: Iterable[str] = ("DFP", "ITR"),
          save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extrai todos os anos e devolve (docs, versions). Salva parquet se save."""
    years = list(years) if years is not None else list(range(min(FIRST_YEAR.values()), pd.Timestamp.today().year + 1))
    docs, versions = [], []
    for form in forms:
        for y in years:
            if y < FIRST_YEAR[form]:
                continue
            logger.info("Extraindo %s %d", form, y)
            d = extract_year(form, y)
            if not d.empty:
                docs.append(d)
            v = extract_index(form, y)
            if not v.empty:
                versions.append(v)
    docs_df = pd.concat(docs, ignore_index=True) if docs else pd.DataFrame()
    ver_df = pd.concat(versions, ignore_index=True).drop_duplicates() if versions else pd.DataFrame()
    if not docs_df.empty:
        docs_df = attach_availability(docs_df, ver_df)
        docs_df = add_ttm(docs_df)
    if save:
        out = processed_dir()
        out.mkdir(parents=True, exist_ok=True)
        docs_df.to_parquet(out / "fundamentals_docs.parquet", index=False)
        ver_df.to_parquet(out / "document_versions.parquet", index=False)
    return docs_df, ver_df


def attach_availability(docs: pd.DataFrame, versions: pd.DataFrame) -> pd.DataFrame:
    v = versions.copy()
    v["versao_n"] = pd.to_numeric(v["versao"], errors="coerce")
    agg = v.groupby(["cd_cvm", "dt_refer", "form"]).agg(
        available_from=("dt_receb", "min"), last_receb=("dt_receb", "max"),
        n_versions=("versao_n", "nunique"), cnpj=("cnpj", "last"), name=("name", "last"),
    ).reset_index()
    out = docs.merge(agg, on=["cd_cvm", "dt_refer", "form"], how="left")
    # Documento sem linha no índice não tem data de entrega: não dá para usar PIT.
    missing = out["available_from"].isna().sum()
    if missing:
        logger.warning("%d documentos sem DT_RECEB no índice — descartados", missing)
    return out.dropna(subset=["available_from"])


def load_docs() -> pd.DataFrame:
    return pd.read_parquet(processed_dir() / "fundamentals_docs.parquet")


def fundamentals_as_of(as_of, docs: Optional[pd.DataFrame] = None,
                       max_staleness_days: int = 500) -> pd.DataFrame:
    """
    Último documento de cada companhia entregue até `as_of` (inclusive).

    Entre documentos já entregues, vale o de DT_REFER mais recente; empate
    (DFP e ITR na mesma data, raro) fica com o entregue por último.
    Companhias cujo último período é mais velho que `max_staleness_days`
    saem (deslistadas ou inadimplentes com a CVM).
    """
    docs = load_docs() if docs is None else docs
    as_of = pd.Timestamp(as_of)
    d = docs[pd.to_datetime(docs["available_from"]) <= as_of].copy()
    if d.empty:
        return d
    d["_ref"] = pd.to_datetime(d["dt_refer"])
    d = d[(as_of - d["_ref"]).dt.days <= max_staleness_days]
    d = d.sort_values(["cd_cvm", "_ref", "available_from"]).groupby("cd_cvm").tail(1)
    return d.drop(columns=["_ref"]).reset_index(drop=True)
