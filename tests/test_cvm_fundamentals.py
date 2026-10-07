import numpy as np
import pandas as pd
import pytest

from src.history.cvm_fundamentals import (
    _flows,
    _is_bank,
    _stocks,
    add_ttm,
    attach_availability,
    fundamentals_as_of,
)
from src.history.cvm_reference import fix_share_scale


def _rows(items):
    """items: [(CD_CONTA, descrição normalizada, valor)]"""
    return pd.DataFrame(items, columns=["CD_CONTA", "ds", "value"])


# ---------------------------------------------------------------------------
# Seleção de contas
# ---------------------------------------------------------------------------

def test_industrial_flows_and_stocks():
    dre = _rows([
        ("3.01", "receita de venda de bens e/ou servicos", 100.0),
        ("3.03", "resultado bruto", 40.0),
        ("3.05", "resultado antes do resultado financeiro e dos tributos", 25.0),
        ("3.07", "resultado antes dos tributos sobre o lucro", 20.0),
        ("3.11", "lucro/prejuizo consolidado do periodo", 15.0),
        ("3.11.01", "atribuido a socios da empresa controladora", 14.0),
        ("3.11.02", "atribuido a socios nao controladores", 1.0),
    ])
    dfc = _rows([
        ("6.01", "caixa liquido atividades operacionais", 30.0),
        ("6.01.01", "caixa gerado nas operacoes", 35.0),
        ("6.01.01.02", "depreciacao, amortizacao e exaustao", 8.0),
        ("6.02", "caixa liquido atividades de investimento", -12.0),
        ("6.02.01", "aquisicoes de imobilizado e intangivel", -10.0),
        ("6.02.02", "recebimento pela venda de imobilizado", 1.0),
    ])
    f = _flows(dre, dfc)
    assert f["revenue"] == 100 and f["ebit"] == 25 and f["pretax_income"] == 20
    assert f["net_income"] == 15 and f["net_income_controlling"] == 14
    assert f["operating_cash_flow"] == 30 and f["capex"] == -10 and f["dna"] == 8

    bpa = _rows([("1", "ativo total", 500.0), ("1.01", "ativo circulante", 200.0),
                 ("1.01.01", "caixa e equivalentes de caixa", 50.0),
                 ("1.01.02", "aplicacoes financeiras", 20.0)])
    bpp = _rows([("2", "passivo total", 500.0),
                 ("2.01.04", "emprestimos e financiamentos", 40.0),
                 ("2.02.01", "emprestimos e financiamentos", 60.0),
                 ("2.03", "patrimonio liquido consolidado", 210.0),
                 ("2.03.09", "participacao dos acionistas nao controladores", 10.0)])
    s = _stocks(bpa, bpp)
    assert s["total_assets"] == 500 and s["cash"] == 50 and s["short_term_investments"] == 20
    assert s["gross_debt"] == 100
    assert s["equity"] == 210 and s["equity_controlling"] == 200


def test_bank_template():
    dre = _rows([
        ("3.01", "receitas da intermediacao financeira", 199.0),
        ("3.03", "resultado bruto intermediacao financeira", 72.0),
        ("3.05", "resultado antes dos tributos sobre o lucro", 27.0),
        ("3.09", "lucro/prejuizo consolidado do periodo", 24.2),
        ("3.09.01", "atribuido a socios da empresa controladora", 23.6),
    ])
    assert _is_bank(dre)
    f = _flows(dre, _rows([]))
    assert np.isnan(f["ebit"])
    assert f["pretax_income"] == 27 and f["net_income_controlling"] == 23.6
    bpp = _rows([("2.08", "patrimonio liquido consolidado", 228.0)])
    s = _stocks(_rows([("1", "ativo total", 3200.0)]), bpp)
    assert s["equity"] == 228 and np.isnan(s["gross_debt"])


def test_individual_statement_without_attribution_uses_net_income():
    dre = _rows([("3.11", "lucro/prejuizo do periodo", 9.0)])
    f = _flows(dre, _rows([]))
    assert f["net_income"] == 9 and f["net_income_controlling"] == 9


# ---------------------------------------------------------------------------
# TTM e point-in-time
# ---------------------------------------------------------------------------

def _doc(form, dt_refer, ytd, ytd_prev, fy_start, months):
    return {"cd_cvm": "1", "form": form, "dt_refer": dt_refer, "fy_start": fy_start,
            "ytd_months": months, "net_income_ytd": ytd, "net_income_ytd_prev": ytd_prev,
            **{f"{f}_{s}": np.nan for f in ("revenue", "gross_profit", "ebit", "pretax_income",
                                            "net_income_controlling", "operating_cash_flow", "capex", "dna")
               for s in ("ytd", "ytd_prev")},
            "available_from": "2026-01-01"}


def test_ttm_from_quarterly_ytd():
    docs = pd.DataFrame([
        _doc("DFP", "2025-12-31", 110.0, 90.0, "2025-01-01", 12),
        _doc("ITR", "2026-06-30", 85.0, 62.0, "2026-01-01", 6),
    ])
    out = add_ttm(docs).set_index("dt_refer")
    assert out.loc["2025-12-31", "net_income_ttm"] == 110
    assert out.loc["2026-06-30", "net_income_ttm"] == pytest.approx(85 + 110 - 62)


def test_ttm_missing_prior_annual_is_nan():
    docs = pd.DataFrame([_doc("ITR", "2026-06-30", 85.0, 62.0, "2026-01-01", 6)])
    assert np.isnan(add_ttm(docs)["net_income_ttm"].iloc[0])


def test_availability_uses_first_version_and_flags_restatement():
    docs = pd.DataFrame([{"cd_cvm": "1", "dt_refer": "2026-03-31", "form": "ITR", "x": 1}])
    versions = pd.DataFrame([
        {"cd_cvm": "1", "cnpj": "c", "name": "N", "dt_refer": "2026-03-31", "versao": "1",
         "dt_receb": "2026-05-11", "form": "ITR"},
        {"cd_cvm": "1", "cnpj": "c", "name": "N", "dt_refer": "2026-03-31", "versao": "2",
         "dt_receb": "2026-07-02", "form": "ITR"},
    ])
    out = attach_availability(docs, versions).iloc[0]
    assert out["available_from"] == "2026-05-11"
    assert out["last_receb"] == "2026-07-02"
    assert out["n_versions"] == 2


def test_fundamentals_as_of_never_looks_ahead():
    docs = pd.DataFrame([
        {"cd_cvm": "1", "dt_refer": "2025-12-31", "available_from": "2026-03-05", "v": "annual"},
        {"cd_cvm": "1", "dt_refer": "2026-03-31", "available_from": "2026-05-11", "v": "q1"},
        {"cd_cvm": "2", "dt_refer": "2023-12-31", "available_from": "2024-03-01", "v": "stale"},
    ])
    assert fundamentals_as_of("2026-05-10", docs)["v"].tolist() == ["annual"]
    assert fundamentals_as_of("2026-05-11", docs)["v"].tolist() == ["q1"]
    assert fundamentals_as_of("2024-04-01", docs)["v"].tolist() == ["stale"]
    assert fundamentals_as_of("2026-03-04", docs).empty  # o de 2023 ficou velho demais


def test_share_scale_fixed_by_fre_and_bvps():
    docs = pd.DataFrame([
        {"cd_cvm": "A", "dt_refer": "2026-03-31", "shares_total": 4_262_534.0,
         "shares_on": 4_262_534.0, "shares_pn": 0.0, "treasury_total": 174_626.0,
         "equity_controlling": 191e9},
        {"cd_cvm": "B", "dt_refer": "2026-03-31", "shares_total": 12_888_732_761.0,
         "shares_on": 7_442_231_382.0, "shares_pn": 5_446_501_379.0, "treasury_total": 0.0,
         "equity_controlling": 445e9},
        {"cd_cvm": "C", "dt_refer": "2026-03-31", "shares_total": 15_763_665.0,
         "shares_on": 15_763_665.0, "shares_pn": 0.0, "treasury_total": 166_971.0,
         "equity_controlling": 90e9},
    ])
    fre = pd.DataFrame([
        {"cd_cvm": "A", "dt_refer": "2026-01-01", "shares_total": 4_539_007_580.0},
        {"cd_cvm": "B", "dt_refer": "2026-01-01", "shares_total": 12_888_732_761.0},
    ])
    out = fix_share_scale(docs, fre).set_index("cd_cvm")
    assert out.loc["A", "shares_total"] == 4_262_534_000 and out.loc["A", "shares_scale_source"] == "fre"
    assert out.loc["A", "treasury_total"] == 174_626_000
    assert out.loc["B", "shares_total"] == 12_888_732_761
    assert out.loc["C", "shares_total"] == 15_763_665_000 and out.loc["C", "shares_scale_source"] == "bvps"
