import numpy as np
import pandas as pd
import pytest

from src.history.adjust import build_series
from src.history.corporate import cash_events, share_multiplier, stock_events
from src.history.cotahist import parse_lines
from src.history.renames import detect_renames, security_ids
from src.history.universe import liquid_universe, rebalance_dates

# Linhas reais do COTAHIST_A2026 (02/10/2026)
PETR4 = "012026100202PETR4       010PETROBRAS   PN      N2   R$  000000000497400000000051350000000004940000000000505000000000051170000000005117000000000511867446000000000057293500000000289368349400000000000000009999123100000010000000000000BRPETRACNPR6230"
PETR4F = "012026100296PETR4F      020PETROBRAS   PN      N2   R$  000000000497700000000051300000000004940000000000504600000000051040000000005104000000000511015793000000000000173594000000000876057434000000000000009999123100000010000000000000BRPETRACNPR6230"
BOVA11 = "012026100214BOVA11      010ISHARES BOVACI           R$  000000001872000000000190000000000018322000000001870300000000190000000000019000000000001900265239000000000006945869000000129908660869000000000000009999123100000010000000000000BRBOVACTF003120"
AAPL34 = "012026100234AAPL34      010APPLE       DRN          R$  000000000866700000000087410000000008642000000000870500000000087060000000008700000000000870601222000000000000237517000000002067729449000000000000009999123100000010000000000000BRAAPLBDR004160"
HEADER = "00COTAHIST.2026BOVESPA 20261005" + " " * 214


def test_parser_reads_real_line_and_keeps_only_lot_spot():
    df = parse_lines([HEADER, PETR4, PETR4F, BOVA11, AAPL34])
    assert df["ticker"].tolist() == ["PETR4"]
    row = df.iloc[0]
    assert row["date"] == pd.Timestamp("2026-10-02")
    assert row["close"] == pytest.approx(51.17)
    assert row["open"] == pytest.approx(49.74)
    assert row["high"] == pytest.approx(51.35)
    assert row["low"] == pytest.approx(49.40)
    assert row["volume"] == pytest.approx(2_893_683_494.0)
    assert row["trades"] == 67446
    assert row["isin"] == "BRPETRACNPR6"
    assert row["spec"] == "PN N2"


def test_parser_divides_by_fatcot():
    line = PETR4[:210] + "0001000" + PETR4[217:]
    assert parse_lines([line]).iloc[0]["close"] == pytest.approx(0.05117)


def test_share_multiplier_conventions():
    assert share_multiplier("DESDOBRAMENTO", 100.0) == 2.0
    assert share_multiplier("BONIFICACAO", 10.0) == pytest.approx(1.10)
    assert share_multiplier("GRUPAMENTO", 0.1) == pytest.approx(0.1)
    assert share_multiplier("INCORPORACAO", 85.0) is None


def test_event_parsing_from_b3_payload():
    issuer = {
        "root": "XPTO",
        "cash": [{"typeStock": "PN", "valueCash": "1,50", "quotedPerShares": "1",
                  "corporateAction": "DIVIDENDO", "lastDatePriorEx": "02/01/2020"},
                 {"typeStock": "PN", "valueCash": "0", "quotedPerShares": "1",
                  "corporateAction": "JRS CAP PROPRIO", "lastDatePriorEx": "02/01/2020"}],
        "stock_events": [{"label": "DESDOBRAMENTO", "factor": "100,00000000000",
                          "lastDatePrior": "06/01/2020", "isinCode": "BRXPTOACNPR0"},
                         {"label": "INCORPORACAO", "factor": "50,0",
                          "lastDatePrior": "06/01/2020", "isinCode": "BRXPTOACNPR0"}],
    }
    c = cash_events(issuer)
    assert len(c) == 1 and c.iloc[0]["value"] == 1.5
    s = stock_events(issuer)
    assert len(s) == 1 and s.iloc[0]["multiplier"] == 2.0


def _px(closes, ticker="XPTO4", isin="BRXPTOACNPR0", start="2020-01-01", spec="PN N1"):
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"date": dates, "ticker": ticker, "isin": isin, "spec": spec,
                         "close": closes, "volume": 1e7, "trades": 1000, "quantity": 1e5})


def test_total_return_with_split_and_dividend():
    # 02/01 data-com de R$1 → ex em 03/01; desdobramento 2:1 com data-com 06/01 → ex 07/01
    px = _px([20.0, 20.0, 19.0, 19.0, 9.5, 9.5])
    cash = pd.DataFrame({"root": ["XPTO"], "type": ["PN"], "value": [1.0],
                         "last_date_prior": [pd.Timestamp("2020-01-02")], "label": ["DIVIDENDO"]})
    stock = pd.DataFrame({"isin": ["BRXPTOACNPR0"], "last_date_prior": [pd.Timestamp("2020-01-06")],
                          "multiplier": [2.0]})
    s, jumps = build_series(px, {"XPTO4": "XPTO4"}, cash, stock)
    s = s.set_index("date")
    assert s.loc["2020-01-03", "ret_total"] == pytest.approx(0.0)      # 19 + 1 = 20
    assert s.loc["2020-01-03", "ret_price"] == pytest.approx(-0.05)
    assert s.loc["2020-01-07", "ret_price"] == pytest.approx(0.0)      # 9.5 * 2 = 19
    assert jumps.empty
    assert s["tr_close"].iloc[-1] == pytest.approx(9.5)
    # retorno total nulo em todo o período; preço caiu só o dividendo (5%)
    assert s["tr_close"].iloc[0] == pytest.approx(9.5)
    assert s["adj_close"].iloc[0] == pytest.approx(10.0)


def test_unexplained_split_is_detected_from_price_jump():
    px = _px([30.0, 30.0, 10.1, 10.0])
    s, jumps = build_series(px, {"XPTO4": "XPTO4"},
                            pd.DataFrame(columns=["root", "type", "value", "last_date_prior", "label"]),
                            pd.DataFrame(columns=["isin", "last_date_prior", "multiplier"]))
    assert len(jumps) == 1 and bool(jumps.iloc[0]["accepted"])
    assert s.set_index("date")["ret_price"].iloc[2] == pytest.approx(10.1 * 3 / 30 - 1)


def test_rename_links_series():
    a = _px([10.0] * 5, ticker="OLDX3", isin="BROLDXACNOR0", spec="ON NM")
    b = _px([10.1] * 5, ticker="NEWX3", isin="BRNEWXACNOR0", spec="ON NM",
            start=str((a["date"].iloc[-1] + pd.offsets.BDay(1)).date()))
    other = _px([5.0] * 10, ticker="ZZZZ3", isin="BRZZZZACNOR0", spec="ON NM")
    df = pd.concat([a, b, other], ignore_index=True)
    df["open"] = df["close"]
    r = detect_renames(df, min_avg_volume=0)
    assert r[["old", "new"]].values.tolist() == [["OLDX3", "NEWX3"]]
    assert security_ids(df, r)["NEWX3"] == "OLDX3"


def test_universe_rule_uses_only_past_data_and_filters():
    dates = pd.bdate_range("2019-01-01", "2019-12-31")
    rows = []
    for t, vol, presence, price in [("AAAA3", 1e9, 1.0, 10), ("BBBB4", 5e8, 1.0, 10),
                                    ("CCCC3", 1e10, 0.5, 10), ("DDDD3", 1e9, 1.0, 0.5),
                                    ("EEEE34", 1e10, 1.0, 10)]:
        for i, d in enumerate(dates):
            if i % int(1 / presence) == 0:
                rows.append({"date": d, "ticker": t, "isin": t, "close": price,
                             "volume": vol, "trades": 100,
                             "spec": "DRN" if t.endswith("34") else "ON NM"})
    df = pd.DataFrame(rows)
    # pregão futuro enorme não pode influenciar
    df = pd.concat([df, pd.DataFrame([{"date": pd.Timestamp("2020-01-03"), "ticker": "BBBB4",
                                       "isin": "BBBB4", "close": 10, "volume": 1e15,
                                       "trades": 10**6, "spec": "ON NM"}])])
    u = liquid_universe(df, pd.Timestamp("2020-01-02"))
    assert u["ticker"].tolist() == ["AAAA3", "BBBB4"]


def test_rebalance_dates_first_session_of_jan_may_sep():
    sessions = pd.bdate_range("2020-01-01", "2020-12-31")
    d = rebalance_dates(sessions, 2020, 2020)
    assert [x.month for x in d] == [1, 5, 9] and d[0] == pd.Timestamp("2020-01-01")
