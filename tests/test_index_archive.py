import json

import pytest

import tools.archive_index_portfolios as arc


def _raw(qty_petr="4.566.445.248"):
    return {
        "header": {"date": "06/10/26", "reductor": "14.011.295,44870142",
                   "theoricalQty": "93.803.750.400"},
        "results": [
            {"cod": "VALE3", "asset": "VALE", "type": "ON      NM",
             "part": "10,123", "theoricalQty": "4.196.924.316"},
            {"cod": "PETR4", "asset": "PETROBRAS", "type": "PN      N2",
             "part": "8,500", "theoricalQty": qty_petr},
        ],
    }


class _Resp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class _Session:
    def __init__(self, data):
        self.data = data

    def get(self, url, timeout):
        return _Resp(self.data)


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(arc, "OUT_DIR", tmp_path)
    monkeypatch.setattr(arc, "INDICES", ["IBOV"])
    monkeypatch.setattr(arc, "CALLS", {"day": "GetPortfolioDay"})
    return tmp_path


def test_normalize_parses_brazilian_numbers():
    snap = arc.normalize(_raw(), "IBOV", "day", "2026-10-06")
    assert [m["ticker"] for m in snap["members"]] == ["PETR4", "VALE3"]
    assert snap["members"][1]["weight_pct"] == pytest.approx(10.123)
    assert snap["members"][0]["theoretical_qty"] == 4_566_445_248
    assert snap["reductor"] == pytest.approx(14_011_295.44870142)
    assert snap["members"][0]["type"] == "PN N2"


def test_writes_only_when_composition_changes(out):
    assert len(arc.archive(_Session(_raw()), "2026-10-06")) == 1
    assert arc.archive(_Session(_raw()), "2026-10-07") == []
    written = arc.archive(_Session(_raw(qty_petr="4.600.000.000")), "2026-10-08")
    assert [p.name for p in written] == ["IBOV_day_2026-10-08.json"]
    saved = json.loads(written[0].read_text(encoding="utf-8"))
    assert saved["fetched_on"] == "2026-10-08"


def test_network_failure_is_not_fatal(out):
    class Boom:
        def get(self, url, timeout):
            raise ConnectionError("offline")

    assert arc.archive(Boom(), "2026-10-06") == []
