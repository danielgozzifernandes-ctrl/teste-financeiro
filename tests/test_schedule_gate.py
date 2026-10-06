"""tests/test_schedule_gate.py — quando os workflows agendados rodam."""

from datetime import date

import pytest

from tools.schedule_gate import main, should_run


@pytest.mark.parametrize("kind, d, expected", [
    ("daily",   "2026-10-09", True),
    ("daily",   "2026-10-12", False),   # feriado
    ("daily",   "2026-10-10", False),   # sábado
    ("weekly",  "2026-10-05", True),
    ("weekly",  "2026-10-06", False),
    ("weekly",  "2026-10-12", False),   # segunda feriado...
    ("weekly",  "2026-10-13", True),    # ...roda na terça
    ("weekly",  "2026-02-18", True),    # carnaval seg+ter → quarta
    ("weekly",  "2026-02-16", False),
    ("monthly", "2026-11-03", True),    # 01/11 domingo, 02/11 feriado
    ("monthly", "2026-11-02", False),
    ("monthly", "2027-01-04", True),    # 01/01 sexta feriado
    ("monthly", "2026-12-01", True),
    ("monthly", "2026-12-02", False),
])
def test_should_run(kind, d, expected):
    assert should_run(kind, date.fromisoformat(d)) is expected


def test_writes_github_output(tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    assert main(["weekly", "2026-10-12"]) == 0
    assert out.read_text(encoding="utf-8").strip() == "run=false"
