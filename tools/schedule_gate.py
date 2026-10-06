"""
Decide se um run agendado roda hoje, pelo calendário da B3 (horário de Brasília).

    daily    todo pregão
    weekly   primeiro pregão da semana (segunda; terça se segunda for feriado...)
    monthly  primeiro pregão do mês

Só usa a stdlib, para rodar no workflow antes de instalar dependências:

    python3 tools/schedule_gate.py weekly [YYYY-MM-DD]

Escreve run=true|false em $GITHUB_OUTPUT quando existe.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.b3_calendar import is_trading_day, previous_trading_day, today_brt  # noqa: E402


def should_run(kind: str, d: date) -> bool:
    if not is_trading_day(d):
        return False
    if kind == "daily":
        return True
    prev = previous_trading_day(d)
    if kind == "weekly":
        return prev.isocalendar()[:2] != d.isocalendar()[:2]
    if kind == "monthly":
        return (prev.year, prev.month) != (d.year, d.month)
    raise ValueError(f"tipo desconhecido: {kind}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    kind = argv[0]
    d = date.fromisoformat(argv[1]) if len(argv) > 1 else today_brt()
    run = should_run(kind, d)
    print(f"{kind} {d}: {'roda' if run else 'pula'}")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"run={'true' if run else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
