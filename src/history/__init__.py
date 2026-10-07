"""Bases históricas para pesquisa (fora do pipeline semanal)."""

import os
from pathlib import Path


def data_dir() -> Path:
    """Raiz dos dados brutos/derivados; nunca dentro do repositório."""
    return Path(os.environ.get("B3_DATA_DIR", Path.home() / "b3_data"))
