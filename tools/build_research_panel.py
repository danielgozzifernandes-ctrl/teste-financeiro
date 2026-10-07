"""Monta e salva o painel mensal do estudo de fatores em B3_DATA_DIR/derived."""
import logging
import time

from src.history.paths import DERIVED_DIR
from src.history.research import build_panel

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    t0 = time.time()
    p = build_panel()
    p.attrs["ibov"].rename("ibov").to_frame().to_parquet(DERIVED_DIR / "bench_ibov.parquet")
    p.attrs["cdi"].rename("cdi").to_frame().to_parquet(DERIVED_DIR / "bench_cdi.parquet")
    p.attrs = {}
    p.to_parquet(DERIVED_DIR / "research_panel.parquet")
    print(p.shape, round(time.time() - t0), "s")
