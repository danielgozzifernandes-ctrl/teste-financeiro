"""
Monta a base de fundamentos point-in-time da CVM em <B3_DATA_DIR>/cvm/processed.

    python tools/build_fundamentals.py             # baixa o que falta e reconstrói
    python tools/build_fundamentals.py --no-download
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.history.cvm_download import download_all  # noqa: E402
from src.history.cvm_fundamentals import build, processed_dir  # noqa: E402
from src.history.cvm_reference import build_reference, fix_share_scale  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--skip-extract", action="store_true",
                    help="reaproveita fundamentals_docs_raw.parquet da última extração")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not args.no_download:
        download_all()
    if args.skip_extract:
        docs = pd.read_parquet(processed_dir() / "fundamentals_docs_raw.parquet")
        versions = pd.read_parquet(processed_dir() / "document_versions.parquet")
    else:
        docs, versions = build(save=False)
        processed_dir().mkdir(parents=True, exist_ok=True)
        docs.to_parquet(processed_dir() / "fundamentals_docs_raw.parquet", index=False)
        versions.to_parquet(processed_dir() / "document_versions.parquet", index=False)
    ref = build_reference(save=True)
    docs = fix_share_scale(docs, ref["fre_shares"])
    sec = ref["sectors"].sort_values("year").drop_duplicates("cd_cvm", keep="last")[["cd_cvm", "sector", "control"]]
    docs = docs.merge(sec, on="cd_cvm", how="left")
    out = processed_dir()
    out.mkdir(parents=True, exist_ok=True)
    docs.to_parquet(out / "fundamentals_docs.parquet", index=False)
    logging.info("fundamentals_docs: %d documentos, %d companhias", len(docs), docs["cd_cvm"].nunique())
    return 0


if __name__ == "__main__":
    sys.exit(main())
