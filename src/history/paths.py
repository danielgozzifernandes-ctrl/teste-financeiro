import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("B3_DATA_DIR", r"C:\Users\danie\b3_data"))
RAW_DIR = DATA_DIR / "cotahist" / "raw"
PARQUET_DIR = DATA_DIR / "cotahist" / "parquet"
CORP_DIR = DATA_DIR / "corporate_actions"
DERIVED_DIR = DATA_DIR / "derived"
