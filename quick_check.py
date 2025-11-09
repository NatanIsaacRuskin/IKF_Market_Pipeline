from pathlib import Path
import pandas as pd

def info_glob(pattern):
    files = sorted(Path(pattern).glob("*.parquet"))
    rows = 0
    for f in files:
        df = pd.read_parquet(f)
        print(f"{f.name:25s}  rows={len(df):7d}  start={df.index.min()}  end={df.index.max()}")
        rows += len(df)
    print(f"TOTAL files={len(files)} rows={rows}\n")

print("== Equities ==")
info_glob("data/raw/equities")

print("== Futures ==")
info_glob("data/raw/futures")

print("== Rates ==")
info_glob("data/raw/rates")

print("== Options (AAPL calls example) ==")
for f in sorted(Path("data/raw/options/AAPL").glob("calls_*.parquet")):
    df = pd.read_parquet(f)
    print(f"{f.name:30s} rows={len(df):6d}")
    