import time
import pandas as pd
from pandas_datareader import data as pdr
from utils.helpers import parquet_path, incremental_append, load_parquet, compute_fetch_start

def update_rates(cfg: dict, storage_root: str):
    if not cfg.get("enabled", False):
        return

    series        = cfg.get("fred_series", [])
    history_start = cfg.get("start", "2010-01-01")
    overlap_days  = int(cfg.get("overlap_days", 5))

    for s in series:
        path  = parquet_path(storage_root, "rates", s)
        exist = load_parquet(path)
        start = compute_fetch_start(
            exist.index if not exist.empty else None,
            history_start=history_start,
            overlap_days=overlap_days
        )

        try:
            df = pdr.DataReader(s, "fred", start)   # index=Date, col=series
        except Exception as e:
            print(f"[WARN] {s}: download error: {e}")
            continue

        if df.empty:
            print(f"[INFO] {s}: no rows returned for start={start}")
            continue

        df.index = pd.to_datetime(df.index); df.index.name = "Date"
        incremental_append(df, path, index_name="Date")
        print(f"[OK] {s}: window={start}..today rows={len(df):,} -> {path}")
