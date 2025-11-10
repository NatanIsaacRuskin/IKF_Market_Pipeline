import pandas as pd
import yfinance as yf
from pathlib import Path
from utils.helpers import parquet_path, incremental_append, load_parquet

def update_futures(cfg: dict, storage_root: str, default_start="2010-01-01"):
    if not cfg.get("enabled", False):
        return
    tickers = cfg.get("tickers", [])
    interval = cfg.get("price_interval","1d")
    for t in tickers:
        stem = t.replace("=F","")
        path = parquet_path(storage_root, "futures", stem)
        exist = load_parquet(path)
        start = (exist.index.max() + pd.Timedelta(days=1)).date().isoformat() if not exist.empty else default_start
        df = yf.download(t, start=start, interval=interval, auto_adjust=False, progress=False)
        if df.empty:
            continue
        df = df.rename(columns=str.capitalize)
        df.index = pd.to_datetime(df.index); df.index.name = "Date"
        incremental_append(df, path, index_name="Date")
