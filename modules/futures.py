import time
import pandas as pd
import yfinance as yf
from utils.helpers import (
    parquet_path, incremental_append, load_parquet, compute_fetch_start
)

def update_futures(cfg: dict, storage_root: str, default_start="2010-01-01"):
    if not cfg.get("enabled", False):
        return

    tickers       = cfg.get("tickers", [])
    interval      = cfg.get("price_interval", "1d")
    overlap_days  = int(cfg.get("overlap_days", 5))
    history_start = cfg.get("history_start", default_start)

    # reuse gentle knobs if provided
    max_attempts  = int(cfg.get("max_attempts", 3))
    base_sleep    = float(cfg.get("base_sleep", 1.5))
    between_sleep = float(cfg.get("between_sleep", 0.35))

    for t in tickers:
        stem = t.replace("=F","")
        path  = parquet_path(storage_root, "futures", stem)
        exist = load_parquet(path)
        start = compute_fetch_start(
            exist.index if not exist.empty else None,
            history_start=history_start,
            overlap_days=overlap_days
        )

        df = None; err = None
        for attempt in range(1, max_attempts + 1):
            try:
                df = yf.download(
                    t, start=start, interval=interval,
                    auto_adjust=False, progress=False
                )
                break
            except Exception as e:
                err = e
                time.sleep(base_sleep * attempt)

        if df is None or df.empty:
            if err:
                print(f"[WARN] {t}: download error after {max_attempts} attempts: {err}")
            else:
                print(f"[INFO] {t}: no rows returned for start={start}")
            time.sleep(between_sleep)
            continue

        df = df.rename(columns=str.capitalize)
        df.index = pd.to_datetime(df.index); df.index.name = "Date"

        incremental_append(df, path, index_name="Date")
        print(f"[OK] {t}: window={start}..today rows={len(df):,} -> {path}")

        time.sleep(between_sleep)
