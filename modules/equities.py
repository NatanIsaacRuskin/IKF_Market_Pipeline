from pathlib import Path
import pandas as pd
import yfinance as yf
from utils.helpers import parquet_path, incremental_append, load_parquet

def _start_date(existing: pd.DataFrame, default_start: str, mode: str, lookback_days: int | None) -> str:
    if mode == "full":
        return default_start
    if mode == "recent":
        # pull a rolling window regardless of what's on disk
        return (pd.Timestamp.today().normalize() - pd.Timedelta(days=lookback_days or 365)).date().isoformat()
    # incremental (default)
    if existing.empty:
        return default_start
    last = pd.to_datetime(existing.index.max())
    return (last + pd.Timedelta(days=1)).date().isoformat()

def update_equities(cfg: dict, storage_root: str):
    if not cfg.get("enabled", False):
        print("[INFO] equities disabled in config.")
        return

    tickers       = cfg["universe"]
    interval      = cfg.get("price_interval", "1d")
    default_start = cfg.get("history_start", "2010-01-01")
    mode          = cfg.get("mode", "incremental")
    lookback_days = cfg.get("lookback_days", 365)

    for t in tickers:
        path   = parquet_path(storage_root, "equities", t)
        exist  = load_parquet(path)
        start  = _start_date(exist, default_start, mode, lookback_days)

        df = yf.download(t, start=start, interval=interval,
                         auto_adjust=True, progress=False)
        if df.empty:
            print(f"[WARN] {t}: no new data")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).strip().title() for c in df.columns]

        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

        incremental_append(df, path, index_name="Date")
        print(f"[OK] {t}: mode={mode} saved -> {path}")
