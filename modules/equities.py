from pathlib import Path
import time
import pandas as pd
import yfinance as yf
from utils.helpers import (
    parquet_path, incremental_append, load_parquet, compute_fetch_start
)

def _mode_start(mode: str,
                exist_df: pd.DataFrame | None,
                history_start: str,
                lookback_days: int,
                overlap_days: int) -> str:
    """
    Decide start date based on mode:
      - full:        history_start
      - recent:      today - lookback_days
      - incremental: compute_fetch_start(existing_index, history_start, overlap_days)
    Always returns an ISO date string.
    """
    mode = (mode or "incremental").lower()
    today = pd.Timestamp.today().normalize()

    if mode == "full":
        return pd.Timestamp(history_start).date().isoformat()

    if mode == "recent":
        start = (today - pd.Timedelta(days=lookback_days)).normalize()
        return start.date().isoformat()

    # incremental (default): small overlap window ending today
    exist_idx = None if exist_df is None or exist_df.empty else exist_df.index
    return compute_fetch_start(exist_idx, history_start=history_start, overlap_days=overlap_days)

def update_equities(cfg: dict, storage_root: str):
    if not cfg.get("enabled", False):
        print("[INFO] equities disabled in config.")
        return

    tickers        = cfg["universe"]
    interval       = cfg.get("price_interval", "1d")
    history_start  = cfg.get("history_start", "2010-01-01")
    mode           = cfg.get("mode", "incremental")
    lookback_days  = int(cfg.get("lookback_days", 365))
    overlap_days   = int(cfg.get("overlap_days", 5))

    # gentle retry knobs
    max_attempts   = int(cfg.get("max_attempts", 3))
    base_sleep     = float(cfg.get("base_sleep", 1.5))
    between_sleep  = float(cfg.get("between_sleep", 0.35))

    print(f"[INFO] Equities mode={mode} interval={interval} "
          f"(history_start={history_start}, lookback_days={lookback_days}, overlap_days={overlap_days})")
    print(f"[INFO] Rate limit: attempts={max_attempts}, base_sleep={base_sleep}s, between={between_sleep}s")

    for t in tickers:
        path  = parquet_path(storage_root, "equities", t)
        exist = load_parquet(path)

        start = _mode_start(
            mode=mode,
            exist_df=exist,
            history_start=history_start,
            lookback_days=lookback_days,
            overlap_days=overlap_days
        )

        df = None
        err = None
        for attempt in range(1, max_attempts + 1):
            try:
                df = yf.download(
                    t, start=start, interval=interval,
                    auto_adjust=True, progress=False
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

        # tidy columns/index
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).strip().title() for c in df.columns]
        df.index = pd.to_datetime(df.index); df.index.name = "Date"

        # upsert (concat/sort/dedup) handled inside incremental_append
        incremental_append(df, path, index_name="Date")
        print(f"[OK] {t}: mode={mode} window={start}..today rows={len(df):,} -> {path}")

        time.sleep(between_sleep)
