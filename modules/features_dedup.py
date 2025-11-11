# modules/features_dedup.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

def _ensure_date_column(df: pd.DataFrame, target: str = "date") -> pd.DataFrame:
    """
    Make sure a 'date' column exists.
    - If index is DatetimeIndex, bring it out as a column.
    - Else, search for a datetime-like column and rename it to 'date'.
    """
    if target in df.columns:
        # normalize to pandas datetime (no timezone)
        df[target] = pd.to_datetime(df[target], errors="coerce", utc=True).dt.tz_convert(None)
        return df

    # 1) DatetimeIndex → column
    if isinstance(df.index, pd.DatetimeIndex):
        name = df.index.name if df.index.name else "index"
        df = df.reset_index().rename(columns={name: target})
        df[target] = pd.to_datetime(df[target], errors="coerce", utc=True).dt.tz_convert(None)
        return df

    # 2) Look for obvious names
    candidates = [c for c in df.columns if str(c).lower() in {"date", "datetime", "timestamp"}]
    # 3) Or any column that can be parsed as datetime
    if not candidates:
        for c in df.columns:
            try:
                tmp = pd.to_datetime(df[c], errors="coerce")
                if tmp.notna().mean() > 0.9:  # mostly valid datetimes
                    candidates.append(c)
                    break
            except Exception:
                continue

    if candidates:
        pick = candidates[0]
        if pick != target:
            df = df.rename(columns={pick: target})
        df[target] = pd.to_datetime(df[target], errors="coerce", utc=True).dt.tz_convert(None)
        return df

    raise KeyError(
        "Could not find a date column. Looked for index=DatetimeIndex or columns like "
        "'date'/'datetime'/'timestamp'. Available columns: "
        f"{list(df.columns)[:12]}..."
    )

def _ensure_id_column(df: pd.DataFrame, target: str = "ticker") -> pd.DataFrame:
    """
    Make sure a 'ticker' column exists. Try common alternatives, then normalize.
    """
    if target not in df.columns:
        for alt in ("Ticker", "ticker", "symbol", "Symbol"):
            if alt in df.columns:
                df = df.rename(columns={alt: target})
                break
    if target not in df.columns:
        raise KeyError(f"Missing identifier column '{target}'. Columns: {list(df.columns)[:12]}...")

    df[target] = df[target].astype(str).str.upper()
    return df

def finalize_equity_features_file(
    processed_dir: str = "data/processed",
    fname: str = "equity_features.parquet",
    *,
    date_col: str = "date",
    id_col: str = "ticker",
) -> str:
    """
    Enforce unique primary key (date, ticker) on the engineered features parquet.
    Run this right after feature engineering (and again before ranking as a guard).
    """
    path = Path(processed_dir) / fname
    if not path.exists():
        print(f"[INFO] finalize_equity_features_file: {path} not found, skipping.")
        return str(path)

    df = pd.read_parquet(path)
    n0 = len(df)

    # Ensure keys exist and are normalized
    df = _ensure_date_column(df, target=date_col)
    df = _ensure_id_column(df, target=id_col)

    # Add a deterministic preference column if none exists, so 'keep=last' is meaningful
    prefer_cols = []
    if "ingest_ts" in df.columns: prefer_cols.append("ingest_ts")
    if "calc_ts"   in df.columns: prefer_cols.append("calc_ts")
    if not prefer_cols:
        df["ingest_ts"] = pd.Timestamp.utcnow()
        prefer_cols = ["ingest_ts"]

    # Sort so the last record per (date, ticker) is the one we keep
    sort_cols = [date_col, id_col] + prefer_cols
    df = df.sort_values(sort_cols).drop_duplicates(subset=[date_col, id_col], keep="last")

    # Sanity check
    dup_ct = df.duplicated(subset=[date_col, id_col]).sum()
    assert dup_ct == 0, f"Dedup failed: {dup_ct} duplicate (date,ticker) rows remain."

    # Write back
    df.to_parquet(path)
    n1 = len(df)
    print(f"[OK] Features deduplicated → {path} (rows: {n0} → {n1}, removed {n0 - n1})")
    return str(path)
