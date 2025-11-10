from pathlib import Path
import pandas as pd

def ensure_dir(path: Path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)

def parquet_path(root: str, *parts: str) -> Path:
    """Build a .parquet file path under the given root."""
    dirp = Path(root, *parts[:-1])
    ensure_dir(dirp)
    stem = parts[-1]
    return dirp / f"{stem}.parquet"

def load_parquet(path: Path) -> pd.DataFrame:
    """Read parquet file if it exists, else return empty DataFrame."""
    return pd.read_parquet(path) if Path(path).exists() else pd.DataFrame()

def incremental_append(df_new: pd.DataFrame, path: Path, index_name: str = None):
    """
    Append new data to an existing parquet file without duplicates.
    Upsert semantics: concat -> sort -> drop duplicate index (keep last).
    """
    df_old = load_parquet(path)
    if df_old.empty:
        df_all = df_new.copy()
    else:
        df_all = pd.concat([df_old, df_new]).sort_index()
        df_all = df_all[~df_all.index.duplicated(keep="last")]
    if index_name:
        df_all.index.name = index_name
    df_all.to_parquet(path)

# ---------- NEW: smart fetch window ----------
def compute_fetch_start(existing_index, history_start="2010-01-01", overlap_days=5) -> str:
    """
    Decide a safe, small window to fetch:
      start = max(history_start, last_date - overlap_days), clamped to today.
    Returns ISO date string.
    """
    today = pd.Timestamp.today().normalize()
    if existing_index is None or len(existing_index) == 0:
        return pd.Timestamp(history_start).date().isoformat()
    last = pd.to_datetime(max(existing_index)).normalize()
    start = (last - pd.Timedelta(days=overlap_days)).normalize()
    start = max(pd.Timestamp(history_start), start)
    start = min(start, today)
    return start.date().isoformat()
