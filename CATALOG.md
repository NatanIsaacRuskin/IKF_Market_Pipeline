# IKF Market Pipeline – Code Catalog

_Branch: **main** | Files: **15**_

## Index

- `CATALOG.md`  (25 KB, modified 2025-11-10 13:07:18Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/CATALOG.md)
- `config/config.yaml`  (1 KB, modified 2025-11-10 13:07:18Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/config/config.yaml)
- `modules/__init__.py`  (0 B, modified 2025-11-10 13:07:18Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/__init__.py)
- `modules/equities.py`  (2 KB, modified 2025-11-10 13:07:18Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/equities.py)
- `modules/features.py`  (6 KB, modified 2025-11-10 13:07:18Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/features.py)
- `modules/futures.py`  (929 B, modified 2025-11-10 13:07:18Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/futures.py)
- `modules/options.py`  (1 KB, modified 2025-11-10 13:07:18Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/options.py)
- `modules/rates.py`  (776 B, modified 2025-11-10 13:07:18Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/rates.py)
- `quick_check.py`  (714 B, modified 2025-11-10 13:07:18Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/quick_check.py)
- `Readme.md`  (229 B, modified 2025-11-10 13:07:18Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/Readme.md)
- `requirements.txt`  (135 B, modified 2025-11-10 13:07:18Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/requirements.txt)
- `run_pipeline.py`  (2 KB, modified 2025-11-10 13:07:18Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/run_pipeline.py)
- `utils/__init__.py`  (0 B, modified 2025-11-10 13:07:18Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/utils/__init__.py)
- `utils/build_catalog.py`  (2 KB, modified 2025-11-10 13:07:18Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/utils/build_catalog.py)
- `utils/helpers.py`  (1 KB, modified 2025-11-10 13:07:18Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/utils/helpers.py)

---

## File Previews


### `CATALOG.md`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/CATALOG.md)

```md
# IKF Market Pipeline – Code Catalog

_Branch: **main** | Files: **15**_

## Index

- `CATALOG.md`  (25 KB, modified 2025-11-10 12:54:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/CATALOG.md)
- `config/config.yaml`  (1 KB, modified 2025-11-10 12:54:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/config/config.yaml)
- `modules/__init__.py`  (0 B, modified 2025-11-10 12:54:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/__init__.py)
- `modules/equities.py`  (2 KB, modified 2025-11-10 12:54:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/equities.py)
- `modules/features.py`  (7 KB, modified 2025-11-10 12:54:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/features.py)
- `modules/futures.py`  (862 B, modified 2025-11-10 12:54:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/futures.py)
- `modules/options.py`  (1 KB, modified 2025-11-10 12:54:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/options.py)
- `modules/rates.py`  (709 B, modified 2025-11-10 12:54:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/rates.py)
- `quick_check.py`  (714 B, modified 2025-11-10 12:54:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/quick_check.py)
- `Readme.md`  (229 B, modified 2025-11-10 12:54:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/Readme.md)
- `requirements.txt`  (94 B, modified 2025-11-10 12:54:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/requirements.txt)
- `run_pipeline.py`  (2 KB, modified 2025-11-10 12:54:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/run_pipeline.py)
- `utils/__init__.py`  (0 B, modified 2025-11-10 12:54:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/utils/__init__.py)
- `utils/build_catalog.py`  (2 KB, modified 2025-11-10 12:54:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/utils/build_catalog.py)
- `utils/helpers.py`  (1 KB, modified 2025-11-10 12:54:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/utils/helpers.py)

---

## File Previews


### `CATALOG.md`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/CATALOG.md)

```md
# IKF Market Pipeline – Code Catalog

_Branch: **main** | Files: **15**_

## Index

- `CATALOG.md`  (21 KB, modified 2025-11-09 14:34:28Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/CATALOG.md)
- `config/config.yaml`  (1 KB, modified 2025-11-09 13:37:06Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/config/config.yaml)
- `modules/__init__.py`  (0 B, modified 2025-11-08 17:53:54Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/__init__.py)
- `modules/equities.py`  (2 KB, modified 2025-11-08 22:10:48Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/equities.py)
- `modules/features.py`  (7 KB, modified 2025-11-08 22:10:38Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/features.py)
- `modules/futures.py`  (862 B, modified 2025-11-08 17:58:04Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/futures.py)
- `modules/options.py`  (1 KB, modified 2025-11-08 17:56:51Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/options.py)
...
[truncated]
```

### `config/config.yaml`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/config/config.yaml)

```yaml
storage:
  format: parquet            # how we save files
  root: "data/raw"           # where raw market files go
  timezone: "US/Eastern"

defaults:
  history_start: "2010-01-01"

equities:
  enabled: true
  universe: ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM","XOM"]
  price_interval: "1d"

  # pick ONE mode:
  # - full         : fetch from history_start → today (rebuilds files)
  # - incremental  : append from last saved date → today (typical daily use)
  # - recent       : pull a rolling window (lookback_days) and merge
  mode: "incremental"        # "full" | "incremental" | "recent"
  lookback_days: 365         # only used when mode = "recent"

options:
  enabled: true
  underlying: ["AAPL","MSFT","SPY"]
  expires: "nearest_3"       # nearest N expiries
  chains: ["calls","puts"]

futures:
  enabled: true
  tickers: ["ES=F","NQ=F","CL=F","GC=F","ZN=F"]
  price_interval: "1d"

rates:
  enabled: true
  fred_series:
    - "SOFR"
    - "DGS2"
    - "DGS5"
    - "DGS10"
    - "DGS30"
  start: "2010-01-01"

features:
  equities:
    enabled: true
    win_ret: 20
    win_vol: 60
    win_sharpe: 60
    sma_windows: [20, 50]
    ema_windows: [20, 50]
    rsi_period: 14
    plots: true
    processed_path: "data/processed"
    plots_path: "output/plots"

```

### `modules/__init__.py`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/__init__.py)

```py

```

### `modules/equities.py`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/equities.py)

```py
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

```

### `modules/features.py`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/features.py)

```py
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from utils.helpers import ensure_dir

RAW_EQUITIES_DIR = Path("data/raw/equities")

# ---------- helpers ----------
def _safe_to_datetime_idx(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

def _to_series(df: pd.DataFrame, col: str) -> pd.Series:
    canon = {c: str(c).strip() for c in df.columns}
    df = df.rename(columns=canon)
    if col not in df.columns:
        alt = {c.lower(): c for c in df.columns}
        if col.lower() in alt:
            col = alt[col.lower()]
    s = df.get(col, None)
    if s is None:
        return pd.Series(dtype="float64", index=df.index)
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    if not isinstance(s, pd.Series):
        s = pd.Series(s, index=df.index)
    s = pd.to_numeric(s, errors="coerce"); s.name = col
    return s

def _compute_one_ticker_features(df: pd.DataFrame, ticker: str, cfg: dict) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    # normalize index & columns
    df = _safe_to_datetime_idx(df)
    if isinstance(df.columns, pd.MultiIndex):               # flatten if needed
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip() for c in df.columns]

    # config
    WIN_RET     = int(cfg.get("win_ret", 20))
    WIN_VOL     = int(cfg.get("win_vol", 60))
    WIN_SHARPE  = int(cfg.get("win_sharpe", 60))
    SMA_WINDOWS = list(cfg.get("sma_windows", [20, 50]))
    EMA_WINDOWS = list(cfg.get("ema_windows", [20, 50]))
    RSI_PERIOD  = int(cfg.get("rsi_period", 14))

    # robust close selection
    candidates = ["Adj Close", "Adj close", "close", "Close"]
    px = None
    for c in candidates:
        if c in df.columns:
            s = df[c]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            px = pd.to_numeric(s, errors="coerce")
            break
    if px is None or px.size == 0:
        lower = {str(c).lower(): c for c in df.columns}
        for want in ["adj close", "close"]:
            if want in lower:
                s = df[lower[want]]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                px = pd.to_numeric(s, errors="coerce")
                break
    if px is None or px.size == 0:
        return pd.DataFrame()

    px = px.reindex(df.index)  # align

    # features
    ret = np.log(px).diff()
    vol = ret.rolling(WIN_VOL).std() * np.sqrt(252)
    sharpe = (ret.rolling(WIN_SHARPE).mean() * 252) / vol
    mom = px.pct_change(WIN_RET)

    feats = pd.DataFrame(index=df.index)
    feats["close"] = px
    feats["ret_1d_log"] = ret
    feats["roll_vol_ann"] = vol
    feats["roll_sharpe"] = sharpe
    feats[f"mom_{WIN_RET}d"] = mom

    for w in SMA_WINDOWS:
        feats[f"sma_{w}"] = SMAIndicator(px, window=w).sma_indicator()
    for w in EMA_WINDOWS:
        feats[f"ema_{w}"] = EMAIndicator(px, window=w).ema_indicator()
    feats[f"rsi_{RSI_PERIOD}"] = RSIIndicator(px, window=RSI_PERIOD).rsi()

    macd = MACD(px)
    feats["macd"] = macd.macd()
    feats["macd_signal"] = macd.macd_signal()
    feats["macd_hist"] = macd.macd_diff()

    feats["ticker"] = ticker
    return feats.dropna(how="all")

# ---------- public entry point used by run_pipeline.py ----------
def run_equity_features(cfg_eq_feat: dict):
    """
    Orchestrates equity feature engineering + basic plots.
    Expects keys in config:
      processed_path, plots_path, plots (bool)
    """
    processed_path = Path(cfg_eq_feat.get("processed_path", "data/processed"))
    plots_path     = Path(cfg_eq_feat.get("plots_path", "output/plots"))
    ensure_dir(processed_path)
    ensure_dir(plots_path)

    feats_list = []
    files = sorted(RAW_EQUITIES_DIR.glob("*.parqu
...
[truncated]
```

### `modules/futures.py`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/futures.py)

```py
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

```

### `modules/options.py`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/options.py)

```py
import pandas as pd
import yfinance as yf
from utils.helpers import parquet_path, incremental_append

def _nearest_expiries(tkr: yf.Ticker, n: int = 3) -> list[str]:
    exps = tkr.options or []
    return exps[:n]

def update_options(cfg: dict, storage_root: str):
    if not cfg.get("enabled", False):
        return
    underlyings = cfg.get("underlying", [])
    n = int(cfg.get("expiries","nearest_3").split("_")[-1])
    chains = cfg.get("chains", ["calls","puts"])

    for u in underlyings:
        tk = yf.Ticker(u)
        expiries = _nearest_expiries(tk, n)
        for exp in expiries:
            chain = tk.option_chain(exp)
            for side in chains:
                df: pd.DataFrame = getattr(chain, side, None)
                if df is None or df.empty:
                    continue
                df["underlying"] = u
                df["expiry"] = exp
                df.set_index(["expiry","contractSymbol"], inplace=True)
                path = parquet_path(storage_root, "options", u, f"{side}_{exp}")
                incremental_append(df, path)

```

### `modules/rates.py`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/rates.py)

```py
import pandas as pd
from pandas_datareader import data as pdr
from utils.helpers import parquet_path, incremental_append, load_parquet

def update_rates(cfg: dict, storage_root: str):
    if not cfg.get("enabled", False):
        return
    series = cfg.get("fred_series", [])
    start = cfg.get("start","2010-01-01")
    for s in series:
        path = parquet_path(storage_root, "rates", s)
        exist = load_parquet(path)
        sdate = (exist.index.max() + pd.Timedelta(days=1)).date().isoformat() if not exist.empty else start
        df = pdr.DataReader(s, "fred", sdate)   # index=Date, col=series
        if df.empty:
            continue
        df.index = pd.to_datetime(df.index); df.index.name = "Date"
        incremental_append(df, path, index_name="Date")

```

### `quick_check.py`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/quick_check.py)

```py
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
    
```

### `Readme.md`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/Readme.md)

```md
# IKF Market Pipeline

Universal market data pipeline (equities, options, futures, rates).

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

```

### `requirements.txt`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/requirements.txt)

```txt
# Core
pandas
numpy
pyyaml
yfinance
ta
matplotlib
seaborn

# Parquet + data sources
pyarrow
pandas-datareader

# Optional
rich
jupyter

```

### `run_pipeline.py`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/run_pipeline.py)

```py
import argparse, yaml
from modules.equities import update_equities
from modules.options  import update_options
from modules.futures  import update_futures
from modules.rates    import update_rates
from modules.features import run_equity_features

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser(description="IKF Universal Market Pipeline")

    # what to run
    ap.add_argument("--asset",
                    choices=["all","equities","options","futures","rates"],
                    default="all",
                    help="which asset updater(s) to run")
    ap.add_argument("--features",
                    action="store_true",
                    help="run equity feature engineering after updates")

    # config path
    ap.add_argument("--config",
                    default="config/config.yaml",
                    help="path to YAML config")

    # NEW: one-off backfill controls for equities
    ap.add_argument("--full",
                    action="store_true",
                    help="force full backfill for equities (start from history_start)")
    ap.add_argument("--recent",
                    type=int, default=None,
                    help="force recent backfill for equities using N lookback days (ignores on-disk)")

    args = ap.parse_args()

    cfg  = load_config(args.config)
    root = cfg["storage"]["root"]

    # ---- equities updater (merge defaults + equities block) ----
    eq_cfg = (cfg.get("defaults", {}) | cfg.get("equities", {})).copy()

    # apply CLI overrides (these override config.yaml for this run only)
    if args.full:
        eq_cfg["mode"] = "full"
    if args.recent is not None:
        eq_cfg["mode"] = "recent"
        eq_cfg["lookback_days"] = args.recent

    # ---- run updaters ----
    if args.asset in ("all", "equities"):
        update_equities(eq_cfg, root)
    if args.asset in ("all", "options"):
        update_options(cfg.get("options", {}), root)
    if args.asset in ("all", "futures"):
        update_futures(cfg.get("futures", {}), root)
    if args.asset in ("all", "rates"):
        update_rates(cfg.get("rates", {}), root)

    # ---- features (equities) ----
    if args.features:
        eq_feat_cfg = cfg.get("features", {}).get("equities", {})
        run_equity_features(eq_feat_cfg)
        print("[OK] Equity feature engineering complete.")

if __name__ == "__main__":
    main()

```

### `utils/__init__.py`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/utils/__init__.py)

```py

```

### `utils/build_catalog.py`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/utils/build_catalog.py)

```py
from __future__ import annotations
import os, pathlib, sys, datetime

REPO_USER = "NatanIsaacRuskin"
REPO_NAME = "IKF_Market_Pipeline"
BRANCH = os.getenv("CATALOG_BRANCH", "main")
RAW_BASE = f"https://raw.githubusercontent.com/{REPO_USER}/{REPO_NAME}/{BRANCH}/"

IGNORE_DIRS = {".git", ".github", "__pycache__", ".venv", "backups", "data/cache"}
ALLOWED_EXT = {".py", ".ipynb", ".md", ".yaml", ".yml", ".toml", ".json", ".csv", ".txt"}

def should_include(rel: pathlib.Path) -> bool:
    if any(part in IGNORE_DIRS for part in rel.parts):
        return False
    if rel.name.startswith("."):  # hide dotfiles except root configs if you want
        return False
    if rel.suffix.lower() in ALLOWED_EXT:
        return True
    return False

def human_size(n: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024: return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.0f} PB"

def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    files = []
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if p.is_file() and should_include(rel):
            files.append(p)
    files.sort(key=lambda x: str(x).lower())

    lines = []
    lines.append("# IKF Market Pipeline – Code Catalog\n")
    lines.append(f"_Branch: **{BRANCH}** | Files: **{len(files)}**_\n")
    lines.append("## Index\n")
    for p in files:
        rel = p.relative_to(root).as_posix()
        stat = p.stat()
        mtime = datetime.datetime.utcfromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%SZ")
        size = human_size(stat.st_size)
        raw = RAW_BASE + rel
        lines.append(f"- `{rel}`  ({size}, modified {mtime} UTC) → [raw]({raw})")

    lines.append("\n---\n")
    lines.append("## File Previews\n")
    for p in files:
        rel = p.relative_to(root).as_posix()
        raw = RAW_BASE + rel
        ext = p.suffix.lower().lstrip(".")
        lang = {"yml":"yaml"}.get(ext, ext)  # nicer fencing
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            text = f"<<could not read: {e}>>"

        # Keep the catalog small but useful (preview only)
        snippet = text if len(text) <= 4000 else text[:4000] + "\n...\n[truncated]"

        lines.append(f"\n### `{rel}`  •  [raw]({raw})\n")
        lines.append(f"```{lang}\n{snippet}\n```")

    (root / "CATALOG.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote CATALOG.md")

if __name__ == "__main__":
    sys.exit(main())

```

### `utils/helpers.py`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/utils/helpers.py)

```py
import pandas as pd
from pathlib import Path

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
    """Append new data to an existing parquet file without duplicates."""
    df_old = load_parquet(path)
    if df_old.empty:
        df_all = df_new.copy()
    else:
        df_all = pd.concat([df_old, df_new]).sort_index()
        df_all = df_all[~df_all.index.duplicated(keep="last")]
    if index_name:
        df_all.index.name = index_name
    df_all.to_parquet(path)

```