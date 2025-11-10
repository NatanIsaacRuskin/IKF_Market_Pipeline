# IKF Market Pipeline – Code Catalog

_Branch: **main** | Files: **15**_

## Index

- `CATALOG.md`  (25 KB, modified 2025-11-10 14:26:10Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/CATALOG.md)
- `config/config.yaml`  (2 KB, modified 2025-11-10 14:26:10Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/config/config.yaml)
- `modules/__init__.py`  (0 B, modified 2025-11-10 14:26:10Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/__init__.py)
- `modules/equities.py`  (4 KB, modified 2025-11-10 14:26:10Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/equities.py)
- `modules/features.py`  (10 KB, modified 2025-11-10 14:26:10Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/features.py)
- `modules/futures.py`  (2 KB, modified 2025-11-10 14:26:10Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/futures.py)
- `modules/options.py`  (1 KB, modified 2025-11-10 14:26:10Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/options.py)
- `modules/rates.py`  (1 KB, modified 2025-11-10 14:26:10Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/rates.py)
- `quick_check.py`  (714 B, modified 2025-11-10 14:26:10Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/quick_check.py)
- `Readme.md`  (229 B, modified 2025-11-10 14:26:10Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/Readme.md)
- `requirements.txt`  (135 B, modified 2025-11-10 14:26:10Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/requirements.txt)
- `run_pipeline.py`  (2 KB, modified 2025-11-10 14:26:10Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/run_pipeline.py)
- `utils/__init__.py`  (0 B, modified 2025-11-10 14:26:10Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/utils/__init__.py)
- `utils/build_catalog.py`  (2 KB, modified 2025-11-10 14:26:10Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/utils/build_catalog.py)
- `utils/helpers.py`  (2 KB, modified 2025-11-10 14:26:10Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/utils/helpers.py)

---

## File Previews


### `CATALOG.md`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/CATALOG.md)

```md
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
- `modules/options.py`  (1 KB, modified 2025-11-10 12:54:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/options.py
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
  universe: ["SPY","AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM","XOM"]
  price_interval: "1d"

  # pick ONE mode:
  # - full         : fetch from history_start → today (rebuilds files)
  # - incremental  : append small window near today (uses overlap_days)
  # - recent       : pull a rolling window (lookback_days) and merge
  mode: "incremental"          # "full" | "incremental" | "recent"
  lookback_days: 365           # only used when mode = "recent"

  # smart incremental window
  overlap_days: 5              # re-fetch last N days each run to heal revisions

  # gentle rate limiting / retries
  max_attempts: 3
  base_sleep: 1.5
  between_sleep: 0.35

options:
  enabled: true
  underlying: ["AAPL","MSFT","SPY"]
  expires: "nearest_3"         # nearest N expiries
  chains: ["calls","puts"]

futures:
  enabled: true
  tickers: ["ES=F","NQ=F","CL=F","GC=F","ZN=F"]
  price_interval: "1d"
  history_start: "2010-01-01"
  overlap_days: 5
  # (optional) reuse gentle knobs; uncomment if you want:
  # max_attempts: 3
  # base_sleep: 1.5
  # between_sleep: 0.35

rates:
  enabled: true
  fred_series:
    - "SOFR"
    - "DGS2"
    - "DGS5"
    - "DGS10"
    - "DGS30"
  start: "2010-01-01"
  overlap_days: 5

features:
  equities:
    enabled: true
    benchmark: "SPY"           # for beta/alpha
    win_ret: 20
    win_vol: 60
    win_sharpe: 60
    sma_windows: [20, 50]
    ema_windows: [20, 50]
    rsi_period: 14
    boll_window: 20
    atr_window: 14
    skew_window: 60
    kurt_window: 60
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
def _rolling_beta_alpha(ret_stock: pd.Series, ret_bench: pd.Series, win: int = 60):
    both = pd.concat([ret_stock, ret_bench], axis=1).dropna()
    both.columns = ["s", "m"]
    cov = both["s"].rolling(win).cov(both["m"])
    var = both["m"].rolling(win).var()
    beta = cov / var
    alpha = (both["s"].rolling(win).mean() - beta * both["m"].rolling(win).mean()) * 252
    return beta, alpha

def _drawdown(px: pd.Series):
    roll_max = px.cummax()
    dd = px / roll_max - 1.0
    mdd = dd.min()
    return dd, mdd

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14):
    hl = (high - low).abs()
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def _bollinger(px: pd.Series, window: int = 20, k: float = 2.0):
    ma = px.rolling(window).mean()
    sd = px.rolling(window).std()
    upper = ma + k * sd
    lower = ma - k * sd
    pct_band = (px - lower) / (upper - lower)  # 0..1 position in band
    return ma, upper, lower, pct_band

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
    """Build per-ticker feature frame. RETURNS empty df if no usable price series."""
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
    BOLL_W      = int(cfg.get("boll_window", 20))
    ATR_W       = int(cfg.get("atr_window", 14))
    SKEW_W      = int(cfg.get("skew_window", 60))
    KURT_W      = int(cfg.get("kurt_window", 60))

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

    # base features
    ret
...
[truncated]
```

### `modules/futures.py`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/futures.py)

```py
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

```