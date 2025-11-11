# IKF Market Pipeline – Code Catalog

_Branch: **main** | Files: **21**_

## Index

- `CATALOG.md`  (49 KB, modified 2025-11-11 20:21:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/CATALOG.md)
- `config/composites_specs/IKF_AI_Megacap.csv`  (75 B, modified 2025-11-11 20:21:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/config/composites_specs/IKF_AI_Megacap.csv)
- `config/composites_specs/IKF_EnergyPulse.csv`  (59 B, modified 2025-11-11 20:21:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/config/composites_specs/IKF_EnergyPulse.csv)
- `config/config.yaml`  (2 KB, modified 2025-11-11 20:21:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/config/config.yaml)
- `modules/__init__.py`  (0 B, modified 2025-11-11 20:21:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/__init__.py)
- `modules/composites.py`  (7 KB, modified 2025-11-11 20:21:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/composites.py)
- `modules/equities.py`  (4 KB, modified 2025-11-11 20:21:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/equities.py)
- `modules/features.py`  (10 KB, modified 2025-11-11 20:21:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/features.py)
- `modules/features_dedup.py`  (4 KB, modified 2025-11-11 20:21:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/features_dedup.py)
- `modules/futures.py`  (2 KB, modified 2025-11-11 20:21:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/futures.py)
- `modules/options.py`  (1 KB, modified 2025-11-11 20:21:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/options.py)
- `modules/ranking.py`  (9 KB, modified 2025-11-11 20:21:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/ranking.py)
- `modules/rates.py`  (1 KB, modified 2025-11-11 20:21:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/rates.py)
- `modules/report_equities.py`  (2 KB, modified 2025-11-11 20:21:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/report_equities.py)
- `quick_check.py`  (714 B, modified 2025-11-11 20:21:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/quick_check.py)
- `Readme.md`  (2 KB, modified 2025-11-11 20:21:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/Readme.md)
- `requirements.txt`  (144 B, modified 2025-11-11 20:21:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/requirements.txt)
- `run_pipeline.py`  (6 KB, modified 2025-11-11 20:21:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/run_pipeline.py)
- `utils/__init__.py`  (0 B, modified 2025-11-11 20:21:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/utils/__init__.py)
- `utils/build_catalog.py`  (2 KB, modified 2025-11-11 20:21:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/utils/build_catalog.py)
- `utils/helpers.py`  (2 KB, modified 2025-11-11 20:21:19Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/utils/helpers.py)

---

## File Previews


### `CATALOG.md`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/CATALOG.md)

```md
# IKF Market Pipeline – Code Catalog

_Branch: **main** | Files: **21**_

## Index

- `CATALOG.md`  (42 KB, modified 2025-11-11 20:09:27Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/CATALOG.md)
- `config/composites_specs/IKF_AI_Megacap.csv`  (75 B, modified 2025-11-11 20:09:27Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/config/composites_specs/IKF_AI_Megacap.csv)
- `config/composites_specs/IKF_EnergyPulse.csv`  (59 B, modified 2025-11-11 20:09:27Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/config/composites_specs/IKF_EnergyPulse.csv)
- `config/config.yaml`  (2 KB, modified 2025-11-11 20:09:27Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/config/config.yaml)
- `modules/__init__.py`  (0 B, modified 2025-11-11 20:09:27Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/__init__.py)
- `modules/composites.py`  (7 KB, modified 2025-11-11 20:09:27Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/composites.py)
- `modules/equities.py`  (4 KB, modified 2025-11-11 20:09:27Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/equities.py)
- `modules/features.py`  (10 KB, modified 2025-11-11 20:09:27Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/features.py)
- `modules/features_dedup.py`  (4 KB, modified 2025-11-11 20:09:27Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/features_dedup.py)
- `modules/futures.py`  (2 KB, modified 2025-11-11 20:09:27Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/futures.py)
- `modules/options.py`  (1 KB, modified 2025-11-11 20:09:27Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/options.py)
- `modules/ranking.py`  (9 KB, modified 2025-11-11 20:09:27Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/ranking.py)
- `modules/rates.py`  (1 KB, modified 2025-11-11 20:09:27Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/rates.py)
- `modules/report_equities.py`  (2 KB, modified 2025-11-11 20:09:27Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/report_equities.py)
- `quick_check.py`  (714 B, modified 2025-11-11 20:09:27Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/quick_check.py)
- `Readme.md`  (2 KB, modified 2025-11-11 20:09:27Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/Readme.md)
- `requirements.txt`  (144 B, modified 2025-11-11 20:09:27Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/requirements.txt)
- `run_pipeline.py`  (6 KB, modified 2025-11-11 20:09:27Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/run_pipeline.py)
- `utils/__init__.py`  (0 B, modified 2025-11-11 20:09:27Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/utils/__init__.py)
- `utils/build_catalog.py`  (2 KB, modified 2025-11-11 20:09:27Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/utils/build_catalog.py)
- `utils/helpers.py`  (2 KB, modified 2025-11-11 20:09:27Z UTC) → [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/utils/helpers.py)

---

## File Previews


### `CATALOG.md`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/CATALOG.md)

```md
# IKF Market Pipeline – Code Catalog

_Branch: **main** | Files: **18**_

## Index

- `CATALOG.md`  (30 KB, mod
...
[truncated]
```

### `config/composites_specs/IKF_AI_Megacap.csv`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/config/composites_specs/IKF_AI_Megacap.csv)

```csv
ticker,weight
NVDA,0.22
MSFT,0.20
AAPL,0.16
GOOGL,0.16
META,0.13
AMZN,0.13

```

### `config/composites_specs/IKF_EnergyPulse.csv`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/config/composites_specs/IKF_EnergyPulse.csv)

```csv
ticker,weight
XOM,0.25
CVX,0.25
SLB,0.20
COP,0.15
EOG,0.15

```

### `config/config.yaml`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/config/config.yaml)

```yaml
# FILE: config/config.yaml
# IKF MARKET PIPELINE — CONFIG

storage:
  root: "data/raw"
  format: "parquet"
  timezone: "US/Eastern"

defaults:
  history_start: "2010-01-01"

equities:
  enabled: true
  universe: ["SPY"]         # runtime auto-adds composite members + benchmarks
  price_interval: "1d"
  mode: "incremental"       # overridden by --recent/--full
  lookback_days: 365
  overlap_days: 5
  max_attempts: 3
  base_sleep: 1.5
  between_sleep: 0.35

options:
  enabled: true
  underlying: ["SPY"]
  expires: "nearest_3"
  chains: ["calls","puts"]

futures:
  enabled: true
  tickers: ["ES=F"]
  price_interval: "1d"
  history_start: "2010-01-01"
  overlap_days: 5

rates:
  enabled: true
  fred_series: ["SOFR","DGS2","DGS5","DGS10","DGS30"]
  start: "2010-01-01"
  overlap_days: 5

features:
  equities:
    enabled: true
    benchmark: "SPY"
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

analysis:
  benchmark: "SPY"
  topk: 25
  deciles: 10

# IKF COMPOSITES (CSV > inline > equal-weight)
composites:
  - name: "IKF_AI_Megacap"
    tickers: ["NVDA","MSFT","AAPL","GOOGL","META","AMZN"]
    weights_csv: "config/composites_specs/IKF_AI_Megacap.csv"   # ← UNCOMMENT / add this
    # weights: [0.22, 0.20, 0.16, 0.16, 0.13, 0.13]             # ← optional to leave, ignored when CSV present
    benchmark: "SPY"
    rebalance: "none"
    start: "2010-01-01"

  - name: "IKF_EnergyPulse"
    tickers: ["XOM","CVX","SLB","COP","EOG"]
    weights_csv: "config/composites_specs/IKF_EnergyPulse.csv"  # ← UNCOMMENT / add this
    # weights: [0.25, 0.25, 0.20, 0.15, 0.15]                  # ← optional to leave, ignored when CSV present
    benchmark: "SPY"
    rebalance: "monthly"
    start: "2010-01-01"


```

### `modules/__init__.py`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/__init__.py)

```py

```

### `modules/composites.py`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/composites.py)

```py
# FILE: modules/composites.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
import numpy as np

# expects utils.helpers to provide these (already in your repo)
from utils.helpers import parquet_path, load_parquet

"""
Composites builder
- Builds composite price time series from RAW equities parquet (no feature dependency)
- IKF weights via inline `weights:` or `weights_csv:` (CSV wins if both supplied)
- Rebalancing: "none" | "monthly" | "quarterly" (default "none")
- Writes data/processed/composites/<name>_prices.parquet
- Optionally writes output/snapshots/<name>_snapshot.csv if rank_today provided
CSV format for weights:
    ticker,weight
    NVDA,0.22
    MSFT,0.20
"""

@dataclass
class CompositeSpec:
    name: str
    tickers: List[str]
    weights: Optional[List[float]] = None
    weights_csv: Optional[str] = None
    benchmark: Optional[str] = None
    rebalance: str = "none"
    start: Optional[str] = None

def _ensure_cols(df: pd.DataFrame, cols: List[str]):
    miss = [c for c in cols if c not in df.columns]
    if miss: raise KeyError(f"Missing columns: {miss}")

def _read_prices_from_raw(storage_root: str, tickers: List[str]) -> pd.DataFrame:
    out = []
    for t in tickers:
        p = parquet_path(storage_root, "equities", t)
        df = load_parquet(p)
        if df.empty: continue
        cols = {str(c).lower(): c for c in df.columns}
        close_col = None
        for cand in ("adj close", "close"):
            if cand in cols: close_col = cols[cand]; break
        if close_col is None: continue
        s = pd.to_numeric(df[close_col].squeeze(), errors="coerce")
        s.index = pd.to_datetime(s.index); s.name = "close"
        tmp = s.reset_index().rename(columns={s.index.name or "index": "date"})
        tmp["ticker"] = t.upper()
        out.append(tmp[["date","ticker","close"]])
    if not out:
        return pd.DataFrame(columns=["date","ticker","close"])
    dfall = pd.concat(out, ignore_index=True)
    return dfall.sort_values(["ticker","date"]).dropna(subset=["close"])

def _normalize_weights(tickers: List[str], weights: Optional[List[float]], weights_csv: Optional[str]) -> pd.DataFrame:
    """Priority: weights_csv > inline weights > equal-weight fallback."""
    if weights_csv:
        wdf = pd.read_csv(weights_csv)
        _ensure_cols(wdf, ["ticker","weight"])
        wdf["ticker"] = wdf["ticker"].astype(str).str.upper()
        wdf = wdf[wdf["ticker"].isin([t.upper() for t in tickers])].copy()
        if wdf.empty: raise ValueError(f"weights_csv={weights_csv} has no rows for declared tickers.")
        w = wdf["weight"].astype(float).values
        s = np.nansum(w)
        if not np.isfinite(s) or s <= 0: raise ValueError("Invalid weights in CSV; sum must be > 0.")
        wdf["weight"] = wdf["weight"] / s
        return wdf[["ticker","weight"]]
    if weights is not None:
        if len(weights) != len(tickers): raise ValueError("Inline 'weights' length must match 'tickers'.")
        w = np.asarray(weights, dtype=float)
        if np.any(~np.isfinite(w)) or w.sum() <= 0: raise ValueError("Inline weights must be finite and sum>0.")
        w = w / w.sum()
        return pd.DataFrame({"ticker":[t.upper() for t in tickers], "weight": w})
    n = len(tickers)
    if n == 0: raise ValueError("No tickers provided.")
    eq = np.repeat(1.0/n, n)
    return pd.DataFrame({"ticker":[t.upper() for t in tickers], "weight": eq})

def _period_starts(dates: pd.DatetimeIndex, freq: str) -> pd.Series:
    if freq == "M":
        lab = pd.Series(dates.to_period("M").astype(str), index=dates)
    elif freq == "Q":
        lab = pd.Series(dates.to_period("Q").astype(str), index=dates)
    else:
        return pd.Series(False, index=dates)
    return lab.ne(lab.shift(1)).fillna(True)

def _build_composite_series(prices: pd.DataFrame, spec: CompositeSpec) -> pd.DataFrame:
    _ensure_cols(prices, ["date","
...
[truncated]
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

### `modules/features_dedup.py`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/features_dedup.py)

```py
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
    print(f"[OK] Features ded
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

### `modules/ranking.py`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/ranking.py)

```py
# modules/ranking.py
from __future__ import annotations
import re
import numpy as np
import pandas as pd

EPS = 1e-12


# ------------------------ Robust transforms ------------------------ #
def _mad(x: np.ndarray) -> float:
    med = np.nanmedian(x)
    return np.nanmedian(np.abs(x - med))


def _safe_robust_z(s: pd.Series) -> pd.Series:
    """Robust z; returns NaNs if too few valid points or constant series."""
    s = pd.to_numeric(s, errors="coerce")
    n = s.notna().sum()
    if n < 3:
        return pd.Series(np.nan, index=s.index)
    med = np.nanmedian(s.values)
    mad = _mad(s.values)
    if not np.isfinite(med) or mad == 0 or not np.isfinite(mad):
        return pd.Series(np.nan, index=s.index)
    out = (s - med) / (1.4826 * (mad + EPS))
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def winsorize(s: pd.Series, lo: float = -3.0, hi: float = 3.0) -> pd.Series:
    return s.clip(lo, hi)


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    m = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - m) / (sd + EPS)


# ------------------------ Neutralization ------------------------ #
def _residualize(y: pd.Series, X: pd.DataFrame) -> pd.Series:
    """Cross-sectional OLS residuals: y ~ X (safe to NaNs)."""
    X = X.copy()
    mask = y.notna()
    for c in X.columns:
        mask &= X[c].notna()
    if mask.sum() < 3:
        yc = y - y.mean(skipna=True)
        return yc.fillna(np.nan)
    yv = y[mask].values.astype(float)
    XV = X[mask].values.astype(float)
    XV = np.c_[XV, np.ones(len(XV))]
    beta = np.linalg.pinv(XV.T @ XV) @ (XV.T @ yv)
    y_hat = XV @ beta
    resid = yv - y_hat
    out = pd.Series(np.nan, index=y.index)
    out.loc[mask.index[mask]] = resid
    return out


# ------------------------ Input normalization ------------------------ #
def _normalize_input(
    features: pd.DataFrame,
    *,
    date_col: str,
    id_col: str,
    sector_col: str | None,
) -> pd.DataFrame:
    df = features.copy()

    if date_col not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        candidates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        candidates += [date_col, "date", "Date", "datetime", "Datetime", "timestamp", "Timestamp"]
        picked = next((c for c in candidates if c in df.columns), None)
        if picked is None:
            first = df.columns[0]
            try:
                pd.to_datetime(df[first]); picked = first
            except Exception:
                pass
        if picked is None:
            raise KeyError(f"Could not find a datetime column for '{date_col}'. Columns: {list(df.columns)[:12]} ...")
        if picked != date_col:
            df = df.rename(columns={picked: date_col})
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if id_col not in df.columns:
        for alt in ("Ticker", "ticker", "symbol", "Symbol"):
            if alt in df.columns:
                df = df.rename(columns={alt: id_col})
                break
    if id_col not in df.columns:
        raise KeyError(f"Missing identifier column '{id_col}'. Columns: {list(df.columns)[:12]} ...")

    if sector_col and sector_col not in df.columns:
        df[sector_col] = "ALL"

    if "ln_mcap" not in df.columns and "market_cap" in df.columns:
        df["ln_mcap"] = np.log(pd.to_numeric(df["market_cap"], errors="coerce").replace(0, np.nan))

    return df


# ------------------------ Auto-detect feature columns ------------------------ #
_ALIAS_PATTERNS: dict[str, list[tuple[str, bool]]] = {
    "momentum": [
        (r"^mom(_\d+d)?$", True),
        (r"^ret_\d+d$", True),
        (r"^momentum.*$", True),
        (r"^roc(_\d+d)?$", True),
    ],
    "volatility": [
        (r"^vol(_\d+d)?$", False),
        (r"^stdev(_\d+d)?$", False),
        (r"^atr(_\d+)
...
[truncated]
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

### `modules/report_equities.py`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/modules/report_equities.py)

```py
# modules/report_equities.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

def _to_md_table(df: pd.DataFrame) -> str:
    """Use pandas.to_markdown if available (tabulate installed); else fallback to code block."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        # Fallback: simple code block table
        return "```\n" + df.to_string(index=False) + "\n```"

def make_equity_report(
    *,
    features_path="data/processed/equity_features.parquet",
    ranks_path="output/equity_rank_snapshot.csv",
    plots_dir="output/plots",
    out_md="output/reports/equities_report.md",
    top_k=25,
    bottom_k=25,
) -> str:
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    ranks = pd.read_csv(ranks_path, parse_dates=["date"])
    latest = ranks["date"].max()
    today = ranks[ranks["date"] == latest].copy()

    top = today.nlargest(top_k, "score")[["ticker", "score", "rank_pct", "decile"]]
    bot = today.nsmallest(bottom_k, "score")[["ticker", "score", "rank_pct", "decile"]]
    summary = today["score"].describe()[["mean", "std", "min", "25%", "50%", "75%", "max"]]

    md = []
    md.append(f"# IKF Equities — Composite Snapshot ({latest.date()})")
    md.append("")
    md.append("**Composite Score Meaning:** standardized cross-sectional z-score (per date). "
              "0≈average, +1≈one standard deviation above peers.")
    md.append("")
    md.append("## Summary Statistics")
    md.append(_to_md_table(summary.to_frame("value").reset_index().rename(columns={"index":"metric"})))

    md.append("\n## Top Ranked Tickers")
    md.append(_to_md_table(top))

    md.append("\n## Bottom Ranked Tickers")
    md.append(_to_md_table(bot))

    md.append("\n## Plots")
    for p in ["risk_return.png", "corr_heatmap.png", "ic_timeseries.png"]:
        f = Path(plots_dir) / p
        if f.exists():
            md.append(f"\n![{p}]({f.as_posix()})")

    Path(out_md).write_text("\n".join(md), encoding="utf-8")
    print(f"[OK] Markdown report written → {out_md}")
    return out_md

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

A universal market data and analytics pipeline for report generation
Fetches and updates raw market data, engineers features, computes rankings, and generates daily reports.

---

## 🚀 Quick Start

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

### Run the Pipeline

• **Default (full analysis)**  
Fetches data, builds features, computes rankings, and writes a Markdown report.  
`python run_pipeline.py`

• **Data-only mode**  
Skip analytics and just update raw data.  
`python run_pipeline.py --raw-only`

• **Optional flags**  
--recent N → rebuild last N days (e.g. `--recent 30`)  
--full   → full backfill from history_start  
--asset X → run a single asset updater (equities, futures, rates, options)  
--config Y → custom config path (default `config/config.yaml`)

---

## ⚙️ Output

data/raw/    incrementally updated market data  
data/processed/equity_features.parquet engineered features  
output/equity_rank_snapshot.csv    latest composite rankings  
output/reports/equities_report.md   markdown report summary  

---

## 🧩 Key Features

• Incremental daily updates with overlap healing  
• Feature engineering: momentum, volatility, RSI, SMA/EMA, beta, etc.  
• Cross-sectional ranking and composite scoring  
• Automated reporting and persistent rank history  

---

## 🕒 Example Cron (Linux)

# Run every weekday at 07:30 Israel time  
TZ=Asia/Jerusalem  
30 7 * * 1-5 /usr/bin/env bash -lc 'cd /path/to/IKF_Market_Pipeline && python run_pipeline.py >> logs/daily.log 2>&1'

---

Maintained as part of the **I Know First Market Intelligence Pipeline**.

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
tabulate

# Parquet + data sources
pyarrow
pandas-datareader

# Optional
rich
jupyter

```

### `run_pipeline.py`  •  [raw](https://raw.githubusercontent.com/NatanIsaacRuskin/IKF_Market_Pipeline/main/run_pipeline.py)

```py
# FILE: run_pipeline.py
from __future__ import annotations
import argparse, sys, yaml
from pathlib import Path
import pandas as pd

from modules.equities import update_equities
from modules.options import update_options
from modules.futures import update_futures
from modules.rates import update_rates
from modules.features import run_equity_features
from modules.features_dedup import finalize_equity_features_file
from modules.ranking import compute_composite_scores, save_rank_snapshot, build_metrics_cfg_from_df
from modules.report_equities import make_equity_report
from modules.composites import build_composites_from_raw

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _parse_composite_selection(arg: str | None) -> tuple[str, set[str]]:
    if not arg or arg.strip().lower() == "all": return ("all", set())
    if arg.strip().lower() == "none": return ("none", set())
    names = {x.strip() for x in arg.split(",") if x.strip()}
    return ("some", names)

def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(
        description="IKF Market Pipeline — default runs full analysis. Use --raw-only to skip analytics; use --composites to control composites."
    )
    ap.add_argument("--asset", choices=["all","equities","options","futures","rates"], default="all")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--raw-only", action="store_true", help="Only update raw data; skip features/ranking/report")
    ap.add_argument("--full", action="store_true", help="Full backfill for equities from history_start")
    ap.add_argument("--recent", type=int, default=None, help="Rebuild last N days for equities (e.g., 30)")
    ap.add_argument("--composites", default="all",
                    help="Build 'all' (default), 'none', or a comma-separated list of composite names")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    root = cfg["storage"]["root"]

    # equities updater config
    eq_cfg = (cfg.get("defaults", {}) | cfg.get("equities", {})).copy()
    if args.full:
        eq_cfg["mode"] = "full"
    if args.recent is not None:
        eq_cfg["mode"] = "recent"; eq_cfg["lookback_days"] = int(args.recent)

    # composites selection + auto-extend universe
    mode, pick = _parse_composite_selection(args.composites)
    all_comps = cfg.get("composites", []) or []
    selected_comps = [] if mode=="none" else ([c for c in all_comps if c.get("name") in pick] if mode=="some" else all_comps)
    if selected_comps:
        composite_members = {t.upper() for c in selected_comps for t in c.get("tickers", [])}
        composite_bench   = {str(c.get("benchmark")).upper() for c in selected_comps if c.get("benchmark")} - {None,"NONE","NULL"}
        base_universe = set(t.upper() for t in eq_cfg.get("universe", []))
        merged_universe = sorted(base_universe | composite_members | composite_bench)
        if merged_universe != sorted(base_universe):
            print(f"[INFO] Expanding equities.universe with composites ({len(merged_universe)} tickers).")
        eq_cfg["universe"] = merged_universe

    # raw updaters
    if args.asset in ("all","equities"): update_equities(eq_cfg, root)
    if args.asset in ("all","options"):  update_options(cfg.get("options", {}), root)
    if args.asset in ("all","futures"):  update_futures(cfg.get("futures", {}), root)
    if args.asset in ("all","rates"):    update_rates(cfg.get("rates", {}), root)

    # build composites from RAW (post raw updates)
    rank_today_df = None
    if selected_comps:
        built = build_composites_from_raw(selected_comps, root, rank_today=None,
                                          select_names=set(c.get("name") for c in selected_comps))
        if built: print("[OK] Composite price series written:", built)

    # analysis bundle (default ON unless --raw-only)
    if args.raw_only:
        return 0

    feats_conf = cfg.get("f
...
[truncated]
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