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
    _ensure_cols(prices, ["date","ticker","close"])
    df = prices.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["date"] = pd.to_datetime(df["date"])
    if spec.start: df = df[df["date"] >= pd.to_datetime(spec.start)]
    df = df[df["ticker"].isin([t.upper() for t in spec.tickers])]
    if df.empty: raise ValueError(f"No prices for composite '{spec.name}'.")
    wdf = _normalize_weights(spec.tickers, spec.weights, spec.weights_csv)
    px = df.pivot(index="date", columns="ticker", values="close").sort_index().ffill()
    wdf = wdf[wdf["ticker"].isin(px.columns)]
    if wdf.empty: raise ValueError("Weighted tickers missing in price matrix.")
    base = px / px.iloc[0]

    if spec.rebalance == "none":
        w = wdf.set_index("ticker")["weight"].reindex(base.columns).fillna(0.0).values
        comp = (base * w).sum(axis=1)
        out = comp.rename("close").to_frame(); out["name"] = spec.name
        return out.reset_index()[["date","name","close"]]

    freq = "M" if spec.rebalance == "monthly" else "Q"
    starts = _period_starts(base.index, freq)
    w = wdf.set_index("ticker")["weight"].reindex(base.columns).fillna(0.0).values
    rets = base.pct_change().fillna(0.0)
    pv = pd.Series(index=base.index, dtype=float); pv.iloc[0] = float((base.iloc[0]*w).sum())
    curr_w = w.copy()
    for i in range(1, len(base)):
        pv.iloc[i] = pv.iloc[i-1] * (1.0 + float((rets.iloc[i]*curr_w).sum()))
        if starts.iloc[i]: curr_w = w.copy()
    out = pv.rename("close").to_frame(); out["name"] = spec.name
    return out.reset_index()[["date","name","close"]]

def build_composites_from_raw(
    cfg_composites: List[Dict],
    storage_root: str,
    rank_today: pd.DataFrame | None = None,
    select_names: Optional[set[str]] = None,
) -> Dict[str,str]:
    """Build selected composites from RAW equities. Returns {name: parquet_path}."""
    results: Dict[str,str] = {}
    wanted = select_names if select_names else None
    for c in (cfg_composites or []):
        name = c.get("name")
        if not name: continue
        if wanted is not None and name not in wanted: continue
        spec = CompositeSpec(
            name=name,
            tickers=list(c.get("tickers", [])),
            weights=c.get("weights"),
            weights_csv=c.get("weights_csv"),
            benchmark=c.get("benchmark"),
            rebalance=str(c.get("rebalance","none")).lower(),
            start=c.get("start"),
        )
        prices = _read_prices_from_raw(storage_root, spec.tickers)
        comp = _build_composite_series(prices, spec)
        out_dir = Path("data/processed/composites"); out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{spec.name}_prices.parquet"
        comp.to_parquet(out_path)
        results[spec.name] = str(out_path)

        if rank_today is not None and not rank_today.empty:
            wdf = _normalize_weights(spec.tickers, spec.weights, spec.weights_csv)
            wdf["ticker"] = wdf["ticker"].str.upper()
            cols = [c for c in ["ticker","score","rank_pct","decile"] if c=="ticker" or c in rank_today.columns]
            snap = wdf.merge(rank_today[cols], on="ticker", how="left").sort_values("weight", ascending=False)
            sdir = Path("output/snapshots"); sdir.mkdir(parents=True, exist_ok=True)
            snap.to_csv(sdir / f"{spec.name}_snapshot.csv", index=False)
    return results
