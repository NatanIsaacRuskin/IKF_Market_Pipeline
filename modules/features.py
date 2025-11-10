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
    files = sorted(RAW_EQUITIES_DIR.glob("*.parquet"))
    for f in files:
        tkr = f.stem
        try:
            df = pd.read_parquet(f)
            one = _compute_one_ticker_features(df, tkr, cfg_eq_feat)
            if not one.empty:
                feats_list.append(one)
        except Exception as e:
            print(f"[WARN] features {tkr}: {e}")

    if not feats_list:
        print("[WARN] No features built (no equities parquet files found).")
        return

    all_feats = pd.concat(feats_list, axis=0, sort=False)
    all_feats.index.name = "Date"
    all_feats.to_parquet(processed_path / "equity_features.parquet")
    print(f"[OK] Saved features -> {processed_path/'equity_features.parquet'}  rows={len(all_feats):,}")

    if not cfg_eq_feat.get("plots", True):
        return

    # --- Plot 1: correlation heatmap of daily returns by ticker ---
    try:
        piv = all_feats.pivot_table(index=all_feats.index, columns="ticker", values="ret_1d_log")
        corr = piv.corr(min_periods=60)
        plt.figure(figsize=(10,8))
        sns.heatmap(corr, cmap="vlag", center=0, annot=False, linewidths=0.2)
        plt.title("Equities: 1D Log Return Correlations")
        plt.tight_layout()
        out1 = plots_path / "equities_corr_heatmap.png"
        plt.savefig(out1, dpi=150); plt.close()
        print(f"[OK] Plot -> {out1}")
    except Exception as e:
        print(f"[WARN] corr heatmap: {e}")

    # --- Plot 2: risk–return scatter (annualized) ---
    try:
        ann_mean = piv.mean(skipna=True) * 252
        ann_vol  = piv.std(skipna=True) * np.sqrt(252)
        rr = pd.DataFrame({"ann_mean": ann_mean, "ann_vol": ann_vol}).dropna()
        plt.figure(figsize=(10,7))
        plt.scatter(rr["ann_vol"], rr["ann_mean"], s=40, alpha=0.8)
        for tkr, row in rr.iterrows():
            plt.text(row["ann_vol"], row["ann_mean"], tkr, fontsize=8, ha="left", va="bottom")
        plt.xlabel("Annualized Volatility"); plt.ylabel("Annualized Return")
        plt.title("Equities: Risk–Return")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        out2 = plots_path / "equities_risk_return.png"
        plt.savefig(out2, dpi=150); plt.close()
        print(f"[OK] Plot -> {out2}")
    except Exception as e:
        print(f"[WARN] risk-return plot: {e}")
