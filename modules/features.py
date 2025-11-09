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

def build_equity_features(cfg_eq_feat: dict) -> pd.DataFrame:
    files = sorted(RAW_EQUITIES_DIR.glob("*.parquet"))
    out = []
    for f in files:
        tkr = f.stem
        try:
            df = pd.read_parquet(f)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):       # flatten if needed
                df.columns = df.columns.get_level_values(0)
            df.columns = [str(c).strip() for c in df.columns]

            feats = _compute_one_ticker_features(df, tkr, cfg_eq_feat)
            if not feats.empty:
                out.append(feats)
        except Exception as e:
            print(f"[WARN] {tkr}: {e}")
    if not out:
        return pd.DataFrame()
    feats_all = pd.concat(out).sort_index()
    cols = ["ticker"] + [c for c in feats_all.columns if c != "ticker"]
    return feats_all[cols]

def _plot_risk_return(df_feats: pd.DataFrame, plots_dir: Path):
    ensure_dir(plots_dir)
    tmp = (df_feats.groupby("ticker")
           .agg(ret_ann=("ret_1d_log", lambda x: x.mean()*252),
                vol_ann=("ret_1d_log", lambda x: x.std()*np.sqrt(252)))).dropna()
    if tmp.empty:
        print("[INFO] Not enough data to draw risk/return scatter.")
        return
    plt.figure(figsize=(8,6))
    sns.scatterplot(x="vol_ann", y="ret_ann", data=tmp)
    for t, r in tmp.iterrows():
        plt.text(r["vol_ann"], r["ret_ann"], t, fontsize=8)
    plt.axhline(0, color="gray", lw=0.8)
    plt.xlabel("Annualized Volatility"); plt.ylabel("Annualized Return")
    plt.title("Equities Risk/Return (annualized)")
    out = plots_dir / "equities_risk_return.png"
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"[OK] plot -> {out}")

def _plot_corr_heatmap(df_feats: pd.DataFrame, plots_dir: Path):
    ensure_dir(plots_dir)
    mat = df_feats.pivot_table(index=df_feats.index, columns="ticker", values="ret_1d_log")
    mat = mat.dropna(how="all").dropna(axis=1, how="all")

    # require a minimum number of observations per ticker
    min_obs = 60
    if mat.shape[0] == 0:
        print("[INFO] No return rows available for correlation; skipping heatmap.")
        return
    keep = mat.count() >= min_obs
    mat = mat.loc[:, keep]

    if mat.shape[1] < 2:
        print("[INFO] Not enough overlapping data to draw correlation heatmap (need ≥2 tickers).")
        return

    corr = mat.corr()
    if corr.size == 0:
        print("[INFO] Correlation matrix empty after cleaning; skipping heatmap.")
        return

    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap="RdBu_r", center=0, square=True)
    plt.title("Equities Return Correlation")
    out = plots_dir / "equities_corr_heatmap.png"
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"[OK] plot -> {out}")

def run_equity_features(cfg: dict):
    """cfg should be config['features']['equities']"""
    if not cfg.get("enabled", True):
        print("[INFO] equity features disabled in config.")
        return

    processed_dir = Path(cfg.get("processed_path", "data/processed"))
    plots_dir     = Path(cfg.get("plots_path", "output/plots"))

    feats = build_equity_features(cfg)
    if feats.empty:
        print("[WARN] no equity raw data in data/raw/equities. Run the pipeline first.")
        return

    ensure_dir(processed_dir)
    out_path = processed_dir / "equities_features.parquet"
    feats.to_parquet(out_path)
    print(f"[OK] saved -> {out_path}")

    if cfg.get("plots", True):
        try:
            _plot_risk_return(feats, plots_dir)
            _plot_corr_heatmap(feats, plots_dir)
        except Exception as e:
            print(f"[WARN] plotting failed: {e}")
