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

    # ---- EXTRA FEATURES FOR REPORTS (this is the part you asked to insert) ----
    # drawdown stats
    dd, mdd = _drawdown(px)
    feats["dd"] = dd
    feats["mdd_static"] = float(mdd)

    # Bollinger features
    ma, up, lo, pct_band = _bollinger(px, window=BOLL_W)
    feats[f"boll_ma_{BOLL_W}"] = ma
    feats[f"boll_up_{BOLL_W}"] = up
    feats[f"boll_lo_{BOLL_W}"] = lo
    feats[f"boll_pos_{BOLL_W}"] = pct_band

    # ATR (uses OHLC if present)
    hi = _to_series(df, "High")
    lo_ = _to_series(df, "Low")
    cl = _to_series(df, "Close")
    feats[f"atr_{ATR_W}"] = _atr(hi, lo_, cl, window=ATR_W)

    # Higher-moment signals
    feats[f"ret_skew_{SKEW_W}"] = ret.rolling(SKEW_W).skew()
    feats[f"ret_kurt_{KURT_W}"] = ret.rolling(KURT_W).kurt()

    # final tag
    feats["ticker"] = ticker
    return feats.dropna(how="all")

# ---------- public entry point used by run_pipeline.py ----------
def run_equity_features(cfg_eq_feat: dict):
    """
    Orchestrates equity feature engineering + basic plots.
    Expects keys in config:
      processed_path, plots_path, plots (bool), benchmark (ticker)
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

    # ---- Beta/Alpha vs benchmark ----
    try:
        bench = cfg_eq_feat.get("benchmark", "SPY")
        bpath = RAW_EQUITIES_DIR / f"{bench}.parquet"
        if bpath.exists():
            bdf = pd.read_parquet(bpath)
            if isinstance(bdf.columns, pd.MultiIndex):
                bdf.columns = bdf.columns.get_level_values(0)
            bdf.columns = [str(c).strip() for c in bdf.columns]
            b_close = _to_series(bdf, "Adj Close") if "Adj Close" in bdf.columns else _to_series(bdf, "Close")
            b_ret = np.log(b_close).diff()

            out_rows = []
            for tkr, g in all_feats.groupby("ticker"):
                s_ret = g["ret_1d_log"]
                beta, alpha = _rolling_beta_alpha(s_ret, b_ret.reindex(s_ret.index), win=60)
                tmp = pd.DataFrame({"beta_60": beta, "alpha_60_ann": alpha})
                tmp["ticker"] = tkr
                tmp.index.name = "Date"
                out_rows.append(tmp)

            if out_rows:
                ba = pd.concat(out_rows).dropna(how="all")
                merged = all_feats.join(ba[["beta_60","alpha_60_ann"]], how="left")
                merged.to_parquet(processed_path / "equity_features.parquet")
                all_feats = merged
                print("[OK] Added beta/alpha vs benchmark to features.")
        else:
            print(f"[INFO] Benchmark parquet not found: {bpath} (skip beta/alpha)")
    except Exception as e:
        print(f"[WARN] beta/alpha: {e}")

    # ---- Cross-sectional ranking snapshot (today) ----
    try:
        last = all_feats.groupby("ticker").tail(1).set_index("ticker")
        score = (
            last[f"mom_{int(cfg_eq_feat.get('win_ret',20))}d"].rank(pct=True) +
            (-last["roll_vol_ann"]).rank(pct=True) +
            (-last[f'rsi_{int(cfg_eq_feat.get("rsi_period",14))}'].abs()).rank(pct=True)
        )
        rank = pd.DataFrame({"score": score}).sort_values("score", ascending=False)
        top = rank.head(10); bot = rank.tail(10)
        outp = processed_path / "equity_rank_snapshot.csv"
        rank.to_csv(outp)
        print(f"[OK] Cross-sectional ranks -> {outp}")
        # (You can print(top) or bot here if you want console output)
    except Exception as e:
        print(f"[WARN] ranking: {e}")

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
