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
        (r"^atr(_\d+)?$", False),
    ],
    "sharpe": [(r"^sharpe(_\d+d)?$", True)],
    "rsi": [(r"^rsi(_\d+)?$", False)],
    "skew": [(r"^skew(_\d+d)?$", True)],
    "kurtosis": [(r"^kurt(_\d+d)?$", False)],
    "boll_pos": [(r"^boll(_pct|_pos)(_?\d+d)?$", True)],
    "quality": [(r"^roic.*$", True), (r"^roe.*$", True), (r"^gross_margin.*$", True)],
    "value": [(r"^value_ep$", True), (r"^earnings_yield$", True)],
}


def _find_first(df_cols_lower: list[str], patterns: list[tuple[str, bool]]) -> tuple[str | None, bool | None]:
    for pat, hib in patterns:
        rx = re.compile(pat)
        for col in df_cols_lower:
            if rx.match(col):
                return col, hib
    return None, None


def build_metrics_cfg_from_df(df: pd.DataFrame) -> tuple[dict, dict]:
    cols_lower = [c.lower() for c in df.columns]
    lower_to_actual = {c.lower(): c for c in df.columns}
    chosen, cfg = {}, {}
    order = ["momentum", "volatility", "sharpe", "skew", "kurtosis", "boll_pos", "quality", "value", "rsi"]
    base_weights = {
        "momentum": 0.30,
        "volatility": 0.15,
        "sharpe": 0.20,
        "skew": 0.10,
        "kurtosis": 0.05,
        "boll_pos": 0.10,
        "quality": 0.07,
        "value": 0.03,
        "rsi": 0.00,
    }
    for canon in order:
        col_lower, hib = _find_first(cols_lower, _ALIAS_PATTERNS.get(canon, []))
        if col_lower is None:
            continue
        actual = lower_to_actual[col_lower]
        chosen[canon] = actual
        if canon == "rsi":
            hib = True
        cfg[actual] = {"w": base_weights[canon], "higher_is_better": bool(hib)}
    wsum = sum(v["w"] for v in cfg.values())
    if wsum > 0:
        for k in cfg:
            cfg[k]["w"] = cfg[k]["w"] / wsum
    return cfg, chosen


# ------------------------ Main API ------------------------ #
def compute_composite_scores(
    features: pd.DataFrame,
    *,
    date_col: str = "date",
    id_col: str = "ticker",
    sector_col: str = "sector",
    metrics_cfg: dict | None = None,
    winsor: tuple[float, float] = (-3, 3),
    neutralize_vs: tuple | None = ("beta_60d", "ln_mcap"),
) -> pd.DataFrame:
    df = _normalize_input(features, date_col=date_col, id_col=id_col, sector_col=sector_col)

    if metrics_cfg is None or not metrics_cfg:
        metrics_cfg, chosen = build_metrics_cfg_from_df(df)
        if not metrics_cfg:
            raise ValueError("Could not auto-detect any usable feature columns for ranking.")
        print("[INFO] Composite metrics selected:")
        for canon, actual in chosen.items():
            hib_flag = "T" if metrics_cfg[actual]["higher_is_better"] else "F"
            w = metrics_cfg[actual]["w"]
            print(f"   {canon:10s} -> {actual}  (w={w:.3f}, hib={hib_flag})")

    use_metrics = [m for m in metrics_cfg.keys() if m in df.columns]
    if not use_metrics:
        raise ValueError("None of the metrics in metrics_cfg are in the features DataFrame.")

    group_keys = [date_col] + ([sector_col] if sector_col else [])
    zcols = []

    for m in use_metrics:
        cfg = metrics_cfg[m]
        df[m] = pd.to_numeric(df[m], errors="coerce")
        if re.match(r"(?i)^rsi(_\d+)?$", m):
            df[m] = (df[m] - 50.0).abs() * -1.0
            cfg = {"w": cfg["w"], "higher_is_better": True}
        zname = f"z_{m}"
        zcols.append(zname)
        sign = 1.0 if cfg.get("higher_is_better", True) else -1.0
        df[zname] = (
            df.groupby(group_keys, observed=True)[m]
              .transform(lambda s: winsorize(_safe_robust_z(s), winsor[0], winsor[1]))
              .pipe(zscore) * sign
        )

    w = np.array([metrics_cfg[m]["w"] for m in use_metrics], dtype=float)
    w = w / (np.sqrt((w ** 2).sum()) + EPS)
    df["score_raw"] = np.nansum(df[[f"z_{m}" for m in use_metrics]].values * w, axis=1)

    base = "score_raw"
    if neutralize_vs:
        def _per_date(g: pd.DataFrame) -> pd.DataFrame:
            controls = [c for c in neutralize_vs if c in g.columns]
            if not controls:
                g["score_neutral"] = g[base]
                return g
            g["score_neutral"] = _residualize(g[base].astype(float), g[controls].astype(float))
            return g
        df = df.groupby(date_col, group_keys=False).apply(_per_date)
        base = "score_neutral"

    # --- Final ranks (NaN-safe) ---
    df["score"] = df.groupby(date_col)[base].transform(zscore)
    df["rank_pct"] = df.groupby(date_col)["score"].rank(ascending=False, pct=True)

    decile_raw = np.ceil(df["rank_pct"] * 10)
    decile_raw = decile_raw.clip(1, 10)
    df["decile"] = decile_raw.where(np.isfinite(decile_raw)).astype("Int64")  # NA-safe integers

    cols = [date_col, id_col, "score", "rank_pct", "decile"] + [f"z_{m}" for m in use_metrics]
    return df[cols]


def save_rank_snapshot(df_scores: pd.DataFrame, path: str) -> str:
    latest = pd.to_datetime(df_scores["date"]).max()
    snap = df_scores[df_scores["date"] == latest].copy().sort_values("score", ascending=False)
    snap.to_csv(path, index=False)
    print(f"[OK] Rank snapshot saved → {path}")
    return path
