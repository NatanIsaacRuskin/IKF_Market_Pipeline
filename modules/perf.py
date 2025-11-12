# modules/perf.py
from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252

def _to_nav_from_prices(prices: pd.Series) -> pd.Series:
    """Convert a price series to NAV=100 base."""
    s = prices.dropna().astype(float)
    if s.empty: return s
    ret = s.pct_change().fillna(0.0)
    nav = (1.0 + ret).cumprod() * 100.0
    nav.name = getattr(prices, "name", "NAV")
    return nav

def build_composite_nav(prices: pd.DataFrame, weights: pd.DataFrame, rebalance: str) -> pd.Series:
    """
    prices: wide DF (date index, tickers columns) with close prices
    weights: wide DF (date index, tickers columns) with target weights on rebalance dates
    rebalance: "none" | "monthly" | "quarterly"
    Returns NAV=100 pd.Series.
    """
    if prices.empty or weights.empty:
        return pd.Series(dtype=float, name="NAV")

    px = prices.sort_index().ffill().dropna(how="all", axis=1)
    rets = px.pct_change().fillna(0.0)

    if rebalance.lower() == "none":
        # Use first available row of weights and keep static weights (renormalize to 1)
        w0 = weights.iloc[0].reindex(px.columns).fillna(0.0)
        w = (w0 / w0.clip(lower=0).sum()).fillna(0.0)
        port_ret = (rets * w).sum(axis=1)
    else:
        if rebalance.lower() == "monthly":
            rule = "M"
        elif rebalance.lower() == "quarterly":
            rule = "Q"
        else:
            rule = "M"

        w_sched = weights.reindex(px.index).ffill().groupby(pd.Grouper(freq=rule)).head(1)
        w_sched = w_sched.reindex(px.index).ffill().reindex(columns=px.columns).fillna(0.0)
        w_sched = w_sched.div(w_sched.clip(lower=0).sum(axis=1), axis=0).fillna(0.0)
        port_ret = (rets * w_sched).sum(axis=1)

    nav = (1.0 + port_ret).cumprod() * 100.0
    nav.name = "NAV"
    return nav

def _cagr(nav: pd.Series) -> float:
    if nav.empty: return np.nan
    n_days = (nav.index[-1] - nav.index[0]).days
    if n_days <= 0: return np.nan
    total = nav.iloc[-1] / nav.iloc[0]
    years = n_days / 365.25
    return float(total ** (1/years) - 1) if years > 0 else np.nan

def drawdown_series(nav: pd.Series) -> pd.Series:
    if nav.empty: return nav
    peak = nav.cummax()
    dd = nav / peak - 1.0
    dd.name = "drawdown"
    return dd

def rolling_sharpe(returns: pd.Series, window_days: int) -> pd.Series:
    if returns.empty: return returns
    mu = returns.rolling(window_days).mean()
    sd = returns.rolling(window_days).std()
    rs = (mu / sd) * np.sqrt(TRADING_DAYS)
    rs.name = "rolling_sharpe"
    return rs

def _ann_vol(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(TRADING_DAYS)) if returns.size else np.nan

def _max_dd(nav: pd.Series) -> float:
    dd = drawdown_series(nav)
    return float(dd.min()) if dd.size else np.nan

def perf_table(nav: pd.Series, spy_nav: pd.Series, asof: pd.Timestamp) -> pd.DataFrame:
    """Return summary metrics for nav vs SPY: YTD, 1Y, Max, Vol, Sharpe, MaxDD, Calmar."""
    def _period_ret(s: pd.Series, start: pd.Timestamp) -> float:
        s = s.loc[s.index >= start]
        return float(s.iloc[-1] / s.iloc[0] - 1.0) if s.size > 1 else np.nan

    idx = pd.to_datetime(nav.index)
    asof = pd.to_datetime(asof) if pd.notna(asof) else idx.max()

    d = {}
    nav_ret = nav.pct_change().fillna(0.0)
    spy_ret = spy_nav.pct_change().fillna(0.0)

    for lbl, days in (("YTD", None), ("1Y", 365), ("MAX", "max")):
        if lbl == "YTD":
            start = pd.Timestamp(year=asof.year, month=1, day=1)
        elif days == "max":
            start = idx.min()
        else:
            start = asof - pd.Timedelta(days=days)
        d[f"ret_{lbl.lower()}"] = _period_ret(nav, start)
    d["vol_ann"] = _ann_vol(nav_ret)
    d["sharpe"] = float((nav_ret.mean() / nav_ret.std()) * np.sqrt(TRADING_DAYS)) if nav_ret.std() > 0 else np.nan
    d["max_dd"] = _max_dd(nav)
    cagr = _cagr(nav)
    d["calmar"] = (cagr / abs(d["max_dd"])) if (pd.notna(cagr) and d["max_dd"] != 0) else np.nan

    out = pd.DataFrame([d])
    out.index = [asof.date()]
    return out
