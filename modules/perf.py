# modules/perf.py
import pandas as pd

def build_composite_nav(prices: pd.DataFrame, weights: pd.DataFrame, rebalance: str) -> pd.Series:
    """Return NAV index (start=100)."""
    pass

def perf_table(nav: pd.Series, spy_nav: pd.Series, asof: pd.Timestamp) -> pd.DataFrame:
    """Return summary metrics (YTD, 1Y, 3Y, Max, Vol, Sharpe, MaxDD, Calmar)."""
    pass

def drawdown_series(nav: pd.Series) -> pd.Series:
    """Return underwater series (0 to negative values)."""
    pass

def rolling_sharpe(returns: pd.Series, window_days: int) -> pd.Series:
    """Return annualized rolling Sharpe."""
    pass
