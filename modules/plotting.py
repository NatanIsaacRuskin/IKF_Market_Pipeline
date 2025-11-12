# modules/plotting.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class PlotCfg:
    colors: Dict[str, str]
    dpi: int
    marker: Dict[str, float]
    line: Dict[str, float]
    style: str

def load_plot_cfg(cfg) -> PlotCfg:
    p = cfg.get("plot", {})
    colors = p.get("colors", {}) or {}
    out = PlotCfg(
        colors={
            "spy": colors.get("spy", "#9E9E9E"),       # grey
            "ikf_green": colors.get("ikf_green", "#1BAA4A"),
            "accent": colors.get("accent", "#3B82F6"),
        },
        dpi=int(p.get("dpi", 140)),
        marker=p.get("marker", {"size_normal": 60, "size_highlight": 110}),
        line=p.get("line", {"width_normal": 2.0, "width_highlight": 3.0}),
        style=p.get("style", "seaborn-whitegrid"),
    )
    plt.style.use("seaborn-v0_8-whitegrid" if out.style.startswith("seaborn") else out.style)
    plt.rcParams.update({"figure.dpi": out.dpi, "axes.titlesize": 13, "axes.labelsize": 11, "legend.frameon": False})
    return out

def _fmt_pct(ax):
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")

def plot_cum_return(spy_nav: pd.Series, composite_navs: Dict[str, pd.Series], out_path: str, cfg: PlotCfg):
    if spy_nav is None or len(composite_navs) == 0: return
    plt.figure(figsize=(9.5, 5.2))
    # Convert to cumulative return (since first point)
    spy = spy_nav / spy_nav.iloc[0] - 1.0
    plt.plot(spy.index, spy.values, color=cfg.colors["spy"], linewidth=cfg.line["width_highlight"], label="S&P (SPY)")
    for name, nav in composite_navs.items():
        cr = nav / nav.iloc[0] - 1.0
        plt.plot(cr.index, cr.values, color=cfg.colors["ikf_green"], linewidth=cfg.line["width_highlight"], label=f"IKF {name}")
        # annotate last point
        plt.text(cr.index[-1], cr.values[-1], f"  {name} {cr.values[-1]:+.1%}", color=cfg.colors["ikf_green"], va="center")
    plt.text(spy.index[-1], spy.values[-1], f"  S&P {spy.values[-1]:+.1%}", color=cfg.colors["spy"], va="center")
    _fmt_pct(plt.gca())
    plt.title("Cumulative Return")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_rolling_sharpe(spy_returns: pd.Series, comp_returns: Dict[str, pd.Series], window: int, out_path: str, cfg: PlotCfg):
    if spy_returns is None or len(comp_returns) == 0: return
    plt.figure(figsize=(9.5, 5.2))
    rs_spy = (spy_returns.rolling(window).mean() / spy_returns.rolling(window).std()) * np.sqrt(252)
    plt.plot(rs_spy.index, rs_spy.values, color=cfg.colors["spy"], linewidth=cfg.line["width_highlight"], label="S&P (SPY)")
    for name, r in comp_returns.items():
        rs = (r.rolling(window).mean() / r.rolling(window).std()) * np.sqrt(252)
        plt.plot(rs.index, rs.values, color=cfg.colors["ikf_green"], linewidth=cfg.line["width_highlight"], label=f"IKF {name}")
    plt.axhline(0, color="#999", linewidth=1, linestyle="--")
    plt.title(f"Rolling Sharpe ({window}D)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path); plt.close()

def plot_drawdown(spy_nav: pd.Series, comp_navs: Dict[str, pd.Series], out_path: str, cfg: PlotCfg):
    if spy_nav is None or len(comp_navs) == 0: return
    def dd(s: pd.Series):
        return s / s.cummax() - 1.0
    plt.figure(figsize=(9.5, 5.2))
    plt.plot(spy_nav.index, dd(spy_nav).values, color=cfg.colors["spy"], linewidth=cfg.line["width_highlight"], label="S&P (SPY)")
    for name, nav in comp_navs.items():
        plt.plot(nav.index, dd(nav).values, color=cfg.colors["ikf_green"], linewidth=cfg.line["width_highlight"], label=f"IKF {name}")
    _fmt_pct(plt.gca())
    plt.title("Drawdown")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path); plt.close()

def plot_risk_return(spy_returns: pd.Series, comp_returns: Dict[str, pd.Series], out_path: str, cfg: PlotCfg):
    if spy_returns is None or len(comp_returns) == 0: return
    def ann_vol(r): return r.std() * np.sqrt(252)
    def ann_ret(r):
        cr = (1.0 + r).prod()
        n = r.shape[0] / 252.0
        return cr ** (1/n) - 1 if n > 0 else np.nan
    plt.figure(figsize=(8.5, 5.5))
    # SPY point
    x_spy, y_spy = ann_vol(spy_returns), ann_ret(spy_returns)
    plt.scatter([x_spy], [y_spy], s=cfg.marker["size_highlight"]*1.5, color=cfg.colors["spy"], edgecolors="black", linewidths=1.0, label="S&P (SPY)", zorder=3)
    # Composites
    for name, r in comp_returns.items():
        x, y = ann_vol(r), ann_ret(r)
        plt.scatter([x], [y], s=cfg.marker["size_highlight"]*1.5, color=cfg.colors["ikf_green"], edgecolors="black", linewidths=1.0, label=f"IKF {name}", zorder=3)
        plt.text(x, y, f"  {name}", color="black", va="center")
    plt.xlabel("Annualized Volatility"); plt.ylabel("Annualized Return"); _fmt_pct(plt.gca())
    plt.title("Risk–Return")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path); plt.close()

def plot_corr_heatmap(returns: pd.DataFrame, out_path: str, cfg: PlotCfg):
    if returns is None or returns.empty: return
    corr = returns.corr()
    plt.figure(figsize=(6.8, 5.6))
    sns.heatmap(corr, cmap="vlag", vmin=-1, vmax=1, annot=False, square=True, cbar_kws={"shrink": .8})
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_path); plt.close()
