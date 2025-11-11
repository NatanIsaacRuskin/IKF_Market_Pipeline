# modules/plotting.py
from dataclasses import dataclass
from typing import Dict, Optional
import matplotlib.pyplot as plt

@dataclass
class PlotCfg:
    colors: Dict[str, str]
    dpi: int
    marker: Dict[str, float]
    line: Dict[str, float]
    style: str

def load_plot_cfg(cfg) -> PlotCfg:
    """Extract plot config dict -> PlotCfg."""
    pass

def plot_cum_return(spy_nav, composite_navs: Dict[str, object], out_path: str, cfg: PlotCfg):
    pass

def plot_rolling_sharpe(spy_rs, comp_rs: Dict[str, object], out_path: str, cfg: PlotCfg):
    pass

def plot_drawdown(spy_dd, comp_dd: Dict[str, object], out_path: str, cfg: PlotCfg):
    pass

def plot_risk_return(spy_point, comp_points: Dict[str, tuple], out_path: str, cfg: PlotCfg):
    pass

def plot_corr_heatmap(df_corr, out_path: str, cfg: PlotCfg):
    pass
