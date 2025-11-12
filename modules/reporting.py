# modules/reporting.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional
import os
import pandas as pd

PLOT_FILES = {
    "cum_return": "equities_cum_return.png",
    "rolling_sharpe": "equities_rolling_sharpe.png",
    "drawdown": "equities_drawdown.png",
    "risk_return": "equities_risk_return.png",
    "corr_heatmap": "equities_corr_heatmap.png",
}

def _md_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_(no data)_\n"
    return df.to_markdown(index=False)

def write_equities_report(
    out_md: Path | str,
    asof: str,
    scorecard: pd.DataFrame | None,
    leaders: pd.DataFrame | None,
    laggards: pd.DataFrame | None,
    plots_dir: Path,
    include_sections: list[str],
    time_windows: list[str] | None = None,
) -> None:
    out_md = Path(out_md)
    plots_dir = Path(plots_dir)

    def _rel(rel: str) -> str:
        """relative path from report file to plot file"""
        fp = (plots_dir / rel)
        return os.path.relpath(fp, start=out_md.parent)

    def _img(rel: str) -> str:
        fp = plots_dir / rel
        return f"![{rel}]({_rel(rel)})" if fp.exists() else ""

    def _link(rel: str) -> str:
        fp = plots_dir / rel
        return f"[{rel}]({_rel(rel)})" if fp.exists() else ""

    lines: list[str] = []
    lines.append(f"# IKF Equities — Composite Snapshot ({asof})")

    # summary table
    if scorecard is not None and not scorecard.empty:
        lines.append("\n## Composite Scorecard\n")
        lines.append(_md_table(scorecard))

    # leaders/laggards
    if leaders is not None and not leaders.empty:
        lines.append("\n## Top Ranked Tickers\n")
        lines.append(_md_table(leaders))
    if laggards is not None and not laggards.empty:
        lines.append("\n## Bottom Ranked Tickers\n")
        lines.append(_md_table(laggards))

    # full-period charts (show only if file exists / section enabled)
    if "cum_return" in include_sections:
        img = _img("equities_cum_return.png")
        if img:
            lines.append("\n## Cumulative Return (Full Period)\n")
            lines.append(img)
    if "drawdown" in include_sections:
        img = _img("equities_drawdown.png")
        if img:
            lines.append("\n## Drawdown (Full Period)\n")
            lines.append(img)
    if "risk_return" in include_sections:
        img = _img("equities_risk_return.png")
        if img:
            lines.append("\n## Risk vs Return (Full Period)\n")
            lines.append(img)
    if "rolling_sharpe" in include_sections:
        img = _img("equities_rolling_sharpe.png")
        if img:
            lines.append("\n## Rolling Sharpe (Full Period)\n")
            lines.append(img)
    if "corr_heatmap" in include_sections:
        img = _img("equities_corr_heatmap.png")
        if img:
            lines.append("\n## Correlation Heatmap (1D Log Returns)\n")
            lines.append(img)

    # IKF time-window snapshots
    if time_windows:
        lines.append("\n## Time-Window Snapshots (IKF)\n")
        lines.append("_Each plot compares IKF composite(s) (green) vs SPY (grey) at the specified horizon._\n")

        for lab in time_windows:
            # Generate file paths
            cum = plots_dir / f"equities_cum_return_{lab}.png"
            dd  = plots_dir / f"equities_drawdown_{lab}.png"
            rr  = plots_dir / f"equities_risk_return_{lab}.png"
            rs  = plots_dir / f"equities_rolling_sharpe_{lab}.png"

            # Build markdown links (only if file exists)
            links = []
            if cum.exists():
                links.append(f"[Cumulative Return]({cum.as_posix()})")
            if dd.exists():
                links.append(f"[Drawdown]({dd.as_posix()})")
            if rr.exists():
                links.append(f"[Risk / Return]({rr.as_posix()})")
            if rs.exists():
                links.append(f"[Rolling Sharpe]({rs.as_posix()})")

            # Write section if anything exists
            if links:
                lines.append(f"\n### {lab.upper()}\n")
                for link in links:
                    lines.append(f"- {link}")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))