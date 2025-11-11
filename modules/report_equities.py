# modules/report_equities.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

def _to_md_table(df: pd.DataFrame) -> str:
    """Use pandas.to_markdown if available (tabulate installed); else fallback to code block."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        # Fallback: simple code block table
        return "```\n" + df.to_string(index=False) + "\n```"

def make_equity_report(
    *,
    features_path="data/processed/equity_features.parquet",
    ranks_path="output/equity_rank_snapshot.csv",
    plots_dir="output/plots",
    out_md="output/reports/equities_report.md",
    top_k=25,
    bottom_k=25,
) -> str:
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    ranks = pd.read_csv(ranks_path, parse_dates=["date"])
    latest = ranks["date"].max()
    today = ranks[ranks["date"] == latest].copy()

    top = today.nlargest(top_k, "score")[["ticker", "score", "rank_pct", "decile"]]
    bot = today.nsmallest(bottom_k, "score")[["ticker", "score", "rank_pct", "decile"]]
    summary = today["score"].describe()[["mean", "std", "min", "25%", "50%", "75%", "max"]]

    md = []
    md.append(f"# IKF Equities — Composite Snapshot ({latest.date()})")
    md.append("")
    md.append("**Composite Score Meaning:** standardized cross-sectional z-score (per date). "
              "0≈average, +1≈one standard deviation above peers.")
    md.append("")
    md.append("## Summary Statistics")
    md.append(_to_md_table(summary.to_frame("value").reset_index().rename(columns={"index":"metric"})))

    md.append("\n## Top Ranked Tickers")
    md.append(_to_md_table(top))

    md.append("\n## Bottom Ranked Tickers")
    md.append(_to_md_table(bot))

    md.append("\n## Plots")
    for p in ["risk_return.png", "corr_heatmap.png", "ic_timeseries.png"]:
        f = Path(plots_dir) / p
        if f.exists():
            md.append(f"\n![{p}]({f.as_posix()})")

    Path(out_md).write_text("\n".join(md), encoding="utf-8")
    print(f"[OK] Markdown report written → {out_md}")
    return out_md
