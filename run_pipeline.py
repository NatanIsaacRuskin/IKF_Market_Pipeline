from __future__ import annotations
import argparse, sys, yaml
from pathlib import Path
import pandas as pd

from modules.equities import update_equities
from modules.options import update_options
from modules.futures import update_futures
from modules.rates import update_rates
from modules.features import run_equity_features
from modules.features_dedup import finalize_equity_features_file
from modules.ranking import compute_composite_scores, save_rank_snapshot, build_metrics_cfg_from_df
from modules.report_equities import make_equity_report
from modules.composites import build_composites_from_raw

# --- Optional clean-outputs orchestrator imports (safe even if stubs) ---
from modules.perf import (
    build_composite_nav, perf_table, drawdown_series, rolling_sharpe  # noqa: F401
)
from modules.plotting import (
    load_plot_cfg, plot_cum_return, plot_rolling_sharpe, plot_drawdown, plot_risk_return, plot_corr_heatmap  # noqa: F401
)
from modules.reporting import write_equities_report  # noqa: F401
# -----------------------------------------------------------------------

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _parse_composite_selection(arg: str | None) -> tuple[str, set[str]]:
    if not arg or arg.strip().lower() == "all": return ("all", set())
    if arg.strip().lower() == "none": return ("none", set())
    names = {x.strip() for x in arg.split(",") if x.strip()}
    return ("some", names)

# ---------------------------------------------------------------------
# Optional: clean outputs orchestrator (SAFE no-op until stubs are filled)
# ---------------------------------------------------------------------
def run_clean_outputs(cfg: dict) -> None:
    """
    Safe optional step. Ensures output dirs exist and applies plotting style
    if available. Does nothing else until modules/perf|plotting|reporting
    functions are implemented. Will not raise.
    """
    try:
        paths = cfg.get("paths", {})
        plots_dir = Path(paths.get("plots", "output/plots"))
        reports_dir = Path(paths.get("reports", "output/reports"))
        snapshots_dir = Path(paths.get("snapshots", "output/snapshots"))
        processed_dir = Path(paths.get("processed", "data/processed"))

        plots_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        (processed_dir / "composites").mkdir(parents=True, exist_ok=True)

        # Try to load/apply plotting style (ok if still a stub)
        try:
            _ = load_plot_cfg(cfg)  # type: ignore
        except Exception as _e:
            print(f"[clean-outputs] plotting cfg not active (ok): {_e}")

        print("[clean-outputs] stage ready (no-op until plotting/perf/reporting implemented).")
    except Exception as e:
        print(f"[clean-outputs] skipped due to: {e}")
# ---------------------------------------------------------------------

def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(
        description="IKF Market Pipeline — default runs full analysis. Use --raw-only to skip analytics; use --composites to control composites."
    )
    ap.add_argument("--asset", choices=["all","equities","options","futures","rates"], default="all")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--raw-only", action="store_true", help="Only update raw data; skip features/ranking/report")
    ap.add_argument("--full", action="store_true", help="Full backfill for equities from history_start")
    ap.add_argument("--recent", type=int, default=None, help="Rebuild last N days for equities (e.g., 30)")
    ap.add_argument("--composites", default="all",
                    help="Build 'all' (default), 'none', or a comma-separated list of composite names")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    root = cfg["storage"]["root"]

    # equities updater config
    eq_cfg = (cfg.get("defaults", {}) | cfg.get("equities", {})).copy()
    if args.full:
        eq_cfg["mode"] = "full"
    if args.recent is not None:
        eq_cfg["mode"] = "recent"; eq_cfg["lookback_days"] = int(args.recent)

    # composites selection + auto-extend universe
    mode, pick = _parse_composite_selection(args.composites)
    all_comps = cfg.get("composites", []) or []
    selected_comps = [] if mode=="none" else ([c for c in all_comps if c.get("name") in pick] if mode=="some" else all_comps)
    if selected_comps:
        composite_members = {t.upper() for c in selected_comps for t in c.get("tickers", [])}
        composite_bench   = {str(c.get("benchmark")).upper() for c in selected_comps if c.get("benchmark")} - {None,"NONE","NULL"}
        base_universe = set(t.upper() for t in eq_cfg.get("universe", []))
        merged_universe = sorted(base_universe | composite_members | composite_bench)
        if merged_universe != sorted(base_universe):
            print(f"[INFO] Expanding equities.universe with composites ({len(merged_universe)} tickers).")
        eq_cfg["universe"] = merged_universe

    # raw updaters
    if args.asset in ("all","equities"): update_equities(eq_cfg, root)
    if args.asset in ("all","options"):  update_options(cfg.get("options", {}), root)
    if args.asset in ("all","futures"):  update_futures(cfg.get("futures", {}), root)
    if args.asset in ("all","rates"):    update_rates(cfg.get("rates", {}), root)

    # build composites from RAW (post raw updates)
    rank_today_df = None
    if selected_comps:
        built = build_composites_from_raw(selected_comps, root, rank_today=None,
                                          select_names=set(c.get("name") for c in selected_comps))
        if built: print("[OK] Composite price series written:", built)

    # analysis bundle (default ON unless --raw-only)
    if args.raw_only:
        return 0

    feats_conf = cfg.get("features", {}).get("equities", {})
    processed_dir = feats_conf.get("processed_path", "data/processed")
    feats_path = f"{processed_dir}/equity_features.parquet"

    # 1) features
    try:
        run_equity_features(feats_conf); print("[OK] Equity feature engineering complete.")
    except Exception as e:
        print(f"[ERR] Feature engineering failed: {e}"); return 2

    # 2) dedup
    finalize_equity_features_file(processed_dir=processed_dir, fname="equity_features.parquet")

    # 3) ranking
    try:
        feats = pd.read_parquet(feats_path)
    except FileNotFoundError:
        print(f"[ERR] Missing features parquet: {feats_path}"); return 2

    auto_cfg, chosen = build_metrics_cfg_from_df(feats)
    print("[INFO] Composite metrics selected:")
    for canon, actual in chosen.items(): print(f"   {canon:10s} -> {actual}")

    scores = compute_composite_scores(
        feats, date_col="date", id_col="ticker", sector_col="sector",
        metrics_cfg=auto_cfg, neutralize_vs=("beta_60d","ln_mcap"),
    )

    # 4) snapshot + history
    snap_path = "output/equity_rank_snapshot.csv"
    hist_path = "output/equity_rank_history.parquet"
    latest_dt = scores["date"].max()
    today_cs = scores[scores["date"] == latest_dt].copy()
    save_rank_snapshot(today_cs, snap_path)

    try:
        if Path(hist_path).exists():
            hist = pd.read_parquet(hist_path)
            hist = pd.concat([hist, today_cs]).drop_duplicates(subset=["date","ticker"], keep="last")
        else:
            hist = today_cs
        hist.to_parquet(hist_path)
        print(f"[OK] Appended to rank history → {hist_path}")
    except Exception as e:
        print(f"[WARN] Could not update rank history parquet: {e}")

    # 4b) composite member snapshots joined with today's ranks
    if selected_comps:
        rank_today_df = today_cs[[c for c in ("ticker","score","rank_pct","decile") if c in today_cs.columns]]
        build_composites_from_raw(selected_comps, root, rank_today=rank_today_df,
                                  select_names=set(c.get("name") for c in selected_comps))

    # 5) report
    try:
        make_equity_report(
            features_path=feats_path, ranks_path=snap_path,
            plots_dir="output/plots", out_md="output/reports/equities_report.md",
            top_k=25, bottom_k=25,
        )
    except Exception as e:
        print(f"[WARN] Report generation failed: {e}")

    # 6) optional clean outputs (no-op until modules are implemented)
    try:
        if cfg.get("report", {}):  # any truthy 'report' section toggles it on
            run_clean_outputs(cfg)
    except Exception as e:
        print(f"[WARN] Clean outputs step skipped: {e}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
