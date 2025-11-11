# run_pipeline.py
from __future__ import annotations

import argparse
import yaml
import pandas as pd
import numpy as np

from modules.equities import update_equities
from modules.options import update_options
from modules.futures import update_futures
from modules.rates import update_rates
from modules.features import run_equity_features
from modules.features_dedup import finalize_equity_features_file
from modules.ranking import compute_composite_scores, save_rank_snapshot, build_metrics_cfg_from_df
from modules.report_equities import make_equity_report

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser(description="IKF Universal Market Pipeline")

    ap.add_argument("--asset", choices=["all","equities","options","futures","rates"], default="all",
                    help="which asset updater(s) to run")
    ap.add_argument("--features", action="store_true",
                    help="run equity feature engineering")
    ap.add_argument("--rank", action="store_true",
                    help="compute cross-sectional composite ranks")
    ap.add_argument("--report", action="store_true",
                    help="generate Markdown equities report")

    ap.add_argument("--config", default="config/config.yaml",
                    help="path to YAML config")

    # one-off backfill controls for equities
    ap.add_argument("--full", action="store_true",
                    help="force full backfill for equities (start from history_start)")
    ap.add_argument("--recent", type=int, default=None,
                    help="force recent backfill for equities using N lookback days")

    args = ap.parse_args()
    cfg = load_config(args.config)

    root = cfg["storage"]["root"]

    # ---- equities updater config (merge defaults + equities block) ----
    eq_cfg = (cfg.get("defaults", {}) | cfg.get("equities", {})).copy()
    if args.full:
        eq_cfg["mode"] = "full"
    if args.recent is not None:
        eq_cfg["mode"] = "recent"
        eq_cfg["lookback_days"] = args.recent

    # ---- run updaters ----
    if args.asset in ("all", "equities"):
        update_equities(eq_cfg, root)
    if args.asset in ("all", "options"):
        update_options(cfg.get("options", {}), root)
    if args.asset in ("all", "futures"):
        update_futures(cfg.get("futures", {}), root)
    if args.asset in ("all", "rates"):
        update_rates(cfg.get("rates", {}), root)

    # ---- features (equities) ----
    feats_conf = cfg.get("features", {}).get("equities", {})
    processed_dir = feats_conf.get("processed_path", "data/processed")
    feats_path = f"{processed_dir}/equity_features.parquet"

    if args.features:
        run_equity_features(feats_conf)
        print("[OK] Equity feature engineering complete.")
        # EARLY DEDUP: enforce unique (date,ticker) before anything downstream
        finalize_equity_features_file(processed_dir=processed_dir, fname="equity_features.parquet")

    # ---- ranking + report ----
    if args.rank or args.report:
        # Guard: ensure features parquet exists and is deduped even if user didn't run --features in this session
        finalize_equity_features_file(processed_dir=processed_dir, fname="equity_features.parquet")

        feats = pd.read_parquet(feats_path)

        # Auto-detect available metrics from the features parquet (robust to schema)
        auto_cfg, chosen = build_metrics_cfg_from_df(feats)
        print("[INFO] Composite metrics selected:")
        for canon, actual in chosen.items():
            print(f"   {canon:10s} -> {actual}")

        scores = compute_composite_scores(
            feats,
            date_col="date",
            id_col="ticker",
            sector_col="sector",
            metrics_cfg=auto_cfg,                # use auto-detected config
            neutralize_vs=("beta_60d", "ln_mcap"),
        )
        snap_path = "output/equity_rank_snapshot.csv"
        save_rank_snapshot(scores, snap_path)

        if args.report:
            make_equity_report(
                features_path=feats_path,
                ranks_path=snap_path,
                plots_dir=feats_conf.get("plots_path", "output/plots"),
                out_md="output/reports/equities_report.md",
            )
            print("[OK] Equities report written.")

if __name__ == "__main__":
    main()
