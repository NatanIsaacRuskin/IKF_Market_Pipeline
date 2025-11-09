import argparse, yaml
from modules.equities import update_equities
from modules.options  import update_options
from modules.futures  import update_futures
from modules.rates    import update_rates
from modules.features import run_equity_features

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser(description="IKF Universal Market Pipeline")

    # what to run
    ap.add_argument("--asset",
                    choices=["all","equities","options","futures","rates"],
                    default="all",
                    help="which asset updater(s) to run")
    ap.add_argument("--features",
                    action="store_true",
                    help="run equity feature engineering after updates")

    # config path
    ap.add_argument("--config",
                    default="config/config.yaml",
                    help="path to YAML config")

    # NEW: one-off backfill controls for equities
    ap.add_argument("--full",
                    action="store_true",
                    help="force full backfill for equities (start from history_start)")
    ap.add_argument("--recent",
                    type=int, default=None,
                    help="force recent backfill for equities using N lookback days (ignores on-disk)")

    args = ap.parse_args()

    cfg  = load_config(args.config)
    root = cfg["storage"]["root"]

    # ---- equities updater (merge defaults + equities block) ----
    eq_cfg = (cfg.get("defaults", {}) | cfg.get("equities", {})).copy()

    # apply CLI overrides (these override config.yaml for this run only)
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
    if args.features:
        eq_feat_cfg = cfg.get("features", {}).get("equities", {})
        run_equity_features(eq_feat_cfg)
        print("[OK] Equity feature engineering complete.")

if __name__ == "__main__":
    main()
