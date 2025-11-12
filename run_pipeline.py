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
from modules.perf import (  # noqa: F401
    build_composite_nav, perf_table, drawdown_series, rolling_sharpe
)
from modules.plotting import (  # noqa: F401
    load_plot_cfg, plot_cum_return, plot_rolling_sharpe, plot_drawdown, plot_risk_return, plot_corr_heatmap
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

def run_clean_outputs(cfg: dict) -> None:
    """
    Build SPY (grey) vs IKF composites (green) plots and a markdown report.
    Uses NAVs from data/processed/{benchmarks,composites} and honors
    report.ikf_windows in config for time-window charts.
    Never raises.
    """
    try:
        from glob import glob
        from pathlib import Path
        import pandas as pd
        from modules.plotting import (
            load_plot_cfg, plot_cum_return, plot_rolling_sharpe,
            plot_drawdown, plot_risk_return, plot_corr_heatmap
        )
        from modules.reporting import write_equities_report
        from modules.perf import perf_table

        # --- paths ---
        paths = cfg.get("paths", {})
        plots_dir = Path(paths.get("plots", "output/plots"));       plots_dir.mkdir(parents=True, exist_ok=True)
        reports_dir = Path(paths.get("reports", "output/reports")); reports_dir.mkdir(parents=True, exist_ok=True)
        snapshots_dir = Path(paths.get("snapshots", "output/snapshots")); snapshots_dir.mkdir(parents=True, exist_ok=True)
        processed_dir = Path(paths.get("processed", "data/processed"))
        (processed_dir / "composites").mkdir(parents=True, exist_ok=True)
        composites_dir = processed_dir / "composites"

        # --- load composite NAVs ---
        comp_navs: dict[str, pd.Series] = {}
        for fp in glob(str(composites_dir / "*_nav.parquet")):
            name = Path(fp).stem.replace("_nav", "")
            s = pd.read_parquet(fp).squeeze()
            s.index = pd.to_datetime(s.index)
            comp_navs[name] = s.sort_index()

        # --- load/derive SPY NAV (benchmark only) ---
        spy_nav = None
        for cand in (processed_dir / "benchmarks" / "spy_nav.parquet",
                     processed_dir / "SPY_nav.parquet"):
            if Path(cand).exists():
                spy_nav = pd.read_parquet(cand).squeeze()
                spy_nav.index = pd.to_datetime(spy_nav.index)
                spy_nav = spy_nav.sort_index()
                break

        if spy_nav is None:
            # best-effort fallback from any prices panel if configured
            try:
                root = cfg["storage"]["root"]
                px_path = Path(root) / "equities" / "prices.parquet"
                if px_path.exists():
                    df = pd.read_parquet(px_path)
                    if "SPY" in df.columns:
                        ps = df["SPY"].dropna()
                        spy_nav = (1.0 + ps.pct_change().fillna(0.0)).cumprod() * 100.0
                        spy_nav.name = "SPY"
                        (processed_dir / "benchmarks").mkdir(parents=True, exist_ok=True)
                        spy_nav.to_frame(name="NAV").to_parquet(processed_dir / "benchmarks" / "spy_nav.parquet")
            except Exception:
                pass

        if spy_nav is None or not comp_navs:
            print("[clean-outputs] Missing SPY or composite NAVs — skipping charts.")
            return

        # --- returns & helpers ---
        spy_ret = spy_nav.pct_change().dropna()
        comp_returns = {k: v.pct_change().dropna() for k, v in comp_navs.items()}
        returns_df = pd.concat([spy_ret.rename("SPY")] + [r.rename(k) for k, r in comp_returns.items()], axis=1).dropna(how="all")

        # --- plotting: full period + IKF windows (config override) ---
        plt_cfg = load_plot_cfg(cfg)

        # full period
        plot_cum_return(spy_nav, comp_navs, (plots_dir / "equities_cum_return.png").as_posix(), plt_cfg)
        plot_drawdown(spy_nav, comp_navs, (plots_dir / "equities_drawdown.png").as_posix(), plt_cfg)
        plot_risk_return(spy_ret, comp_returns, (plots_dir / "equities_risk_return.png").as_posix(), plt_cfg)
        plot_corr_heatmap(returns_df, (plots_dir / "equities_corr_heatmap.png").as_posix(), plt_cfg)

        rs_win = int(cfg.get("backtest", {}).get("rolling_sharpe_window_days", 60))
        if len(spy_ret) >= rs_win:
            plot_rolling_sharpe(spy_ret, comp_returns, rs_win, (plots_dir / "equities_rolling_sharpe.png").as_posix(), plt_cfg)

        # IKF windows — read from config if present
        asof = spy_nav.index.max()
        _map = {
            "3d": pd.Timedelta(days=3),
            "7d": pd.Timedelta(days=7),
            "14d": pd.Timedelta(days=14),
            "1m": pd.DateOffset(months=1),
            "3m": pd.DateOffset(months=3),
            "1y": pd.DateOffset(years=1),
        }
        cfg_w = (cfg.get("report", {}) or {}).get("ikf_windows")
        if cfg_w:
            windows = [(w, _map[w]) for w in cfg_w if w in _map]
        else:
            windows = [("3d", _map["3d"]), ("7d", _map["7d"]), ("14d", _map["14d"]),
                       ("1m", _map["1m"]), ("3m", _map["3m"]), ("1y", _map["1y"])]

        def _slice_navs(start_dt):
            s_spy = spy_nav[spy_nav.index >= start_dt]
            s_comp = {k: v[v.index >= start_dt] for k, v in comp_navs.items()}
            r_spy = s_spy.pct_change().dropna()
            r_comp = {k: s.pct_change().dropna() for k, s in s_comp.items()}
            rr_df = pd.concat([r_spy.rename("SPY")] + [r.rename(k) for k, r in r_comp.items()], axis=1).dropna(how="all")
            return s_spy, s_comp, r_spy, r_comp, rr_df

        for label, delta in windows:
            start = (asof - delta) if isinstance(delta, pd.Timedelta) else asof - delta
            s_spy, s_comp, r_spy, r_comp, rr_df = _slice_navs(start)
            if len(s_spy) >= 2 and any(len(s) >= 2 for s in s_comp.values()):
                plot_cum_return(s_spy, s_comp, (plots_dir / f"equities_cum_return_{label}.png").as_posix(), plt_cfg)
                plot_drawdown(s_spy, s_comp, (plots_dir / f"equities_drawdown_{label}.png").as_posix(), plt_cfg)
                if len(r_spy) >= 5:
                    plot_risk_return(r_spy, r_comp, (plots_dir / f"equities_risk_return_{label}.png").as_posix(), plt_cfg)
                if len(r_spy) >= rs_win:
                    plot_rolling_sharpe(r_spy, r_comp, rs_win, (plots_dir / f"equities_rolling_sharpe_{label}.png").as_posix(), plt_cfg)

        window_labels = [w[0] for w in windows]

        # --- scorecard ---
        rows = []
        for name, nav in comp_navs.items():
            row = perf_table(nav, spy_nav, asof)
            row.insert(0, "composite", name)
            rows.append(row)
        scorecard = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if not scorecard.empty:
            scorecard = scorecard.rename(columns={
                "ret_ytd": "YTD", "ret_1y": "1Y", "ret_max": "Max",
                "vol_ann": "Vol (ann)", "sharpe": "Sharpe",
                "max_dd": "Max DD", "calmar": "Calmar",
            })
            order = ["composite", "YTD", "1Y", "Max", "Vol (ann)", "Sharpe", "Max DD", "Calmar"]
            scorecard = scorecard[[c for c in order if c in scorecard.columns]]

        # --- leaders/laggards ---
        leaders = laggards = None
        try:
            snap = pd.read_csv("output/equity_rank_snapshot.csv")
            cols = [c for c in ("ticker","score","rank_pct","decile") if c in snap.columns]
            if cols:
                snap = snap[cols].copy().sort_values("score", ascending=False)
                top_n = int(cfg.get("report", {}).get("top_n", 10))
                leaders = snap.head(top_n).reset_index(drop=True)
                laggards = snap.tail(top_n).iloc[::-1].reset_index(drop=True)
        except Exception:
            pass

        # --- write report ---
        include_sections = cfg.get("report", {}).get(
            "include_sections",
            ["cum_return", "rolling_sharpe", "drawdown", "risk_return", "corr_heatmap"],
        )
        write_equities_report(
            out_md=reports_dir / "equities_report.md",
            asof=str(asof.date()),
            scorecard=scorecard,
            leaders=leaders,
            laggards=laggards,
            plots_dir=plots_dir,
            include_sections=include_sections,
            time_windows=window_labels,
        )
        print("[clean-outputs] Report + plots written.")
    except Exception as e:
        print(f"[clean-outputs] skipped due to: {e}")

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

    # 5b) auto-build/extend SPY + composite NAVs (incremental; cheap)
    try:
        import os
        from utils.build_navs_from_prices import main as _build_navs
        force_full = bool(os.getenv("IKF_FORCE_FULL_NAV"))
        _build_navs(force_full=force_full)
    except Exception as e:
        print(f"[WARN] NAV build skipped: {e}")

    # 6) optional clean outputs (no-op until modules are implemented)
    try:
        if cfg.get("report", {}):  # any truthy 'report' section toggles it on
            run_clean_outputs(cfg)
    except Exception as e:
        print(f"[WARN] Clean outputs step skipped: {e}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
