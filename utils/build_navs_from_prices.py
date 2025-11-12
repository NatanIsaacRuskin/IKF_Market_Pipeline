# utils/build_navs_from_prices.py
from __future__ import annotations
from pathlib import Path
import os, json, yaml, hashlib
from typing import Optional, Dict, Iterable
import pandas as pd

PANEL_PATH = Path("data/processed/equities_prices.parquet")
RAW_DIR    = Path("data/raw/equities")
BENCH_DIR  = Path("data/processed/benchmarks")
COMP_DIR   = Path("data/processed/composites")

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ---------- io + math helpers ----------
def _read_existing_series(path: Path) -> Optional[pd.Series]:
    if not path.exists(): return None
    s = pd.read_parquet(path).squeeze()
    s.index = pd.to_datetime(s.index)
    return s.sort_index()

def _continue_nav(existing: Optional[pd.Series], daily_ret: pd.Series) -> pd.Series:
    daily_ret = daily_ret.sort_index()
    if existing is None or existing.empty:
        nav = (1.0 + daily_ret.fillna(0.0)).cumprod() * 100.0
        nav.name = "NAV"
        return nav
    last = existing.index.max()
    tail = daily_ret[daily_ret.index > last]
    if tail.empty:  # nothing new
        return existing
    ext = (1.0 + tail.fillna(0.0)).cumprod() * float(existing.iloc[-1])
    ext.name = "NAV"
    return pd.concat([existing, ext])

def _eqw_daily_returns(px: pd.DataFrame) -> pd.Series:
    px = px.sort_index().ffill().dropna(how="all", axis=1)
    if px.shape[1] == 0:
        return pd.Series(dtype=float, name="ret")
    rets = px.pct_change()
    return rets.mean(axis=1, skipna=True).rename("ret").dropna()

def _fingerprint(d: Dict) -> str:
    blob = json.dumps(d, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]

# ---------- robust series reader ----------
_PREFER = ("Adj Close","AdjClose","adj_close","Close","close","Price","price","PX","px","Value","value","Last","last")

def _series_from_any_parquet(fp: Path, rename_to: Optional[str] = None) -> pd.Series:
    """Read parquet (Series/DF/MultiIndex), set a real datetime index, pick a price column,
    and return a single float Series named by the ticker."""
    obj = pd.read_parquet(fp)

    # If it's already a Series, try to fix index then return
    if isinstance(obj, pd.Series):
        s = obj
        idx = s.index
        # if there's a 'date' attribute/column, ignore for Series
    else:
        df = obj

        # 1) Ensure datetime index
        #    - If a 'date' column exists, use it
        #    - Else if index looks like epoch ints, parse with s/ms
        #    - Else try default to_datetime on index
        date_col = next((c for c in df.columns if str(c).lower() in ("date","datetime","asof")), None)
        if date_col is not None:
            idx = pd.to_datetime(df[date_col])
        else:
            # index might be ints; try epoch seconds then millis
            try:
                idx_try = pd.to_datetime(df.index)
                if idx_try.dtype.kind == "M":
                    idx = idx_try
                else:
                    raise ValueError
            except Exception:
                # if integer-like, try epoch s then ms
                try:
                    idx = pd.to_datetime(df.index.astype("int64"), unit="s")
                except Exception:
                    idx = pd.to_datetime(df.index.astype("int64"), unit="ms")

        df = df.copy()
        df.index = idx

        # 2) choose price column
        prefer = ("Adj Close","AdjClose","adj_close","Close","close",
                  "Price","price","PX","px","Value","value","Last","last")
        if isinstance(df.columns, pd.MultiIndex):
            target = None
            for want in prefer:
                cand = [c for c in df.columns
                        if any(str(lv).strip().lower() == want.lower()
                               for lv in (c if isinstance(c, tuple) else (c,)))]
                if cand:
                    target = cand[0]; break
            if target is None:
                # fallback to first numeric subcolumn
                for c in df.columns:
                    if pd.api.types.is_numeric_dtype(df[c]): target = c; break
            s = df[target]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:,0]
        else:
            col = next((c for c in prefer if c in df.columns), None)
            if col is None:
                nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                col = nums[0] if nums else df.columns[0]
            s = df[col]

    # 3) final clean + naming
    s = s.squeeze()
    s = s.astype(float)
    s.index = pd.to_datetime(s.index).sort_values()
    s = s[~s.index.duplicated(keep="last")]
    s.name = rename_to or (s.name if s.name else fp.stem.upper())
    return s


# ---------- price loading paths ----------
def _load_prices_panel() -> Optional[pd.DataFrame]:
    """
    Prefer combined processed panel; if missing/empty, merge per-ticker raws,
    picking a sensible price column per file and renaming it to the TICKER.
    """
    if PANEL_PATH.exists():
        df = pd.read_parquet(PANEL_PATH)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        if df.shape[1] > 0:
            return df

    if not RAW_DIR.exists():
        return None

    frames = []
    for fp in sorted(RAW_DIR.glob("*.parquet")):
        try:
            tkr = fp.stem.upper()
            s = _series_from_any_parquet(fp, rename_to=tkr)
            frames.append(s)
        except Exception as e:
            print(f"[NAV] warn: could not read {fp.name}: {e}")
    if not frames:
        return None
    return pd.concat(frames, axis=1).sort_index()

def _normalize_to_available(tickers: Iterable[str], available: set[str]) -> list[str]:
    """Map composite tickers to available columns (case/dot/dash tolerant)."""
    out = []
    for t in tickers:
        u = t.upper().replace(" ", "")
        candidates = [u, u.replace(".", "-"), u.replace("-", "."), u.replace("/", "-")]
        hit = next((c for c in candidates if c in available), None)
        if hit: out.append(hit)
    return sorted(set(out))

# ---------- main ----------
def main(force_full: bool = False):
    cfg = load_config()
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    COMP_DIR.mkdir(parents=True, exist_ok=True)

    # 0) Load a price panel (SPY + members). If absent, we’ll still handle composites
    #    directly from precomputed *_prices.parquet files under data/processed/composites.
    px = _load_prices_panel()
    avail = set(map(str.upper, px.columns.astype(str))) if (px is not None and not px.empty) else set()

    # 1) SPY benchmark NAV (never part of composites)
    spy_out = BENCH_DIR / "spy_nav.parquet"
    try:
        if "SPY" in avail:
            spy_ret = px["SPY"].pct_change()
        else:
            spy_fp = RAW_DIR / "SPY.parquet"
            if spy_fp.exists():
                spy_s = _series_from_any_parquet(spy_fp, rename_to="SPY")
                spy_ret = spy_s.pct_change()
            else:
                spy_ret = None
        if spy_ret is not None:
            existing_spy = None if force_full else _read_existing_series(spy_out)
            spy_nav = _continue_nav(existing_spy, spy_ret)
            if existing_spy is None or spy_nav.index.max() != existing_spy.index.max():
                spy_nav.to_frame(name="NAV").to_parquet(spy_out)
                print(f"[OK] SPY NAV → {spy_out} (thru {spy_nav.index.max().date()})")
        else:
            print("[NAV] warn: no SPY price series found; benchmark not written.")
    except Exception as e:
        print(f"[NAV] warn: failed to build SPY NAV: {e}")

    # 2) Composites (each becomes ONE line). Prefer precomputed composite price files.
    composites = cfg.get("composites", []) or []
    for comp in composites:
        name = comp.get("name")
        if not name:
            continue

        # Prefer a precomputed composite prices parquet if present
        comp_price_fp = COMP_DIR / f"{name}_prices.parquet"
        if comp_price_fp.exists():
            try:
                s = _series_from_any_parquet(comp_price_fp, rename_to=name)
                port_ret = s.pct_change().dropna()
                src = "prices-file"
            except Exception as e:
                print(f"[NAV] warn: {name} prices file unreadable ({e}); will try member merge")
                port_ret = None
        else:
            port_ret = None

        # If no direct composite prices, merge from members via panel/raws
        if port_ret is None:
            raw_members = [t for t in (comp.get("tickers") or [])]
            benchmark   = str(comp.get("benchmark", "SPY") or "SPY").upper()
            members = _normalize_to_available(raw_members, avail)
            members = [t for t in members if t not in {"SPY", benchmark}]

            if not members or px is None or px.empty:
                print(f"[NAV] skip {name}: no valid members or prices panel not available.")
                continue

            sub = px[members].dropna(how="all")
            port_ret = _eqw_daily_returns(sub)
            src = "members-panel"

        out_nav  = COMP_DIR / f"{name}_nav.parquet"
        out_meta = COMP_DIR / f"{name}_nav.meta.json"

        info = {
            "source": src,
            "benchmark": str(comp.get("benchmark", "SPY") or "SPY").upper(),
            "weights": comp.get("weights", "equal"),
            "rebalance": comp.get("rebalance", "none"),
        }
        if src == "members-panel":
            info["tickers"] = sorted(sub.columns.tolist())
        fp = _fingerprint(info)

        stored = {}
        if out_meta.exists():
            try: stored = json.loads(out_meta.read_text())
            except Exception: stored = {}
        stored_fp = stored.get("fingerprint")
        full_rebuild = force_full or (stored_fp != fp)
        if stored_fp and stored_fp != fp:
            print(f"[NAV] {name}: definition changed → full rebuild")

        existing = None if full_rebuild else _read_existing_series(out_nav)
        nav = _continue_nav(existing, port_ret)

        if existing is None or nav.index.max() != existing.index.max() or full_rebuild:
            nav.to_frame(name="NAV").to_parquet(out_nav)
            out_meta.write_text(json.dumps({"fingerprint": fp, "asof": str(nav.index.max().date())}, indent=2))
            print(f"[OK] {name} NAV → {out_nav} (thru {nav.index.max().date()})")

if __name__ == "__main__":
    main(force_full=bool(os.getenv("IKF_FORCE_FULL_NAV")))
