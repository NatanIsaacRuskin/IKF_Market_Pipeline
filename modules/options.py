import pandas as pd
import yfinance as yf
from utils.helpers import parquet_path, incremental_append

def _nearest_expiries(tkr: yf.Ticker, n: int = 3) -> list[str]:
    exps = tkr.options or []
    return exps[:n]

def update_options(cfg: dict, storage_root: str):
    if not cfg.get("enabled", False):
        return
    underlyings = cfg.get("underlying", [])
    n = int(cfg.get("expiries","nearest_3").split("_")[-1])
    chains = cfg.get("chains", ["calls","puts"])

    for u in underlyings:
        tk = yf.Ticker(u)
        expiries = _nearest_expiries(tk, n)
        for exp in expiries:
            chain = tk.option_chain(exp)
            for side in chains:
                df: pd.DataFrame = getattr(chain, side, None)
                if df is None or df.empty:
                    continue
                df["underlying"] = u
                df["expiry"] = exp
                df.set_index(["expiry","contractSymbol"], inplace=True)
                path = parquet_path(storage_root, "options", u, f"{side}_{exp}")
                incremental_append(df, path)
