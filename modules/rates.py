import pandas as pd
from pandas_datareader import data as pdr
from utils.helpers import parquet_path, incremental_append, load_parquet

def update_rates(cfg: dict, storage_root: str):
    if not cfg.get("enabled", False):
        return
    series = cfg.get("fred_series", [])
    start = cfg.get("start","2010-01-01")
    for s in series:
        path = parquet_path(storage_root, "rates", s)
        exist = load_parquet(path)
        sdate = (exist.index.max() + pd.Timedelta(days=1)).date().isoformat() if not exist.empty else start
        df = pdr.DataReader(s, "fred", sdate)   # index=Date, col=series
        if df.empty: 
            continue
        incremental_append(df, path, index_name="Date")
