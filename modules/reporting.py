# modules/reporting.py
from pathlib import Path
from typing import List, Dict
import pandas as pd

def write_equities_report(
    out_md: Path,
    asof: str,
    scorecard: pd.DataFrame,
    leaders: pd.DataFrame,
    laggards: pd.DataFrame,
    plots_dir: Path,
    include_sections: List[str],
):
    """Render markdown with conditional plot embeds (only if files exist)."""
    pass
