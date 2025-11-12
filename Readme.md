# IKF Market Pipeline

A universal market data and analytics pipeline for report generation.  
Fetches and updates raw market data, engineers features, computes rankings, builds composite NAVs, and generates daily reports and plots.

---

## 🚀 Quick Start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Pipeline

- **Default (full analysis)**  
  Fetches data, builds features, computes rankings, builds NAVs, and writes a Markdown report.  
  ```bash
  python run_pipeline.py
  ```

- **Data-only mode**  
  Skip analytics and just update raw data (no features/ranking/report).  
  ```bash
  python run_pipeline.py --raw-only
  ```

- **Optional flags**
  - `--recent N` → rebuild last N days (e.g., `--recent 30`)
  - `--full` → full backfill from `history_start`
  - `--asset X` → run a single asset updater (`equities`, `futures`, `rates`, `options`)
  - `--config Y` → custom config path (default `config/config.yaml`)
  - `--composites <all|none|CSV>` → control which composites to build and auto-include in universe

---

## ⚙️ Output

- `data/raw/` — incrementally updated market data  
- `data/processed/equity_features.parquet` — engineered features  
- `data/processed/composites/<NAME>_prices.parquet` — per-composite price panels  
- `data/processed/composites/<NAME>_nav.parquet` — per-composite NAV series (rebased to 100)  
- `data/processed/benchmarks/spy_nav.parquet` — SPY benchmark NAV (grey)  
- `output/equity_rank_snapshot.csv` — latest composite rankings (cross-sectional)  
- `output/equity_rank_history.parquet` — persistent rank history  
- `output/plots/*.png` — charts (full period + IKF windows)  
- `output/reports/equities_report.md` — markdown report summary

> **Branding / colors:** SPY is **grey** (`#9E9E9E`). IKF composites are **green** (`#1BAA4A`).  
> Charts show **two lines**: SPY (benchmark) vs each composite NAV.

---

## 🧩 Key Features

- **Incremental daily updates** with overlap healing  
- **Feature engineering**: momentum, volatility, RSI, SMAs/EMAs, beta, size, etc.  
- **Cross-sectional ranking & composite scoring** (z-score based, neutralization options)  
- **Composite NAV builder** from component prices (equal weight by default; config-driven)  
- **Automated reporting** with full-period visuals **+ IKF time windows** (3d/7d/14d/1m/3m/1y)  
- **Clean outputs orchestrator** that renders plots and a Markdown report

---

## 📈 IKF Windowed Reporting

The pipeline now renders **six windowed charts** matching IKF forecast horizons:

- `3d`, `7d`, `14d`, `1m`, `3m`, `1y`

For each window you get:
- **Cumulative Return** (SPY grey vs composites green)
- **Drawdown**
- **Risk vs Return**
- **Rolling Sharpe** (if enough history for the chosen RS window)

These images are saved to `output/plots/` with window suffixes, e.g.:
```
equities_cum_return_3d.png
equities_cum_return_7d.png
...
equities_drawdown_1y.png
```
and are included in the Markdown report under **“Time-Window Snapshots (IKF)”**.

---

## 🔧 Configuration

Everything important lives in `config/config.yaml`.

### Plot look & feel
```yaml
plot:
  dpi: 140
  colors:
    spy: "#9E9E9E"      # grey benchmark
    ikf_green: "#1BAA4A" # IKF green
  marker:
    size_normal: 60
    size_highlight: 110
  line:
    width_normal: 2.0
    width_highlight: 3.0
```

### Report sections & IKF windows
```yaml
report:
  top_n: 10
  include_sections:
    - leaders_laggards
    - cum_return
    - rolling_sharpe
    - drawdown
    - risk_return
    - corr_heatmap
  ikf_windows: ["3d","7d","14d","1m","3m","1y"]  # <-- override these anytime
```

### Backtest / analytics knobs
```yaml
backtest:
  rolling_sharpe_window_days: 60
  rebal_freq: "monthly"   # none|monthly|quarterly
```

### Paths
```yaml
paths:
  plots: "output/plots"
  reports: "output/reports"
  snapshots: "output/snapshots"
  processed: "data/processed"
```

---

## 🧱 Composites

Define composites in `config.yaml` under `composites:` (name + member tickers, optional benchmark).  
When you run the pipeline:
1. The equities universe auto-expands to include composite members & benchmarks.  
2. Per-composite price panels → `data/processed/composites/<NAME>_prices.parquet`.  
3. NAV series → `data/processed/composites/<NAME>_nav.parquet`.  
4. Charts compare each composite vs SPY (grey).

Force a full recompute of NAVs (e.g. after changing members):
```bash
IKF_FORCE_FULL_NAV=1 python run_pipeline.py
```

---

## 🗂️ Repo Hygiene

We ignore all generated artifacts to keep the repo lean:

```
data/**
output/**
*.parquet
*.meta.json
```

> Only source, configs, and docs are versioned; all outputs are reproducible locally.

---

## 🧪 Sanity Checks

- **Active windows:** check `report.ikf_windows` in `config.yaml` — each should have plots in `output/plots/`.  
- **SPY not found:** we derive it from prices automatically and write to `data/processed/benchmarks/spy_nav.parquet`.  
- **Composite starts at 1970:** means parquet index wasn’t datetime; force rebuild:
  ```bash
  IKF_FORCE_FULL_NAV=1 python run_pipeline.py
  ```

---

## 🕒 Example Cron (Linux)

Run every weekday at 07:30 Israel time:
```cron
# /etc/cron.d/ikf
TZ=Asia/Jerusalem
30 7 * * 1-5  /usr/bin/env bash -lc 'cd /path/to/IKF_Market_Pipeline && python run_pipeline.py >> logs/daily.log 2>&1'
```

---

## 🧾 Example Composites (config.yaml)
```yaml
composites:
  - name: "IKF_AI_Megacap"
    benchmark: "SPY"
    tickers: ["AAPL", "MSFT", "GOOGL", "NVDA", "META"]

  - name: "IKF_EnergyPulse"
    benchmark: "XLE"
    tickers: ["XOM", "CVX", "COP", "EOG", "PSX"]
```

---

Maintained as part of the **I Know First Market Intelligence Pipeline**.
