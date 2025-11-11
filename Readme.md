# IKF Market Pipeline

A universal market data and analytics pipeline for report generation
Fetches and updates raw market data, engineers features, computes rankings, and generates daily reports.

---

## 🚀 Quick Start

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

### Run the Pipeline

• **Default (full analysis)**  
Fetches data, builds features, computes rankings, and writes a Markdown report.  
`python run_pipeline.py`

• **Data-only mode**  
Skip analytics and just update raw data.  
`python run_pipeline.py --raw-only`

• **Optional flags**  
--recent N → rebuild last N days (e.g. `--recent 30`)  
--full   → full backfill from history_start  
--asset X → run a single asset updater (equities, futures, rates, options)  
--config Y → custom config path (default `config/config.yaml`)

---

## ⚙️ Output

data/raw/    incrementally updated market data  
data/processed/equity_features.parquet engineered features  
output/equity_rank_snapshot.csv    latest composite rankings  
output/reports/equities_report.md   markdown report summary  

---

## 🧩 Key Features

• Incremental daily updates with overlap healing  
• Feature engineering: momentum, volatility, RSI, SMA/EMA, beta, etc.  
• Cross-sectional ranking and composite scoring  
• Automated reporting and persistent rank history  

---

## 🕒 Example Cron (Linux)

# Run every weekday at 07:30 Israel time  
TZ=Asia/Jerusalem  
30 7 * * 1-5 /usr/bin/env bash -lc 'cd /path/to/IKF_Market_Pipeline && python run_pipeline.py >> logs/daily.log 2>&1'

---

Maintained as part of the **I Know First Market Intelligence Pipeline**.
