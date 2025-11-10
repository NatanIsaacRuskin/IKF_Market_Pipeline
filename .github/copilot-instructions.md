# Copilot Instructions for IKF Market Pipeline

## Project Overview
- **Purpose:** Universal market data pipeline for equities, options, futures, and rates. Modular, extensible, and designed for robust daily data ingestion and feature engineering.
- **Key Entry Point:** `run_pipeline.py` orchestrates all asset updaters and feature engineering. Use CLI flags to control which assets and features to update.

## Architecture & Data Flow
- **Config-driven:** All behavior is controlled by `config/config.yaml` (assets, universe, modes, feature params, storage paths).
- **Modules:**
  - `modules/equities.py`, `modules/options.py`, `modules/futures.py`, `modules/rates.py`: Each handles fetching, updating, and storing data for its asset class. All use helper functions for Parquet I/O and incremental updates.
  - `modules/features.py`: Computes equity features, cross-sectional ranks, and generates plots. Reads raw equity data, writes processed features and plots.
  - `utils/helpers.py`: Shared utilities for file I/O, directory management, and incremental Parquet appends.
- **Data Storage:**
  - Raw data: `data/raw/<asset>/`
  - Processed features: `data/processed/`
  - Outputs (plots, reports): `output/`

## Developer Workflows
- **Setup:**
  - Create a virtual environment and install dependencies:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
- **Run Pipeline:**
  - Update all assets and features:
    ```bash
    python run_pipeline.py --asset all --features
    ```
  - Update only equities, with full backfill:
    ```bash
    python run_pipeline.py --asset equities --full
    ```
  - See all CLI options in `run_pipeline.py`.
- **Quick Data Check:**
  - Use `quick_check.py` to print row counts and date ranges for all Parquet files.

## Project Conventions & Patterns
- **Incremental Updates:** All asset updaters append new data to existing Parquet files, minimizing redundant downloads.
- **Gentle Rate Limiting:** Configurable retry and sleep parameters for API calls (see `config.yaml`).
- **Feature Engineering:** Feature computation is modular and config-driven. Plots and cross-sectional ranks are auto-generated.
- **No Hardcoded Paths:** All file paths are relative and configurable.
- **Extensibility:** To add a new asset class, create a new module in `modules/`, update `run_pipeline.py`, and extend `config.yaml`.

## Integration & Dependencies
- **External APIs:**
  - Equities, options, futures: Yahoo Finance via `yfinance`
  - Rates: FRED via `pandas-datareader`
- **Data Format:** All data is stored as Parquet for efficiency and compatibility.

## Examples
- To add a new equity to the universe, edit `config/config.yaml` under `equities: universe:`.
- To change feature windows or add new features, update the `features:` block in `config.yaml` and/or extend `modules/features.py`.

## Key Files & Directories
- `run_pipeline.py` — main orchestrator
- `config/config.yaml` — all pipeline settings
- `modules/` — asset updaters and feature engineering
- `data/`, `output/` — data and results
- `quick_check.py` — data inspection utility

---
For questions about conventions or unclear patterns, review `run_pipeline.py`, `config/config.yaml`, and the relevant module in `modules/`.
