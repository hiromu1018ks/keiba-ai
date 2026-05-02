# External Integrations

**Analysis Date:** 2026-05-02

## APIs & External Services

**Slack Notifications:**
- Service: Slack Incoming Webhook
- Purpose: Paper trading bet recommendations, daily result summaries, alerts
- SDK/Client: `urllib.request` (stdlib, no third-party HTTP client)
- Auth: Webhook URL passed at construction (`src/monitoring/notifier.py`)
- Classes: `SlackNotifier` (send, send_prediction, send_daily_result)

## Data Sources

**EveryDB2 / JRA-VAN DataLab:**
- Provider: JRA-VAN DataLab via PostgreSQL external tables
- Purpose: Horse racing historical data, race entries, odds, payouts
- Connection: PostgreSQL at `localhost:5432/everydb2`
- Auth: `PGPASSWORD` environment variable (overrides `config/settings.yaml` empty default)
- Client: `psycopg2` for direct queries (`src/db/everydb2_queries.py`), `sqlalchemy` for ETL (`src/db/connection.py`)
- Tables: 103 tables -- n_ prefix (53 raced/master) + s_ prefix (50 delta)
- ETL config: `config/etl_tables.yaml`

## Data Storage

### Databases

**PostgreSQL (EveryDB2):**
- Purpose: Source-of-truth for raw racing data (ETL input only)
- Connection: `localhost:5432/everydb2`, user `postgres`
- Connection string: Built from `config/settings.yaml` with `PGPASSWORD` env var override
- Client: `sqlalchemy.create_engine` with `NullPool` (`src/db/connection.py`)
- Direct query: `psycopg2.connect` in `src/db/everydb2_queries.py`
- Schema definitions: `src/db/schema.py` (5 schemas: raw, odds_history, feature, prediction, betting)
- Read-only access to EveryDB2 external tables (n_race, n_uma_race, n_harai, etc.)

### Parquet File Storage (Primary Data Layer)

**Location:** `data/` directory (project root)

**Raw Data (`data/raw/`):**
- `races.parquet` - Race metadata
- `entries.parquet` - Horse entries (n_uma_race)
- `payouts.parquet` - Payout data (n_harai)
- `horses.parquet` - Horse master data
- `chokyo_seiseki.parquet` - Training results
- `kisyu_seiseki.parquet` - Jockey results
- `horse_career_stats.parquet` - Precomputed career stats
- `jockey_stats.parquet` - Precomputed jockey stats
- `sire_career_stats.parquet` - Precomputed sire stats
- `trainer_stats.parquet` - Precomputed trainer stats

**Odds Data (`data/odds/`):**
- `snapshots.parquet` - Odds snapshots
- `wide.parquet` - Wide (quinella place) odds
- `time_series/` - Year-partitioned time-series odds (year=2015 through year=2024)
- `odds_tanpuku.parquet` - Win/place odds
- `odds_wide.parquet` - Wide odds
- `odds_waku.parquet` - Bracket odds
- `jodds_tanpuku.parquet` - Time-series win/place odds (partitioned)
- `jodds_umaren.parquet` - Time-series exacta odds
- Plus head/summary variants

**Model Artifacts (`data/models/`):**
- `stage1_turf.lgb`, `stage1_dirt.lgb` - LightGBM Stage1 ability models
- `ev_corrector_p_turf.lgb`, `ev_corrector_p_dirt.lgb` - P-correction models
- `ev_corrector_e_turf.lgb`, `ev_corrector_e_dirt.lgb` - E-correction models
- `market_turf.lgb`, `market_dirt.lgb` - Market models
- `regime_detector.lgb` - Regime classification model
- `race_quality.lgb` - Race quality screening model
- `place_hit_turf.joblib`, `place_hit_dirt.joblib` - Place hit models (joblib)
- `place_ret_turf.lgb`, `place_ret_dirt.lgb` - Place return models
- `win_hit_turf.joblib`, `win_hit_dirt.joblib` - Win hit models
- `win_ret_turf.lgb`, `win_ret_dirt.lgb` - Win return models
- `wide_hit_turf.lgb`, `wide_hit_dirt.lgb` - Wide hit models
- `wide_ret_turf.lgb`, `wide_ret_dirt.lgb` - Wide return models
- `isotonic_place_turf.joblib`, `isotonic_place_dirt.joblib` - Isotonic calibration
- `benter_combo_turf.json`, `benter_combo_dirt.json` - Benter combination params
- `confidence_params.json` - Confidence estimation parameters
- `meta.json` - Model metadata (train dates, versions)
- Year-stamped subdirs: `2023/`, `2024/`, `2025/`, `2025_alpha_capped/`

**Backtest Models (`data/models-backtest/`):**
- Per-year model snapshots (separate from production `data/models/`)

**Backtest Output (`data/backtest/`):**
- `backtest_result.json` - Single-year result
- `multi_year_result.json` - Multi-year result
- `bet_history.json`, `multi_year_bet_history.json` - Bet histories
- `backtest_report.html` - Jinja2-generated HTML report
- `bt_*_diagnostics.csv`, `bt_*_features.parquet` - Per-year diagnostic data
- `feature_importance/` - Feature importance logs
- `predictions/` - Backtest predictions

**Paper Trading (`data/paper_trading/`):**
- `bets.parquet` - Bet records
- `daily_summary/` - Daily summary files
- `predictions/` - Prediction outputs
- `diag_*_diagnostics.csv`, `diag_*_features.parquet` - Diagnostic data
- `dry_run/`, `bets/`, `model/` subdirectories

**Feature Cache:**
- `data/raw/horse_career_stats.parquet` - Precomputed horse career statistics
- `data/raw/sire_career_stats.parquet` - Precomputed sire statistics
- Feature engine output cached as needed

**ETL State:**
- `data/etl_state.json` - Tracks last delta update timestamps

### MLflow Artifact Storage

**Location:** `mlruns/` directory (local filesystem, `file:///mlruns` URI)
- Multiple experiment directories: `mlruns/0/`, `mlruns/2/`, `mlruns/3/`, `mlruns/4/`
- Model artifacts stored with `requirements.txt` for reproducibility
- Also present under `notebooks/mlruns/`

**File Storage:**
- Local filesystem only (no S3, GCS, or Azure Blob)
- Parquet is the primary storage format
- Models stored as `.lgb` (LightGBM native), `.joblib` (sklearn/serialized), `.json` (params)

**Caching:**
- No Redis or external cache
- ParquetStore uses in-process pandas DataFrames
- Feature cache in `data/raw/` for precomputed stats

## Authentication & Identity

**Auth Provider:**
- PostgreSQL password authentication only
- No OAuth, JWT, or API key management detected
- `PGPASSWORD` environment variable for database access

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry, Rollbar, or equivalent)

**Logs:**
- Python stdlib `logging` module
- Format: `"%(asctime)s [%(levelname)s] %(name)s: %(message)s"`
- Level: INFO (configurable in `config/settings.yaml`)
- No structured logging or log aggregation

**Model Monitoring:**
- `src/monitoring/model_monitor.py` - Performance and drift detection
- `src/monitoring/auto_retrain_trigger.py` - Automatic retrain trigger based on drift/decay
- `src/monitoring/notifier.py` - Slack notifications for alerts

## CI/CD & Deployment

**Hosting:**
- Local machine only (no cloud deployment, Docker, or serverless detected)
- No Dockerfile or docker-compose.yml present

**CI Pipeline:**
- None (no `.github/workflows/`, `.gitlab-ci.yml`, or equivalent)

**Execution:**
- CLI scripts in `scripts/` directory
- Manual execution in prescribed order: ETL -> Train -> Backtest

## Environment Configuration

**Required env vars:**
- `PGPASSWORD` - PostgreSQL password for EveryDB2 connection

**Optional env vars:**
- Slack webhook URL (configured programmatically, not via env var)

**Secrets location:**
- `PGPASSWORD` environment variable (not committed to git)
- `config/settings.yaml` has empty password field with comment to use env var
- `.gitignore` present (exact contents not verified but `.env` files are typically excluded)

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- Slack Incoming Webhook via `urllib.request` (`src/monitoring/notifier.py`)
  - Bet recommendations, daily summaries, alerts
  - Only triggered during paper trading execution

## Script Execution Pipeline

**ETL (`scripts/run_etl.py`):**
- Input: EveryDB2 PostgreSQL (103 tables)
- Output: Parquet files in `data/raw/` and `data/odds/`
- Modes: `full` (date range filter) or `delta` (incremental merge)
- Duration: ~10 minutes

**Training (`scripts/run_train.py`):**
- Input: Parquet files
- Output: Trained models to `data/models/`, MLflow artifacts
- Duration: ~44 minutes

**Backtest (`scripts/run_backtest.py`):**
- Input: Parquet files + trained models (re-trains each run for reproducibility)
- Output: `data/backtest/backtest_result.json` or `data/backtest/multi_year_result.json`
- Modes: Single year or multi-year (`--years` flag)
- Duration: ~57 minutes per year

**Supporting Scripts:**
- `scripts/precompute_career_stats.py` - Horse career stat precomputation
- `scripts/precompute_sire_stats.py` - Sire stat precomputation
- `scripts/run_paper_trading.py` - Paper trading execution
- `scripts/run_tuning.py` - Optuna hyperparameter tuning
- `scripts/analyze_odds_movement.py` - Odds movement analysis
- `scripts/compare_bt_pt_features.py` - Backtest vs paper trading feature comparison
- `scripts/scrape_everydb2_manual.py` - Manual data scraping (uses Playwright)

---

*Integration audit: 2026-05-02*
