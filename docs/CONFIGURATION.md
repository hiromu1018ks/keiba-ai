<!-- generated-by: gsd-doc-writer -->

# Configuration

This document describes all configuration mechanisms for the keiba-ai prediction system, including environment variables, YAML config files, and hardcoded defaults in the codebase.

## Environment Variables

The project uses `python-dotenv` to load variables from a `.env` file at the project root. Only two environment variables are referenced in the codebase.

| Variable | Required | Default | Description |
|---|---|---|---|
| `PGPASSWORD` | Yes (for ETL/paper trading) | `""` (empty) | PostgreSQL password. Overrides the empty `password` field in `config/settings.yaml`. Used by `DatabaseConnection` and `run_paper_trading.py`. |
| `SLACK_WEBHOOK_URL` | No | `""` (empty) | Slack Incoming Webhook URL for paper trading notifications. When empty, Slack notifications are disabled with a warning. |

The `.env.example` file at the project root contains only `PGPASSWORD=` as a template.

## Config File Format

### `config/settings.yaml` -- Primary Application Settings

The main configuration file, loaded by `src/db/connection.py` via `yaml.safe_load`. If the file is missing, the application raises `FileNotFoundError` at startup.

```yaml
# Database connection (ETL only)
database:
  host: "localhost"
  port: 5432
  dbname: "everydb2"
  user: "postgres"
  password: ""  # Overridden by PGPASSWORD env var

# File system paths
paths:
  data_dir: "data"
  model_dir: "models"
  mlflow_tracking_uri: "file:///mlruns"

# Logging
logging:
  level: "INFO"
  format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# Feature engine
feature_engine:
  exclude_steeple: true  # Exclude steeplechase races (TrackCD 51-59)

# Late money filter thresholds
late_money:
  cancel_threshold: 0.25     # Cancel if win odds drops >= 25%
  add_rise_threshold: 0.30   # Add candidate if win odds rises >= 30%
  cancel_time_minutes: 3     # Evaluate at T-3 minutes
  log_time_minutes: 2        # Log at T-2 minutes

# Submodel configuration
submodel:
  surfaces: ["turf", "dirt"]
  distance_bands:
    turf:
      sprint: [0, 1400]
      mile: [1401, 1700]
      intermediate: [1701, 2100]
      long: [2101, 9999]
    dirt:
      sprint: [0, 1400]
      mile: [1401, 1700]
      intermediate: [1701, 9999]

# Betting strategy
betting_strategy:
  default_fractional_kelly: 0.5
  kelly_fraction_cap: 0.25
  target_ev: 1.10
  max_scale: 2.0
  regime_fractions:
    aggressive: 0.50
    conservative: 0.25
    collapsed: 0.00
```

### `config/etl_tables.yaml` -- ETL Table Definitions

Defines 103 tables (53 `n_` full-load tables + 50 `s_` delta tables) for the EveryDB2-to-Parquet ETL pipeline. Each table entry specifies:

- `db_table`: PostgreSQL table name
- `parquet_key`: Output Parquet file name (without extension)
- `category`: Output directory under `data/` -- either `raw` or `odds`
- `type`: Load strategy -- `raced` (date-filtered), `master` (full dump), or `delta` (incremental merge)
- `pk`: Primary key columns used for delta merge deduplication
- `partition_cols`: (Optional) Hive partitioning columns for large tables (e.g., `jodds_tanpuku` partitions by `year, month`)

Loaded by `src/db/etl.py` via the `load_table_config()` function.

### `config/backtest_config.yaml` -- Backtest Validation Criteria

Defines pass/fail thresholds for the backtest validation suite (`src/backtest/validation_suite.py`).

```yaml
walk_forward:
  train_years: 4
  test_years: 1
  step_years: 1

holdout:
  start: "2022-01-01"
  end: "2024-12-31"

pass_criteria:
  place_roi: 1.00        # Place ROI >= 100%
  wide_roi: 1.03         # Wide ROI >= 103%
  overall_roi: 1.01      # Overall ROI >= 101%
  max_drawdown: 0.16     # Max DD <= 16%
  min_profitable_months: 22  # Monthly 100%+ >= 22/36

ev_correction:
  mae_improvement: 0.10
  mid_range_improvement: 0.15

validation:
  min_submodel_samples: 20000
  p_e_correlation_max: 0.30
```

## Required vs Optional Settings

### Settings that cause startup failure if misconfigured

| Setting | File | Failure Mode |
|---|---|---|
| `config/settings.yaml` missing | `settings.yaml` | `FileNotFoundError` in `_load_settings()` |
| `config/etl_tables.yaml` missing | `etl_tables.yaml` | `FileNotFoundError` in `load_table_config()` |
| `PGPASSWORD` missing (ETL mode) | `.env` / env var | PostgreSQL connection fails with authentication error |
| `database.host/port/dbname/user` missing | `settings.yaml` | `KeyError` in `DatabaseConnection.__init__()` |

### Settings that degrade gracefully when absent

| Setting | Default Behavior |
|---|---|
| `SLACK_WEBHOOK_URL` | Slack notifications disabled; warning logged |
| `settings.yaml password` field | Empty string; expects `PGPASSWORD` env var override |
| `mlflow_tracking_uri` | Defaults to `"file:///mlruns"` in `ModelLoader` and `PaperTradingConfig` |

## Defaults

### Dataclass Defaults in Source Code

Several configuration objects use Python dataclass defaults rather than YAML files. These are the runtime defaults if no override is provided.

**`RegimeConfig`** (`src/domain/models.py`):

| Parameter | Default | Description |
|---|---|---|
| `window` | `200` | Rolling window for regime detection |
| `min_samples` | `100` | Minimum samples before regime detection activates |
| `fav_rate_aggressive` | `0.28` | Favorite rate threshold for aggressive regime |
| `fav_rate_collapsed` | `0.50` | Favorite rate threshold for collapsed regime |
| `overround_base` | `0.20` | Baseline overround value |
| `retrain_trigger` | `100` | Consecutive collapsed detections to trigger retrain |

**`TwoStageConfig`** (`src/domain/models.py`):

| Parameter | Default | Description |
|---|---|---|
| `hit_metric` | `"auc"` | Stage1 (hit) evaluation metric |
| `hit_leaves` | `15` | LightGBM num_leaves for hit model |
| `hit_lr` | `0.03` | Learning rate for hit model |
| `hit_rounds` | `300` | Boosting rounds for hit model |
| `return_metric` | `"mae"` | Stage2 (return) evaluation metric |
| `return_leaves` | `15` | LightGBM num_leaves for return model |
| `return_lr` | `0.03` | Learning rate for return model |
| `return_rounds` | `300` | Boosting rounds for return model |
| `min_hit_samples` | `200` | Minimum samples required to train Stage2 |

**`SafetyConfig`** (`src/domain/models.py`):

| Parameter | Default | Description |
|---|---|---|
| `min_bankroll` | `10000.0` | Minimum bankroll to allow betting (yen) |
| `max_daily_loss` | `10000.0` | Maximum daily loss before stop (yen) |
| `max_weekly_loss` | `30000.0` | Maximum weekly loss before stop (yen) |
| `max_consecutive_losses` | `10` | Maximum consecutive losses before stop |

**`StakeCalculator`** (`src/betting/stake_calculator.py`):

| Parameter | Default | Description |
|---|---|---|
| `fractional_kelly` | `0.5` | Kelly fraction (half-Kelly) |
| `kelly_fraction_cap` | `0.25` | Maximum Kelly fraction cap |
| `target_ev` | `1.10` | Target EV for stake scaling |
| `max_scale` | `2.0` | Maximum EV scaling multiplier |
| `MIN_EDGE_THRESHOLD` | `0.005` | Minimum edge to consider betting (0.5%) |
| `RACE_EXPOSURE_CAP` | `0.02` | Maximum race exposure (2% of bankroll) |
| `MIN_STAKE` | `100` | Minimum stake in yen |
| `MAX_STAKE` | `10000` | Maximum stake in yen |

**`PaperTradingConfig`** (`src/paper_trading/config.py`):

| Parameter | Default | Description |
|---|---|---|
| `mlflow_run_id` | `None` | MLflow run ID (uses latest if None) |
| `mlflow_tracking_uri` | `"file:///mlruns"` | MLflow tracking URI |
| `ev_threshold` | `1.0` | Minimum EV for paper trading bets |
| `initial_bankroll` | `100000.0` | Starting bankroll (yen) |
| `stake` | `100.0` | Fixed stake per bet (yen) |
| `watch_lead_minutes` | `5` | Minutes before race start to begin watching |
| `retry_count` | `3` | Number of retry attempts |
| `retry_interval_seconds` | `60` | Seconds between retries |
| `query_timeout_seconds` | `30` | Database query timeout |
| `paper_trading_dir` | `Path("data/paper_trading")` | Output directory |

### Regime Strategy Parameters

The `RegimeDetector` (`src/models/regime_detector.py`) returns hardcoded strategy parameters per regime state. These are defined in `_get_base_params()` and can be overridden at runtime via the `override_params` constructor argument.

| Parameter | Aggressive | Conservative | Collapsed |
|---|---|---|---|
| `ev_threshold` | 1.10 | 1.30 | 1.50 |
| `edge_threshold` | 0.05 | 0.06 | 0.09 |
| `fractional_kelly` | 0.50 | 0.25 | 0.00 |
| `min_place_prob` | 0.08 | 0.09 | 0.10 |
| `max_place_odds` | 18.0 | 18.0 | 16.0 |
| `wide_enabled` | false | false | false |
| `score_threshold` | 0.010 | 0.020 | 0.050 |
| `max_bets_per_race` | 1 | 1 | 1 |

The `settings.yaml` `betting_strategy.regime_fractions` section provides parallel values (`aggressive: 0.50`, `conservative: 0.25`, `collapsed: 0.00`) that correspond to the `fractional_kelly` in each regime.

## Per-Environment Overrides

The project does not use per-environment config files (no `.env.development` / `.env.production` / `.env.test` files exist). Environment-specific behavior is controlled through:

1. **`PGPASSWORD` environment variable** -- The only runtime override. Set differently per environment to point to the correct PostgreSQL instance.

2. **MLflow tracking URI** -- Configurable in `settings.yaml` (`paths.mlflow_tracking_uri`) and as a default in `ModelLoader` and `PaperTradingConfig`. The default `"file:///mlruns"` uses local filesystem storage.

3. **Script arguments** -- Pipeline scripts accept `--start`, `--end`, `--years`, `--run-id`, and other CLI arguments that override defaults without modifying config files.

4. **`config/settings.yaml` `database` section** -- The host, port, dbname, and user fields can be changed to target different PostgreSQL instances for development vs production.

## File Layout Reference

```
config/
  settings.yaml          # Primary config (database, paths, logging, features, betting)
  etl_tables.yaml        # ETL table definitions (103 tables)
  backtest_config.yaml   # Backtest pass/fail criteria

.env                     # Environment variables (PGPASSWORD, SLACK_WEBHOOK_URL)
.env.example             # Template for .env

data/                    # Data directory (paths.data_dir)
  raw/*.parquet          # Raw race/entry/payout data
  odds/*.parquet         # Odds data (snapshots, time-series)
  features/              # Feature cache
  predictions/           # Model predictions
  bets/                  # Bet records
  models/                # Production model artifacts
  models-backtest/       # Backtest model artifacts (isolated from production)
  paper_trading/         # Paper trading outputs
  backtest/              # Backtest results

mlruns/                  # MLflow experiment tracking (file-based)
```
