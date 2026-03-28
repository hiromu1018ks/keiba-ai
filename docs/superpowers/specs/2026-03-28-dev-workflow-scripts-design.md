# Dev Workflow Scripts Design

Date: 2026-03-28
Status: Approved

## Goal

Create 3 standalone executable scripts that cover the development pipeline:
ETL (EveryDB2 → Parquet) → Training → Backtest.

Each script uses argparse for parameter input and can run independently.

## Context

Currently, only `scripts/run_backtest.py` exists (hardcoded dates, no argparse).
ETL and training have no entry points — they require inline Python.
The production automation layer (Phase F) is out of scope.

## Date Format Convention

CLI arguments accept `YYYYMMDD` format (e.g., `20200101`).
- `run_etl.py` passes dates directly to ETL functions (expect `YYYYMMDD`).
- `run_train.py` and `run_backtest.py` convert `YYYYMMDD` → `YYYY-MM-DD`
  before passing to `TrainingPipelineV5.run()` and `BacktestEngine.run()`.

Conversion helper (in each script):
```python
def to_dash_date(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"
```

## Scripts

### 1. `scripts/run_etl.py` — Data Export

Exports data from EveryDB2 external tables to Parquet files.

```bash
python scripts/run_etl.py --start 20150101 --end 20241231
```

| Argument   | Required | Description           |
|------------|----------|-----------------------|
| `--start`  | Yes      | ETL start date (YYYYMMDD) |
| `--end`    | Yes      | ETL end date (YYYYMMDD)   |

**Flow**:
1. `db = DatabaseConnection()` → PostgreSQL connection
2. `store = ParquetStore()` → Parquet I/O
3. `db.etl_to_parquet(store, start, end)` → 6 tables exported
   (internally calls `run_full_etl_to_parquet(db.get_engine(), store, start, end)`)
4. Log row counts and file paths per table

**Dependencies**: PostgreSQL (EveryDB2) must be running at localhost:5432.

### 2. `scripts/run_train.py` — Model Training

Trains all models via `TrainingPipelineV5` and logs to MLflow.

```bash
python scripts/run_train.py --start 20200101 --end 20231231
python scripts/run_train.py --start 20200101 --end 20231231 --experiment keiba-v5.5
```

| Argument         | Required | Default     | Description                |
|------------------|----------|-------------|----------------------------|
| `--start`        | Yes      | -           | Training start date (YYYYMMDD) |
| `--end`          | Yes      | -           | Training end date (YYYYMMDD)   |
| `--experiment`   | No       | `keiba-v5`  | MLflow experiment name     |

**Flow**:
1. `store = ParquetStore()` → `repo = DataRepository(store)`
2. `pipeline = TrainingPipelineV5(repo)` (uses default `settings_path`)
3. `pipeline.run(to_dash_date(start), to_dash_date(end))` → trains all models
4. Log MLflow run ID and key metrics

**Dependencies**: Parquet files must exist (run `run_etl.py` first).

### 3. `scripts/run_backtest.py` — Backtest (Refactored)

Refactored from existing script. Adds argparse. Always trains fresh models
(no MLflow loading in v1 — see Future Work).

```bash
# Full run: train + backtest
python scripts/run_backtest.py \
  --train-start 20200101 --train-end 20231231 \
  --test-start 20240101 --test-end 20241231
```

| Argument          | Required | Description                    |
|-------------------|----------|--------------------------------|
| `--train-start`   | Yes      | Training start date (YYYYMMDD) |
| `--train-end`     | Yes      | Training end date (YYYYMMDD)   |
| `--test-start`    | Yes      | Test start date (YYYYMMDD)     |
| `--test-end`      | Yes      | Test end date (YYYYMMDD)       |

**Flow**:
1. `store = ParquetStore()` → `repo = DataRepository(store)`
2. `pipeline = TrainingPipelineV5(repo)`
3. `models = pipeline.run(to_dash_date(train_start), to_dash_date(train_end))`
4. `engine = BacktestEngine(models=models, repo=repo)`
5. `result = engine.run(to_dash_date(test_start), to_dash_date(test_end))`
6. Print `result.total_roi`, max drawdown, final bankroll, bet count
7. Save results to `backtest_result.json`

**Note**: `BacktestEngine.__init__` signature is `(models, initial_bankroll=100_000, repo=None)`.
Must use keyword argument `repo=repo` to avoid passing repo as `initial_bankroll`.

**Dependencies**: Parquet files must exist (run `run_etl.py` first).

## Execution Order

```bash
# Step 1: Export data (one-time or incremental)
python scripts/run_etl.py --start 20150101 --end 20241231

# Step 2: Train models
python scripts/run_train.py --start 20200101 --end 20231231

# Step 3: Backtest (trains fresh models, then tests)
python scripts/run_backtest.py \
  --train-start 20200101 --train-end 20231231 \
  --test-start 20240101 --test-end 20241231
```

## Error Handling

- All scripts return exit code 0 on success, 1 on failure.
- `run_etl.py`: catches `sqlalchemy.OperationalError` (PostgreSQL unreachable)
  and prints actionable error message.
- `run_train.py`: validates that Parquet files exist before starting
  (`store.exists("raw", "races")` check).
- `run_backtest.py`: same Parquet validation.
- All scripts catch `KeyboardInterrupt` and print partial progress.

## Implementation Notes

- Each script initializes its own `ParquetStore` and `DataRepository` (no shared state).
- `ParquetStore()` uses default data directory (`data/`) resolved from project root
  via `_PROJECT_ROOT` in `connection.py`. No `--data-dir` argument needed in v1.
- All scripts use `logging` with basicConfig for consistent output.

## Future Work

- `--mlflow-run-id` for `run_backtest.py`: load `TrainedModelsV5` from MLflow
  artifacts to skip training. Deferred because reconstructing 14+ models
  (including runtime state like `current_regime`) requires a serialization protocol.
- `--settings-path` for `run_train.py`: custom config file path.
- `--data-dir` for all scripts: custom Parquet data directory.
