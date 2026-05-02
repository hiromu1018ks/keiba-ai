# Technology Stack

**Analysis Date:** 2026-05-02

## Languages

**Primary:**
- Python 3.11 - All source code (`src/`, `scripts/`, `tests/`)

**Secondary:**
- YAML - Configuration (`config/settings.yaml`, `config/etl_tables.yaml`, `config/backtest_config.yaml`)
- SQL - PostgreSQL queries embedded in Python (`src/db/everydb2_queries.py`, `src/db/etl.py`)
- Jinja2 HTML templates - Backtest reports (`src/backtest/report.py`)

## Runtime

**Environment:**
- Python 3.11 (pinned via `mise.toml` with `[tools] python = "3.11"`)
- `requires-python = ">=3.11"` in `pyproject.toml`

**Package Manager:**
- pip with editable install (`pip install -e ".[dev]"`)
- `mise` for Python version management
- `setuptools>=69.0` as build backend
- Lockfile: Not present (no `requirements.lock` or `pip freeze` output committed)

## Frameworks

**Core ML:**
- LightGBM >=4.3 - Primary gradient boosting model (lambdarank ranker for ability, binary for hit/EV correction)
- scikit-learn >=1.4 - IsotonicRegression calibration, Ridge meta-learner, CalibratedClassifierCV, KFold, metrics
- XGBoost >=2.0 - Stacked ensemble Level-1 model (in `src/models/stacked_ensemble.py`)
- CatBoost >=1.2 - Stacked ensemble Level-1 model (in `src/models/stacked_ensemble.py`)
- Optuna >=3.5 - Hyperparameter tuning (in `src/tuning/optuna_tuner.py`)

**Experiment Tracking:**
- MLflow >=2.12 - Model versioning, experiment tracking, artifact storage (local `file:///mlruns`)
  - Used in `src/pipelines/training_pipeline.py`, `src/db/model_loader.py`

**Data Processing:**
- pandas >=2.2 - Primary data manipulation throughout codebase
- numpy >=1.26 - Numerical computation
- pyarrow >=14.0 - Parquet file I/O with predicate pushdown (`src/db/parquet_store.py`)

**Testing:**
- pytest >=8.0 - Test runner (all tests use mocks, no DB required)
- pytest-cov >=5.0 - Coverage reporting

**Build/Dev:**
- ruff >=0.4 - Linting and formatting (target py311, line-length=100, rules E/F/I/N/W)
- mypy >=1.10 - Static type checking (strict: `disallow_untyped_defs = true`)
- ipykernel >=6.29 - Jupyter notebook support

**Optional:**
- playwright >=1.40 - Web scraping (optional `[scraping]` dependency)

## Key Dependencies

**Critical:**
- `lightgbm` >=4.3 - Core prediction models (Stage1 ability, EV correction, regime detector, market model)
- `scikit-learn` >=1.4 - Calibration (IsotonicRegression), StackedEnsemble meta-learner (Ridge), metrics
- `pandas` >=2.2 - All data pipeline operations
- `pyarrow` >=14.0 - Parquet I/O layer for all data access
- `mlflow` >=2.12 - Model artifact storage and versioning

**Infrastructure:**
- `sqlalchemy` >=2.0 - PostgreSQL connection for ETL (`src/db/connection.py`)
- `psycopg2-binary` >=2.9 - PostgreSQL adapter for EveryDB2 queries (`src/db/everydb2_queries.py`)
- `joblib` - Model serialization (.joblib files for sklearn models, `src/db/model_loader.py`)
- `pyyaml` >=6.0 - Configuration file parsing

**Mathematics:**
- `scipy` (transitive via scikit-learn) - Optimization (`scipy.optimize.minimize` in `src/models/benter_combination.py`)

**Reporting:**
- `jinja2` >=3.1 - HTML report generation (`src/backtest/report.py`, `src/paper_trading/report.py`)
- `tqdm` >=4.66 - Progress bars in ETL (`src/db/etl.py`)

## Configuration

**Environment:**
- `mise.toml` - Python version pinning
- `pyproject.toml` - Project metadata, dependencies, tool configs (ruff, mypy, pytest)
- `requirements.txt` - Subset of `pyproject.toml` dependencies (does not include xgboost, catboost, optuna, pyarrow, tqdm, jinja2, joblib)

**Application Config:**
- `config/settings.yaml` - Database connection, paths, logging, feature engine, late money thresholds, submodel configuration
- `config/etl_tables.yaml` - 103 EveryDB2 table definitions (n_ raced/master + s_ delta)
- `config/backtest_config.yaml` - Walk-forward, holdout, pass criteria, EV correction, validation settings

**Key configs required:**
- `PGPASSWORD` env var - PostgreSQL password (overrides empty default in settings.yaml)
- MLflow tracking URI defaults to `file:///mlruns` (local filesystem)

## Platform Requirements

**Development:**
- Python 3.11 via mise
- PostgreSQL accessible at `localhost:5432/everydb2` (for ETL only)
- Local filesystem for Parquet data, MLflow artifacts, model files

**Production:**
- Local deployment (no containerization or cloud hosting detected)
- PostgreSQL with EveryDB2/JRA-VAN DataLab external tables
- Local Parquet file storage (`data/` directory)
- Slack Incoming Webhook for notifications (optional)

---

*Stack analysis: 2026-05-02*
