<!-- refreshed: 2026-05-02 -->
# Architecture

**Analysis Date:** 2026-05-02

## System Overview

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                        Entry Points (scripts/)                          │
│  run_etl.py · run_train.py · run_backtest.py · run_paper_trading.py    │
│  run_tuning.py · precompute_*.py · analyze_odds_movement.py            │
├──────────────────────────────────────────────────────────────────────────┤
│                    Orchestration Layer                                   │
│  TrainingPipelineV5           BacktestEngine          PaperPredictor    │
│  `pipelines/training_pipeline` `backtest/engine`      `paper_trading/`  │
│                     \              |              /                      │
│                      RacePredictor (shared inference)                   │
│                      `backtest/race_predictor`                          │
├──────────────┬──────────────────┬──────────────────┬────────────────────┤
│   Features   │     Models       │     Betting      │    Domain          │
│ FeatureEngine│ MarketModel      │ Orchestrator     │ types.py (enums)   │
│ 14 modules   │ AbilityModel     │ StakeCalculator  │ models.py (data)   │
│              │ Win/Place 2Stage │ DDController     │                    │
│              │ EVCorrection     │ GateKeeper       │                    │
│              │ RegimeDetector   │ LateMoneyFilter  │                    │
│              │ QualityScreener  │ MetaSwitcher     │                    │
│              │ BenterCombination│ Win/Place/Wide   │                    │
│              │ PlaceSelectionGate│   Strategy      │                    │
├──────────────┴──────────────────┴──────────────────┴────────────────────┤
│                         Data Layer                                       │
│  ParquetStore ── DataReaders ── ETL Engine ── DatabaseConnection        │
│  `db/parquet_store` `db/readers` `db/etl`    `db/connection`           │
│         │               │                                               │
│         ▼               ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Parquet Files (data/)                                          │    │
│  │  raw/ · odds/ · features/ · predictions/ · bets/               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
         │
         ▼ (ETL only)
┌──────────────────────────────────────────────────────────────────────────┐
│  PostgreSQL (EveryDB2) — READ ONLY                                      │
│  n_race · n_uma_race · n_odds_tanpuku · s_odds_tanpuku · etc.          │
└──────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| TrainingPipelineV5 | Orchestrate full training: data load, features, all models, MLflow logging | `src/pipelines/training_pipeline.py` |
| BacktestEngine | Historical simulation using trained models, bankroll tracking | `src/backtest/engine.py` |
| RacePredictor | Shared inference pipeline for single race: predict, select candidates, generate bets | `src/backtest/race_predictor.py` |
| PaperPredictor | Live paper-trading inference using RacePredictor + pre-computed features | `src/paper_trading/predictor.py` |
| FeatureEngine | Build all features from raw data; orchestrates 14+ feature modules | `src/features/feature_engine.py` |
| ParquetStore | Low-level Parquet file I/O with pyarrow predicate pushdown | `src/db/parquet_store.py` |
| DataReaders | Typed Parquet readers with date filters, type coercion, steeple exclusion | `src/db/readers.py` |
| DatabaseConnection | PostgreSQL connection (EveryDB2), ETL entry point | `src/db/connection.py` |
| ETL Engine | YAML-driven full/delta load from EveryDB2 to Parquet | `src/db/etl.py` |
| ModelLoader | Load TrainedModelsV5 from local dir or MLflow | `src/db/model_loader.py` |

## Pattern Overview

**Overall:** Pipeline architecture with 2-stage ML model decomposition

**Key Characteristics:**
- Parquet-based data layer (PostgreSQL is ETL-only, not queried at inference)
- 2-stage model: P(hit) x E(odds|hit) for each bet type (win, place, wide)
- Surface-based submodels: separate model sets for turf and dirt
- OOF (out-of-fold) predictions to prevent training leakage
- Regime-adaptive betting with 3 market states (aggressive/conservative/collapsed)
- All tests mock external dependencies (no DB needed)

## Layers

**Data Layer (`src/db/`):**
- Purpose: Read/write Parquet files and ETL from PostgreSQL
- Location: `src/db/`
- Contains: `ParquetStore`, `DataReaders`, `ETL`, `DatabaseConnection`, `ModelLoader`, `OddsExtractor`
- Depends on: pyarrow, pandas, SQLAlchemy, psycopg2
- Used by: TrainingPipeline, BacktestEngine, PaperPredictor, all feature modules

**Feature Layer (`src/features/`):**
- Purpose: Transform raw race/entry/odds data into ML features
- Location: `src/features/`
- Contains: FeatureEngine (orchestrator) + 14 feature modules
- Depends on: `src/db/` (ParquetStore via readers), `src/domain/`
- Used by: TrainingPipeline, BacktestEngine

**Model Layer (`src/models/`):**
- Purpose: Train and run LightGBM-based ML models
- Location: `src/models/`
- Contains: 14 model classes covering ability, market, 2-stage EV, correction, regime, calibration
- Depends on: `src/domain/`, lightgbm, scikit-learn, pandas
- Used by: TrainingPipeline, RacePredictor, ModelLoader

**Betting Layer (`src/betting/`):**
- Purpose: Bet selection, stake sizing, risk management
- Location: `src/betting/`
- Contains: Orchestrator, strategies (win/place/wide), StakeCalculator, DDController, GateKeeper
- Depends on: `src/domain/`, `src/models/`
- Used by: RacePredictor, BacktestEngine

**Domain Layer (`src/domain/`):**
- Purpose: Shared type definitions and data classes
- Location: `src/domain/`
- Contains: `types.py` (enums), `models.py` (dataclasses)
- Depends on: scikit-learn (IsotonicRegression type hint only)
- Used by: All other layers

**Orchestration Layer (`src/pipelines/`, `src/backtest/`):**
- Purpose: Coordinate data, features, models for training and evaluation
- Location: `src/pipelines/training_pipeline.py`, `src/backtest/`
- Contains: TrainingPipelineV5, BacktestEngine, RacePredictor, ValidationSuite
- Depends on: All other layers
- Used by: Script entry points

## Data Flow

### Primary Request Path (Training)

1. **ETL** — `scripts/run_etl.py` invokes `DatabaseConnection.etl_to_parquet()` (`src/db/connection.py:81`)
2. **ETL Engine** — `run_full_load()` reads EveryDB2 tables via SQLAlchemy, writes Parquet (`src/db/etl.py:200`)
3. **Training** — `scripts/run_train.py` invokes `TrainingPipelineV5.run()` (`src/pipelines/training_pipeline.py:83`)
4. **Data Load** — Readers load races, entries, odds from Parquet (`src/db/readers.py`)
5. **Feature Build** — `FeatureEngine.build_all()` orchestrates all feature modules (`src/features/feature_engine.py`)
6. **Submodel Train** — For each surface (turf/dirt): Market -> Ability (OOF) -> Win2Stage -> Place2Stage -> Wide2Stage -> EV Correction (`src/pipelines/training_pipeline.py:282`)
7. **Global Models** — RaceQualityScreener + RegimeDetector trained on race-level features (`src/pipelines/training_pipeline.py:259-270`)
8. **Save** — Models saved to `data/models/` (.lgb, .joblib, .json) + MLflow logging (`src/pipelines/training_pipeline.py:931-1045`)

### Backtest Flow

1. **Run** — `scripts/run_backtest.py` trains then runs `BacktestEngine.run()` (`src/backtest/engine.py:210`)
2. **Data Load** — Same readers as training, plus pre-post odds extraction (`src/db/odds_extractor.py`)
3. **Feature Build** — Same FeatureEngine pipeline as training
4. **Race Loop** — For each race: `RacePredictor.predict()` -> `should_bet()` -> `select_bets()` -> settle (`src/backtest/engine.py:420`)
5. **Result** — `BacktestResult` with ROI, bankroll curve, bet history (`src/backtest/engine.py:805`)

### Paper Trading Flow

1. **Setup** — `scripts/run_paper_trading.py --mode setup` fetches today's race cards from EveryDB2 (`src/paper_trading/predictor.py:42`)
2. **Predict** — `--mode predict` loads trained model, runs inference via RacePredictor, saves bets (`src/paper_trading/predictor.py`)
3. **Reconcile** — `--mode reconcile` compares predictions with actual results, calculates ROI (`src/paper_trading/reconciler.py`)

**State Management:**
- Parquet files are the single source of truth (no database at inference time)
- Model state stored as .lgb/.joblib/.json files in `data/models/`
- MLflow tracks experiment runs in `mlruns/` directory
- ETL state persisted in `data/etl_state.json`

## Key Abstractions

**TrainedModelsV5:**
- Purpose: Container for all trained models (the output of training pipeline)
- Definition: `src/domain/models.py:252`
- Contains: `dict[str, SubmodelSet]` (keyed by "turf"/"dirt"), `RaceQualityScreener`, `RegimeDetector`
- Pattern: Dataclass container with all model objects

**SubmodelSet:**
- Purpose: All models for one surface (turf or dirt)
- Definition: `src/domain/models.py:229`
- Contains: MarketModel, AbilityModel, Win/Place/Wide 2-stage, EV correctors, confidence estimator, Benter combination, calibration
- Pattern: One SubmodelSet per surface, trained independently

**ParquetStore:**
- Purpose: Abstracted Parquet I/O with predicate pushdown
- Examples: `src/db/parquet_store.py`
- Pattern: category/name addressing (e.g., `store.read("raw", "races", filters=...)`)

**RacePredictor:**
- Purpose: Single-race inference shared between BacktestEngine and PaperPredictor
- Examples: `src/backtest/race_predictor.py`
- Pattern: Strategy pattern — inject TrainedModelsV5 + optional StakeCalculator/DDController

**2-Stage Model:**
- Purpose: Decompose EV = P(hit) x E(odds|hit) to avoid zero-inflation
- Examples: `src/models/two_stage_return_model.py` (WinTwoStageModel, PlaceTwoStageModel)
- Pattern: Two separate LightGBM models per bet type per surface

## Entry Points

**`scripts/run_etl.py`:**
- Location: `scripts/run_etl.py`
- Triggers: Manual (CLI)
- Responsibilities: PostgreSQL (EveryDB2) -> Parquet extraction

**`scripts/run_train.py`:**
- Location: `scripts/run_train.py`
- Triggers: Manual (CLI)
- Responsibilities: Full model training pipeline, MLflow logging

**`scripts/run_backtest.py`:**
- Location: `scripts/run_backtest.py`
- Triggers: Manual (CLI)
- Responsibilities: Train + historical simulation, supports single-year and multi-year modes

**`scripts/run_paper_trading.py`:**
- Location: `scripts/run_paper_trading.py`
- Triggers: Manual (CLI, typically cron)
- Responsibilities: Setup/predict/reconcile/dry-run modes for paper trading

**`scripts/run_tuning.py`:**
- Location: `scripts/run_tuning.py`
- Triggers: Manual (CLI)
- Responsibilities: Optuna hyperparameter optimization

## Architectural Constraints

- **Data flow direction:** PostgreSQL -> Parquet -> Features -> Models -> Predictions -> Bets. Never reversed.
- **No DB at inference:** All inference (backtest, paper trading) reads only Parquet files. PostgreSQL is ETL-only.
- **Surface partition:** Models are trained independently per surface (turf/dirt). No cross-surface model sharing.
- **OOF leakage prevention:** Stage1 (AbilityModel) uses out-of-fold predictions. All time-series data sorted by race_date before train/valid splits.
- **POST_RACE exclusion:** `POST_RACE_COLS` (kakuteijyuni, confirmed_odds, etc.) are stripped before inference in backtest (`src/backtest/engine.py:472-475`).
- **Global state:** `RegimeDetector` maintains `_current_regime` state across races in backtest loop. Not thread-safe.
- **Circular imports:** Avoided via `TYPE_CHECKING` guards throughout `src/domain/models.py`.
- **Threading:** `TrainingPipelineV5._train_submodel()` uses `ThreadPoolExecutor(max_workers=2)` for parallel surface training (`src/pipelines/training_pipeline.py:207`). LightGBM's own threading is controlled by `num_threads` parameter.

## Anti-Patterns

### Inline feature computation in BacktestEngine

**What happens:** `BacktestEngine.run()` contains 100+ lines of feature pre-computation (horse history, jockey context, sire features, pace aptitude, course features) that duplicates logic from `TrainingPipelineV5._train_submodel()`.
**Why it's wrong:** Changes to feature computation in training must be manually mirrored in backtest. Risk of feature drift between training and evaluation.
**Do this instead:** Extract shared feature pre-computation into a reusable function called by both `TrainingPipelineV5._train_submodel()` and `BacktestEngine.run()`. See `src/backtest/engine.py:328-405` vs `src/pipelines/training_pipeline.py:290-371`.

### Model loading path fragmentation

**What happens:** `ModelLoader._load_from_local()` in `src/db/model_loader.py:365` has model loading logic separate from `TrainingPipelineV5._save_models_local()`. When a new model is added to `SubmodelSet`, both save and load code must be updated independently.
**Why it's wrong:** Easy to forget updating one side, causing silent model loading failures (model silently uses defaults).
**Do this instead:** Use a declarative model registry (dict of name -> file_pattern) that drives both save and load. See `src/db/model_loader.py:365-570` and `src/pipelines/training_pipeline.py:934-1045`.

## Error Handling

**Strategy:** Fail-safe with logging

**Patterns:**
- Empty DataFrame checks after every data load (`if df.empty: return BacktestResult(...)`)
- Try/except around optional model loading (PlaceEVCorrectionModel, PlaceSelectionGate) with fallback to passthrough
- Warning logs for missing columns, falling back to defaults
- LightGBM exceptions in inference caught in `RacePredictor.predict()` (`src/backtest/race_predictor.py:89`)

## Cross-Cutting Concerns

**Logging:** Python `logging` module. Configured in `config/settings.yaml`. Per-module loggers via `logging.getLogger(__name__)`.

**Validation:** Type coercion in `src/db/readers.py:_coerce_types()` handles legacy string-typed Parquet columns. Feature column existence checked before use.

**Authentication:** PostgreSQL password via environment variable `PGPASSWORD`, overriding `config/settings.yaml` default.

**Timing:** `src/utils/timing.py:TimingContext` context manager for pipeline step timing.

**Experiment Tracking:** MLflow for model versioning, parameters, and metrics. Local file-based tracking (`mlruns/`).

---

*Architecture analysis: 2026-05-02*
