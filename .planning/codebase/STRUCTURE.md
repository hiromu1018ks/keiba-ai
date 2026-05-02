# Codebase Structure

**Analysis Date:** 2026-05-02

## Directory Layout

```
keiba-ai/
├── config/              # Configuration files (settings, ETL tables, backtest)
├── data/                # Parquet data files (not committed)
│   ├── raw/             # Raw race/entry/payout/horse data
│   ├── odds/            # Odds snapshots, time series, wide odds
│   ├── features/        # Feature cache (horse_features.parquet)
│   ├── predictions/     # Model predictions
│   ├── bets/            # Bet records
│   ├── models/          # Production trained models
│   ├── models-backtest/ # Backtest-specific model copies (per year)
│   ├── backtest/        # Backtest results, diagnostics
│   └── paper_trading/   # Paper trading state (bets, predictions, summaries)
├── docs/                # Design document (design.md)
├── mlruns/              # MLflow experiment tracking (local)
├── notebooks/           # Jupyter notebooks for analysis
├── output/              # Generated reports (HTML, JSON)
├── scripts/             # CLI entry points
├── src/                 # Main source code
│   ├── automation/      # Scheduler, safety guard, PAT voter
│   ├── backtest/        # Backtest engine, race predictor, validation, diagnostics
│   ├── betting/         # Betting strategies, stake calc, risk management
│   ├── db/              # Data layer: Parquet I/O, ETL, readers, model loading
│   ├── domain/          # Shared types and dataclass definitions
│   ├── features/        # Feature engineering modules (14 modules)
│   ├── ingestion/       # JVLink fetcher, odds collector
│   ├── models/          # ML model implementations
│   ├── monitoring/      # Model monitoring, auto-retrain, notifications
│   ├── paper_trading/   # Paper trading pipeline
│   ├── pipelines/       # Training pipeline orchestration
│   ├── tuning/          # Optuna hyperparameter tuning
│   └── utils/           # Shared utilities (timing)
├── tests/               # Test files (mock-based, no DB required)
├── pyproject.toml       # Project config, dependencies, tool settings
├── mise.toml            # Python version pin (3.11)
└── CLAUDE.md            # AI assistant instructions
```

## Directory Purposes

**`src/db/`:**
- Purpose: Data access layer -- Parquet I/O, PostgreSQL ETL, data readers
- Contains: 8 Python files covering storage, reading, ETL, schema, model loading
- Key files: `parquet_store.py`, `readers.py`, `etl.py`, `connection.py`, `model_loader.py`

**`src/features/`:**
- Purpose: Feature engineering for ML models
- Contains: `feature_engine.py` (orchestrator) + 14 specialized feature modules
- Key files: `feature_engine.py`, `horse_history_features.py`, `odds_dynamics_features.py`, `intra_race_features.py`

**`src/models/`:**
- Purpose: ML model implementations (LightGBM, calibration, ensemble)
- Contains: 14 model classes covering the full prediction pipeline
- Key files: `stage1_ability_model.py`, `two_stage_return_model.py`, `market_model.py`, `regime_detector.py`, `ev_correction_model.py`

**`src/betting/`:**
- Purpose: Bet selection, sizing, and risk management
- Contains: 8 files for strategies (win/place/wide), stake calculation, drawdown control
- Key files: `orchestrator.py`, `stake_calculator.py`, `drawdown_controller.py`, `place_strategy.py`

**`src/domain/`:**
- Purpose: Shared type definitions (no business logic)
- Contains: 2 files: `types.py` (enums) and `models.py` (dataclasses)
- Key files: `types.py` (BetType, Surface, RegimeState, RecoveryState), `models.py` (Race, Entry, Bet, TrainedModelsV5, SubmodelSet)

**`src/backtest/`:**
- Purpose: Historical simulation and evaluation
- Contains: Engine, race predictor, validation suite, diagnostics, reporting
- Key files: `engine.py`, `race_predictor.py`, `validation_suite.py`, `report.py`

**`src/pipelines/`:**
- Purpose: Training pipeline orchestration
- Contains: Single file `training_pipeline.py`
- Key files: `training_pipeline.py` (TrainingPipelineV5 class)

**`src/automation/`:**
- Purpose: Automated operation (scheduling, safety, voting)
- Contains: 3 files for scheduling, safety guard, PAT voter
- Key files: `scheduler.py`, `safety_guard.py`, `pat_voter.py`

**`src/monitoring/`:**
- Purpose: Model performance monitoring and alerting
- Contains: Model monitor, auto-retrain trigger, notifier
- Key files: `model_monitor.py`, `auto_retrain_trigger.py`, `notifier.py`

**`src/paper_trading/`:**
- Purpose: Paper trading simulation pipeline
- Contains: Config, predictor, reconciler, report, watcher
- Key files: `predictor.py`, `reconciler.py`, `config.py`

**`src/ingestion/`:**
- Purpose: Data ingestion interfaces (JVLink, odds collection)
- Contains: 2 files for race data fetching and odds collection
- Key files: `jvlink_fetcher.py`, `odds_collector.py`

**`src/tuning/`:**
- Purpose: Hyperparameter optimization
- Contains: Single file `optuna_tuner.py`
- Key files: `optuna_tuner.py`

## Key File Locations

**Entry Points:**
- `scripts/run_etl.py`: ETL -- PostgreSQL (EveryDB2) to Parquet extraction
- `scripts/run_train.py`: Full model training pipeline
- `scripts/run_backtest.py`: Train + historical simulation (single-year or multi-year)
- `scripts/run_paper_trading.py`: Paper trading with setup/predict/reconcile/dry-run modes
- `scripts/run_tuning.py`: Optuna hyperparameter search
- `scripts/precompute_sire_stats.py`: Pre-compute sire statistics cache
- `scripts/precompute_career_stats.py`: Pre-compute career statistics cache
- `scripts/analyze_odds_movement.py`: Odds movement analysis utility

**Configuration:**
- `config/settings.yaml`: Database connection, paths, logging, feature engine, late money, submodel settings
- `config/etl_tables.yaml`: YAML-driven table definitions for ETL (table names, parquet keys, partition columns, primary keys)
- `config/backtest_config.yaml`: Backtest configuration
- `pyproject.toml`: Dependencies, pytest/ruff/mypy settings
- `mise.toml`: Python 3.11 version pin

**Core Logic:**
- `src/db/parquet_store.py`: ParquetStore -- low-level Parquet read/write with pyarrow pushdown
- `src/db/readers.py`: Typed data loaders (load_races, load_entries, load_odds_*, etc.)
- `src/db/etl.py`: ETL engine -- full load and delta merge from PostgreSQL to Parquet
- `src/db/connection.py`: DatabaseConnection -- PostgreSQL connection singleton, ETL entry point
- `src/db/model_loader.py`: ModelLoader -- reconstruct TrainedModelsV5 from files/MLflow
- `src/db/odds_extractor.py`: extract_pre_post_odds -- N-minutes-before-post odds snapshot extraction
- `src/pipelines/training_pipeline.py`: TrainingPipelineV5 -- full training orchestration
- `src/backtest/engine.py`: BacktestEngine -- historical simulation with bankroll tracking
- `src/backtest/race_predictor.py`: RacePredictor -- shared single-race inference pipeline
- `src/features/feature_engine.py`: FeatureEngine -- main feature orchestration
- `src/domain/models.py`: TrainedModelsV5, SubmodelSet, Bet, Race, Entry dataclasses
- `src/domain/types.py`: BetType, Surface, RegimeState, RecoveryState enums

**Testing:**
- `tests/`: All test files, mock-based (no DB required)
- See Test Structure section below

## Model Architecture Files

The model pipeline is defined across these files, executed in this order during training:

| Order | Model | File | Purpose |
|-------|-------|------|---------|
| 1 | MarketModel | `src/models/market_model.py` | Predict market probability, compute log_error |
| 2 | AbilityModel | `src/models/stage1_ability_model.py` | LightGBM Ranker for horse ability (OOF) |
| 3 | PlaceAbilityModel | `src/models/place_ability_model.py` | Place-specific ability (CalibratedClassifierCV) |
| 4 | WinTwoStageModel | `src/models/two_stage_return_model.py` | P(win) x E(win_odds|win) |
| 5 | EVCorrectionModel | `src/models/ev_correction_model.py` | P/E decomposition correction |
| 6 | PlaceTwoStageModel | `src/models/two_stage_return_model.py` | P(place) x E(place_odds|place) |
| 7 | PlaceEVCorrectionModel | `src/models/ev_correction_model.py` | Place-specific EV correction |
| 8 | WideTwoStageModel | `src/models/wide_two_stage_model.py` | P(both top3) x E(wide_odds|hit) |
| 9 | BenterCombination | `src/models/benter_combination.py` | Fundamental + market probability fusion |
| 10 | RobustConfidenceEstimator | `src/models/robust_confidence_estimator.py` | Confidence interval calibration |
| 11 | PlaceSelectionGateModel | `src/models/place_selection_gate.py` | Learned gate for place candidate selection |
| 12 | RaceQualityScreener | `src/models/race_quality_screener.py` | Binary: should we bet on this race? |
| 13 | RegimeDetector | `src/models/regime_detector.py` | 3-state market regime classification |
| 14 | SubModelManager | `src/models/submodel_manager.py` | Surface-based model routing + distance band features |
| 15 | WidePairBuilder | `src/models/wide_pair_builder.py` | Construct wide bet pairs |
| 16 | WalkForwardCV | `src/models/walk_forward_cv.py` | Time-series cross-validation |
| 17 | StackedEnsemble | `src/models/stacked_ensemble.py` | LGBM + XGB + CatBoost -> Ridge meta-learner |

## Feature Module Files

| Module | File | Purpose |
|--------|------|---------|
| FeatureEngine | `src/features/feature_engine.py` | Main orchestrator: basic features, difficulty, popularity |
| BloodlineFeatures | `src/features/bloodline_features.py` | Bloodline surface/distance/condition win rates |
| CourseFeatures | `src/features/course_features.py` | Course-specific win rates |
| FormCycleFeatures | `src/features/form_cycle_features.py` | Recent form cycle scoring |
| HorseCareerStats | `src/features/horse_career_stats.py` | Pre-computed career statistics |
| HorseHistoryFeatures | `src/features/horse_history_features.py` | Past race performance features |
| InfoAsymmetryFeatures | `src/features/info_asymmetry_features.py` | Expanding-window historical statistics |
| InteractionFeatures | `src/features/interaction_features.py` | Feature cross-terms |
| IntraRaceFeatures | `src/features/intra_race_features.py` | Within-race relative features |
| JockeyContextFeatures | `src/features/jockey_context_features.py` | Jockey performance context |
| JockeyTrainerCombo | `src/features/jockey_trainer_combo.py` | Jockey-trainer combination stats |
| LeakageValidators | `src/features/leakage_validators.py` | Feature leakage detection |
| MarketBiasFeatures | `src/features/market_bias_features.py` | Market structure features (overround, entropy) |
| OddsDynamicsFeatures | `src/features/odds_dynamics_features.py` | Odds change rates, volatility |
| PaceAptitudeFeatures | `src/features/pace_aptitude_features.py` | Pace running style aptitude |
| RaceDifficultyModel | `src/features/race_difficulty_model.py` | Race difficulty scoring |
| SireFeatures | `src/features/sire_features.py` | Sire (father horse) progeny statistics |
| TrainerContextFeatures | `src/features/trainer_context_features.py` | Trainer performance context |

## Betting Module Files

| Module | File | Purpose |
|--------|------|---------|
| BettingOrchestrator | `src/betting/orchestrator.py` | Main orchestrator with finalize_bets |
| StakeCalculator | `src/betting/stake_calculator.py` | Fractional Kelly stake sizing |
| DrawdownController | `src/betting/drawdown_controller.py` | Bankroll drawdown management |
| GateKeeper | `src/betting/gate_keeper.py` | EV edge threshold filter |
| LateMoneyFilter | `src/betting/late_money_filter.py` | Late odds movement detection |
| MetaSwitcher | `src/betting/meta_switcher.py` | Strategy parameter switching |
| PlaceStrategy | `src/betting/place_strategy.py` | Place bet generation |
| WideStrategy | `src/betting/wide_strategy.py` | Wide bet generation |
| WinStrategy | `src/betting/win_strategy.py` | Win bet generation |

## Data Directory Structure

```
data/
├── raw/
│   ├── races.parquet          # Race metadata (2015-2025)
│   ├── entries.parquet        # Horse entries with results
│   ├── payouts.parquet        # Payout data (win/place/wide)
│   ├── horses.parquet         # Horse master data
│   ├── kisyu_seiseki.parquet  # Jockey statistics
│   ├── chokyo_seiseki.parquet # Trainer statistics
│   ├── horse_career_stats.parquet  # Pre-computed career stats
│   ├── sire_career_stats.parquet   # Pre-computed sire stats
│   └── keito.parquet          # Bloodline code master
├── odds/
│   ├── odds_tanpuku.parquet   # Single/place odds snapshots
│   ├── odds_wide.parquet      # Wide odds
│   ├── jodds_tanpuku/         # Time-series odds (year/month partitions)
│   │   └── year=YYYY/month=M/*.parquet
│   └── time_series/           # Legacy time-series (deprecated, fallback)
├── features/
│   └── horse_features.parquet # Feature cache
├── models/                    # Production trained models
│   ├── meta.json              # Training metadata
│   ├── confidence_params.json # Confidence estimator params
│   ├── *.lgb                  # LightGBM model files
│   ├── *.joblib               # sklearn/ensemble model files
│   ├── *.json                 # Benter/temp scaler params
│   └── (surface-specific pairs)
├── models-backtest/           # Per-year model copies for backtest isolation
│   └── YYYY/
├── backtest/
│   ├── predictions/           # Backtest prediction outputs
│   └── feature_importance/    # Feature importance logs
├── paper_trading/
│   ├── bets/                  # Paper trading bet records
│   ├── predictions/           # Paper trading predictions
│   ├── daily_summary/         # Daily ROI summaries
│   ├── model/                 # Paper trading model snapshot
│   └── dry_run/               # Dry run outputs
└── etl_state.json             # ETL incremental state
```

## Test Structure

**Location:** `tests/` (flat directory, no subdirectories)

**Naming:** `test_<module_name>.py`

**Pattern:** 88 test files, one per source module or cross-cutting concern.

**Test execution:**
```bash
python -m pytest tests/ -v              # All tests
python -m pytest tests/test_domain.py   # Single module
python -m pytest tests/ --cov=src       # With coverage
```

**Test dependencies:** All tests use `unittest.mock` -- no database or external service required. `pythonpath = [".", "src"]` configured in `pyproject.toml`.

## Import Path Conventions

**Python path configuration** (from `pyproject.toml`):
```toml
[tool.pytest.ini_options]
pythonpath = [".", "src"]
```

**Import patterns used throughout:**
```python
# Data layer
from db.parquet_store import ParquetStore
from db.readers import load_races, load_entries, load_odds_snapshots
from db.connection import DatabaseConnection
from db.etl import _compute_race_date, _compute_race_id

# Domain
from domain.types import BetType, Surface, RegimeState, RecoveryState
from domain.models import TrainedModelsV5, SubmodelSet, Bet, Race, Entry

# Features
from features.feature_engine import FeatureEngine
from features.horse_history_features import HorseHistoryFeatures

# Models
from models.stage1_ability_model import AbilityModel
from models.two_stage_return_model import WinTwoStageModel, PlaceTwoStageModel
from models.market_model import MarketModel
from models.regime_detector import RegimeDetector

# Betting
from betting.stake_calculator import StakeCalculator
from betting.drawdown_controller import DrawdownController
```

**TYPE_CHECKING pattern:** Heavy use of `if TYPE_CHECKING:` for circular import avoidance in `src/domain/models.py` and `src/backtest/race_predictor.py`.

**Deferred imports:** Some heavy imports are done inside function bodies (e.g., `from features.horse_history_features import HorseHistoryFeatures` inside `TrainingPipelineV5._train_submodel()`).

## Naming Conventions

**Files:**
- Python modules: `snake_case.py`
- Test files: `test_<module_name>.py`
- Scripts: `snake_case.py` (run_*.py for entry points)
- Config: `snake_case.yaml`
- Model files: `<model_name>_<surface>.lgb` or `.joblib`

**Directories:**
- Source packages: `snake_case` (no hyphens)
- Data partitions: `year=YYYY/month=M/` (Hive-style partitioning)

## Where to Add New Code

**New Feature Module:**
- Implementation: `src/features/<feature_name>.py`
- Import in: `src/features/feature_engine.py` (if used during base feature build)
- Import in: `src/pipelines/training_pipeline.py:_train_submodel()` (if used during model training)
- Import in: `src/backtest/engine.py:run()` (if needed during backtest)
- Tests: `tests/test_<feature_name>.py`

**New Model:**
- Implementation: `src/models/<model_name>.py`
- Add to: `src/domain/models.py:SubmodelSet` (new field)
- Add to: `src/pipelines/training_pipeline.py:_train_submodel()` (training logic)
- Add to: `src/pipelines/training_pipeline.py:_save_models_local()` (save logic)
- Add to: `src/db/model_loader.py:_load_from_local()` (load logic)
- Add to: `src/backtest/race_predictor.py:predict()` (inference chain)
- Tests: `tests/test_<model_name>.py`

**New Betting Strategy:**
- Implementation: `src/betting/<strategy_name>.py`
- Register in: `src/betting/__init__.py`
- Add to: `src/backtest/race_predictor.py:select_bets()` (if needed)
- Tests: `tests/test_<strategy_name>.py`

**New Script:**
- Implementation: `scripts/run_<name>.py`
- Must include: `sys.path.insert(0, ROOT)` and `sys.path.insert(0, os.path.join(ROOT, "src"))`

**New Parquet Data Source:**
- Add table config: `config/etl_tables.yaml`
- Add reader function: `src/db/readers.py`
- Add type rules: `src/db/etl.py:_TABLE_TYPE_RULES`
- Add `_STRING_COLUMNS` exclusion: `src/db/readers.py:_STRING_COLUMNS` (if string column)

## Special Directories

**`data/`:**
- Purpose: All Parquet data files and model artifacts
- Generated: Yes (by ETL and training scripts)
- Committed: No (should be in .gitignore)

**`mlruns/`:**
- Purpose: MLflow experiment tracking data
- Generated: Yes (by training scripts)
- Committed: No

**`src/keiba_ai.egg-info/`:**
- Purpose: Setuptools package metadata
- Generated: Yes (`pip install -e .`)
- Committed: No

**`data/models-backtest/`:**
- Purpose: Isolated model copies per backtest year (avoids overwriting production models)
- Generated: Yes (by `scripts/run_backtest.py`)
- Committed: No

---

*Structure analysis: 2026-05-02*
