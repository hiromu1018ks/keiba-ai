# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

競馬AI予測システム v6.0 -- 統計的 horse racing prediction system (単勝/複勝/ワイド)。
LightGBM + XGBoost + CatBoost 3モデルスタッキングで ML モデルを構築し、PostgreSQL (EveryDB2/JRA-VAN DataLab) をデータソースとする。
MLflow で実験管理。設計書は `docs/design.md`。

**Milestone:** v2.2 ROI Recovery Analysis (Phases 43-46 in progress)
**Shipped:** v2.1 (42 phases, 12 milestones). See `.planning/ROADMAP.md` for full history.

## Development Environment

### Python (mise)

```bash
# Python 3.11 を mise でインストール・アクティベート
mise install
mise activate

# 依存インストール
pip install -e ".[dev]"
```

`mise.toml` で Python 3.11 を固定。`pip install -e ".[dev]"` で ruff, mypy, ipykernel も含む。

### Common Commands

```bash
# テスト実行（DB不要、全テスト mock 使用）
python -m pytest tests/ -v

# 単一テストファイル
python -m pytest tests/test_domain.py -v

# カバレッジ付き
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# リント (ruff)
ruff check src/ tests/

# フォーマット
ruff format --check src/ tests/

# 型チェック (mypy)
mypy src/
```

### Database

PostgreSQL が `localhost:5432/everydb2` で稼働前提。パスワードは環境変数 `PGPASSWORD` で上書き。
設定ファイル: `config/settings.yaml`

## Architecture

### Data Layer (Parquet-based)

```
EveryDB2外部テーブル → PostgreSQL (ETL入力のみ) → Parquetファイル群
                                                      |
                 +------------------------------------+--------------------------------+
                 |                  |                      |                           |
           ParquetStore      FeatureEngine          ModelLoader              OddsCollector
                 |                  |                      |                           |
           DataRepository    horse_features.parquet  TrainedModelsV5         pre-post snapshots
                 |                  |                      |                           |
                 +------------------+----------------------+---------------------------+
                                    |
                          TrainingPipelineV5 / BacktestEngine / PaperPredictor
                                    |
                              RacePredictor (shared inference)
```

### Package Layout (16 src/ packages)

| Package | Key Classes / Functions | Description |
|---------|------------------------|-------------|
| `src/domain/` | `Surface`, `BetType`, `RecoveryState`, `RegimeState`, `POST_RACE_COLS`, `Race`, `Entry`, `Bet`, `TrainedModelsV5`, `SubmodelSet` | Enums, type aliases, domain dataclasses |
| `src/db/` | `ParquetStore`, `DataRepository`, `DatabaseConnection` | Parquet I/O, ML data access facade, PostgreSQL ETL |
| `src/features/` | `FeatureEngine`, `TargetEncoder` + 25 feature modules | Feature engineering orchestrator + individual feature modules |
| `src/models/` | `AbilityModel`, `WinTwoStageModel`, `PlaceTwoStageModel`, `MarketModel`, `EVCorrectionModel`, `RegimeDetector`, `RaceQualityScreener`, `PlaceAbilityModel`, `ConformalEVModel`, `StackedEnsemble`, `WideTwoStageModel`, `BenterCombination`, `MarketAwareWinCalibrator`, `RaceLevelRanker`, `WinSelectionGateModel`, `WinSelectionPolicy`, `WinProfitSelector` | ML model definitions |
| `src/betting/` | `WinStrategy`, `PlaceStrategy`, `WideStrategy`, `StakeCalculator`, `DrawdownController`, `LateMoneyFilter`, `OddsBandFilter`, `GateKeeper`, `MetaSwitcher`, `EVTtailCalibrator` | Betting strategy and risk management |
| `src/backtest/` | `BacktestEngine`, `RacePredictor`, `BacktestValidationSuite`, `ParameterFreezeProtocol`, `ShadowComparisonFramework`, `ShadowDiagnosis`, `DeploymentGateEvaluator` | Backtest infrastructure + deployment safety |
| `src/pipelines/` | `TrainingPipelineV5` | Full training pipeline orchestration |
| `src/tuning/` | `tune_model`, `StrategyOptimizer` | Optuna hyperparameter + strategy optimization |
| `src/ingestion/` | `JVLinkFetcher`, `OddsCollector` | Race data and odds snapshot ingestion |
| `src/paper_trading/` | `PaperPredictor`, `RaceWatcher`, `PaperReconciler` | Live paper trading system |
| `src/automation/` | `RaceScheduler`, `SafetyGuard`, `PatVoter` | Race-day automation and safety |
| `src/monitoring/` | `ModelMonitor`, `AutoRetrainTrigger`, `SlackNotifier` | Drift detection and notification |
| `src/investment/` | `InvestmentFeatureFrameBuilder`, `InvestmentFrameCache` | Investment feature schema (94 specs / 9 categories) |
| `src/validation/` | `OOFHealthValidator`, `CalibratorArtifactProfile`, `RankerArtifactProfile` | OOF health + artifact profile validation |
| `src/audit/` | `FeatureRoutingAuditRegistry` | Feature routing leak detection (50+28 forbidden features) |
| `src/utils/` | `wf_splits`, `TimingContext`, `ProfileContext` | Shared utilities |

### Model Hierarchy

```
TrainedModelsV5
  |-- submodels: dict[str, SubmodelSet]  (key: "turf" | "dirt")
  |     |-- market: MarketModel
  |     |-- stage1: AbilityModel
  |     |-- win: WinTwoStageModel
  |     |-- ev_corrector: EVCorrectionModel
  |     |-- conformal_ev_model: ConformalEVModel?
  |     |-- place_ability: PlaceAbilityModel?
  |     |-- place: PlaceTwoStageModel?
  |     |-- place_ev_corrector: PlaceEVCorrectionModel?
  |     |-- place_selection_gate: PlaceSelectionGateModel?
  |     |-- wide: WideTwoStageModel?
  |     |-- benter_combo: BenterCombination?
  |     |-- isotonic_calibrator: IsotonicRegression?
  |     |-- temperature_scaler: TemperatureScaling?
  |     |-- win_selection_gate: WinSelectionGateModel?
  |     |-- win_selection_policy: WinSelectionPolicy?
  |     |-- win_profit_selector: WinProfitSelector?
  |     |-- market_aware_win_calibrator: MarketAwareWinCalibrator?  (Phase 39)
  |     +-- win_race_level_ranker: RaceLevelRanker?                  (Phase 40, shadow mode)
  |-- quality_screener: RaceQualityScreener
  +-- regime_detector: RegimeDetector
```

### Inference Pipeline (RacePredictor)

`RacePredictor.predict_race()` runs the full prediction stack per race:
1. Quality screening (RaceQualityScreener)
2. Regime detection (RegimeDetector)
3. Stage1 ability -> MarketModel error -> Win/Place EV prediction
4. EV correction (EVCorrectionModel)
5. Conformal EV intervals (ConformalEVModel)
6. MarketAwareWinCalibrator (Phase 39)
7. RaceLevelRanker scoring (Phase 40, shadow mode)
8. Win/Place selection gates -> tail calibration -> bet generation

### Parquet Files

```
data/raw/          races.parquet, entries.parquet, payouts.parquet, horses.parquet,
                   horse_career_stats.parquet, sire_career_stats.parquet,
                   chokyo_seiseki.parquet, kisyu_seiseki.parquet, keito.parquet, mining.parquet

data/odds/         snapshots.parquet, wide.parquet,
                   time_series/ (year/month partitions),
                   jodds_tanpuku.parquet, jodds_umaren.parquet, jodds_waku.parquet,
                   + 16 bet-type odds parquet files

data/features/     horse_features.parquet (feature cache)
data/predictions/  predictions.parquet
data/bets/         bets.parquet
data/oof/          oof_predictions.parquet, win_selection_oof.parquet
data/models/       production model artifacts (.lgb, .joblib, .json)
data/models-backtest/  per-year backtest model cache
data/paper_trading/    predictions, bets, daily_summary
data/backtest/     bt_{year}_*.csv/parquet, wf results, shadow artifacts
data/audit/        feature routing + tier reports
data/validation/   validation_report.json
data/baseline/     ic_baseline.json
data/strategy_manifest.json
```

全テーブルに `race_date` (datetime64) 列を含む。
`race_id` は `_compute_race_id()` でpandas計算（PostgreSQL GENERATED COLUMN不使用）。

### Import Path Conventions

```python
# pythonpath = [".", "src"] 設定済み (pyproject.toml)
from db.repository import DataRepository              # MLパイプライン用データアクセス
from db.parquet_store import ParquetStore              # 低レベルParquet I/O
from db.connection import DatabaseConnection           # PostgreSQL ETL専用
from db.model_loader import ModelLoader                # MLflow/local model loading
from domain.types import Surface, BetType, POST_RACE_COLS
from domain.models import TrainedModelsV5, SubmodelSet, Race, Entry, Bet
from features.feature_engine import FeatureEngine
from models.market_aware_win_calibrator import MarketAwareWinCalibrator
from models.race_level_ranker import RaceLevelRanker
from backtest.race_predictor import RacePredictor
from backtest.shadow_comparison import ShadowComparisonFramework
from backtest.deployment_gates import evaluate_deployment_gates
from validation.oof_health_validator import OOFHealthValidator
from audit.feature_routing_registry import run_feature_audit
from utils.wf_splits import walk_forward_race_splits
```

## Key Design Decisions

- **2-stage model**: P(hit) x E(odds|hit) decomposition eliminates zero-inflation bias. Stage1=ability model, Stage2=return-on-hit regression
- **Market Model**: Only log_error deltas pass to Stage2. Raw p_market_pred excluded from downstream features
- **EV correction**: P-correction (binary objective, init_score=logit(p_pred)) x E-correction (weight=1/sqrt(p)) as two separate models
- **RegimeDetector**: 3-state classifier (aggressive/conservative/collapsed) via market indicators (fav_rate x overround), with hysteresis transition control
- **Submodels**: Turf/Dirt 2-split only, with distance-band one-hot features via SubModelManager
- **MarketAwareWinCalibrator** (Phase 39): Replaced WinBenterGate + WinSegmentCalibrator with single Benter logit-blend (51-dim segment conditioning via L2-regularized LogisticRegression). Prevents double-correction
- **RaceLevelRanker** (Phase 40): Ridge-based relevance+value scoring in shadow mode. investment_score = 0.35*rel + 0.35*val + 0.20*log_ev - 0.10*uncertainty
- **Shadow-first deployment**: New pipeline components stay in shadow mode until all quality gates (probability quality, bet count, reproducibility, diagnostics) pass via DeploymentGateEvaluator
- **Feature routing audit**: 50 calibrator + 28 ranker forbidden features verified by CI to prevent information leakage into target models
- **OOF health validation**: fail-fast OOF validation with SHA256 manifest verification before downstream consumption
- **Win EV/Odds Safety Filter** (Phase 43.5): `--min-win-ev 1.03 --min-win-odds 3.0` as provisional default. Decision-time filter (5min odds, win_selection_ev) that excludes low-quality bets. NOT a profit maximizer -- removes loss sources (filter-off ROI 94.0% → filter-on 102.37%). Tiered stake rejected (DD 1.5-1.8x worse). CLI defaults stay 0.0; specify via strategy manifest or CLI args. Validated: offline post-hoc and actual BT Jaccard 100%, 2,025 bets, profit +4,790

## Pipeline Scripts

### Core Pipeline

#### run_etl.py -- ETL (PostgreSQL -> Parquet)

```bash
python scripts/run_etl.py --mode full --start 20140101 --end 20251231
python scripts/run_etl.py --mode delta
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | `full`\|`delta` | **required** | `full`=full extraction, `delta`=incremental merge |
| `--start` | YYYYMMDD | -- | Start date (required for `full`) |
| `--end` | YYYYMMDD | -- | End date (required for `full`) |
| `--tables` | list[str] | -- | Target tables (default: all) |

Runtime: ~10 min (full)

#### run_train.py -- Model Training

```bash
python scripts/run_train.py --start 20200101 --end 20231231 --ensemble
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--start` | YYYYMMDD | **required** | Training start date |
| `--end` | YYYYMMDD | **required** | Training end date |
| `--ensemble` | flag | False | StackedEnsemble (LGBM+XGB+CB->Ridge) |
| `--experiment` | str | `keiba-v5` | MLflow experiment name |

Runtime: ~17 min / Output: MLflow models + `data/features/horse_features.parquet`

#### run_backtest.py -- Backtest Simulation

Trains then tests on historical data. Retrains each time for reproducibility. Models saved to `data/models-backtest/`.

```bash
# Single year
python scripts/run_backtest.py \
  --train-start 20200101 --train-end 20231231 \
  --test-start 20240101 --test-end 20241231 --ensemble

# Multi-year
python scripts/run_backtest.py --years 2023 2024 2025 --train-window 4 --ensemble

# With Optuna-optimized manifest
python scripts/run_backtest.py \
  --ensemble --strategy-manifest data/strategy_manifest.json \
  --train-start 20200101 --train-end 20231231 \
  --test-start 20240101 --test-end 20241231
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--train-start` | YYYYMMDD | -- | Train start (single-year mode) |
| `--train-end` | YYYYMMDD | -- | Train end (single-year mode) |
| `--test-start` | YYYYMMDD | -- | Test start (single-year mode) |
| `--test-end` | YYYYMMDD | -- | Test end (single-year mode) |
| `--years` | list[int] | -- | Test years (multi-year mode) |
| `--train-window` | int | 4 | Training years (multi-year mode) |
| `--ensemble` | flag | False | Enable stacked ensemble |
| `--betting-mode` | `flat`\|`kelly` | `flat` | `flat`=100yen fixed, `kelly`=Fractional Kelly |
| `--betting-target` | `win`\|`place`\|`wide` | `win` | Bet target |
| `--report` | flag | False | HTML report + parquet output |
| `--skip-train` | flag | False | Skip training (cached models, requires `--ensemble`) |
| `--profile` | flag | False | pyinstrument profiling (`data/profiles/`) |
| `--strategy-manifest` | str | -- | Optuna manifest JSON (requires `--ensemble`) |
| `--calibration-bt` | flag | False | Lightweight BT for OddsBandFilter calibration (last 12 months) |
| `--min-win-ev` | float | 0.0 | Min EV threshold for win bet safety filter (provisional: 1.03) |
| `--min-win-odds` | float | 0.0 | Min odds threshold for win bet safety filter (provisional: 3.0) |
| `--win-ev-stake-threshold` | float | 0.0 | EV threshold for tiered stake (experimental, NOT recommended) |
| `--win-ev-stake-multiplier` | float | 1.0 | Stake multiplier above threshold (experimental, NOT recommended) |

**Mode selection:** `--years` -> multi-year; 4 train/test args -> single-year. One required.
**Output:** `backtest_result.json`, `data/validation/validation_report.json`, `data/backtest/bt_{year}_*.{csv,parquet}`
Runtime: ~41 min/year (no manifest), ~57 min/year (with manifest)

#### run_wf_validation.py -- Walk-Forward Validation

2-fold WF for overfitting detection. Measures feature importance stability (Spearman rho) and ROI gap.

```bash
python scripts/run_wf_validation.py --ensemble
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ensemble` | flag | False | Enable stacked ensemble |
| `--betting-target` | `win`\|`place`\|`wide` | `win` | Bet target |
| `--profile` | flag | False | pyinstrument profiling |

**Fold definitions (hardcoded):** Fold0=train 2020-2023/test 2024, Fold1=train 2021-2024/test 2025
**Output:** `data/backtest/wf_validation_result.json`
Runtime: ~2h/fold (~4h total)

#### run_strategy_optimization.py -- Optuna Strategy Parameter Optimization

16-dimensional strategy parameter optimization via Optuna TPE.

```bash
# Single seed
python scripts/run_strategy_optimization.py \
  --n-trials 100 --models-dir data/models-backtest \
  --output data/strategy_manifest.json

# Multi-seed stability verification
python scripts/run_strategy_optimization.py \
  --n-trials 100 --seeds 42,43,44 \
  --models-dir data/models-backtest \
  --output data/stability/strategy_manifest.json
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--n-trials` | int | 100 | Optuna trials |
| `--seed` | int | 42 | TPESampler seed |
| `--models-dir` | str | `data/models` | Trained model directory |
| `--output` | str | `data/strategy_manifest.json` | Output manifest path |
| `--min-bets` | int | 1000 | Min bets per fold |
| `--seeds` | str | -- | Multi-seed stability (comma-separated, e.g. `42,43,44`) |

Runtime: ~2.5h/trial

#### run_tuning.py -- Hyperparameter Tuning

Optuna tuning for individual models.

```bash
python scripts/run_tuning.py --model win_hit --start 20200101 --end 20231231 --trials 50
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | `win_hit`\|`win_return`\|`place_hit`\|`place_return`\|`ability` | **required** | Target model |
| `--start` | YYYYMMDD | **required** | Training start |
| `--end` | YYYYMMDD | **required** | Training end |
| `--trials` | int | 50 | Optuna trials |

**Output:** `data/tuning/{model}_best_params.json`
Runtime: ~30 min (50 trials)

### Diagnostics & Analysis

#### run_paper_trading.py -- Paper Trading

Live paper trading with 5 modes.

```bash
python scripts/run_paper_trading.py --mode setup --date 2026-05-30
python scripts/run_paper_trading.py --mode predict --ensemble
python scripts/run_paper_trading.py --mode reconcile
python scripts/run_paper_trading.py --mode dry-run --start 2026-04-01 --end 2026-04-30
python scripts/run_paper_trading.py --mode diagnose --start 2026-04-01 --end 2026-04-30
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | `setup`\|`predict`\|`reconcile`\|`dry-run`\|`diagnose` | **required** | Operation mode |
| `--date` | YYYY-MM-DD | -- | Target date (setup/predict) |
| `--start` | YYYY-MM-DD | -- | Period start (diagnose/dry-run) |
| `--end` | YYYY-MM-DD | -- | Period end (diagnose/dry-run) |
| `--run-id` | str | -- | MLflow run ID (default: latest) |
| `--ensemble` | flag | False | Use ensemble models |
| `--minutes-before` | int | -- | Minutes before post for odds snapshot |

#### run_shadow_comparison.py -- Shadow Model Comparison (Phase 41)

Compares baseline vs shadow model on fixed test folds.

```bash
python scripts/run_shadow_comparison.py \
  --baseline-root data/models-backtest \
  --shadow-root data/models-backtest \
  --folds 2024 2025 --report
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--baseline-root` | str | **required** | Baseline model directory |
| `--shadow-root` | str | **required** | Shadow model directory |
| `--folds` | list[int] | `[2024, 2025]` | Test fold years |
| `--train-window` | int | 4 | Training window years |
| `--betting-target` | `win`\|`place`\|`wide` | `win` | Bet target |
| `--output-dir` | str | `data/backtest/shadow` | Output directory |
| `--report` | flag | False | Generate HTML report |
| `--baseline-name` | str | `baseline` | Baseline variant name |
| `--shadow-name` | str | `shadow` | Shadow variant name |
| `--betting-mode` | `flat`\|`kelly` | `flat` | Betting mode |

**Output:** JSON metrics, Parquet race/horse diffs, CSV diff, HTML report, SHA256 manifest

#### run_shadow_diagnosis.py -- Shadow Diagnosis (Phase 43)

3-step progressive exclusion diagnosis on shadow comparison artifacts.

```bash
python scripts/run_shadow_diagnosis.py --input-dir data/backtest/shadow --report
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input-dir` | str | **required** | Shadow comparison artifact directory |
| `--output-dir` | str | `data/backtest/shadow` | Output directory |
| `--report` | flag | False | Generate HTML report |

**Output:** JSON/Markdown/HTML diagnostic report

#### run_feature_routing_audit.py -- Feature Routing Audit (Phase 42)

Verifies calibrator/ranker features are not leaking into target models.

```bash
python scripts/run_feature_routing_audit.py --output-dir data/audit
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output-dir` | str | `data/audit` | Output directory |
| `--registry-version` | str | -- | Registry version filter |

**Output:** JSON + Markdown audit reports

#### run_gpd.py -- Gain per Depth Diagnostic

Per-model charts showing gain contribution by tree depth and feature category.

```bash
python scripts/run_gpd.py --models-dir data/models --output-dir data/gpd --ensemble
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--models-dir` | str | `data/models` | Model directory |
| `--output-dir` | str | `data/gpd` | Output directory |
| `--ensemble` | flag | False | Use ensemble models |

**Output:** `data/gpd/gpd_report.json` + PNG charts

#### run_ic_eval.py -- Information Coefficient Evaluation

IC evaluation for OOF predictions.

```bash
python scripts/run_ic_eval.py data/oof/oof_predictions.parquet --output data/baseline/ic_baseline.json
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| positional | str | **required** | OOF predictions parquet path |
| `--output` | str | -- | Output JSON path |
| `--mlflow` | flag | False | Log to MLflow |
| `--experiment` | str | -- | MLflow experiment name |

### Precomputation & Utility Scripts

| Script | Usage | Description |
|--------|-------|-------------|
| `precompute_career_stats.py` | `python scripts/precompute_career_stats.py` | PIT-safe horse career statistics -> `data/raw/horse_career_stats.parquet` |
| `precompute_sire_stats.py` | `python scripts/precompute_sire_stats.py` | PIT-safe sire offspring statistics -> `data/raw/sire_career_stats.parquet` |
| `freeze_feature_manifest.py` | `python scripts/freeze_feature_manifest.py` | Freeze all 12 models' FEATURE_COLS to JSON with SHA256 hashes |
| `prune_noise_features.py` | `python scripts/prune_noise_features.py [--apply] [--full-bt]` | Tier 1 noise feature pruning with OOF safety check |
| `analyze_feature_importance.py` | `python scripts/analyze_feature_importance.py [--all-models]` | SHAP/gain/permutation importance analysis |

### Analysis Scripts (ad-hoc)

| Script | Description |
|--------|-------------|
| `analyze_odds_movement.py` | Odds movement analysis (Steamer/Stable/Drifter) |
| `analyze_high_odds.py` | High-odds hit pattern analysis (Cohen's d + TreeSHAP) |
| `analyze_loss_segments.py` | 12-dimension backtest loss segment analysis |
| `analysis_distribution_shift.py` | Distribution shift and leak detection between train/BT data |
| `compare_bt_pt_features.py` | BT(2025) vs PT(2026/4) feature distribution comparison |
| `diagnose_phase36_diff.py` | Diff diagnostic between baseline and current backtest |
| `scrape_everydb2_manual.py` | Playwright-based scraper for EveryDB2 documentation |

## Code Style

- Ruff: target py311, line-length=100, rules=E/F/I/N/W
- Mypy: `disallow_untyped_defs = true` (all functions require type annotations)
- Tests: DB not required (all use `unittest.mock`)
- Commit messages: Conventional Commits (Japanese)

## Configuration

| File | Contents |
|------|----------|
| `config/settings.yaml` | Database connection, data paths, logging, feature_engine (steeple exclusion), late_money thresholds, submodel surface/distance_band definitions, betting_strategy defaults (Kelly fraction, EV targets, regime fractions) |
| `config/backtest_config.yaml` | Walk-forward parameters, holdout period, pass criteria (ROI targets, max drawdown), EV correction improvement thresholds, validation constraints |
| `config/etl_tables.yaml` | 103 ETL table mappings (53 n_ tables for full extraction, 50 s_ tables for delta merge) |
| `pyproject.toml` | Project metadata, dependencies, pytest/ruff/mypy configuration |

DB connection uses `_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent` for project root resolution.

## Important Notes

### Data Pipeline

- PostgreSQL GENERATED columns (`distance_band`, `surface` via `track_cd`) are NOT included in Parquet ETL -> recomputed in `FeatureEngine._map_basic_features()` via Python
- `race_id` is computed via `_compute_race_id()` in pandas (not a PostgreSQL GENERATED COLUMN)
- `run_multi_year_backtest.py` is deprecated. Use `run_backtest.py --years` instead

### Safety & Reproducibility Infrastructure

- **POST_RACE_COLS** (`domain/types.py`): 41 columns forbidden as ML features (3-layer CI detection)
- **OOFHealthValidator** (`validation/oof_health_validator.py`): fail-fast OOF validation with SHA256 manifest
- **FeatureRoutingAuditRegistry** (`audit/feature_routing_registry.py`): 50+28 forbidden features CI check
- **ParameterFreezeProtocol** (`backtest/parameter_freeze_protocol.py`): JSON manifest + SHA256 tamper detection
- **DeploymentGateEvaluator** (`backtest/deployment_gates.py`): 4-gate evaluation (probability quality, bet count, reproducibility, diagnostics). Report-only; never modifies deployment_status

### Known Issues

- BT ROI: 87.8% (v2.0 close). v1.7 achieved 97.8%. v2.2 targets recovery via structural fix
- Win EV/Odds safety filter (`--min-win-ev 1.03 --min-win-odds 3.0`) improves BT ROI to 102.37% (2024-2025, 2,025 bets, +4,790 profit). Still provisional; 2024 year is -5,860
- `test_training_pipeline.py` has 3 known failures
- `training_pipeline._build_race_level_features()` rl_* column processing not fully integrated
- Turf conservative regime unprofitable -- largest improvement opportunity
- Paper trading diagnose mode uses Parquet readers (not EveryDB2) for historical inference

### Phase History Summary

| Milestone | Phases | Key Deliverable |
|-----------|--------|----------------|
| v1.0 | 1-4 | Win model: feature analysis + calibration + selection gate + WF validation |
| v1.1 | 5-7 | ROI advanced: EMA features + odds deviation + 3-model stacking |
| v1.2 | 8-10 | Win backtest: settlement fix + bet history + pipeline optimization |
| v1.3 | 11-13 | Betting strategy: EV_lower + OddsBandFilter + DD control + Optuna 16-dim |
| v1.4 | 14-18 | Ensemble filter: WinSelectionGate retrain + dynamic EV_lower + drift diagnostics |
| v1.5 | 19-22 | Model accuracy: Isotonic EV calibration + high-odds 18 features + CQR intervals |
| v1.6 | 23-28 | Feature overhaul: POST_RACE leak removal + 22 new features + target encoding |
| v1.7 | 29-34 | Market-independent edge: race-level features + Harville cross + IC/GPD frameworks |
| v1.8 | 35-36.1.1 | Turf precision: haron/lap features + relative features + MarketModel fix |
| v2.0 | 37-38 | Investment pipeline: OOFHealthValidator + InvestmentFeatureFrame (94 specs) |
| v2.1 | 39-42 | MAWC + Ranker (shadow mode) + Shadow Comparison + Deployment Gates |
| v2.2 | 43-46 | ROI Recovery Analysis (in progress) |
