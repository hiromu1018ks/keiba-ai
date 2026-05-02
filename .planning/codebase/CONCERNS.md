# Codebase Concerns

**Analysis Date:** 2026-05-02

## Tech Debt

### PlaceSelectionGateModel: God Class at 1,401 Lines

- Issue: `PlaceSelectionGateModel` is the largest file in the codebase (1,401 lines) with 17+ instance attributes, multiple score table builders, OOF walk-forward logic, threshold grid search, and expansion scoring. It combines training, scoring, threshold optimization, and candidate ranking in a single class.
- Files: `src/models/place_selection_gate.py`
- Impact: Difficult to test individual components. High cognitive load for modifications. Any change to scoring risks breaking threshold optimization.
- Fix approach: Extract `ScoreTableBuilder`, `ThresholdOptimizer`, and `ExpansionScorer` as separate classes. Keep `PlaceSelectionGateModel` as a facade that delegates.

### TrainingPipelineV5: 1,045-Line Monolith with Implicit State

- Issue: `TrainingPipelineV5` stores `self._race_df` and `self._entry_df` on the instance during `run()` for use by `_train_submodel()`. This implicit state coupling means `_train_submodel()` cannot be called independently or tested in isolation without first running `run()`.
- Files: `src/pipelines/training_pipeline.py` (lines 119-121)
- Impact: Cannot unit-test `_train_submodel` without full pipeline setup. State leakage between calls if `run()` is invoked multiple times.
- Fix approach: Pass `race_df` and `entry_df` as explicit parameters to `_train_submodel()`.

### EveryDB2Queries: Unverified SQL Queries

- Issue: EveryDB2Queries contains 5 `TODO` comments indicating that table names and column names are unverified guesses (e.g., `s_bataijyu`, `s_odds_tanpuku`, `n_race`). All queries use hardcoded table names that have never been validated against the actual EveryDB2 schema.
- Files: `src/db/everydb2_queries.py` (lines 49, 83, 103, 124, 147)
- Impact: Every method in this class will fail at runtime in production. The paper trading and live betting automation paths are entirely non-functional.
- Fix approach: Run schema discovery queries (`SELECT table_name FROM information_schema.tables`) against the actual EveryDB2 instance and update all queries. Add integration tests.

### Validation Suite Incomplete Metrics

- Issue: Two metrics (`logloss` and `spearman_rho`) are hardcoded to `None` with `TODO` comments. Walk-forward validation reports are incomplete without these.
- Files: `src/backtest/validation_suite.py` (lines 610-611)
- Impact: Cannot compare model predictive quality across runs via logloss. No rank-correlation metric for prediction ordering quality.
- Fix approach: Compute logloss from predictions and spearman_rho from rank comparison against actual finish order.

### Broad Exception Handling Throughout

- Issue: The codebase uses `except Exception:` in 35+ locations, especially in `src/db/everydb2_queries.py` (20 occurrences) and `src/db/model_loader.py` (14 occurrences). While logging is present, the pattern silently swallows errors and returns empty/None defaults.
- Files: `src/db/everydb2_queries.py`, `src/db/model_loader.py`
- Impact: Production failures in data loading or model loading are masked. The system continues with degraded data rather than failing fast.
- Fix approach: For non-optional data paths, catch specific exceptions (e.g., `psycopg2.OperationalError`, `FileNotFoundError`) and re-raise with context. Reserve `except Exception` only for fallback paths where degraded operation is acceptable.

## Known Bugs

### Password Embedded in Connection URL

- Symptoms: Database password is read from `config/settings.yaml` and interpolated into a connection URL string in `DatabaseConnection.__init__`.
- Files: `src/db/connection.py` (lines 41-49)
- Trigger: Password stored in `config/settings.yaml` under `database.password` key. If `PGPASSWORD` env var is not set, the YAML value is used.
- Workaround: Always set `PGPASSWORD` environment variable.

### BacktestEngine.run() Uses f-string Logging Without Lazy Evaluation

- Symptoms: `logger.warning(f"No races found in {test_start} ~ {test_end}")` evaluates the f-string even when the log level is above WARNING.
- Files: `src/backtest/engine.py` (line 232), `src/pipelines/training_pipeline.py` (line 101, 194, 205)
- Impact: Minor performance overhead. Not a bug per se, but violates Python logging best practices and makes log-level changes at runtime ineffective for cost savings.
- Workaround: None needed (cosmetic).

## Security Considerations

### Database Password in Configuration File

- Risk: `config/settings.yaml` may contain database credentials in plaintext. The code at `src/db/connection.py:41` reads `db.get("password", "")` directly from YAML.
- Files: `src/db/connection.py`, `config/settings.yaml`
- Current mitigation: `PGPASSWORD` environment variable can override the YAML value.
- Recommendations: (1) Remove password from `settings.yaml` entirely. (2) Require `PGPASSWORD` env var and fail fast if not set. (3) Add `config/settings.yaml` to `.gitignore` or ensure it is templated.

### No SQL Injection Protection in EveryDB2Queries

- Risk: `EveryDB2Queries._query()` accepts raw SQL strings. While current code uses parameterized queries (`%s` placeholders with params), the architecture allows arbitrary SQL to be passed.
- Files: `src/db/everydb2_queries.py` (line 35)
- Current mitigation: All current callers use parameterized queries.
- Recommendations: Add type hints restricting SQL to specific query types. Consider using SQLAlchemy text() for additional safety.

## Performance Concerns

### BacktestEngine: Per-Race DataFrame Filtering in Loop

- Problem: The main backtest loop (line 420+) filters the full `feat_df` for each `race_id` using `feat_df[feat_df["race_id"] == race_id]`. For thousands of races, this creates O(n^2) pandas boolean indexing operations.
- Files: `src/backtest/engine.py` (line 421)
- Cause: Sequential iteration over `race_ids` with full-DataFrame filter per iteration.
- Improvement path: Pre-group by `race_id` using `feat_df.groupby("race_id")` or build a dict of `{race_id: DataFrame}` before the loop.

### iterrows() Usage in Hot Paths

- Problem: `iterrows()` is used in payout map construction (`build_payout_map`, `build_wide_payout_map`), diagnostic logging, and candidate selection. These iterate row-by-row creating `pd.Series` objects per row.
- Files: `src/backtest/engine.py` (lines 112, 151, 278, 451, 545, 632, 892), `src/models/place_selection_gate.py` (lines 270, 297, 320, 444, 483), `src/backtest/race_predictor.py` (lines 583, 657)
- Cause: `iterrows()` is 100-1000x slower than vectorized operations.
- Improvement path: Replace `build_payout_map` with vectorized melt + division. Replace diagnostic logging loops with batch dict construction. For `place_selection_gate.py`, use vectorized groupby aggregation instead of iterrows over grouped DataFrames.

### Pre-Computation Loads All Features into Memory

- Problem: `BacktestEngine.run()` loads all races, entries, odds, payouts, and wide odds for the entire test period at once, then computes HorseHistoryFeatures, JockeyContextFeatures, TrainerContextFeatures, JockeyTrainerComboFeatures, SireFeatures, PaceAptitudeFeatures, and CourseFeatures for all races simultaneously.
- Files: `src/backtest/engine.py` (lines 328-405)
- Cause: No chunked or streaming processing. For multi-year backtests, this can require several GB of RAM.
- Improvement path: Process in monthly or yearly chunks. Release feature DataFrames after each race is processed.

### PlaceSelectionGateModel Row-by-Row Scoring

- Problem: `_score_frame_from_tables()` uses a Python list comprehension over every row of the DataFrame (line 402-413), calling `_score_row_from_tables()` per row. This is inherently single-threaded and slow.
- Files: `src/models/place_selection_gate.py` (lines 402-413)
- Cause: Dictionary lookups (combo_scores, pair_scores, single_scores) are not vectorizable with pandas.
- Improvement path: Convert score tables to pandas merge-based lookups using bin indices as join keys.

## Data Quality Issues

### Feature Column Availability Varies Between Train and Predict

- Issue: `PlaceAbilityModel.FEATURE_COLS` lists 45 features, but `train()` and `predict()` both filter with `available_cols = [c for c in self.FEATURE_COLS if c in df.columns]`. Different code paths (training pipeline vs. backtest engine vs. paper trading predictor) may produce different sets of available columns, leading to train/test feature skew.
- Files: `src/models/place_ability_model.py` (lines 118, 176), `src/models/stage1_ability_model.py` (similar pattern)
- Impact: Model may be trained with features that are absent during inference, or vice versa. LightGBM silently handles missing features by using default values, but this degrades prediction quality unpredictably.
- Fix approach: Log the set of missing features during both train and predict. Raise an error if critical features are absent. Add a feature availability validator.

### NaN Propagation in Multi-Stage Pipeline

- Issue: The prediction pipeline is a chain: MarketModel -> Stage1 -> PlaceAbility -> WinTwoStage -> PlaceTwoStage -> EVCorrection -> PlaceEVCorrection -> ConfidenceEstimator -> PlaceSelectionGate. If any upstream model produces NaN, all downstream models receive NaN inputs. LightGBM handles NaN natively, but intermediate pandas operations (e.g., `groupby.transform("sum")`, `clip()`) may propagate or amplify NaN in unexpected ways.
- Files: `src/backtest/race_predictor.py` (lines 88-191), `src/models/place_ability_model.py` (line 207)
- Impact: Entire races may produce all-NaN predictions if a single upstream feature is missing. The `race_sum.clip(lower=1e-6)` in PlaceAbilityModel prevents division by zero but does not fix the underlying NaN issue.
- Fix approach: Add NaN audits between pipeline stages. Log the percentage of NaN values in key columns after each model. Consider adding fallback values for critical intermediate columns.

### fillna(0) Masking Missing Data in Race-Level Features

- Issue: Multiple locations use `.fillna(0.0)` or `.fillna(0)` for race-level statistics that should never be zero (e.g., `fav_won`, `topk_hit`, `positive_return`). Zero-filling makes it impossible to distinguish "data not available" from "actual zero value."
- Files: `src/pipelines/training_pipeline.py` (lines 653, 684, 686, 687, 737)
- Impact: Model trains on synthetic zeros that don't represent real race outcomes. `hist_roi_topk` defaults to 1.0 when no data exists, implying profitable favorites when data is simply missing.
- Fix approach: Use sentinel values (e.g., -1) or explicit NaN with indicator columns. Let LightGBM's native NaN handling deal with missing values rather than imputing zeros.

### PostgreSQL GENERATED Columns Not in Parquet ETL

- Issue: `distance_band`, `surface` (derived from `track_cd`), and other PostgreSQL GENERATED columns are not included in the Parquet ETL output. `FeatureEngine._map_basic_features()` recomputes these in Python, creating a divergence risk if the PostgreSQL logic and Python logic are not kept in sync.
- Files: `src/features/feature_engine.py` (lines 270-408)
- Impact: If the PostgreSQL GENERATED column logic changes (e.g., distance band boundaries), the Python recomputation must be manually updated to match. No automated test verifies parity.
- Fix approach: Add a parity test that compares PostgreSQL-generated values with Python-computed values on a sample dataset. Document the exact SQL GENERATED column definitions alongside the Python code.

## Architecture Concerns

### Tight Coupling Between BacktestEngine and Feature Pre-Computation

- Issue: `BacktestEngine.run()` contains 80+ lines of feature pre-computation (lines 328-405) that duplicate logic from `TrainingPipelineV5._train_submodel()`. The same imports and computation patterns appear in both files.
- Files: `src/backtest/engine.py`, `src/pipelines/training_pipeline.py`
- Why fragile: Any change to feature computation order, feature names, or merge logic must be applied in both places. The two implementations can drift.
- Safe modification: Extract a shared `FeaturePreComputer` class that both `BacktestEngine` and `TrainingPipelineV5` use.
- Test coverage: No test verifies that training features and backtest features are computed identically.

### SubmodelSet Has 13 Constructor Parameters

- Issue: `SubmodelSet` dataclass has 13 fields including optional ones (`benter_combo`, `isotonic_calibrator`, `temperature_scaler`). It serves as a catch-all container for all per-surface models.
- Files: `src/domain/models.py`
- Why fragile: Adding a new model requires modifying SubmodelSet, ModelLoader, TrainingPipelineV5, and RacePredictor. The blast radius of any model addition is 4+ files.
- Safe modification: Consider using a registry pattern or typed dict to decouple model storage from model usage.

### RacePredictor Imports Inside Method Body

- Issue: `RacePredictor.predict()` imports `HorseHistoryFeatures` and `compute_interaction_features` inside the method body (lines 64-65). Similarly, it imports `traceback` inside an except block (line 91).
- Files: `src/backtest/race_predictor.py` (lines 64-65, 91)
- Why fragile: Import errors at runtime are only discovered when the specific code path executes. Circular import risks.
- Safe modification: Move imports to the module top level or use TYPE_CHECKING for type-only imports. The `import traceback` inside the except block is a standard pattern for lazy heavy imports, but `HorseHistoryFeatures` is a core dependency and should be at the top level.

## ML-Specific Concerns

### No Reproducibility Seed in LightGBM Training

- Issue: None of the LightGBM model training calls set `random_state` or `seed` parameters. The models produce different results on each training run even with identical data.
- Files: `src/models/stage1_ability_model.py`, `src/models/place_ability_model.py`, `src/models/market_model.py`, `src/models/ev_correction_model.py`, `src/models/regime_detector.py`, `src/models/two_stage_return_model.py`
- Impact: Cannot reproduce exact training results. Backtest results vary between runs with the same data. Makes A/B comparison of model changes unreliable.
- Fix approach: Add `random_state=42` (or configurable seed) to all `LGBMClassifier`, `LGBMRegressor`, and `lgb.train()` calls. Also set `deterministic=True` and `force_row_wise=True` for full reproducibility.

### Isotonic Calibration Skipped on Insufficient Data

- Issue: `PlaceAbilityModel.train()` skips isotonic calibration entirely when `len(X_calib) < 50` (line 160-171). The model falls back to raw LightGBM probabilities, which are known to be poorly calibrated.
- Files: `src/models/place_ability_model.py` (lines 160-171)
- Impact: Small datasets (e.g., dirt-only training) may produce uncalibrated predictions, leading to overconfident probability estimates and poor EV calculations downstream.
- Fix approach: Log a warning when calibration is skipped (already done). Consider using a Platt Scaling fallback which requires less data than isotonic calibration.

### Benter Combination Disabled in v5.6

- Issue: The Isotonic calibration step after Benter combination is commented out in `RacePredictor.predict()` (lines 144-148) with a note that "Isotonic post-Benter is too aggressive (pushes mean 0.224 vs true ~0.375)." The temperature scaler is still applied.
- Files: `src/backtest/race_predictor.py` (lines 144-148)
- Impact: The Benter combination's probability output is only temperature-scaled, not isotonic-calibrated. This may leave residual calibration error. The root cause (isotonic overcorrection) suggests the Benter alpha/beta fitting may need adjustment rather than skipping calibration.
- Fix approach: Investigate whether the Benter parameters (alpha, beta, gamma) are correctly fitted. Consider calibrating on a held-out set that is separate from the Benter fitting set.

### PlaceAbilityModel Race Normalization May Clip Probabilities

- Issue: The race-level normalization (line 207-214) enforces `sum(p_place) = 3` per race and then applies `clip(upper=1.0)`. For small fields (e.g., 8 horses), individual probabilities can exceed 1.0 after normalization and get clipped, causing the sum to be less than 3.
- Files: `src/models/place_ability_model.py` (lines 207-217)
- Impact: Violates the probability constraint that exactly 3 horses should place. Downstream EV calculations assume well-calibrated probabilities.
- Fix approach: Use the iterative normalization from `EVCorrectionModel._normalize_probability_array()` which caps probabilities at 1.0 and redistributes the remainder.

### RegimeDetector Labels Are Heuristic, Not Learned

- Issue: The RegimeDetector's training labels (AGGRESSIVE/CONSERVATIVE/COLLAPSED) are computed from hardcoded thresholds on `market_condition_score` and `entropy` (lines 94-101 in regime_detector.py). The LightGBM model then learns to predict these heuristic labels.
- Files: `src/models/regime_detector.py` (lines 94-101)
- Impact: The model can never be more accurate than the heuristic used to create labels. If the thresholds are wrong, the model learns wrong patterns. The thresholds (0.28, 0.50 for market_condition_score; median entropy) are not validated.
- Fix approach: Validate regime labels against actual profitability (e.g., AGGRESSIVE regime should correspond to periods where betting more aggressively is profitable). Consider unsupervised clustering (GMM) as an alternative.

### Model Train/Test Split Uses Percentage, Not Date

- Issue: Several models use `split = int(n * 0.8)` for train/calibration splits (PlaceAbilityModel, RegimeDetector). While data is sorted by date before splitting, using a percentage rather than a fixed date means the exact split point changes as data grows.
- Files: `src/models/place_ability_model.py` (line 134), `src/models/regime_detector.py` (line 106)
- Impact: Adding new training data shifts the calibration/validation boundary, potentially changing model behavior on existing data.
- Fix approach: Use a fixed calibration start date (e.g., last 6 months of training data) rather than a percentage split.

## Operational Concerns

### No Model Versioning Beyond MLflow

- Issue: Model artifacts are stored in `data/models/` (local filesystem) with MLflow as the tracking backend. The `ModelLoader` falls back to filesystem loading if MLflow fails (line 66-68). There is no model checksumming or validation that loaded models match the expected version.
- Files: `src/db/model_loader.py`
- Impact: Cannot verify that a loaded model is the correct version. Silent corruption or accidental overwrite of model files would go undetected.
- Fix approach: Add SHA256 checksums to model metadata. Verify checksums after loading. Store training data hash alongside model artifacts.

### Diagnostic Logging in BacktestEngine Duplicates Significant Code

- Issue: The diagnostic logging block (logging horse-level features for quality-passed and quality-failed races) is nearly identical between lines 544-591 and 630-678 of `backtest/engine.py`. This ~100-line block is duplicated with minor differences.
- Files: `src/backtest/engine.py` (lines 544-591, 630-678)
- Impact: Any change to logged fields must be made in two places. Risk of divergence.
- Fix approach: Extract a `_log_race_diagnostics()` method that accepts a `quality_passed` flag.

### Silent Degradation in Paper Trading Components

- Issue: Multiple paper trading and automation components return empty lists or `None` on failure: `PaperPredictor.predict()` returns `[]` on 5 different error paths, `Scheduler` returns `[]` on 3 paths, `PatVoter` returns `[]` on 2 paths.
- Files: `src/paper_trading/predictor.py`, `src/automation/scheduler.py`, `src/automation/pat_voter.py`
- Impact: Paper trading system silently produces no output when data is unavailable or errors occur. No alerting or health monitoring.
- Fix approach: Add structured error reporting (exception types or error result objects) instead of empty returns. Add health check endpoints.

## Dependency Risks

### psycopg2 Direct Usage Alongside SQLAlchemy

- Issue: `EveryDB2Queries` uses `psycopg2.connect()` directly while `DatabaseConnection` uses SQLAlchemy. This creates two independent database access patterns with different connection management.
- Files: `src/db/everydb2_queries.py`, `src/db/connection.py`
- Impact: Connection pooling and lifecycle management differ between the two modules. `EveryDB2Queries` opens a new connection for every query (no pooling).
- Fix approach: Standardize on SQLAlchemy for all database access. Use `psycopg2` only as the SQLAlchemy driver.

### LightGBM Version Sensitivity

- Issue: The codebase uses LightGBM-specific features like `FrozenEstimator` (from sklearn) and `lgb.early_stopping` callback. These APIs changed between LightGBM versions. No version pin is visible in the source.
- Files: `src/models/place_ability_model.py` (line 162), `src/models/regime_detector.py` (line 130)
- Impact: Upgrading LightGBM could break training if APIs change.
- Fix approach: Pin `lightgbm` version in `pyproject.toml` or `requirements.txt`. Test against specific versions in CI.

### sklearn.calibration.FrozenEstimator Availability

- Issue: `FrozenEstimator` was introduced in scikit-learn 1.6. If an older version is installed, `PlaceAbilityModel.train()` will fail with `ImportError` at runtime.
- Files: `src/models/place_ability_model.py` (line 162)
- Impact: Training fails on environments with scikit-learn < 1.6.
- Fix approach: Pin `scikit-learn>=1.6` or add a version check with fallback to `CalibratedClassifierCV(estimator=...)` without freezing.

## Test Coverage Gaps

### PlaceSelectionGateModel Training Path Untested

- What's not tested: `PlaceSelectionGateModel.train()` with its full OOF walk-forward fold logic, threshold grid search, and expansion scoring. The test file `tests/test_place_selection_gate.py` exists but may not cover the multi-fold OOF path.
- Files: `src/models/place_selection_gate.py`, `tests/test_place_selection_gate.py`
- Risk: Changes to score table construction or threshold optimization may silently break the gate.
- Priority: High

### End-to-End Feature Parity Between Training and Backtest

- What's not tested: No test verifies that `TrainingPipelineV5._train_submodel()` and `BacktestEngine.run()` produce identical feature sets for the same input data. Feature drift between these two paths is a silent failure mode.
- Files: `src/pipelines/training_pipeline.py`, `src/backtest/engine.py`
- Risk: Model trained on one feature set, predictions made on a different feature set. Backtest results are unreliable.
- Priority: High

### Automation and Paper Trading Modules

- What's not tested: `src/automation/pat_voter.py`, `src/automation/safety_guard.py`, `src/automation/scheduler.py`, `src/paper_trading/watcher.py` have limited or no meaningful test coverage (test files exist but may be stub-level).
- Files: `src/automation/`, `src/paper_trading/`
- Risk: Production automation failures are discovered only at runtime.
- Priority: Medium

### Live Inference Path

- What's not tested: `RacePredictor.predict()` with full model chain (MarketModel -> Stage1 -> PlaceAbility -> WinTwoStage -> EVCorrection -> Confidence -> Benter -> Gate). Individual models are tested but the integration of the full chain is only exercised through backtests.
- Files: `src/backtest/race_predictor.py`
- Risk: A change in one model's output column name breaks the entire inference chain.
- Priority: High

### Calibration Quality

- What's not tested: No automated test checks whether isotonic calibration, temperature scaling, or Benter combination actually improves calibration metrics (e.g., ECE, Brier score) on held-out data.
- Files: `src/models/place_ability_model.py`, `src/models/benter_combination.py`
- Risk: Calibration steps may be degrading rather than improving predictions.
- Priority: Medium

---

*Concerns audit: 2026-05-02*
