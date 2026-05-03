# Architecture Patterns: v1.1 ROI Advanced Model

**Domain:** Horse racing prediction -- ensemble stacking, odds deviation EV, time-series features
**Researched:** 2026-05-03
**Parent system:** keiba-ai v5.5 (LightGBM + 2-stage decomposition P(hit) x E(odds|hit))

## Recommended Architecture

The v1.1 milestone extends the existing 2-stage decomposition architecture in three targeted layers: (1) replacing single-GBM models with a 3-model stacking ensemble, (2) adding odds-deviation features that feed directly into EV calculation, and (3) introducing time-series/pace features that improve the Stage1 ability estimate. The existing pipeline structure (FeatureEngine -> AbilityModel -> TwoStageModel -> EVCorrection -> WinBenterGate -> WinSelectionGate) is preserved; only the internal implementations of certain stages change.

### Current Pipeline Flow (v5.5, unchanged foundation)

```
ParquetStore -> DataRepository -> FeatureEngine.build_all()
  -> SubModelManager (turf/dirt split)
    -> [per surface]:
       MarketModel (OOF) -> AbilityModel.train_oof() (LightGBM Ranker)
       -> PlaceAbilityModel -> WinTwoStageModel (hit + return)
       -> JockeyContext / TrainerContext / JTCombo features
       -> EVCorrectionModel (P-correction + E-correction)
       -> PlaceTwoStageModel -> BenterCombination -> Calibration
       -> WinBenterGate (Benter + calibration + temperature)
       -> WinSelectionGate (walk-forward OOF gate)
       -> RobustConfidenceEstimator
```

### v1.1 Extension Points

The three feature areas map to specific insertion points in this pipeline:

```
                         NEW/CHANGED COMPONENTS
                         ======================

[A] Stacking Ensemble (replaces single LightGBM in hit_model)
    Location: WinTwoStageModel.hit_model, PlaceTwoStageModel.hit_model
    Mechanism: StackedEnsemble class already exists (src/models/stacked_ensemble.py)
    Change: Enable use_ensemble=True in pipeline, enhance StackedEnsemble

[B] Odds Deviation EV Features (new features in FeatureEngine + TwoStageModel)
    Location: FeatureEngine.build_all() + WinTwoStageModel.FEATURE_COLS
    Mechanism: New feature module (odds_deviation_features.py)
    Change: Compute market-vs-model deviation metrics before EV correction

[C] Time-Series / Pace Features (new features in FeatureEngine)
    Location: FeatureEngine.build_all() + HorseHistoryFeatures
    Mechanism: New feature modules (time_series_features.py, pace_prediction_features.py)
    Change: Extend horse history with temporal trend features and pace prediction
```

## Component Boundaries

### Components to MODIFY (existing)

| Component | File | What Changes | Why |
|-----------|------|-------------|-----|
| StackedEnsemble | `src/models/stacked_ensemble.py` | Add Ranker support, hyperparameter tuning, model persistence | Current implementation only supports binary classification; AbilityModel uses LightGBM Ranker (lambdarank) which requires group-aware training |
| WinTwoStageModel | `src/models/two_stage_return_model.py` | Add odds-deviation features to FEATURE_COLS, support stacked hit_model | New features need to flow through the 2-stage model |
| TrainingPipelineV5 | `src/pipelines/training_pipeline.py` | Wire new feature modules, improve ensemble integration | Orchestrator must call new modules in correct order |
| FeatureEngine | `src/features/feature_engine.py` | Add calls to new feature modules in build_all() | New features must be computed during batch feature generation |
| HorseHistoryFeatures | `src/features/horse_history_features.py` | Add time-series trend features to BASE_COLS | Temporal features computed from past performance data |
| BacktestEngine | `src/evaluation/backtest_engine.py` | Support loading stacked models | Must handle joblib-pickled ensembles alongside .lgb files |

### Components to CREATE (new)

| Component | File | Responsibility | Inserted Before |
|-----------|------|---------------|----------------|
| OddsDeviationFeatures | `src/features/odds_deviation_features.py` | Compute model-vs-market probability deviation, deviation trends, deviation-adjusted EV | WinTwoStageModel training |
| TimeSeriesFeatures | `src/features/time_series_features.py` | Temporal features from past runs: time progression, closing-speed trend, consistency metrics | HorseHistoryFeatures.compute() |
| PacePredictionFeatures | `src/features/pace_prediction_features.py` | Predicted pace scenario from entry field composition, position-taking probability | AbilityModel training |
| StackedRankerEnsemble | (extend `src/models/stacked_ensemble.py`) | 3-model stacking for ranking (lambdarank) with group-aware OOF | AbilityModel.train_oof() |

### Components UNCHANGED

| Component | Why Unchanged |
|-----------|--------------|
| ParquetStore / DataRepository | Data layer is stable; new features derive from existing data |
| EVCorrectionModel | Operates downstream of TwoStageModel; gets better inputs automatically |
| WinBenterGate | Benter combination already works with any probability source |
| WinSelectionGate | Walk-forward gate learns from realized ROI; improved model inputs help automatically |
| RegimeDetector | Market regime detection uses pre-race features only |
| BettingOrchestrator / WinStrategy | Betting logic is downstream; benefits from better predictions |
| SubModelManager | turf/dirt split remains the same |

## Data Flow Changes

### Current Flow (simplified, per surface)

```
raw data -> FeatureEngine.build_all()
  -> HorseHistoryFeatures.compute()    [past performance stats]
  -> compute_odds_dynamics()           [odds velocity/volatility]
  -> compute_market_bias()             [market entropy, overround]
  -> MarketModel.train() + OOF         [log_error features]
  -> AbilityModel.train_oof()          [p_ability_win via LightGBM Ranker]
  -> WinTwoStageModel.train_hit_model() [p_win via LightGBM binary]
  -> WinTwoStageModel.train_return_model() [E(odds|win) via LightGBM regression]
  -> EVCorrectionModel.train()         [P/E correction]
```

### New Flow (v1.1, additions in **bold**)

```
raw data -> FeatureEngine.build_all()
  -> HorseHistoryFeatures.compute()    [past performance stats]
  **-> TimeSeriesFeatures.compute()**   [time progression, closing-speed trend]
  **-> PacePredictionFeatures.compute()** [predicted pace, position probability]
  -> compute_odds_dynamics()           [odds velocity/volatility]
  -> compute_market_bias()             [market entropy, overround]
  -> MarketModel.train() + OOF         [log_error features]
  -> AbilityModel.train_oof()
     **[uses StackedRankerEnsemble if enabled]** [LightGBM + XGBoost + CatBoost Ranker]
  **-> OddsDeviationFeatures.compute()** [model-vs-market deviation at this stage]
  -> WinTwoStageModel.train_hit_model()
     **[uses StackedEnsemble if enabled]** [LightGBM + XGBoost + CatBoost binary]
  -> WinTwoStageModel.train_return_model()
  -> EVCorrectionModel.train()         [P/E correction, now sees deviation features]
```

### Key Data Dependency Chain

```
TimeSeriesFeatures      -- requires --> HorseHistoryFeatures (past run data)
PacePredictionFeatures  -- requires --> entry_df (field composition)
OddsDeviationFeatures   -- requires --> p_ability_win (from AbilityModel)
                           requires --> p_market_win_adj (from market_bias)
StackedRankerEnsemble   -- requires --> same features as AbilityModel
StackedEnsemble (binary) -- requires --> same features as WinTwoStageModel.hit_model
```

This dependency chain determines the build order. Time-series and pace features can be computed early (they depend only on raw data and past performance). Odds deviation features must wait until after AbilityModel produces p_ability_win.

## Patterns to Follow

### Pattern 1: Feature Module Pattern

**What:** Each feature group lives in its own module under `src/features/`, follows the same interface pattern.
**When:** Adding any new feature group.
**Example:**

```python
# src/features/time_series_features.py

def compute_time_series_features(
    df: pd.DataFrame,
    hist_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute temporal trend features from past performance data.

    Args:
        df: Base DataFrame with race_id, umaban, kettonum columns
        hist_df: Pre-loaded history DataFrame (optional, loaded from ParquetStore if None)

    Returns:
        DataFrame with new feature columns, merged on (race_id, umaban)
    """
    df = df.copy()
    # ... compute features ...
    return df
```

This pattern is already established by `compute_odds_dynamics()`, `compute_market_bias()`, `compute_intra_race_features()`, etc. New modules must follow it.

### Pattern 2: Stacked Ensemble Drop-In Replacement

**What:** The StackedEnsemble class replaces `lgb.Booster` via duck typing (`.predict()` method + `.best_iteration` attribute).
**When:** Using ensemble mode in the pipeline.
**Example:**

```python
# Current (single model):
hit_model = lgb.train(params, train_data, num_boost_round=500)
p = hit_model.predict(features, num_iteration=hit_model.best_iteration)

# Ensemble (drop-in replacement):
ensemble = StackedEnsemble(cat_cols=["surface", "distance_bin"])
ensemble.train(X_train, y_train, X_valid, y_valid)
p = ensemble.predict(X)  # same signature
```

The existing `TrainingPipelineV5._train_submodel()` already handles this via `use_ensemble=True` flag (lines 439-454). The StackedEnsemble class must maintain `.best_iteration = 0` and a compatible `.predict()` signature.

### Pattern 3: PIT (Point-in-Time) Feature Computation

**What:** All features must be computed using only data available before the target race. No future leakage.
**When:** Computing any feature from historical data.
**Example:** HorseHistoryFeatures uses `searchsorted` to find data before `target_date`, and expanding window statistics that only look backward.

New time-series features MUST follow this pattern:
```python
# CORRECT: only use past data
past = history[history["race_date"] < target_date]
stats = expanding_mean(past["harontimel3"])

# WRONG: uses future data
stats = history["harontimel3"].rolling(5).mean()  # may include future races
```

### Pattern 4: Surface-Parallel Training

**What:** turf and dirt models are trained independently in parallel via ThreadPoolExecutor.
**When:** Any model that varies by surface.
**Implementation:** TrainingPipelineV5._train_submodel() handles this. New features added within _train_submodel() (like time-series features) automatically get the surface parallelism. Features added in build_all() (before the split) are computed once for both surfaces.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Stacking Without Time-Series Awareness

**What:** Using random KFold for OOF generation in stacking.
**Why bad:** Horse racing data is temporal. Random folds leak future information (training on 2023 to predict 2022).
**Instead:** Use expanding window folds, exactly like the existing `AbilityModel.train_oof()` does with `race_date`-sorted boundaries. The current `StackedEnsemble.train()` uses approximate position-based folds; these must be replaced with date-aware expanding windows.

### Anti-Pattern 2: Mixing Ranker and Binary Stacking

**What:** Using binary classification stacking for the Stage1 Ranker model.
**Why bad:** The AbilityModel uses `lambdarank` objective which requires group information (horses per race). Binary classification ignores this race-level grouping, losing critical ranking signal.
**Instead:** Create a separate `StackedRankerEnsemble` that passes `group` information to each base model's training. XGBoost supports `rank:pairwise`, `rank:ndcg` objectives with group data. CatBoost supports `PairLogit` and `YetiRank` ranking objectives with group data.

### Anti-Pattern 3: Odds Leakage in Feature Computation

**What:** Using confirmed (post-race) odds in features that feed into pre-race prediction.
**Why bad:** The system already handles this (confirmed_odds vs tanodds separation in FeatureEngine.build_all()), but new feature modules must respect the convention.
**Instead:** Always use `tanodds` (pre-race snapshot) for features, `confirmed_odds` only for training labels (E-correction target).

### Anti-Pattern 4: Feature Computation After Model Training

**What:** Computing new features after models are already trained on the old feature set.
**Why bad:** The pipeline has a strict ordering -- features must exist before model training begins.
**Instead:** Add new feature computation to FeatureEngine.build_all() (for features available pre-surface-split) or to _train_submodel() (for features computed within a surface group). Never compute features after model training.

## Architecture for Each Feature Area

### Area 1: 3-Model Stacking Ensemble

**Current state:** `StackedEnsemble` class exists in `src/models/stacked_ensemble.py` with LightGBM + XGBoost + CatBoost binary classification and Ridge meta-learner. It is already wired into TrainingPipelineV5 via `use_ensemble=True`.

**Gaps to address:**

1. **Ranker stacking does not exist.** The current StackedEnsemble only supports binary classification (`objective="binary"`). The AbilityModel uses LightGBM Ranker (lambdarank) which requires group information. We need to either:
   - Option A: Keep AbilityModel as single LightGBM Ranker, only stack the binary models (hit_model in WinTwoStageModel). This is the simpler approach and targets the biggest ROI impact since binary hit prediction is where stacking helps most.
   - Option B: Build a StackedRankerEnsemble that handles group-aware training for all 3 frameworks. This is more complex but theoretically better.

   **Recommendation: Option A for v1.1.** The hit_model (binary) is where stacking has the most empirical benefit. Ranker stacking adds complexity with unclear payoff. The AbilityModel's p_ability_win feeds into WinTwoStageModel via init_score/logit anyway, so improving the binary model downstream captures most of the ensemble benefit.

2. **Time-series OOF generation.** Current StackedEnsemble uses positional folds (`val_start = int(n * (i+1) / (n_folds+1))`) which may not align with race_date boundaries. Must use race_date-aware expanding window splits like AbilityModel.train_oof() does.

3. **Hyperparameter tuning.** Current StackedEnsemble uses hardcoded parameters (learning_rate=0.03, num_leaves=31, max_depth=6, num_boost_round=300). These are reasonable defaults but each GBM framework benefits from slightly different tuning. At minimum, XGBoost should use `max_depth=4-6` and CatBoost should use `l2_leaf_reg=3`.

4. **Model persistence.** StackedEnsemble is saved via joblib (line 1182-1243 in training_pipeline.py). This works but the Ridge meta-learner coefficients should be logged to MLflow for traceability.

**Proposed component structure:**

```
StackedEnsemble (enhanced, src/models/stacked_ensemble.py)
  +-- train(): Use race_date-aware expanding window for OOF
  +-- predict(): Unchanged (3 predictions -> Ridge -> clip)
  +-- save/load(): Add to existing joblib pattern

No new files needed for this area -- extend existing StackedEnsemble.
```

### Area 2: Odds Deviation EV Features

**Current state:** The market_bias_features module computes `p_market_win_adj` (normalized market probability). The pipeline computes `odds_to_ability_ratio = p_market / p_ability` (line 422-423 in training_pipeline.py). The EVCorrectionModel uses `signed_log_error_win` and `abs_log_error_win` from MarketModel.

**What to add:** A dedicated feature module that computes richer odds-deviation metrics:

```python
# src/features/odds_deviation_features.py

def compute_odds_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features capturing model-vs-market probability divergence.

    These features quantify how much the model disagrees with the market,
    which is directly related to expected value.

    Features:
      - odds_deviation_signed: log(p_model/p_market), positive = model thinks horse is better
      - odds_deviation_abs: absolute deviation (uncertainty measure)
      - odds_deviation_squared: squared deviation (larger penalties for big disagreements)
      - odds_deviation_zscore: standardized deviation within race
      - model_confidence_gap: p_model - p_market (raw probability difference)
      - market_efficiency_score: entropy ratio (model vs market distribution)
      - deviation_adjusted_ev: p_model * odds * (1 - abs(deviation)) -- discounted EV
    """
```

**Integration point:** This module must be called AFTER AbilityModel.train_oof() (because it needs p_ability_win) but BEFORE WinTwoStageModel.train_hit_model() (because the deviation features feed into the hit model). This means it goes inside `_train_submodel()`, after the `odds_to_ability_ratio` computation (line 422) and before `WinTwoStageModel` training (line 435).

**Data flow:**

```
AbilityModel.train_oof() -> p_ability_win
compute_market_bias()     -> p_market_win_adj
                                    |
                            compute_odds_deviation_features()
                                    |
                            WinTwoStageModel.FEATURE_COLS += deviation features
```

### Area 3: Time-Series / Pace Features

**Current state:** HorseHistoryFeatures computes static aggregates of past performance (averages over last N runs). `harontime_late_trend` (last 2 vs first 3 runs) is the only temporal feature. The pace_aptitude_features module computes pace aptitude from historical data but does not predict the pace scenario for the upcoming race.

**What to add:**

1. **Time-series features** (temporal trends from past runs):
   - `time_progression_slope`: linear regression slope of finishing times over last 5 runs (positive = getting slower, negative = improving)
   - `closing_speed_trend`: trend in closing index over last 5 runs
   - `form_volatility`: standard deviation of normalized finish positions (consistency measure)
   - `recent_improvement_rate`: rate of change in the last 3 runs vs preceding 3 runs
   - `peak_form_indicator`: whether the horse is within 10% of its best recent performance

2. **Pace prediction features** (predicted race scenario from field composition):
   - `predicted_pace_scenario`: classify race as "fast pace" / "moderate" / "slow pace" based on the number of front-running horses in the field
   - `front_runner_count`: number of horses with kyakusitukubun_cd == "escape" (front-running style) in the race
   - `pace_pressure_index`: competition for the lead (how many horses prefer early position)
   - `position_fit_score`: how well the horse's running style fits the predicted pace

**Integration point:**

Time-series features go inside HorseHistoryFeatures.compute() -- they use the same past-performance data and follow the same PIT pattern. They add columns to the output DataFrame.

Pace prediction features are different: they require looking at ALL entries in the same race, not just one horse's history. This means they need to be computed at the race level, similar to `compute_intra_race_features()`. The best insertion point is within `_train_submodel()`, after HorseHistoryFeatures and before AbilityModel, since pace features are inputs to the ability model.

```
HorseHistoryFeatures.compute()  [includes time-series trends]
     |
PacePredictionFeatures.compute() [race-level pace analysis]
     |
AbilityModel.train_oof()         [sees pace features in FEATURE_COLS]
```

## Scalability Considerations

| Concern | Current (v5.5) | With Stacking | With New Features |
|---------|---------------|---------------|-------------------|
| Training time | ~44 min (2 surfaces) | ~2-3x increase (3 models per slot) | ~10-15% increase (more columns) |
| Memory (training) | ~4 GB | ~8-12 GB (3 models in memory) | ~5 GB (more feature columns) |
| Prediction latency | ~10 ms/race | ~30 ms/race (3 models + meta) | ~12 ms/race (more features) |
| Model storage | ~50 MB (.lgb files) | ~150 MB (3x models + joblib) | ~55 MB (feature metadata) |
| Backtest time | ~57 min/year | ~120-170 min/year | ~65 min/year |

The training time increase from stacking is the most significant concern. Mitigation: the pipeline already trains turf/dirt in parallel via ThreadPoolExecutor, and each surface's models can internally parallelize via num_threads. With 4+ CPU cores (typical), the wall-clock increase is closer to 1.5-2x rather than 3x.

## Build Order Recommendation

The build order follows strict data dependencies. Features must exist before models that consume them. Single models must work before stacking is layered on top.

```
Phase 1: Time-Series Features (no model changes)
  1a. Create src/features/time_series_features.py
  1b. Add time-series features to HorseHistoryFeatures.compute()
  1c. Add time-series columns to AbilityModel.FEATURE_COLS
  1d. Validate via backtest (features only, no stacking)

  Rationale: Features are the foundation. Adding them first means all
  subsequent model improvements benefit from richer inputs. No risk of
  regression since we only add columns (LightGBM ignores unused features).

Phase 2: Pace Prediction Features (no model changes)
  2a. Create src/features/pace_prediction_features.py
  2b. Wire into _train_submodel() after HorseHistoryFeatures
  2c. Add pace features to AbilityModel.FEATURE_COLS
  2d. Validate via backtest

  Rationale: Pace features are independent of time-series features
  but follow the same "features first" principle.

Phase 3: Odds Deviation Features (feature module + feature list update)
  3a. Create src/features/odds_deviation_features.py
  3b. Wire into _train_submodel() after AbilityModel.train_oof()
  3c. Add deviation features to WinTwoStageModel.FEATURE_COLS
  3d. Validate via backtest

  Rationale: Deviation features depend on p_ability_win existing,
  so they must come after Phase 1 (which improves p_ability_win).

Phase 4: Stacking Ensemble Enhancement (model change)
  4a. Enhance StackedEnsemble with time-aware OOF splits
  4b. Add hyperparameter profiles per GBM framework
  4c. Enable use_ensemble=True and validate
  4d. Add MLflow logging for meta-learner coefficients

  Rationale: Stacking is the last layer because it amplifies the
  quality of the underlying features. Better features first means
  the stacking meta-learner has a stronger signal to combine.

  NOTE: Stacking is applied only to the binary hit_model, not to
  the AbilityModel Ranker. This is a deliberate scope reduction.
  Ranker stacking can be evaluated in a future iteration.
```

## Integration Verification Checklist

After each phase, verify:

- [ ] New features have no NaN values for >95% of training data
- [ ] New features do not use post-race information (PIT compliance)
- [ ] Backtest with new features shows no regression vs baseline
- [ ] Feature importance (SHAP or gain) shows non-zero contribution from new features
- [ ] Training pipeline completes without errors
- [ ] All existing tests pass (`python -m pytest tests/ -v`)
- [ ] mypy type checking passes (`mypy src/`)

## Sources

- Existing codebase analysis (src/models/stacked_ensemble.py, src/pipelines/training_pipeline.py, src/features/*.py, src/models/stage1_ability_model.py, src/models/two_stage_return_model.py)
- XGBoost Learning to Rank documentation: https://xgboost.readthedocs.io/en/latest/tutorials/learning_to_rank.html
- CatBoost Ranking loss functions documentation: https://catboost.ai/docs/en/concepts/loss-functions-ranking
- Stacking ensemble best practices: Kaggle community discussion, ResearchGate publication on XGBoost+LightGBM+CatBoost stacking
- XGBoost v3.2.0 and CatBoost v1.2.10 verified as installed in the project environment
