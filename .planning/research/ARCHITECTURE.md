# Architecture: Ensemble Filter Recalibration (v1.4)

**Project:** keiba-ai v1.4 -- Ensemble Filter Recalibration
**Researched:** 2026-05-05
**Scope:** How ensemble-aware betting filters integrate with existing architecture. Data flow changes needed for filter recalibration across model changes. Filter parameter management across single-model to ensemble transition.
**Confidence:** HIGH (verified against full source code)

## Executive Summary

The v1.4 milestone requires recalibrating three existing filters (WinSelectionGate, EV_lower threshold, OddsBandFilter) to work with the 3-model stacked ensemble instead of the single LightGBM model they were calibrated against in Phase 11-12. The core problem is a probability distribution mismatch: the ensemble outputs probabilities with different statistical properties than single LightGBM, causing the fixed thresholds to over-exclude ensemble candidates (only 7 bets/year vs the needed 100+).

The architecture change is modest in scope but precise in placement. No new components are needed. The changes fall into three categories: (1) re-training `WinSelectionGateModel` with ensemble OOF predictions instead of single-model OOF, (2) making the EV_lower threshold dynamic by computing it from ensemble prediction statistics rather than hardcoding 1.0, and (3) re-calibrating `OddsBandFilter` with ensemble-derived `training_bet_history`. The Optuna 14-dimensional parameter search wires all of these together through the existing `StrategyOptimizer` -> `BacktestEngine` -> `RacePredictor` pipeline.

## Recommended Architecture

### Current Data Flow (Single Model, Phase 11-12)

```
TrainingPipelineV5
  |
  +-> stage1.add_ability_probs(df)     # LightGBM only: p_win_pred
  +-> win.predict_ev(df)               # LightGBM only: ev_win, EV_lower_win_corrected
  +-> ev_corrector.correct_ev(df)      # LightGBM only: ev_win_corrected
  +-> confidence.predict_interval(df)  # LightGBM only: EV_lower_win_corrected
  |
  v
RacePredictor.get_win_candidates(result_df)
  |  Filter 1: win_selection_edge > 0.0    (fixed)
  |  Filter 2: EV_lower_win_corrected >= 1.0  (hardcoded for single model)
  |  Sort: win_gate_score DESC              (calibrated on single-model OOF)
  |  Max: 2 candidates
  v
BacktestEngine.run()
  |  Filter 3: COLLAPSED regime skip
  |  Filter 4: OddsBandFilter.filter()     (calibrated from single-model bet history)
  v
select_bets() -> Bet objects
```

### Target Data Flow (Ensemble, v1.4)

```
TrainingPipelineV5 (--ensemble flag)
  |
  +-> stage1.add_ability_probs(df)        # StackedEnsemble: p_win_pred (Ridge blend of 3 GBM)
  +-> win.predict_ev(df)                  # Ensemble probabilities feed EV computation
  +-> ev_corrector.correct_ev(df)         # EV corrections use ensemble probabilities
  +-> confidence.predict_interval(df)     # Conformal intervals adapted for ensemble
  |
  +-> [RETRAIN] WinSelectionGateModel.train(df)  # Retrained on ensemble OOF distribution
  |     Uses: p_win_final (ensemble), ev_win_corrected (ensemble), tanoddslow
  |     Produces: adaptive min_prob/min_edge/max_odds thresholds
  |
  v
RacePredictor.get_win_candidates(result_df)
  |  Filter 1: win_selection_edge > 0.0    (unchanged)
  |  Filter 2: EV_lower_win_corrected >= dynamic_threshold  (was hardcoded 1.0)
  |    dynamic_threshold from gate model or Optuna parameter
  |  Sort: win_gate_score DESC              (retrained for ensemble distribution)
  |  Max: 2 candidates
  v
BacktestEngine.run()
  |  Filter 3: COLLAPSED regime skip       (unchanged)
  |  Filter 4: OddsBandFilter.filter()     (re-calibrated from ensemble bet history)
  v
select_bets() -> Bet objects

Optuna 14-dim search (or 15-dim with EV_lower threshold)
  |
  +-> StrategyOptimizer._run_single_backtest()
       |_ training backtest -> training_bet_history (ensemble-derived)
       |_ OddsBandFilter.calibrate(training_bet_history)  # auto-recalibrated
       |_ test backtest with all filters active
       |_ objective = mean ROI across WF folds
```

## Component Boundaries

| Component | File | Responsibility | Current State | v1.4 Change |
|-----------|------|----------------|---------------|-------------|
| `StackedEnsemble` | `models/stacked_ensemble.py` | 3-model GBM stacking (LGBM+XGB+CatBoost -> Ridge) | Fully implemented | No change -- produces probabilities via `predict()` |
| `WinTwoStageModel` | `models/two_stage_return_model.py` | P(hit) * E(odds\|hit) decomposition | Uses `hit_model` (single or ensemble) | No change -- delegates to `hit_model.predict()` |
| `WinSelectionGateModel` | `models/win_selection_gate.py` | Learned OOF gate + reranker for win bet selection | Trained on single-model OOF | **RETRAIN** with ensemble OOF predictions |
| `RacePredictor.get_win_candidates()` | `backtest/race_predictor.py` | Win candidate filtering (edge, EV, gate score) | Hardcoded EV_lower >= 1.0 | **MODIFY** threshold to be dynamic |
| `EVCorrectionModel` | `models/ev_correction_model.py` | P-correction and E-correction for win EV | Calibrated for single model | Auto-adapts (uses ensemble p_win) |
| `RobustConfidenceEstimator` | `models/robust_confidence_estimator.py` | Conformal prediction intervals | Produces EV_lower_win_corrected | Auto-adapts (wider intervals from ensemble) |
| `OddsBandFilter` | `betting/odds_band_filter.py` | Dynamic odds band ROI exclusion | Calibrated from single-model bet history | **RECALIBRATE** from ensemble bet history |
| `StrategyOptimizer` | `tuning/strategy_optimizer.py` | Optuna 14-dim parameter search | Fully implemented | **EXECUTE** (not yet run) |
| `BacktestEngine` | `backtest/engine.py` | Historical simulation with filter pipeline | Integrated in Phase 11 | No structural change |
| `RegimeDetector` | `models/regime_detector.py` | 3-state market regime classification | Independent of model type | No change |
| `DrawdownController` | `betting/drawdown_controller.py` | DD%-based stake sizing | Independent of model type | No change |
| `ModelLoader` | `db/model_loader.py` | Load models from disk/MLflow | Supports ensemble via `use_ensemble_override` | No change |

## Data Flow Changes

### CHANGE 1: WinSelectionGateModel Ensemble Retraining

The `WinSelectionGateModel.train()` is already model-agnostic -- it takes a DataFrame with `win_selection_prob`, `win_selection_edge`, `tanoddslow`, `kakuteijyuni` columns and never touches the underlying model directly. When the upstream model changes from single to ensemble, the same `train()` method is called with ensemble-derived values in those columns. The gate model automatically adapts its thresholds to the new distribution.

```
TrainingPipelineV5.run() --ensemble
  |
  v
ensemble OOF predictions (from StackedEnsemble.train() K-fold loop)
  |
  +-> df with p_win_final, ev_win_corrected, tanoddslow, kakuteijyuni
  |
  v
WinSelectionGateModel.train(df)  <-- RECALLED with ensemble-derived df
  |
  +-> Walk-forward threshold grid search over (min_prob, min_edge, max_odds)
  +-> Quantile binning of ensemble probability/edge/odds distributions
  +-> Score tables: combo_scores, pair_scores, single_scores
  +-> New thresholds: self.min_prob, self.min_edge, self.max_odds
  +-> Save to win_selection_gate_{surface}.joblib
  |
  v
ModelLoader.load_from_dir() loads retrained gate model
  |
  v
RacePredictor.predict() -> win_gate_model.score(df) -> uses new thresholds
```

**Critical insight:** The gate model stores quantile bin edges (`prob_edges`, `edge_edges`, `odds_edges`) computed from the training distribution. If trained on single-model data but used with ensemble predictions, these bins would misclassify candidates because ensemble probabilities have a different distribution. This is the root cause of the 7-bets/year problem.

### CHANGE 2: Dynamic EV_lower Threshold

Two viable approaches for making the EV_lower threshold adaptive:

**Option A (Recommended): Gate Model-Managed Threshold**
- During `WinSelectionGateModel.train()`, compute the ensemble EV_lower distribution from OOF predictions
- Store the threshold as a gate model parameter (e.g., 25th percentile of positive-edge EV_lower values)
- `get_win_candidates()` reads the threshold from the gate model instead of hardcoding 1.0

**Option B: Optuna-Managed Threshold**
- Add `ev_lower_threshold` as a 15th dimension in the Optuna search space (range 0.8 to 1.2)
- StrategyOptimizer finds the optimal threshold across WF folds
- Stored in `strategy_manifest.json` alongside the other 14 parameters

Both approaches are viable. Option A is simpler and self-contained within the gate model. Option B provides more search flexibility but adds another dimension to an already large search space.

```
# Option A: In WinSelectionGateModel.train()
ev_lower_positive = ev_lower[ev_lower.notna() & (ev_lower > 0)]
self.ev_lower_threshold = float(ev_lower_positive.quantile(0.25)) if len(ev_lower_positive) > 50 else 1.0

# In get_win_candidates():
ev_lower_threshold = win_gate_model.ev_lower_threshold if win_gate_model else 1.0
ev_mask = ev_lower.fillna(ev_lower_threshold) >= ev_lower_threshold
```

### CHANGE 3: OddsBandFilter Ensemble Recalibration

The `StrategyOptimizer._run_single_backtest()` (strategy_optimizer.py:118-192) already runs a training-phase backtest with ensemble models and passes the resulting `training_bet_history` to `OddsBandFilter.calibrate()`. This means the filter is automatically recalibrated for the ensemble when the optimizer runs.

No code changes needed -- the existing wiring is correct. The key is ensuring the optimizer actually executes (it has not been run yet).

```
StrategyOptimizer._run_single_backtest()
  |
  +-> ModelLoader.load_from_dir(use_ensemble_override=True)  # loads ensemble models
  +-> Training-phase backtest with ensemble
  |     produces: training_bet_history (ensemble-derived odds/edges/outcomes)
  |
  v
OddsBandFilter.calibrate(training_bet_history)
  |
  +-> Per-band ROI computed from ensemble bet outcomes
  +-> Bands with ROI < roi_threshold excluded
  |
  v
Test-phase backtest uses re-calibrated OddsBandFilter
```

### CHANGE 4: Optuna 14-dim Parameter Execution

The `StrategyOptimizer` is fully implemented but has never been executed. Running it is the final step of v1.4.

```
StrategyOptimizer.optimize()
  |
  +-> TPE sampler explores 14 dimensions:
  |     - fk_aggressive, fk_conservative (Kelly fractions)
  |     - ev_aggressive, ev_conservative (EV thresholds)
  |     - edge_aggressive, edge_conservative (edge thresholds)
  |     - dd_threshold_1, dd_threshold_2 (DD control)
  |     - multiplier_reduced, rolling_window, min_stay_races
  |     - target_ev, max_scale (EV scaling)
  |     - roi_threshold (OddsBandFilter)
  |
  +-> Each trial: build strategy_config -> WF backtest -> ROI
  +-> Best params -> save_strategy_manifest() -> JSON + SHA256
  |
  v
Verified manifest loaded for OOS validation
```

## Patterns to Follow

### Pattern 1: Model-Agnostic Filter Interface

**What:** Filters operate on DataFrame columns (probabilities, edges, odds) without knowing whether values came from a single model or ensemble.

**When:** All filter components consume columns, not models.

**Why:** The existing architecture already achieves this. `WinSelectionGateModel.train()` takes a DataFrame with `win_selection_prob`, `win_selection_edge`, `tanoddslow` -- it never touches the model directly. Similarly, `OddsBandFilter.calibrate()` takes bet history dicts. This means retraining is simply calling the same methods with ensemble-derived data.

**Example:**
```python
# WinSelectionGateModel.train() is already model-agnostic:
def train(self, df: pd.DataFrame) -> None:
    prepared = self._prepare_training_frame(df)
    # Uses: win_selection_prob, win_selection_edge, tanoddslow, kakuteijyuni
    # All columns come from whatever model produced them

# For ensemble: the DataFrame just has different values in those columns
# (wider probability distribution, different edge distribution)
# The gate model adapts its thresholds to the new distribution automatically
```

### Pattern 2: Training-Data-Driven Filter Calibration

**What:** Filters derive their parameters from training data, not hardcoded values. When the upstream model changes, filters re-calibrate from the new data distribution.

**When:** `OddsBandFilter.calibrate()` and `WinSelectionGateModel.train()`.

**Implementation:**
```
Model change (single -> ensemble)
  -> New probability distribution in OOF predictions
  -> WinSelectionGateModel.train() adapts thresholds
  -> BacktestEngine generates new training_bet_history
  -> OddsBandFilter.calibrate() adapts excluded bands
```

### Pattern 3: Optuna Parameter Freeze Protocol

**What:** After Optuna finds optimal parameters, freeze them with SHA256 manifest to prevent drift during OOS evaluation.

**When:** After `StrategyOptimizer.optimize()` completes successfully.

**Implementation:** Existing `save_strategy_manifest()` and `verify_strategy_manifest()` -- no changes needed.

### Pattern 4: Walk-Forward Validation for Filter Parameters

**What:** Filter thresholds are validated using walk-forward folds, not in-sample.

**When:** `WinSelectionGateModel.train()` already implements this internally (`_build_walk_forward_folds`). `StrategyOptimizer` implements this externally (2-fold WF across years).

**Why this matters for ensemble:** The ensemble has different overfitting characteristics than single LightGBM. Walk-forward validation ensures filter thresholds generalize.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Mixing Single-Model and Ensemble Gate Models

**What:** Loading a WinSelectionGateModel trained on single-model OOF while using ensemble predictions at inference time.

**Why bad:** The quantile bins (`prob_edges`, `edge_edges`, `odds_edges`) and score tables (`combo_scores`, `pair_scores`) are calibrated to the single-model probability distribution. Ensemble probabilities have different quantiles -- the bins would misclassify candidates, producing garbage scores. This is the root cause of the current 7-bets/year problem.

**Instead:** Always retrain `WinSelectionGateModel` when switching from single model to ensemble.

**Detection:** If `meta.json` has `use_ensemble=true` but `win_selection_gate_{surface}.joblib` was trained before ensemble was enabled, the gate model is stale.

### Anti-Pattern 2: Hardcoded EV_lower Threshold Across Model Changes

**What:** Keeping `EV_lower_win_corrected >= 1.0` fixed regardless of whether the underlying model is single or ensemble.

**Why bad:** The ensemble produces different `EV_lower` values because conformal intervals are computed from ensemble residuals (different variance) and EV corrections use ensemble probabilities. The ensemble's `EV_lower` distribution is shifted relative to single model, so the same threshold over-excludes (current problem: 3,594 excluded, only 7 bets/year).

**Instead:** Make the threshold dynamic -- either derived from ensemble distribution statistics or searched by Optuna.

### Anti-Pattern 3: Calibrating OddsBandFilter with Wrong Model's Bet History

**What:** Running a training-phase backtest with single-model, then passing that `training_bet_history` to an ensemble test-phase backtest.

**Why bad:** Band ROI statistics would reflect single-model performance, not ensemble performance. Different models have different edge distributions across odds bands.

**Instead:** `StrategyOptimizer._run_single_backtest()` already runs the training-phase backtest with the SAME models (ensemble) as the test-phase. The `training_bet_history` is always model-consistent.

### Anti-Pattern 4: Bypassing Filters During Optuna Search

**What:** Disabling filters during Optuna search to maximize bet count, then enabling them during OOS evaluation.

**Why bad:** Optuna would optimize parameters for a configuration that never runs in production.

**Instead:** Keep all filters active during Optuna search. The optimizer adjusts filter parameters themselves (roi_threshold, ev/edge thresholds) as part of the search space.

## Detailed Integration Points

### Integration Point 1: Training Pipeline -> WinSelectionGateModel

**File to modify:** `src/training/pipeline.py` or equivalent training script

The training pipeline must pass ensemble-derived DataFrame to `WinSelectionGateModel.train()`. The column flow is:

```
StackedEnsemble.predict(X) -> p_win_pred
WinTwoStageModel.predict_ev(df) -> ev_win, p_win_combined
EVCorrectionModel.correct_ev(df) -> ev_win_corrected
RobustConfidenceEstimator.predict_interval(df) -> EV_lower_win_corrected
ensure_win_selection_columns(df) -> win_selection_ev, win_selection_edge, win_selection_prob
```

The `win_selection_prob` column is what the gate model uses for quantile binning. It must reflect ensemble probabilities. This column comes from `p_win_final` (or `p_win_combined` or `p_win_corrected` -- whichever is available, see `ensure_win_selection_columns()` in win_selection_gate.py:33-54).

### Integration Point 2: StrategyOptimizer -> BacktestEngine

**File:** `src/tuning/strategy_optimizer.py` (already correct)

Key insight: Line 137 already loads ensemble models:
```python
models, info = loader.load_from_dir(self.models_dir, use_ensemble_override=True)
```

This means:
- Training bet history is ensemble-derived (correct for OddsBandFilter)
- EV_lower values are ensemble-derived (correct for dynamic threshold)
- Win gate scores come from retrained gate model (correct for candidate ranking)

The StrategyOptimizer is already wired correctly. The missing piece is retraining `WinSelectionGateModel` with ensemble data BEFORE running the optimizer.

### Integration Point 3: Optuna Search Space -> Filter Parameters

**Current 14-dimensional search space (strategy_optimizer.py:51-81):**

| Dimension | Parameter | Affected Component | Ensemble Impact |
|-----------|-----------|-------------------|-----------------|
| 1-2 | fk_aggressive, fk_conservative | StakeCalculator | Indirect (edge distribution changes) |
| 3-4 | ev_aggressive, ev_conservative | RegimeDetector params | Direct (EV threshold shifts) |
| 5-6 | edge_aggressive, edge_conservative | RegimeDetector params | Direct (edge distribution changes) |
| 7-8 | dd_threshold_1, dd_threshold_2 | DrawdownController | Indirect |
| 9 | multiplier_reduced | DrawdownController | Indirect |
| 10-11 | rolling_window, min_stay_races | DrawdownController | Indirect |
| 12-13 | target_ev, max_scale | StakeCalculator.apply_ev_scaling() | Direct (EV scaling) |
| 14 | roi_threshold | OddsBandFilter | Direct (band exclusion threshold) |

**Optional 15th dimension:** An explicit `ev_lower_threshold` parameter would allow Optuna to find the optimal EV_lower cutoff for the ensemble distribution. Currently hardcoded at 1.0 in `get_win_candidates()` (race_predictor.py:441).

### Integration Point 4: ModelLoader -> WinSelectionGate Loading

**File:** `src/db/model_loader.py` (no change needed)

Lines 588-595 already load whatever gate model file exists:
```python
wsg_file = models_dir / f"win_selection_gate_{surface}.joblib"
if wsg_file.is_file():
    win_selection_gate = WinSelectionGateModel.load(wsg_file)
```

If retraining produces a new `.joblib` file with ensemble-calibrated parameters, it will be loaded automatically. The gate model serialization is model-agnostic (stores thresholds, bin edges, score tables -- not the underlying model).

## Build Order (Dependency-Driven)

### Phase 1: WinSelectionGate Ensemble Retraining
**Dependencies:** Existing ensemble training pipeline
**Changes:** Modify training pipeline to pass ensemble-derived DataFrame to `WinSelectionGateModel.train()`
**Files:** Training script or pipeline module
**Verification:** Gate model thresholds change from single-model values to ensemble-adapted values

### Phase 2: Dynamic EV_lower Threshold
**Dependencies:** Phase 1 (gate model retraining)
**Changes:** Either (a) add `ev_lower_threshold` as a WinSelectionGateModel parameter computed during `train()`, or (b) add it as Optuna search dimension
**Files:** `src/backtest/race_predictor.py` (~10 lines in filter mask), optionally `src/tuning/strategy_optimizer.py`
**Verification:** Ensemble backtest produces 100+ bets/year (not 7)

### Phase 3: OddsBandFilter Ensemble Recalibration
**Dependencies:** Phase 1 (gate model retrained for correct candidate selection)
**Changes:** None to OddsBandFilter itself -- it already receives ensemble-derived `training_bet_history` from StrategyOptimizer
**Files:** None
**Verification:** Band exclusion reflects ensemble ROI, not single-model ROI

### Phase 4: Optuna 14-dim (or 15-dim) Execution
**Dependencies:** Phases 1-3 (all filters ensemble-aware)
**Changes:** Execute existing `StrategyOptimizer.optimize()`. Optionally add `ev_lower_threshold` as 15th dimension.
**Files:** `scripts/run_strategy_optimization.py` (execution only)
**Verification:** Best ROI across WF folds > 1.0 (target: ROI > 100%)

### Phase 5: Manifest Freeze and OOS Validation
**Dependencies:** Phase 4 (optimal parameters found)
**Changes:** Freeze parameters via `save_strategy_manifest()`, run final OOS backtest with frozen params
**Files:** None (existing protocol)
**Verification:** `verify_strategy_manifest()` passes, OOS ROI > 100%

## Modified vs New Components

### Modified Components

| File | Change | Scope |
|------|--------|-------|
| Training pipeline (pipeline.py or run_train.py) | Pass ensemble OOF df to `WinSelectionGateModel.train()` | ~20 lines wiring |
| `src/backtest/race_predictor.py` | Dynamic EV_lower threshold in `get_win_candidates()` | ~10 lines in filter mask |
| `src/tuning/strategy_optimizer.py` | Optionally add ev_lower_threshold as 15th dimension | ~5 lines |

### New Components

None. All required infrastructure exists from Phase 11-12 and v1.1 ensemble work.

### Unchanged Components

| File | Why Unchanged |
|------|--------------|
| `src/models/stacked_ensemble.py` | Already produces correct probabilities |
| `src/models/win_selection_gate.py` | `train()` is already model-agnostic |
| `src/betting/odds_band_filter.py` | Already takes bet_history as input |
| `src/backtest/engine.py` | Filter pipeline already integrated |
| `src/backtest/parameter_freeze_protocol.py` | Already handles parameter freezing |
| `src/models/regime_detector.py` | Model-type independent |
| `src/betting/drawdown_controller.py` | Model-type independent |
| `src/domain/models.py` | No new data structures needed |
| `src/db/model_loader.py` | Already loads ensemble models and gate models |
| `config/settings.yaml` | No new configuration needed |

## Scalability Considerations

| Concern | At current scale (~9K single-model bets/year) | At target (100+ ensemble bets/year) | At high volume (10K+ ensemble bets/year) |
|---------|-----------------------------------------------|--------------------------------------|------------------------------------------|
| WinSelectionGate retraining | ~30 seconds (100+ features, 200+ races) | Same -- retraining is one-time | May need subsampling for larger datasets |
| OddsBandFilter calibration | O(n) single pass over bet_history | Same -- negligible | Same |
| Optuna 100 trials x 2 folds | ~200 backtests (~57 min each = ~190 hours total) | Same -- this is the bottleneck | Consider parallel Optuna with n_jobs>1 |
| Strategy manifest verification | O(1) SHA256 hash | Same | Same |

## Sources

- Code analysis: `src/models/stacked_ensemble.py` -- ensemble predict/produce interface, K-fold OOF generation
- Code analysis: `src/models/win_selection_gate.py` -- OOF gate training, walk-forward threshold search, quantile binning
- Code analysis: `src/backtest/race_predictor.py` -- get_win_candidates() filter chain, EV_lower hardcoded threshold
- Code analysis: `src/backtest/engine.py` -- BacktestEngine.run() race loop with filter pipeline
- Code analysis: `src/betting/odds_band_filter.py` -- calibrate/filter interface
- Code analysis: `src/tuning/strategy_optimizer.py` -- Optuna parameter search, ensemble model loading
- Code analysis: `src/db/model_loader.py` -- ensemble model loading, gate model loading, _load_hit_model
- Code analysis: `src/backtest/parameter_freeze_protocol.py` -- manifest freeze/verify
- Code analysis: `src/models/regime_detector.py` -- regime params independent of model type
- Code analysis: `src/betting/drawdown_controller.py` -- DD control independent of model type
- Code analysis: `.planning/phases/11-bet-selection-filters/11-RESEARCH.md` -- Phase 11 architecture decisions
- Code analysis: `.planning/PROJECT.md` -- v1.4 milestone context (7 bets/year problem, 3,594 EV_lower exclusions)
