# Architecture: Investment Pipeline Restructuring (v2.0)

**Domain:** Horse racing prediction system -- probability calibration, feature engineering, race-level ranking
**Researched:** 2026-05-27
**Overall confidence:** HIGH (based on direct codebase analysis of all integration points)

## Executive Summary

The v2.0 milestone restructures the probability estimation and horse selection pipeline. Four new components (OOF Health, InvestmentFeatureFrame, MarketAwareWinCalibrator, Race-Level Ranker) insert into the existing training and inference paths at specific, well-defined points. The architecture follows the established pattern of extend-not-replace: new models are added as optional fields on SubmodelSet, with graceful fallback when absent. The critical architectural invariant is OOF safety: MarketAwareWinCalibrator and Race-Level Ranker both consume predictions from earlier models and must train on out-of-fold predictions using the existing walk-forward race-split pattern. The existing WinBenterGate is retained as fallback but superseded by MarketAwareWinCalibrator.

The recommended build order follows a strict dependency chain: Phase 0 (OOF Health) first because all downstream training depends on valid OOF artifacts; Phase 1 (InvestmentFeatureFrame) second because it provides the feature substrate for both the calibrator and ranker; Phase 2+3 (MarketAwareWinCalibrator with Segment Calibration) third because it produces p_win_market_aware which the ranker consumes; Phase 4 (Race-Level Ranker) last because it depends on both investment features and market-aware probabilities.

## Recommended Architecture

### Current Pipeline (v1.8)

```
TrainingPipelineV5.run()
  |
  +-- 1. Data load (ParquetStore -> races, entries, odds)
  +-- 2. FeatureEngine.build_all() -> feat_df (368-438 cols)
  +-- 3. SubModelManager.add_distance_band_features()
  +-- 4. Per-surface training (_train_submodel):
  |     +-- MarketModel.train() -> predict_and_calc_error()
  |     +-- AbilityModel.train_oof() -> p_ability_win (Stage1 OOF)
  |     +-- TargetEncoder.fit_transform_oof() [OOF-safe TE]
  |     +-- compute_stage2_relative_features()
  |     +-- compute_odds_deviation_features()
  |     +-- WinTwoStageModel (train_hit + train_return + predict_ev)
  |     +-- EVCorrectionModel.train() + correct_ev()
  |     +-- EV Isotonic Calibration + Odds Band Scaling (OOF)
  |     +-- WinSelectionGate OOF generation (walk-forward)
  |     +-- WinSelectionPolicy.train() + apply()
  |     +-- WinProfitSelector.train() + score()
  |     +-- WinSegmentCalibrator.train() (turf only)
  |     +-- Returns SubmodelSet + df_oof + wsg_train_df
  +-- 5. _build_race_level_features() -> race_feat_df
  +-- 6. RaceQualityScreener.train()
  +-- 7. RegimeDetector.train()
  +-- 8. MLflow logging
  +-- Returns TrainedModelsV5
```

```
RacePredictor.predict() (inference path)
  |
  +-- Submodel selection by surface
  +-- HorseHistoryFeatures merge
  +-- Race-rank columns computation
  +-- compute_interaction_features()
  +-- compute_relative_features()
  +-- MarketModel.predict_and_calc_error()
  +-- AbilityModel.add_ability_probs()
  +-- compute_stage2_relative_features()
  +-- compute_odds_deviation_features()
  +-- WinTwoStageModel.predict_ev()
  +-- EVCorrectionModel.correct_ev()
  +-- WinBenterGate.apply() [if win_benter exists]
  +-- WinSelectionGate.score() [if trained]
  +-- Returns df with EV/prob columns

RacePredictor.get_win_candidates()
  |
  +-- EV tail calibration
  +-- WinSegmentCalibrator.apply() [if trained]
  +-- Selection score computation (surface-aware base)
  +-- WinProfitSelector.score() [if trained]
  +-- Sort by profit_score -> market_score -> edge -> prob
  +-- Return top-1 (or max_per_race if profit selector enabled)
```

### Proposed Pipeline (v2.0)

```
TrainingPipelineV5.run() [MODIFIED]
  |
  +-- 1-3. (unchanged) Data load, FeatureEngine, SubModelManager
  +-- 4. Per-surface training (_train_submodel) [MODIFIED]:
  |     +-- (unchanged) MarketModel through WinTwoStageModel
  |     +-- (unchanged) EVCorrectionModel + Isotonic Calibration
  |     +-- [NEW] OOF Health validation
  |     +-- [NEW] InvestmentFeatureFrame.build()
  |     |     -- Derives 80-150 investment-specific columns from base features
  |     |     -- Produces: data/features/investment_features.parquet
  |     |     -- Produces: data/oof/win_investment_oof.parquet
  |     +-- [NEW] MarketAwareWinCalibrator.train(oof_df)
  |     |     -- Input: win_investment_oof (OOF predictions + investment features)
  |     |     -- Label: is_win (kakuteijyuni == 1)
  |     |     -- Model: LightGBM binary with logit(p_model)+logit(p_market)+segment features
  |     |     -- Output: p_win_market_aware
  |     |     -- Replaces win_benter as primary probability source
  |     +-- [MODIFIED] Segment Calibration features added to InvestmentFeatureFrame
  |     |     -- segment_actual_pred_ratio, segment_sample_count, etc.
  |     |     -- NOT a standalone model; features consumed by MarketAwareWinCalibrator
  |     +-- (unchanged) WinSelectionGate OOF generation
  |     +-- (unchanged) WinSelectionPolicy + WinProfitSelector
  |     +-- [NEW] Race-Level Ranker training
  |     |     -- Input: InvestmentFeatureFrame + p_win_market_aware
  |     |     -- Two sub-models: win rate ranker + value ranker
  |     |     -- Uses LightGBM ranker (LambdaRank)
  |     |     -- Output: investment_score per horse within race
  |     +-- Returns SubmodelSet [EXTENDED] + investment_oof + ranker_oof
  +-- 5-8. (unchanged) Race features, QualityScreener, RegimeDetector, MLflow
  +-- Returns TrainedModelsV5 [EXTENDED]
```

```
RacePredictor.predict() [MODIFIED]
  |
  +-- (unchanged) Steps 1-8: through WinTwoStageModel.predict_ev()
  +-- (unchanged) EVCorrectionModel.correct_ev()
  +-- [NEW] InvestmentFeatureFrame.build() (inference-time)
  +-- [NEW] MarketAwareWinCalibrator.apply()
  |     -- Produces p_win_market_aware, replaces p_win_final for downstream
  +-- [FALLBACK] WinBenterGate.apply() -- only if MarketAwareWinCalibrator absent
  +-- (unchanged) WinSelectionGate.score()

RacePredictor.get_win_candidates() [MODIFIED]
  |
  +-- (unchanged) EV tail calibration
  +-- (unchanged) WinSegmentCalibrator.apply()
  +-- [NEW] Race-Level Ranker.score()
  |     -- Replaces manual selection_score computation
  |     -- investment_score = value_ranker + win_rate_ranker - uncertainty_penalty
  +-- (unchanged) WinProfitSelector.score()
  +-- Sort by investment_score -> profit_score -> prob -> edge
  +-- Return top candidate(s)
```

## Component Boundaries

### NEW Components

| Component | File | Responsibility | Communicates With |
|-----------|------|----------------|-------------------|
| OOF Health Checker | `src/validation/oof_health.py` | Validates OOF artifacts: empty check, race_id dedup, anomaly detection, fold integrity | TrainingPipelineV5 (called after OOF generation) |
| InvestmentFeatureFrame | `src/features/investment_features.py` | Builds 80-150 investment-judgment features from base features + model outputs | FeatureEngine (reads base features), MarketAwareWinCalibrator (provides input), RacePredictor (inference) |
| MarketAwareWinCalibrator | `src/models/market_aware_win_calibrator.py` | Benter-type logit blend + segment calibration features -> p_win_market_aware | TrainingPipelineV5 (train), SubmodelSet (stored), RacePredictor (inference) |
| Win Race-Level Ranker | `src/models/win_race_level_ranker.py` | LightGBM ranker for race-internal horse ranking (value + win rate) | InvestmentFeatureFrame (features), RacePredictor (inference) |
| Calibration Report | `src/validation/calibration_report.py` | Generates probability quality reports (Brier, ECE, actual/pred) | MarketAwareWinCalibrator (validation) |

### MODIFIED Components

| Component | File | Change Type | What Changes |
|-----------|------|-------------|-------------|
| TrainingPipelineV5 | `src/pipelines/training_pipeline.py` | Extension | Add InvestmentFeatureFrame generation, MarketAwareWinCalibrator training, Ranker training. Add OOF health checks. Extend `_prepare_win_selection_oof_artifact()` to include investment features |
| SubmodelSet | `src/domain/models.py` | Extension | Add fields: `market_aware_calibrator`, `win_race_level_ranker` |
| TrainedModelsV5 | `src/domain/models.py` | No change | Already contains submodels dict, quality_screener, regime_detector |
| RacePredictor | `src/backtest/race_predictor.py` | Extension | Add InvestmentFeatureFrame.build() call, MarketAwareWinCalibrator.apply(), Ranker.score(). Modify get_win_candidates() to use ranker output |
| ModelLoader | `src/db/model_loader.py` | Extension | Load MarketAwareWinCalibrator and WinRaceLevelRanker artifacts from MLflow/local |

### UNCHANGED Components

| Component | File | Reason |
|-----------|------|--------|
| FeatureEngine | `src/features/feature_engine.py` | Base feature generation unchanged; InvestmentFeatureFrame is a separate consumer |
| ParquetStore | `src/db/parquet_store.py` | Data I/O layer unchanged |
| DataRepository | `src/db/repository.py` | Data access unchanged |
| MarketModel | `src/models/market_model.py` | Produces market error features; consumed by InvestmentFeatureFrame |
| AbilityModel | `src/models/stage1_ability_model.py` | Stage1 output p_ability_win is input to InvestmentFeatureFrame |
| WinTwoStageModel | `src/models/two_stage_return_model.py` | Core model unchanged; its output feeds InvestmentFeatureFrame |
| EVCorrectionModel | `src/models/ev_correction_model.py` | EV correction unchanged; feeds InvestmentFeatureFrame |
| BacktestEngine | `src/backtest/engine.py` | Uses RacePredictor; benefits from changes without modification |
| BettingOrchestrator | `src/betting/orchestrator.py` | Uses RacePredictor output; no changes needed |
| StakeCalculator | `src/betting/stake_calculator.py` | Stake computation unchanged |
| DrawdownController | `src/betting/drawdown_controller.py` | DD control unchanged |
| RegimeDetector | `src/models/regime_detector.py` | Regime detection unchanged (v2.0 does not depend on regime) |

## Data Flow

### Training Data Flow

```
                                    +-----------------------+
                                    |  FeatureEngine output  |
                                    |  (368-438 cols)        |
                                    +-----------+-----------+
                                                |
                                    +-----------v-----------+
                                    |  WinTwoStageModel      |
                                    |  + EVCorrectionModel   |
                                    |  -> p_win_pred,        |
                                    |     p_win_corrected,   |
                                    |     ev_win_corrected   |
                                    +-----------+-----------+
                                                |
                        +-----------------------v------------------------+
                        |  [NEW] OOF Health Check                       |
                        |  validate_oof_health(df_oof)                   |
                        |  - Empty guard (0-row prohibition)             |
                        |  - race_id dedup across folds                  |
                        |  - Top1 hit rate / ROI anomaly detection       |
                        |  RAISES on failure (prevents downstream use)   |
                        +-----------------------+------------------------+
                                                |
                        +-----------------------v------------------------+
                        |  [NEW] InvestmentFeatureFrame.build()          |
                        |  Input: base features + model outputs +        |
                        |         market probs + race-level features      |
                        |  Output: 80-150 investment-specific features    |
                        |  Artifact: data/oof/win_investment_oof.parquet |
                        +-----------------------+------------------------+
                                                |
                        +-----------------------v------------------------+
                        |  [NEW] MarketAwareWinCalibrator.train()        |
                        |  Input: win_investment_oof (OOF predictions)   |
                        |  Label: is_win                                |
                        |  Features: logit(p_model), logit(p_market),   |
                        |            segment features, investment feats  |
                        |  Model: LightGBM binary (initial)              |
                        |  Output: p_win_market_aware                   |
                        |  Validation: Brier, ECE, actual/pred          |
                        |  Must pass OOF deployment gate before saving   |
                        +-----------------------+------------------------+
                                                |
                        +-----------------------v------------------------+
                        |  [NEW] Win Race-Level Ranker.train()           |
                        |  Input: InvestmentFeatureFrame +               |
                        |         p_win_market_aware                     |
                        |  Sub-model 1: win_rate_ranker (target: is_win) |
                        |  Sub-model 2: value_ranker (target: ev, CLV)   |
                        |  Model: LightGBM ranker (LambdaRank)           |
                        |  Output: investment_score per horse per race   |
                        |  Must not reduce bet count vs baseline         |
                        +-----------------------------------------------+
```

### Inference Data Flow (within RacePredictor)

```
race_df (single race)
     |
     v
[Existing inference chain: Market -> Stage1 -> WinTwoStage -> EVCorrection]
     |
     v
df with p_win_pred, p_win_corrected, ev_win_corrected
     |
     v
[NEW] InvestmentFeatureFrame.build(df)
     |
     v
df with 80-150 investment features
     |
     v
[NEW] MarketAwareWinCalibrator.apply(df)
     |
     v
df with p_win_market_aware, p_win_market_aware_raw,
    market_aware_segment, market_aware_uncertainty
     |
     v
[FALLBACK if no calibrator: WinBenterGate.apply()]
     |
     v
ConformalEVModel.predict_interval()  -- (existing)
     |
     v
WinSelectionGate.score()  -- (existing)
     |
     v
[NEW] WinRaceLevelRanker.score(df)
     |
     v
df with investment_score, value_ranker_score,
    win_rate_ranker_score
     |
     v
get_win_candidates() uses investment_score as primary sort key
     |
     v
WinProfitSelector.score()  -- (existing, secondary filter)
     |
     v
final bet candidates
```

## Patterns to Follow

### Pattern 1: OOF-Safe Training (existing, extend to new models)

**What:** All models that consume predictions from earlier models must train on OOF (out-of-fold) predictions, never in-sample predictions. Walk-forward race splits prevent race-level leakage.

**When:** MarketAwareWinCalibrator and WinRaceLevelRanker both consume p_win_pred and must use OOF versions.

**Implementation:**
```python
# In _train_submodel, after WinTwoStageModel + EVCorrection:
# 1. Generate investment features on the full OOF df_oof
inv_features = InvestmentFeatureFrame.build(df_oof)

# 2. Generate OOF predictions for MarketAwareWinCalibrator
#    using walk-forward race splits (same pattern as generate_win_selection_oof_frame)
calibrator_oof = MarketAwareWinCalibrator.train_oof(
    inv_features,
    n_splits=5,
    target_col="is_win",
)

# 3. Train final calibrator on full OOF data
calibrator = MarketAwareWinCalibrator()
calibrator.train(calibrator_oof)
```

**Existing reference:** `_walk_forward_race_splits()` in training_pipeline.py (line 197), `generate_win_selection_oof_frame()` (line 1646).

### Pattern 2: SubmodelSet Extension (existing pattern)

**What:** New per-surface models are stored as optional fields on SubmodelSet. ModelLoader discovers them from artifacts. Backward compatible: None default means old models still load.

**When:** Adding MarketAwareWinCalibrator and WinRaceLevelRanker.

**Implementation:**
```python
# In domain/models.py SubmodelSet:
market_aware_calibrator: MarketAwareWinCalibrator | None = None
win_race_level_ranker: WinRaceLevelRanker | None = None

# In model_loader.py load_from_dir(), for each surface:
market_aware_calibrator = None
mac_file = models_dir / f"market_aware_calibrator_{surface}.joblib"
if mac_file.is_file():
    market_aware_calibrator = MarketAwareWinCalibrator.load(mac_file)

win_ranker = None
ranker_file = models_dir / f"win_race_level_ranker_{surface}.joblib"
if ranker_file.is_file():
    win_ranker = WinRaceLevelRanker.load(ranker_file)
```

**Existing reference:** SubmodelSet already has 15+ optional fields (conformal_ev_model, place_selection_gate, win_benter, etc.) all following this pattern.

### Pattern 3: Feature Module Independence (existing pattern)

**What:** Each feature module is a standalone function that takes a DataFrame and returns new columns. No cross-module dependencies except through the DataFrame itself.

**When:** InvestmentFeatureFrame follows this pattern.

**Implementation:**
```python
# src/features/investment_features.py
def build_win_investment_features(
    feature_df: pd.DataFrame,
    prediction_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Derive investment-judgment features from base features + model outputs.

    Input: feature_df (from FeatureEngine + model prediction columns)
    Output: DataFrame with 80-150 investment-specific columns.
    """
    ...
```

**Existing reference:** `compute_relative_features()`, `compute_interaction_features()`, `compute_odds_deviation_features()` all follow this pattern.

### Pattern 4: Deployment Gate via Probability Quality (new pattern for v2.0)

**What:** Models are deployed based on calibration metrics (Brier, ECE, actual/pred ratio), never ROI alone. Multiple years must show stability. Single-year improvement is insufficient.

**When:** MarketAwareWinCalibrator and WinRaceLevelRanker deployment decisions.

**Implementation:**
```python
# In calibration_report.py
def assess_deployment(calibrator, oof_df):
    metrics = {
        "brier": compute_brier(oof_df["is_win"], oof_df["p_win_market_aware"]),
        "logloss": compute_logloss(oof_df["is_win"], oof_df["p_win_market_aware"]),
        "ece": compute_ece(oof_df["is_win"], oof_df["p_win_market_aware"]),
        "actual_pred_ratio_by_odds_band": ...,
        "actual_pred_ratio_by_surface": ...,
    }
    deployable = (
        metrics["brier"] <= baseline_brier
        and metrics["logloss"] <= baseline_logloss
        and metrics["ece"] <= baseline_ece
        and no_year_degrades(metrics, baseline_metrics)
    )
    return {"deployable": deployable, "metrics": metrics}
```

**Absolute constraint from PROJECT.md:** "2024/2025にだけ合う係数調整はしない。OOF、ウォークフォワード、年別安定性で配備可否を判定する。"

## Anti-Patterns to Avoid

### Anti-Pattern 1: Replacing WinBenterGate Before MarketAwareWinCalibrator is Validated

**What:** Removing the existing WinBenterGate before the new calibrator passes OOF validation.
**Why bad:** Regression with no fallback. WinBenterGate currently provides the only market-blended probability for win betting.
**Instead:** Keep WinBenterGate as fallback. In RacePredictor, use p_win_market_aware if MarketAwareWinCalibrator is present and passes health check, else fall back to WinBenterGate output (p_win_final from the existing Benter combination).
**Detection:** If p_win_market_aware column is absent or all-NaN in inference, fallback triggers automatically.

### Anti-Pattern 2: InvestmentFeatureFrame Depending on Post-Race Columns

**What:** Including kakuteijyuni, confirmed_odds, or other post-race columns in InvestmentFeatureFrame feature computation.
**Why bad:** These are not available at inference time (except in backtest), causing silent NaN differences between training and production paths.
**Instead:** Explicitly exclude POST_RACE_COLS (already defined in domain/types.py). InvestmentFeatureFrame should only use columns that exist in RacePredictor's inference path. Market probability features (p_market_win_adj, tanodds as market proxy) are available at inference and are safe. kakuteijyuni is the label, not a feature.

### Anti-Pattern 3: Training MarketAwareWinCalibrator on In-Sample Predictions

**What:** Using p_win_pred (full-training predictions) instead of p_win_oof (out-of-fold predictions) as input features to the calibrator.
**Why bad:** p_win_pred is trained on the same data it evaluates, inflating calibration quality. The calibrator would learn to trust overconfident predictions.
**Instead:** Use the same walk-forward OOF pattern as generate_win_selection_oof_frame() (line 1646). Generate calibrator OOF predictions where each fold is predicted by a calibrator trained on earlier data only. The existing _walk_forward_race_splits() utility can be reused.

### Anti-Pattern 4: Ranker Optimizing ROI Directly

**What:** Training the Race-Level Ranker with ROI or profit as the direct optimization target.
**Why bad:** Single-win ROI is extremely noisy (10% hit rate means 90% of samples have zero return). The model would overfit to the few high-odds winners in the training set.
**Instead:** Use two separate objectives: (1) win_rate_ranker with is_win target (binary classification ranking), (2) value_ranker with calibrated_ev target (regression ranking). Combine scores with fixed weights, not end-to-end ROI optimization. The investment_score formula from ROI_IMPROVEMENT_PLAN.md (calibrated_log_ev + value_ranker_score + clv_score - uncertainty_penalty) uses multiple orthogonal signals.

### Anti-Pattern 5: Reducing Bet Count to Inflate ROI

**What:** Adding aggressive filters that cut bet count significantly while reporting improved ROI.
**Why bad:** Violates the absolute constraint "ベット数を過剰に減らしてROIを上げる方針は採用しない." A cherry-picking filter can trivially improve ROI by only betting on the most confident signals, but this is not a robust strategy.
**Instead:** Deployment criteria must include bet count stability. Race-Level Ranker must demonstrate equal or greater race coverage compared to the existing selection_score approach.

## Integration Points Detail

### 1. OOF Health in TrainingPipelineV5

**Where:** After `_train_submodel()` returns df_oof for each surface, before saving OOF artifacts.

**Current code:** `_validate_win_selection_oof_health()` already exists (training_pipeline.py line 237) and checks top1 hit rate / ROI. Extend this pattern.

**Integration:**
```python
# New file: src/validation/oof_health.py
class OOFHealthChecker:
    def validate(self, df: pd.DataFrame, *, context: str) -> dict:
        """Run all OOF health checks. Returns report dict.
        Raises ValueError if critical checks fail."""
        checks = {
            "empty_guard": len(df) > 0,
            "is_oof_flag": "is_oof" in df.columns,
            "race_id_dedup": self._check_race_id_dedup(df),
            "top1_hit_rate": self._check_top1_hit_rate(df, max_rate=0.35),
            "top1_roi": self._check_top1_roi(df, max_roi=2.0),
            "min_rows": len(df) >= self._expected_min_rows(df),
        }
        ...

# In training_pipeline.py, after line 1549 (after _validate_win_selection_oof_health):
from validation.oof_health import OOFHealthChecker

health_checker = OOFHealthChecker()
full_oof_health = health_checker.validate(full_features_df, context="full_oof_save")
logger.info("Full OOF health: %s", full_oof_health)
```

**Files modified:** `src/pipelines/training_pipeline.py` (add import + call after OOF save)
**Files created:** `src/validation/oof_health.py`, `tests/test_oof_health.py`
**Artifacts produced:** `data/oof/oof_health_report.json`, `data/oof/win_selection_oof_health_report.json`

### 2. InvestmentFeatureFrame in TrainingPipelineV5

**Where:** Inside `_train_submodel()`, after EVCorrectionModel.correct_ev() (line ~1138), before WinSelectionGate OOF generation (line ~1500).

**Rationale:** InvestmentFeatureFrame needs model outputs (p_win_pred, p_win_corrected, ev_win_corrected, p_market_win_adj) which are available after EVCorrection. It must be computed before MarketAwareWinCalibrator training.

**Integration in training:**
```python
# In _train_submodel, after ev_corrector.correct_ev() and EV Isotonic:
from features.investment_features import build_win_investment_features

with TimingContext(f"{surface}/investment_features"):
    inv_df = build_win_investment_features(df_oof)
    # Merge investment features into df_oof for downstream consumers
    for col in inv_df.columns:
        if col not in df_oof.columns:
            df_oof[col] = inv_df[col].values

# Save investment feature artifact
inv_oof_path = Path("data/oof/win_investment_oof.parquet")
...
```

**Integration in inference (RacePredictor.predict):**
```python
# After EVCorrectionModel.correct_ev(), before MarketAwareWinCalibrator:
from features.investment_features import build_win_investment_features

inv_df = build_win_investment_features(df)
for col in inv_df.columns:
    if col not in df.columns:
        df[col] = inv_df[col].values
```

**Files modified:** `src/pipelines/training_pipeline.py`, `src/backtest/race_predictor.py`
**Files created:** `src/features/investment_features.py`, `tests/test_investment_features.py`
**Artifacts produced:** `data/features/investment_features.parquet`, `data/oof/win_investment_oof.parquet`

### 3. MarketAwareWinCalibrator in TrainingPipelineV5

**Where:** Inside `_train_submodel()`, after InvestmentFeatureFrame, before WinSelectionGate OOF generation.

**Relationship to existing WinBenterGate:** MarketAwareWinCalibrator replaces and generalizes WinBenterGate. Where WinBenterGate does `logit(p_c) = alpha*logit(p_fund) + beta*logit(p_market) + gamma` (3 scalar parameters), MarketAwareWinCalibrator does `logit(p_aware) = f(logit(p_model), logit(p_market), segment_features, investment_features)` with a learned LightGBM model.

**Integration in training:**
```python
# In _train_submodel, after investment features:
from models.market_aware_win_calibrator import MarketAwareWinCalibrator

with TimingContext(f"{surface}/market_aware_calibrator_oof"):
    # Generate OOF predictions for the calibrator itself
    # (same walk-forward pattern as generate_win_selection_oof_frame)
    calibrator_oof_df = generate_market_aware_calibrator_oof(
        df_oof,  # has investment features + model outputs
        n_splits=5,
        num_threads=num_threads,
    )

with TimingContext(f"{surface}/market_aware_calibrator_train"):
    market_aware_cal = MarketAwareWinCalibrator()
    market_aware_cal.train(calibrator_oof_df, target_col="is_win")

    # Apply to full df_oof for downstream consumers
    df_oof = market_aware_cal.apply(df_oof)
    # df_oof now has p_win_market_aware, p_win_market_aware_raw,
    # market_aware_segment, market_aware_uncertainty
```

**In SubmodelSet (domain/models.py):**
```python
# Add to SubmodelSet dataclass:
market_aware_calibrator: MarketAwareWinCalibrator | None = None
```

**In ModelLoader:** Load from `market_aware_calibrator_{surface}.joblib`

**In RacePredictor.predict():**
```python
# After InvestmentFeatureFrame.build():
calibrator = getattr(submodel, 'market_aware_calibrator', None)
if calibrator is not None:
    df = calibrator.apply(df)
    # p_win_market_aware now available
else:
    # Fallback to existing WinBenterGate
    if getattr(submodel, 'win_benter', None) is not None:
        from models.win_benter_gate import WinBenterGate
        win_gate = WinBenterGate(
            benter=submodel.win_benter,
            calibrator=getattr(submodel, 'win_isotonic_calibrator', None),
            temp_scaler=getattr(submodel, 'win_temperature_scaler', None),
        )
        df = win_gate.apply(df)
```

**Files modified:** `src/pipelines/training_pipeline.py`, `src/domain/models.py`, `src/backtest/race_predictor.py`, `src/db/model_loader.py`
**Files created:** `src/models/market_aware_win_calibrator.py`, `tests/test_market_aware_win_calibrator.py`
**Artifacts produced:** `market_aware_calibrator_{surface}.joblib` per surface, `data/validation/market_aware_calibration_report.json`

### 4. Segment Calibration as Features (not standalone model)

**Where:** Inside InvestmentFeatureFrame.build().

**Current WSC:** WinSegmentCalibrator is a standalone model (turf only) that applies probability shrinkage per segment (surface x odds_band x prob_rank_band). Per ROI_IMPROVEMENT_PLAN Option B, segment features are integrated into MarketAwareWinCalibrator's input rather than keeping WSC as a standalone model.

**Integration:**
```python
# In investment_features.py, within build_win_investment_features():
# Compute segment features (feature engineering, not a model)
segment_features = compute_segment_calibration_features(df)
# Returns: segment_actual_pred_ratio, segment_sample_count,
#          segment_win_count, segment_roi, segment_shrinkage_factor,
#          segment_reliability_weight

# These become columns in the InvestmentFeatureFrame
# MarketAwareWinCalibrator sees them as input features
# The calibrator learns how much to trust each segment
```

**Note:** WinSegmentCalibrator remains in the codebase (backward compatible) but is superseded. Its apply() step in RacePredictor.get_win_candidates() becomes optional/secondary.

**Files modified:** `src/features/investment_features.py` (includes segment feature computation)
**Files unchanged:** `src/models/win_segment_calibrator.py` (kept for backward compat)

### 5. Race-Level Ranker in RacePredictor

**Where:** In RacePredictor.get_win_candidates(), after MarketAwareWinCalibrator.apply(), replacing the manual selection_score computation.

**Current flow (get_win_candidates, line 586):**
1. Compute tail EV calibration
2. Compute selection_score from surface-aware base + late_odds_drop + log_odds_penalty + prob_rank_bonus + ev_tail_pressure + risk_penalty (lines 862-877)
3. WinProfitSelector.score()
4. Sort by composite score

**New flow:**
1. Compute tail EV calibration
2. If WinRaceLevelRanker is available:
   - Compute investment_score = ranker.score(df)
   - Use investment_score as primary sort key
3. Else: use existing selection_score computation (backward compat)
4. WinProfitSelector.score() (unchanged)
5. Sort by investment_score -> profit_score -> existing tiebreakers

**Integration:**
```python
# In get_win_candidates():
ranker = self._get_win_race_level_ranker(prepared)
if ranker is not None:
    prepared = ranker.score(prepared)
    # prepared now has: investment_score, value_ranker_score, win_rate_ranker_score
    # Use investment_score as primary sort key
    sort_cols = ["investment_score", PROFIT_SCORE_COL, ...]
else:
    # Existing selection_score computation (lines 862-877)
    prepared["win_market_selection_score"] = (
        selection_score
        - late_odds_drop_weight * late_odds_drop_z
        - log_odds_penalty * log_odds
        + prob_rank_bonus * model_prob_rank
        - ev_tail_penalty_weight * ev_tail_risk
        - market_risk_penalty_weight * risk_penalty
    )
    sort_cols = [PROFIT_SCORE_COL, "_win_market_selection_score_num", ...]
```

**Files modified:** `src/backtest/race_predictor.py` (get_win_candidates method)
**Files created:** `src/models/win_race_level_ranker.py`, `tests/test_win_race_level_ranker.py`
**Artifacts produced:** `win_race_level_ranker_{surface}.joblib` per surface, `data/validation/win_ranker_report.json`

### 6. Model Serialization (MLflow + Local)

**Where:** `_log_to_mlflow()` in TrainingPipelineV5 and `load_from_dir()` / `load()` in ModelLoader.

**New artifacts to save/load per surface:**
- `market_aware_calibrator_{surface}.joblib` -- MarketAwareWinCalibrator model (joblib for LightGBM + metadata)
- `win_race_level_ranker_{surface}.joblib` -- WinRaceLevelRanker model (joblib for LightGBM ranker)
- `investment_feature_cols.json` -- Column manifest for InvestmentFeatureFrame (deterministic feature list)

**Files modified:** `src/pipelines/training_pipeline.py` (MLflow logging), `src/db/model_loader.py` (loading)

## Build Order and Dependencies

### Dependency Graph

```
Phase 0: OOF Health
  |
  v
Phase 1: InvestmentFeatureFrame
  |
  +---> Phase 2: MarketAwareWinCalibrator
  |       |
  |       +---> Phase 3: Segment Calibration (features into calibrator)
  |               |
  |               +---> Phase 4: Race-Level Ranker
  |
  +---> Phase 9: Validation Design (can run in parallel after Phase 2)
```

### Phase Ordering Rationale

**Phase 0 (OOF Health) first** because:
- All subsequent phases depend on OOF quality
- Existing `_validate_win_selection_oof_health()` is limited (only checks top1 metrics)
- New checks (empty OOF guard, race_id dedup, fold count) protect against silent corruption
- Without this, a corrupted OOF could silently train a bad calibrator or ranker
- Estimated effort: Small (new module, ~150 lines + tests)

**Phase 1 (InvestmentFeatureFrame) second** because:
- MarketAwareWinCalibrator needs investment features as input
- Race-Level Ranker needs investment features as input
- Feature audit (column counts, missing rates, nunique) provides visibility before building models
- Does not modify existing pipeline -- pure addition
- Estimated effort: Medium (feature engineering, ~400 lines + tests)

**Phase 2 (MarketAwareWinCalibrator) third** because:
- Depends on InvestmentFeatureFrame from Phase 1
- Replaces WinBenterGate as primary probability source
- Must be validated (Brier/ECE/actual-pred) before Phase 4 can use p_win_market_aware
- Deployment condition is probability quality, not ROI
- Estimated effort: Medium-Large (new model, OOF generation, calibration report, ~500 lines + tests)

**Phase 3 (Segment Calibration integration) merges into Phase 2** because:
- Segment features are computed inside InvestmentFeatureFrame (Phase 1)
- MarketAwareWinCalibrator consumes them as features
- No standalone model -- just feature engineering columns
- Can be implemented as additional feature columns in InvestmentFeatureFrame

**Phase 4 (Race-Level Ranker) last** because:
- Depends on both InvestmentFeatureFrame and p_win_market_aware from Phase 2
- Replaces manual selection_score computation in get_win_candidates()
- LightGBM ranker needs group-aware training (race_id groups)
- Must validate that ranker does not reduce bet count
- Estimated effort: Large (ranker model, OOF validation, integration testing, ~600 lines + tests)

### Final Recommended Build Order

1. **Phase 0**: OOF Health (no dependencies, protects all downstream work)
2. **Phase 1**: InvestmentFeatureFrame (depends on Phase 0 for OOF quality)
3. **Phase 2+3**: MarketAwareWinCalibrator with Segment Calibration features (depends on Phase 1)
4. **Phase 4**: Race-Level Ranker (depends on Phases 1+2)

## Scalability Considerations

| Concern | Current (v1.8) | After v2.0 | Mitigation |
|---------|----------------|------------|------------|
| Training time | ~17 min | ~25-30 min (calibrator + ranker training) | Both are LightGBM models with small feature sets (under 200 cols). OOF generation adds ~5 min per model. |
| Inference latency (per race) | ~50ms | ~80ms (feature frame + calibrator + ranker) | All operations are vectorized per-race. No network calls. |
| Model artifact size | ~50MB | ~70MB (two new models per surface) | joblib compression. Negligible disk impact. |
| Memory (training) | ~4GB | ~6GB (investment features + calibrator OOF) | Investment features are computed in-place. Calibrator OOF reuses existing walk-forward infrastructure. |
| Feature columns | 368-438 | 368-438 (base) + 80-150 (investment, separate) | Investment features are a separate DataFrame, not merged into base. Consumed only by calibrator/ranker. |

## Sources

- Direct codebase analysis: `src/pipelines/training_pipeline.py` (2567 lines, training orchestration)
- Direct codebase analysis: `src/backtest/race_predictor.py` (1538 lines, inference pipeline)
- Direct codebase analysis: `src/domain/models.py` (305 lines, SubmodelSet/TrainedModelsV5 definitions)
- Direct codebase analysis: `src/db/model_loader.py` (943 lines, MLflow/local artifact loading)
- Direct codebase analysis: `src/models/benter_combination.py` (157 lines, existing Benter logit blend)
- Direct codebase analysis: `src/models/win_benter_gate.py` (80+ lines, existing Win Benter pipeline)
- Direct codebase analysis: `src/features/` (30 feature modules, feature engineering patterns)
- Project plan: `.planning/PROJECT.md` (v2.0 milestone definition, constraints, decisions)
- Improvement plan: `ROI_IMPROVEMENT_PLAN.md` (1021 lines, detailed implementation plan with phases)
