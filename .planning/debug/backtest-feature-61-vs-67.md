---
status: resolved
trigger: "LightGBMError: The number of features in data (61) is not the same as it was in training data (67)"
created: 2026-05-17T10:00:00+09:00
updated: 2026-05-17T12:00:00+09:00
---

## Current Focus

root_cause_found: true
hypothesis: "6 features missing in inference path: 3 from compute_stage2_relative_features and 3 from TargetEncoder -- both computed during training but never applied during inference"
next_action: "Fix applied"

## Symptoms

expected: BacktestEngine inference should provide all 67 features that LightGBM model was trained on
actual: "LightGBMError: The number of features in data (61) is not the same as it was in training data (67)"
errors: "lightgbm.basic.LightGBMError in StackedEnsemble.predict() -> lgbm_model.predict()"
reproduction: "Run backtest with --ensemble: training succeeds, then _collect_training_bet_history -> train_engine.run() -> RacePredictor.predict() -> WinTwoStageModel.predict_ev() -> StackedEnsemble.predict() fails with 61 vs 67"
started: "After commit 471decf which fixed relative_features timing (92 vs 95)"

## Eliminated

- FeatureEngine.build_all() feature generation -- produces same base features
- HorseHistoryFeatures merge timing -- already fixed in prior session
- DamPedigreeFeatures/RecordFeatures/MiningFeatures merge keys -- already fixed in prior sessions

## Evidence

- timestamp: 2026-05-17T10:30:00
  checked: "WinTwoStageModel.FEATURE_COLS (81 total) vs training-time availability"
  found: "Training produces 67 available features (14 not available). Inference produces only 61 (20 not available). Difference = 6 features present in training but missing in inference."
  implication: "6 features computed during training are never computed during inference"

- timestamp: 2026-05-17T10:45:00
  checked: "Feature generation order comparison: training (_train_submodel) vs inference (RacePredictor.predict)"
  found: |
    Training path before WinTwoStageModel.predict_ev():
    1. compute_stage2_relative_features -> rel_p_ability_win_zscore, rel_p_ability_win_rank, rel_odds_ability_deviation (3 features)
    2. TargetEncoder.fit_transform_oof -> te_blood_keito_cd, te_kisyucode, te_chokyosicode (3 features)

    Inference path before WinTwoStageModel.predict_ev():
    - Neither compute_stage2_relative_features nor TargetEncoder.transform are called
    - These 6 features are silently skipped by _prepare_features() (line 232: available_cols = [c for c in FEATURE_COLS if c in df.columns])
  implication: "Exactly 6 features missing = 3 stage2 relative + 3 target encoding. Matches 67-61=6 gap."

- timestamp: 2026-05-17T11:00:00
  checked: "TargetEncoder storage -- is it persisted in SubmodelSet?"
  found: "TargetEncoder is NOT stored in SubmodelSet. It is a local variable in _train_submodel that is discarded after training."
  implication: "TargetEncoder must be added to SubmodelSet and stored during training for inference-time application"

- timestamp: 2026-05-17T11:30:00
  checked: "odds_to_ability_ratio availability for compute_stage2_relative_features"
  found: "compute_stage2_relative_features needs odds_to_ability_ratio as a base column. In inference, odds_to_ability_ratio is only computed inside _prepare_features() (fallback). Must compute it explicitly before compute_stage2_relative_features."
  implication: "Need to add odds_to_ability_ratio computation in RacePredictor.predict() before calling compute_stage2_relative_features"

## Resolution

root_cause: |
  Two feature groups computed during training but missing from the inference path:

  1. compute_stage2_relative_features (3 features): rel_p_ability_win_zscore,
     rel_p_ability_win_rank, rel_odds_ability_deviation. Called in training_pipeline.py
     _train_submodel() line 586 after p_ability_win is available. Never called in
     RacePredictor.predict().

  2. TargetEncoder (3 features): te_blood_keito_cd, te_kisyucode, te_chokyosicode.
     Created and fit_transform_oof'd in _train_submodel() line 565-570 but never stored
     in SubmodelSet. Not applied during inference.

  Additionally, odds_to_ability_ratio (a dependency of compute_stage2_relative_features)
  must be computed before calling compute_stage2_relative_features, since it is only
  computed as a fallback inside _prepare_features() which runs too late.

fix: |
  Three changes:
  1. src/domain/models.py: Added `target_encoder: TargetEncoder | None = None` to SubmodelSet dataclass
  2. src/pipelines/training_pipeline.py: Store te_encoder in SubmodelSet constructor
  3. src/backtest/race_predictor.py: Added three steps before WinTwoStageModel.predict_ev():
     a. Compute odds_to_ability_ratio (if p_market_win_adj and p_ability_win are available)
     b. Call compute_stage2_relative_features(df)
     c. Apply target_encoder.transform(df) (if encoder is available)
  4. tests: Added `target_encoder = None` to all mock submodels

verification: |
  - All 1526 tests pass (0 failures)
  - No new ruff errors introduced (only pre-existing ones)
  - Feature count at inference should now be 67, matching training

files_changed:
  - src/domain/models.py
  - src/pipelines/training_pipeline.py
  - src/backtest/race_predictor.py
  - tests/test_backtest_engine.py
  - tests/test_race_predictor.py
