---
phase: 36-feature-computation
plan: 01
subsystem: features
tags: [trf, int, race-rank, weighted-recent-form, interaction-features, model-registration]
dependency_graph:
  requires: []
  provides: [trf-01, trf-02, trf-03, int-01, int-02, int-03, int-04]
  affects: [horse_history_features, interaction_features, race_predictor, all-models]
tech_stack:
  added: []
  patterns: [ema-halflife3, race-rank-groupby, nan-safe-where-notna, feature-cols-registration]
key_files:
  created:
    - tests/test_trf_features.py
  modified:
    - src/features/horse_history_features.py
    - src/features/interaction_features.py
    - src/backtest/race_predictor.py
    - src/models/stage1_ability_model.py
    - src/models/two_stage_return_model.py
    - src/models/ev_correction_model.py
    - src/models/conformal_ev_model.py
    - src/models/market_model.py
    - src/models/place_ability_model.py
    - src/models/race_quality_screener.py
    - src/models/regime_detector.py
    - src/models/wide_two_stage_model.py
    - tests/test_interaction_features.py
    - tests/test_horse_history_features.py
decisions:
  - D-07: weighted_recent_form uses EMA halflife=3, same as harontimel5_avg
  - D-08: Two indicators - finish logit (absolute level) and timediff (time gap)
  - D-10: INT-01 = grade_code x form_trend numeric product
  - D-11: INT-02 = kyori x closing_index_avg numeric product
  - D-12: INT-03 = grade_code x blood_prize_log numeric product with NaN-safe .where()
  - D-13: Three new race_rank cols mirror existing groupby().rank(pct=True) pattern
  - D-15: All 8 new features registered in all 12 model FEATURE_COLS lists
metrics:
  duration: 746s
  completed: "2026-05-20"
  tasks: 2
  files: 15
---

# Phase 36 Plan 01: TRF/INT Feature Computation Summary

TRF race-rank transforms, weighted_recent_form (EMA halflife=3), 3 interaction products, and full model FEATURE_COLS registration across 12 models.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | TRF-01 race-rank + TRF-02 weighted_recent_form | 30b8c34 | horse_history_features.py, test_trf_features.py, test_horse_history_features.py |
| 2 | INT-01/02/03 + TRF-03/INT-04 model registration + RacePredictor mirror | 12001c4 | interaction_features.py, race_predictor.py, 9 model files, test_interaction_features.py |

## Changes Made

### TRF-01: add_race_transforms race_rank extensions
- Added `form_trend`, `blood_total_wr`, `blood_surface_wr` to race_rank_cols in `add_race_transforms()`
- These follow the same `groupby("race_id").rank(pct=True, method="average")` pattern as the existing 7 columns
- Gracefully skipped when columns not present via `if col not in df.columns: continue`

### TRF-02: weighted_recent_form (EMA halflife=3)
- `weighted_recent_form_finish`: EMA-weighted average of norm_finish_logit values from past races
- `weighted_recent_form_time`: EMA-weighted average of timediff (winner gap) values from past races
- Uses the same decay formula as harontimel5_avg: `decay = ln(2)/halflife`, weights reversed so newest has highest weight
- Both default to NaN when n_past == 0
- Both added to BASE_COLS (now 50 entries)

### INT-01: grade_x_form_trend
- Numeric product of grade_code (mapped via _GRADE_MAP: G1=5, G2=4, G3=3, OP=2) and form_trend
- NaN-safe via `.where(notna)` pattern

### INT-02: distance_x_closing_index
- Numeric product of kyori (continuous distance in meters) and closing_index_avg
- NaN-safe via `.where(notna)` pattern

### INT-03: grade_x_blood_prize_log
- Numeric product of grade_code numeric mapping and blood_prize_log
- Reuses grade_num computed for weight_x_class; NaN-safe

### TRF-03 + INT-04: Model FEATURE_COLS Registration
All 8 new features registered in all 12 model FEATURE_COLS:
- AbilityModel.FEATURE_COLS
- WinTwoStageModel.FEATURE_COLS
- PlaceTwoStageModel.HIT_FEATURE_COLS + RETURN_FEATURE_COLS
- EVCorrectionModel.FEATURE_COLS + PlaceEVCorrectionModel.FEATURE_COLS
- ConformalEVModel.FEATURE_COLS
- MarketModel.FEATURE_COLS
- PlaceAbilityModel.FEATURE_COLS
- RaceQualityScreener.FEATURE_COLS
- RegimeDetector.FEATURE_COLS
- WideTwoStageModel.SHARED_FEATURE_COLS

StackedEnsemble was NOT modified (has no FEATURE_COLS per Pitfall 6 in RESEARCH.md).

### RacePredictor Mirror
- `_race_rank_cols` extended from 7 to 10 items (added form_trend, blood_total_wr, blood_surface_wr)
- Inference path now generates the same race_rank columns as training path

## Deviations from Plan

None - plan executed exactly as written.

## Test Results

- test_trf_features.py: 5 passed (compute values, NaN verification, race_rank verification, BASE_COLS check)
- test_interaction_features.py: 32 passed (including 7 new INT tests + model registration verification)
- test_horse_history_features.py: 68 passed (BASE_COLS count updated from 48 to 50)
- test_post_race_leakage.py: 13 passed (Layer 2 still passes - no POST_RACE cols in FEATURE_COLS)
- Total: 118 passed

## Known Stubs

None.

## Threat Flags

None. All new features are derived from BASE_COLS (history) or pre-race columns. No POST_RACE data enters any feature computation.

## Self-Check: PASSED

All created/modified files verified present. Both task commits verified in git log.
