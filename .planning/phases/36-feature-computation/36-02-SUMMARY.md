---
phase: 36-feature-computation
plan: 02
subsystem: features
tags: [hlf, harontime-l4, laptime-pace, race-rank, model-registration, pit-safe]
dependency_graph:
  requires: ["36-01"]
  provides: [hlf-01, hlf-02, hlf-03, hlf-04, hlf-05]
  affects: [horse_history_features, race_predictor, all-models]
tech_stack:
  added: []
  patterns: [ema-halflife3, expanding-stats-hierarchical, searchsorted-pit-safe, pace-ratio-segments, np-array-split]
key_files:
  created:
    - tests/test_hlf_features.py
  modified:
    - src/features/horse_history_features.py
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
    - tests/test_horse_history_features.py
decisions:
  - D-01: DISTANCE_THRESHOLD=2000 for harontime_last3f unified column
  - D-02: L3 expanding_stats used as z-score proxy for unified column (more coverage)
  - D-04: n_laps = kyori / 200 for LapTime segment division
  - D-05: pace_ratio = late_avg / early_avg (< 1.0 = closing fast)
  - D-06: LapTime pace_ratio built from races_hist lookup, not past_df merge
metrics:
  duration: 693s
  completed: "2026-05-20"
  tasks: 2
  files: 13
---

# Phase 36 Plan 02: HLF HaronTime L4 / LapTime Pace Features Summary

HaronTime L4 history stats (avg/zscore/trend), harontime_last3f unified column with distance-based auto-selection, LapTime pace_ratio features from past races, and full model FEATURE_COLS registration across 10 model classes.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | HLF-01/02 HaronTime L4 history + unified column + race_rank | db59071 | horse_history_features.py, race_predictor.py, test_hlf_features.py |
| 2 | HLF-03 LapTime pace + HLF-04 model registration + HLF-05 dual-path | fef7110 | horse_history_features.py, 8 model files, test_hlf_features.py, test_horse_history_features.py |

## Changes Made

### HLF-01: HaronTime L4 History Stats

- Added `harontimel4` to `cols_horse` list for dict-of-numpy array lookup
- Added `_has_harontimel4` flag (backward compatible: NaN when column absent from data)
- Pre-computed `expanding_stats_hl4` dict parallel to existing `expanding_stats` for L3, using same FALLBACK_LEVELS structure
- Computed `harontimel4_avg` using EMA halflife=3 (same pattern as harontimel5_avg)
- Computed `harontimel4_zscore` using hierarchical expanding_stats lookup
- Computed `harontimel4_trend` as linear regression slope of last 3 valid L4 values

### HLF-01: harontime_last3f Unified Column

- `DISTANCE_THRESHOLD = 2000` constant at module level
- Per-race selection: if current_kyori >= 2000, prefer L4 with L3 fallback; if < 2000, prefer L3 with L4 fallback
- Computed `harontime_last3f_avg` (EMA halflife=3 of unified values)
- Computed `harontime_last3f_zscore` using L3 expanding_stats as proxy (more data coverage)
- Computed `harontime_last3f_trend` as linear regression slope

### HLF-02: Race-Rank Extensions

- Extended `add_race_transforms()` race_rank_cols with `harontimel4_avg` and `harontime_last3f_avg`
- Mirrored in `RacePredictor._race_rank_cols` for inference path parity

### HLF-03: LapTime Pace Features

- Pre-computed per-race pace_ratio from LapTime1~25 in races_hist (not past_df)
- Used np.array_split for 3-segment division; n_laps = kyori / 200
- pace_ratio = late_avg / early_avg (D-05)
- Built per-horse pace history lookup via entries_hist x race_pace_lookup join
- Computed `pace_ratio_avg` (EMA halflife=3), `pace_ratio_zscore` (global), `pace_ratio_trend` (linear regression)
- Computed `pace_early_avg`, `pace_mid_avg`, `pace_late_avg` as simple means of past segment values

### HLF-04: Model FEATURE_COLS Registration

All 14 HLF features registered in all 10 model classes (12 FEATURE_COLS lists):
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

StackedEnsemble NOT modified (has no FEATURE_COLS per Pitfall 6 in RESEARCH.md).

### BASE_COLS Extension

BASE_COLS expanded from 50 to 62 entries:
- 6 HaronTime features (harontimel4_avg/zscore/trend, harontime_last3f_avg/zscore/trend)
- 6 LapTime pace features (pace_ratio_avg/zscore/trend, pace_early_avg, pace_mid_avg, pace_late_avg)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] LapTime data source mismatch**
- **Found during:** Task 2 implementation
- **Issue:** Plan assumed LapTime columns would be available in past_df_sorted, but the race_cols_all merge excludes LapTime1~25 columns
- **Fix:** Built _pace_lookup from races_hist directly (per-race pace_ratio computation), then matched to horses via entries_hist race_id. This avoids modifying the race_cols_all merge while maintaining PIT-safety
- **Files modified:** horse_history_features.py
- **Commit:** fef7110

**2. [Rule 1 - Bug] Test data syussotosu conflict**
- **Found during:** Task 1 TDD RED phase
- **Issue:** Test past_entries included syussotosu column, which conflicts with the same column from races_hist merge (creates syussotosu_x / syussotosu_y)
- **Fix:** Removed syussotosu from past_entries test data (it comes from races_hist)
- **Files modified:** tests/test_hlf_features.py
- **Commit:** 4c9f3d0

## Test Results

- test_hlf_features.py: 157 passed (14 HaronTime + 3 LapTime + 140 model registration)
- test_horse_history_features.py: 68 passed (BASE_COLS count updated from 50 to 62)
- test_post_race_leakage.py: 13 passed (Layer 2 still passes)
- test_trf_features.py: 5 passed (Plan 01 tests still pass)
- test_interaction_features.py: 32 passed (Plan 01 tests still pass)
- Total: 275 passed

## Known Stubs

None.

## Threat Flags

None. All HLF features are derived from past-race data only. LapTime and HaronTime columns are POST_RACE_COLS but only accessed via searchsorted(target_date, side="left") which ensures strictly past data. Current-race LapTime/HaronTime values are never used in feature computation.

## Self-Check: PASSED

All created/modified files verified present. All three task commits verified in git log.
