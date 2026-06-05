---
phase: 49-derived-higher-order-features
verified: 2026-06-05T02:56:49Z
status: passed
score: 4/4 must-haves verified
overrides_applied: 0
---

# Phase 49: Derived & Higher-Order Features Verification Report

**Phase Goal:** 馬個体の馬場状態適性・ペース予測・異常値検出・既存特徴量インタラクションが実装され、全特徴量層が揃う
**Verified:** 2026-06-05T02:56:49Z
**Status:** PASSED
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | 馬個体の含水率/クッション値適性(horse_dirt_wet_hit_rate等)がPIT-safeに計算され、過走履歴から適性カテゴリ(湿得意/乾得意/万能)が分類される | VERIFIED | `src/features/horse_track_aptitude.py`: `precompute_track_aptitude()` with expanding window + shift(1). 4 hit rates + `horse_condition_type` (wet_good/dry_good/balanced/unknown) with min_starts=3, threshold=0.3. 19 tests pass including PIT-safety, classification, versatility. |
| 2 | クッション値/含水率のコース別月別偏差(season_deviation)が計算される | VERIFIED | `src/features/track_condition_features.py`: `_compute_track_month_stats()` produces trackcd x month statistics. `cushion_season_deviation` and `moisture_season_deviation` computed as zscore in `compute_track_condition_features()`. 4 dedicated tests pass. |
| 3 | 含水率/クッション値から先行バイアススコア・蹴り返りリスク・ペース予測が算出され、レースフィールド条件マッチスコアが計算される | VERIFIED | T4-01: `track_front_bias_score` (linear interpolation), `kickback_risk_score` (inverse), `expected_pace_class` (3-level). T4-02: `race_condition_match_score`, `race_condition_match_max`, `race_condition_match_ratio`, `race_field_front_bias` in `compute_race_condition_features()`. 15 dedicated tests pass. |
| 4 | クッション/含水率異常値検出(2sigma逸脱) + 既存特徴量とのインタラクション(距離/馬齢/脚質等)が全て計算される | VERIFIED | T4-03: `cushion_anomaly_flag` and `moisture_extreme_flag` (|deviation| > 2). T4-04: `cushion_x_distance`, `moisture_x_weight`, `cushion_x_age`, `surface_condition_transition`. 12 dedicated tests pass. |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/features/horse_track_aptitude.py` | Core precompute with PIT-safe logic | VERIFIED | 263 lines. `precompute_track_aptitude()` + `APTITUDE_COLS` (14 columns). Expanding window + shift(1). Exports match plan. |
| `scripts/precompute_track_aptitude.py` | CLI precompute script | VERIFIED | 72 lines. Follows `precompute_career_stats.py` pattern. Loads entries/races/track_conditions parquets, writes horse_track_aptitude.parquet. |
| `src/db/readers.py` | `load_horse_track_aptitude(store)` | VERIFIED | Line 334. Standalone loader with exists-check returning empty DataFrame. |
| `src/db/repository.py` | `DataRepository.load_horse_track_aptitude(start, end)` | VERIFIED | Line 93. Date-filtered loader following `load_track_conditions` pattern. |
| `src/features/feature_engine.py` | T3 merge in build_all() | VERIFIED | Line 418-428. Left join on race_id + kettonum after track_conditions merge. Guarded with TimingContext. |
| `src/features/track_condition_features.py` | Extended with T3-04/T4-01/T4-03/T4-04 row + T4-02 race features | VERIFIED | 631 lines. TRACK_DERIVED_COLS (11), RACE_CONDITION_COLS (4), `_compute_track_month_stats()`, `compute_race_condition_features()`. All functions substantive. |
| `src/pipelines/training_pipeline.py` | track_month_stats + race_condition_features integration | VERIFIED | Lines 972-990: Computes track_month_stats, passes to feature functions, calls race_condition_features. Line 1569: Saves on SubmodelSet. |
| `src/backtest/race_predictor.py` | race_condition_features + track_month_stats inference | VERIFIED | Lines 256-266: Reads track_month_stats from SubmodelSet via getattr, calls both compute functions. |
| `src/domain/models.py` | SubmodelSet.track_month_stats field | VERIFIED | Line 276: `track_month_stats: dict | None = None` immediately after track_stats. |
| `config/settings.yaml` | track_condition thresholds | VERIFIED | Lines 49-55: dirt_wet_threshold (12.0), dirt_dry_threshold (3.0), turf_hard_threshold (10.0), turf_soft_threshold (8.0), hit_rate_threshold (0.3), min_starts (3). |
| `tests/test_horse_track_aptitude.py` | Test suite for T3 precompute | VERIFIED | 19 tests: unit + integration. All pass. |
| `tests/test_track_condition_features.py` | Test suite extended with T3/T4 tests | VERIFIED | 55 total tests (22 existing + 33 new). All pass. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `scripts/precompute_track_aptitude.py` | `data/raw/horse_track_aptitude.parquet` | `ParquetStore.write("raw", "horse_track_aptitude", stats)` | WIRED | Line 53 of precompute script calls `store.write()`. |
| `src/features/feature_engine.py` | `data/raw/horse_track_aptitude.parquet` | `DataRepository.load_horse_track_aptitude()` | WIRED | Lines 418-428: loads via repo, merges on race_id + kettonum with left join. |
| `src/pipelines/training_pipeline.py` | `src/features/track_condition_features.py` | `_compute_track_month_stats` + `compute_track_condition_features` + `compute_race_condition_features` | WIRED | Lines 972-990: computes stats, passes to both functions in correct order. |
| `src/backtest/race_predictor.py` | `src/features/track_condition_features.py` | `compute_track_condition_features` + `compute_race_condition_features` | WIRED | Lines 256-266: reads track_month_stats from SubmodelSet, calls both functions. |
| 6 included model FEATURE_COLS | TRACK_DERIVED_COLS + RACE_CONDITION_COLS | Direct inclusion | WIRED | All 6 models + PlaceTwoStageModel 3 lists verified via programmatic check: all 15 features present in each. |
| 4 excluded models | TRACK_DERIVED_COLS + RACE_CONDITION_COLS | Explicit exclusion | WIRED | MarketModel, RaceQualityScreener, RegimeDetector, ConformalEVModel: grep confirms zero matches for new feature names. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `compute_track_condition_features()` | track_front_bias_score | dirt_moisture/turf_cushion via linear interpolation | Yes (computed from real track condition values) | FLOWING |
| `compute_track_condition_features()` | cushion_season_deviation | track_month_stats dict lookup + turf_cushion | Yes (requires training period stats) | FLOWING |
| `compute_race_condition_features()` | race_condition_match_score | horse_dirt_wet_hit_rate/horse_cushion_hard_hit_rate from T3 aptitude | Yes (aggregated from per-horse aptitude) | FLOWING |
| `precompute_track_aptitude()` | horse_dirt_wet_hit_rate | expanding window + shift(1) on entries with kakuteijyuni | Yes (PIT-safe cumulative) | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| APTITUDE_COLS count = 14 | `python -c "from features.horse_track_aptitude import APTITUDE_COLS; print(len(APTITUDE_COLS))"` | 14 | PASS |
| TRACK_DERIVED_COLS = 11, RACE_CONDITION_COLS = 4 | `python -c "from features.track_condition_features import TRACK_DERIVED_COLS, RACE_CONDITION_COLS; print(len(TRACK_DERIVED_COLS), len(RACE_CONDITION_COLS))"` | 11 4 | PASS |
| All horse track aptitude tests pass | `python -m pytest tests/test_horse_track_aptitude.py -v` | 19/19 passed | PASS |
| All track condition feature tests pass | `python -m pytest tests/test_track_condition_features.py -v` | 55/55 passed | PASS |
| Domain tests (no regression) | `python -m pytest tests/test_domain.py -v` | 26/26 passed | PASS |
| Interaction feature tests (no regression) | `python -m pytest tests/test_interaction_features.py -v` | 32/32 passed | PASS |
| Surgical routing (all included models) | Programmatic check via Python | 9/9 model lists have all 15 features | PASS |
| Surgical routing (all excluded models) | grep on MarketModel, RegimeDetector, RaceQualityScreener, ConformalEVModel | 0 matches in all 4 | PASS |

### Probe Execution

Step 7c: SKIPPED -- no probe scripts defined for this phase.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| T3-01 | 49-01 | horse_dirt_wet_hit_rate / horse_dirt_dry_hit_rate | SATISFIED | `horse_track_aptitude.py` lines 139-152, 194-195. Tests: test_hit_definition, test_pit_safe_first_start. |
| T3-02 | 49-01 | horse_cushion_hard_hit_rate / horse_cushion_soft_hit_rate | SATISFIED | `horse_track_aptitude.py` lines 143-156, 196-197. Tests: test_turf_cushion_classification. |
| T3-03 | 49-01 | horse_condition_type (wet_good/dry_good/balanced/unknown) | SATISFIED | `horse_track_aptitude.py` lines 205-237. Tests: test_condition_type_wet_good, dry_good, balanced, unknown. |
| T3-04 | 49-02 | cushion_season_deviation / moisture_season_deviation | SATISFIED | `track_condition_features.py` lines 96-145, 308-374. Tests: test_cushion_season_deviation, test_season_deviation_std_zero, test_season_deviation_no_stats, test_compute_track_month_stats_basic. |
| T4-01 | 49-02 | track_front_bias_score / kickback_risk_score / expected_pace_class | SATISFIED | `track_condition_features.py` lines 266-306. Tests: test_dirt_front_bias_high_moisture, test_dirt_kickback_dry, test_turf_front_bias_mid_cushion, test_pace_class_slow/fast/neutral, test_nan_propagation (11 tests total). |
| T4-02 | 49-02 | race_condition_match_score / race_field_front_bias | SATISFIED | `track_condition_features.py` lines 452-630. Tests: test_race_condition_match_score, test_race_condition_match_max, test_race_condition_match_ratio, test_race_field_front_bias. |
| T4-03 | 49-02 | cushion_anomaly_flag / moisture_extreme_flag | SATISFIED | `track_condition_features.py` lines 379-389. Tests: test_cushion_anomaly_flag_triggered, test_cushion_anomaly_flag_normal, test_anomaly_flag_nan_propagation. |
| T4-04 | 49-02 | cushion_x_distance / moisture_x_weight / cushion_x_age / surface_condition_transition | SATISFIED | `track_condition_features.py` lines 391-442. Tests: test_cushion_x_distance, test_moisture_x_weight, test_cushion_x_age, test_surface_condition_transition_dirt/turf, test_nan variants (9 tests total). Note: moisture_x_prev_kyakusitu from REQUIREMENTS.md intentionally excluded per amended D-16 (Phase 48 dirt_moisture_x_kyakusitu already covers moisture x leg-type interaction). |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/pipelines/training_pipeline.py` | 7 | I001 import sort | Info | Pre-existing, not introduced by Phase 49. Not a blocker. |
| `src/backtest/race_predictor.py` | 1098, 1103, 1239 | TODO comments | Info | Pre-existing regime-related TODOs, not introduced by Phase 49. Not a blocker. |

No TBD, FIXME, or XXX markers found in any Phase 49 file.

### Deferred Items

Items not yet met but explicitly addressed in later milestone phases.

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | Feature Routing Audit (REG-02) for new features | Phase 50 | Phase 50 Success Criteria #1: "Feature Routing Audit pass for surgical routing" |
| 2 | POST_RACE CI verification (REG-03) | Phase 50 | Phase 50 Success Criteria #2: "POST_RACE 3-layer CI verification" |
| 3 | BT ROI 97%+ validation (VLD-01) | Phase 50 | Phase 50 Success Criteria #3: "BT ROI 97%+ achieved" |
| 4 | IC evaluation (VLD-02) | Phase 50 | Phase 50 Success Criteria #4: "IC evaluation for independence" |
| 5 | WF NaN availability check (VLD-03) | Phase 50 | Phase 50 Success Criteria #5: "WF Fold0 NaN rate check" |

### Gaps Summary

No gaps found. All 4 ROADMAP success criteria verified as TRUE in the codebase:

1. **PIT-safe aptitude precompute** -- expanding window + shift(1) pattern correctly implemented, 19 tests pass, FeatureEngine merge wired.
2. **Season deviation** -- trackcd x month zscore computation via `_compute_track_month_stats()`, integrated into training pipeline and race predictor.
3. **Bias/pace/race-level features** -- 7 features (T4-01 + T4-02) correctly computed via linear interpolation and race-level aggregation, 15 tests pass.
4. **Anomaly detection + interactions** -- 6 features (T4-03 + T4-04) correctly implemented with NaN-safe products and 2-sigma thresholds, 12 tests pass.

Surgical routing verified programmatically: all 9 included model FEATURE_COLS lists contain all 15 new features; all 4 excluded models contain zero new features.

Total test count: 74 tests across test_horse_track_aptitude.py (19) and test_track_condition_features.py (55), all passing. No regression in domain (26/26) or interaction features (32/32).

---

_Verified: 2026-06-05T02:56:49Z_
_Verifier: Claude (gsd-verifier)_
