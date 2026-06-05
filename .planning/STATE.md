---
gsd_state_version: 1.0
milestone: v2.3
milestone_name: Track Condition Feature Integration
status: executing
stopped_at: Phase 50 context gathered
last_updated: "2026-06-05T05:55:55.788Z"
last_activity: 2026-06-05 -- Phase 50 execution started
progress:
  total_phases: 4
  completed_phases: 3
  total_plans: 7
  completed_plans: 5
  percent: 71
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-04)

**Core value:** 単勝モデルのバックテストROIを100%超えにすること
**Current focus:** Phase 50 — safety-validation

## Current Position

Phase: 50 (safety-validation) — EXECUTING
Plan: 1 of 2
Status: Executing Phase 50
Last activity: 2026-06-05 -- Phase 50 execution started

Progress: [████████░░] 75%

## Accumulated Context

### Decisions

- D-04: entry_id(18-digit) = race_id(first 16) + umaban(last 2) (47-01)
- D-05: ValueError on multiple distinct non-NaN values per race_id (47-01)
- D-06: NaN/non-NaN mix resolves to non-NaN (47-01)
- D-07: Cross-validation against races.parquet logs only (47-01)
- D-08: NaN values preserved as-is (no statistical imputation) (47-01)
- D-09: Physical outliers NaN-ified (dirt: 0<x<100, turf: x>0) (47-01)
- D-10: load_track_conditions follows existing load_* pattern with exists-check gate (47-02)
- D-11: POST_RACE_COLS CI test ensures track condition columns are safe for ML features (47-02)
- Coarse granularity: 4 phases compressed from 7 requirement categories
- Tier 1+2 combined (Phase 48): Both are direct interaction features from same data source
- Tier 3+4 combined (Phase 49): Derived features depend on Tier 1/2 being registered
- REG-01 (feature registration) moved to Phase 48: Features must be registered to be usable
- REG-02/REG-03 in Phase 50: Routing audit and POST_RACE CI validate the complete feature set

### Phase 47 Artifacts

- data/raw/track_conditions.parquet — 23,259 rows, 13,323 dirt_moisture, 9,936 turf_cushion (2018-07 ~ 2026-05)
- DataRepository.load_track_conditions(start, end) — date-filtered access to track conditions
- 22 new tests: 16 (track_condition_data) + 4 (repository) + 2 (POST_RACE CI)

### Phase 48 Artifacts

- src/features/track_condition_features.py — 8 T1/T2 features: compute_track_condition_features() + _compute_track_stats()
- FeatureEngine.build_all() merges dirt_moisture/turf_cushion from track_conditions.parquet
- _train_submodel() / RacePredictor: track_condition_features computed between HorseHistoryFeatures and interaction_features
- SubmodelSet.track_stats: training-period statistics for T1-02 relative/zscore features
- Surgical routing: 8 models included (AbilityModel, Win/PlaceTwoStage, EVCorrection, PlaceEVCorrection, PlaceAbility, WideTwoStage), 4 excluded (MarketModel, RaceQualityScreener, RegimeDetector, ConformalEVModel)
- 22 new tests + 4 downstream test propagation fixes

### Phase 49 Plan 01 Artifacts

- src/features/horse_track_aptitude.py — precompute_track_aptitude() + APTITUDE_COLS (14 columns)
- scripts/precompute_track_aptitude.py — CLI precompute script (career_stats pattern)
- src/db/readers.py — load_horse_track_aptitude(store) standalone function
- src/db/repository.py — DataRepository.load_horse_track_aptitude(start, end)
- src/features/feature_engine.py — T3 merge in build_all() on race_id + kettonum (left join)
- 19 new tests: 16 unit + 3 integration

### Phase 49 Plan 02 Artifacts

- src/features/track_condition_features.py — TRACK_DERIVED_COLS (11), RACE_CONDITION_COLS (4), _compute_track_month_stats(), compute_race_condition_features()
- src/pipelines/training_pipeline.py — track_month_stats computation + race_condition_features call
- src/backtest/race_predictor.py — race_condition_features call + track_month_stats inference
- src/domain/models.py — SubmodelSet.track_month_stats field
- 6 included models: TRACK_DERIVED_COLS + RACE_CONDITION_COLS in FEATURE_COLS (AbilityModel, Win/PlaceTwoStage, EVCorrection, PlaceEVCorrection, PlaceAbility, WideTwoStage)
- 4 excluded models: MarketModel, RaceQualityScreener, RegimeDetector, ConformalEVModel
- config/settings.yaml — track_condition thresholds section
- 33 new tests (55 total)

### Blockers/Concerns

- クッション値データは2020/09開始のためWF Fold0(2020学習)でNaN率高い可能性 (VLD-03で検証)

## Deferred Items

| Category | Item | Status |
|----------|------|--------|
| Validation | WF検証スクリプトの実際の実行(~4時間、PostgreSQL環境必要) | Pending since v1.0 |
| UAT | Human UAT 5項目 (PostgreSQL依存) | Pending since v1.4 |
| Bug | test_training_pipeline.py 3件既知失敗 | Pending since v1.6 |
| Bug | 4 pre-existing test failures (observed_true, blood_keito, ev_oof, profit_selector) | Pending |
| Cleanup | WinSegmentCalibrator dead code removal (WRN-01) | Pending since v2.1 |
| Optimization | Optuna 19次元パラメータ最適化 (DEP-02) | Deferred to v2.4+ |
| Automation | デプロイゲート自動判定 (DEP-01) | Deferred to v2.4+ |
| Calibration | Conservative MAWC redesign / selective interaction experiment | Deferred to v2.4+ |

## Session Continuity

Last session: 2026-06-05T04:33:47.951Z
Stopped at: Phase 50 context gathered
Resume file: .planning/phases/50-safety-validation/50-CONTEXT.md
