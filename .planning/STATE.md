---
gsd_state_version: 1.0
milestone: v2.3
milestone_name: Track Condition Feature Integration
status: planning
stopped_at: Phase 48 verified PASSED
last_updated: "2026-06-05T00:00:00.000Z"
last_activity: 2026-06-05 — Phase 48 VERIFICATION.md PASSED (human_needed, approved)
progress:
  total_phases: 4
  completed_phases: 2
  total_plans: 3
  completed_plans: 3
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-04)

**Core value:** 単勝モデルのバックテストROIを100%超えにすること
**Current focus:** v2.3 Track Condition Feature Integration — 含水率/クッション値特徴量統合でBT ROI 97%+回復

## Current Position

Phase: 49 of 50 (Derived & Higher-Order Features) — next
Plan: —
Status: Phase 48 verified PASSED, ready for Phase 49 planning
Last activity: 2026-06-05 — Phase 48 VERIFICATION.md PASSED (human_needed, approved)

Progress: [████░░░░░░] 50%

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

Last session: 2026-06-05T00:00:00.000Z
Stopped at: Phase 48 verified PASSED
Resume file: .planning/phases/49-derived-features/
