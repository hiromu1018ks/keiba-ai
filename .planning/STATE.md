---
gsd_state_version: 1.0
milestone: v2.4
milestone_name: Paper Trading Pipeline Integration
status: planning
last_updated: "2026-06-05T22:48:08.770Z"
last_activity: 2026-06-05
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-05)

**Core value:** 単勝モデルのバックテストROIを100%超えにすること
**Current focus:** Planning next milestone

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-06-05 — Milestone v2.4 started

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

### v2.3 Shipped Artifacts

- data/raw/track_conditions.parquet — 23,259 rows, 13,323 dirt_moisture, 9,936 turf_cushion
- data/raw/horse_track_aptitude.parquet — 14-column PIT-safe precompute
- src/features/track_condition_features.py — 23 features: TRACK_CONDITION_COLS(8) + TRACK_DERIVED_COLS(11) + RACE_CONDITION_COLS(4)
- src/features/horse_track_aptitude.py — precompute_track_aptitude() + APTITUDE_COLS(14)
- FeatureEngine.build_all() — track_conditions + horse_track_aptitude merge integrated
- Surgical routing: 6 included models (all 23 features), 4 excluded (0 features)
- CI: 55 track condition tests + 17 safety CI tests + 16 IC eval tests
- IC evaluation script: scripts/run_track_condition_ic_eval.py
- NaN diagnostic script: scripts/validate_track_condition_nan.py

### Blockers/Concerns

- なし (次マイルストーンの要件定義待ち)

## Deferred Items

| Category | Item | Status |
|----------|------|--------|
| Validation | WF検証スクリプトの実際の実行(~4時間、PostgreSQL環境必要) | Pending since v1.0 |
| UAT | Human UAT 5項目 (PostgreSQL依存) | Pending since v1.4 |
| Bug | test_training_pipeline.py 3件既知失敗 | Pending since v1.6 |
| Bug | 4 pre-existing test failures (observed_true, blood_keito, ev_oof, profit_selector) | Pending |
| Cleanup | WinSegmentCalibrator dead code removal (WRN-01) | Pending since v2.1 |
| Feature | 4 RACE_CONDITION特徴量100% NaN修正 (track_month_stats availability) | Pending since v2.3 |
| Feature | sire_x_cushion_band 51% NaN改善 (種牡馬×クッション交差データ不足) | Pending since v2.3 |
| Optimization | Optuna 19次元パラメータ最適化 (DEP-02) | Deferred to v2.4+ |
| Automation | デプロイゲート自動判定 (DEP-01) | Deferred to v2.4+ |
| Calibration | Conservative MAWC redesign / selective interaction experiment | Deferred to v2.4+ |
| Validation | IC評価レポート生成 (OOF予測必要、別途run_train.py) | Pending since v2.3 |

## Session Continuity

Last session: 2026-06-05
Stopped at: v2.3 shipped and archived
