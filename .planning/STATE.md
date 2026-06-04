---
gsd_state_version: 1.0
milestone: v2.3
milestone_name: Track Condition Feature Integration
status: executing
stopped_at: Phase 47 Plan 02 complete
last_updated: "2026-06-04T20:50:18Z"
last_activity: 2026-06-04 — Plan 47-02 complete (DataRepository.load_track_conditions + POST_RACE CI test)
progress:
  total_phases: 4
  completed_phases: 0
  total_plans: 8
  completed_plans: 2
  percent: 25
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-04)

**Core value:** 単勝モデルのバックテストROIを100%超えにすること
**Current focus:** v2.3 Track Condition Feature Integration — 含水率/クッション値特徴量統合でBT ROI 97%+回復

## Current Position

Phase: 47 of 50 (ETL Data Pipeline)
Plan: 02 of 02 (complete)
Status: Plan 47-02 done — DataRepository.load_track_conditions + POST_RACE CI test
Last activity: 2026-06-04 — Plan 47-02 complete

Progress: [██░░░░░░░░] 25%

## Accumulated Context

### Decisions

- D-10: load_track_conditions follows existing load_* pattern with exists-check gate (47-02)
- D-11: POST_RACE_COLS CI test ensures track condition columns are safe for ML features (47-02)
- Coarse granularity: 4 phases compressed from 7 requirement categories
- Tier 1+2 combined (Phase 48): Both are direct interaction features from same data source
- Tier 3+4 combined (Phase 49): Derived features depend on Tier 1/2 being registered
- REG-01 (feature registration) moved to Phase 48: Features must be registered to be usable
- REG-02/REG-03 in Phase 50: Routing audit and POST_RACE CI validate the complete feature set

### Blockers/Concerns

- クッション値データは2020/09開始のためWF Fold0(2020学習)でNaN率高い可能性 (VLD-03で検証)

## Deferred Items

| Category | Item | Status |
|----------|------|--------|
| Validation | WF検証スクリプトの実際の実行(~4時間、PostgreSQL環境必要) | Pending since v1.0 |
| UAT | Human UAT 5項目 (PostgreSQL依存) | Pending since v1.4 |
| Bug | test_training_pipeline.py 3件既知失敗 | Pending since v1.6 |
| Cleanup | WinSegmentCalibrator dead code removal (WRN-01) | Pending since v2.1 |
| Optimization | Optuna 19次元パラメータ最適化 (DEP-02) | Deferred to v2.4+ |
| Automation | デプロイゲート自動判定 (DEP-01) | Deferred to v2.4+ |
| Calibration | Conservative MAWC redesign / selective interaction experiment | Deferred to v2.4+ |

## Session Continuity

Last session: 2026-06-04T20:50:18Z
Stopped at: Phase 47 Plan 02 complete
Resume file: .planning/phases/47-etl-data-pipeline/47-CONTEXT.md
