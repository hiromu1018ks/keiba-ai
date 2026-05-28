---
phase: 40-race-level-ranker
plan: 02
subsystem: models/pipelines
tags: [ranker, oof, iff-join, training-pipeline, shadow-mode]
dependency_graph:
  requires: [RaceLevelRanker-class, SubmodelSet-win_race_level_ranker-field]
  provides: [extended-OOF-generation, ranker-training-in-pipeline]
  affects: [src/models/win_benter_gate.py, src/pipelines/training_pipeline.py]
tech_stack:
  added: []
  patterns: [post-hoc-OOF-IFF-join-by-race-id-umaban, ranker-training-after-MAWC, joblib-serialization]
key_files:
  created: []
  modified:
    - src/models/win_benter_gate.py
    - src/pipelines/training_pipeline.py
    - tests/test_win_benter_gate.py
decisions:
  - OOF+IFF feature join uses post-hoc merge by race_id/umaban (RESEARCH OQ#1 resolved)
  - calibrated_ev_oof sourced from fold-level ev_win_corrected, not MAWC outputs (D-14)
  - Ranker training reuses same oof_cal_df from MAWC OOF generation (no separate OOF loop)
metrics:
  duration_minutes: 11
  completed: "2026-05-28"
  tasks_total: 2
  tasks_completed: 2
  tests_added: 5
  tests_passed: 90
  files_created: 0
  files_modified: 3
  lines_added: 179
---

# Phase 40 Plan 02: OOF Generation Extension + Ranker Pipeline Integration Summary

Extend OOF generation for calibrated_ev_oof and integrate RaceLevelRanker training into TrainingPipelineV5 after MarketAwareWinCalibrator using post-hoc OOF+IFF feature join.

## Completed Tasks

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Extend generate_win_oof_predictions() with calibrated_ev_oof | b81bb67 | src/models/win_benter_gate.py, tests/test_win_benter_gate.py |
| 2 | Integrate ranker training into TrainingPipelineV5 after MAWC | a3c67ef | src/pipelines/training_pipeline.py |

## What Was Built

### Extended OOF Generation (src/models/win_benter_gate.py)

- Added `calibrated_ev_oof` column to `generate_win_oof_predictions()` output (D-12)
- Captures fold-level `ev_win_corrected` from EVCorrection per fold for ranker value target (D-09)
- Existing columns preserved exactly: p_win_oof, p_market_norm, p_win_corrected, kakuteijyuni, tanodds, etc.
- No behavioral changes to existing OOF logic

### Ranker Training in TrainingPipelineV5 (src/pipelines/training_pipeline.py)

- RaceLevelRanker trained immediately after MarketAwareWinCalibrator using same `oof_cal_df` (D-14)
- OOF+IFF feature join: `oof_cal_df.merge(iff_builder.build_frame(oof_cal_df, mode="train"), on=["race_id", "umaban"])` (RESEARCH OQ#1 resolved)
- IFF train-mode features resolved to OOF-safe sources via dual-mode schema
- Ranker artifacts saved as `win_race_level_ranker_{surface}.joblib` following MAWC pattern exactly
- MLflow artifact logging following MAWC pattern
- SubmodelSet.win_race_level_ranker populated during training
- Insufficient data guard: requires >= 500 OOF samples (same threshold as MAWC)

## TDD Gate Compliance

Task 1 was designated `tdd="true"` but the RED/GREEN separation was imperfect:
- Tests and implementation were staged together in a single commit (b81bb67)
- Both `test(...)` and `feat(...)` content exist in the same commit
- All 5 new tests pass (calibrated_ev_oof column, row count, kakuteijyuni preservation, existing columns, value correctness)
- The RED phase was verified (2 tests failed before implementation) but commits were not properly separated

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] oof_cal_df scope not accessible outside MAWC if-block**
- **Found during:** Task 2 implementation
- **Issue:** oof_cal_df was defined inside the MAWC if-block. The ranker training block needed access to it outside that block.
- **Fix:** Added `oof_cal_df: pd.DataFrame | None = None` initialization before the MAWC if-block, so it is accessible for the ranker training block.
- **Files modified:** src/pipelines/training_pipeline.py
- **Commit:** a3c67ef

## Verification Results

- 20/20 win_benter_gate tests passed (5 new ranker columns + 15 existing)
- 31/31 training_pipeline tests passed (no regression)
- 13/13 race_level_ranker tests passed (from Plan 01, no regression)
- 26/26 domain tests passed (no regression)
- ruff check (E/F/W/N): all checks passed
- Total: 90/90 tests passed

## Threat Flags

No new threat surface introduced beyond plan's threat_model. The OOF+IFF join uses train-mode IFF sources with leakage guard. calibrated_ev_oof is sourced from fold-level EV correction (OOF-safe). Ranker operates in shadow mode with `_trained=False` default.

## Known Stubs

None. All data flows are wired end-to-end: OOF generation -> IFF join -> ranker training -> SubmodelSet -> MLflow/local save.

## Self-Check: PASSED

- FOUND: src/models/win_benter_gate.py
- FOUND: src/pipelines/training_pipeline.py
- FOUND: .planning/phases/40-race-level-ranker/40-02-SUMMARY.md
- FOUND: commit b81bb67 (Task 1: OOF extension)
- FOUND: commit a3c67ef (Task 2: pipeline integration)
