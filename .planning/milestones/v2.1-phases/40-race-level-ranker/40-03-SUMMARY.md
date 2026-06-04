---
phase: 40-race-level-ranker
plan: 03
subsystem: backtest/db
tags: [ranker, shadow-mode, D-18-diagnostics, model-loader, investment-score]
dependency_graph:
  requires: [RaceLevelRanker-class, OOF-generation-extension, ranker-training-in-pipeline]
  provides: [ranker-inference-integration, ranker-persistence, D-18-shadow-diagnostics]
  affects: [src/backtest/race_predictor.py, src/db/model_loader.py]
tech_stack:
  added: []
  patterns: [shadow-mode-via-getattr-is_trained, post-score-D-18-comparison, MAWC-load-pattern-reuse]
key_files:
  created: []
  modified:
    - src/backtest/race_predictor.py
    - src/db/model_loader.py
    - tests/test_race_predictor.py
decisions:
  - D-18 diagnostics computed in get_win_candidates() not predict() because win_market_selection_score
    is only available after baseline scoring in get_win_candidates()
  - Ranker scoring block in predict() adds investment_score columns to ALL runners (D-20)
  - ModelLoader follows MAWC pattern exactly: local + MLflow paths with try/except
metrics:
  duration_minutes: 12
  completed: "2026-05-28"
  tasks_total: 2
  tasks_completed: 2
  tests_added: 10
  tests_passed: 120
  files_created: 0
  files_modified: 3
  lines_added: 407
---

# Phase 40 Plan 03: RacePredictor Integration + ModelLoader Persistence Summary

Integrate trained ranker into RacePredictor inference path with D-18 shadow diagnostics and ModelLoader persistence for end-to-end integration.

## Completed Tasks

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Ranker scoring block + D-18 shadow diagnostics in RacePredictor | 8040908 | src/backtest/race_predictor.py, tests/test_race_predictor.py |
| 2 | Ranker load/save in ModelLoader | eb81a3f | src/db/model_loader.py |

## What Was Built

### Ranker Scoring Block (src/backtest/race_predictor.py)

- **predict()**: Added `ranker.score(df)` call after MAWC, before WinSelectionGate
  - Uses `getattr(submodel, "win_race_level_ranker", None)` + `is_trained` guard
  - When ranker is trained, adds investment_score columns to ALL runners (D-20)
  - When ranker is None or not trained, predict() behavior identical to before
  - investment_score is shadow-only diagnostic, does NOT alter baseline selection

- **get_win_candidates()**: Added D-18 per-race shadow diagnostics after win_market_selection_score computation
  - `baseline_selected_umaban`: umaban of horse with highest win_market_selection_score per race
  - `ranker_selected_umaban`: umaban of horse with highest investment_score per race
  - `baseline_ranker_agreement`: True/False whether both selectors picked the same horse
  - These columns enable agreement rate calculation in backtest reporting (Phase 41)

### ModelLoader Persistence (src/db/model_loader.py)

- **load_from_dir()** (local path): Added RaceLevelRanker loading after MAWC
  - File pattern: `data/models/win_race_level_ranker_{surface}.joblib`
  - try/except with warning on failure, same as MAWC pattern

- **load()** (MLflow path): Added RaceLevelRanker loading after MAWC
  - Downloads artifacts from `runs:/{run_id}/win_race_level_ranker_{surface}`
  - Falls back to `_find_artifact_dir()` if MLflow download fails
  - try/except with warning on failure

- Both SubmodelSet constructions updated with `win_race_level_ranker=win_race_level_ranker`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] D-18 diagnostics placement moved from predict() to get_win_candidates()**
- **Found during:** Task 1 implementation
- **Issue:** The plan specified D-18 diagnostics in predict() after MAWC, but win_market_selection_score is computed inside get_win_candidates(), not predict(). The comparison between baseline and ranker requires both scores to exist.
- **Fix:** Split implementation: ranker.score() in predict() (adds investment_score), D-18 comparison in get_win_candidates() (where win_market_selection_score exists).
- **Files modified:** src/backtest/race_predictor.py
- **Commit:** 8040908

## Verification Results

- 74/74 race_predictor tests passed (64 existing + 10 new)
- 7/7 model_loader tests passed (no regression)
- 13/13 race_level_ranker tests passed (from Plan 01, no regression)
- 26/26 domain tests passed (no regression)
- ruff check: all checks passed
- Total: 120/120 tests passed

## Threat Flags

No new threat surface introduced. The ranker scoring follows the exact shadow-mode pattern: getattr + is_trained guard ensures no impact when ranker is unavailable. D-18 diagnostic columns are clearly labeled shadow diagnostics and are not used in any selection/sorting logic. ModelLoader uses try/except with graceful degradation on load failure.

## Known Stubs

None. All data flows are wired end-to-end: ranker.score() -> investment_score columns -> D-18 diagnostics in get_win_candidates() -> ModelLoader load/save.

## Self-Check: PASSED

- FOUND: src/backtest/race_predictor.py
- FOUND: src/db/model_loader.py
- FOUND: .planning/phases/40-race-level-ranker/40-03-SUMMARY.md
- FOUND: commit 8040908 (Task 1: ranker scoring + D-18 diagnostics)
- FOUND: commit eb81a3f (Task 2: ModelLoader persistence)
