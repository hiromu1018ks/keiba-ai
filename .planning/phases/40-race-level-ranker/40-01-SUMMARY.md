---
phase: 40-race-level-ranker
plan: 01
subsystem: models
tags: [ranker, ridge, shadow-mode, investment-score, tdd]
dependency_graph:
  requires: []
  provides: [RaceLevelRanker-class, SubmodelSet-win_race_level_ranker-field]
  affects: [src/models/race_level_ranker.py, src/domain/models.py]
tech_stack:
  added: [sklearn.linear_model.Ridge, sklearn.metrics.ndcg_score, scipy.stats.spearmanr]
  patterns: [shadow-mode-via-is_trained, per-surface-independent-models, wf-alpha-grid-selection, joblib-serialization]
key_files:
  created:
    - src/models/race_level_ranker.py
    - tests/test_race_level_ranker.py
  modified:
    - src/domain/models.py
decisions:
  - Per-surface Ridge models (4 total) stored as flat fields on single RaceLevelRanker container
  - rel_p_ability_win_rank computed at training/scoring time from if_p_ability_win groupby rank (not added to IFF schema)
  - if_odds_rank and if_abs_logit_gap derived at training/scoring time inside ranker
  - df_surf.reset_index(drop=True) required for WF splits to match numpy array indices
metrics:
  duration_minutes: 15
  completed: "2026-05-28"
  tasks_total: 2
  tasks_completed: 2
  tests_added: 13
  tests_passed: 13
  files_created: 2
  files_modified: 1
  lines_added: 621
---

# Phase 40 Plan 01: RaceLevelRanker Class Summary

Per-surface Ridge ranker (relevance + value) with shadow mode, alpha grid selection, D-11 diagnostics, and SubmodelSet integration.

## Completed Tasks

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | RaceLevelRanker class with training, scoring, persistence, and D-11 diagnostics | f5be1bd (test), 7954e10 (feat) | src/models/race_level_ranker.py, tests/test_race_level_ranker.py |
| 2 | Add win_race_level_ranker field to SubmodelSet | 31356d5 | src/domain/models.py |

## What Was Built

### RaceLevelRanker Class (src/models/race_level_ranker.py, 621 lines)

- **4 Ridge models**: relevance_scorer_turf/dirt, value_scorer_turf/dirt
- **Alpha grid selection**: [0.03, 0.1, 0.3, 1.0, 3.0, 10.0] with WF fold validation
  - Relevance: NDCG@3 primary metric, tie-breaker larger alpha
  - Value: Spearman rank correlation primary metric, tie-breaker larger alpha
- **Targets**:
  - Relevance: graded {1.00, 0.55, 0.30, 0.10, 0.00} by kakuteijyuni (D-08)
  - Value: composite clipped_log_ev + mispricing_bonus - uncertainty_penalty (D-09)
- **Score output columns**: relevance_score, value_score, relevance_score_pct, value_score_pct, calibrated_log_ev_pct, uncertainty_penalty_pct, investment_score
- **Combination**: investment_score = 0.35*rel_pct + 0.35*val_pct + 0.20*log_ev_pct - 0.10*uncertainty_pct (D-03)
- **D-11 diagnostics per surface**: top1_win_rate, ndcg_at_3, rank_of_actual_winner, top3_contains_winner
- **Shadow mode**: is_trained property, deployment_status="shadow_only", score() returns df unchanged when not trained
- **Persistence**: save()/load() via joblib following MAWC pattern exactly

### SubmodelSet Field (src/domain/models.py)

- Added `win_race_level_ranker: RaceLevelRanker | None = None` field
- Import under TYPE_CHECKING following existing MarketAwareWinCalibrator pattern

### Features Used

**Relevance (15 features, D-23):**
if_p_win_final, if_p_win_race_rank, if_p_ability_win, rel_p_ability_win_rank (derived), if_norm_finish_avg, if_closing_index, if_weighted_recent_form, if_jockey_wr, if_trainer_wr, if_blood_surface_wr, if_class_level, if_surface, if_distance_bin, if_grade_code, if_n_horses

**Value (15 features, D-24):**
if_logit_gap, if_edge_win, if_ev_calibrated, if_odds_log, if_odds_band_id, if_odds_drop_60_10, if_odds_drop_30_10, if_overround, if_market_entropy, if_conformal_width, if_ev_uncertainty_ratio, if_p_win_race_rank, if_n_horses, if_odds_rank (derived), if_abs_logit_gap (derived)

## TDD Gate Compliance

- RED commit: f5be1bd -- 13 tests failing (ModuleNotFoundError)
- GREEN commit: 7954e10 -- 13 tests passing
- No separate REFACTOR commit needed (ruff/mypy fixes included in GREEN)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed positional index mismatch in WF alpha selection**
- **Found during:** Task 1 GREEN phase
- **Issue:** df_surf extracted via .loc[mask] preserves original DataFrame index, but X is a continuous numpy array. _walk_forward_race_splits returns indices from df_surf.index which don't align with X[positional_index].
- **Fix:** Added df_surf.reset_index(drop=True) before building feature matrices and passing to alpha selection.
- **Files modified:** src/models/race_level_ranker.py
- **Commit:** 7954e10

**2. [Rule 3 - Blocking] Test data missing if_surface column**
- **Found during:** Task 1 GREEN phase
- **Issue:** score() tests used "surface" column name but score() method accesses "if_surface" (IFF schema name).
- **Fix:** Changed test data to use "if_surface" column name matching the actual IFF schema.
- **Files modified:** tests/test_race_level_ranker.py
- **Commit:** 7954e10

**3. [Rule 1 - Bug] NameError in persistence test**
- **Found during:** Task 1 GREEN phase
- **Issue:** test_save_load_roundtrip referenced RaceLevelRanker class name that wasn't in scope at test method level.
- **Fix:** Added local import `from models.race_level_ranker import RaceLevelRanker as _RLR` inside the test method.
- **Files modified:** tests/test_race_level_ranker.py
- **Commit:** 7954e10

## Verification Results

- 13/13 race_level_ranker tests passed
- 26/26 domain tests passed (no regression)
- ruff check: all checks passed (src/models/race_level_ranker.py, src/domain/models.py)
- mypy: success, no issues found (src/models/race_level_ranker.py)

## Threat Flags

No new threat surface introduced beyond what was documented in the plan's threat_model. The RaceLevelRanker uses OOF-only training data, operates in shadow mode by default, and does not add network endpoints or auth paths.
