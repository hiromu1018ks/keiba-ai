---
phase: 41-shadow-comparison-framework
plan: 01
subsystem: backtest
tags: [shadow-comparison, feature-flags, alignment, metrics, race-predictor]
dependency_graph:
  requires: [BacktestEngine, ModelLoader, RacePredictor, TrainedModelsV5]
  provides: [ShadowComparisonFramework, FoldDefinition, VariantConfig, ComparisonMetrics, enable_market_aware_calibrator, enable_race_level_ranker]
  affects: [src/backtest/shadow_comparison.py, src/backtest/race_predictor.py, tests/test_shadow_comparison.py]
tech_stack:
  added: []
  patterns: [N-way variant comparison, feature flag injection, post-hoc alignment, ECE computation]
key_files:
  created:
    - src/backtest/shadow_comparison.py
    - tests/test_shadow_comparison.py
  modified:
    - src/backtest/race_predictor.py
decisions:
  - Feature flags injected via TrainedModelsV5._shadow_flags dict, read by RacePredictor constructor (D-19)
  - MAWC baseline path computes p_win_final from p_win_corrected with race normalization (D-18)
  - Strict mode raises ValueError when flag=True but artifact missing (D-21)
  - CLV null with clv_available=false when <10% of bets have valid inputs (D-14)
  - ECE uses 10 equal-width bins over [0, 1] probability range
metrics:
  duration: 15m
  completed: "2026-05-28T09:38:02Z"
  tasks: 3
  tests: 33
  files: 3
  loc_added: 1563
---

# Phase 41 Plan 01: Shadow Comparison Framework Core Summary

ShadowComparisonFramework with N-way BacktestEngine comparison, post-hoc alignment, comprehensive metrics (Brier/logloss/ECE/ROI/HR/DD/CLV/selection agreement), and feature flag injection into RacePredictor for MAWC and ranker control.

## Completed Tasks

| Task | Name | Status |
|------|------|--------|
| 1 | Create dataclasses and ShadowComparisonFramework class | Done |
| 2 | Inject feature flags into RacePredictor for MAWC and ranker control | Done |
| 3 | Wire ShadowComparisonFramework to BacktestEngine with feature flag injection | Done |

## Key Artifacts

### src/backtest/shadow_comparison.py (new, ~620 lines)
- **FoldDefinition** (frozen dataclass): Fixed fold definitions matching WF validation (Fold 2024: train 2020-2023, Fold 2025: train 2021-2024)
- **VariantConfig** (frozen dataclass): variant_name/model_dir + enable_market_aware_calibrator/enable_race_level_ranker flags
- **ComparisonMetrics** (dataclass): Brier, logloss, ECE, ROI, HR, bet_count, avg_odds, max_DD, CLV, selection_agreement, avg_investment_score, actual_predicted_ratio
- **VariantResult** (dataclass): BacktestResult + flag_states
- **ShadowComparisonResult** (dataclass): fold, variants dict, race_diff/horse_diff DataFrames, metrics dict, alignment_succeeded
- **ShadowComparisonFramework**: run_fold(), run(), _align_race_level(), _align_horse_level(), compute_metrics(), compute_metrics_by_group()
- D-21 strict mode validation: _validate_artifacts() raises ValueError on flag/artifact mismatch

### src/backtest/race_predictor.py (modified)
- Added `enable_market_aware_calibrator: bool = True` and `enable_race_level_ranker: bool = True` constructor kwargs
- _shadow_flags propagation from TrainedModelsV5 to RacePredictor
- MAWC guard: `if self.enable_market_aware_calibrator and mawc is not None and mawc.is_trained`
- Ranker guard: `if self.enable_race_level_ranker and ranker is not None and ranker.is_trained`
- Baseline path: computes p_win_final from p_win_corrected with race normalization when flag=False

### tests/test_shadow_comparison.py (new, 921 lines)
- 33 tests covering: FoldDefinition, VariantConfig, ComparisonMetrics, ShadowComparisonResult
- Race-level alignment: two variants, odds preservation
- Horse-level alignment: probability/investment_score matching, selected flag
- Metrics: Brier, logloss, ECE, ROI/HR, actual/predicted ratio, CLV valid/invalid
- Selection agreement, odds band grouping
- N-way framework: 3-variant alignment
- Strict mode: D-21 flag/artifact mismatch ValueError
- RacePredictor flags: default True, disabled skips, _shadow_flags propagation, override behavior
- Integration: run_fold with mock ModelLoader/BacktestEngine, default folds, custom folds

## Decisions Made

1. **Feature flag injection via _shadow_flags dict** on TrainedModelsV5: ShadowComparisonFramework sets `models._shadow_flags = {...}` before constructing BacktestEngine. RacePredictor reads it in constructor. This avoids changing BacktestEngine's constructor signature.
2. **MAWC baseline path**: When `enable_market_aware_calibrator=False`, p_win_final is computed from p_win_corrected with group-level normalization (existing fallback behavior).
3. **Strict mode (D-21)**: Raises ValueError immediately when flag=True but no trained artifact exists. Prevents silent wrong-path execution.
4. **CLV availability threshold**: If fewer than 10% of bets have valid closing_odds/tanodds, reports clv=None with clv_available=False.
5. **Post-commit deletion check**: No files deleted in this commit.

## Deviations from Plan

None - plan executed exactly as written. All 3 tasks completed in a single commit due to tight coupling (Task 3 wires Tasks 1 and 2 together).

## Verification Results

- `python -m pytest tests/test_shadow_comparison.py -v`: 33/33 passed
- `python -m pytest tests/test_backtest_engine.py tests/test_backtest_engine_autocalibrate.py -v -q`: 76/77 passed (1 pre-existing failure unrelated to this plan)
- `python -m ruff check src/backtest/shadow_comparison.py src/backtest/race_predictor.py`: All checks passed
- Backward compatibility: RacePredictor defaults to enable_market_aware_calibrator=True, enable_race_level_ranker=True

## Self-Check: PASSED

- FOUND: src/backtest/shadow_comparison.py
- FOUND: src/backtest/race_predictor.py
- FOUND: tests/test_shadow_comparison.py
- FOUND: .planning/phases/41-shadow-comparison-framework/41-01-SUMMARY.md
- FOUND: commit fce95c2
