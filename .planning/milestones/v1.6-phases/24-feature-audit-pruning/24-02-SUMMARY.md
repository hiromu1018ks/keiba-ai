---
phase: 24-feature-audit-pruning
plan: 02
subsystem: features
tags: [pruning, oof-safety, roi-verification, rollback, tier-report]
dependency_graph:
  requires: [24-01 (tier classification)]
  provides: [prune_noise_features.py, generate_tier_report, rollback mechanism]
  affects: [win_feature_analysis, model FEATURE_COLS (editing)]
tech_stack:
  added: []
  patterns: [multi-stage pruning pipeline, per-model independent pruning, OOF safety gate, ROI baseline comparison]
key_files:
  created:
    - scripts/prune_noise_features.py
    - tests/test_prune_noise_features.py
  modified:
    - src/features/win_feature_analysis.py
decisions:
  - Binary models only get OOF logloss/AUC safety check; regression models skip (validate_noise_removal is binary-only)
  - Regression model safety deferred to full BT validation (RESEARCH Pitfall 6)
  - File editing uses line-based text approach (not AST) for FEATURE_COLS modification
  - Backup files (.backup) created before any FEATURE_COLS editing; rollback deletes them after restore
  - Path separator handling uses os.path.normpath for cross-platform comparison in tests
metrics:
  duration_minutes: 23
  completed: "2026-05-12"
  tasks_completed: 2
  tests_added: 8
  files_modified: 1
  files_created: 2
  total_tests_passing: 1419
---

# Phase 24 Plan 02: Tier 1 Pruning + OOF Safety + Rollback Summary

Integrated pruning script with per-model Tier 1 removal, OOF logloss/AUC safety gating for binary models, full BT ROI verification against v1.5 baseline (84.4%), and automatic rollback with cause analysis on degradation.

## Changes Made

### Task 1: prune_noise_features.py (commit a06b7d2)
- Created `scripts/prune_noise_features.py` -- integrated CLI for Tier 1 pruning pipeline
- CLI arguments: `--model-dir`, `--features-path`, `--apply`, `--full-bt`, `--rollback`, `--output`, `--safety-threshold`, `--baseline-roi`
- Processing flow: model loading -> importance computation -> Tier classification -> OOF safety check -> optional FEATURE_COLS editing -> optional full BT ROI verification -> optional rollback
- Model type classification: `BINARY_MODELS = {stage1, win_hit, place_hit}`, `REGRESSION_MODELS = {win_return, place_return, ev_correction, place_ev_correction, conformal_ev}`
- Binary models: `validate_noise_removal()` OOF logloss/AUC comparison with 0.5% degradation threshold
- Regression models: OOF safety skipped (safety_passed=True), full BT validates comprehensively
- Per-model independent pruning via `_MODEL_COL_MAP` (8 model-to-file mappings)
- File editing: line-based regex approach, handles `FEATURE_COLS: list[str] = [` format correctly
- Backup mechanism: `.backup` files created before editing, `rollback_files()` restores and cleans up
- ROI comparison: subprocess runs full BT, reads `backtest_result.json`, compares against baseline
- Rollback + cause analysis: restores backups, reads BT CSV, calls `generate_cause_analysis()`, saves report
- Output: `data/audit/pruning_validation.json`, `data/audit/roi_comparison.json`, `data/audit/cause_analysis.json`
- 8 tests in `TestPruneNoiseFeatures` (all mock-based, DB-free)

### Task 2: generate_tier_report() (commit d5fd5de)
- Added `generate_tier_report()` to `src/features/win_feature_analysis.py`
- Per-model Tier 1/2 lists with gain and perm values for each feature
- Cross-model Tier 1 frequency analysis (which features are Tier 1 across multiple models)
- Summary statistics: total counts, unique features, model count
- Recommendations: Tier 1 = "auto-remove", Tier 2 = "review manually"
- Callable from `analyze_feature_importance.py --tier-report` mode

## Test Results

- 1419 passed, 1 skipped, 0 failures
- 8 new tests added (TestPruneNoiseFeatures)
- ruff check: all modified files pass
- POST_RACE leakage tests: all 4 pass

## Key Decisions

1. **Binary-only OOF safety**: `validate_noise_removal()` uses `objective="binary"` and `metric="binary_logloss"`, making it inappropriate for regression models. Regression safety is deferred to full BT (RESEARCH Pitfall 6).
2. **Line-based file editing**: Python AST was not used for FEATURE_COLS editing; instead regex/line-based text processing handles `FEATURE_COLS: list[str] = [...]` format including type annotations.
3. **`=` sign detection for list boundaries**: The initial `] in stripped` check incorrectly matched `list[str]` type annotations. Fixed by checking `after_eq = stripped[eq_pos+1:]` to only detect list-ending `]` after the assignment operator.
4. **Path normalization in rollback**: `rollback_files()` uses `os.path.join(ROOT, rel_path)` which may produce different separators than test assertions. Tests use `os.path.normpath()` for comparison.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed `]` detection in _edit_feature_cols_in_file**
- **Found during:** Task 1 test execution
- **Issue:** `"] in stripped"` matched `list[str]` type annotation, causing `in_target_list` to immediately reset to False
- **Fix:** Changed detection to `stripped == "]"` or `stripped.startswith("]")` for list-end detection
- **Files modified:** scripts/prune_noise_features.py
- **Commit:** a06b7d2

**2. [Rule 3 - Blocking] Fixed patch target for validate_noise_removal in tests**
- **Found during:** Task 1 test execution
- **Issue:** `patch("prune_noise_features.validate_noise_removal")` failed because the module was dynamically loaded via importlib
- **Fix:** Changed to `patch.object(wfa_mod, "validate_noise_removal", ...)` patching the source module directly
- **Files modified:** tests/test_prune_noise_features.py
- **Commit:** a06b7d2

## Commits

| Commit | Description |
|--------|-------------|
| a06b7d2 | feat(24-02): Tier 1プルーニング統合スクリプト + OOF安全性確認 + ロールバック |
| d5fd5de | feat(24-02): generate_tier_report() でTier 1/2包括レポート生成 |

## Self-Check: PASSED

- scripts/prune_noise_features.py: FOUND
- tests/test_prune_noise_features.py: FOUND
- src/features/win_feature_analysis.py: FOUND
- a06b7d2: FOUND
- d5fd5de: FOUND
- 1419 tests passing
