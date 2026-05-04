---
phase: 10-pipeline-performance
fixed_at: 2026-05-04T13:00:00Z
review_path: .planning/phases/10-pipeline-performance/10-REVIEW.md
iteration: 1
findings_in_scope: 5
fixed: 4
skipped: 1
status: partial
---

# Phase 10: Code Review Fix Report

**Fixed at:** 2026-05-04T13:00:00Z
**Source review:** `.planning/phases/10-pipeline-performance/10-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 5 (1 Critical, 4 Warning)
- Fixed: 4
- Skipped: 1

## Fixed Issues

### CR-01: Duplicate argparse in run_wf_validation.py makes --betting-target unusable

**Files modified:** `scripts/run_wf_validation.py`
**Commit:** `9a91be0`
**Applied fix:** Consolidated two separate `ArgumentParser` instances into a single parser in `main()` that handles both `--profile` and `--betting-target`. `_run_validation()` now accepts `betting_target` as a parameter instead of re-parsing `sys.argv`. This fixes the runtime error where `--betting-target` was unreachable because `main()`'s parser consumed `sys.argv` first and raised `SystemExit(2)` on unrecognized arguments.

### WR-01: is_cache_valid docstring claims hybrid timestamp+content-hash but only does timestamp check

**Files modified:** `src/features/feature_engine.py`
**Commit:** `354fd06`
**Applied fix:** Updated the docstring of `is_cache_valid()` to accurately describe timestamp-based cache invalidation. The previous docstring claimed "hybrid invalidation: timestamp comparison (fast) -> content hash (strict)" but the function body only performs mtime comparison with no content hash fallback.

### WR-02: Feature cache read failure is silently swallowed, returning potentially empty DataFrame

**Files modified:** `src/features/feature_engine.py`
**Commit:** `b4e0402`
**Applied fix:** Added an explicit warning log when `is_cache_valid()` returns True but the cached DataFrame is empty, making the fall-through-to-recomputation path visible. The control flow is now clearer: cache hit + empty data logs a warning before recomputing.

### WR-04: Redundant JRA filter applied twice in BacktestEngine.run()

**Files modified:** `src/backtest/engine.py`
**Commit:** `7891aed`
**Applied fix:** Converted the second JRA filter on `feat_df` (lines 473-483) from a silent re-filter into a safety assertion that warns if NAR entries leaked through the feature pipeline. The original filter at lines 387-392 on raw `race_df`/`entry_df` remains the primary filter. The second check now only acts (and warns) if entries somehow leaked in, indicating a pipeline bug.

## Skipped Issues

### WR-03: Variable name `race_df` shadows the outer-scope variable in BacktestEngine.run()

**File:** `src/backtest/engine.py:586,589`
**Reason:** The rename from `race_df` to `race_df_all` would require changing 10+ references across 120+ lines of the `run()` method. Other methods in the same class also use `race_df` as a parameter name, so renaming only the outer scope would create inconsistency. The loop already uses the distinct name `race_df_single`, making the shadowing risk low. This is a code quality suggestion that should be addressed in a dedicated refactoring pass rather than a minimal fix.
**Original issue:** In the `run()` method, the outer scope variable `race_df` (all races) could be confused with `race_df_single` (per-race slice) inside the per-race loop. Renaming the outer variable to `race_df_all` would improve clarity.

## Verification

- All 1162 tests pass (`python -m pytest tests/ -x -q`)
- No regressions introduced by fixes
- Syntax checks passed for all modified files

---

_Fixed: 2026-05-04T13:00:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
