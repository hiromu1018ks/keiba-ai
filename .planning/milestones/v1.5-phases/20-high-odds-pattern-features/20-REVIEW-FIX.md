---
phase: 20-high-odds-pattern-features
fixed_at: 2026-05-09T01:45:00Z
review_path: .planning/phases/20-high-odds-pattern-features/20-REVIEW.md
iteration: 1
findings_in_scope: 4
fixed: 4
skipped: 0
status: all_fixed
---

# Phase 20: Code Review Fix Report

**Fixed at:** 2026-05-09T01:45:00Z
**Source review:** `.planning/phases/20-high-odds-pattern-features/20-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 4
- Fixed: 4
- Skipped: 0

## Fixed Issues

### WR-01: Two test methods are no-ops (assert True / bare pass)

**Files modified:** `tests/test_high_odds_features.py`
**Commit:** df2273c
**Applied fix:** Removed `test_exp_count_accuracy` (assert True no-op) and `test_exp_count_three_matches` (pass no-op). Both are superseded by `test_dist_change_exp_count` which has proper assertions.

### WR-02: current_db variable defined in sibling if-block

**Files modified:** `src/features/horse_history_features.py`
**Commit:** ee10fa5
**Applied fix:** Extracted `current_db` computation to before both the `distance_change` if-block and the `env_adaptability` if-block. The distance_change block now only reads `current_db` without redefining it, eliminating the fragile cross-block variable dependency.

### WR-03: Divergent _is_nan vs pd.notna

**Files modified:** `src/features/high_odds_features.py`
**Commit:** cf9bfb8
**Applied fix:** Added `value is None` check to `_is_nan()` function. Previously `_is_nan` only checked for `float` NaN via `np.isnan`, missing `None` values that `pd.notna()` would catch. The function now has equivalent coverage to `pd.notna()` for the values it encounters, without adding a pandas import dependency to this module.

### WR-04: Overly broad exception handler

**Files modified:** `scripts/analyze_high_odds.py`
**Commit:** 1b59334
**Applied fix:** Replaced bare `except Exception as e` with two handlers: (1) specific `except (FileNotFoundError, ValueError, KeyError, RuntimeError)` for expected model/data errors logged as warnings, and (2) a fallback `except Exception as e` with `exc_info=True` for unexpected errors logged as errors with full traceback. This prevents programming bugs from being silently swallowed while still gracefully handling legitimate failures.

---

_Fixed: 2026-05-09T01:45:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
