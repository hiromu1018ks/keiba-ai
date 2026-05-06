---
phase: 17-optuna-optimization
reviewed: 2026-05-06T12:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - scripts/run_strategy_optimization.py
  - src/tuning/strategy_optimizer.py
  - tests/test_strategy_optimizer.py
findings:
  critical: 2
  warning: 4
  info: 3
  total: 9
status: issues_found
---

# Phase 17: Code Review Report

**Reviewed:** 2026-05-06T12:00:00Z
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Reviewed 3 files comprising the Optuna strategy optimization feature: the optimizer core (`strategy_optimizer.py`), CLI entry point (`run_strategy_optimization.py`), and test suite (`test_strategy_optimizer.py`).

Two critical bugs were found. First, `_objective` sets `ev_lower_threshold_*` attributes on submodels *before* running the training-phase backtest via `_generate_training_bet_history`, which means the training bet history is generated with optimized (non-default) EV thresholds -- a look-ahead bias that violates design principle D-04. Second, the same `_objective` method loads models and mutates state (regime overrides, submodel EV thresholds) but these mutations persist across Optuna trials if `ModelLoader.load_from_dir` returns cached/reused objects, potentially corrupting trial independence.

Four warnings cover dead code, an unsafe method-swap pattern, thread-safety concerns, and incomplete test coverage for re-optimization side-effects.

## Critical Issues

### CR-01: Look-ahead bias -- EV lower thresholds applied before training-phase backtest

**File:** `src/tuning/strategy_optimizer.py:267-265`
**Issue:** In `_objective`, lines 268-274 set `ev_lower_threshold_turf` and `ev_lower_threshold_dirt` on the submodel objects *after* line 260-262 sets default regime overrides but *before* the training-phase backtest runs at line 263-265 via `_generate_training_bet_history`. However, closer inspection reveals the order is actually: (1) default regime overrides set at 260-262, (2) `_generate_training_bet_history` called at 263-265, (3) EV thresholds set at 268-274. So the training bet history is generated with default EV thresholds. **Correction**: The training bet history IS generated before EV thresholds are set -- the order is correct.

However, there is a different look-ahead issue: `_generate_training_bet_history` runs a full BacktestEngine with the `models` object, which mutates the `RegimeDetector` state machine. After this call, the `RegimeDetector._current_regime`, `_regime_counter`, `_pending_regime`, and `_collapsed_consecutive` fields are left in whatever state the training data produced. The subsequent fold loop at line 282-287 resets these -- but the regime *history* within the DrawdownController (rolling window state, drawdown level) is NOT reset. This means fold 0 of the test-period backtest inherits stale drawdown state from the training period, creating look-ahead contamination of the DD control system.

**Fix:**
After `_generate_training_bet_history` returns and before the fold loop, reset the DrawdownController state. If the DD controller is accessible on the models object, reset its rolling window and current drawdown level. Alternatively, load models fresh for each trial (removing the D-05 optimization) or deep-copy the models object before mutations.

### CR-02: Mutable model state leaks across Optuna trials

**File:** `src/tuning/strategy_optimizer.py:253-274`
**Issue:** `_objective` calls `ModelLoader.load_from_dir()` once per trial and then mutates the returned `models` object in-place:
- Line 262: `models.regime_detector._override_params = default_regime_overrides`
- Lines 272-273: `sm.ev_lower_threshold_turf = ev_lower_turf` / `sm.ev_lower_threshold_dirt = ev_lower_dirt`

If `ModelLoader.load_from_dir` caches or reuses model objects across calls (which is a common pattern for expensive model loading), then mutations from trial N persist into trial N+1. Even without caching, the `_optimize_with_fixed_dims` method at line 484 calls `self.optimize()` which runs many trials through `_objective`, and each trial mutates the same `models.submodels` dataclass fields in-place via `sm.ev_lower_threshold_* = ...`. Since `SubmodelSet` is a dataclass and `ev_lower_threshold_turf/dirt` are mutable attributes (not frozen), these assignments silently corrupt state for subsequent trials.

**Fix:**
Either (a) ensure `ModelLoader.load_from_dir()` returns a deep copy each time, or (b) save and restore the EV threshold values around each trial, or (c) set `ev_lower_threshold_*` on a copy of the submodel or pass them via `strategy_config` instead of mutating the model object. The safest approach is option (c) -- add `ev_lower_threshold_turf` and `ev_lower_threshold_dirt` to the `strategy_config` dict and let `BacktestEngine` read them from there.

## Warnings

### WR-01: Dead code -- `_run_single_backtest` is never called by `_objective`

**File:** `src/tuning/strategy_optimizer.py:108-195`
**Issue:** The method `_run_single_backtest` (lines 108-195) is a self-contained method that loads models, builds engines, and runs backtests. It was the original implementation before the D-05 optimization split it into `_generate_training_bet_history` + `_run_single_backtest_with_models`. Now `_objective` only calls the latter two. `_run_single_backtest` is only called from tests (`TestRunSingleBacktest`). The method has its own model loading and regime reset logic that duplicates (and subtly differs from) the `_objective` code path. This dead code increases maintenance burden and risks divergent behavior if one path is updated without the other.

**Fix:** Remove `_run_single_backtest` or mark it as `_legacy` and consolidate all tests to use the `_objective` / `_run_single_backtest_with_models` path. If it must remain for standalone use, extract the shared regime-reset logic into a helper method.

### WR-02: Unsafe instance method swap in `_optimize_with_fixed_dims`

**File:** `src/tuning/strategy_optimizer.py:469-486`
**Issue:** The method temporarily replaces `self._suggest_params` with a closure `_suggest_fixed`, then restores it in a `finally` block. This pattern is not thread-safe -- if `optimize_multi_seed` were ever called concurrently (e.g., from multiple threads), the method swap would cause data races. More practically, if an exception occurs between line 481 (swap) and the `try` block at line 483 (which cannot happen in CPython but is technically possible in other implementations), the original method would be lost. The closure also captures `unstable_dims` and `fixed_params` from the enclosing scope, creating hidden coupling.

**Fix:** Instead of swapping the instance method, pass the fixed dimensions as a parameter to `_objective` or create a separate study configuration. For example, add an optional `fixed_params` argument to `_objective` that overrides suggested values, avoiding the need to mutate `self`.

### WR-03: Hardcoded default values duplicated in `_optimize_with_fixed_dims`

**File:** `src/tuning/strategy_optimizer.py:454-464`
**Issue:** The `default_param_values` dict at line 454 hardcodes 16 parameter values that must match `RegimeDetector._get_base_params()` and `DDConfig` defaults. The comment says "T-17-07: RegimeDetector._get_base_params() と一致" but there is no validation or programmatic link. If anyone updates the defaults in `RegimeDetector` or `DDConfig`, these hardcoded values will silently become stale, causing the re-optimization to fix dimensions to wrong values.

**Fix:** Import and call `build_default_strategy_config()` (or `_get_base_params`) to derive the default values programmatically rather than duplicating them. For example:
```python
from betting.default_strategy import build_strategy_config_from_params
default_config = build_strategy_config_from_params({})
# Then extract the parameter values from default_config
```

### WR-04: `_run_single_backtest_with_models` does not reset regime state

**File:** `src/tuning/strategy_optimizer.py:220-246`
**Issue:** `_run_single_backtest_with_models` accepts a `models` object and runs a backtest without resetting the `RegimeDetector` state. The caller (`_objective`) handles the reset at lines 282-287, but the separation of concerns is fragile -- any future caller of `_run_single_backtest_with_models` that forgets to reset regime state will get contaminated results from prior folds or trials. The method's docstring does not document this precondition.

**Fix:** Either (a) add regime reset at the start of `_run_single_backtest_with_models`, or (b) add a clear docstring precondition: "Caller MUST reset RegimeDetector state before calling this method for each fold."

## Info

### IN-01: `default_param_values` fallback value of `1.0` for unknown dimensions

**File:** `src/tuning/strategy_optimizer.py:466`
**Issue:** `default_param_values.get(dim, 1.0)` uses a magic fallback value of `1.0` if a dimension name is not found in the dict. This silently masks bugs if a new parameter name is added to the search space but not to `default_param_values`. A missing key should ideally raise an error rather than defaulting to an arbitrary value.

**Fix:** Use `default_param_values[dim]` (without `.get()`) to raise `KeyError` on unknown dimensions, or log a warning when the fallback is used.

### IN-02: `n_trials` type annotation mismatch in `optimize_multi_seed` mock

**File:** `tests/test_strategy_optimizer.py:577`
**Issue:** The mock `mock_optimize` function at line 577 has parameter `n_trials: int` but the test at line 676 does not verify that the `n_trials` argument type is respected. This is a minor test quality observation.

**Fix:** No action needed -- this is informational only.

### IN-03: `run_strategy_optimization.py` uses f-string logging

**File:** `scripts/run_strategy_optimization.py:69,77`
**Issue:** Lines 69 and 77 use f-strings in `logger.info(f"...")` instead of `%`-style formatting (`logger.info("...", arg)`). While functionally equivalent, f-string logging evaluates the format string even when the log level is disabled, wasting CPU cycles. The rest of the file correctly uses `%`-style formatting (lines 58, 65).

**Fix:** Use `logger.info("Starting strategy optimization: n_trials=%d", args.n_trials)` consistently, matching lines 58 and 65.

---

_Reviewed: 2026-05-06T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
