---
phase: 13-risk-calibration-parameter-optimization
reviewed: 2026-05-05T00:00:00Z
depth: standard
files_reviewed: 13
files_reviewed_list:
  - src/betting/drawdown_controller.py
  - src/domain/types.py
  - src/domain/models.py
  - src/backtest/engine.py
  - tests/test_drawdown_controller.py
  - tests/test_domain.py
  - src/models/regime_detector.py
  - src/betting/meta_switcher.py
  - src/backtest/parameter_freeze_protocol.py
  - tests/test_parameter_freeze.py
  - src/tuning/strategy_optimizer.py
  - scripts/run_strategy_optimization.py
  - tests/test_strategy_optimizer.py
findings:
  critical: 1
  warning: 5
  info: 3
  total: 9
status: issues_found
---

# Phase 13: Code Review Report

**Reviewed:** 2026-05-05
**Depth:** standard
**Files Reviewed:** 13
**Status:** issues_found

## Summary

Reviewed 13 files across the risk-calibration and parameter-optimization phase. The codebase is generally well-structured with good test coverage at the unit level. However, one critical bug was found where `StrategyOptimizer` replaces a trained `RegimeDetector` with an untrained instance that will crash at runtime when `detect()` is called. Several warnings address type annotation errors, missing edge-case guards, unused fields, and incomplete exception handling.

## Critical Issues

### CR-01: StrategyOptimizer replaces trained RegimeDetector with untrained instance, causing AttributeError crash

**File:** `src/tuning/strategy_optimizer.py:148-149`
**Severity:** BLOCKER

**Issue:** In `_run_single_backtest()`, when `regime_overrides` is present, a brand-new `RegimeDetector(override_params=regime_overrides)` is created and assigned to `models.regime_detector`. This new instance has never had `train()` called on it, so `self.model` does not exist. When `BacktestEngine.run()` later calls `models.regime_detector.detect()` (after accumulating >= `min_samples` races in `engine.py:725`), the call reaches `self.model.best_iteration` at `regime_detector.py:149`, which raises `AttributeError: 'RegimeDetector' object has no attribute 'model'`.

The existing tests in `test_strategy_optimizer.py` do not catch this because `_run_single_backtest` is mocked in all optimization-level tests, and the integration test (`test_calls_model_loader`, `test_injects_regime_overrides`) patches both `ModelLoader` and `BacktestEngine`, preventing the actual execution path from being exercised.

**Fix:**
```python
# In _run_single_backtest(), line 143-149:
# Instead of replacing the RegimeDetector, update only the override_params
# on the existing trained detector:
regime_overrides = strategy_config.get("regime_overrides")
if regime_overrides:
    models.regime_detector._override_params = regime_overrides
```

## Warnings

### WR-01: callable used as type annotation instead of Callable in RegimeDetectorProtocol

**File:** `src/betting/meta_switcher.py:14`
**Severity:** WARNING

**Issue:** `should_retrain: callable` uses the built-in function `callable` instead of the proper type `Callable`. This will fail `mypy` type checking (the project requires `disallow_untyped_defs = true` per `CLAUDE.md`) and does not provide correct type information. The `callable` builtin is not a type and cannot be used in type annotations.

**Fix:**
```python
from typing import Protocol
from collections.abc import Callable
from domain.types import RegimeState

class RegimeDetectorProtocol(Protocol):
    current_regime: RegimeState
    should_retrain: Callable[[], bool]
```

### WR-02: ZeroDivisionError when peak_bankroll is zero

**File:** `src/betting/drawdown_controller.py:78,122`
**Severity:** WARNING

**Issue:** `update()` and `get_state()` both compute `dd = (self.peak_bankroll - bankroll) / self.peak_bankroll` without guarding against `peak_bankroll == 0`. The constructor accepts `peak_bankroll: float` without validation, so `DrawdownController(peak_bankroll=0)` is possible and will cause a `ZeroDivisionError` on the first call to `update()` or `get_state()`.

**Fix:**
```python
# In __init__, add validation:
if peak_bankroll <= 0:
    raise ValueError(f"peak_bankroll must be > 0, got {peak_bankroll}")

# Or defensively in the division:
dd = (self.peak_bankroll - bankroll) / self.peak_bankroll if self.peak_bankroll > 0 else 0.0
```

### WR-03: ParameterFreezeProtocol._serialize catches incomplete set of pickle exceptions

**File:** `src/backtest/parameter_freeze_protocol.py:101`
**Severity:** WARNING

**Issue:** The `except` clause only catches `(pickle.PicklingError, TypeError)` but `pickle.dumps()` can also raise `AttributeError` (e.g., when objects have `__slots__` without `__getstate__`) or `RuntimeError`. If any of these exceptions propagate, `_serialize` crashes instead of falling back to `repr` hashing. The freeze protocol would then fail entirely rather than degrading gracefully.

**Fix:**
```python
except Exception:
    # pickle can raise AttributeError, RuntimeError, etc. for non-picklable objects
    return hashlib.sha256(repr(obj).encode()).digest()
```

### WR-04: DrawdownController.rolling_window is defined but never used

**File:** `src/betting/drawdown_controller.py:21`
**Severity:** WARNING

**Issue:** `DDConfig.rolling_window` is declared and validated in `__post_init__` but is never referenced anywhere in `DrawdownController`. The `update()` method computes DD as a simple `(peak - current) / peak` ratio without any rolling window logic. This dead field suggests incomplete implementation -- the rolling window was likely intended for a more sophisticated DD calculation (e.g., peak over last N races rather than all-time peak).

**Fix:** Either implement rolling-window DD tracking or remove the field to avoid confusion. If keeping it, document that it is reserved for future use.

### WR-05: BacktestResult.monthly_returns is always empty -- never populated

**File:** `src/backtest/engine.py:627,1104`
**Severity:** WARNING

**Issue:** `monthly_returns` is initialized as `{}` on line 627 and returned as-is in the `BacktestResult` on line 1104, but no code ever populates it. All per-race data goes into `bet_history` with `race_date` fields. This dead field misleads consumers who might expect populated monthly aggregation.

**Fix:** Either populate `monthly_returns` from `bet_history` before returning, or remove the field and add it as a computed property on `BacktestResult`.

## Info

### IN-01: Unused import of IsotonicRegression at runtime in domain/models.py

**File:** `src/domain/models.py:8`
**Severity:** INFO

**Issue:** `from sklearn.isotonic import IsotonicRegression` is imported at the module level (not inside `TYPE_CHECKING`), making `sklearn` a hard runtime dependency for any module that imports from `domain.models`. Since the type hint `isotonic_calibrator: IsotonicRegression | None` is already under `from __future__ import annotations`, the import could be moved into the `TYPE_CHECKING` block to reduce the dependency footprint of the domain layer.

**Fix:** Move the import inside the `if TYPE_CHECKING:` block.

### IN-02: StrategyOptimizer reloads models from disk on every fold of every trial

**File:** `src/tuning/strategy_optimizer.py:136-137`
**Severity:** INFO

**Issue:** `ModelLoader().load_from_dir()` is called inside `_run_single_backtest()`, meaning models are loaded from disk for every fold of every Optuna trial (up to `n_trials * n_folds` times, e.g., 200 loads for 100 trials). The loaded models never change across trials. This should load once in `optimize()` and reuse the models object.

**Fix:** Load models once in `optimize()` (or lazily in `__init__`) and pass them to `_run_single_backtest`.

### IN-03: DDConfig.max_adjustment_per_n_bets window-reset timing is off-by-one

**File:** `src/betting/drawdown_controller.py:134-137`
**Severity:** INFO

**Issue:** In `get_multiplier()`, `_bets_in_window` is incremented before the reset check. This means when `_bets_in_window` reaches `max_adjustment_per_n_bets`, the current call's result is already computed with the old window, and the reset happens for the *next* call. The `_multiplier_at_window_start` is then set to the *current clamped* `_current_multiplier`, not the raw target. This causes the rate-limiting window to be one bet behind, though the practical impact is minimal since the clamping logic converges quickly.

**Fix:** This is a minor asymmetry. Consider resetting before incrementing, or documenting the intentional design.

---

_Reviewed: 2026-05-05_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
