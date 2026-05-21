---
phase: 13-risk-calibration-parameter-optimization
fixed_at: 2026-05-05T12:00:00Z
review_path: .planning/phases/13-risk-calibration-parameter-optimization/13-REVIEW.md
iteration: 1
findings_in_scope: 6
fixed: 6
skipped: 0
status: all_fixed
---

# Phase 13: Code Review Fix Report

**Fixed at:** 2026-05-05
**Source review:** .planning/phases/13-risk-calibration-parameter-optimization/13-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 6
- Fixed: 6
- Skipped: 0

## Fixed Issues

### CR-01: StrategyOptimizer replaces trained RegimeDetector with untrained instance, causing AttributeError crash

**Files modified:** `src/tuning/strategy_optimizer.py`, `tests/test_strategy_optimizer.py`
**Commit:** 2c63899
**Applied fix:** Replaced `RegimeDetector(override_params=...)` constructor call with `models.regime_detector._override_params = regime_overrides` to update only the override params on the existing trained instance, preventing AttributeError when `detect()` accesses `self.model.best_iteration`.

### WR-01: callable used as type annotation instead of Callable in RegimeDetectorProtocol

**Files modified:** `src/betting/meta_switcher.py`
**Commit:** e11789f
**Applied fix:** Changed `should_retrain: callable` to `should_retrain: Callable[[], bool]` with proper `from collections.abc import Callable` import.

### WR-02: ZeroDivisionError when peak_bankroll is zero

**Files modified:** `src/betting/drawdown_controller.py`
**Commit:** b466359
**Applied fix:** Added `ValueError` validation in `DrawdownController.__init__()` rejecting `peak_bankroll <= 0`, preventing ZeroDivisionError in `update()` and `get_state()`.

### WR-03: ParameterFreezeProtocol._serialize catches incomplete set of pickle exceptions

**Files modified:** `src/backtest/parameter_freeze_protocol.py`
**Commit:** e94f8b9
**Applied fix:** Changed `except (pickle.PicklingError, TypeError)` to `except Exception` so that `AttributeError`, `RuntimeError`, and other pickle-related exceptions also fall back to `repr` hashing.

### WR-04: DrawdownController.rolling_window is defined but never used

**Files modified:** `src/betting/drawdown_controller.py`
**Commit:** e38bc11
**Applied fix:** Added documentation comment indicating `rolling_window` is reserved for future rolling-window DD tracking (currently unused). Kept the field since StrategyOptimizer passes it to DDConfig.

### WR-05: BacktestResult.monthly_returns is always empty -- never populated

**Files modified:** `src/backtest/engine.py`, `tests/test_backtest_engine.py`
**Commit:** 65b2ae5
**Applied fix:** Converted `monthly_returns` from a never-populated dataclass field to a computed `@property` that aggregates monthly ROI from `bet_history`. Removed the unused local variable and constructor argument. Updated test to remove the now-invalid constructor argument.

## Skipped Issues

None -- all findings were fixed.

---

_Fixed: 2026-05-05_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
