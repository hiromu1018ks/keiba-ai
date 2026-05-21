---
phase: 16-odds-band-rebuild
reviewed: 2026-05-06T19:50:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - src/betting/default_strategy.py
  - tests/test_default_strategy.py
  - src/tuning/strategy_optimizer.py
  - tests/test_strategy_optimizer.py
  - tests/test_backtest_engine_autocalibrate.py
  - src/backtest/engine.py
  - scripts/run_backtest.py
findings:
  critical: 1
  warning: 4
  info: 3
  total: 8
status: issues_found
---

# Phase 16: Code Review Report

**Reviewed:** 2026-05-06T19:50:00Z
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

Reviewed 7 files across the odds band rebuild feature: the shared default strategy config utility, the Optuna strategy optimizer, the backtest engine, the run_backtest script, and their respective tests. The codebase shows solid test coverage and good separation of concerns. However, one significant bug was found: the `StrategyOptimizer._run_single_backtest` method leaks mutable RegimeDetector state from the training-phase backtest into the test-phase backtest, which can bias Optuna's parameter search. Several quality issues were also identified including dead code, duplicated logic, and tests that do not exercise the actual code paths they claim to validate.

## Critical Issues

### CR-01: RegimeDetector state leaks from training to test backtest in StrategyOptimizer

**File:** `src/tuning/strategy_optimizer.py:155-201`
**Issue:** `_run_single_backtest` executes a training-phase backtest (line 175) using the same `models` object that is later used for the test-phase backtest (line 201). During `BacktestEngine.run()`, the `RegimeDetector` accumulates mutable state: `_current_regime`, `_regime_counter`, `_pending_regime`, and `_collapsed_consecutive` (see `regime_detector.py:72-76`). After the training backtest completes, these fields reflect training-period regime transitions. Line 188 only resets `_override_params` but does not reset the detector's internal hysteresis state. Consequently, the test-phase backtest begins with a contaminated regime detector that may start in AGGRESSIVE or COLLAPSED state rather than the default CONSERVATIVE, and with non-zero `_regime_counter`/`_collapsed_consecutive` values that affect transition behavior. This biases the Optuna parameter search because the same training contamination is applied to every trial's test evaluation, distorting ROI measurements.

**Fix:**
```python
# After line 188 in strategy_optimizer.py, reset regime detector internal state:
models.regime_detector._override_params = regime_overrides
# Reset mutable state to prevent training-to-test leakage
models.regime_detector._current_regime = RegimeState.CONSERVATIVE
models.regime_detector._regime_counter = 0
models.regime_detector._pending_regime = None
models.regime_detector._collapsed_consecutive = 0
```
Alternatively, reload the models after the training backtest, or construct a fresh `RegimeDetector` instance for the test-phase engine.

## Warnings

### WR-01: Dead code in _settle_bet for WIN fallback

**File:** `src/backtest/engine.py:1251`
**Issue:** The `elif bet.bet_type == BetType.WIN:` branch at line 1251 is unreachable dead code. The WIN case is fully handled earlier at lines 1219-1234, which always returns (every path through that block ends with `return`). The fallback block at lines 1248-1253 only handles PLACE bets that fall through from the `payout_map` lookup, but the WIN branch will never be reached. This dead code suggests an incomplete refactoring and may confuse future maintainers who expect WIN fallback logic here.

**Fix:** Remove lines 1251-1253 (the `elif bet.bet_type == BetType.WIN:` block).

### WR-02: Duplicated strategy config builder in run_backtest.py

**File:** `scripts/run_backtest.py:198-236`
**Issue:** `_build_strategy_config_from_manifest` duplicates the logic from `StrategyOptimizer._build_strategy_config`. Both convert Optuna-style flat params into a `BacktestEngine strategy_config` dict with the same structure, the same `dd_threshold_2 > dd_threshold_1` guard, and the same default values. Having two independent implementations creates a maintenance risk: if one is updated (e.g., new parameter added, default changed), the other can be overlooked. The comment "StrategyOptimizer._build_strategy_config と同じロジック" acknowledges this but does not solve it.

**Fix:** Extract a shared utility function (e.g., in `betting/default_strategy.py`) that both `StrategyOptimizer._build_strategy_config` and `run_backtest._build_strategy_config_from_manifest` call. The existing `default_strategy.py` module is the natural home since it already houses shared config logic.

### WR-03: Inconsistent default min_stay_races between manifest builder and DDConfig

**File:** `scripts/run_backtest.py:218`
**Issue:** `_build_strategy_config_from_manifest` uses `min_stay_races` default of `15` (line 218: `params.get("min_stay_races", 15)`), while `DDConfig.__post_init__` and `build_default_strategy_config` both use `10` as the default. Similarly, the `DDConfig` dataclass default is `10`. If a manifest is missing the `min_stay_races` key, the script will use `15` instead of the project-standard `10`.

**Fix:** Change line 218 to use the same default as `DDConfig`:
```python
min_stay_races=params.get("min_stay_races", 10),
```

### WR-04: E2E tests manually re-implement logic instead of calling the actual code

**File:** `tests/test_backtest_engine_autocalibrate.py:141-193`
**Issue:** The `TestAutoCalibrateE2E` class (lines 107-193) claims to test E2E flows (e.g., "run()がtraining_bet_history=Noneの場合に_generate_training_bet_history()を呼び出し"), but instead of calling `engine.run()` and verifying its behavior, each test manually re-implements the if/else logic from `engine.py:705-712`:
```python
training_bet_history = None
if engine._odds_band_filter is not None:
    if training_bet_history is None:
        training_bet_history = engine._generate_training_bet_history()
    if training_bet_history:
        engine._odds_band_filter.calibrate(training_bet_history)
```
These tests verify that the manually-copied logic behaves as expected, not that `BacktestEngine.run()` actually follows that logic. If the logic in `engine.py` changes (e.g., the condition order is modified, or a new branch is added), these tests will continue to pass without catching the regression.

**Fix:** Restructure the E2E tests to call `engine.run()` with appropriate mocks for data loading and prediction, then assert on the observable side effects (e.g., `calibrate` was called, `_generate_training_bet_history` was called) via `patch.object`.

## Info

### IN-01: Broad Exception catch in _run_single_backtest swallows training errors silently

**File:** `src/tuning/strategy_optimizer.py:181-183`
**Issue:** The `except Exception` block at line 181 catches all exceptions from the training backtest and logs a warning, then proceeds with `training_bet_history = None`. This means if the training backtest fails for a systemic reason (e.g., corrupt model files, missing data), the optimizer will silently skip calibration for every trial rather than failing fast. The `-1.0` penalty from `_objective` only triggers for low bet counts, not for missing calibration.

**Fix:** Consider narrowing to specific expected exceptions or adding a counter that aborts optimization after N consecutive training failures.

### IN-02: build_wide_payout_map 3-char kumi heuristic can misparse horse number pairs

**File:** `src/backtest/engine.py:244-262`
**Issue:** The length-3 heuristic for `kumi` parsing assumes that if the first two digits form a number <= 18, it should be parsed as (first-two, last-one). For example, "910" would be parsed as (9, 10) correctly, but "101" would be parsed as (10, 1) even though it could mean (1, 01) in zero-padded form. The heuristic is documented inline, but ambiguous inputs may produce incorrect pair mappings that silently corrupt wide payout calculations.

**Fix:** No immediate code change needed since the heuristic matches the data format documented in JRA-VAN specs, but adding a runtime validation step (e.g., asserting parsed horse numbers are within 1-18 range and exist in the entry data) would catch edge cases.

### IN-03: _collect_training_bet_history uses caller's betting_target but internal engine uses "place"

**File:** `scripts/run_backtest.py:275-283` and `src/backtest/engine.py:434-441`
**Issue:** `run_backtest.py:_collect_training_bet_history` creates a `BacktestEngine` with the caller's `betting_target` (line 281), which may be "win". This engine's `run()` then calls `_generate_training_bet_history()`, which internally creates another engine with `betting_target="place"`. The outer engine with `betting_target="win"` also has an `_odds_band_filter` that gets auto-calibrated. The net effect is that the outer `_collect_training_bet_history` engine produces a training bet history from place bets but calibrates a win-odds band filter. The function then returns only the `bet_history` from the outer run, not the inner one. The semantics are confusing, though in practice the outer engine's run also returns its own bet_history (from the outer betting_target), so the calibration data matches the outer target.

**Fix:** Add a clarifying comment explaining the two-layer engine construction, or refactor to make the flow more explicit.

---

_Reviewed: 2026-05-06T19:50:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
