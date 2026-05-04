---
phase: 11-bet-selection-filters
reviewed: 2026-05-05T12:00:00Z
depth: standard
files_reviewed: 10
files_reviewed_list:
  - src/backtest/engine.py
  - src/backtest/race_predictor.py
  - src/backtest/report.py
  - src/betting/odds_band_filter.py
  - src/models/regime_detector.py
  - tests/test_backtest_engine.py
  - tests/test_backtest_report.py
  - tests/test_odds_band_filter.py
  - tests/test_race_predictor.py
  - tests/test_regime_detector.py
findings:
  critical: 0
  warning: 4
  info: 4
  total: 8
status: issues_found
---

# Phase 11: Code Review Report

**Reviewed:** 2026-05-05T12:00:00Z
**Depth:** standard
**Files Reviewed:** 10
**Status:** issues_found

## Summary

Reviewed all 10 source and test files from Phase 11 (bet-selection-filters). The implementation adds COLLAPSED regime race-level skip, EV lower bound filtering for win candidates, OddsBandFilter calibration from training bet history, and exclusion stats in BacktestResult and report/AI diagnostics output.

The core filtering logic is correct and well-structured. The main concern is a logic bug in `RegimeDetector.should_retrain()` that renders it dead code -- it can never return `True` due to the hysteresis counter being reset to zero whenever the current regime matches the raw regime. There are also minor quality issues including dead code branches, an odds classification edge case in OddsBandFilter, and band boundary inconsistency between the filter and the report.

## Warnings

### WR-01: RegimeDetector.should_retrain() is dead code -- can never return True

**File:** `src/models/regime_detector.py:235-240`
**Issue:** The `should_retrain()` method checks `self._regime_counter >= self.cfg.retrain_trigger` (default 100) while `self._current_regime == COLLAPSED`. However, the hysteresis logic in `detect()` (lines 154-156) resets `_regime_counter = 0` whenever `raw_regime == self._current_regime`. Once the current regime becomes COLLAPSED, every subsequent call to `detect()` that also predicts COLLAPSED resets the counter to 0. If a non-COLLAPSED regime is predicted, the counter also resets to 0 (new pending regime). Therefore `_regime_counter` is always 0 when `_current_regime == COLLAPSED`, and `should_retrain()` always returns False.

The docstring says "COLLAPSED state for 100 consecutive races triggers retraining" but the current counter semantics track transitions, not consecutive detections of the same state.
**Fix:** Change the counter semantics to track consecutive detections of COLLAPSED regardless of current state, or add a separate counter:
```python
def detect(self, recent_stats: pd.DataFrame) -> RegimeState:
    # ... existing hysteresis logic ...

    # Track consecutive COLLAPSED detections for retrain trigger
    if raw_regime == RegimeState.COLLAPSED:
        self._collapsed_consecutive += 1
    else:
        self._collapsed_consecutive = 0

    # ... rest of method ...

def should_retrain(self) -> bool:
    return self._collapsed_consecutive >= self.cfg.retrain_trigger
```

### WR-02: OddsBandFilter misclassifies odds=0 (or missing odds) into "30.0+" band

**File:** `src/betting/odds_band_filter.py:30-35,50-52`
**Issue:** `_get_band_name()` checks `lo <= odds < hi` for each band. When `odds=0.0` (the default from `bet.get("odds", 0)` at line 51), no band matches because `1.0 <= 0.0` is False. The fallback `return "30.0+"` at line 35 incorrectly assigns zero-odds bets to the highest odds band. This pollutes the "30.0+" band's ROI calculation during calibration, potentially causing the band to be excluded incorrectly.
**Fix:** Add a guard in `calibrate()` to skip bets with invalid odds:
```python
for bet in bet_history:
    odds = float(bet.get("odds", 0))
    if odds < 1.0:
        continue  # Skip bets with invalid/missing odds
    band = self._get_band_name(odds)
    # ... rest of loop
```

### WR-03: Dead code branch in engine.py win candidate selection

**File:** `src/backtest/engine.py:746-754`
**Issue:** The code uses `getattr(self._race_predictor, "get_win_candidates", None)` to check if the method exists. However, `get_win_candidates` is always defined on the `RacePredictor` class (line 408 of `race_predictor.py`), so `callable(get_win)` is always `True`. The `else` branch (lines 751-754, calling `get_place_candidates` as fallback) is unreachable dead code. While not a bug (the correct path always executes), the dead code is misleading and the fallback would produce wrong results if it were reached (using place candidates for win betting).
**Fix:** Remove the `else` branch and simplify:
```python
if self.betting_target == "win":
    candidate_df = self._race_predictor.get_win_candidates(result_df)
    n_ev_excluded += int(candidate_df.attrs.get("n_ev_excluded", 0))
else:
    candidate_df = self._race_predictor.get_place_candidates(
        result_df, regime_params=regime_params,
    )
```

### WR-04: Dead code methods _generate_bets and _build_race_features in BacktestEngine

**File:** `src/backtest/engine.py:1113-1198`
**Issue:** Both `_build_race_features()` and `_generate_bets()` on BacktestEngine have docstrings saying they are "kept for compatibility" and delegated to `RacePredictor`. Neither method is called from any production code or test. They add ~85 lines of dead code to an already large file (~1260 lines).
**Fix:** Remove both methods. If needed for backward compatibility, add a deprecation comment pointing to the RacePredictor equivalents. If external callers exist, they should migrate.

## Info

### IN-01: Odds band boundary mismatch between OddsBandFilter and report.py

**File:** `src/betting/odds_band_filter.py:30-34` vs `src/backtest/report.py:420-424`
**Issue:** `OddsBandFilter._get_band_name()` uses inclusive lower bound (`1.0 <= odds < 3.0`), while `report.py`'s `_compute_condition_stats` lambda uses exclusive upper bound only (`b.get("tanoddslow", 0) < 3.0` with no lower bound check). For odds < 1.0, the report lambda assigns to "1.0-3.0" while the filter assigns to "30.0+" (fallback). This means the band statistics in reports will not match the filter's band classification for edge cases (odds < 1.0).
**Fix:** Align the report lambda to use the same boundaries as OddsBandFilter:
```python
lambda b: (
    "1.0-3.0" if 1.0 <= b.get("tanoddslow", 0) < 3.0
    else "3.0-10.0" if b.get("tanoddslow", 0) < 10.0
    else "10.0-30.0" if b.get("tanoddslow", 0) < 30.0
    else "30.0+"
),
```

### IN-02: Missing test coverage for training_bet_history calibration path

**File:** `tests/test_backtest_engine.py`
**Issue:** `BacktestEngine.run()` accepts a `training_bet_history` parameter (D-05) that calibrates the `OddsBandFilter` before the test loop. No test exercises this path -- specifically, no test verifies that when `training_bet_history` is provided with `betting_target="win"`, the `OddsBandFilter` is calibrated and subsequently excludes candidates during the test period.
**Fix:** Add a test that provides `training_bet_history` to `engine.run()` and verifies `n_odds_band_excluded > 0` in the result.

### IN-03: RegimeDetector.detect() will AttributeError if called before train()

**File:** `src/models/regime_detector.py:141-143`
**Issue:** `detect()` accesses `self.model.best_iteration` and `self.model.predict()` without checking if `self.model` exists. If `detect()` is called before `train()`, this raises `AttributeError: 'RegimeDetector' object has no attribute 'model'`. In the current system, models are always pre-trained, so this is not a runtime issue, but it is a fragile API contract.
**Fix:** Add a guard at the start of `detect()`:
```python
if not hasattr(self, "model"):
    return RegimeState.CONSERVATIVE
```

### IN-04: subprocess.check_output in report.py has no timeout

**File:** `src/backtest/report.py:67-68,110-111`
**Issue:** Two calls to `subprocess.check_output(["git", "rev-parse", ...])` have no `timeout` parameter. If the git command hangs (e.g., requiring authentication), the entire report generation process will block indefinitely.
**Fix:** Add `timeout=5` to both calls:
```python
subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"],
    stderr=subprocess.DEVNULL,
    timeout=5,
)
```

---

_Reviewed: 2026-05-05T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
