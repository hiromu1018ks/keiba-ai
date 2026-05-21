---
phase: 04-walk-forward-validation
reviewed: 2026-05-03T00:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - src/models/walk_forward_cv.py
  - scripts/run_wf_validation.py
  - tests/test_walk_forward_cv.py
findings:
  critical: 2
  warning: 4
  info: 2
  total: 8
status: issues_found
---

# Phase 4: Code Review Report

**Reviewed:** 2026-05-03
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Reviewed 3 files implementing walk-forward cross-validation infrastructure: the core `WalkForwardCV` class and Phase 4 validation functions in `walk_forward_cv.py`, the execution script `run_wf_validation.py`, and accompanying tests. Two critical correctness bugs were found: `_add_years_dt` produces off-by-one fold boundaries due to leap-year miscalculation, and `np.std` uses population standard deviation instead of sample standard deviation, understating reported ROI variability. Four warnings cover missing MLflow error handling, NaN stability verdict logic, duplicated fold definitions, and potential train data leakage through shared model state.

## Critical Issues

### CR-01: _add_years_dt produces off-by-one fold boundaries for leap-year spans

**File:** `src/models/walk_forward_cv.py:193`
**Issue:** `_add_years_dt` uses `int(365.25 * years)` as a day offset, which is inaccurate for any date range crossing a leap year boundary. The error compounds: for `years=3` starting from 2020-01-01 (a leap year), the function returns 2022-12-31 instead of the correct 2023-01-01 (off by 1 day). For `years=2` starting from 2015-01-01, it returns 2016-12-30 instead of 2016-12-31. This causes `generate_folds()` to produce fold boundaries that are 1 day short, meaning the last day of a training period is silently dropped or test/train periods have a 1-day gap/overlap.

The test at `tests/test_walk_forward_cv.py:296-300` validates the buggy behavior rather than the correct behavior, encoding the wrong expectation.

**Fix:**
```python
def _add_years_dt(dt: datetime, years: int) -> datetime:
    """datetime に年を加算 (calendar-aware)"""
    try:
        return dt.replace(year=dt.year + years)
    except ValueError:
        # Handle Feb 29 in non-leap result year
        return dt.replace(year=dt.year + years, day=28)
```

### CR-02: np.std uses population std (ddof=0) instead of sample std (ddof=1)

**File:** `src/models/walk_forward_cv.py:185`
**Issue:** `result.std_roi = float(np.std(rois))` computes the population standard deviation. Walk-forward CV typically has very few folds (2-5), and with `ddof=0` the reported standard deviation systematically understates the true variability. For 2 folds, `np.std([a, b])` with `ddof=0` is `sqrt(2)/2 ≈ 0.707x` the correct sample standard deviation (`ddof=1`). This directly impacts the reliability of overfitting detection, since the std is used as a gauge of cross-fold consistency.

**Fix:**
```python
result.std_roi = float(np.std(rois, ddof=1)) if len(rois) > 1 else 0.0
```

## Warnings

### WR-01: MLflow logging failure prevents final result from being saved

**File:** `scripts/run_wf_validation.py:271-295`
**Issue:** The MLflow logging block (step 9) is not wrapped in a try/except. If the MLflow server is unreachable or the logging call raises an exception, the script crashes before reaching step 10 (final result save at line 298-299). Although intermediate fold-by-fold results are saved (step 5), the final aggregated metrics (pool_roi, weighted_roi, spearman_rho, verdicts) would be lost.

**Fix:** Wrap the MLflow block in a try/except, logging the error but proceeding to save the final result:
```python
try:
    mlflow.set_experiment("wf_validation")
    with mlflow.start_run(...):
        ...
except Exception as e:
    logger.warning("MLflow logging failed: %s", e)

# 10. Final save always executes
result_dict = asdict(wf_result)
_save_intermediate_result(result_dict, output_path)
```

### WR-02: NaN spearman_rho treated as PASS verdict in judge_overfitting

**File:** `src/models/walk_forward_cv.py:337-342`
**Issue:** When `spearman_rho` is NaN (fewer than 2 folds, or fewer than 3 common features), the `stability_verdict` remains at its default value `"PASS"`. This means the stability check silently passes when there is insufficient data to compute it. A NaN result should be treated as `"WARNING"` to signal that stability could not be assessed, rather than implying it was confirmed.

**Fix:**
```python
# 基準3: 安定性
if np.isnan(result.spearman_rho):
    result.stability_verdict = "WARNING"
elif result.spearman_rho >= min_rho:
    result.stability_verdict = "PASS"
else:
    result.stability_verdict = "WARNING"
```

### WR-03: Hardcoded FOLDS definition duplicates WalkForwardCV.generate_folds logic

**File:** `scripts/run_wf_validation.py:45-58`
**Issue:** The script defines `FOLDS` as a hardcoded list rather than using `WalkForwardCV.generate_folds()`. This creates two sources of truth for fold definitions. If the fold logic in `WalkForwardCV` is updated (e.g., to fix CR-01), the script's hardcoded folds would not reflect the change. The script already imports `WalkForwardCV` indirectly but never calls `generate_folds`.

**Fix:** Generate folds programmatically or at minimum validate that the hardcoded FOLDS match `WalkForwardCV.generate_folds()` output at startup:
```python
cv = WalkForwardCV(train_years=4, test_years=1, step_years=1)
expected_folds = cv.generate_folds("2020-01-01", "2025-12-31")
# Validate FOLDS matches expected_folds
```

### WR-04: train-engine uses same models object; state mutation risk

**File:** `scripts/run_wf_validation.py:187-190`
**Issue:** The train-period backtest engine is created with the same `models` object returned by the pipeline. If `BacktestEngine.run()` mutates any state on the models (e.g., caching predictions, updating internal counters), the train backtest results could contaminate the test backtest results. The comment says "Per Pitfall 2" and uses a separate engine instance, but both engines share the same `models` object. A deep copy of models or explicit documentation that `BacktestEngine.run()` is side-effect-free on models would be needed.

**Fix:** Either document that `BacktestEngine.run()` is read-only with respect to models, or create a separate copy of models for the train backtest:
```python
import copy
train_models = copy.deepcopy(models)
train_engine = BacktestEngine(models=train_models, ...)
```

## Info

### IN-01: Top-level lightgbm import couples WalkForwardCV to LightGBM

**File:** `src/models/walk_forward_cv.py:200`
**Issue:** `import lightgbm as lgb` is at module level (after the class definitions but still loaded on import). Any code that only needs `WalkForwardCV` (lines 54-188) must also have LightGBM installed. The Phase 4 functions (`extract_feature_ranking`, etc.) require it, but the core `WalkForwardCV` class does not. Consider moving Phase 4 functions to a separate module to allow lightweight imports.

**Fix:** Move `FoldResult`, `WFValidationResult`, `extract_feature_ranking`, `compute_feature_stability`, and `judge_overfitting` to a separate file (e.g., `src/models/wf_validation.py`) to decouple from `WalkForwardCV`.

### IN-02: test_add_years validates buggy behavior

**File:** `tests/test_walk_forward_cv.py:296-300`
**Issue:** The test `test_add_years` asserts that `_add_years_dt(datetime(2020,1,1), 1)` returns December 31, 2020 (year=2020). Adding 1 year to Jan 1, 2020 should yield Jan 1, 2021. The test validates the buggy 365.25-day approximation rather than correct calendar-year addition. This test will need updating once CR-01 is fixed.

**Fix:** After fixing `_add_years_dt` per CR-01, update the test to assert the correct result:
```python
def test_add_years(self) -> None:
    from datetime import datetime
    from models.walk_forward_cv import _add_years_dt

    dt = datetime(2020, 1, 1)
    result = _add_years_dt(dt, 1)
    assert result.year == 2021
    assert result.month == 1
    assert result.day == 1
```

---

_Reviewed: 2026-05-03_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
