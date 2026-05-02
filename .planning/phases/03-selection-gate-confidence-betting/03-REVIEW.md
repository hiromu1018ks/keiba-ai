---
phase: 03-selection-gate-confidence-betting
reviewed: 2026-05-03T00:30:00Z
depth: standard
files_reviewed: 13
files_reviewed_list:
  - src/models/win_selection_gate.py
  - tests/test_win_selection_gate.py
  - src/domain/models.py
  - src/pipelines/training_pipeline.py
  - src/db/model_loader.py
  - src/backtest/race_predictor.py
  - src/models/robust_confidence_estimator.py
  - src/models/regime_detector.py
  - src/betting/gate_keeper.py
  - src/betting/meta_switcher.py
  - tests/test_race_predictor.py
  - tests/test_gate_keeper.py
  - tests/test_meta_switcher.py
findings:
  critical: 1
  warning: 5
  info: 5
  total: 11
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-05-03T00:30:00Z
**Depth:** standard
**Files Reviewed:** 13
**Status:** issues_found

## Summary

Phase 3 adds WinSelectionGateModel (mechanical clone of PlaceSelectionGateModel with win-specific column names), edge threshold adjustments for JRA takeout, and supporting integration in RacePredictor, ModelLoader, TrainingPipeline, and MetaSwitcher. The implementation is broadly sound but contains one critical data-loss bug in confidence estimator serialization and several warning-level inconsistencies between parallel parameter sources.

## Critical Issues

### CR-01: Race-condition-dependent CP quantile not serialized -- silently lost at inference

**File:** `src/pipelines/training_pipeline.py:1122-1130`, `src/db/model_loader.py:608-617`
**Issue:** `RobustConfidenceEstimator._win_cp_quantile_by_condition` (dict) is populated during `calibrate()` when `surface` and `distance_bin` columns are present. However, neither `_log_to_mlflow` nor `_save_models_local` writes this dict to `confidence_params.json` -- they only write 6 scalar fields. Similarly, `ModelLoader._load_from_local` only restores those 6 scalars and never restores `_win_cp_quantile_by_condition`.

At inference, `_win_cp_quantile_by_condition` is always `{}` (default), so the per-condition quantile fallback path in `predict_lower_bound` (lines 121-131 of `robust_confidence_estimator.py`) is never used. The race-condition-dependent confidence calibration (SELC-02) is therefore completely dead in production.

**Fix:**

In `training_pipeline.py`, add the dict to `conf_params`:

```python
# In _log_to_mlflow and _save_models_local, add:
conf_params = {
    "alpha": conf.alpha,
    "rolling_window": conf.rolling_window,
    "win_cp_quantile": conf._win_cp_quantile,
    "place_cp_quantile": conf._place_cp_quantile,
    "win_rolling_quantile": conf._win_rolling_quantile,
    "place_rolling_quantile": conf._place_rolling_quantile,
    "win_cp_quantile_by_condition": conf._win_cp_quantile_by_condition,  # ADD
}
```

In `model_loader.py`, restore it after loading:

```python
# After loading conf_data:
if "win_cp_quantile_by_condition" in conf_data:
    confidence._win_cp_quantile_by_condition = conf_data["win_cp_quantile_by_condition"]
```

## Warnings

### WR-01: Edge threshold mismatch between RegimeDetector and MetaSwitcher

**File:** `src/models/regime_detector.py:183,211,224` vs `src/betting/meta_switcher.py:47,55,63`
**Issue:** Two parallel sources of regime-based strategy parameters exist with different values:

| Regime      | RegimeDetector edge_threshold | MetaSwitcher edge_threshold |
|-------------|-------------------------------|----------------------------|
| AGGRESSIVE  | 0.05                          | 0.05                       |
| CONSERVATIVE| 0.06                          | 0.07                       |
| COLLAPSED   | 0.09                          | 0.10                       |

CONSERVATIVE and COLLAPSED differ by +0.01 in MetaSwitcher. If MetaSwitcher is ever used as the strategy source instead of RegimeDetector (or both are consulted), bets will be filtered at inconsistent thresholds. This is a logic divergence that risks over- or under-filtering depending on code path.

**Fix:** Consolidate to a single source of truth. Either have MetaSwitcher delegate to `RegimeDetector.get_strategy_params()` or extract constants to a shared config. The simplest fix is to make MetaSwitcher wrap the detector's params rather than maintaining its own copy.

### WR-02: Protocol `should_retrain` typed as bare `callable` -- no type safety

**File:** `src/betting/meta_switcher.py:14`
**Issue:** `RegimeDetectorProtocol.should_retrain` is declared as `callable` (bare type), which in Python typing means `Callable[..., Any]`. This bypasses mypy checking on the return type and signature. If the concrete `RegimeDetector.should_retrain` method signature changes, mypy will not catch the mismatch through the protocol.

**Fix:**
```python
from typing import Callable

class RegimeDetectorProtocol(Protocol):
    current_regime: RegimeState
    should_retrain: Callable[[], bool]
```

### WR-03: GateKeeper.should_bet hardcodes edge threshold, ignores regime

**File:** `src/betting/gate_keeper.py:28`
**Issue:** `should_bet` uses a hardcoded `0.04` threshold while `filter_bets` accepts `edge_threshold` as a parameter. This means `should_bet` cannot respect regime-based thresholds (e.g., CONSERVATIVE=0.06, COLLAPSED=0.09). The `ev_threshold` parameter is accepted but explicitly documented as "unused" -- dead parameter.

**Fix:** Either make `should_bet` accept `edge_threshold` as a parameter (matching `filter_bets`), or remove the method if it is not called in production paths.

### WR-04: build_race_features uses mismatched pd.Series index for fallback

**File:** `src/backtest/race_predictor.py:747-754`
**Issue:** When columns `signed_log_error_win` or `abs_log_error_win` are missing, fallback `pd.Series([0.0])` is created with a default integer index (0). Subsequent operations like `.mean()`, `.std()`, `.quantile()` work but produce results for a single-element series rather than the expected per-horse distribution. While the scalar results are "safe" (0.0), this silently degrades quality scoring by masking missing features.

**Fix:** Use a Series of the same length as `race_df` for fallback:
```python
signed_error = (
    race_df["signed_log_error_win"]
    if "signed_log_error_win" in race_df.columns
    else pd.Series([0.0] * len(race_df), index=race_df.index)
)
```

### WR-05: WinSelectionGateModel.train silently returns on empty data with no logging

**File:** `src/models/win_selection_gate.py:740-751`
**Issue:** `train()` returns silently when `_prepare_training_frame` produces an empty DataFrame (line 741) or when no folds can be built (line 751). The model stays untrained (`_trained=False`), and callers have no indication why training failed. This matches the pattern from PlaceSelectionGateModel but makes debugging production issues difficult.

**Fix:** Add `logger.warning` calls at both early-return points:
```python
if prepared.empty:
    logger.warning("WinSelectionGate training skipped: no data after preparation")
    return
# ...
if not folds:
    logger.warning("WinSelectionGate training skipped: insufficient folds (need %d races)", self.min_train_races)
    return
```

## Info

### IN-01: Duplicate artifact_uri call in ModelLoader.load

**File:** `src/db/model_loader.py:65,73`
**Issue:** `mlflow.get_artifact_uri(run_id)` is called twice: once inside the try block (line 65) and again at line 73, unconditionally overwriting the value. The second call is redundant.

**Fix:** Remove line 73 or restructure to only call once.

### IN-02: Dead `self.threshold` field in WinSelectionGateModel

**File:** `src/models/win_selection_gate.py:131`
**Issue:** `self.threshold` is initialized to `0.0`, saved, and loaded, but never read or used anywhere in the class logic. It appears to be a leftover from an earlier design.

**Fix:** Remove `threshold` from `__init__`, `save`, and `load` to avoid confusion.

### IN-03: RacePredictor._build_place_selection_ev is a static passthrough

**File:** `src/backtest/race_predictor.py:47-49`
**Issue:** `_build_place_selection_ev` is a `@staticmethod` that simply calls `build_place_selection_ev(df)`. It adds no value beyond indirection.

**Fix:** Consider inlining the call or removing the wrapper.

### IN-04: Test data duplication across test_win_selection_gate tests

**File:** `tests/test_win_selection_gate.py:17-35,122-141,156-175`
**Issue:** The same 120-race dataset construction appears in 3 test functions (`test_win_selection_gate_trains_and_scores`, `test_win_selection_gate_hit_condition`, `test_win_selection_gate_save_load_roundtrip`). Extracting to a shared fixture would reduce duplication.

**Fix:** Extract a `_build_120_race_df()` helper or pytest fixture.

### IN-05: MetaSwitcher ev_threshold values diverge from RegimeDetector

**File:** `src/betting/meta_switcher.py:46,53,60`
**Issue:** `MetaSwitcher._default_params` uses `ev_threshold` values 1.15/1.35/1.55 while `RegimeDetector.get_strategy_params` uses 1.10/1.30/1.50. The MetaSwitcher values are all +0.05 higher. Combined with WR-01 (edge_threshold divergence), the two param sources are inconsistent across two dimensions. This does not cause a crash since MetaSwitcher appears unused in the critical path, but creates confusion for future consumers.

**Fix:** See WR-01 -- consolidate to single source of truth.

---

_Reviewed: 2026-05-03T00:30:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
