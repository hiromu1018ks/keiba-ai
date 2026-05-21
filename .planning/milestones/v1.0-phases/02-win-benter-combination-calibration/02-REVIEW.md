---
phase: 02-win-benter-combination-calibration
reviewed: 2026-05-02T23:00:00Z
depth: standard
files_reviewed: 8
files_reviewed_list:
  - src/models/win_benter_gate.py
  - src/domain/models.py
  - src/pipelines/training_pipeline.py
  - src/db/model_loader.py
  - src/backtest/race_predictor.py
  - tests/test_win_benter_gate.py
  - tests/test_race_predictor.py
  - pyproject.toml
findings:
  critical: 2
  warning: 5
  info: 3
  total: 10
status: issues_found
---

# Phase 2: Code Review Report

**Reviewed:** 2026-05-02T23:00:00Z
**Depth:** standard
**Files Reviewed:** 8
**Status:** issues_found

## Summary

Phase 2 adds Win Benter combination (market probability blending for win predictions), calibration comparison (Beta vs Isotonic), and integration into training pipeline, model loader, and race predictor. The implementation is well-structured with good test coverage, but contains two critical defects that will cause silent inference degradation in production:

1. **MLflow model loading path is missing all Benter/Place/Benter calibration fields**, meaning models loaded from MLflow will silently skip probability blending at inference time.
2. **Malformed try/except in model_loader.py** makes the fallback confidence estimator loading unreachable, silently using defaults when MLflow download fails.

Additional issues include a redundant grid-search loop that re-computes invariant data 36 times, variable-scope fragility in the calibration comparison block, and a variable naming inconsistency where `win_isotonic_calibrator` may hold a `BetaCalibrationManual` object.

## Critical Issues

### CR-01: MLflow load path missing Benter and calibration fields in SubmodelSet construction

**File:** `src/db/model_loader.py:188-199`
**Issue:** The `SubmodelSet` constructor call in the MLflow `load()` method (lines 188-199) does not pass `benter_combo`, `isotonic_calibrator`, `temperature_scaler`, `win_benter`, `win_isotonic_calibrator`, or `win_temperature_scaler`. These fields default to `None`. By contrast, the `_load_from_local()` method (lines 560-577) correctly loads and passes all six fields.

This means: when models are loaded from MLflow (fallback path or direct MLflow URI), the inference pipeline in `race_predictor.py` will skip both Place Benter combination (line 147: `benter = submodel.benter_combo` -> `None`) and Win Benter gate (line 116: `getattr(submodel, "win_benter", None)` -> `None`). Both Place and Win predictions will fall back to raw model probabilities without market blending, silently degrading prediction quality.

The MLflow path is the primary production loading mechanism when local `data/models/` does not exist, and is also used for loading specific run IDs. Any deployment relying on MLflow will be affected.

**Fix:**
```python
# In model_loader.py load() method, inside the surface loop,
# add Benter/calibration loading (similar to _load_from_local lines 502-558):
# Then update the SubmodelSet constructor:

submodels[surface] = SubmodelSet(
    market=market,
    stage1=ability,
    place_ability=pa,
    win=win,
    ev_corrector=ev_corr,
    place=place,
    place_ev_corrector=place_ev_corr,
    wide=wide,
    confidence=confidence,
    place_selection_gate=place_selection_gate,
    # Add these missing fields:
    benter_combo=benter_combo,
    isotonic_calibrator=isotonic_calibrator,
    temperature_scaler=temperature_scaler,
    win_benter=win_benter,
    win_isotonic_calibrator=win_isotonic_calibrator,
    win_temperature_scaler=win_temperature_scaler,
)
```

### CR-02: Malformed try/except makes RobustConfidenceEstimator fallback unreachable

**File:** `src/db/model_loader.py:166-186`
**Issue:** The `try/except` block for loading `RobustConfidenceEstimator` parameters has an invalid structure. There is a `try` at line 168, an `except Exception` at line 172 (for MLflow download failure), and another `except Exception` at line 185. The second `except` is syntactically unreachable -- Python will never reach it because the first `except Exception` already catches all exceptions from the `try` block.

The code at lines 174-184 (inside the first `except`) attempts a filesystem fallback but is NOT wrapped in its own try/except. If the fallback also fails (e.g., `conf_path` does not exist, invalid JSON), the exception propagates unhandled and crashes the entire model loading.

The intended design appears to be: try MLflow download -> if fails, try filesystem -> if that also fails, use defaults. But the second fallback (using defaults) at line 185-186 can never execute.

**Fix:**
```python
confidence = RobustConfidenceEstimator()
try:
    conf_path = mlflow.artifacts.download_artifacts(
        f"runs:/{run_id}/confidence_params.json"
    )
    with open(conf_path) as f:
        conf_data = json.load(f)
    confidence.alpha = conf_data["alpha"]
    confidence.rolling_window = conf_data["rolling_window"]
    confidence._win_cp_quantile = conf_data["win_cp_quantile"]
    confidence._place_cp_quantile = conf_data["place_cp_quantile"]
    confidence._win_rolling_quantile = conf_data["win_rolling_quantile"]
    confidence._place_rolling_quantile = conf_data["place_rolling_quantile"]
    confidence._calibrated = True
except Exception:
    # Fallback: filesystem
    try:
        conf_dir = self._find_artifact_dir(run_id, "confidence_params.json")
        conf_path = str(conf_dir / "confidence_params.json")
        with open(conf_path) as f:
            conf_data = json.load(f)
        confidence.alpha = conf_data["alpha"]
        confidence.rolling_window = conf_data["rolling_window"]
        confidence._win_cp_quantile = conf_data["win_cp_quantile"]
        confidence._place_cp_quantile = conf_data["place_cp_quantile"]
        confidence._win_rolling_quantile = conf_data["win_rolling_quantile"]
        confidence._place_rolling_quantile = conf_data["place_rolling_quantile"]
        confidence._calibrated = True
    except Exception:
        logger.warning(
            "RobustConfidenceEstimator params not found, using defaults"
        )
```

## Warnings

### WR-01: Grid search re-computes invariant logit arrays 36 times

**File:** `src/pipelines/training_pipeline.py:596-628`
**Issue:** Inside the `for a0, b0, g0 in iter_product(...)` loop (36 iterations), `BenterCombination._logit(oof_p_fund)` and `BenterCombination._logit(oof_p_market)` are computed on every iteration despite the data being invariant. Additionally, `y_arr = oof_y.astype(float)` is also recomputed each time. These should be hoisted before the loop.

**Fix:** Move the three invariant computations before line 596:
```python
logit_f = BenterCombination._logit(oof_p_fund)
logit_m = BenterCombination._logit(oof_p_market)
y_arr = oof_y.astype(float)

for a0, b0, g0 in iter_product(alpha_grid, beta_grid, gamma_grid):
    ...
```

### WR-02: `win_isotonic_calibrator` field may hold BetaCalibrationManual object

**File:** `src/pipelines/training_pipeline.py:678-679`
**Issue:** When the calibration comparison winner is "beta", line 679 stores `cal_result["beta_calibrator"]` into `win_isotonic_cal`. This variable is then assigned to `SubmodelSet.win_isotonic_calibrator`. The field name implies it holds an `IsotonicRegression` object, but it may hold either `BetaCalibration` (from the betacal package) or `BetaCalibrationManual` (the fallback). While `.transform()` works on all three types, the naming is misleading and could confuse future maintainers or code that checks `isinstance(calibrator, IsotonicRegression)`.

**Fix:** Rename the field to `win_calibrator` (or document that it may hold any object with a `.transform()` method). Alternatively, store the calibrator type in metadata:
```python
# In SubmodelSet, rename:
win_calibrator: object | None = None  # IsotonicRegression, BetaCalibration, or BetaCalibrationManual
```

### WR-03: Variable scope fragility in calibration comparison block

**File:** `src/pipelines/training_pipeline.py:659`
**Issue:** The calibration comparison block at line 659 references `oof_p_fund`, `oof_p_market`, and `oof_y` which are defined only inside the `if "tanodds" in df_oof.columns and len(df_oof) >= 500:` block at line 571. While the `win_benter is not None` guard at line 659 ensures these variables exist (since `win_benter` can only become non-None inside the line 571 block), this is fragile: a future refactor that sets `win_benter` elsewhere would cause an `UnboundLocalError`. The variables should be initialized at the same scope as `win_benter` to prevent this.

**Fix:** Initialize at the same scope as the `win_benter = None` line (569):
```python
win_benter = None
win_isotonic_cal = None
win_temp_scaler = None
oof_p_fund = np.array([])  # sentinel
oof_p_market = np.array([])
oof_y = np.array([])
```

### WR-04: `generate_win_oof_predictions` uses NaN fill of 0.5 for market probability instead of median

**File:** `src/models/win_benter_gate.py:126-130`
**Issue:** When `tanodds <= 0`, the market probability is set to `np.nan`, which is then NOT filled before clipping (unlike `extract_market_probability` which fills with 0.5). The `np.clip(0.01, 0.99)` on `np.nan` returns `np.nan`, which is then filtered out by the `valid` mask at line 134. This is correct but inconsistent with `extract_market_probability` which fills NaN with 0.5 before clipping. If the intent is to exclude invalid odds, the comment should note this. If the intent is to impute, the behavior differs from `WinBenterGate.extract_market_probability`.

**Fix:** Either align with `extract_market_probability` (fill NaN with 0.5 before clip) or add a comment explaining the intentional difference:
```python
p_market = np.where(
    df["tanodds"] > 0,
    1.0 / df["tanodds"].values,
    np.nan,  # intentionally NaN: invalid odds excluded via valid mask below
)
```

### WR-05: `compare_calibrations` sets `has_beta = True` even when betacal import failed

**File:** `src/models/win_benter_gate.py:274-279`
**Issue:** When the `betacal` package import or fit fails, the code falls back to `BetaCalibrationManual` and sets `has_beta = True` on line 279. This means the "beta" path is always taken, and the `has_beta = False` initial state (line 266) is effectively dead code. The `beta_calibrator` in the return dict will always be set (never None), which is fine functionally but makes the `has_beta` flag meaningless.

**Fix:** Either remove the `has_beta` flag entirely (since it's always True) or distinguish between the real `betacal` and the manual fallback:
```python
has_beta = False
beta_cal = None
try:
    from betacal import BetaCalibration
    beta_cal = BetaCalibration(parameters="abc")
    beta_cal.fit(p_train, y_train)
    p_beta = np.asarray(beta_cal.transform(p_val), dtype=float)
    has_beta = True
except (ImportError, Exception) as e:
    logger.warning("betacal unavailable or failed (%s), using manual fallback", e)
    beta_cal = BetaCalibrationManual()
    beta_cal.fit(p_train, y_train)
    p_beta = np.asarray(beta_cal.transform(p_val), dtype=float)
    has_beta = True  # or False, and handle downstream
```

## Info

### IN-01: Redundant logit computation inside grid-search loop

**File:** `src/pipelines/training_pipeline.py:596-628`
**Issue:** (Duplicate of WR-01 but noted for completeness.) The loop body recomputes `BenterCombination._logit()` on invariant arrays 36 times. Hoisting these out of the loop would reduce computational overhead during training.

### IN-02: `_save_models_local` saves confidence_params.json per surface but overwrites

**File:** `src/pipelines/training_pipeline.py:1213-1226`
**Issue:** The `confidence_params.json` file is written once per surface inside the `for surface, sub in models.items()` loop (line 1213), meaning the last surface's confidence parameters overwrite all previous ones. Since `RobustConfidenceEstimator` is trained per-surface but the confidence params file is shared, the turf confidence params may be silently overwritten by dirt. This is pre-existing behavior (not introduced in Phase 2) but worth noting.

### IN-03: `pyproject.toml` declares `betacal>=1.0` as a required dependency

**File:** `pyproject.toml:23`
**Issue:** `betacal>=1.0` is listed as a required (non-optional) dependency, but `win_benter_gate.py` implements `BetaCalibrationManual` as a fallback when betacal is unavailable. If the manual fallback is considered sufficient, `betacal` should be an optional dependency to avoid installation failures on environments where betacal has compatibility issues.

---

_Reviewed: 2026-05-02T23:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
