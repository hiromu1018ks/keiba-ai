---
phase: 33-gain-per-depth-diagnostic
reviewed: 2026-05-18T12:00:00Z
depth: standard
files_reviewed: 4
files_reviewed_list:
  - src/models/gpd_diagnostics.py
  - tests/test_gpd_diagnostics.py
  - scripts/run_gpd.py
  - tests/test_run_gpd.py
findings:
  critical: 0
  warning: 3
  info: 4
  total: 7
status: issues_found
---

# Phase 33: Code Review Report

**Reviewed:** 2026-05-18T12:00:00Z
**Depth:** standard
**Files Reviewed:** 4
**Status:** issues_found

## Summary

Reviewed four files implementing the Gain per Depth Diagnostic feature: a core module (`gpd_diagnostics.py`), a CLI script (`run_gpd.py`), and their test files. The code is generally well-structured with thorough test coverage for the GPD pipeline. However, there are a few defects: a silent data drop when Market/EV/Wide models are not Boosters, an inconsistency in StackedEnsemble unwrapping logic between `win` and `place` paths, a fragile test assertion that depends on import ordering, and some minor code quality concerns.

## Critical Issues

No critical issues found.

## Warnings

### WR-01: Non-Booster models silently dropped without `_is_booster` guard

**File:** `src/models/gpd_diagnostics.py:276-309`
**Issue:** The `_extract_boosters` function applies `_is_booster()` validation for `stage1`, `win.hit_model`, `win.return_model`, and `place` models, but does NOT apply it for `market.model` (line 277-278), `ev_corrector.*` (lines 281-284), `wide.*` (lines 306-309), or `conformal_ev_model.*` (lines 313-316). These are stored directly via `is not None` checks only. If any of these fields were set to a non-Booster object (e.g., a wrapper or a different model type after a refactoring), `_extract_boosters` would silently include it and `_compute_depth_gains` would crash with an `AttributeError` on `trees_to_dataframe()`. The `compute_gpd_diagnostics` function does catch this at line 508 via a broad `except Exception` and logs a warning, but the booster is then silently omitted from the report with no indication of *why* it failed. This inconsistency between validated and unvalidated paths is a maintenance hazard.
**Fix:** Apply `_is_booster()` consistently to all extracted models, or at minimum wrap each extraction in a try/except with a specific log message:

```python
# Market Model
if sub.market.model is not None:
    model = sub.market.model
    if _is_booster(model):
        boosters[f"market_{surface}"] = model
    else:
        logger.warning("market_%s is not a Booster (type=%s), skipping", surface, type(model).__name__)
```

Apply the same pattern to `ev_corrector`, `wide`, and `conformal_ev_model` extractions.

### WR-02: Inconsistent StackedEnsemble unwrapping -- `place` path does not unwrap

**File:** `src/models/gpd_diagnostics.py:287-291`
**Issue:** The `win` extraction path (lines 267-271) handles the case where `hit_model` is a `StackedEnsemble` by unwrapping to `.lgbm_model`. The `place` extraction path (lines 287-291) does not perform this unwrapping -- it only checks `_is_booster()`. While `PlaceTwoStageModel.hit_model` is currently always an `lgb.Booster` (never wrapped in `StackedEnsemble`), this asymmetry means that if `PlaceTwoStageModel` ever gains ensemble support (like `WinTwoStageModel` did), the place booster would be silently skipped. This violates the principle of least surprise and creates an inconsistency in how the two two-stage model types are handled.
**Fix:** Add StackedEnsemble unwrapping logic to the place extraction path, mirroring the win path:

```python
# Place (optional)
if sub.place is not None:
    place_hit = sub.place.hit_model
    if _is_booster(place_hit):
        boosters[f"place_hit_{surface}"] = place_hit
    elif hasattr(place_hit, "lgbm_model") and place_hit.lgbm_model is not None:
        boosters[f"place_ensemble_lgbm_{surface}"] = place_hit.lgbm_model
    if _is_booster(sub.place.return_model):
        boosters[f"place_ret_{surface}"] = sub.place.return_model
```

### WR-03: Broad exception catch silently skips boosters

**File:** `src/models/gpd_diagnostics.py:506-510`
**Issue:** The `compute_gpd_diagnostics` function wraps `_compute_depth_gains` in a bare `except Exception` block and continues to the next booster. While this is robust against individual booster failures, it silently omits failed boosters from the report. If a critical model (e.g., `stage1_turf`) fails, the user gets a partial report with no indication that data is missing. The logged warning uses `exc_info=True` which helps, but there is no summary count of failed boosters in the final output.
**Fix:** Track failed boosters and include the count in the metadata:

```python
failed_names: list[str] = []
for name, booster in boosters.items():
    try:
        depth_gains = _compute_depth_gains(booster)
    except Exception:
        logger.warning("Failed to compute depth gains for %s", name, exc_info=True)
        failed_names.append(name)
        continue
    # ...

result["metadata"]["failed_boosters"] = failed_names
result["metadata"]["num_failed"] = len(failed_names)
```

## Info

### IN-01: Unused import `np` in test file

**File:** `tests/test_gpd_diagnostics.py:13`
**Issue:** `import numpy as np` is imported at the top of the test file but `np` is only used inside `test_nan_split_gain_filled_to_zero` (to create `np.nan`). While this is not technically unused, it is worth noting as the test file imports numpy solely for one constant. A `float("nan")` would avoid the import.
**Fix:** Replace `np.nan` with `float("nan")` at line 360 and remove the `import numpy as np` statement, or keep as-is if numpy is expected in the test suite.

### IN-02: `MagicMock` with `spec` does not perfectly simulate `StackedEnsemble`

**File:** `tests/test_gpd_diagnostics.py:180`
**Issue:** The mock ensemble is created as `MagicMock(spec=["lgbm_model", "best_iteration", "predict"])`. This means `hasattr(mock_ensemble, "trees_to_dataframe")` returns `False`, which causes `_is_booster()` to return `False`. This happens to work correctly for the test, but it tests the negative case by construction rather than by simulating a real `StackedEnsemble` object. If `_is_booster` ever changes its duck-type check criteria, this test might break or pass incorrectly.
**Fix:** Consider creating a more realistic mock that explicitly sets `trees_to_dataframe` and `feature_importance` to `None` or removes them, rather than relying on `spec` to implicitly exclude them.

### IN-03: `plot_gpd_charts` does not handle `output_dir` as string

**File:** `scripts/run_gpd.py:95`
**Issue:** The function signature declares `output_dir: Path`, but the first line wraps it with `Path(output_dir)`. While this handles string input gracefully, callers from `main()` already pass `args.output_dir` which is a `Path` object (due to `type=Path` in argparse). The redundant `Path()` call is harmless but suggests uncertainty about the type contract.
**Fix:** Either remove the `Path()` wrap (trusting the type annotation) or change the annotation to `str | Path` to document the flexibility.

### IN-04: Test `test_main_default_args` assertion may be fragile

**File:** `tests/test_run_gpd.py:412`
**Issue:** The assertion `call_kwargs.kwargs.get("use_ensemble_override") is False` uses `is` identity comparison with the boolean `False`. While this works in CPython (which caches `False` as a singleton), the `mock_loader.return_value.load_from_dir.call_args` returns a `unittest.mock.call` object whose `.kwargs` dict may contain the value `False` directly from argparse, so `is` comparison should work. However, the idiomatic and safer comparison is `== False` or `assert not ...`.
**Fix:** Use `assert call_kwargs.kwargs.get("use_ensemble_override") == False` or `assert not call_kwargs.kwargs.get("use_ensemble_override")` for clarity.

---

_Reviewed: 2026-05-18T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
