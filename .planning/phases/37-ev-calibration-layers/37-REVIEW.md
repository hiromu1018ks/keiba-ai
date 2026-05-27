---
phase: 37-ev-calibration-layers
reviewer: gsd-code-reviewer
depth: standard
status: issues_found
files_reviewed:
  - src/validation/__init__.py
  - src/validation/oof_health_validator.py
  - src/models/stage1_ability_model.py
  - src/pipelines/training_pipeline.py
findings:
  critical: 2
  warning: 4
  info: 2
  total: 8
---

# Phase 37: Code Review Report

**Reviewed:** 2026-05-27T12:00:00Z
**Depth:** standard
**Files Reviewed:** 4
**Status:** issues_found

## Summary

Review of OOF Health Infrastructure: OOFHealthValidator with 8 health checks, fold column additions to AbilityModel and EV pipeline, and pipeline wiring for pre-save validation + manifest generation.

Two critical bugs will prevent the pipeline from running: (1) the validator profiles require columns (`is_oof`, `oof_artifact_version`) that the pipeline never adds to the DataFrame, causing validate() to always raise; (2) the fallback path in AbilityModel.train_oof() omits the `ability_oof_fold` column. Additionally, there is a path traversal vulnerability in `load_validated_oof` and a zero-division risk in the anomaly check.

## Critical Issues

### CR-01: Pipeline validation always fails -- required columns `is_oof` and `oof_artifact_version` never added

**File:** `src/pipelines/training_pipeline.py:284-289`
**Issue:** The pipeline calls `oof_validator.validate(full_features_df, OOF_PREDICTIONS_PROFILE, ...)` where the profile requires columns `("race_id", "race_date", "is_oof", "oof_artifact_version")`. However, `full_features_df` is built from `pd.concat(oof_dfs, ...)` where no code ever adds `is_oof` or `oof_artifact_version` columns. The OOF-07 check inside `validate()` will always detect these as missing and raise `ValueError("Missing required columns: ['is_oof', 'oof_artifact_version'] (OOF-07)")`. The pipeline will crash at every training run.

**Fix:** Add the required columns to `full_features_df` before validation, or remove them from `OOF_PREDICTIONS_PROFILE.required_columns`:
```python
# Option A: Add missing columns before validation
full_features_df["is_oof"] = True
full_features_df["oof_artifact_version"] = 1

# Option B: Remove from profile (if not yet needed)
OOF_PREDICTIONS_PROFILE = OOFHealthProfile(
    required_columns=("race_id", "race_date"),
    ...
)
```

### CR-02: AbilityModel.train_oof() fallback path omits `ability_oof_fold` column

**File:** `src/models/stage1_ability_model.py:369-371`
**Issue:** When `n_dates < n_folds + 1` (insufficient data for expanding window), the fallback calls `self.add_ability_probs(df)` and returns the result. `add_ability_probs()` does NOT add an `ability_oof_fold` column. Downstream code in `training_pipeline.py` expects this column (e.g., for OOF validation at line 284 where `OOF_PREDICTIONS_PROFILE.fold_col = "ability_oof_fold"`). This causes a crash when training with small datasets.

**Fix:** Add the fold column in the fallback path:
```python
if n_dates < n_folds + 1:
    self.train(df, num_threads=num_threads)
    df = self.add_ability_probs(df)
    df["ability_oof_fold"] = pd.NA  # or pd.array([pd.NA] * len(df), dtype=pd.Int64Dtype())
    return df
```

## Warnings

### WR-01: Path traversal vulnerability in `load_validated_oof`

**File:** `src/validation/oof_health_validator.py:369-379`
**Issue:** `manifest_path = Path(index_data[artifact_name])` and `artifact_path = Path(manifest["artifact_path"])` construct filesystem paths directly from JSON content without any validation. A malicious or corrupted `index.json` or manifest file could contain paths like `../../etc/passwd` or `C:\Windows\System32\config\SAM`, causing the function to read arbitrary files. The SHA256 hash verification only verifies file integrity, not file location.

**Fix:** Validate that resolved paths stay within the expected base directory:
```python
_BASE_DIR = Path("data/oof").resolve()

def _validate_path(path: Path) -> Path:
    resolved = path.resolve()
    if not str(resolved).startswith(str(_BASE_DIR)):
        raise ValueError(f"Path escapes OOF directory: {path}")
    return resolved
```

### WR-02: Zero-division risk in `_check_top1_anomaly` when all score values are NaN

**File:** `src/validation/oof_health_validator.py:179-189`
**Issue:** `df.groupby(...)[profile.score_col].idxmax()` with all-NaN groups returns NaN indices, which when passed to `df.loc[...]` may produce an empty DataFrame (or error depending on pandas version). If `n_races` ends up as 0, line 189 `hit_rate = float((top1_rows["kakuteijyuni"] == 1).sum() / n_races)` raises `ZeroDivisionError`.

**Fix:** Guard against empty result:
```python
top1_idx = df.groupby("race_id", observed=True)[profile.score_col].idxmax()
top1_idx = top1_idx.dropna()
if len(top1_idx) == 0:
    return
top1_rows = df.loc[top1_idx]
n_races = len(top1_rows)
if n_races == 0:
    return
```

### WR-03: `expected_row_count=len(full_features_df)` makes OOF-04 coverage check vacuous

**File:** `src/pipelines/training_pipeline.py:288`
**Issue:** The coverage check computes `len(df) / expected_row_count`. Since `expected_row_count` is set to `len(full_features_df)` (the same DataFrame being validated), coverage will always be exactly 1.0 (100%). This means OOF-04 can never fail, defeating the purpose of the check.

**Fix:** The expected row count should come from an independent source -- e.g., the number of rows in the original training data before feature generation, or a previously recorded count:
```python
expected_row_count = len(feat_df)  # original feature data before OOF subsetting
```

### WR-04: Redundant validation call -- `generate_manifest()` re-runs full validation

**File:** `src/pipelines/training_pipeline.py:284-300` and `src/validation/oof_health_validator.py:246-251`
**Issue:** The pipeline calls `oof_validator.validate()` at line 284, checks the result, then calls `generate_manifest()` at line 300 which internally calls `validate()` again (line 246 of oof_health_validator.py). This doubles the computational cost of validation unnecessarily, especially for the groupby operations in OOF-03 and OOF-06.

**Fix:** Either (a) pass the already-computed validation result to `generate_manifest()`, or (b) remove the explicit `validate()` call in the pipeline and rely on `generate_manifest()` to do it internally:
```python
# Option A: Pass result to generate_manifest
manifest = oof_validator.generate_manifest(
    full_features_df,
    OOF_PREDICTIONS_PROFILE,
    artifact_hash,
    train_date_range=train_date_range,
    _cached_result=oof_result,  # reuse
)
```

## Info

### IN-01: `generate_manifest` includes fold key `"<NA>"` for nullable Int64 columns

**File:** `src/validation/oof_health_validator.py:254-261`
**Issue:** When the fold column is `pd.Int64Dtype()` (nullable integer), `value_counts()` includes `pd.NA` as a key. `str(pd.NA)` produces the string `"<NA>"`, which gets written as a fold key in the JSON manifest. This is not a crash bug but produces misleading manifest entries where `"<NA>"` appears as a fold number alongside valid fold indices like `"0"`, `"1"`, `"2"`.

**Fix:** Filter out NA fold entries before building the manifest:
```python
fold_series = df[fold_col].dropna()
fold_counts = fold_series.value_counts().to_dict()
```

### IN-02: Private function `_update_index` exported via module-level import

**File:** `src/validation/oof_health_validator.py:334` and `src/pipelines/training_pipeline.py:57`
**Issue:** `_update_index` is a module-level function with a private naming convention (underscore prefix), but it is explicitly imported in `training_pipeline.py` via `from validation.oof_health_validator import ... _update_index`. The underscore prefix suggests it should not be part of the public API. If it is intended for cross-module use, it should be renamed without the underscore prefix.

**Fix:** Rename to `update_oof_index` if it is part of the public API, or move the index update logic into the pipeline itself.

---

_Reviewed: 2026-05-27T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
