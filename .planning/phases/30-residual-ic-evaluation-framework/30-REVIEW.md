---
phase: 30-residual-ic-evaluation-framework
reviewed: 2026-05-18T05:07:10Z
depth: standard
files_reviewed: 4
files_reviewed_list:
  - src/models/ic_evaluator.py
  - tests/test_ic_evaluator.py
  - src/pipelines/training_pipeline.py
  - scripts/run_ic_eval.py
findings:
  critical: 2
  warning: 3
  info: 3
  total: 8
status: issues_found
---

# Phase 30: Code Review Report

**Reviewed:** 2026-05-18T05:07:10Z
**Depth:** standard
**Files Reviewed:** 4
**Status:** issues_found

## Summary

Reviewed 4 files implementing the Residual IC Evaluation Framework. Found 2 critical bugs in
`src/models/ic_evaluator.py`: (1) a crash when Spearman rho is exactly 0.0 due to Python's falsy
short-circuit in an `or` chain, and (2) a sign-convention mismatch between the per-race IC
formulation and the other three formulations (B/C/E), which invalidates the direction consistency
check. Also found 3 warnings (missing validation on implied_prob, untested public internal function,
and inconsistent data subsets across formulations) and 3 info-level items.

## Critical Issues

### CR-01: `_check_direction_consistency` crashes when Spearman rho is exactly 0.0

**File:** `src/models/ic_evaluator.py:152`
**Issue:**
The expression `metric.get("rho") or metric.get("delta_ic") or metric.get("mean_rho")` uses
Python's `or` operator, which short-circuits on falsy values. When `rho = 0.0` (a legitimate
Spearman correlation value indicating no monotonic relationship), `0.0` is falsy in Python, so
the chain falls through to `metric.get("delta_ic")` (returns `None` for b_difference/c_orthogonal
metrics) and then `metric.get("mean_rho")` (also `None`). The result is `rho = None`, and the
subsequent `np.isfinite(None)` on line 153 raises `TypeError`.

This will crash `run_ic_evaluation()` any time a Spearman rho is exactly 0.0, which is a realistic
scenario especially in low-signal regimes or with small sample sizes.

**Fix:**
```python
# Line 152: Replace or-chain with explicit None-check
rho = metric.get("rho")
if rho is None:
    rho = metric.get("delta_ic")
if rho is None:
    rho = metric.get("mean_rho")
```

### CR-02: Per-race IC has inverted sign convention vs. B/C/E formulations

**File:** `src/models/ic_evaluator.py:240` (per-race) vs lines 237-239 (B/C/E)
**Issue:**
The B-difference, C-orthogonal, and E-incremental formulations use binary `y` (1=win, 0=lose)
where a positive Spearman rho indicates the model adds predictive value. However, `_compute_per_race_ic`
uses raw `kakuteijyuni` values (1=1st place, 2=2nd place, ...) where a positive Spearman rho between
model predictions and finishing position means higher predictions correspond to *worse* finishes --
the opposite of the desired interpretation.

This means:
1. For a well-calibrated model, B/C/E ICs will be positive while per-race IC will be negative.
2. The `_check_direction_consistency` function will always flag a working model as "inconsistent."
3. The direction consistency check, which is a core verification mechanism (RIC-06), produces
   misleading results.

**Fix:**
```python
# Option A: Negate per-race rho so positive = good (consistent with B/C/E)
rho, _ = spearmanr(pred.loc[common].values, actual.loc[common].values)
rho = -rho  # Invert: lower kakuteijyuni (better finish) -> positive IC

# Option B: Convert kakuteijyuni to binary before per-race IC
# In run_ic_evaluation, add a binary column:
sub_df_bin = sub_df.copy()
sub_df_bin["_y_binary"] = (pd.to_numeric(sub_df_bin[IC_TARGET_COLUMN], errors="coerce") == 1).astype(float)
"per_race": _compute_per_race_ic(sub_df_bin, pred_col, "_y_binary"),
```

## Warnings

### WR-01: `_get_market_probability` returns unvalidated values for `implied_prob` column

**File:** `src/models/ic_evaluator.py:45-46`
**Issue:**
When the `implied_prob` column exists, values are returned raw via `pd.to_numeric(..., errors="coerce").values`
without any bounds checking. In contrast, the `tanodds` fallback path (lines 47-48) clips output to
[0.01, 0.99]. Invalid `implied_prob` values (0.0, negative, or > 1.0) would propagate into delta
computations and OLS regression, producing nonsensical IC values without any warning.

**Fix:**
```python
if "implied_prob" in df.columns:
    raw = pd.to_numeric(df["implied_prob"], errors="coerce").values
    return np.clip(np.where(np.isfinite(raw), raw, np.nan), 0.01, 0.99)
```

### WR-02: `model_prob_filter` is public and untested, but should be private

**File:** `src/models/ic_evaluator.py:265`
**Issue:**
`model_prob_filter` is a module-level public function (no `_` prefix) used only internally by
`run_ic_evaluation`. It is not imported by any test or external module. Per the project convention
(private helpers use `_` prefix), it should be renamed to `_model_prob_filter`. Additionally,
it has zero direct test coverage (only tested indirectly via `run_ic_evaluation`).

**Fix:** Rename to `_model_prob_filter` and add direct unit tests for edge cases (empty mask,
all-NaN subset, exact MIN_SAMPLE_SIZE boundary).

### WR-03: Per-race IC operates on a different data subset than B/C/E formulations

**File:** `src/models/ic_evaluator.py:229-240`
**Issue:**
B/C/E formulations receive NaN-filtered arrays from `model_prob_filter` (lines 230, 235), which
removes rows where *any* of model_pred, market_prob, or y is NaN. Per-race IC receives the raw
`sub_df` DataFrame (line 229/240) and performs its own independent NaN handling inside
`_compute_per_race_ic`. This means the four IC formulations being compared may operate on different
row subsets, making their comparison semantically inconsistent. For example, rows with NaN
`market_prob` but valid `model_pred` and `kakuteijyuni` would be excluded from B/C/E but included
in per-race IC.

**Fix:** Pass the same validated sub_df (with NaN rows removed) to `_compute_per_race_ic`, or
document that per-race IC intentionally includes more data points.

## Info

### IN-01: Duplicate `import mlflow` in run_ic_eval.py

**File:** `scripts/run_ic_eval.py:56,68`
**Issue:** `import mlflow` appears twice inside the `if args.mlflow:` block (lines 56 and 68).
Python caches module imports so this is functionally harmless, but it is unnecessary.
**Fix:** Move `import mlflow` to the top of the `if args.mlflow:` block (line 55-56) and remove
the second import at line 68.

### IN-02: Loose `dict` return type annotations

**File:** `src/models/ic_evaluator.py:55,70,89,116,147,178`
**Issue:** Six functions return bare `dict` without type parameters (e.g., `-> dict` instead of
`-> dict[str, Any]`). While mypy with `disallow_untyped_defs = true` accepts bare `dict` as a
valid annotation, it provides no type safety for callers. This is inconsistent with the project's
stated goal of strict typing.
**Fix:** Add type parameterization, e.g., `-> dict[str, Any]`.

### IN-03: Tests use binary `kakuteijyuni` values that don't match production data shape

**File:** `tests/test_ic_evaluator.py:30,41`
**Issue:** Test helpers `_make_arrays` and `_make_oof_df` generate `kakuteijyuni` as binary (0/1)
values via `(rng.rand(n) < model_pred).astype(float)`. In production data, `kakuteijyuni` contains
finishing positions (1, 2, 3, ...). This means the per-race IC tests do not exercise the sign-
convention issue identified in CR-02, because binary targets happen to have the same rank ordering
as the "lower = better" convention.
**Fix:** Generate realistic `kakuteijyuni` values (1..field_size) in test helpers and add a test
that verifies per-race IC sign convention matches B/C/E formulations.

---

_Reviewed: 2026-05-18T05:07:10Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
