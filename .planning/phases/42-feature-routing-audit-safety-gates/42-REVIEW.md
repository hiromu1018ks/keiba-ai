---
phase: 42
phase_name: Feature Routing Audit & Safety Gates
reviewed: 2026-05-28T12:00:00Z
reviewer: code-review
depth: standard
status: issues_found
critical: 0
warning: 4
info: 3
files_reviewed:
  - src/audit/__init__.py
  - src/audit/feature_routing_registry.py
  - src/backtest/deployment_gates.py
  - src/validation/artifact_profiles.py
  - scripts/run_feature_routing_audit.py
  - tests/test_feature_routing_audit.py
  - tests/test_artifact_profiles.py
  - tests/test_deployment_gates.py
---

# Phase 42: Code Review Report

**Reviewed:** 2026-05-28
**Depth:** standard
**Files Reviewed:** 8
**Status:** issues_found

## Summary

Reviewed 8 source files for Phase 42 (SAF-01/02/03: Feature Routing Audit, Artifact Profiles, Deployment Gates). The audit registry and gate evaluator are well-structured with frozen dataclasses, proper error handling, and comprehensive test coverage. However, several issues were found: inconsistent fold column naming between `required_columns` and `fold_col` defaults in artifact profiles, `sys.exit()` in a library function, a dead code path in the CLI script, and mutable default arguments in non-frozen dataclasses.

## Warnings

### WR-01: Inconsistent fold_col vs required_columns defaults in CalibratorArtifactProfile

**File:** `src/validation/artifact_profiles.py:34-39`
**Issue:** `required_columns` includes `"fold"` but `fold_col` defaults to `"ability_oof_fold"`. These represent the same conceptual column with two different names. The `validate()` method checks `required_columns` for `"fold"` (line 57-59), then separately checks `self.fold_col` for `"ability_oof_fold"` (line 65). When `"fold"` is present but `"ability_oof_fold"` is absent, the code falls through to a silent pass at line 67-68 rather than flagging a mismatch. A real MAWC OOF artifact will have `ability_oof_fold` (matching `fold_col`), but then the `required_columns` check looking for `"fold"` would succeed because `"fold"` is not in the dataframe -- wait, `"fold"` IS in `required_columns`, so it WOULD be flagged as missing. The net effect: real MAWC artifacts with `ability_oof_fold` (not `"fold"`) will produce a spurious "Required column 'fold' missing" failure because `required_columns` demands `"fold"` but the actual column is `"ability_oof_fold"`. Tests pass only because test data uses the short alias `"fold"` which matches `required_columns` but is not the real column name.
**Fix:** Either change `required_columns` default to include `"ability_oof_fold"` instead of `"fold"`, or change `fold_col` default to `"fold"`. The two must agree on the same column name. The `required_columns` check should reference the same column as `fold_col`:
```python
required_columns: tuple[str, ...] = (
    "race_id", "p_win_combined", "p_win_final", "ability_oof_fold",
),
```

### WR-02: Same fold_col / required_columns inconsistency in RankerArtifactProfile

**File:** `src/validation/artifact_profiles.py:143-149`
**Issue:** Identical to WR-01. `required_columns` includes `"fold"` while `fold_col` defaults to `"ability_oof_fold"`. The same silent-fallback logic at lines 169-171 masks the mismatch in tests, but real OOF artifacts would produce incorrect validation results.
**Fix:** Same as WR-01 -- align `required_columns` and `fold_col` on the same column name.

### WR-03: sys.exit() in library function run_deployment_gates()

**File:** `src/backtest/deployment_gates.py:826`
**Issue:** `run_deployment_gates()` calls `sys.exit(1)` on FAIL (line 826). This function is importable as a library function (`from backtest.deployment_gates import run_deployment_gates`), but calling `sys.exit()` terminates the entire process, making it unusable from other Python code or tests without catching `SystemExit`. The function already returns the `GateEvaluationResult`, so callers can check `result.overall_status` themselves. The `sys.exit()` should only be in a CLI `if __name__ == "__main__"` block.
**Fix:** Remove `sys.exit(1)` from `run_deployment_gates()`. Add a separate CLI wrapper:
```python
def run_deployment_gates(...) -> GateEvaluationResult:
    # ... existing logic minus sys.exit ...
    if result.overall_status == "FAIL":
        logger.error("Deployment gate evaluation FAILED -- see report for details")
    return result
```

### WR-04: GateConditionResult and GateEvaluationResult are mutable dataclasses

**File:** `src/backtest/deployment_gates.py:51-69`
**Issue:** `GateConditionResult` and `GateEvaluationResult` are plain `@dataclass` (mutable), while `GatePolicy` is `@dataclass(frozen=True)`. The phase conventions specify frozen dataclasses for config objects, and both of these carry audit/result data that should be immutable after construction. `GateConditionResult` is appended to a list and never modified, and `GateEvaluationResult` contains the final verdict. Neither should be mutated post-construction.
**Fix:** Add `frozen=True` to both dataclasses:
```python
@dataclass(frozen=True)
class GateConditionResult:
    ...

@dataclass(frozen=True)
class GateEvaluationResult:
    ...
```

## Info

### IN-01: Dead code path for --registry-version check in run_audit()

**File:** `scripts/run_feature_routing_audit.py:67-69`
**Issue:** The `run_audit()` function checks `if "--registry-version" in sys.argv` and then does nothing (`pass`). The comment says "Already validated by argparse above" but this block is unreachable dead code. The actual version validation happens in `main()` at line 171. The `run_audit()` function does not accept a `registry_version` parameter, so this check can never do anything useful.
**Fix:** Remove lines 66-69 from `run_audit()` entirely, or pass `registry_version` as a parameter and validate there.

### IN-02: Markdown table delimiter misalignment in _build_markdown_report()

**File:** `scripts/run_feature_routing_audit.py:121-122`
**Issue:** The Markdown table header uses 25 characters for the "Forbidden Intersections" column (`|------------------------|`) but the header cell text is only 25 characters. This is cosmetic and renders correctly, but the table delimiters are inconsistent widths across the two table sections (critical vs advisory), which could confuse maintenance.

### IN-03: Bare Exception catch in _get_model_features()

**File:** `src/audit/feature_routing_registry.py:189`
**Issue:** `_get_model_features()` catches `Exception` broadly when importing model modules. While this is intentional for robustness (logging a warning instead of crashing), it could silently swallow `ImportError` due to a real bug (e.g., syntax error in a model file, missing dependency). The `logger.warning()` at least makes it visible in logs, and the function returns `None` which produces an "ERROR" status in the audit result. Acceptable for this use case, but worth noting.

---

## REVIEW COMPLETE

_Reviewed: 2026-05-28_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
