---
phase: 46-quality-gate-verification
review_depth: standard
status: issues_found
reviewed_at: 2026-06-01T12:00:00Z
findings:
  critical: 0
  warning: 5
  info: 4
  total: 9
files_reviewed: 3
files_reviewed_list:
  - scripts/run_phase46_quality_gates.py
  - tests/test_phase46_quality_gates.py
  - tests/conftest.py
---

# Phase 46 Code Review

**Reviewed:** 2026-06-01T12:00:00Z
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Reviewed the Phase 46 Quality Gate Orchestration CLI (604 lines), its test suite (796 lines, 30 tests), and the conftest.py path setup. The overall architecture is sound: 2-stage flow, 3-label framework, stop-on-first-fail orchestration. Subprocess calls use `shell=False` (list-form `cmd`) correctly, preventing command injection.

Key concerns: `sys.exit(1)` calls inside class methods prevent proper error aggregation; shadow comparison subprocess return code is unchecked; several methods lack try/except around I/O and third-party calls that can fail; the `_extract_roi` method contains unreachable dead code; and global warning suppression is overly aggressive for a quality gate tool. Test suite has coverage gaps for failure paths and edge cases.

## Warnings

### WR-01: `sys.exit(1)` in class methods prevents error aggregation and testability

**File:** `scripts/run_phase46_quality_gates.py:125,129,133,186`
**Issue:** Four `sys.exit(1)` calls exist inside `QualityGateOrchestrator` methods (`_run_stage1` at lines 125/129/133, `_run_shadow_comparison` at line 186). When these methods encounter failures, they terminate the entire process instead of returning an error result. This means:
1. No JSON result file is written on failure, so downstream consumers get nothing.
2. The `_aggregate_results` method is never called for failure cases, making the 3-label framework incomplete.
3. Unit testing these failure paths requires `pytest.raises(SystemExit)` workarounds rather than asserting on return values.

The `main()` function at line 598 already correctly uses `sys.exit(1)` after aggregation -- the class methods should follow the same pattern by raising exceptions or returning error dicts and letting `main()` handle the exit.
**Fix:**
```python
def _run_stage1(self, args: argparse.Namespace) -> Path:
    # ...
    if result.returncode != 0:
        raise RuntimeError(f"Stage 1 FAILED: {result.stderr}")
    if not manifest_path.exists():
        raise RuntimeError("Stage 1 FAILED: manifest.json not created")
    if not self._check_manifest_deployed(manifest_path):
        raise RuntimeError("Stage 1 FAILED: No surfaces deployed")
    # ...

# In main(), wrap in try/except:
try:
    manifest_path = orch._run_stage1(args)
except RuntimeError as e:
    logger.error(str(e))
    stage_results["stage1"] = {"status": "FAIL", "error": str(e)}
    # proceed to aggregate and write results
```

### WR-02: Shadow comparison subprocess return code unchecked

**File:** `scripts/run_phase46_quality_gates.py:182`
**Issue:** `subprocess.run(cmd, cwd=ROOT)` on line 182 does not check `returncode`. The only validation is whether the output file exists (line 184). If the subprocess crashes with a non-zero exit code but leaves a stale/partial output file from a previous run, the orchestrator will silently treat it as success. In contrast, Stage 1's subprocess call at line 120-125 properly checks `result.returncode != 0`.
**Fix:**
```python
result = subprocess.run(cmd, cwd=ROOT)  # noqa: S603
if result.returncode != 0:
    logger.error("Shadow Comparison subprocess failed with return code %d", result.returncode)
    sys.exit(1)
```

### WR-03: `_extract_roi` contains unreachable dead code

**File:** `scripts/run_phase46_quality_gates.py:266-268`
**Issue:** The second block of `_extract_roi` (lines 266-268) is dead code. The logic is:
1. Line 263: `if variant in overall` -- returns immediately if found.
2. Line 266-268: `for key in ("baseline", "mawc_conservative", "shadow"): if key in overall and key == variant:` -- this can only return when `key == variant`, meaning `variant in overall` must be True. But line 263 already returned for that case. So this loop body is never reached.

This means `_extract_roi("baseline", ...)` and `_extract_roi("mawc_conservative", ...)` work correctly via line 263, but the function's fallback logic is misleading dead code.
**Fix:**
```python
def _extract_roi(self, shadow_result: dict[str, Any], variant: str) -> float | None:
    try:
        overall = shadow_result["overall"]["metrics"]
        if variant in overall:
            return overall[variant].get("roi")
        return None
    except (KeyError, TypeError):
        return None
```

### WR-04: `_run_shadow_diagnosis` has no exception handling for I/O failures

**File:** `scripts/run_phase46_quality_gates.py:199-201`
**Issue:** `ShadowDiagnosis(input_dir)` constructor reads three parquet files and two JSON files from disk (see `shadow_diagnosis.py:143-154`). If any of these files are missing, corrupted, or malformed, the constructor raises `FileNotFoundError`, `json.JSONDecodeError`, or `pd.errors.EmptyDataError`. None of these are caught. Similarly, `save_diagnosis_results(result, diagnosis_dir)` can raise I/O errors. The `_run_oof_validation` method at line 140 has the same issue with `pd.read_parquet(args.oof_path)`.

In a quality gate orchestrator, individual step failures should be reported in the aggregated result rather than crashing the entire process.
**Fix:**
```python
def _run_shadow_diagnosis(self, args: argparse.Namespace) -> dict[str, Any]:
    # ...
    try:
        sd = ShadowDiagnosis(input_dir)
        result = sd.run()
        save_diagnosis_results(result, diagnosis_dir)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logger.error("Shadow Diagnosis FAILED: %s", e)
        return {"status": "FAIL", "error": str(e)}
    return {"status": "PASS", "path": str(diagnosis_result_path)}
```

### WR-05: Global `warnings.filterwarnings("ignore")` suppresses all warnings

**File:** `scripts/run_phase46_quality_gates.py:43`
**Issue:** `warnings.filterwarnings("ignore")` at module level suppresses all warnings globally for the entire process. This is a quality gate verification tool whose purpose is to detect problems -- suppressing warnings from all libraries (including deprecation warnings, future warnings, and potential data pipeline warnings) is counter to its purpose. At minimum, this should be scoped to specific warning categories or moved to only surround known-noisy calls.
**Fix:**
```python
# Option 1: Remove entirely and let warnings surface
# Option 2: Scope to specific categories
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
```

## Info

### IN-01: No tests for `_extract_roi` method

**File:** `tests/test_phase46_quality_gates.py`
**Issue:** The `_extract_roi` method has no dedicated test coverage. Given that this method has dead code (WR-03) and is responsible for extracting the key ROI metric that drives the 3-label framework, it deserves direct testing. Edge cases worth testing: missing variant key, `None` ROI value, empty `shadow_result`, malformed nested structure.
**Fix:** Add test class `TestExtractRoi` with cases for found/not-found/None/malformed inputs.

### IN-02: No test for subprocess failure paths

**File:** `tests/test_phase46_quality_gates.py`
**Issue:** The test suite has no test where `subprocess.run` returns a non-zero `returncode` for Stage 1, nor for Shadow Comparison. The test at line 204 always returns `returncode=0`. This means the `sys.exit(1)` error paths in `_run_stage1` (lines 123-133) and the unchecked return code in `_run_shadow_comparison` (line 182) are untested.
**Fix:** Add tests that mock `subprocess.run` to return `returncode=1` and verify the error behavior (whether `SystemExit` is raised or error is propagated).

### IN-03: No test for ROI trend edge cases (unknown, boundary values)

**File:** `tests/test_phase46_quality_gates.py`
**Issue:** `TestComputeRoiTrend` tests three clear cases (>=90, 87.8-89.9, <87.8) but misses:
- ROI exactly at the baseline threshold (87.8) -- confirmed `weak_recovery` but not explicitly tested
- `shadow_result` with no `mawc_conservative` or `shadow` key (should return `"unknown"`)
- `shadow_result` with empty `overall` dict (should return `"unknown"`)
- `shadow_result` with `roi: None` (should return `"unknown"`)
- `shadow_result` with `roi: 0` (zero -- falsy but valid)
**Fix:** Add boundary and edge case tests to `TestComputeRoiTrend`.

### IN-04: No test for `_run_shadow_diagnosis` skip logic

**File:** `tests/test_phase46_quality_gates.py:319-343`
**Issue:** The test for `_run_shadow_diagnosis` (Test 11) always runs with the diagnosis result not pre-existing, so it tests the execution path. The skip path (when `diagnosis_result_path` already exists and `force=False`) is never tested. Similarly, the `_run_shadow_comparison` skip path is untested.
**Fix:** Add a test that pre-creates `diagnosis_result_path` and verifies `_run_shadow_diagnosis` returns `{"status": "SKIP", ...}` without calling `ShadowDiagnosis`.

---

_Reviewed: 2026-06-01T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
