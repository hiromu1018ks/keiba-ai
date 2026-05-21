---
phase: 18-validation-freeze
reviewed: 2026-05-07T12:00:00Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - scripts/run_backtest.py
  - src/backtest/engine.py
  - src/backtest/validation_report.py
  - tests/test_backtest_engine.py
  - tests/test_backtest_validation.py
findings:
  critical: 1
  warning: 5
  info: 4
  total: 10
status: issues_found
---

# Phase 18: Code Review Report

**Reviewed:** 2026-05-07T12:00:00Z
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found

## Summary

Reviewed 5 files comprising the Phase 18 validation/freeze changes: BacktestEngine PFP integration, new validation_report module, run_backtest.py manifest handling, and corresponding test suites. Found 1 critical bug (hardcoded `sha256_verified=True` without actual verification), 5 warnings (logic errors, defensive gaps, error handling), and 4 info items (unused variables, magic numbers, dead code). The core PFP dual-verification protocol in engine.py is sound, but validation_report.py introduces misleading data by claiming SHA256 verification passed without performing any hash check.

## Critical Issues

### CR-01: sha256_verified hardcoded to True without performing actual SHA256 verification

**File:** `src/backtest/validation_report.py:69`
**Issue:** When `manifest_path is not None`, the code unconditionally sets `"sha256_verified": True`. However, `generate_validation_report()` never reads the manifest file or computes its SHA256 hash. The actual SHA256 verification (`verify_strategy_manifest`) happens in `engine.py` at line 520, but its result is not passed to `generate_validation_report()`. The validation report therefore records `sha256_verified: True` even if the manifest was tampered with between the engine verify call and the report generation. This is a data integrity issue -- the report falsely claims verification passed.

Furthermore, `"sha256_hash"` is always set to `None` (line 70), meaning no hash is recorded in the report even when a manifest is present.

**Fix:**

```python
# In engine.py, around line 1264, capture the SHA256 hash from verify_strategy_manifest:
# (Before run() body, save the manifest hash)
manifest_sha256: str | None = None
if self._manifest_path is not None:
    # verify_strategy_manifest already called at line 520
    # Re-read the manifest to get the hash for reporting
    import json as _json_module
    try:
        _manifest_data = _json_module.loads(self._manifest_path.read_text(encoding="utf-8"))
        manifest_sha256 = _manifest_data.get("sha256")
    except Exception:
        pass

# Then pass it:
report = generate_validation_report(
    result=backtest_result,
    test_start=test_start,
    test_end=test_end,
    train_start=train_start_val,
    train_end=train_end_val,
    manifest_path=self._manifest_path,
    manifest_sha256=manifest_sha256,
    pfp_result=pfp_result,
)

# In validation_report.py, update generate_validation_report signature:
def generate_validation_report(
    result: object,
    test_start: str,
    test_end: str,
    train_start: str,
    train_end: str,
    manifest_path: Path | None = None,
    manifest_sha256: str | None = None,
    pfp_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # ...
    manifest_info: dict[str, Any] = {
        "path": str(manifest_path) if manifest_path is not None else None,
        "sha256_verified": manifest_sha256 is not None,
        "sha256_hash": manifest_sha256,
    }
```

## Warnings

### WR-01: generate_cause_analysis misclassifies odds=0 bets into "1.0-2.0" band

**File:** `src/backtest/validation_report.py:146`
**Issue:** When a bet record lacks both `final_odds` and `odds` keys, the fallback is `0` (line 143: `odds = b.get("final_odds", b.get("odds", 0))`). An odds of `0` satisfies `odds <= 2.0` and gets classified into the `"1.0-2.0"` band, polluting that band's ROI statistics with non-betting data. Zero-odds entries should be excluded or handled separately.

**Fix:**

```python
for b in bet_history:
    odds = b.get("final_odds", b.get("odds", 0))
    if odds <= 0:
        continue  # Skip entries without valid odds
    stake = float(b.get("stake", 0))
    # ... rest of classification
```

### WR-02: generate_validation_report uses getattr instead of typed access on BacktestResult

**File:** `src/backtest/validation_report.py:60-64`
**Issue:** The `result` parameter is typed as `object` and accessed via `getattr()` with default fallbacks. While this works, if the caller passes a malformed object (e.g., missing `total_roi`), `getattr` returns `0.0` silently, and the validation report would show `total_roi=0.0` instead of raising an error. This defeats the purpose of validation. The function should either type `result` as `BacktestResult` or validate that required attributes exist.

**Fix:**

```python
# Either type properly:
def generate_validation_report(
    result: BacktestResult,  # not object
    ...
) -> dict[str, Any]:
    total_bets = result.total_bets
    total_stake = result.total_stake
    # ... direct attribute access

# Or validate:
total_bets: int = getattr(result, "total_bets", None)
if total_bets is None:
    raise ValueError("result must have 'total_bets' attribute")
```

### WR-03: Multi-year validation report silently swallows all errors

**File:** `scripts/run_backtest.py:759`
**Issue:** The multi-year validation report generation at lines 722-760 wraps the entire block in a bare `except Exception`. If the report generation fails for any reason (corrupted data, disk full, schema mismatch), the user sees only a generic warning and the validation report is silently not generated. This is a significant oversight in a "validation freeze" phase -- the report IS the validation artifact.

**Fix:** At minimum, log the full traceback:

```python
except Exception as e:
    logger.warning("マルチ年度検証レポート生成失敗: %s", e, exc_info=True)
```

Better: consider re-raising for critical failures (e.g., `TypeError`, `AttributeError` indicating a code bug rather than a data issue).

### WR-04: _compute_yearly_breakdown only counts positive result values, contradicting engine.py sum

**File:** `src/backtest/validation_report.py:270`
**Issue:** Line 270 has `if result_val > 0: yearly[year]["return"] += result_val`. This means negative or zero results are excluded from the yearly return sum. However, in `engine.py` line 1172, `total_return` is computed as `sum(b["result"] for b in bet_history if b["result"] > 0)` -- the same pattern. But the validation report's yearly breakdown ROI values will not sum to `total_roi` if any bet has a `result` that is negative (which should not happen for payout-style results) OR if `result_val` is exactly 0.0 (lost bets). For lost bets, `result_val=0` means they do not contribute to the return sum, which is correct. However, the asymmetry with how `total_return` is computed in BacktestResult vs. yearly breakdown should be noted -- they happen to match only because both filter `> 0`. If the semantics change in one place but not the other, ROI will silently disagree.

**Fix:** This is a latent consistency risk. Add a comment documenting the invariant:

```python
# Note: Only positive returns are summed, consistent with
# BacktestResult.total_return computation in engine.py line 1172.
if result_val > 0:
    yearly[year]["return"] += result_val
```

### WR-05: PFP verify() called twice in run() -- _verify_pfp() + inline verify

**File:** `src/backtest/engine.py:496,1220`
**Issue:** The `_verify_pfp()` helper method (line 490-499) is called on early-return paths (lines 536, 562, 569, 579), but on the normal exit path (line 1218-1223), the PFP verify is done inline with slightly different logic (captures `pfp_result` for the report). This means PFP `verify()` is called potentially multiple times: once per early return path via `_verify_pfp()`, and once at the end inline. The `_verify_pfp()` helper and the inline verify at line 1220 are logically identical, creating a maintenance risk where one path could be updated without the other.

**Fix:** Unify by having `_verify_pfp()` return the pfp_result dict, and use it consistently:

```python
def _verify_pfp(self) -> dict[str, Any] | None:
    """D-03(2): PFP verify. Returns result dict or None."""
    if self._pfp is not None:
        pfp_result = self._pfp.verify()
        if not pfp_result["passed"]:
            raise RuntimeError(pfp_result["message"])
        logger.info("PFP verification passed: %s", pfp_result["message"])
        return pfp_result
    return None
```

Then replace both the early-return calls and the inline verify at line 1218 with `pfp_result = self._verify_pfp()`.

## Info

### IN-01: strategy_params argument in _collect_training_bet_history is unused

**File:** `scripts/run_backtest.py:218`
**Issue:** The `strategy_params` parameter is accepted and documented as "呼び出し側インターフェース互換のために保持するが、関数内部では使用しない". While documented, this dead parameter increases cognitive load and could mislead future callers into thinking their passed `strategy_params` affect the training bet history.

**Fix:** Remove the parameter or deprecate with a clear `_` prefix:

```python
def _collect_training_bet_history(
    models: Any,
    store: Any,
    train_start: str,
    train_end: str,
    betting_mode: str,
    betting_target: str,
    _strategy_params: dict[str, Any] | None = None,  # Unused, kept for interface compat
) -> list[dict[str, Any]]:
```

### IN-02: Magic number 100 in evaluate_validation and generate_validation_report

**File:** `src/backtest/validation_report.py:32,105`
**Issue:** The minimum bet count threshold of `100` appears as a magic number in both `evaluate_validation()` (line 32) and the `roi` dict (line 105: `"target_bets": 100`). This should be a named constant for maintainability.

**Fix:**

```python
MIN_BET_COUNT_THRESHOLD = 100

def evaluate_validation(roi: float, total_bets: int) -> str:
    if roi > 1.0 and total_bets >= MIN_BET_COUNT_THRESHOLD:
        return "PASS"
    return "FAIL"
```

### IN-03: Duplicate ROI evaluation logic in generate_validation_report and evaluate_validation

**File:** `src/backtest/validation_report.py:81,91`
**Issue:** The ROI pass/fail logic is computed twice: once at line 81 (`roi_passed = total_roi > 1.0 and total_bets >= 100`) and again via the call to `evaluate_validation()` at line 91 (`validation_result = evaluate_validation(total_roi, total_bets)`). These produce the same boolean vs. string but are redundant.

**Fix:** Compute once and derive:

```python
validation_result = evaluate_validation(total_roi, total_bets)
roi_passed = validation_result == "PASS"
```

### IN-04: Broad except in engine.py validation report generation

**File:** `src/backtest/engine.py:1284`
**Issue:** The `except Exception` at line 1284 silently swallows validation report generation errors. While this is intentionally non-fatal (the backtest result is still returned), it could hide issues like missing imports or schema mismatches that should be caught during development.

**Fix:** At minimum, add `exc_info=True` for debuggability:

```python
except Exception as e:
    logger.warning("Validation report generation failed: %s", e, exc_info=True)
```

---

_Reviewed: 2026-05-07T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
