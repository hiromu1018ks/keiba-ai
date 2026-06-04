---
phase: 44-roi-bisect
reviewed: 2026-05-30T12:00:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - src/backtest/component_attribution.py
  - src/backtest/historical_bisect.py
  - src/backtest/component_attribution_report.py
  - src/backtest/templates/component_attribution_report.html
  - scripts/run_component_attribution.py
  - tests/test_component_attribution.py
  - tests/test_historical_bisect.py
findings:
  critical: 2
  warning: 6
  info: 3
  total: 11
status: issues_found
---

# Phase 44: Code Review Report

**Reviewed:** 2026-05-30T12:00:00Z
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

Reviewed Phase 44 (ROI Bisect / Component Attribution) -- 7 files implementing a post-hoc diagnostic engine, historical bisect, CLI, HTML report, and tests. Two critical bugs were found: a potential `NameError` crash when `race_diff` is empty but has columns, and a logic error in `_resolve_col` that can return the wrong variant's column. Several warnings address edge cases, test reliability issues, and defensive coding gaps.

## Critical Issues

### CR-01: NameError when race_diff is empty but has columns (bl_stake/sh_stake scope leak)

**File:** `src/backtest/component_attribution.py:331-336`
**Issue:** Variables `bl_stake` and `sh_stake` are defined inside the `if not self.race_diff.empty:` block (line 292) but referenced later inside the `if "selected_changed" in self.race_diff.columns:` block (line 326). In Pandas, an empty DataFrame (zero rows) can still have columns. If `self.race_diff` has columns (including `"selected_changed"`) but zero rows, the first `if` block is skipped (`.empty == True`), so `bl_stake` and `sh_stake` are never defined. The second `if` block passes (column exists), and lines 331/336 crash with `NameError: name 'bl_stake' is not defined`.

This is not theoretical -- `pd.DataFrame(columns=["selected_changed"])` has `.empty == True` but `"selected_changed" in df.columns == True`.

**Fix:**
```python
def attribute_bet_count_loss(self) -> dict[str, Any]:
    baseline_bet_count = 0
    shadow_bet_count = 0
    bl_stake: str | None = None  # Initialize at function scope
    sh_stake: str | None = None

    if not self.race_diff.empty:
        bl_stake = self._resolve_col(
            self.race_diff, self.baseline_name, "stake"
        )
        # ... rest unchanged
```

### CR-02: _resolve_col returns wrong variant's column as fallback

**File:** `src/backtest/component_attribution.py:870-878`
**Issue:** The `_resolve_col` method has a hardcoded `"shadow_"` fallback that ignores the `variant_name` parameter:

```python
candidates = [
    f"{variant_name}_{metric}",
    f"shadow_{metric}",  # BUG: always falls back to shadow
]
```

When `variant_name="baseline"` and the baseline column is missing but the shadow column exists, this method silently returns the shadow column. The caller then computes "baseline" metrics using shadow data, producing incorrect attribution results. This affects bet count attribution (lines 293-307) and OBF analysis (lines 366-371) where baseline metrics could be computed from shadow data.

**Fix:**
```python
@staticmethod
def _resolve_col(
    df: pd.DataFrame,
    variant_name: str,
    metric: str,
) -> str | None:
    """Resolve variant-prefixed column name in DataFrame."""
    col = f"{variant_name}_{metric}"
    if col in df.columns:
        return col
    return None
```

If callers need a broader fallback, they should implement it explicitly at the call site rather than encoding a hidden assumption in the helper.

## Warnings

### WR-01: compare_oof_metrics can call spearmanr with mismatched array lengths

**File:** `src/backtest/historical_bisect.py:172-183`
**Issue:** When `has_kakuteijyuni` is False, `y_vals` becomes an empty `pd.Series(dtype=float)` (line 169) with no index alignment to `p_vals`. After filtering with `valid`, `y_valid = y_vals[valid].values` produces an empty array while `p_valid` may have 10+ elements. The subsequent `spearmanr(p_valid, y_valid)` at line 183 receives arrays of different lengths, which raises a ValueError from scipy.

**Fix:** Guard the `valid` mask to also require non-empty `y_vals`:
```python
if not has_kakuteijyuni:
    result["current_oof"] = {"ic": 0.0, "brier": 0.0, "ece": 0.0,
                              "note": "kakuteijyuni column not available"}
    return result
```

### WR-02: Subprocess calls without `shell=False` and without `cwd` specification

**File:** `src/backtest/historical_bisect.py:93-97, 215-218`
**Issue:** `subprocess.run(["git", "tag", "-l"], ...)` and `subprocess.run(["git", "log", "v1.7..v2.0", "--oneline"], ...)` do not set `cwd`. If the working directory is not the repo root at call time, these commands may fail silently or operate on a different repository. The subprocess calls are not dangerous (list form, no shell), but the missing `cwd` makes behavior environment-dependent.

**Fix:** Pass `cwd=str(project_root)` or the known repo root:
```python
result = subprocess.run(
    ["git", "tag", "-l"],
    capture_output=True, text=True, timeout=10,
    cwd=str(Path(__file__).resolve().parent.parent.parent),
)
```

### WR-03: Tests for compare_phase_artifacts and compare_oof_metrics do not mock subprocess

**File:** `tests/test_historical_bisect.py:137, 150, 172`
**Issue:** `TestComparePhaseArtifacts` (lines 137, 150) and `TestCompareOOFMetrics` (line 172) do not `@patch("subprocess.run")`, so the `HistoricalBisect.__init__` call triggers real `git tag -l` execution. This makes tests environment-dependent: they pass in a git repo but may fail or behave differently in CI containers without git or with unexpected tags.

**Fix:** Either patch `subprocess.run` in all HistoricalBisect test classes, or patch `_detect_git_tags` to return a fixed list.

### WR-04: Hardcoded Phase 35-38 estimation in _estimate_degradation_phase ignores input data

**File:** `src/backtest/historical_bisect.py:355-395`
**Issue:** The `_estimate_degradation_phase` method returns a hardcoded string implicating Phase 35-36 regardless of the `phase_changes` data. The `phase_changes` parameter is only used to append git commit count text, but the core attribution is always the same. If git history shows zero Phase 35-36 commits, the method still blames Phase 35-36. This makes the analysis misleading.

**Fix:** Use `phase_changes` to drive the actual estimation logic, not just append supplementary text. If no Phase 35-36 commits are found, either lower confidence or report "unknown."

### WR-05: HTML template truncates estimated_degradation_phase at 80 chars without indication

**File:** `src/backtest/templates/component_attribution_report.html:369`
**Issue:** `{{ historical_result.estimated_degradation_phase[:80] }}` silently truncates the degradation phase string at 80 characters. The full estimation (which may be 300+ characters of diagnostic text) is lost in the HTML report. Users reading only the HTML report get incomplete information.

**Fix:** Use Jinja2's `truncate` filter with ellipsis, or show the full text in a `<details>` element:
```html
<td><details><summary>{{ historical_result.estimated_degradation_phase[:80] }}...</summary>
{{ historical_result.estimated_degradation_phase }}</details></td>
```

### WR-06: run_component_attribution.py swallows HistoricalBisect exceptions silently

**File:** `scripts/run_component_attribution.py:94-101`
**Issue:** The `try/except Exception` block around HistoricalBisect catches and logs a warning but continues silently. If HistoricalBisect fails due to a real bug (e.g., corrupted artifact file), the user gets only a warning and proceeds without historical context, potentially missing critical diagnostic information.

**Fix:** At minimum, log the full traceback:
```python
except Exception as e:
    logger.warning("Historical bisect skipped: %s", e, exc_info=True)
```

## Info

### IN-01: Inline scipy import inside compare_oof_metrics

**File:** `src/backtest/historical_bisect.py:182-183`
**Issue:** `from scipy.stats import spearmanr` is imported inside the function body rather than at module level. While this works, it introduces a hidden dependency and can cause a late ImportError at runtime if scipy is not installed. Other files in this project import scipy at the top level.

**Fix:** Move to module-level import.

### IN-02: Unused import `warnings` pattern in CLI script

**File:** `scripts/run_component_attribution.py:23, 26`
**Issue:** `import warnings` followed by `warnings.filterwarnings("ignore")` suppresses all warnings globally. While this is a common pattern in CLI scripts, it can hide deprecation warnings or other important signals during development.

**Fix:** Consider using `warnings.filterwarnings("ignore", category=FutureWarning)` or a more specific filter.

### IN-03: Unused `HistoricalBisectResult` import in component_attribution.py used only for type hint

**File:** `src/backtest/component_attribution.py:25`
**Issue:** `HistoricalBisectResult` is imported but only used in function signatures (`_build_bisect_summary_md`, `save_attribution_results`). This is correct and intentional for type checking, but creates a circular dependency risk: `component_attribution.py` imports from `historical_bisect.py`, while `component_attribution_report.py` imports from both. The current structure avoids actual circularity, but the coupling could be reduced by using `TYPE_CHECKING` guard.

**Fix:**
```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backtest.historical_bisect import HistoricalBisectResult
```

---

_Reviewed: 2026-05-30T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
