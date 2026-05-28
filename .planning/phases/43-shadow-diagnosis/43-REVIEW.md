---
phase: 43-shadow-diagnosis
reviewed: 2026-05-29T12:00:00Z
depth: standard
files_reviewed: 4
files_reviewed_list:
  - src/backtest/shadow_diagnosis.py
  - scripts/run_shadow_diagnosis.py
  - src/backtest/templates/shadow_diagnosis_report.html
  - tests/test_shadow_diagnosis.py
findings:
  critical: 2
  warning: 4
  info: 3
  total: 9
status: issues_found
---

# Phase 43: Code Review Report

**Reviewed:** 2026-05-29T12:00:00Z
**Depth:** standard
**Files Reviewed:** 4
**Status:** issues_found

## Summary

Reviewed all 4 source files for Phase 43 (Shadow Diagnosis). The implementation reads Phase 41 shadow comparison artifacts and performs a 3-step diagnostic analysis. The overall structure is sound, but there are two critical bugs: one in missing-inputs tracking that incorrectly reports `closing_win_odds` as missing even after successful fallback merge, and one where a pandas merge on `selected_changed` will crash with a `KeyError` if that column already exists in `horse_diff`. Several warnings relate to robustness gaps and inconsistent metric semantics in Step 2.

## Critical Issues

### CR-01: False positive in missing_inputs for closing_win_odds after successful merge

**File:** `src/backtest/shadow_diagnosis.py:457-459`
**Issue:** When `horse_diff` lacks `closing_win_odds` but the code successfully falls back to merging `baseline_tanodds` from `race_diff` (lines 436-456), the check on line 457 still evaluates `self.horse_diff` (the original, unmodified DataFrame) rather than `horse_work` (the copy that was enriched by the merge). This means `closing_win_odds` gets appended to `self.missing_inputs` even though the data was successfully recovered and `odds_band` was computed. The downstream consumer of `missing_inputs` (JSON report, HTML report, CLI summary) will show a misleading "Missing Inputs: closing_win_odds" when the column was actually used successfully.

**Fix:**
```python
# Line 457: check the working copy, not the original
if "closing_win_odds" not in horse_work.columns:
    if "closing_win_odds" not in self.missing_inputs:
        self.missing_inputs.append("closing_win_odds")
```

Alternatively, track whether `odds_band` was successfully created and only add to `missing_inputs` when it was not.

### CR-02: merge on selected_changed will crash if column already exists in horse_diff

**File:** `src/backtest/shadow_diagnosis.py:474-481`
**Issue:** The code unconditionally merges `selected_changed` from `race_diff` into `horse_work`. If `horse_diff` already contains a `selected_changed` column (a plausible scenario since Phase 41 might have joined it earlier), the `merge` call on line 478 will produce columns named `selected_changed_x` and `selected_changed_y` instead of `selected_changed`. The subsequent `.map(...)` on line 479 will then try to access `horse_work["selected_changed"]`, which no longer exists, raising a `KeyError`.

**Fix:**
```python
# --- selected_changed ---
if not self.race_diff.empty and "selected_changed" in self.race_diff.columns:
    if "selected_changed" in horse_work.columns:
        horse_work = horse_work.drop(columns=["selected_changed"])
    sc_lookup = self.race_diff[["race_id", "selected_changed"]].drop_duplicates(
        subset=["race_id"]
    )
    horse_work = horse_work.merge(sc_lookup, on="race_id", how="left")
    horse_work["selected_changed"] = horse_work["selected_changed"].map(
        {True: "changed", False: "unchanged"}
    )
```

## Warnings

### WR-01: Step 2 _compute_group_metrics always uses baseline columns for both groups

**File:** `src/backtest/shadow_diagnosis.py:276-333`
**Issue:** The `_compute_group_metrics` method hardcodes `baseline_stake`, `baseline_result`, `baseline_tanodds`, and `{baseline_name}_p_win_final` for both the "changed" and "unchanged" groups. This means `changed_metrics.roi` measures baseline ROI in races where the shadow disagreed, and `unchanged_metrics.roi` measures baseline ROI in races where the shadow agreed. While this is a valid diagnostic question ("how does baseline perform when shadow disagrees?"), the `SelectionGroupMetrics` dataclass has no field indicating which variant was measured. A consumer reading `step2.changed.roi` could reasonably assume it reflects shadow performance in changed races. The semantics are ambiguous and undocumented.

**Fix:** Either (a) rename the metrics to clarify they are baseline-only (e.g., `baseline_roi_in_changed_races`), or (b) compute both baseline and shadow metrics per group and expose both. At minimum, add a docstring to `SelectionPatternResult` clarifying that all metrics are baseline-only.

### WR-02: No error handling for missing Phase 41 artifact files

**File:** `src/backtest/shadow_diagnosis.py:140-151`
**Issue:** The `__init__` method reads four files directly with `json.loads(...read_text())` and `pd.read_parquet(...)` without try/except. If any file is missing or malformed, the user gets an unhandled `FileNotFoundError`, `json.JSONDecodeError`, or parquet error with no guidance about what is wrong. The CLI script has no top-level error handling either.

**Fix:** Wrap file loading in try/except with descriptive error messages:
```python
try:
    self.result_json = json.loads(
        (input_dir / "shadow_comparison_result.json").read_text(encoding="utf-8")
    )
except FileNotFoundError:
    raise FileNotFoundError(
        f"Phase 41 artifact not found: {input_dir / 'shadow_comparison_result.json'}. "
        "Run shadow comparison first."
    )
```

### WR-03: _add_segment_columns mutates missing_inputs after _detect_missing_inputs has already run

**File:** `src/backtest/shadow_diagnosis.py:457-471`
**Issue:** `_detect_missing_inputs` (called in `__init__`) populates `self.missing_inputs` based on column availability at init time. Then `_add_segment_columns` (called in `_step3_calibration_by_segment`, which is called in `run()`) can append additional items to `self.missing_inputs`. This means calling `run()` has a side effect on the instance state. If `run()` is called twice, duplicates could accumulate (the `if col not in self.missing_inputs` guards prevent this, but the pattern is fragile). More importantly, the split responsibility for tracking missing inputs between two methods makes the logic hard to follow and error-prone.

**Fix:** Move all missing-inputs detection into `_detect_missing_inputs`, or make `_add_segment_columns` return a separate list and merge it in `run()`.

### WR-04: HTML template uses loop.previtem which is not a standard Jinja2 built-in

**File:** `src/backtest/templates/shadow_diagnosis_report.html:153`
**Issue:** Line 153 uses `loop.previtem.segment_name`, which is a Jinja2 extension (`do`/`loop` extension) that requires `jinja2.ext.loopcontrols` or similar. In standard Jinja2, `loop.previtem` is available only if `jinja2.ext.loopcontrols` is loaded, or in Jinja2 3.1.x+ where `previtem`/`nextitem` are available by default. If the project uses an older Jinja2 version, this will raise `UndefinedError`.

**Fix:** Either ensure Jinja2 >= 3.1 is pinned in dependencies, or rewrite the template logic to avoid `loop.previtem`:
```html
{% set current_name = "" %}
{% for seg in step3_segments %}
{% if seg.segment_name != current_name %}
{% if not loop.first %}
</table>
{% endif %}
<h3>{{ seg.segment_name }}</h3>
<table>...</table>
{% set current_name = seg.segment_name %}
{% endif %}
```

## Info

### IN-01: Debug print statements in CLI script

**File:** `scripts/run_shadow_diagnosis.py:93-152`
**Issue:** The `main()` function contains 20+ `print()` statements for summary output. While this is intentional CLI behavior (stdout summary), using `logging.info` with a console handler would be more consistent with the project's logging pattern (line 35-39 sets up `logging.basicConfig`). The `print()` calls bypass the logging framework.

**Fix:** Consider using `logger.info()` for structured output and reserving `print()` only for machine-readable output (JSON).

### IN-02: Test for missing inputs does not assert all expected missing columns

**File:** `tests/test_shadow_diagnosis.py:305-318`
**Issue:** `test_missing_inputs_detection` creates a horse_df without `popularity`, `surface`, and `tanodds`/`closing_win_odds`, but only asserts that `popularity` and `surface` are in `missing_inputs`. It does not check whether `tanodds` or `closing_win_odds` are also reported as missing. Given CR-01 (false positive bug), the test would pass even with the bug because it never asserts the absence of `closing_win_odds` in `missing_inputs`.

**Fix:** Add explicit assertions:
```python
# Also verify odds-related missing columns
assert "tanodds" in result.missing_inputs
```

### IN-03: `subprocess` import in test file is unused for non-CLI tests

**File:** `tests/test_shadow_diagnosis.py:12`
**Issue:** `import subprocess` is only used in `TestCLIDryRun.test_cli_dry_run`. The import at module level is fine but worth noting that only one of 12 tests uses it.

**Fix:** No action needed; module-level import is acceptable for a test file.

---

_Reviewed: 2026-05-29T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
