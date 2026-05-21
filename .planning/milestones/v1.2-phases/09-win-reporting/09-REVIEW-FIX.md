---
phase: 09-win-reporting
fixed_at: 2026-05-04T13:30:00Z
review_path: .planning/phases/09-win-reporting/09-REVIEW.md
iteration: 1
findings_in_scope: 7
fixed: 5
skipped: 2
status: partial
---

# Phase 09: Code Review Fix Report

**Fixed at:** 2026-05-04T13:30:00Z
**Source review:** .planning/phases/09-win-reporting/09-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 7 (1 Critical + 6 Warnings)
- Fixed: 5
- Skipped: 2

## Fixed Issues

### WR-02: `monthly_returns` dict initialized but never updated (dead code)

**Files modified:** `src/backtest/engine.py`
**Commit:** 0b8acd0
**Applied fix:** Removed `monthly_returns: dict[str, float] = {}` variable initialization and replaced the reference in `BacktestResult()` constructor with inline `{}` literal.

### WR-03: `_derive_fields` doesn't validate race_id length

**Files modified:** `src/backtest/report.py`
**Commit:** ea8b668
**Applied fix:** Added length check with `rid = str(bet.get("race_id", ""))` and `if len(rid) >= 8` guard before parsing date. Short race_ids produce empty string for race_date.

### WR-04: Skipped test with outdated reason

**Files modified:** `tests/test_backtest_report.py`
**Commits:** 9c8f772, bff8da0
**Applied fix:** Removed `@pytest.mark.skip` decorator and `@patch("db.repository.DataRepository")` mock. Removed unused `mock_repo` parameter and setup. Also cleaned up the now-unused `import pandas as pd` that was only needed for the deleted mock setup. Test now runs and passes.

### WR-05: No tests for `save_ai_diagnostics`

**Files modified:** `tests/test_backtest_report.py`
**Commit:** 1cd8cb8
**Applied fix:** Added `TestSaveAiDiagnostics` class with 6 tests: win mode normal case (valid JSON with highlights), empty data (returns None), place mode (returns None), trend improving, trend declining, and trend stable.

### WR-06: `_build_race_features` and `_generate_bets` are dead code

**Files modified:** `src/backtest/engine.py`
**Commit:** 61fe0aa
**Applied fix:** Removed both methods (~87 lines total) from BacktestEngine. Verified no external references exist.

## Skipped Issues

### CR-01: KPI cards not shown for non-win modes

**File:** `src/backtest/templates/report.html:110-123`
**Reason:** Already fixed in current code. The bet count, investment, and return KPI cards (lines 110-121) are outside any `{% if betting_target == "win" %}` block. The only conditional block is at line 239 which wraps regime and odds multiplier bands (win-specific features). The reviewer's observation no longer applies to the current code state.

### WR-01: `_parse_kumi` 3-char parse ambiguity - lo/hi sort missing

**File:** `src/backtest/engine.py:165-190`
**Reason:** Already fixed in current code. Line 190 already returns `(min(lo, hi), max(lo, hi))` ensuring lo <= hi ordering. The fix suggested in the review is already present in the codebase (likely from Phase 08 fix commit `3470f57`).

---

_Fixed: 2026-05-04T13:30:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
