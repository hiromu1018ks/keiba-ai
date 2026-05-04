---
phase: 08-win-backtest-core
fixed_at: 2026-05-04T12:30:00Z
review_path: .planning/phases/08-win-backtest-core/08-REVIEW.md
iteration: 1
findings_in_scope: 6
fixed: 6
skipped: 0
status: all_fixed
---

# Phase 8: Code Review Fix Report

**Fixed at:** 2026-05-04T12:30:00Z
**Source review:** .planning/phases/08-win-backtest-core/08-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 6
- Fixed: 6
- Skipped: 0

## Fixed Issues

### CR-01: `_parse_kumi` mishandles 3-digit kumi when first horse has 2 digits

**Files modified:** `src/backtest/engine.py`
**Commit:** 3470f57
**Applied fix:** Rewrote `_parse_kumi` to try both possible splits (X|YZ and XY|Z) for 3-character kumi strings, validating each part against the valid horse number range (1-18). Added `min()/max()` sorting to guarantee `lo <= hi` in the return value, which also resolves WR-05.

### WR-01: `get_payouts()` n_harai fallback SQL missing win payout columns

**Files modified:** `src/db/everydb2_queries.py`
**Commit:** 33304c0
**Applied fix:** Added `paytansyoumaban1, paytansyopay1` to the `n_harai` SELECT query so that `build_win_payout_map()` can find win payout data even when falling back from `s_harai`.

### WR-02: `_load_cached_models` accesses private method `_load_from_local`

**Files modified:** `src/db/model_loader.py`, `scripts/run_backtest.py`
**Commit:** cfecc94
**Applied fix:** Renamed `_load_from_local` to `load_from_dir` and made it a public method with a proper docstring. Updated the internal call from `load()` and the external call from `run_backtest.py` to use the new public name.

### WR-03: `display_single_year_result` uses wrong field for average win odds

**Files modified:** `scripts/run_backtest.py`
**Commit:** ae83288
**Applied fix:** Changed the average odds calculation from `b.get("tanoddslow", 0)` to `b.get("final_odds", b.get("odds", 0))` so the display reflects actual settlement odds rather than pre-race odds.

### WR-04: Hardcoded `before_roi = 0.638` benchmark value

**Files modified:** `scripts/run_backtest.py`
**Commit:** 3f07e4e
**Applied fix:** Extracted `0.638` to a module-level constant `BASELINE_ROI` with a comment documenting its origin (Phase 7 backtest, 2024 test, place mode, flat betting). Replaced the hardcoded value in `display_single_year_result` with a reference to the constant.

### WR-05: `wide_payout_map` key ordering does not guarantee `lo < hi`

**Files modified:** `src/backtest/engine.py`
**Commit:** 3470f57 (same commit as CR-01)
**Applied fix:** Resolved as part of CR-01 fix. The new `_parse_kumi` always returns `(min(lo, hi), max(lo, hi))`, which guarantees the key ordering matches the wide odds column convention.

## Verification

- All 1141 tests passed (2 skipped, 0 failures) after all fixes were applied
- Python syntax checks passed for all modified files
- No uncommitted changes remain

---

_Fixed: 2026-05-04T12:30:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
