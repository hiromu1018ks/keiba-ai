---
phase: 51-settlement-integrity-training-pipeline
fixed_at: 2026-06-06T15:00:00Z
review_path: .planning/phases/51-settlement-integrity-training-pipeline/51-REVIEW.md
iteration: 1
findings_in_scope: 8
fixed: 8
skipped: 0
status: all_fixed
---

# Phase 51: Code Review Fix Report

**Fixed at:** 2026-06-06T15:00:00Z
**Source review:** `.planning/phases/51-settlement-integrity-training-pipeline/51-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 8 (2 Critical, 6 Warning; WR-02 skipped per instructions)
- Fixed: 8
- Skipped: 0

## Fixed Issues

### CR-01: PaperReconciler silently mishandles wide bets

**Files modified:** `src/paper_trading/reconciler.py`, `tests/test_paper_reconciler.py`
**Commit:** `3e2fb4c`
**Applied fix:** Imported `build_wide_payout_map`, constructed `wide_map` in `reconcile()`, and added full wide bet settlement logic using `(race_id, lo, hi)` lookup with `umaban_b`. Wide bets missing `umaban_b` are kept pending with a warning log instead of being silently lost. Updated `_make_payouts_df` test helper to include `paywidekumi`/`paywidepay` columns.

### CR-02: Atomic write on Windows may crash when target file is open

**Files modified:** `src/paper_trading/reconciler.py`
**Commit:** `c8436f4`
**Applied fix:** Replaced `Path.replace()` with `os.replace()` wrapped in a retry loop (3 attempts, 0.1s sleep between retries) to handle Windows `PermissionError` when the target file is held open by another process.

### WR-01: n_refunded and n_voided counters never incremented

**Files modified:** `src/paper_trading/reconciler.py`
**Commit:** `ced4d62`
**Applied fix:** Removed dead `n_refunded` and `n_voided` local variables from `reconcile()` and their arguments from the `_compute_roi()` call. `_compute_roi` already computes these from the DataFrame directly.

### WR-03: Missing boundary tests for wide kumi parsing

**Files modified:** `tests/test_payout_maps.py`
**Commit:** `4204c98`
**Applied fix:** Added 4 boundary test cases: `"118"` -> (8,11), `"181"` -> (1,18), `"918"` -> (9,18), `"109"` -> (9,10). Validates the `first_two <= 18` heuristic at the boundary where first_two equals 18.

### WR-04: compute_bet_id cannot produce unique IDs for wide bets

**Files modified:** `src/paper_trading/reconciler.py`
**Commit:** `c560789`
**Applied fix:** Added optional `umaban_b: int | None = None` parameter to `compute_bet_id()`. When provided, `umaban_b` is appended to the hash input, ensuring wide bets on different horse pairs produce unique IDs. Backward compatible (existing callers pass only 4 args).

### WR-05: PaperReconciler store parameter unused

**Files modified:** `src/paper_trading/reconciler.py`, `tests/test_paper_reconciler.py`, `scripts/run_paper_trading.py`
**Commit:** `9dee050`
**Applied fix:** Removed the `store` parameter from `PaperReconciler.__init__()` and removed `self.store` assignment. Updated all callers in test files and `run_paper_trading.py`. Removed unused `MagicMock` import from `_run_reconcile`.

### WR-06: race_date type mismatch between predict and reconcile

**Files modified:** `scripts/run_paper_trading.py`
**Commit:** `ef2eac5`
**Applied fix:** Changed `"race_date": ymd` (string) to `"race_date": pd.Timestamp(ymd)` in `_run_predict` bet records, ensuring consistent `pd.Timestamp` comparison in `reconcile()`.

---

_Fixed: 2026-06-06T15:00:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
