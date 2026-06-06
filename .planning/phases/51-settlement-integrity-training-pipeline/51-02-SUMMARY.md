---
phase: 51
plan: 02
subsystem: paper_trading
tags: [reconciler, settlement, roi, 3-column-state, payout-maps, D-01, D-02, D-03, D-05, D-06, D-07, D-08, D-09, D-11, D-18, D-19, D-20]
dependency_graph:
  requires: [51-01]
  provides: [PaperReconciler, compute_bet_id, _validate_bet_schema, _atomic_write_parquet, retry_pending, _compute_roi]
  affects: [src/paper_trading/reconciler.py, scripts/run_paper_trading.py, tests/test_paper_reconciler.py]
tech_stack:
  added: []
  patterns: [3-column-state-model, atomic-parquet-write, schema-validation, thin-cli-wrapper]
key_files:
  created: []
  modified:
    - src/paper_trading/reconciler.py
    - scripts/run_paper_trading.py
    - tests/test_paper_reconciler.py
decisions:
  - D-01 PaperReconciler is sole settlement implementation, _run_reconcile is thin CLI wrapper
  - D-02 bet_id = SHA256(session_id|race_id|bet_type|umaban)[:32]
  - D-03 3-column state model: settlement_status/outcome/payout
  - D-05 ROI over effective_stake (won+lost only), excluding refunded/voided
  - D-06 Retry mechanism at 60s intervals up to 600s timeout
  - D-07 Atomic Parquet write via NamedTemporaryFile + replace
  - D-08 Cumulative bets.parquet as source of truth, written at predict time
  - D-09 Shared payout_maps.py for Win/Place settlement
  - D-11 Settlement order: refund/void first, then payout map lookup
  - D-18 Old schema (result column) explicitly rejected
  - D-19 schema_version=2 column added
  - D-20 Pre-write schema validation (pending NULLs, settled non-NULLs, bet_id unique, stake>0)
metrics:
  duration: 6m 25s
  completed: 2026-06-06
  tasks_total: 2
  tasks_completed: 2
  files_created: 0
  files_modified: 3
  tests_added: 25
  tests_passed: 25
---

# Phase 51 Plan 02: PaperReconciler Overhaul Summary

Overhauled PaperReconciler with 3-column state model (settlement_status/outcome/payout), Win/Place settlement via shared payout_maps, correct ROI calculation including losses, retry mechanism, and thinned _run_reconcile to a CLI wrapper.

## Tasks Completed

| Task | Name | Status | Commit |
|------|------|--------|--------|
| 1 | Overhaul PaperReconciler with 3-column state model, settlement logic, and retry | Done | a41972b |
| 2 | Add new schema columns to _run_predict, thin _run_reconcile, update tests | Done | a41972b |

## Key Changes

### src/paper_trading/reconciler.py (OVERHAULED)
- **compute_bet_id**: SHA256(session_id|race_id|bet_type|umaban)[:32] static method (D-02)
- **_atomic_write_parquet**: NamedTemporaryFile + replace pattern for crash safety (D-07)
- **_validate_bet_schema**: Pre-write validation enforcing all D-20 constraints (old schema rejection, pending NULLs, settled non-NULLs, payout>=0, bet_id unique, stake>0, schema_version=2)
- **reconcile()**: Loads bets.parquet, filters by settlement_status="pending", builds win/place payout maps via shared payout_maps.py, settles per D-11 order (invalid payout keeps pending), atomic write
- **retry_pending()**: Loop at retry_interval (60s) up to retry_timeout (600s), returns exit_code=2 if pending remain (D-06)
- **_compute_roi()**: effective_stake = sum(stake WHERE outcome IN won/lost), excludes refunded/voided (D-05)
- Removed all references to `result == 0.0` filtering, replaced with `settlement_status == "pending"`
- Constructor extended with `retry_interval` and `retry_timeout` parameters

### scripts/run_paper_trading.py (MODIFIED)
- **_run_predict**: Session ID generation with crash recovery (D-02), bet records use new schema columns (bet_id, session_id, schema_version=2, settlement_status="pending", outcome=None, payout=None), removed `"result": 0.0`
- **_run_predict**: After predictions saved, appends new bets to cumulative bets.parquet with dedup by bet_id, schema validation, and atomic write (D-08)
- **_run_reconcile**: Thinned from ~220 lines to 58 non-blank lines. All inline settlement logic deleted (payout_map dicts, iterrows, direct Parquet writes, inline ROI). Now calls PaperReconciler.reconcile() and formats results.

### tests/test_paper_reconciler.py (REWRITTEN)
- 25 tests in 10 test classes covering:
  - TestComputeBetId (deterministic, length, different inputs)
  - TestValidateBetSchema (old schema rejection, pending/settled constraints, duplicate bet_id, zero stake, wrong schema_version)
  - TestWinSettlement (won and lost win bets)
  - TestPlaceSettlement (won and lost place bets)
  - TestROI (effective_stake excludes refunded, ROI includes losses)
  - TestOldSchemaRejection (result column raises ValueError)
  - TestInvalidPayout (zero multiplier keeps pending per D-11 item 6)
  - TestIdempotency (already-settled bets skipped)
  - TestEdgeCases (no file, no pending, no payout data)
  - TestAtomicWrite (creates file in subdirectory)

## Verification Results

| Check | Result |
|-------|--------|
| `python -m pytest tests/test_paper_reconciler.py -v` | 25/25 PASSED |
| _run_reconcile non-blank lines | 58 (< 70) |
| `"result": 0.0` not in _run_predict section | OK |
| `settlement_status` in run_paper_trading.py | OK |
| `compute_bet_id` in run_paper_trading.py | OK |
| `session_id` in run_paper_trading.py | OK |
| `schema_version` in run_paper_trading.py | OK |
| `from betting.payout_maps import` in reconciler.py | OK |
| No iterrows/payout_map in _run_reconcile | OK |

## Decisions Made

1. **race_in_payouts detection** -- Check both win_map and place_map keys for race_id existence, since some races may only have win or place payouts
2. **pending-only filtering** -- settlement_status column used instead of result==0.0, enabling correct handling of already-settled bets
3. **Loss recording** -- Bets not found in payout map (but race exists) get outcome="lost", payout=0.0, settling the critical ROI overestimation bug

## Deviations from Plan

None -- plan executed exactly as written.

## Self-Check: PASSED

- `src/paper_trading/reconciler.py` exists and imports from `betting.payout_maps`
- `tests/test_paper_reconciler.py` exists with 25 passing tests
- `scripts/run_paper_trading.py` has new schema columns in _run_predict
- Commit `a41972b` exists in git log
