---
phase: 54-automation-reporting
plan: 01
subsystem: paper_trading
tags: [exit-code, race-progress, report-aggregator, tdd]
requires: []
provides: [exit_codes, race_progress, report_aggregator]
affects: []
tech_stack:
  added: [IntEnum, StrEnum, tempfile.mkstemp+os.replace atomic write]
  patterns: [atomic JSON write, effective_stake won+lost only, ISO week range]
key_files:
  created:
    - src/paper_trading/exit_codes.py
    - src/paper_trading/race_progress.py
    - src/paper_trading/report_aggregator.py
    - tests/test_race_progress.py
    - tests/test_report_aggregator.py
  modified: []
decisions:
  - D-17 ExitCode IntEnum (8 codes, SUCCESS=0 to SIGINT=130)
  - D-18 EXIT_SEVERITY dict for severity ordering
  - D-06 RaceProgress 5-state machine (PENDING/PROCESSING/PREDICTED/NO_BET/FAILED)
  - D-05 effective_stake = won + lost only (excludes refunded/voided)
  - D-10 bets.parquet as sole cumulative history source
  - D-11 PaperTradingReportAggregator as single aggregation engine
  - D-13 JSON output directory structure with common fields
  - D-14 data_completeness (complete/partial/no_data) in daily output
  - D-19 model identity from session_manifest in all reports
metrics:
  duration: 6m
  completed: "2026-06-06"
  tasks: 2
  tests: 22
  files: 5
---

# Phase 54 Plan 01: Foundation Classes Summary

ExitCode taxonomy, RaceProgress state machine, PaperTradingReportAggregator -- foundational building blocks for run mode orchestration and reporting.

## One-liner

ExitCode IntEnum (D-17), RaceProgress atomic state machine (D-06), PaperTradingReportAggregator with daily/weekly/target stats from bets.parquet

## Changes

### Task 1: ExitCode IntEnum and RaceProgress state machine

**Commit:** 7652610

- `src/paper_trading/exit_codes.py`: ExitCode IntEnum with 8 codes per D-17, EXIT_SEVERITY dict per D-18, determine_final_exit_code() function
- `src/paper_trading/race_progress.py`: RaceState StrEnum (5 states), RaceProgress class with atomic JSON writes (tempfile.mkstemp + os.replace + Windows PermissionError retry)
- `tests/test_race_progress.py`: 11 tests (5 ExitCode + 6 RaceProgress)

### Task 2: PaperTradingReportAggregator

**Commit:** 398ac56

- `src/paper_trading/report_aggregator.py`: PaperTradingReportAggregator class with daily/weekly/target aggregation, ISO week calculation, schema validation, model identity extraction
- `tests/test_report_aggregator.py`: 11 tests covering ROI calculation, effective_stake, pending fields, ISO week boundary, JSON output structure

## Verification

```
22 tests passed (11 race_progress + 11 report_aggregator)
ruff check: All checks passed
ruff format: All formatted
```

## Deviations from Plan

None -- plan executed exactly as written.

## Key Design Notes

- **Atomic writes:** Both RaceProgress and ReportAggregator use tempfile.mkstemp + os.replace pattern from session_manifest.py, with Windows PermissionError retry (3 retries, 100ms sleep) from reconciler.py
- **Effective stake:** D-05 compliance -- only won+lost bets contribute to ROI denominator. Refunded/voided stakes are excluded
- **ISO week boundary:** Year-end dates (e.g., 2025-12-30 in ISO week 1 of 2026) handled correctly via Jan 4-based calculation
- **Schema validation:** Old v1 schema (with "result" column, no "payout") is rejected at load time

## Self-Check: PASSED

- All 5 source/test files verified on disk
- All 3 commits verified in git log (7652610, 398ac56, dcd5388)
- 22/22 tests passing
