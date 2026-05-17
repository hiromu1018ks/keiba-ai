---
status: partial
phase: 29-etl-expansion
source: [29-VERIFICATION.md]
started: 2026-05-17
updated: 2026-05-17
---

## Current Test

[awaiting human testing — requires PostgreSQL with EveryDB2 data]

## Tests

### 1. ETL Full Extraction
expected: `python scripts/run_etl.py --mode full --start 20150101 --end 20251231` completes without error and generates Parquet files for odds_sanren, odds_umaren, odds_sanrentan
result: [pending]

### 2. Data Quality Confirmation
expected: Coverage log output shows no missing year warnings and missing rate <= 30% for all 6 new tables (odds_sanren, odds_sanren_head, odds_umaren, odds_umaren_head, odds_sanrentan, odds_sanrentan_head)
result: [pending]

## Summary

total: 2
passed: 0
issues: 0
pending: 2
skipped: 0
blocked: 0

## Gaps
