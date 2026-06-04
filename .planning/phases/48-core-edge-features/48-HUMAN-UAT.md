---
status: partial
phase: 48-core-edge-features
source: [48-VERIFICATION.md]
started: 2026-06-04T12:00:00Z
updated: 2026-06-04T12:00:00Z
---

# Phase 48 Human UAT

## Current Test

[awaiting human testing]

## Tests

### 1. run_train.py End-to-End Training
expected: `python scripts/run_train.py --start 20200101 --end 20231231 --ensemble` がエラーなく完了し、8個の新特徴量を含むモデルがMLflowに記録される。track_statsがSubmodelSetに保存され、horse_features.parquetにdirt_moisture/turf_cushion列が含まれる。
result: [pending]

## Summary

total: 1
passed: 0
issues: 0
pending: 1
skipped: 0
blocked: 0

## Gaps
