---
status: partial
phase: 04-walk-forward-validation
source: [04-01-VERIFICATION.md]
started: 2026-05-03T00:00:00
updated: 2026-05-03T00:00:00
---

## Current Test

[awaiting human testing]

## Tests

### 1. Walk-Forward検証スクリプトの実行
expected: `python scripts/run_wf_validation.py` が2フォールド(2024/2025テスト)を完了し、プールROI>100%を確認できること。実行にはPostgreSQL+Parquetデータが必要、推定4時間。
result: [pending]

### 2. MLflow実験の検証
expected: 実行完了後、MLflow experiment "wf_validation" に正しいparameters/metrics/tagsが記録されていること
result: [pending]

## Summary

total: 2
passed: 0
issues: 0
pending: 2
skipped: 0
blocked: 0

## Gaps
