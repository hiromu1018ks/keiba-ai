---
status: partial
phase: 18-validation-freeze
source: [18-VERIFICATION.md]
started: 2026-05-07T12:00:00.000Z
updated: 2026-05-07T12:00:00.000Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. アンサンブルバックテストROI検証

**テストコマンド:**
```bash
# PostgreSQL環境が必要(EveryDB2)
# Phase 17で生成されたmanifestパスを指定
python scripts/run_backtest.py \
  --ensemble \
  --strategy-manifest data/strategy_manifest.json \
  --years 2024 2025 \
  --train-window 4
```

expected: data/validation/multi_year_validation_report.json が生成され、validation_result=PASS (ROI>100%かつ100+ベット)
result: [pending]

### 2. 単年度バリデーションレポート確認

**テストコマンド:**
```bash
python scripts/run_backtest.py \
  --ensemble \
  --strategy-manifest data/strategy_manifest.json \
  --train-start 20200101 --train-end 20231231 \
  --test-start 20240101 --test-end 20241231
```

expected: data/validation/validation_report.json が生成され、PFP verification passed が含まれる
result: [pending]

### 3. manifest改ざん検知確認

**手順:**
1. data/strategy_manifest.json を開き、パラメータを1つ変更する
2. バックテストを再実行する
3. SHA256不一致でValueErrorが送出されることを確認

expected: SHA256 mismatch エラーでバックテストが即時停止する
result: [pending]

## Summary

total: 3
passed: 0
issues: 0
pending: 3
skipped: 0
blocked: 0

## Gaps
