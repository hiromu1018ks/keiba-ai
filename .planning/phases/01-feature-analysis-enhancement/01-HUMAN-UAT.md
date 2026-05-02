---
status: partial
phase: 01-feature-analysis-enhancement
source: [01-VERIFICATION.md]
started: 2026-05-02
updated: 2026-05-02
---

## Current Test

[awaiting human testing]

## Tests

### 1. フルバックテストによるlogloss/AUC維持確認 (SC-4)
expected: 新特徴量追加後のバックテストでlogloss/AUCがベースライン（既存モデル）以上であること
result: [pending]

```bash
python scripts/run_backtest.py \
  --train-start 20200101 --train-end 20231231 \
  --test-start 20240101 --test-end 20241231
```

所要時間: ~100分（DB接続必須）

## Summary

total: 1
passed: 0
issues: 0
pending: 1
skipped: 0
blocked: 0

## Gaps
