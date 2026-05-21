---
status: complete
phase: 08-win-backtest-core
source: [08-VERIFICATION.md]
started: 2026-05-05T17:34:00Z
updated: 2026-05-06T01:18:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Full Pipeline Win Backtest (--ensemble付き)
expected: `python scripts/run_backtest.py --betting-target win --ensemble --profile --report` が正常完了し、ROI/的中率/バンクロール推移が表示される
result: pass
note: "正常完了。ROI 0.0% (7 bet, 0 hit)。実行時間9,067秒。"

### 2. Walk-Forward Validation (Win Mode, --ensemble付き)
expected: `python scripts/run_wf_validation.py --betting-target win --ensemble` が完了する
result: pass
note: "正常完了（verdict: FAIL）。Pool ROI 0.0%。テスト期間ベット0件。"

## Summary

total: 2
passed: 2
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps
