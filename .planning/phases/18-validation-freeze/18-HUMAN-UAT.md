---
status: partial
phase: 18-validation-freeze
source: [18-VERIFICATION.md]
started: 2026-05-07T12:00:00.000Z
updated: 2026-05-07T18:30:00.000Z
---

## Current Test

[partially verified — 5/6 UAT items passed, 1 pending Optuna再最適化後再検証]

## Tests

### 1. アンサンブルバックテストROI検証

**テストコマンド:**
```bash
python scripts/run_backtest.py \
  --ensemble \
  --train-start 20200101 --train-end 20231231 \
  --test-start 20240101 --test-end 20241231
```

expected: data/validation/validation_report.json が生成され、validation_result=PASS (ROI>100%かつ100+ベット)
result: **FAIL** — デフォルトパラメータでROI 83.1% (2,321ベット)。Optuna最適化後にmanifest付きで再検証が必要

**Actual results:**
- ROI: 83.1% (目標 >100%)
- ベット数: 2,321 (目標 100+ はクリア)
- 投資額: 232,100円 / 払戻額: 192,870円 / 利益: -39,230円
- 改善幅: 63.8% → 83.1% (+19.3pt)
- validation_result: "FAIL"
- cause_analysis生成あり (5項目)

**残タスク:** Optuna最適化完了後、`--strategy-manifest` 付きで再実行してROI>100%を再検証

### 2. 検証レポート内容確認

**テスト:** data/validation/validation_report.json の内容を目視確認
expected: ROI、ベット数、年別内訳、PFP検証結果、(ROI<=1.0の場合) cause_analysisが含まれる
result: **PASS**

**Actual results:**
- total_roi: 0.831 ✓
- total_bets: 2321 ✓
- yearly_breakdown.2024 (roi/bets/stake/return) ✓
- pfp_verification.passed: null, message: "PFP not used" ✓ (manifestなし実行のため正常)
- manifest.path: null ✓
- validation_result: "FAIL" (ROI<100%で正しい判定) ✓
- cause_analysis (5項目全て): odds_band_roi, regime_roi, ev_diagnosis, bet_count_sufficiency, surface_roi ✓

### 3. manifest改ざん検知確認 (自動テストで検証済み)

result: **PASS** (自動テストで確認)

### 4. evaluate_validation判定ロジック (自動テストで検証済み)

result: **PASS** (自動テストで確認)

### 5. generate_cause_analysis原因分析 (自動テストで検証済み)

result: **PASS** (自動テストで確認)

### 6. generate_validation_report全体レポート (自動テストで検証済み)

result: **PASS** (自動テストで確認)

## Summary

total: 6
passed: 5
issues: 0
pending: 1 (UAT #1: Optuna最適化後のROI再検証)
skipped: 0
blocked: 0

## Gaps

- UAT #1: Optuna 10-trial最適化をバックグラウンド実行中 (task: brj2pdt65)。完了後manifest付きバックテストで再検証予定
