---
status: partial
phase: 18-validation-freeze
source: [18-VERIFICATION.md]
started: 2026-05-07T12:00:00.000Z
updated: 2026-05-07T12:30:00.000Z
---

## Current Test

[partially verified — 3/6 UAT items passed via automated testing, 2 deferred to actual backtest execution]

## Tests

### 1. アンサンブルバックテストROI検証

**テストコマンド:**
```bash
# Phase 17で生成されたmanifestパスを指定
python scripts/run_backtest.py \
  --ensemble \
  --strategy-manifest data/strategy_manifest.json \
  --years 2024 2025 \
  --train-window 4
```

expected: data/validation/multi_year_validation_report.json が生成され、validation_result=PASS (ROI>100%かつ100+ベット)
result: [deferred] — バックテスト実行に~57分/年が必要。strategy_manifest.jsonがPhase 17 Optuna実行後に生成される

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
result: [deferred] — バックテスト実行に~57分が必要

### 3. manifest改ざん検知確認 (自動テストで検証済み)

**テスト結果:**
- save_strategy_manifest() → verify_strategy_manifest() 正常系: PASS
- manifest改ざん(ev_lower変更) → SHA256 mismatch ValueError: PASS
- 存在しないmanifest → FileNotFoundError: PASS

result: PASS (自動テストで確認)

### 4. evaluate_validation判定ロジック (自動テストで検証済み)

**テスト結果:**
- ROI=1.05, bets=200 → PASS: 確認
- ROI=0.89, bets=200 → FAIL: 確認
- ROI=1.05, bets=50 → FAIL (100+ベット不足): 確認

result: PASS (自動テストで確認)

### 5. generate_cause_analysis原因分析 (自動テストで検証済み)

**テスト結果:**
- 空bet_history → error返却: PASS
- サンプルデータ(3件) → odds_band_roi(4バンド), regime_roi, surface_roi, ev_diagnosis, bet_count_sufficiency: PASS
- 欠損フィールド → .get()で安全処理: PASS

result: PASS (自動テストで確認)

### 6. generate_validation_report全体レポート (自動テストで検証済み)

**テスト結果:**
- ROI>100% → validation_result=PASS, cause_analysis=None: PASS
- ROI<100% → validation_result=FAIL, cause_analysis含む(5項目): PASS

result: PASS (自動テストで確認)

## Summary

total: 6
passed: 4
issues: 0
pending: 2
skipped: 0
blocked: 0

## Gaps
