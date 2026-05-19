---
gsd_state_version: 1.0
milestone: v1.8
milestone_name: Turf Precision Calibration
status: planning
stopped_at:
last_updated: "2026-05-19T14:00:00.000Z"
last_activity: 2026-05-19 -- v1.8 milestone started
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-19)

**Core value:** 単勝モデルのバックテストROIを100%超えにすること
**Current focus:** v1.8 Turf Precision Calibration — 芝IC改善 + ROI 100%超え

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-05-19 -- Milestone v1.8 started

Progress: [                    ] 0% (v1.8 phases to be defined)

## v1.8 Goal

芝モデルのIC b_differenceを負から正に転換し、ROI 97.8%→100%超えを達成。

### Root Causes (3 layers)

1. 芝Stage1モデルが市場に負けている (IC b_difference = -0.004)
2. 芝中位人気(4-12番)の確率過大推定 (calibration ratio = 0.527)
3. 距離×グレードの交互効果が未学習 (芝一般戦短距離: ROI 51.5%)

### Target Features

- A: 上がりタイム+ラップ特徴量 (期待 +3~5%)
- B: 人気帯キャリブレーション (期待 +1.5~2.5%)
- C: 芝レース内相対特徴量強化 (期待 +1~2%)
- D: 条件交互作用特徴量 (期待 +2~3%)
- E: レジーム×サーフェスEV補正 (期待 +1~2%)

### Absolute Constraint

リーク・PIT安全性への最新の注意。全POST_RACE由来特徴量は過走データのみ集計。

## Performance Metrics

**Historical (v1.0-v1.7):**

- Total phases: 34
- Total plans: 80
- Cumulative LOC: ~24,100
- Tests: 1,540+

**BT ROI progression:**

- v1.0: baseline
- v1.5: 84.4%
- v1.6: 85.7% (+1.3pp)
- v1.7: 97.8% (+12.1pp)
- v1.8: target 100%+ (+2.2pp+ needed)

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
v1.7: C-orthogonal IC 0.2753 confirms market-independent predictive power
v1.8: 芝b_difference負が最大ボトルネック — 上がりタイム+ラップデータ存在確認済 (EveryDB2 RA/SE tables)

### Deferred Items

| Category | Item | Status |
|----------|------|--------|
| Validation | WF検証スクリプトの実際の実行(~4時間、PostgreSQL環境必要) | Pending since v1.0 |
| Validation | バックテストROI検証(run_backtest.py --ensemble --strategy-manifest実行) | Pending since v1.4 |
| UAT | Human UAT 5項目 (PostgreSQL依存) | Pending since v1.4 |
| Feature | n_taisyogata_miningペアワイズ比較特徴量 | Pending since v1.6 |
| Feature | n_sale/n_banusi統計特徴量 | Pending since v1.6 |
| Feature | 坂路調教タイム(37-HANRO) ETL・特徴量化 | Deferred from v1.8 |
| Bug | test_training_pipeline.py 3件既知失敗 (RecordFeatures.compute mock問題) | Pending since v1.6 |
| Verification | Phase 30/34 VERIFICATION.md不足 | Deferred at v1.7 close |

### Known Issues

1. ROI 100%目標未達 (97.8%、あと2.2pp) — v1.8で対応
2. Turf conservative regime unprofitable — v1.8で対応
3. training_pipeline _build_race_level_features() rl_*列処理未追加
4. GPD place model skip issue — v1.7で修正済だがwide model skip残存

## Session Continuity

Last session: 2026-05-19
Status: v1.8 PLANNING — requirements definition in progress
