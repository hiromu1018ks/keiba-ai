---
gsd_state_version: 1.0
milestone: v1.8
milestone_name: Turf Precision Calibration
status: planning
last_updated: "2026-05-21T05:42:49.162Z"
last_activity: 2026-05-21
progress:
  total_phases: 6
  completed_phases: 4
  total_plans: 10
  completed_plans: 10
  percent: 67
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-19)

**Core value:** 単勝モデルのバックテストROIを100%超えにすること
**Current focus:** Phase 37 - EV Calibration Layers

## Current Position

Phase: 37
Plan: Not started
Status: Ready to plan
Last activity: 2026-05-21

Progress: [▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░] 67% (v1.8: 4/6 phases done, Phase 37 next)

## v1.8 Roadmap

| Phase | Goal | Requirements | Status |
|-------|------|--------------|--------|
| 35. ETL Data Foundation | HaronTime/LapTime/Jyuni float64 in Parquet + POST_RACE safety | ETL-01~05 | DONE |
| 36. Feature Computation | Turf relative + conditional interactions + Haron/Lap PIT-safe | TRF-01~03, INT-01~04, HLF-01~05 | DONE |
| 36.1. HaronTime L4/LapTime Redesign | クロスレベル派生特徴量への再設計 + BT hist_features欠落修正 | RED-01~06 | DONE (352 tests) |
| 36.1.1. MarketModel & RaceQuality配線修正 | Phase36特徴量ルーティング修正 + EV Tail Calibration | RTG-01~05 | Next (URGENT) |
| 37. EV Calibration Layers | Pop band calibration + regime-surface EV correction | CAL-01~05 | Pending |
| 38. Integrated Validation | CI tests + IC positive + BT ROI 100%+ + Manifest freeze | VAL-01~06 | Pending |

## Performance Metrics

**Historical (v1.0-v1.7):**

- Total phases: 34
- Total plans: 80
- Cumulative LOC: ~24,100
- Tests: 1,540+

**BT ROI progression:**

- v1.5: 84.4% → v1.6: 85.7% → v1.7: 97.8% → v1.8 post-36.1: 77.3% (regression!) → target: 100%+

## Accumulated Context

### Roadmap Evolution

- Phase 36.1 inserted after Phase 36: HaronTime L4/LapTime Feature Redesign - クロスレベル派生特徴量への再設計 + backtest engine hist_features欠落修正 (URGENT)
- Phase 36.1.1 inserted after Phase 36.1: MarketModel & RaceQuality配線修正 — Phase36強力fundamental特徴量の全モデル一律登録でMarketModel・RaceQualityScreenerの役割崩壊を修復 (URGENT)

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
v1.8: Coarse granularity — 4 phases from 28 requirements (natural dependency ordering)
v1.8: TRF/INT/HLF combined into single Feature Computation phase (all are feature work)
v1.8 Phase 36: 22 new features across 3 categories (TRF=8, INT=3, HLF=14→12 unique + 2 race_rank)

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

### Known Issues

1. ROI 100%目標未達 (97.8%) → Phase36.1後 77.3%に悪化 — Phase 36.1.1で回復
2. Turf conservative regime unprofitable — v1.8で対応
3. Phase36特徴量のMarketModel支配 (gain share 80%+) → Phase 36.1.1で修正
4. RaceQualityScreenerが利益源レースを71.9%除外 → Phase 36.1.1で再設計
5. training_pipeline _build_race_level_features() rl_*列処理未追加
4. LapTime column names in EveryDB2 not yet verified against live schema

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260521-gks | ROI劣化修正: EV Tail Calibration無効化 + 相関ペナルティ無効化 + 高NaN特徴量削除 | 2026-05-21 | 464e0bf | [260521-gks-roi-ev-tail-calibration-nan](./quick/260521-gks-roi-ev-tail-calibration-nan/) |

## Session Continuity

Last session: 2026-05-21T05:37:40.398Z
Status: PHASE 36 COMPLETE — 2 plans, 2 waves, 275 tests passing
