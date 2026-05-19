---
gsd_state_version: 1.0
milestone: v1.8
milestone_name: Turf Precision Calibration
status: ready_to_plan
last_updated: "2026-05-19T11:54:23.855Z"
last_activity: 2026-05-19 -- Phase 35 execution started
progress:
  total_phases: 4
  completed_phases: 1
  total_plans: 2
  completed_plans: 0
  percent: 25
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-19)

**Core value:** 単勝モデルのバックテストROIを100%超えにすること
**Current focus:** Phase 35 — etl-data-foundation

## Current Position

Phase: 36
Plan: Not started
Status: Ready to plan
Last activity: 2026-05-19

Progress: [▓░░░░░░░░░░░░░░░░░░░░] 5% (v1.8: 0/4 phases, Phase 35 planned)

## v1.8 Roadmap

| Phase | Goal | Requirements |
|-------|------|--------------|
| 35. ETL Data Foundation | HaronTime/LapTime/Jyuni float64 in Parquet + POST_RACE safety | ETL-01~05 |
| 36. Feature Computation | Turf relative + conditional interactions + Haron/Lap PIT-safe | TRF-01~03, INT-01~04, HLF-01~05 |
| 37. EV Calibration Layers | Pop band calibration + regime-surface EV correction | CAL-01~05 |
| 38. Integrated Validation | CI tests + IC positive + BT ROI 100%+ + Manifest freeze | VAL-01~06 |

## Performance Metrics

**Historical (v1.0-v1.7):**

- Total phases: 34
- Total plans: 80
- Cumulative LOC: ~24,100
- Tests: 1,540+

**BT ROI progression:**

- v1.5: 84.4% → v1.6: 85.7% → v1.7: 97.8% → v1.8 target: 100%+

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
v1.8: Coarse granularity — 4 phases from 28 requirements (natural dependency ordering)
v1.8: TRF/INT/HLF combined into single Feature Computation phase (all are feature work)

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

1. ROI 100%目標未達 (97.8%) — v1.8で対応
2. Turf conservative regime unprofitable — v1.8で対応
3. training_pipeline _build_race_level_features() rl_*列処理未追加
4. LapTime column names in EveryDB2 not yet verified against live schema

## Session Continuity

Last session: 2026-05-19T11:00:00.000Z
Status: PHASE 35 PLANNED — 2 plans, 1 wave, verification passed
