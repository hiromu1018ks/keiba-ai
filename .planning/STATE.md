---
gsd_state_version: 1.0
milestone: v1.8
milestone_name: Turf Precision Calibration
status: Ready to execute
last_updated: "2026-05-20T16:30:00.000Z"
last_activity: 2026-05-20
progress:
  total_phases: 5
  completed_phases: 2
  total_plans: 4
  completed_plans: 4
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-19)

**Core value:** 単勝モデルのバックテストROIを100%超えにすること
**Current focus:** Phase 36.1 — HaronTime L4/LapTime Feature Redesign (PLANNED — 2 plans)

## Current Position

Phase: 36.1 HaronTime L4/LapTime Feature Redesign (INSERTED)
Plan: 2 plans (36.1-01, 36.1-02)
Status: Ready to execute
Last activity: 2026-05-20

Progress: [▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░] 75% (v1.8: 2/4 phases done, Phase 36.1 planned)

## v1.8 Roadmap

| Phase | Goal | Requirements | Status |
|-------|------|--------------|--------|
| 35. ETL Data Foundation | HaronTime/LapTime/Jyuni float64 in Parquet + POST_RACE safety | ETL-01~05 | DONE |
| 36. Feature Computation | Turf relative + conditional interactions + Haron/Lap PIT-safe | TRF-01~03, INT-01~04, HLF-01~05 | DONE |
| 36.1. HaronTime L4/LapTime Redesign | クロスレベル派生特徴量への再設計 + BT hist_features欠落修正 | RED-01~06 | PLANNED (2 plans) |
| 37. EV Calibration Layers | Pop band calibration + regime-surface EV correction | CAL-01~05 | Blocked by 36.1 |
| 38. Integrated Validation | CI tests + IC positive + BT ROI 100%+ + Manifest freeze | VAL-01~06 | Pending |

## Performance Metrics

**Historical (v1.0-v1.7):**

- Total phases: 34
- Total plans: 80
- Cumulative LOC: ~24,100
- Tests: 1,540+

**BT ROI progression:**

- v1.5: 84.4% → v1.6: 85.7% → v1.7: 97.8% → v1.8 target: 100%+

## Accumulated Context

### Roadmap Evolution

- Phase 36.1 inserted after Phase 36: HaronTime L4/LapTime Feature Redesign - クロスレベル派生特徴量への再設計 + backtest engine hist_features欠落修正 (URGENT)

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

1. ROI 100%目標未達 (97.8%) — v1.8で対応
2. Turf conservative regime unprofitable — v1.8で対応
3. training_pipeline _build_race_level_features() rl_*列処理未追加
4. LapTime column names in EveryDB2 not yet verified against live schema

## Session Continuity

Last session: 2026-05-20T01:56:27.772Z
Status: PHASE 36 COMPLETE — 2 plans, 2 waves, 275 tests passing
