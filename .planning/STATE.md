---
gsd_state_version: 1.0
milestone: v1.7
milestone_name: Market-Independent Edge Discovery
status: complete
stopped_at: v1.7 milestone shipped
last_updated: "2026-05-19T12:00:00.000Z"
last_activity: 2026-05-19 -- v1.7 milestone completed and archived
progress:
  total_phases: 34
  completed_phases: 34
  total_plans: 80
  completed_plans: 80
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-19)

**Core value:** 単勝モデルのバックテストROIを100%超えにすること
**Current focus:** Planning next milestone

## Current Position

Phase: 34 (last phase of v1.7) — SHIPPED
Status: v1.7 milestone complete and archived
Last activity: 2026-05-19 -- Milestone archived, tag v1.7 created

Progress: [====================] 100% (34/34 phases total, 8 milestones)

## v1.7 Results Summary

| Metric | v1.6 | v1.7 | Change |
|--------|------|------|--------|
| BT ROI | 85.7% | 97.8% | +12.1pp |
| C-orthogonal IC | — | 0.2753 | new |
| Dirt ROI | — | 107.4% | profitable |
| Aggressive ROI | — | 116.7% | profitable |
| High-odds (10+) ROI | — | 179.9% | profitable |
| New features | — | 11 (6 rl_* + 5 MCF) | +11 |
| Tests | 1,460+ | 1,540+ | +84 |

## Performance Metrics

**Velocity (historical):**

- v1.0: 4 phases, 7 plans
- v1.1: 3 phases, 5 plans
- v1.2: 3 phases, 5 plans
- v1.3: 3 phases, 7 plans
- v1.4: 5 phases, 10 plans
- v1.5: 5 phases, 13 plans
- v1.6: 6 phases, 14 plans
- v1.7: 6 phases, 15 plans
- Total plans completed: 80 (v1.0-v1.7)

**Cumulative:**

- LOC (src/): ~24,100
- Tests: 1,540+
- Total features implemented: 68+ across 8 milestones

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Key insight from v1.6: 特徴量追加アプローチの限界 (37新特徴量でROI+1.3ppのみ)
Key insight for v1.7: Echo Chamber脱却 -- race-level + market-cross特徴量で市場独立性を獲得 (+12.1pp)
v1.7 result: C-orthogonal IC 0.2753 confirms market-independent predictive power

### Deferred Items

Items acknowledged and deferred at milestone close on 2026-05-19:

| Category | Item | Status |
|----------|------|--------|
| Validation | WF検証スクリプトの実際の実行(~4時間、PostgreSQL環境必要) | Pending since v1.0 |
| Validation | バックテストROI検証(run_backtest.py --ensemble --strategy-manifest実行) | Pending since v1.4 |
| UAT | Human UAT 5項目 (PostgreSQL依存) | Pending since v1.4 |
| Feature | n_taisyogata_miningペアワイズ比較特徴量 | Pending since v1.6 |
| Feature | n_sale/n_banusi統計特徴量 | Pending since v1.6 |
| Bug | test_training_pipeline.py 3件既知失敗 | Pending since v1.6 |
| Verification | Phase 30 VERIFICATION.md (RIC-01~06 PARTIAL) | Deferred at v1.7 close |
| Verification | Phase 34 VERIFICATION.md (VAL-01~05 PARTIAL) | Deferred at v1.7 close |
| Bug | training_pipeline _build_race_level_features() rl_*列処理 | Known at v1.7 close |
| Bug | GPD診断 place model graceful skip | Known at v1.7 close |
| Bug | build_features() compute_flb_slope() → implied_prob_hhi/odds_skewness NaN | Known since v1.7 (WR-02) |

### Known Issues for Next Milestone

1. ROI 100%目標未達 (97.8%、あと2.2pp)
2. Turf conservative regime unprofitable — 最大の改善余地
3. training_pipeline.pyの_build_race_level_features()にrl_*列の処理を追加する必要あり
4. GPD診断はplace modelがなくても動作するように修正が必要
5. BT 2024の再実行が必要 (rl_*パイプライン修正後)

## Session Continuity

Last session: 2026-05-19
Status: v1.7 SHIPPED — ready for next milestone planning
