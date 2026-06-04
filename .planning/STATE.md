---
gsd_state_version: 1.0
milestone: v2.3
milestone_name: Track Condition Feature Integration
status: planning
stopped_at: Phase 47 context gathered
last_updated: "2026-06-04T04:27:36.808Z"
last_activity: 2026-06-04 — Roadmap created (4 phases, 23 requirements mapped)
progress:
  total_phases: 4
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-04)

**Core value:** 単勝モデルのバックテストROIを100%超えにすること
**Current focus:** v2.3 Track Condition Feature Integration — 含水率/クッション値特徴量統合でBT ROI 97%+回復

## Current Position

Phase: 47 of 50 (ETL Data Pipeline)
Plan: —
Status: Ready to plan
Last activity: 2026-06-04 — Roadmap created (4 phases, 23 requirements mapped)

Progress: [░░░░░░░░░░] 0%

## Accumulated Context

### Decisions

- Coarse granularity: 4 phases compressed from 7 requirement categories
- Tier 1+2 combined (Phase 48): Both are direct interaction features from same data source
- Tier 3+4 combined (Phase 49): Derived features depend on Tier 1/2 being registered
- REG-01 (feature registration) moved to Phase 48: Features must be registered to be usable
- REG-02/REG-03 in Phase 50: Routing audit and POST_RACE CI validate the complete feature set

### Blockers/Concerns

- クッション値データは2020/09開始のためWF Fold0(2020学習)でNaN率高い可能性 (VLD-03で検証)

## Deferred Items

| Category | Item | Status |
|----------|------|--------|
| Validation | WF検証スクリプトの実際の実行(~4時間、PostgreSQL環境必要) | Pending since v1.0 |
| UAT | Human UAT 5項目 (PostgreSQL依存) | Pending since v1.4 |
| Bug | test_training_pipeline.py 3件既知失敗 | Pending since v1.6 |
| Cleanup | WinSegmentCalibrator dead code removal (WRN-01) | Pending since v2.1 |
| Optimization | Optuna 19次元パラメータ最適化 (DEP-02) | Deferred to v2.4+ |
| Automation | デプロイゲート自動判定 (DEP-01) | Deferred to v2.4+ |
| Calibration | Conservative MAWC redesign / selective interaction experiment | Deferred to v2.4+ |

## Session Continuity

Last session: 2026-06-04T04:27:36.790Z
Stopped at: Phase 47 context gathered
Resume file: .planning/phases/47-etl-data-pipeline/47-CONTEXT.md
