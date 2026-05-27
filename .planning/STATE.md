---
gsd_state_version: 1.0
milestone: v2.1
milestone_name: MarketAware Calibration + Race-Level Ranker for ROI Recovery
status: executing
stopped_at: Phase 39 Plan 02 complete
last_updated: "2026-05-27T22:13:19Z"
last_activity: 2026-05-27 -- Phase 39 Plan 39-02 completed
progress:
  total_phases: 4
  completed_phases: 0
  total_plans: 11
  completed_plans: 2
  percent: 18
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-27)

**Core value:** 単勝モデルのバックテストROIを100%超えにすること
**Current focus:** Phase 39 — marketawarewincalibrator

## Current Position

Phase: 39 (marketawarewincalibrator) — EXECUTING
Plan: 3 of 3
Status: Phase 39 Plan 02 complete, Plan 03 next
Last activity: 2026-05-27 -- Phase 39 Plan 39-02 completed

Progress: [==........] 18%

## Performance Metrics

**Velocity:**

- Total plans completed (all milestones): 84
- v2.1 plans completed: 0

**Recent Trend:**

- v2.0 (Phases 37-38): 5 plans, all complete
- Trend: Stable

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
v2.1 key decisions:

- MarketAwareWinCalibrator REPLACES WinBenterGate + WinSegmentCalibrator (not augments)
- Segment effects as regularized features/interactions in global calibrator (not per-segment coefficients)
- Race-Level Ranker is a LEARNED ranker (not deterministic weighted sum)
- Shadow comparison includes fixed-fold 2024/2025 validation
- Regime-dependent calibration is OUT OF SCOPE
- Selection agreement is a diagnostic metric, NOT a deployment gate
- generate_win_oof_predictions() returns enriched DataFrame directly (no wrapper) for MarketAwareWinCalibrator training

### Pending Todos

None for v2.1 yet.

### Blockers/Concerns

- BT ROI 87.8% at v2.0 close, target 100%+
- v1.8 ROI collapse (97.8% to 87.8%) from feature routing -- SAF-01 audit essential
- Normalization after calibration breaks probability quality -- must be addressed in Phase 39
- Zero new pip dependencies required

### Deferred Items

| Category | Item | Status |
|----------|------|--------|
| Validation | WF検証スクリプトの実際の実行(~4時間、PostgreSQL環境必要) | Pending since v1.0 |
| UAT | Human UAT 5項目 (PostgreSQL依存) | Pending since v1.4 |
| Bug | test_training_pipeline.py 3件既知失敗 | Pending since v1.6 |
| Pipeline | _build_race_level_features() rl_*列処理 | Pending since v1.7 |
| Validation | 芝IC b_difference正転換 (VAL-02~06) | Deferred to v2.1+ |
| Feature | コーナー通過順位展開特徴量 (HLF-06~08) | Future |
| Feature | E-correction fundamental activation (EFA-01~03) | Future |
| Feature | 坂路調教タイムETL (TDF-01) | Future |
| Optimization | Optuna 19次元パラメータ最適化 (DEP-02) | Deferred to v2.2 |
| Automation | デプロイゲート自動判定 (DEP-01) | Deferred to v2.2 |

## Session Continuity

Last session: 2026-05-27T22:13:19Z
Stopped at: Phase 39 Plan 02 complete
Resume file: .planning/phases/39-marketawarewincalibrator/39-03-PLAN.md
