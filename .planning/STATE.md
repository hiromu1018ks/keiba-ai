---
gsd_state_version: 1.0
milestone: v1.4
milestone_name: Ensemble Filter Recalibration
status: executing
last_updated: "2026-05-06T00:00:00Z"
last_activity: 2026-05-06 — Phase 14 executed (2/2 plans complete)
progress:
  total_phases: 5
  completed_phases: 1
  total_plans: 2
  completed_plans: 2
  percent: 20
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-05)

**Core value:** 単勝モデルのバックテストROIを100%超えにすること
**Current focus:** Phase 14 — Gate Recalibration

## Current Position

Phase: 14 of 18 (Gate Recalibration)
Plan: 2 of 2 in current phase
Status: Phase 14 executed — verification pending
Last activity: 2026-05-06 — Phase 14 executed (2/2 plans complete)

Progress: [==          ] 20%

## Performance Metrics

**Velocity (historical):**
- v1.0: 4 phases, 7 plans, ~3 sessions
- v1.1: 3 phases, 5 plans, ~2 sessions
- v1.2: 3 phases, 5 plans, ~1 session
- v1.3: 3 phases, 7 plans, ~2 sessions
- Total plans completed: 24 (v1.0 + v1.1 + v1.2 + v1.3)
- Average duration: ~12min/plan

**Cumulative:**
- LOC (src/): ~18,820
- Tests: 1,200+
- Total features implemented: 26+ new features across 4 milestones

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Historical decisions archived in:
- .planning/milestones/v1.0-ROADMAP.md
- .planning/milestones/v1.1-ROADMAP.md
- .planning/milestones/v1.3-ROADMAP.md

Recent decisions affecting current work:

- **Build order is dependency-ordered:** Gate → EV_lower → OddsBand → Optuna → Validation (from research)
- **EV_lower dynamic threshold starting value:** Data-dependent, research recommends 25th percentile of positive-edge ensemble OOF winners
- **Optuna fold count:** Must be 4+ (not 2) to prevent overfitting with 14 free parameters
- **Look-ahead bias fix:** strategy_optimizer.py must use default params for training_bet_history generation
- **Drift diagnostics integrated in pipeline:** Pipeline-integrated ks_2samp/wasserstein_distance with JSON+console output (Phase 14 D-01 to D-04)
- **use_ensemble propagation test:** Single integration test with mocks, True path only (Phase 14 D-05 to D-07)
- **Gate retraining verification:** Unit test + pipeline assertion for edge value differences (Phase 14 D-08)

### Pending Todos

- バックテストROI検証(run_backtest.py実行、PostgreSQL環境必要) — addressed by Phase 18
- Optuna最適化実行(run_strategy_optimization.py) — addressed by Phase 17
- WF検証スクリプトの実際の実行(PostgreSQL環境必要、~4時間) — deferred

### Blockers/Concerns

- Look-ahead bias in strategy_optimizer.py — must be fixed in Phase 16 before Optuna (Phase 17)
- EV_lower threshold value is data-dependent — empirical tuning needed in Phase 15
- OOF vs inference distribution shift magnitude unknown — Phase 14 diagnostics will measure this
- RegimeDetector behavior under ensemble needs verification post-recalibration

## Deferred Items

Items acknowledged and deferred at milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Validation | WF検証スクリプトの実際の実行(PostgreSQL環境必要、~4時間) | Pending | v1.0 close |
| UAT | Human UAT 3項目(01-HUMAN-UAT, 04-HUMAN-UAT, 07-UAT) | Pending | v1.1 close |
| Validation | バックテストROI検証(run_backtest.py実行) | Pending | v1.3 close |
| Validation | Optuna最適化実行(run_strategy_optimization.py) | Pending | v1.3 close |

## Session Continuity

Last session: 2026-05-06
Stopped at: Phase 14 context gathered — ready to plan
Resume file: .planning/phases/14-gate-recalibration/14-CONTEXT.md
