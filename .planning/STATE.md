---
gsd_state_version: 1.0
milestone: v1.3
milestone_name: Betting Strategy Optimization
status: executing
last_updated: "2026-05-05T12:00:00Z"
last_activity: 2026-05-05 — Phase 12 complete (2/2 plans, human_needed)
progress:
  total_phases: 3
  completed_phases: 2
  total_plans: 8
  completed_plans: 4
  percent: 50
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-04)

**Core value:** 単勝モデルのバックテストROIを100%超えにすること
**Current focus:** Phase 13 — Risk Calibration & Parameter Optimization (next up)

## Current Position

Phase: 13 of 13 (Risk Calibration & Parameter Optimization)
Plan: TBD
Status: Ready to plan
Last activity: 2026-05-05 — Phase 12 complete (2/2 plans, human_needed)

Progress: [======    ] 67%

## Performance Metrics

**Velocity (historical):**
- v1.0: 4 phases, 7 plans, ~3 sessions
- v1.1: 3 phases, 5 plans, ~2 sessions
- v1.2: 3 phases, 5 plans, ~1 session
- v1.3: Phase 11 done (2 plans), Phase 12 done (2 plans)
- Total plans completed: 21 (v1.0 + v1.1 + v1.2 + v1.3 partial)
- Average duration: ~12min/plan

**Cumulative:**
- LOC (src/): ~21,500
- Tests: 1,203
- Total features implemented: 20+ new features across 3 milestones

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Historical decisions archived in:
- .planning/milestones/v1.0-ROADMAP.md
- .planning/milestones/v1.1-ROADMAP.md

Recent decisions affecting v1.3:
- BettingOrchestrator is NOT in backtest path — target RacePredictor instead
- Only ONE new component: OddsBandFilter; all others modify existing StakeCalculator/DrawdownController/RegimeDetector
- Build order: Filters first → Sizing second → Tuning third (avoids sizing-on-wrong-bets anti-pattern)
- No new production dependencies needed (numpy/pandas/optuna sufficient)
- EV exclusion count propagated via DataFrame.attrs (gap fix in Phase 11)
- StakeCalculator now supports constructor injection for fractional_kelly, enabling regime-based dynamic sizing (Phase 12)

### Pending Todos

- バックテストROI検証(run_backtest.py実行、PostgreSQL環境必要)

### Blockers/Concerns

- Look-ahead bias risk in parameter optimization — walk-forward validation required
- Regime detector oscillation risk — hysteresis counter may need adjustment
- PostgreSQL環境が必要な検証が複数残存(WF検証、バックテスト)

## Deferred Items

Items acknowledged and deferred at milestone close on 2026-05-04:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| Validation | WF検証スクリプトの実際の実行(PostgreSQL環境必要、~4時間) | Pending | v1.0 close |
| UAT | Human UAT 3項目(01-HUMAN-UAT, 04-HUMAN-UAT, 07-UAT) | Pending | v1.1 close |

## Session Continuity

Last session: 2026-05-05
Stopped at: Phase 12 complete — ready for Phase 13
Resume file: .planning/phases/12-stake-sizing-enhancement/12-CONTEXT.md
