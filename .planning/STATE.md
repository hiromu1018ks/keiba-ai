---
gsd_state_version: 1.0
milestone: v2.2
milestone_name: ROI Recovery Analysis
status: Closed -- not_deployable
last_updated: "2026-06-02T00:00:00Z"
last_activity: 2026-06-02 — v2.2 runtime verification closed; conservative MAWC rejected
progress:
  total_phases: 4
  completed_phases: 4
  total_plans: 6
  completed_plans: 6
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-28)

**Core value:** 単勝モデルのバックテストROIを100%超えにすること
**Current focus:** v2.2 closed; next milestone should decide v2.3 direction

## Current Position

Phase: v2.2 closed after Phase 46 runtime verification
Previous: Phase 46 runtime verification complete
Status: Closed -- conservative MAWC failed deployment gates and is not deployable
Last activity: 2026-06-02 -- Archiving v2.2 as not_deployable

Progress: [██████████] 100%

## v2.2 Final Runtime Result

| Item | Result |
|------|--------|
| Candidate | MAWC conservative variant (36-dim, C=0.003, shadow_only) |
| Quality Gate | FAIL |
| Deployment | not_deployable |
| Baseline test ROI | -8.0% |
| Conservative test ROI | -11.3% |
| Decision | Do not replace baseline MAWC |

The original 51-dim MAWC remains the baseline path. The conservative variant is kept only as an experimental artifact under `data/models-backtest-mawc-conservative/`.

## Deferred Items

| Category | Item | Status |
|----------|------|--------|
| Validation | WF検証スクリプトの実際の実行(~4時間、PostgreSQL環境必要) | Pending since v1.0 |
| UAT | Human UAT 5項目 (PostgreSQL依存) | Pending since v1.4 |
| Bug | test_training_pipeline.py 3件既知失敗 | Pending since v1.6 |
| Cleanup | WinSegmentCalibrator dead code removal (WRN-01) | Pending since v2.1 |
| Optimization | Optuna 19次元パラメータ最適化 (DEP-02) | Deferred to v2.3+ |
| Automation | デプロイゲート自動判定 (DEP-01) | Deferred to v2.3+ |
| Calibration | Conservative MAWC redesign / selective interaction experiment | Deferred to v2.3+ |

## Session Continuity

Last session: 2026-06-02T00:00:00Z
Resume file: .planning/v2.2-MILESTONE-SUMMARY.md
