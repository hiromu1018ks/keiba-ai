---
gsd_state_version: 1.0
milestone: v2.1
milestone_name: MarketAware Calibration + Race-Level Ranker
status: executing
stopped_at: context exhaustion at 75% (2026-05-28)
last_updated: "2026-05-28T12:54:16.679Z"
last_activity: 2026-05-28
progress:
  total_phases: 4
  completed_phases: 4
  total_plans: 11
  completed_plans: 11
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-27)

**Core value:** 単勝モデルのバックテストROIを100%超えにすること
**Current focus:** Phase 42 — feature-routing-audit-safety-gates

## Current Position

Phase: 42
Plan: 03 complete (SAF-03 Deployment Gate Evaluator)
Status: Executing — Plan 03 done, Phase 42 complete
Last activity: 2026-05-28

Progress: [=========] 100%

## Completed Plans

| Plan | Name | Commit |
|------|------|--------|
| 42-01 | Feature Routing Audit (SAF-01) | fc567f |
| 42-02 | OOF Artifact Profiles (SAF-02) | 46622fa |
| 42-03 | Deployment Gate Evaluator (SAF-03) | 6ee4ec7 |

## Performance Metrics

**Velocity:**

- Total plans completed (all milestones): 85
- v2.1 plans completed: 11 (planned: 11)

**Recent Trend:**

- v2.0 (Phases 37-38): 5 plans, all complete
- Trend: Stable

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
v2.1 key decisions:

- field_size excluded from FORBIDDEN_CALIBRATOR_FEATURES as raw input (Pitfall 3, SAF-01)
- Advisory model class names corrected to match actual codebase (SAF-01)
- Artifact profiles use regular classes with validate() method, PROFILES dict for plugin discovery (SAF-02)
- Rank determinism check emits WARNING (not failure) for duplicated investment_scores (SAF-02)
- GatePolicy frozen dataclass with explicit thresholds for deployment gate evaluation (SAF-03)
- Actual/predicted ratio degradation is WARN not FAIL per D-11 (SAF-03)
- OOF/audit diagnostic gates are SKIP placeholders requiring manual runs per D-05 (SAF-03)
- Variant names identified from manifest flag_states for Pitfall 4 safety (SAF-03)

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

Last session: 2026-05-28T12:54:16.674Z
Stopped at: context exhaustion at 75% (2026-05-28)
Resume file: None
