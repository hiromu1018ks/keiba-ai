---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: Investment Pipeline Restructuring
status: planning
last_updated: "2026-05-27T00:00:00.000Z"
last_activity: 2026-05-27
progress:
  total_phases: 4
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-27)

**Core value:** 単勝モデルのバックテストROIを100%超えにすること
**Current focus:** Phase 37 — OOF Health Infrastructure

## Current Position

Phase: 37 of 40 (OOF Health Infrastructure)
Plan: 0 of ? in current phase
Status: Ready to plan
Last activity: 2026-05-27 — v2.0 roadmap created

Progress: [░░░░░░░░░░] 0%

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
v2.0 key decisions:

- Benter型市場ブレンド: logit(p_model)+logit(p_market)で市場を強い事前分布として扱う
- Segment Calibration統合(Option B): WSCを単独モデルにせずMarketAwareWinCalibrator特徴量に統合
- 配備条件=確率品質: ROIではなくBrier/logloss/ECE/actual-predで配備判定
- レジーム非依頼 + ベット数削減禁止

### Pending Todos

None yet for v2.0.

### Blockers/Concerns

- BT ROI 87.8% at v1.8 close (#33), target 100%+ — v2.0 structural reform needed
- v1.7 ROI was 97.8% but degraded to 87.8% in v1.8 — Phase 36.1.1 fixes applied but not BT-validated
- WF検証未実行 (~4h) — cross-cutting validation gap

### Deferred Items

| Category | Item | Status |
|----------|------|--------|
| Validation | WF検証スクリプトの実際の実行(~4時間、PostgreSQL環境必要) | Pending since v1.0 |
| UAT | Human UAT 5項目 (PostgreSQL依存) | Pending since v1.4 |
| Bug | test_training_pipeline.py 3件既知失敗 | Pending since v1.6 |
| Pipeline | _build_race_level_features() rl_*列処理 | Pending since v1.7 |
| Future | Portfolio (Phase 5) + Multi-Market (Phase 6) | v2.1+ |
| Future | CLV予測モデル / OOF drift detector | v2.1+ |

## Session Continuity

Last session: 2026-05-27
Stopped at: v2.0 roadmap created, ready to plan Phase 37
Resume file: None
