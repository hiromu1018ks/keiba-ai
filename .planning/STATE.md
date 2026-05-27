---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: Investment Pipeline Restructuring
status: shipped
stopped_at: Milestone complete
last_updated: "2026-05-27T20:00:00Z"
last_activity: 2026-05-27 -- Milestone archived, ready for next
progress:
  total_phases: 38
  completed_phases: 38
  total_plans: 84
  completed_plans: 84
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-27)

**Core value:** 単勝モデルのバックテストROIを100%超えにすること
**Current focus:** Planning next milestone (v2.1+)

## Current Position

Phase: 38 (all complete)
Plan: All plans complete
Status: Shipped
Last activity: 2026-05-27

Progress: [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓] 100%

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
v2.0 key decisions:

- Benter型市場ブレンド: logit(p_model)+logit(p_market)で市場を強い事前分布として扱う
- Segment Calibration統合(Option B): WSCを単独モデルにせずMarketAwareWinCalibrator特徴量に統合
- 配備条件=確率品質: ROIではなくBrier/logloss/ECE/actual-predで配備判定
- レジーム非依存 + ベット数削減禁止
- OOF health fail-fast at save point
- InvestmentFeatureSpec frozen dataclass for schema safety

### Pending Todos

None for v2.0.

### Blockers/Concerns

- BT ROI 87.8% at v2.0 close, target 100%+ — v2.1+ structural reform needed
- v1.7 ROI was 97.8% but degraded to 87.8% in v1.8 — Phase 36.1.1 fixes applied but not BT-validated
- WF検証未実行 (~4h) — cross-cutting validation gap

### Deferred Items

Items acknowledged and deferred at milestone close on 2026-05-27:

| Category | Item | Status |
|----------|------|--------|
| Validation | WF検証スクリプトの実際の実行(~4時間、PostgreSQL環境必要) | Pending since v1.0 |
| UAT | Human UAT 5項目 (PostgreSQL依存) | Pending since v1.4 |
| Bug | test_training_pipeline.py 3件既知失敗 | Pending since v1.6 |
| Pipeline | _build_race_level_features() rl_*列処理 | Pending since v1.7 |
| Feature | 人気帯キャリブレーション (CAL-01~03) | Deferred to Phase 39+ |
| Feature | レジーム×サーフェスEV補正 (CAL-04~05 retired) | Deferred to Phase 39+ |
| Validation | 芝IC b_difference正転換 (VAL-02~06) | Deferred to Phase 39+ |
| Feature | コーナー通過順位展開特徴量 (HLF-06~08) | Future |
| Feature | E-correction fundamental activation (EFA-01~03) | Future |
| Feature | 坂路調教タイムETL (TDF-01) | Future |
| Pipeline | CLV予測モデル / OOF drift detector | v2.1+ |

## Session Continuity

Last session: 2026-05-27T20:00:00Z
Stopped at: Milestone archived
Next: `/gsd:new-milestone` to start v2.1+
