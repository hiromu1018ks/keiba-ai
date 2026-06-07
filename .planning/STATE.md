---
gsd_state_version: 1.0
milestone: v2.4
milestone_name: Paper Trading Pipeline Integration
status: executing
stopped_at: context exhaustion at 75% (2026-06-07)
last_updated: "2026-06-07T06:14:36.795Z"
last_activity: 2026-06-06 -- Phase 54 execution started
progress:
  total_phases: 4
  completed_phases: 4
  total_plans: 13
  completed_plans: 13
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-06)

**Core value:** 単勝モデルのバックテストROIを100%超えにすること
**Current focus:** Phase 54 — automation-reporting

## Current Position

Phase: 54 (automation-reporting) — EXECUTING
Plan: 1 of 3
Status: Executing Phase 54
Last activity: 2026-06-06 -- Phase 54 execution started

Progress: [████░░░░░░] 80%

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- v2.4 is an integration milestone, not a feature milestone
- PT uses retrained models (run_train.py 2022-2025), not BT model artifacts
- 2026 PT uses data up to 2025-12-31 only (no future information)
- Regime state unification decision deferred to Phase 53 planning
- PaperReconciler is the single implementation of settlement logic (D-01, Phase 51)
- bet_id = SHA256(session_id|race_id|bet_type|umaban)[:32] (D-02, Phase 51)
- 3-column state model: settlement_status/outcome/payout (D-03, Phase 51)
- ROI = return / effective_stake (won+lost only), excluding refunded/voided (D-05, Phase 51)
- ModelLoader requires explicit run_id or models_dir, no implicit fallback (D-16, Phase 51)
- Moisture collation uses closest-match algorithm for JRA/CSV rule selection (D-06, Phase 53)
- Live track conditions merged via left join with NaN safety (D-07, Phase 53)

### Phase 51 Deliverables

| Component | Status | Key Change |
|-----------|--------|------------|
| src/betting/payout_maps.py | New | Pure functions for Win/Place/Wide payout maps (20 tests) |
| src/backtest/engine.py | Modified | Imports from payout_maps.py, ~200 lines removed |
| src/paper_trading/reconciler.py | Overhauled | 3-column state model, retry, ROI with losses (25 tests) |
| scripts/run_paper_trading.py | Modified | New schema columns in _run_predict, _run_reconcile thinned to 58 lines |
| scripts/run_train.py | Modified | --betting-target CLI, pre-training Parquet validation |
| src/pipelines/training_pipeline.py | Modified | track_stats JSON persistence, betting_target in meta.json |
| src/features/feature_engine.py | Modified | track_conditions + horse_track_aptitude cache deps |
| src/db/model_loader.py | Modified | Explicit source selection, track_stats restore, betting_target (12 tests) |

### Blockers/Concerns

- Phase 52 (Shared Feature Builder) is HIGH RISK — extraction from BacktestEngine.prepare_data() requires full BT regression test before and after

## Deferred Items

| Category | Item | Status |
|----------|------|--------|
| Validation | WF検証スクリプトの実際の実行(~4時間、PostgreSQL環境必要) | Pending since v1.0 |
| UAT | Human UAT 5項目 (PostgreSQL依存) | Pending since v1.4 |
| Bug | test_training_pipeline.py 3件既知失敗 | Pending since v1.6 |
| Bug | 4 pre-existing test failures (observed_true, blood_keito, ev_oof, profit_selector) | Pending |
| Cleanup | WinSegmentCalibrator dead code removal (WRN-01) | Pending since v2.1 |
| Feature | 4 RACE_CONDITION特徴量100% NaN修正 (track_month_stats availability) | Pending since v2.3 |
| Feature | sire_x_cushion_band 51% NaN改善 (種牡馬×クッション交差データ不足) | Pending since v2.3 |
| Optimization | Optuna 19次元パラメータ最適化 (DEP-02) | Deferred to v2.5+ |
| Automation | デプロイゲート自動判定 (DEP-01) | Deferred to v2.5+ |
| Calibration | Conservative MAWC redesign | Deferred to v2.5+ |
| Validation | IC評価レポート生成 (OOF予測必要、別途run_train.py) | Pending since v2.3 |

## Session Continuity

Last session: 2026-06-07T06:14:36.789Z
Stopped at: context exhaustion at 75% (2026-06-07)
