---
phase: 03-selection-gate-confidence-betting
verified: 2026-05-03T10:00:00Z
status: passed
score: 8/8 must-haves verified
overrides_applied: 0
re_verification: false
deferred:
  - truth: "統合後のバックテストで、ベット数が適切にフィルタリングされROIの改善が確認される"
    addressed_in: "Phase 4"
    evidence: "Phase 4 goal: Walk-Forward Validation; success criteria: ROI>100% multi-year validation"
---

# Phase 3: Selection Gate, Confidence & Betting Verification Report

**Phase Goal:** 学習済みゲートで低信頼レースを除外し、JRA控除率25%を考慮した最適ベッティング戦略を統合する
**Verified:** 2026-05-03
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | WinSelectionGateModelがOOF walk-forward score tablesで学習し、単勝ベットのpass/reject判定を行う | VERIFIED | win_selection_gate.py 1044 lines; train() uses _build_walk_forward_folds + _build_score_tables; score() produces win_gate_pass bool column |
| 2 | Win gateはPlace gateと同じ3次元binning + smoothed scoring + add-second reranker構造を持つ | VERIFIED | _build_score_tables: prob/edge/odds 3D binning; _smoothed_score for Bayesian smoothing; _fit_add_second_reranker with grid search |
| 3 | 低信頼レースがWinSelectionGateのmin_prob/min_edge/max_oddsで完全除外される | VERIFIED | _pass_mask() checks prob >= min_prob AND edge >= min_edge AND odds <= max_odds; _primary_selection_mask applies hard+soft masks |
| 4 | RacePredictor.predict()でWin Benter適用後にWinSelectionGateが適用される | VERIFIED | race_predictor.py lines 127-140: ensure_win_selection_columns -> getattr(submodel, 'win_selection_gate') -> score(df) -> annotate_race_context(df), inserted after Win Benter (line 125), before Place (line 142) |
| 5 | Conformal predictionのrace-condition-dependent calibrationがsurface/distance_binで細分化される | VERIFIED | robust_confidence_estimator.py: _win_cp_quantile_by_condition dict (line 42); calibrate() groupby(["surface","distance_bin"]) with min 30 samples (lines 71-77); predict_lower_bound() uses conditional quantile per row (lines 119-137) |
| 6 | RegimeDetectorのedge_thresholdがJRA控除率25%を考慮して引き上げられている | VERIFIED | regime_detector.py: AGGRESSIVE=0.05, CONSERVATIVE=0.06, COLLAPSED=0.09 (lines 183/211/224); all +0.01 from prior values |
| 7 | MetaSwitcherのedge_thresholdがRegimeDetectorと同期して引き上げられている | VERIFIED | meta_switcher.py: AGGRESSIVE=0.05, CONSERVATIVE=0.07, COLLAPSED=0.10 (lines 47/55/63); maintains +0.01 gap for CONSERVATIVE/COLLAPSED |
| 8 | GateKeeperのデフォルトedge閾値が0.03から0.04に引き上げられている | VERIFIED | gate_keeper.py: should_bet returns bet.edge >= 0.04 (line 28); filter_bets default edge_threshold=0.04 (line 30); no 0.03 references remain |

**Score:** 8/8 truths verified

### Deferred Items

Items not yet met but explicitly addressed in later milestone phases.

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | 統合後のバックテストで、ベット数が適切にフィルタリングされROIの改善が確認される | Phase 4 | Phase 4 success criteria: "複数年度の加重平均ROIが100%を超えている" + "2024-2025のウォークフォワード交差検証が実行され、各テスト年度のROIが個別に確認できる" |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/models/win_selection_gate.py` | WinSelectionGateModel完全実装 | VERIFIED | 1044 lines; class WinSelectionGateModel with train/score/save/load/soft_pass_mask/add_second_reranker; kakuteijyuni==1 hit condition (line 232); tanoddslow as odds source (14 occurrences) |
| `tests/test_win_selection_gate.py` | WinSelectionGateModelテスト | VERIFIED | 6 tests covering train/score, fallback chain, EV computation, hit condition, save/load roundtrip, soft_pass_mask; all 6 PASS |
| `src/domain/models.py` | SubmodelSet dataclass with win_selection_gate | VERIFIED | Line 255: `win_selection_gate: WinSelectionGateModel \| None = None`; TYPE_CHECKING import line 19 |
| `src/backtest/race_predictor.py` | WinSelectionGate integration | VERIFIED | Import lines 18; usage lines 128-140: ensure_win_selection_columns -> getattr -> score -> annotate_race_context |
| `src/models/regime_detector.py` | edge_threshold values | VERIFIED | 0.05/0.06/0.09 with Phase 3 comments |
| `src/betting/meta_switcher.py` | edge_threshold values | VERIFIED | 0.05/0.07/0.10 with Phase 3 comments |
| `src/betting/gate_keeper.py` | default threshold 0.04 | VERIFIED | Both should_bet and filter_bets use 0.04; no 0.03 remains |
| `src/models/robust_confidence_estimator.py` | race-condition-dependent CP quantile | VERIFIED | _win_cp_quantile_by_condition dict (4 occurrences: __init__, calibrate, calibrate groupby, predict_lower_bound conditional logic) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/backtest/race_predictor.py` | `src/models/win_selection_gate.py` | `getattr(submodel, 'win_selection_gate', None)` then `.score(df)` | WIRED | Lines 131-137; score + annotate_race_context called |
| `src/pipelines/training_pipeline.py` | `src/models/win_selection_gate.py` | `WinSelectionGateModel.train()` + `ensure_win_selection_columns()` | WIRED | Import line 42; training block lines 771-778; save to MLflow lines 1066-1074; save to local lines 1194-1196 |
| `src/db/model_loader.py` | `src/models/win_selection_gate.py` | `WinSelectionGateModel.load()` | WIRED | Import line 86; MLflow load lines 148-162; local load lines 585-589; SubmodelSet construction lines 307, 697 |
| `src/models/robust_confidence_estimator.py` | race conditions (surface/distance_bin) | `_win_cp_quantile_by_condition` dict | WIRED | __init__ line 42; calibrate() groupby lines 71-77; predict_lower_bound() conditional per-row lines 119-137 |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| `win_selection_gate.py` score() | win_gate_pass | OOF walk-forward training: min_prob/min_edge/max_odds derived from threshold grid search over _simulate_threshold_surface | Yes -- thresholds learned from realized_win_roi | FLOWING |
| `race_predictor.py` predict() | win_selection_ev, win_selection_prob | EV columns from upstream ML models (win Benter, EV correction) | Yes -- ensure_win_selection_columns builds from ev_win_corrected/EV_lower_win_corrected | FLOWING |
| `robust_confidence_estimator.py` predict_lower_bound() | EV_lower_win_corrected | Conditional CP quantile per surface/distance_bin; falls back to global quantile | Yes -- computed from actual residuals groupby with min 30 samples | FLOWING |
| `regime_detector.py` get_strategy_params() | edge_threshold | Hardcoded per regime with Phase 3 values | Yes -- static config values | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| WinSelectionGate tests pass | `python -m pytest tests/test_win_selection_gate.py -v` | 6/6 passed in 16.23s | PASS |
| RacePredictor tests pass (regression) | `python -m pytest tests/test_race_predictor.py -v` | 28/28 passed in 16.45s | PASS |
| PlaceSelectionGate tests pass (regression) | `python -m pytest tests/test_place_selection_gate.py -v` | 2/2 passed in 6.27s | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-----------|-------------|--------|----------|
| SELC-01 | 03-01-PLAN | PlaceSelectionGateパターンを踏襲したWinSelectionGate実装 | SATISFIED | WinSelectionGateModel 1044 lines, mechanical clone of PlaceSelectionGate with win-specific mappings (kakuteijyuni==1, tanoddslow, EV_lower_win_corrected) |
| SELC-02 | 03-01-PLAN | Conformal predictionに基づく信頼性推定で低信頼度レース除外 | SATISFIED | _win_cp_quantile_by_condition dict with surface/distance_bin groupby; predict_lower_bound() uses conditional quantile per row; WinSelectionGate min_prob/min_edge/max_odds exclusion |
| BETT-01 | 03-02-PLAN | JRA控除率25%を考慮したエッジ閾値設定・調整 | SATISFIED | RegimeDetector: 0.05/0.06/0.09; MetaSwitcher: 0.05/0.07/0.10; GateKeeper: 0.04 default -- all +0.01 from prior values |

No orphaned requirements found -- all three requirements mapped to Phase 3 in REQUIREMENTS.md are covered by plans.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | - | - | No anti-patterns detected in any phase 3 artifacts |

No TODO/FIXME/PLACEHOLDER/stub patterns found. No empty implementations. No hardcoded empty data in production paths.

### Human Verification Required

None -- all must-haves are programmatically verified. The deferred backtest ROI validation (ROADMAP success criterion #4) is explicitly scheduled for Phase 4.

### Gaps Summary

No gaps found. All 8 observable truths verified with codebase evidence. All 3 requirements (SELC-01, SELC-02, BETT-01) satisfied. All artifacts exist, are substantive, and are properly wired into the pipeline.

---

_Verified: 2026-05-03_
_Verifier: Claude (gsd-verifier)_
