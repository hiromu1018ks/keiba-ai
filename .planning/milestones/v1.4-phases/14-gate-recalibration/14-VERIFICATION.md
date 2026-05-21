---
phase: 14-gate-recalibration
verified: 2026-05-06T12:00:00Z
status: passed
score: 3/3 must-haves verified
overrides_applied: 0
re_verification: false
---

# Phase 14: Gate Recalibration Verification Report

**Phase Goal:** WinSelectionGateがアンサンブルOOF予測で再学習され、quantile binとscore tableが新しい分布に適合し、use_ensembleフラグがパイプライン全体で正しく伝播されている状態になる
**Verified:** 2026-05-06T12:00:00Z
**Status:** PASSED
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | run_backtest.py --ensemble 実行時にWinSelectionGateがアンサンブルOOF予測で再学習され、再計算されたprob_edges/edge_edges/odds_edgesが単一モデルと異なる値になる | VERIFIED | (a) pipeline lines 453-468: StackedEnsemble created when use_ensemble=True and assigned to hit_model; (b) pipeline lines 806-807: WinSelectionGateModel.train(wsg_train_df) receives ensemble-derived df_oof; (c) pipeline lines 810-812: runtime assertions verify gate.is_trained and prob_edges non-empty; (d) test_gate_edges_differ_between_single_and_ensemble_oof passes: edges differ between narrow (single) and wide (ensemble) distributions |
| 2 | ks_2samp/wasserstein_distanceで単一モデルとアンサンブルのOOF確率分布を比較した診断レポートが出力され、ドリフト量が定量化されている | VERIFIED | (a) src/models/drift_diagnostics.py: compute_drift_diagnostics() with ks_2samp + wasserstein_distance, surface/year splits, JSON output, console_summary; (b) pipeline lines 792-803: integration guarded by use_ensemble=True, own TimingContext, JSON output to data/backtest/; (c) 8 unit tests all pass covering basic stats, drift detection, surface/year splits, JSON output, missing columns, logging |
| 3 | use_ensemble=TrueがModelLoader→RacePredictor→BacktestEngine→WinSelectionGate全経路で一貫して伝播されていることをテストで確認できる | VERIFIED | (a) TrainingPipelineV5.train() line 85: accepts use_ensemble; line 97: stores self.use_ensemble; line 203: passes to _train_submodel(); (b) _train_submodel() line 284: accepts use_ensemble; line 825: passes to SubmodelSet; (c) ModelLoader._load_hit_model() line 448: accepts use_ensemble, prefers .joblib when True; (d) ModelLoader.load_from_dir() lines 519-522: resolves use_ensemble from override or meta.json; line 695: passes to SubmodelSet (bug fix from 14-02); (e) TestEnsembleFlagPropagation 3 tests all pass verifying StackedEnsemble creation, .joblib loading, and trained gate in SubmodelSet |

**Score:** 3/3 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| src/models/drift_diagnostics.py | compute_drift_diagnostics with KS/Wasserstein stats | VERIFIED | 119 lines; exports compute_drift_diagnostics, console_summary, DRIFT_COLUMNS; uses scipy.stats.ks_2samp and wasserstein_distance |
| src/pipelines/training_pipeline.py | Pipeline integration of drift diagnostics in _train_submodel() | VERIFIED | Lines 792-803: drift diagnostics call guarded by use_ensemble=True with own TimingContext; lines 809-812: runtime assertions |
| tests/test_drift_diagnostics.py | 8 unit tests for drift diagnostics | VERIFIED | 8 tests all pass: basic stats, drift detection, surface/year splits, JSON output, recommendations, missing columns, console summary |
| tests/test_win_selection_gate.py | Gate retraining verification test | VERIFIED | test_gate_edges_differ_between_single_and_ensemble_oof at line 270 passes |
| tests/test_ensemble_gate_propagation.py | 3 integration tests for use_ensemble flag propagation | VERIFIED | TestEnsembleFlagPropagation class with 3 methods all pass |
| src/db/model_loader.py | use_ensemble propagation fix in load_from_dir() | VERIFIED | Line 695: use_ensemble=use_ensemble added to SubmodelSet() call (bug fix from 14-02) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| training_pipeline.py | drift_diagnostics.py | import + compute_drift_diagnostics() call | WIRED | Line 795: `from models.drift_diagnostics import compute_drift_diagnostics, console_summary`; line 798: function call with wsg_train_df |
| training_pipeline.py | data/backtest/ | JSON output path | WIRED | Line 797: `Path("data/backtest") / f"drift_diagnostics_{surface}.json"` |
| test_ensemble_gate_propagation.py | training_pipeline.py | mock StackedEnsemble verification | WIRED | sys.modules patch for models.stacked_ensemble; verifies cat_cols, train() call, hit_model assignment |
| test_ensemble_gate_propagation.py | model_loader.py | mock _load_hit_model + load_from_dir test | WIRED | Direct _load_hit_model test + full load_from_dir integration test with temp dir |
| model_loader.py | SubmodelSet | use_ensemble parameter | WIRED | Line 695: use_ensemble=use_ensemble passed to SubmodelSet constructor |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| drift_diagnostics.py | DRIFT_COLUMNS | df_oof (pipeline) | Yes -- df_oof populated from ensemble prediction pipeline (win_2s.predict_ev at line 475) | FLOWING |
| training_pipeline.py (gate training) | wsg_train_df | df_oof after ensemble prediction + ensure_win_selection_columns | Yes -- df_oof flows through ensemble predict_ev then into gate training | FLOWING |
| model_loader.py | use_ensemble flag | meta.json or use_ensemble_override parameter | Yes -- flag resolved at lines 519-522, propagated through _load_hit_model and SubmodelSet | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Drift diagnostics tests pass | python -m pytest tests/test_drift_diagnostics.py -v | 8 passed | PASS |
| Ensemble propagation tests pass | python -m pytest tests/test_ensemble_gate_propagation.py -v | 3 passed | PASS |
| Gate retraining test passes | python -m pytest tests/test_win_selection_gate.py::test_gate_edges_differ_between_single_and_ensemble_oof -v | 1 passed | PASS |
| All phase 14 tests together | python -m pytest tests/test_drift_diagnostics.py tests/test_ensemble_gate_propagation.py tests/test_win_selection_gate.py::test_gate_edges_differ_between_single_and_ensemble_oof -v | 12 passed in 143.31s | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| GATE-01 | 14-01 | WinSelectionGateをアンサンブルOOF予測で再学習し、prob_edges/edge_edges/odds_edgesを再計算する | SATISFIED | Pipeline integration: StackedEnsemble hit_model (lines 453-468), gate.train(wsg_train_df) (line 807), runtime assertions (lines 810-812), test_gate_edges_differ test |
| GATE-02 | 14-01 | 単一モデルとアンサンブルのOOF確率分布をks_2samp/wasserstein_distanceで比較し、ドリフトを定量化する診断機能を追加する | SATISFIED | drift_diagnostics.py module, pipeline integration (lines 792-803), 8 unit tests |
| GATE-03 | 14-02 | use_ensembleフラグがModelLoader→RacePredictor→BacktestEngine全体で正しく伝播されていることを検証する | SATISFIED | TestEnsembleFlagPropagation 3 tests, ModelLoader bug fix (line 695), full propagation chain verified from TrainingPipelineV5.train through _train_submodel to SubmodelSet |

No orphaned requirements found. All 3 requirement IDs (GATE-01, GATE-02, GATE-03) mapped to Phase 14 in REQUIREMENTS.md are covered by plans and verified in the codebase.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | - | - | No anti-patterns found |

No TODO/FIXME/HACK/PLACEHOLDER markers found in any phase 14 files. No empty return patterns, no console.log-only handlers, no hardcoded empty data flows.

### Human Verification Required

No human verification items required. All truths are verified programmatically:
- Gate retraining verified by unit test (different distributions produce different edges)
- Drift diagnostics verified by 8 unit tests + pipeline integration code review
- Flag propagation verified by 3 integration tests + code review of full chain

### Gaps Summary

No gaps found. All 3 ROADMAP success criteria are satisfied:

1. WinSelectionGate trains on ensemble-derived OOF data and produces different edges than single-model -- verified by test and pipeline code.
2. Drift diagnostics module with ks_2samp/wasserstein_distance integrated into pipeline for ensemble mode -- verified by module code, 8 tests, and pipeline integration.
3. use_ensemble flag propagates correctly through TrainingPipelineV5 and ModelLoader -- verified by 3 integration tests, including discovery and fix of a bug where ModelLoader.load_from_dir() was not passing use_ensemble to SubmodelSet.

---

_Verified: 2026-05-06T12:00:00Z_
_Verifier: Claude (gsd-verifier)_
