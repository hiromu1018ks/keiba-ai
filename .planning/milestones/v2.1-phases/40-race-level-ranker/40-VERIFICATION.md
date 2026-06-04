---
phase: 40-race-level-ranker
verified: 2026-05-28T12:00:00Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
---

# Phase 40: Race-Level Ranker Verification Report

**Phase Goal:** A learned ranker orders horses within each race by combining relevance (win/finishing-position) and value/mispricing signals, producing an investment_score that replaces hand-tuned formulas
**Verified:** 2026-05-28
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A learned Win relevance ranker orders horses within each race using is_win and finishing-position relevance signals | VERIFIED | `race_level_ranker.py:112-123` _compute_relevance_target maps kakuteijyuni to graded [1.00,0.55,0.30,0.10,0.00]; 15 curated relevance features (D-23); per-surface Ridge with NDCG@3 alpha selection; 13/13 tests pass |
| 2 | A learned value/mispricing ranker detects mispriced horses using calibrated EV, model-vs-market gap, and CLV diagnostics (OOF-safe) | VERIFIED | `race_level_ranker.py:125-161` _compute_value_target uses clipped_log_ev + mispricing_bonus - uncertainty_penalty from OOF sources (p_win_oof, p_market_norm); `win_benter_gate.py:170-182` emits calibrated_ev_oof from fold-level EV correction |
| 3 | Win ranker and Value ranker outputs are combined into a single investment_score per horse | VERIFIED | `race_level_ranker.py:563-582` score() computes 0.35*rel_pct + 0.35*val_pct + 0.20*log_ev_pct - 0.10*uncertainty_pct; test_investment_score_formula verifies exact formula match |
| 4 | The ranker operates in shadow mode behind a feature flag, with baseline WinSelectionGate preserved and functional | VERIFIED | `race_level_ranker.py:103-106` is_trained property; line 510-512 shadow guard returns df unchanged; `race_predictor.py:283-285` getattr+is_trained guard; deployment_status="shadow_only"; win_selection_gate.py untouched; test_predict_untrained_ranker_no_effect passes |
| 5 | One-bet-per-race baseline bet count is maintained without explicit approval to reduce it | VERIFIED | `race_predictor.py:888` profit_max_per_race=1 unchanged; investment_score is diagnostic-only column; win_market_selection_score remains sole sorting criterion; test_win_market_selection_score_unchanged_by_ranker passes |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/models/race_level_ranker.py` | RaceLevelRanker class with train/score/save/load/is_trained | VERIFIED | 622 lines; 4 Ridge models; alpha grid selection; D-11 diagnostics; shadow mode; joblib persistence |
| `src/domain/models.py` | win_race_level_ranker field on SubmodelSet | VERIFIED | Line 264: `win_race_level_ranker: RaceLevelRanker \| None = None`; import under TYPE_CHECKING |
| `src/models/win_benter_gate.py` | Extended generate_win_oof_predictions() with calibrated_ev_oof | VERIFIED | Lines 170-182: captures fold-level ev_win_corrected; emits calibrated_ev_oof column |
| `src/pipelines/training_pipeline.py` | RaceLevelRanker training block after MAWC with OOF+IFF join | VERIFIED | Lines 1324-1355: trains ranker after MAWC; OOF+IFF join by race_id/umaban; MLflow (lines 2190-2206) and local save (lines 2384-2391) |
| `src/backtest/race_predictor.py` | Ranker scoring block + D-18 shadow diagnostics | VERIFIED | Lines 279-285: ranker.score() after MAWC; lines 860-884: D-18 diagnostic columns |
| `src/db/model_loader.py` | Ranker load/save following MAWC pattern | VERIFIED | Lines 242-263 (MLflow path), lines 758-767 (local path); both SubmodelSet constructions updated |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| training_pipeline.py | race_level_ranker.py | RaceLevelRanker import and train() call | WIRED | Line 1328: import; line 1345-1346: instantiate and train |
| training_pipeline.py | win_benter_gate.py | generate_win_oof_predictions() extended output | WIRED | OOF data feeds ranker via oof_cal_df with calibrated_ev_oof |
| training_pipeline.py | feature_frame.py | IFF build_frame(mode="train") joined to OOF | WIRED | Line 1333-1341: build_frame(mode="train") + merge on race_id/umaban |
| race_predictor.py | race_level_ranker.py | ranker.score(df) call after MAWC.apply(df) | WIRED | Line 283-285: getattr + is_trained guard + score() |
| model_loader.py | race_level_ranker.py | RaceLevelRanker.load(path) | WIRED | Lines 259-261 (MLflow), lines 763-765 (local) |
| domain/models.py | race_level_ranker.py | SubmodelSet.win_race_level_ranker field | WIRED | Line 264: typed field; import at line 21 under TYPE_CHECKING |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| race_level_ranker.py train() | relevance_scorer_turf/dirt | OOF DataFrame with kakuteijyuni, IFF features | Yes -- Ridge.fit() on graded relevance target | FLOWING |
| race_level_ranker.py train() | value_scorer_turf/dirt | OOF DataFrame with calibrated_ev_oof, p_win_oof, p_market_norm | Yes -- Ridge.fit() on composite value target | FLOWING |
| race_level_ranker.py score() | investment_score | Ridge.predict() on scored features | Yes -- real predictions with fixed-weight combination | FLOWING |
| win_benter_gate.py | calibrated_ev_oof | fold_val["ev_win_corrected"] per fold | Yes -- captured from existing EV correction output | FLOWING |
| race_predictor.py | D-18 diagnostics | win_market_selection_score + investment_score | Yes -- computed from actual score columns via groupby | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| RaceLevelRanker init defaults | `PYTHONPATH=src python -c "from models.race_level_ranker import RaceLevelRanker; rlr=RaceLevelRanker(); assert not rlr.is_trained"` | No output (success) | PASS |
| Shadow mode guard | `PYTHONPATH=src python -c "... rlr.score(df) returns unchanged"` | No output (success) | PASS |
| SubmodelSet field exists | `PYTHONPATH=src python -c "... assert 'win_race_level_ranker' in fields"` | No output (success) | PASS |
| Ruff lint check | `python -m ruff check src/models/race_level_ranker.py` | All checks passed | PASS |
| Test suite: race_level_ranker | `python -m pytest tests/test_race_level_ranker.py -v` | 13/13 passed | PASS |
| Test suite: regression (domain+benter+pipeline+loader) | `python -m pytest tests/test_domain.py tests/test_win_benter_gate.py tests/test_training_pipeline.py tests/test_model_loader.py -v` | 84/84 passed | PASS |
| Test suite: race_predictor | `python -m pytest tests/test_race_predictor.py -v` | 74/74 passed | PASS |

### Probe Execution

Step 7c: SKIPPED -- no probe scripts declared for this phase.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| RNK-01 | 40-01, 40-02 | Learned Win relevance ranker using is_win/finishing-position signals | SATISFIED | race_level_ranker.py: graded relevance target, 15 relevance features, per-surface Ridge |
| RNK-02 | 40-01, 40-02 | Learned value/mispricing ranker using calibrated EV, model-vs-market gap (OOF-safe) | SATISFIED | race_level_ranker.py: composite value target from OOF sources, 15 value features |
| RNK-03 | 40-01, 40-03 | investment_score combination | SATISFIED | race_level_ranker.py: 0.35/0.35/0.20/0.10 fixed-weight combination; race_predictor.py: score() adds columns |
| RNK-04 | 40-01, 40-03 | Shadow mode with baseline WinSelectionGate preserved | SATISFIED | is_trained guard, deployment_status="shadow_only", win_selection_gate.py unchanged |
| RNK-05 | 40-03 | One-bet-per-race baseline bet count maintained | SATISFIED | profit_max_per_race=1 unchanged; investment_score is diagnostic-only; test passes |

No orphaned requirements. All 5 requirements (RNK-01 through RNK-05) from REQUIREMENTS.md are covered by plans and verified in codebase.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | - | - | No blocker or warning patterns found |

No TBD/FIXME/XXX markers in Phase 40 files. Pre-existing TODOs in race_predictor.py (lines 1026, 1031, 1167) are regime-related and not from Phase 40.

### Human Verification Required

None. All success criteria are programmatically verifiable and have been verified.

### Gaps Summary

No gaps found. All 5 ROADMAP success criteria and all 5 REQUIREMENTS.md entries (RNK-01 through RNK-05) are satisfied with substantive, wired, and flowing implementations. The ranker operates in shadow mode without altering baseline selection. Test suites pass with zero regressions (171 total tests across affected modules).

---

_Verified: 2026-05-28_
_Verifier: Claude (gsd-verifier)_
