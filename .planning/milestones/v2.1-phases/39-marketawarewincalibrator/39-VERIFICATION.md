---
phase: 39-marketawarewincalibrator
verified: 2026-05-28T00:00:00Z
status: passed
score: 4/4 roadmap success criteria verified
overrides_applied: 0
---

# Phase 39: MarketAwareWinCalibrator Verification Report

**Phase Goal:** Win probabilities are produced by a single MarketAwareWinCalibrator that blends model and market logits with segment-conditioned regularization, replacing the previous dual WinBenterGate + WinSegmentCalibrator chain and preventing double-correction
**Verified:** 2026-05-28
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths (from ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | MarketAwareWinCalibrator produces calibrated win probabilities via Benter logit(p_model) + logit(p_market) blend, absorbing both WinBenterGate and WinSegmentCalibrator roles at the same pipeline position in RacePredictor.predict() | VERIFIED | `src/models/market_aware_win_calibrator.py`: LogisticRegression L2 blend of logit_model + logit_market (lines 171-176, 364). `src/backtest/race_predictor.py` lines 268-271: `mawc.apply(df)` called after `ev_corrector.correct_ev()`, before `WinSelectionGate`. 11 unit tests pass including race normalization sum-to-1.0. |
| 2 | Segment conditioning uses popularity rank, odds band, and probability rank as regularized features/interactions in a global calibrator (not per-segment coefficients), preventing sparse segment overfitting | VERIFIED | `build_feature_matrix()` produces 51-dim features: 6 main effects + 15 one-hot (7 odds_band + 5 pop_bucket + 3 p_rank) + 30 logit-segment interactions. L2 regularization via LogisticRegression (no penalty param for sklearn 1.8). Test 10 verifies NO segment x segment interactions. |
| 3 | Calibrator output maintains probability quality (Brier, logloss, ECE) after normalization and satisfies sum-to-1.0 constraint per race | VERIFIED | `apply()` lines 472-474: race-level normalization via `groupby("race_id").transform("sum")`. Test 5 (`test_apply_race_normalization_sums_to_one`) verifies sums within 1e-6. C-selection uses logloss primary metric with Brier secondary (lines 299-300). |
| 4 | WinBenterGate and WinSegmentCalibrator are removed from the pipeline with no remaining call sites | VERIFIED | `src/backtest/race_predictor.py`: zero references to WinBenterGate/win_benter/WinSegmentCalibrator (only comments). `src/db/model_loader.py`: zero references to win_benter/win_segment_calibrator loading. `src/domain/models.py`: win_benter, win_isotonic_calibrator, win_temperature_scaler, win_segment_calibrator fields removed from SubmodelSet. `src/pipelines/training_pipeline.py`: only `generate_win_oof_predictions` import retained (needed for OOF DataFrame generation). |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/models/market_aware_win_calibrator.py` | MarketAwareWinCalibrator class with train/apply/save/load/build_feature_matrix | VERIFIED | 513 lines, dataclass with LogisticRegression L2, 51-dim features, C-selection WF grid, beta_market guard, race normalization |
| `tests/test_market_aware_win_calibrator.py` | Unit tests covering feature encoding, training, inference, guards | VERIFIED | 11 tests, all passing (1.80s) |
| `src/pipelines/training_pipeline.py` | MarketAwareWinCalibrator training replacing WinBenterGate + WinSegmentCalibrator | VERIFIED | Lines 1327-1361: MAWC training block. Lines 1503-1504: segment calibrator removal comment. Line 1547: SubmodelSet uses `market_aware_win_calibrator=`. Lines 2180-2195: MLflow save. Lines 2365-2371: local save. |
| `src/domain/models.py` | Updated SubmodelSet with market_aware_win_calibrator field | VERIFIED | Line 261: `market_aware_win_calibrator: MarketAwareWinCalibrator \| None = None`. Old win fields removed. TYPE_CHECKING import added. |
| `src/backtest/race_predictor.py` | RacePredictor using MarketAwareWinCalibrator instead of WinBenterGate + WinSegmentCalibrator | VERIFIED | Lines 268-277: MAWC apply with fallback. Lines 644-648: neutral segment factors. `_get_win_segment_calibrator` removed. |
| `src/db/model_loader.py` | ModelLoader loading MarketAwareWinCalibrator from MLflow/local | VERIFIED | Lines 219-240: MLflow load path. Lines 711-720: local load path. Lines 394, 843: SubmodelSet construction with `market_aware_win_calibrator=`. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/models/market_aware_win_calibrator.py` | `sklearn.linear_model.LogisticRegression` | L2-regularized logit-blend calibrator | WIRED | `from sklearn.linear_model import LogisticRegression` (line 24), used in `_fit_final()` and C-selection grid |
| `build_feature_matrix()` | segment one-hot encoding | ODDS_BAND_NAMES(7) + POP_BUCKET_NAMES(5) + P_RANK_NAMES(3) | WIRED | `_encode_odds_band()`, `_encode_pop_bucket()`, `_encode_p_rank()` all produce guaranteed schema columns |
| `training_pipeline.py` | `market_aware_win_calibrator.py` | import MarketAwareWinCalibrator, call train() | WIRED | `from models.market_aware_win_calibrator import MarketAwareWinCalibrator` (line 1330), `train()` called at line 1345 |
| `training_pipeline.py` | `domain/models.py` | SubmodelSet constructor with market_aware_win_calibrator= | WIRED | Line 1547: `market_aware_win_calibrator=market_aware_calibrator` |
| `race_predictor.py` | `market_aware_win_calibrator.py` | import, call apply() | WIRED | Lines 269-271: `getattr(submodel, "market_aware_win_calibrator", None)` then `mawc.apply(df)` |
| `model_loader.py` | `market_aware_win_calibrator.py` | load from MLflow and local | WIRED | MLflow: lines 236-238. Local: lines 716-718. Both call `MarketAwareWinCalibrator.load()` |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `MarketAwareWinCalibrator.train()` | `X` (feature matrix), `y` (target) | OOF DataFrame from `generate_win_oof_predictions()` | FLOWING | `generate_win_oof_predictions()` returns enriched DataFrame with p_win_oof, p_market_norm, tanodds, popularity_rank, field_size, p_win_race_rank_pct, race_id, race_date, umaban, surface, kakuteijyuni, p_win_corrected |
| `MarketAwareWinCalibrator.apply()` | `p_win_combined`, `p_win_final`, `edge_win` | `calibrator.predict_proba(X)[:, 1]` then race normalization | FLOWING | Real prediction pipeline, not hardcoded. Race normalization divides by groupby sum. |
| `RacePredictor.predict()` | `df` with p_win_final, edge_win | `mawc.apply(df)` or fallback block | FLOWING | Fallback computes p_win_final = p_win_corrected / race_sum, edge_win = p_win_final * tanodds - 1.0 |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All MarketAwareWinCalibrator tests pass | `python -m pytest tests/test_market_aware_win_calibrator.py -v` | 11 passed in 1.80s | PASS |
| Domain + RacePredictor + ModelLoader tests pass | `python -m pytest tests/test_domain.py tests/test_race_predictor.py tests/test_model_loader.py -v` | 97 passed in 3.72s | PASS |
| TrainingPipelineV5 imports successfully | `python -c "from pipelines.training_pipeline import TrainingPipelineV5"` | Import OK | PASS |
| SubmodelSet has correct fields | `python -c "from domain.models import SubmodelSet; s = SubmodelSet(); assert hasattr(s, 'market_aware_win_calibrator'); assert not hasattr(s, 'win_benter')"` | Assertion passed | PASS |

### Probe Execution

Step 7c: SKIPPED (no probe scripts declared or conventional for this phase)

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| CAL-01 | 39-01, 39-02, 39-03 | MarketAwareWinCalibrator produces calibrated win probabilities via Benter logit-blend | SATISFIED | LogisticRegression L2 blend of logit(p_model) + logit(p_market) with 51-dim features, C-selection WF grid search |
| CAL-02 | 39-01 | Segment conditioning from InvestmentFeatureFrame (popularity rank, odds band, probability rank) | SATISFIED | `_encode_odds_band()` (7 bands), `_encode_pop_bucket()` (5 buckets), `_encode_p_rank()` (3 ranks) |
| CAL-03 | 39-01, 39-02 | Segment effects as regularized features/interactions, not per-segment coefficients | SATISFIED | 30 interaction features (logit_model x segment + logit_market x segment) inside L2-regularized LogisticRegression |
| CAL-04 | 39-02, 39-03 | MarketAwareWinCalibrator replaces WinBenterGate + WinSegmentCalibrator, preventing double-correction | SATISFIED | All old references removed from pipeline, SubmodelSet fields updated, neutral segment factors (1.0) |
| CAL-05 | 39-01, 39-02, 39-03 | Calibrator output maintains probability quality with sum-to-1.0 constraint | SATISFIED | Race normalization in `apply()`, C-selection uses logloss/Brier, test verifies sum within 1e-6 |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | - | - | No TBD/FIXME/XXX markers, no stubs, no empty implementations |

### Human Verification Required

None -- all truths are programmatically verified through unit tests, import checks, and code inspection.

---

_Verified: 2026-05-28_
_Verifier: Claude (gsd-verifier)_
