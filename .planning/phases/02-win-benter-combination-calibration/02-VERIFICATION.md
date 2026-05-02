---
phase: 02-win-benter-combination-calibration
verified: 2026-05-02T12:00:00Z
status: gaps_found
score: 6/7 must-haves verified
overrides_applied: 0
re_verification: false
gaps:
  - truth: "Win Benter学習データがOOF(out-of-fold)予測で生成され、データリークがない"
    status: failed
    reason: "generate_win_oof_predictions() calls fold_model.predict_ev() which accesses self.return_model, but only train_hit_model() is called beforehand. self.return_model is never set, causing AttributeError at runtime in production pipeline."
    artifacts:
      - path: "src/models/win_benter_gate.py"
        issue: "Lines 112-118: train_hit_model() only sets self.hit_model, but predict_ev() on line 117 accesses self.return_model which is uninitialized"
    missing:
      - "Fix generate_win_oof_predictions to call a hit-only prediction path instead of predict_ev(), or call train_return_model() in addition to train_hit_model()"
      - "The OOF loop needs p_win_pred which only requires hit_model.predict(), not the full predict_ev()"
---

# Phase 2: Win Benter Combination & Calibration Verification Report

**Phase Goal:** 単勝予測に市場効率信号を組み込み、確率を正規化・キャリブレーションすることでEV推定精度を飛躍的に向上させる
**Verified:** 2026-05-02T12:00:00Z
**Status:** gaps_found
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | WinBenterGateがbasic確率(p_win_corrected)と市場確率(1/tanodds)をブレンドしたp_win_combinedを出力する | VERIFIED | WinBenterGate.apply() lines 61-75: benter.combine(p_fund, p_market) -> p_win_combined. Test test_combined_range passes. |
| 2 | Benter組み合わせ後の確率がレース単位で正規化され、各レースのP合計が1.0になる | VERIFIED | apply() lines 78-79: groupby(race_id).transform("sum") normalization. Test test_race_sums_to_one passes with atol=1e-9. |
| 3 | Win Benter学習データがOOF(out-of-fold)予測で生成され、データリークがない | FAILED | generate_win_oof_predictions() line 115 calls train_hit_model() only, but line 117 calls predict_ev() which accesses self.return_model (uninitialized). Runtime crash: AttributeError. Training pipeline tests fail with this exact error. |
| 4 | Win Benterモデルが保存・読み込み可能で、バックテストパイプラインに統合されている | VERIFIED | training_pipeline.py lines 1181-1195: save logic. model_loader.py lines 617-644: load logic. race_predictor.py lines 116-124: inference wiring. |
| 5 | Beta calibrationとIsotonic calibrationが両方実装され、Brier Score + ECEで定量比較される | VERIFIED | compare_calibrations() in win_benter_gate.py lines 218-319: both Beta and Isotonic fitted, Brier+ECE computed, winner selected. Tests pass. |
| 6 | Beta calibration(3パラメータ)がIsotonicより低いBrier Score/ECEを示す、または両者の定量的な差が記録される | VERIFIED | compare_calibrations() lines 292-299: winner selection with Brier primary, ECE secondary, Beta preferred for stability (5% threshold). Tests confirm selection logic. |
| 7 | 信頼性ダイアグラム(reliability diagram)のデータが生成され、オッズバケット毎のキャリブレーション品質が確認できる | VERIFIED | generate_reliability_data() lines 322-342: returns fraction_of_positives, mean_predicted_value, bin_edges. training_pipeline.py line 669: logs reliability data during training. Tests pass. |

**Score:** 6/7 truths verified

### ROADMAP Success Criteria Coverage

| # | Success Criterion | Status | Evidence |
|---|-------------------|--------|----------|
| 1 | WinBenterGateが実装され、基本確率と市場確率のブレンド済み単勝確率が出力される | VERIFIED | WinBenterGate class with apply(), extract_market_probability(), 15/15 tests pass |
| 2 | Beta calibrationとIsotonic calibrationが比較評価され、単勝に最適な手法が採用されている | VERIFIED | compare_calibrations() with Brier+ECE dual metrics, winner selection logic, pipeline integration |
| 3 | Benter組み合わせ後の確率がレース単位で正規化され、各レースのP合計が1.0になる | VERIFIED | Test test_race_sums_to_one with assert_allclose atol=1e-9 |
| 4 | 信頼性ダイアグラムにより、キャリブレーション品質がオッズバケット毎に視覚的に確認できる | VERIFIED | generate_reliability_data() with n_bins=10, logged during training |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/models/win_benter_gate.py` | WinBenterGate class + calibration functions | VERIFIED | 343 lines. Exports: WinBenterGate, generate_win_oof_predictions, compute_ece, compare_calibrations, generate_reliability_data, BetaCalibrationManual |
| `src/domain/models.py` | SubmodelSet with win_* fields | VERIFIED | Lines 249-252: win_benter, win_isotonic_calibrator, win_temperature_scaler added as Optional with None default |
| `tests/test_win_benter_gate.py` | Unit tests for WinBenterGate + calibration | VERIFIED | 15 tests in 9 classes covering all behaviors. 15/15 pass |
| `src/pipelines/training_pipeline.py` | Win Benter training + calibration integration | WIRED | Lines 567-728: OOF generation, grid search, calibration comparison, temperature scaling. SubmodelSet construction line 784. Save lines 1181-1195. Note: OOF generation has runtime bug (see gap) |
| `src/db/model_loader.py` | Win Benter model loading | VERIFIED | Lines 617-644: load benter_combo_win, isotonic_win, temp_scale_win. SubmodelSet construction lines 660-662 |
| `src/backtest/race_predictor.py` | WinBenterGate application in predict() | VERIFIED | Lines 116-124: getattr guard + WinBenterGate construction + apply(). Position after ev_corrector, before place prediction |
| `pyproject.toml` | betacal dependency | VERIFIED | betacal>=1.0 in dependencies |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| race_predictor.py | win_benter_gate.py | WinBenterGate.apply() after EV correction | WIRED | Lines 116-124: getattr guard, import, construction with benter/calibrator/temp_scaler, apply() |
| training_pipeline.py | win_benter_gate.py | generate_win_oof_predictions() + compare_calibrations() | PARTIAL | Imports correct (lines 573, 660). generate_win_oof_predictions has runtime bug. compare_calibrations wiring correct (line 666) |
| model_loader.py | domain/models.py | benter_combo_win_{surface}.json -> win_benter field | WIRED | Lines 617-644: loads all three win_* artifacts. Lines 660-662: passes to SubmodelSet |
| training_pipeline.py | domain/models.py | win_benter/win_isotonic_calibrator/win_temperature_scaler fields | WIRED | Line 784: win_benter=win_benter. Line 785: win_isotonic_calibrator=win_isotonic_cal. Line 786: win_temperature_scaler=win_temp_scaler |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| WinBenterGate.apply() | p_win_combined | benter.combine(p_fund, p_market) | Yes -- real Benter logit combination | FLOWING |
| WinBenterGate.apply() | p_win_final | p_win_combined / groupby sum | Yes -- race normalization | FLOWING |
| WinBenterGate.apply() | edge_win | p_win_final * tanodds - 1.0 | Yes -- edge calculation | FLOWING |
| generate_win_oof_predictions() | oof_p_fund | predict_ev() -> p_win_corrected | DISCONNECTED -- predict_ev() crashes at self.return_model | DISCONNECTED |
| compare_calibrations() | beta_brier, iso_brier, winner | Benter-combined OOF probabilities | Depends on generate_win_oof_predictions output | STATIC (dependent on broken OOF) |
| generate_reliability_data() | fraction_of_positives, mean_predicted_value | sklearn.calibration_curve | Yes -- produces real data when called | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| WinBenterGate tests pass | python -m pytest tests/test_win_benter_gate.py -v | 15/15 passed | PASS |
| Regression tests pass | python -m pytest tests/test_benter_combination.py tests/test_ev_correction.py -v | 34/34 passed | PASS |
| Race predictor tests pass | python -m pytest tests/test_race_predictor.py -v | 28/28 passed | PASS |
| Training pipeline tests pass | python -m pytest tests/test_training_pipeline.py -v | 3 failed, 17 passed | FAIL -- OOF predict_ev() crash |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| BENT-01 | 02-01 | 単勝予測にBenter組み合わせを実装する | SATISFIED | WinBenterGate class with full Benter combination pipeline, race normalization, edge calculation |
| BENT-02 | 02-02 | Beta/Isotonicキャリブレーションを比較し最適手法を採用する | SATISFIED | compare_calibrations() with dual metrics, winner selection, pipeline integration |
| BENT-03 | 02-01 | Benter組み合わせ後の確率をレース単位で正規化する | SATISFIED | Race normalization in apply() with sum==1.0 verification via tests |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| src/models/win_benter_gate.py | 127 | Inconsistent NaN handling: generate_win_oof_predictions uses np.where+clip without NaN->0.5 replacement, unlike extract_market_probability | Warning | Functional difference from extract_market_probability, but valid mask filters NaN out. No functional impact. |
| src/models/win_benter_gate.py | 112-118 | Bug: train_hit_model() only called, predict_ev() requires return_model | Blocker | Production pipeline crashes when generating Win Benter OOF predictions |

### Human Verification Required

None -- all critical behaviors are testable programmatically and have been tested.

### Gaps Summary

**1 critical gap found in `generate_win_oof_predictions()`:**

The OOF prediction generation function has a method call mismatch. It trains only the hit model via `fold_model.train_hit_model(fold_train)` (line 115), but then calls `fold_model.predict_ev(fold_val)` (line 117) which requires `self.return_model` to be initialized. Since `train_hit_model()` only sets `self.hit_model`, the `self.return_model` attribute is never created, causing an `AttributeError` at runtime.

**Impact:**
- Training pipeline will crash when `tanodds` is present in data and df has >= 500 rows
- 3 test_training_pipeline.py tests fail with this exact error
- The Win Benter model cannot be trained via the production pipeline
- Calibration comparison depends on OOF output, so it is also blocked

**Inference path is unaffected:** WinBenterGate.apply() in race_predictor.py works correctly for applying a pre-trained Benter model. The bug is isolated to the training path.

**Fix required:** Change `generate_win_oof_predictions()` to use hit-only prediction instead of `predict_ev()`. The OOF function only needs `p_win_pred` (from hit_model.predict()), not the full EV prediction. Either:
- (a) Call `fold_model.hit_model.predict(features)` directly after train_hit_model(), or
- (b) Add `fold_model.train_return_model(fold_train)` before predict_ev() -- but this is wasteful since return_model output is discarded

Option (a) is correct: the OOF loop only needs the fundamental probability p_win_pred for Benter fitting, not the full EV decomposition.

---

_Verified: 2026-05-02T12:00:00Z_
_Verifier: Claude (gsd-verifier)_
