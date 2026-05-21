---
phase: 19-ev-calibration
verified: 2026-05-07T22:45:00Z
status: gaps_found
score: 9/10 must-haves verified
overrides_applied: 0
re_verification: false
gaps:
  - truth: "ModelLoaderがIsotonicモデルとオッズバンドスケーリング係数を読み込める"
    status: partial
    reason: "load_from_dir() で RaceQualityScreener の初期化3行がコミット22efbc2で誤削除され、quality変数がNameErrorとなる。MLflow load_pathは正常動作。既存テスト test_model_loader_ensemble_override_loads_joblib がFAILしている。"
    artifacts:
      - path: "src/db/model_loader.py"
        issue: "lines 755-756: quality変数未定義でNameError。RaceQualityScreener()の初期化3行が欠落。"
    missing:
      - "load_from_dir() 内のfor loop直後に quality = RaceQualityScreener() / quality.model = self._load_lgbm(race_quality.lgb) の3行を復元"
---

# Phase 19: EV推定キャリブレーション Verification Report

**Phase Goal:** PxE分解の独立性仮定に依存せず、OOF予測ベースでEVを直接キャリブレーションし、全セグメントのEV過大評価倍率を1.0+/-0.2に収束させる
**Verified:** 2026-05-07T22:45:00Z
**Status:** gaps_found
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | OOF EV予測が_train_submodel()内でK-foldループにより生成される | VERIFIED | `generate_ev_oof_predictions()` in training_pipeline.py:936-975. KFold(n_splits=5, shuffle=False). Called at line 555. |
| 2 | IsotonicRegressionがOOF ev_win_correctedをactual_returnにキャリブレーションする | VERIFIED | `fit_ev_calibration()` in training_pipeline.py:978-1009. IsotonicRegression(y_min=0, out_of_bounds="clip").fit(oof_ev[valid], oof_actual[valid]). |
| 3 | オッズバンド別median残差比スケーリング係数がOOF残差から算出される | VERIFIED | fit_ev_calibration() lines 1000-1009: OddsBandFilter.BANDS別にmedian residual ratio算出, MIN_SAMPLES=50閾値あり. |
| 4 | Isotonicモデルとオッズバンドスケーリング係数がサーフェス別にSubmodelSetに保存される | VERIFIED | SubmodelSet: ev_isotonic_calibrator (line 259), ev_odds_band_scales (line 260) in models.py. _train_submodel() return at line 931-932. _save_models_local() saves ev_isotonic_{surface}.joblib + ev_odds_band_scales_{surface}.json at lines 1447-1456. |
| 5 | ModelLoaderがIsotonicモデルとオッズバンドスケーリング係数を読み込める | FAILED | MLflow load_path (lines 290-312) works correctly. load_from_dir() (lines 710-753) loads files correctly BUT commit 22efbc2 accidentally deleted RaceQualityScreener initialization, causing NameError at line 756. 1 test FAILS. |
| 6 | correct_ev()がIsotonic + オッズバンドスケーリングを適用してev_win_calibratedを生成する | VERIFIED | ev_correction_model.py:343-367. Isotonic transform (344-352), band scaling (355-367). ev_win_calibrated always generated. |
| 7 | Isotonic未設定時はev_win_calibrated = ev_win_correctedでフォールバックする | VERIFIED | ev_correction_model.py:352 "else: df['ev_win_calibrated'] = df['ev_win_corrected'].copy()". Test test_correct_ev_no_isotonic_fallback passes. |
| 8 | OOF EV生成がK-fold(shuffle=False)でlook-ahead biasなく動作する | VERIFIED | training_pipeline.py:950 "KFold(n_splits=n_splits, shuffle=False)". Test test_generate_ev_oof_no_shuffle verifies shuffle=False. |
| 9 | fit_ev_calibration()がIsotonicとオッズバンドスケーリングを正しく返す | VERIFIED | training_pipeline.py:978-1009. Returns (IsotonicRegression, dict[str, float]). Tests verify band scaling per band. |
| 10 | 全テストがDB不要(mockベース)で実行される | VERIFIED | 43 tests in test_ev_isotonic.py + test_ev_correction.py all pass. Mock-based, no DB access. |

**Score:** 9/10 truths verified

### ROADMAP Success Criteria Assessment

These are runtime metrics that require a backtest run to verify numerically. They cannot be confirmed via code inspection alone.

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 1 | IsotonicRegressionでOOF ev_winをactual_returnにキャリブレーションし、ECEが改善 | VERIFIED (infrastructure) | Code implements Isotonic OOF calibration. ECE improvement is a runtime metric -- requires backtest to measure. Deferred to Phase 22. |
| 2 | 高オッズ帯(20+)のEV過大評価倍率が2.08から1.2以下に改善 | UNCERTAIN (runtime) | Requires backtest with real data. Infrastructure in place. |
| 3 | 全セグメントのEV過大評価倍率が1.0+/-0.2に収束 | UNCERTAIN (runtime) | Requires backtest with real data. Infrastructure in place. |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/domain/models.py` | SubmodelSet with ev_isotonic_calibrator + ev_odds_band_scales | VERIFIED | Lines 258-260. IsotonicRegression \| None, dict[str, float] \| None. |
| `src/pipelines/training_pipeline.py` | OOF EV generation + Isotonic fit + band scales | VERIFIED | generate_ev_oof_predictions() (936-975), fit_ev_calibration() (978-1009), _train_submodel() integration (550-576), _save_models_local() (1446-1456). |
| `src/db/model_loader.py` | Isotonic model loading + band scale loading | FAILED | MLflow path: correct. load_from_dir(): RaceQualityScreener initialization accidentally deleted, causing NameError. |
| `src/models/ev_diagnostics.py` | EV_PRED_COLUMN = "ev_win_calibrated" with fallback | VERIFIED | Line 22: EV_PRED_COLUMN = "ev_win_calibrated". Lines 184-200: fallback chain for ev_win_corrected, EV_lower_win, edge_win. |
| `src/models/ev_correction_model.py` | Isotonic + band scaling in correct_ev() | VERIFIED | __init__ (136-146), correct_ev() Isotonic (343-352), band scaling (355-367). |
| `tests/test_ev_isotonic.py` | 4 classes, 18 tests | VERIFIED | 4 classes, 18 tests, all pass. |
| `tests/test_ev_correction.py` | Extended with ev_win_calibrated tests | VERIFIED | 4 new tests added. 10 occurrences of "ev_win_calibrated". All pass. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| training_pipeline.py | models.py | SubmodelSet(ev_isotonic_calibrator=, ev_odds_band_scales=) | WIRED | Line 931-932 in _train_submodel() return. |
| model_loader.py | models.py | SubmodelSet(ev_isotonic_calibrator=, ev_odds_band_scales=) | WIRED | Lines 752-753 in load_from_dir(). Lines 332-333 in MLflow path. |
| model_loader.py | ev_correction_model.py | ev_corr.ev_isotonic_calibrator = loaded_calibrator | WIRED | Lines 730-731 (load_from_dir), lines 311-312 (MLflow). |
| test_ev_isotonic.py | ev_correction_model.py | EVCorrectionModel correct_ev() with Isotonic + band scales | WIRED | Imports EVCorrectionModel, tests Isotonic/band/fallback/integration. |
| test_ev_isotonic.py | training_pipeline.py | generate_ev_oof_predictions + fit_ev_calibration | WIRED | TestOOFEVGeneration patches and calls generate_ev_oof_predictions. |
| ev_diagnostics.py | ev_correction_model.py | EV_PRED_COLUMN = "ev_win_calibrated" with fallback | WIRED | Fallback to ev_win_corrected when calibrated column absent. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| ev_correction_model.py correct_ev() | ev_win_calibrated | ev_win_corrected via IsotonicRegression.transform + band_scale multiplication | Yes (from live PxE values) | FLOWING |
| training_pipeline.py generate_ev_oof_predictions() | oof_ev_corrected | WinTwoStageModel + EVCorrectionModel per fold | Yes (from full retrain per fold) | FLOWING |
| training_pipeline.py fit_ev_calibration() | (IsotonicRegression, band_scales) | OOF EV residuals by odds band | Yes (from OOF actual returns) | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| test_ev_isotonic.py 18 tests | python -m pytest tests/test_ev_isotonic.py -v | 18 passed | PASS |
| test_ev_correction.py 25 tests | python -m pytest tests/test_ev_correction.py -v | 25 passed (includes 4 new) | PASS |
| All test suite | python -m pytest tests/ -q | 1348 passed, 1 failed, 1 skipped | FAIL |
| Failing test | test_model_loader_ensemble_override_loads_joblib | NameError: quality not defined | FAIL |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| EVC-01 | 19-01 | OOF予測ベースのEV直接キャリブレーション (Isotonic Regression) | SATISFIED | generate_ev_oof_predictions() + fit_ev_calibration() with IsotonicRegression(y_min=0, out_of_bounds="clip"). KFold(shuffle=False). |
| EVC-02 | 19-01 | オッズバンド別EV補正層の追加 | SATISFIED | fit_ev_calibration() computes median residual ratio per OddsBandFilter.BANDS. correct_ev() applies band scales to ev_win_calibrated. |
| EVC-03 | 19-02 | EVCorrectionModelへの統合 + テスト | SATISFIED | __init__ accepts calibrator/scales. correct_ev() applies both. 18 + 4 = 22 new tests. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| src/db/model_loader.py | 756 | NameError: quality not defined -- RaceQualityScreener init deleted | Blocker | load_from_dir() crashes at runtime. 1 existing test fails. |
| src/models/ev_correction_model.py | 360 | Delayed import OddsBandFilter inside correct_ev() loop | Info | Intentional for circular dependency avoidance. No real performance impact. |

### Human Verification Required

### 1. Runtime EV Calibration Effectiveness

**Test:** Run full training pipeline (run_train.py) + backtest (run_backtest.py) with Phase 19 enabled
**Expected:** ECE improvement; high-odds band EV overestimation ratio <= 1.2
**Why human:** Requires full training run (~17min) + backtest (~57min) with real database data. Cannot verify programmatically without production data.

### 2. Regression from Deleted RaceQualityScreener Code

**Test:** Run backtest with models loaded from local directory (not MLflow)
**Expected:** load_from_dir() should succeed without NameError
**Why human:** The NameError is confirmed via existing test failure. Requires code fix, then integration testing with backtest pipeline.

### Gaps Summary

Phase 19のインフラ実装はほぼ完全に達成されている。OOF EV生成、Isotonicキャリブレーション、オッズバンド別スケーリング、EVCorrectionModel統合、テストスイート全てが正しく実装されている。

唯一のブロッカーはコミット 22efbc2 (Phase 19 review fix) で `load_from_dir()` 内の RaceQualityScreener 初期化3行が誤って削除されたこと。これにより `quality` 変数が未定義となり、`load_from_dir()` が実行時に NameError でクラッシュする。MLflow 経由のロードパスは正常動作する。

**修正内容**: `src/db/model_loader.py` の `for surface in surfaces:` ループ終了後 (line 754付近) に以下3行を復元:

```python
        # RaceQualityScreener
        quality = RaceQualityScreener()
        quality.model = self._load_lgbm(str(models_dir / "race_quality.lgb"))
```

---

_Verified: 2026-05-07T22:45:00Z_
_Verifier: Claude (gsd-verifier)_
