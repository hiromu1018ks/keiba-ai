---
phase: 21-conformal-ev
plan: 01
subsystem: models
tags: [cqr, conformal-prediction, ev-intervals, lightgbm-quantile, model-replacement]
dependency_graph:
  requires: []
  provides: [ConformalEVModel, cqr-ev-prediction-intervals]
  affects: [training_pipeline, model_loader, race_predictor, SubmodelSet]
tech_stack:
  added: [lightgbm-quantile-objective, numpy-quantile-cqr]
  patterns: [cqr-nonconformity-score, 2-alpha-interval, backward-compat-shim]
key_files:
  created:
    - src/models/conformal_ev_model.py
    - tests/test_conformal_ev_model.py
  modified:
    - src/models/__init__.py
    - src/domain/models.py
    - src/pipelines/training_pipeline.py
    - src/db/model_loader.py
    - src/backtest/race_predictor.py
    - tests/test_backtest_engine.py
    - tests/test_domain.py
    - tests/test_ensemble_gate_propagation.py
    - tests/test_mlflow_logging.py
    - tests/test_odds_deviation.py
    - tests/test_parameter_freeze.py
    - tests/test_race_predictor.py
    - tests/test_training_pipeline.py
    - tests/test_win_benter_gate.py
  deleted:
    - src/models/robust_confidence_estimator.py
    - tests/test_robust_confidence_estimator.py
decisions:
  - CQR非適合スコア max(q_low-y, y-q_high) でRomano et al. 2019標準手法を採用
  - 同一q_low/q_highモデルからalpha=0.1と0.2で別々の補正量子を計算(2-alpha構成)
  - Plan 02完全統合までの間、calibrate/predict_lower_bound後方互換shimを維持
  - confidenceフィールド削除に伴い全参照箇所(training_pipeline, model_loader, race_predictor, テスト群)を一括更新
metrics:
  duration: ~15min
  completed: "2026-05-09"
  tasks: 2
  files_modified: 16
  tests_added: 16
  tests_passing: 16/16
---

# Phase 21 Plan 01: ConformalEVModel新規作成 + RobustConfidenceEstimator削除 Summary

CQR (Conformalized Quantile Regression) ベースのConformalEVModelクラスを実装し、既存RobustConfidenceEstimatorを完全に置き換えた。LightGBM quantile regressionで2つの分位点モデル(q_low, q_high)を学習し、CQR非適合スコアでCP補正を適用する。2-alpha構成(90%/80%)で出力列名が既存と互換。

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | ConformalEVModel新規作成 + RobustConfidenceEstimator削除 | f946f7e | conformal_ev_model.py (新規), robust_confidence_estimator.py (削除), __init__.py, models.py |
| 2 | テスト作成 + 既存テスト移行 | 57f385e | test_conformal_ev_model.py (新規), test_robust_confidence_estimator.py (削除), 13ファイル更新 |

## Key Changes

### ConformalEVModel (src/models/conformal_ev_model.py)
- **train()**: LightGBM quantile regressionでq_low(alpha/2)とq_high(1-alpha/2)を学習
- **CQR非適合スコア**: `max(q_low - y, y - q_high)` から補正量子を計算
- **predict_interval()**: RobustConfidenceEstimatorと同じ出力列名(EV_lower_win_corrected等)を生成
- **2-alpha構成**: 90%区間(_calibration_quantile_90)と80%区間(_calibration_quantile_80)を同時に計算
- **モノトonicity clip**: `q_low = np.minimum(q_low, q_high)` でq_low <= q_highを保証
- **EV_lower 0クリップ**: `np.maximum(q_low - Q_calib, 0.0)` で非負を保証
- **save/load**: .lgb形式でモデル保存、.jsonでパラメータ保存

### Backward Compat (Plan 02まで)
- `calibrate()` shim: _calibrated=Trueを設定するだけ（実際のCQR学習はtrain()で実行）
- `predict_lower_bound()`: predict_interval()のラッパー
- 古い属性(rolling_window, _win_cp_quantile等)を定義済み

### Field Migration
- SubmodelSet: `confidence: RobustConfidenceEstimator` -> `conformal_ev_model: ConformalEVModel | None = None`
- 全参照箇所(training_pipeline.py, model_loader.py, race_predictor.py, テスト群)を一括更新

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Updated all references beyond plan scope**
- **Found during:** Task 1 execution
- **Issue:** Plan specified only updating __init__.py and models.py, but deleting RobustConfidenceEstimator caused collection errors in training_pipeline.py and model_loader.py (import errors), and all tests using SubmodelSet.confidence field broke
- **Fix:** Updated imports and field references in training_pipeline.py, model_loader.py, race_predictor.py, and 10 test files. Added backward-compat shim methods (calibrate, predict_lower_bound) and attributes to ConformalEVModel
- **Files modified:** src/pipelines/training_pipeline.py, src/db/model_loader.py, src/backtest/race_predictor.py, tests/*.py
- **Commit:** 57f385e

**2. [Rule 1 - Bug] Fixed numpy/pandas clip() compatibility**
- **Found during:** Task 2 test execution
- **Issue:** `interval_width.clip(lower=1e-6)` failed because numpy arrays from LightGBM predict don't support pandas-style keyword arguments in clip()
- **Fix:** Wrapped numpy array in pd.Series before calling clip()
- **Files modified:** src/models/conformal_ev_model.py
- **Commit:** 57f385e

**3. [Rule 1 - Bug] Fixed syntax error from incomplete method insertion**
- **Found during:** Task 2 test execution
- **Issue:** predict_lower_bound method body and train() method def got merged during editing, causing SyntaxError
- **Fix:** Corrected the method boundary with proper def train() line
- **Files modified:** src/models/conformal_ev_model.py
- **Commit:** 57f385e

## Verification Results

- test_conformal_ev_model.py: 16/16 passed
- Full test suite: 1389/1389 passed (1373 existing + 16 new)
- No regressions introduced

## Notes for Plan 02

- training_pipeline.py: `conf = RobustConfidenceEstimator()` (alias) + `conf.calibrate()` -- needs replacement with `ConformalEVModel.train()` in _train_submodel()
- model_loader.py: Old attribute loading code (confidence.alpha, rolling_window, etc.) -- needs replacement with ConformalEVModel.load()
- race_predictor.py: `submodel.conformal_ev_model.predict_interval()` -- already updated, works as-is
- Backward-compat shims (calibrate, predict_lower_bound) should be removed after Plan 02 integration

## Self-Check: PASSED

- src/models/conformal_ev_model.py: FOUND
- tests/test_conformal_ev_model.py: FOUND
- src/models/__init__.py: FOUND
- src/domain/models.py: FOUND
- src/models/robust_confidence_estimator.py: DELETED
- tests/test_robust_confidence_estimator.py: DELETED
- Commit f946f7e: FOUND
- Commit 57f385e: FOUND
