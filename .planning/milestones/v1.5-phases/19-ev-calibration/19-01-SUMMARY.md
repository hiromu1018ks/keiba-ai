---
phase: 19-ev-calibration
plan: 01
subsystem: ev-calibration
tags: [isotonic, ev-calibration, oof-prediction, odds-band-scaling]
dependency_graph:
  requires: []
  provides: [ev_isotonic_calibrator, ev_odds_band_scales, ev_win_calibrated]
  affects: [SubmodelSet, EVCorrectionModel, TrainingPipelineV5, ModelLoader, ev_diagnostics]
tech-stack:
  added: [sklearn.isotonic.IsotonicRegression, KFold-OOF EV generation]
  patterns: [Isotonic calibration, Odds band residual scaling, Column fallback chain]
key-files:
  created: []
  modified:
    - src/domain/models.py
    - src/pipelines/training_pipeline.py
    - src/db/model_loader.py
    - src/models/ev_diagnostics.py
    - src/models/ev_correction_model.py
decisions:
  - IsotonicRegression(y_min=0, out_of_bounds="clip") でEV負値防止 + 範囲外クリップ
  - KFold(shuffle=False) でlook-ahead bias防止 (時系列順序保持)
  - MIN_SAMPLES=50 未満のバンドはスケール1.0で外れ値耐性確保
  - ev_win_calibrated常に生成 + ev_win_corrected保持で下位互換性維持
  - OddsBandFilter遅延importで循環依存回避
metrics:
  duration: 2158s
  completed: "2026-05-07T12:35:24Z"
  tasks: 2
  files: 5
  tests_passed: 1327
---

# Phase 19 Plan 01: EV Isotonic Calibration Base + Odds Band Scaling Summary

OOF予測ベースのIsotonic EVキャリブレーション基盤とオッズバンド別補正層を構築。TrainingPipelineV5内で5-fold OOF EV予測を生成し、IsotonicRegressionでev_win_correctedをactual_returnにキャリブレーションし、オッズバンド別median残差比でセグメントバイアスを是正する。

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | SubmodelSet拡張 + OOF EV生成 + Isotonic/band scale fit | 5cf91b7 | models.py, training_pipeline.py, model_loader.py, ev_diagnostics.py |
| 2 | correct_ev()へのIsotonic + オッズバンド補正統合 | be90575 | ev_correction_model.py |

## Key Changes

### Task 1: SubmodelSet拡張 + パイプライン統合

- **SubmodelSet**: `ev_isotonic_calibrator: IsotonicRegression | None` と `ev_odds_band_scales: dict[str, float] | None` を追加。既存のPxE補正フローに影響なし。
- **generate_ev_oof_predictions()**: 5-fold KFold(shuffle=False)でOOF EV予測を生成。各foldでWinTwoStageModel + EVCorrectionModelを再学習し、ev_win_correctedをOOF生成。
- **fit_ev_calibration()**: IsotonicRegression(y_min=0, out_of_bounds="clip")でEVのIsotonicキャリブレーション + OddsBandFilter.BANDS別にmedian残差比スケーリング係数を算出(MIN_SAMPLES=50)。
- **_train_submodel()**: EV補正後にev_isotonic_oof + ev_isotonic_fitを追加。500+サンプル + confirmed_odds列存在時のみ実行。
- **_save_models_local()**: ev_isotonic_{surface}.joblib + ev_odds_band_scales_{surface}.jsonを保存。
- **ModelLoader**: load_from_dir() と MLflow load() の両方にEV Isotonic/band scalesの読み込みを追加。
- **ev_diagnostics.py**: EV_PRED_COLUMNを"ev_win_calibrated"に更新。フォールバックチェーンで"ev_win_corrected"列もサポート。

### Task 2: EVCorrectionModel統合

- **__init__()**: ev_isotonic_calibrator/ev_odds_band_scalesを受け取るコンストラクタを追加。OddsBandFilterの遅延importで循環依存回避。
- **correct_ev()**: ev_win_corrected生成後にIsotonic transformでev_win_calibrated列を生成。オッズバンド別スケーリングを乗算。IsotonicがNoneの場合はev_win_calibrated = ev_win_correctedでフォールバック。

## Deviations from Plan

None - plan executed exactly as written.

## Verification Results

- Tests: 1327 passed, 1 skipped
- grep verification: all counts meet or exceed thresholds
- `ev_isotonic_calibrator` in models.py: 1
- `ev_odds_band_scales` in models.py: 1
- `generate_ev_oof_predictions` in training_pipeline.py: 2
- `fit_ev_calibration` in training_pipeline.py: 2
- `ev_isotonic` in model_loader.py: 8
- `ev_win_calibrated` in ev_correction_model.py: 4

## Self-Check: PASSED

All 6 modified files exist. Both task commits (5cf91b7, be90575) found in git history.
