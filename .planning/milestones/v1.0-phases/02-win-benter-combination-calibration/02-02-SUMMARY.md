---
phase: 02-win-benter-combination-calibration
plan: 02
subsystem: models
tags: [calibration, beta, isotonic, ece, reliability-diagram, brier-score]
dependency_graph:
  requires: [win_benter_gate, benter_combination, training_pipeline]
  provides: [calibration_comparison, ece_computation, reliability_diagram, beta_calibration]
  affects: [training_pipeline, win_benter_gate]
tech_stack:
  added: [betacal>=1.0, sklearn.calibration.calibration_curve, sklearn.metrics.brier_score_loss]
  patterns: [TDD, Beta vs Isotonic comparison, ECE metric, reliability diagram, temperature scaling gate]
key_files:
  created: []
  modified:
    - src/models/win_benter_gate.py
    - src/pipelines/training_pipeline.py
    - tests/test_win_benter_gate.py
    - pyproject.toml
decisions:
  - "Beta calibration 3-parameter (betacal)を推奨、Isotonicは5%以上Brier改善時のみ選択"
  - "Temperature scalingはBrier Score改善時のみ適用(D-06)"
  - "betacal import失敗時はBetaCalibrationManual(scipy)でフォールバック"
  - "キャリブレーション比較は時系列80/20分割でデータリーク防止(T-02-04)"
metrics:
  duration: "9m"
  completed: "2026-05-02"
  tasks_completed: 2
  files_modified: 4
  tests_added: 6
---

# Phase 2 Plan 02: Beta vs Isotonic Calibration Comparison Summary

Beta calibration(3-parameter)とIsotonic calibrationをBrier Score + ECEで定量的に比較する機能を実装し、学習パイプラインに統合した。信頼性ダイアグラムデータ生成、ECE計算、温度スケーリングの条件付き適用を含む。

## Changes

### Task 1: Calibration comparison with ECE + reliability diagram (TDD)

**Modified `src/models/win_benter_gate.py`:**
- `compute_ece()`: ECE (Expected Calibration Error) をn_bins=10で計算。Guo et al., 2017に基づく
- `BetaCalibrationManual`: betacalパッケージ互換性問題時のscipy.optimizeフォールバック
- `compare_calibrations()`: Beta(3-param) vs IsotonicをBrier Score + ECEで比較。時系列80/20分割でリーク防止。勝者を選択してキャリブレーターを返す
- `generate_reliability_data()`: sklearn.calibration_curveを使って信頼性ダイアグラム用データを生成

**Modified `tests/test_win_benter_gate.py`:**
- 6テスト追加: ECE非負返却、完全キャリブレーションECE≈0、必須キー確認、Brier Score勝者選択、信頼性データキー確認、完全キャリブレーション一致

**Modified `pyproject.toml`:**
- `betacal>=1.0` をdependenciesに追加

### Task 2: Training pipeline integration

**Modified `src/pipelines/training_pipeline.py`:**
- Win Benter学習後に`compare_calibrations()`を呼び出し、Beta vs Isotonicを比較
- 勝者キャリブレーターを`win_isotonic_cal`に格納(Beta/Isotonic自動選択)
- `generate_reliability_data()`で信頼性ダイアグラムデータをログ出力
- Temperature scalingはBrier Score改善時のみ適用(D-06)、改善なき場合はスキップ
- キャリブレーション比較は`win_benter is not None and len(oof_p_fund) >= 500`の条件下でのみ実行

## Deviations from Plan

None - plan executed exactly as written.

## TDD Gate Compliance

- RED gate: `c739de2` - test(02-02): 6 failing tests for calibration comparison
- GREEN gate: `b246df5` - feat(02-02): implementation passes all 15 tests
- No REFACTOR needed - code is clean

## Test Results

- **WinBenterGate tests:** 15/15 passed (6 new calibration + 9 existing)
- **Regression tests (benter_combination + ev_correction):** 34/34 passed
- **Total:** 49/49 passed, 0 failures
- **Known pre-existing failures (out of scope):** 3 in test_training_pipeline.py (WinTwoStageModel mock issue from Plan 01)

## Verification

1. `grep -c "compare_calibrations" src/pipelines/training_pipeline.py` = 2
2. `grep -c "generate_reliability_data" src/pipelines/training_pipeline.py` = 2
3. `grep -c "betacal" pyproject.toml` = 1
4. `grep -c "win_isotonic_cal" src/pipelines/training_pipeline.py` = 9

## Threat Flags

No new threat surfaces beyond plan's threat_model. Time-series 80/20 split prevents data leakage (T-02-04 mitigated). betacal has manual fallback (T-02-05 accepted). Temperature scaling gated on Brier Score improvement (T-02-06 mitigated).

## Self-Check: PASSED

- All 4 key files verified modified in git log
- 3 task commits verified (c739de2 RED, b246df5 GREEN, e621aac pipeline)
- 49/49 tests passing
