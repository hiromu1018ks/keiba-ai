---
phase: 19-ev-calibration
plan: 02
subsystem: ev-calibration
tags: [isotonic, ev-calibration, odds-band-scaling, test-suite, oof-prediction]
dependency_graph:
  requires: [19-01]
  provides: [test_ev_isotonic, test_ev_correction_extended]
  affects: []
tech-stack:
  added: []
  patterns: [mock-based testing, IsotonicRegression integration testing, KFold shuffle verification]
key-files:
  created:
    - tests/test_ev_isotonic.py
  modified:
    - tests/test_ev_correction.py
decisions:
  - Isotonicテストでは実際のsklearn IsotonicRegressionを使用 (mock不使用) で正確な動作検証
  - OOF生成テストではWinTwoStageModelとEVCorrectionModelをmock化してfold学習チェーンをシミュレート
  - KFold(shuffle=False)テストにはwrapsパターンを使用してrecursive mock問題を回避
metrics:
  duration: 1013s
  completed: "2026-05-07T21:03:28Z"
  tasks: 2
  files: 2
  tests_passed: 1349
---

# Phase 19 Plan 02: EV Isotonic Calibration Test Suite Summary

Isotonic EVキャリブレーション + オッズバンドスケーリング + OOF生成の包括的テストスイートを構築。4クラス18テストの新規ファイルと4テストの拡張で、correct_ev()のIsotonic適用、フォールバック、非負制約、順序維持、バンド別スケーリング、KFold(shuffle=False)ルックアヘッドバイアス防止を全て検証。

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Isotonic + オッズバンド別補正 + OOF生成のテストスイート作成 | 37411c1 | tests/test_ev_isotonic.py |
| 2 | 既存テスト拡張 + 全テスト通過確認 | 53b0c10 | tests/test_ev_correction.py |

## Key Changes

### Task 1: test_ev_isotonic.py (新規)

4クラス18テストのテストスイートを作成:

- **TestEVIsotonicCalibration** (5テスト): Isotonic適用時のev_win_calibrated列生成、高EV過大評価抑制、非負制約(y_min=0)、順序維持(単調増加)、Isotonic未設定時フォールバック
- **TestOddsBandScaling** (4テスト): バンドスケーリング適用(高オッズ帯縮小)、未設定時Isotonicのみ、バンド別スケール検証、odds列不在時スキップ
- **TestOOFEVGeneration** (4テスト): 3配列出力形式、KFold(shuffle=False)検証、race_dateソート、全インデックスカバー(NaNなし)
- **TestEVCorrectionIntegration** (5テスト): フルパイプライン(Isotonic+band)、下位互換性(ev_win_corrected不変)、Isotonic初期化受け入れ、band_scales初期化受け入れ、SubmodelSetデフォルトNone

### Task 2: test_ev_correction.py (拡張)

TestEVCorrectionModelクラスに4テストを追加:

- test_correct_ev_produces_ev_win_calibrated: 列生成確認
- test_correct_ev_calibrated_equals_corrected_without_isotonic: フォールバック値の等価性
- test_correct_ev_with_isotonic_produces_different_calibrated: Isotonic適用時の差異
- test_correct_ev_calibrated_non_negative: 非負制約

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] KFold import path for mock patching**
- **Found during:** Task 1 test execution
- **Issue:** `pipelines.training_pipeline.KFold` patch failed because KFold is imported locally inside the function via `from sklearn.model_selection import KFold`
- **Fix:** Changed patch target to `sklearn.model_selection.KFold` with `wraps=SklearnKFold` pattern to avoid recursive mock
- **Files modified:** tests/test_ev_isotonic.py
- **Commit:** 37411c1

**2. [Rule 1 - Bug] test_ev_corrected_column_unchanged test logic**
- **Found during:** Task 1 test execution
- **Issue:** Test compared fixture's ev_win_corrected with correct_ev() output, but correct_ev() always recalculates ev_win_corrected from P/E correction models
- **Fix:** Changed test to compare ev_win_corrected output between two models (with/without Isotonic) using identical mock boosters to verify Isotonic independence
- **Files modified:** tests/test_ev_isotonic.py
- **Commit:** 37411c1

## Verification Results

- Tests: 1349 passed, 1 skipped (pre-existing), 0 failed
- test_ev_isotonic.py: 18 tests in 4 classes -- all passed
- test_ev_correction.py: 4 new tests added (total now 16 in TestEVCorrectionModel)
- grep "ev_win_calibrated" in test_ev_correction.py: 10 occurrences
- grep "def test_" in test_ev_isotonic.py: 18 tests

## Self-Check: PASSED

Both created/modified files exist:
- tests/test_ev_isotonic.py: EXISTS (18 tests, 4 classes)
- tests/test_ev_correction.py: EXISTS (4 new tests added)

Both task commits found in git history:
- 37411c1: test(19-02): Isotonic EVキャリブレーション + オッズバンドスケーリングテストスイート
- 53b0c10: test(19-02): ev_win_calibrated列の検証テストをtest_ev_correction.pyに追加
