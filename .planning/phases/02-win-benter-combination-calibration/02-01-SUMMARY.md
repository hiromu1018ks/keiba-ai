---
phase: 02-win-benter-combination-calibration
plan: 01
subsystem: models
tags: [benter, win, calibration, pipeline-integration]
dependency_graph:
  requires: [ev_correction_model, benter_combination, two_stage_return_model]
  provides: [win_benter_gate, win_benter_integration]
  affects: [training_pipeline, model_loader, race_predictor, domain_models]
tech_stack:
  added: [numpy, pandas, sklearn.model_selection.KFold, scipy.optimize.minimize]
  patterns: [TDD, Benter logit combination, OOF prediction, race normalization]
key_files:
  created:
    - src/models/win_benter_gate.py
    - tests/test_win_benter_gate.py
  modified:
    - src/domain/models.py
    - src/pipelines/training_pipeline.py
    - src/db/model_loader.py
    - src/backtest/race_predictor.py
    - tests/test_race_predictor.py
decisions:
  - "extract_market_probabilityでNaN/ゼロオッズを0.5で補完後にクリップ (planのNaNクリップでは不十分)"
  - "grid searchでNLL最小の初期値を探索するD-13パターンを採用"
  - "race_predictorのWin Benter適用はgetattr()で安全チェック"
metrics:
  duration: "8m"
  completed: "2026-05-02"
  tasks_completed: 2
  files_modified: 6
  tests_added: 9
---

# Phase 2 Plan 01: WinBenterGate + Pipeline Integration Summary

WinBenterGateクラスを実装し、Benter組み合わせ(alpha*logit(p_fund) + beta*logit(p_market) + gamma)による単勝確率ブレンド、レース正規化(Sigma P=1.0)、OOF予測生成を学習/推論/保存/読み込みパイプラインに完全統合した。

## Changes

### Task 1: WinBenterGate class (TDD)

**Created `src/models/win_benter_gate.py`:**
- `WinBenterGate` クラス: `apply()` メソッドで Benter combine -> calibration -> temperature scaling -> race normalization -> edge計算を実行
- `extract_market_probability()` static method: tanoddsを市場確率に変換(クリップ[0.01, 0.99])
- `generate_win_oof_predictions()` 関数: KFold CVでリークなしOOF予測生成

**Modified `src/domain/models.py`:**
- `SubmodelSet` dataclassに3フィールド追加: `win_benter`, `win_isotonic_calibrator`, `win_temperature_scaler` (全て `| None = None`)

**Created `tests/test_win_benter_gate.py`:**
- 9テスト: 市場確率変換、結合確率範囲、レース正規化、edge計算、OOF予測生成、SubmodelSetフィールド

### Task 2: Pipeline Integration

**`src/pipelines/training_pipeline.py`:**
- Win Benter学習ブロック追加: OOF予測生成 -> グリッドサーチ(alpha/beta/gamma) -> BenterCombination生成
- SubmodelSet構築にwin_*フィールド追加
- モデル保存: `benter_combo_win_{surface}.json`, `isotonic_win_{surface}.joblib`, `temp_scale_win_{surface}.json`

**`src/db/model_loader.py`:**
- Win Benter/Isotonic/TempScale読み込み + SubmodelSet構築に統合

**`src/backtest/race_predictor.py`:**
- EV補正後にWinBenterGate.apply()を適用
- `p_win_combined`, `p_win_final`, `edge_win` 列を生成

**`tests/test_race_predictor.py`:**
- モックSubmodelSetに `win_benter=None`, `win_isotonic_calibrator=None`, `win_temperature_scaler=None` 追加 (MagicMock自動属性アクセス回避)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] extract_market_probability NaN handling**
- **Found during:** Task 1 GREEN phase
- **Issue:** `np.where(tanodds > 0, ..., np.nan)` produces NaN which `np.clip` passes through as-is, not a valid probability
- **Fix:** Added intermediate NaN replacement to 0.5 before clipping to [0.01, 0.99]
- **Files modified:** `src/models/win_benter_gate.py`
- **Commit:** 6a149c9

**2. [Rule 3 - Blocking] test helper _make_race_df array length mismatch**
- **Found during:** Task 1 GREEN phase
- **Issue:** When n_races * horses_per_race != base list length (e.g., 3*6 vs 5-element lists), DataFrame construction fails
- **Fix:** Changed to dynamic length calculation with slice
- **Files modified:** `tests/test_win_benter_gate.py`
- **Commit:** 6a149c9

**3. [Rule 2 - Missing] test_race_predictor mock missing win_benter=None**
- **Found during:** Task 2 verification
- **Issue:** MagicMock auto-creates attributes, so `getattr(submodel, "win_benter", None)` returns a MagicMock instead of None, triggering WinBenterGate.apply() on test data without p_win_corrected column
- **Fix:** Added `win_benter=None`, `win_isotonic_calibrator=None`, `win_temperature_scaler=None` to mock helper
- **Files modified:** `tests/test_race_predictor.py`
- **Commit:** 5b5717f

## Test Results

- **WinBenterGate tests:** 9/9 passed
- **Regression tests (benter_combination + ev_correction + race_predictor):** 62/62 passed
- **Total:** 71/71 passed, 0 failures

## Verification

1. `grep -c "win_benter" src/domain/models.py` = 1
2. `grep -c "benter_combo_win" src/pipelines/training_pipeline.py` = 1
3. `grep -c "benter_combo_win" src/db/model_loader.py` = 1
4. `grep -c "WinBenterGate" src/backtest/race_predictor.py` = 2

## Threat Flags

No new threat surfaces beyond plan's threat_model. OOF uses KFold with shuffle=False (T-02-01 mitigated). tanodds column name verified (T-02-02 mitigated). Grid search falls back to standard fit on convergence failure (T-02-03 accepted).

## Self-Check: PASSED

- All 8 key files verified present
- 2 task commits verified in git log (6a149c9, 5b5717f)
- 71/71 tests passing (9 WinBenterGate + 62 regression)
