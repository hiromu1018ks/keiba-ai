---
phase: 01-feature-analysis-enhancement
plan: 01
subsystem: features
tags: [shap, feature-importance, noise-detection, lightgbm, tdd]
dependency_graph:
  requires: [WinTwoStageModel.hit_model]
  provides: [analyze_feature_importance, identify_noise_features, validate_noise_removal, remove_noise_features]
  affects: [WinTwoStageModel.FEATURE_COLS]
tech_stack:
  added: []
  patterns: [LightGBM pred_contrib TreeSHAP, sklearn.metrics log_loss/roc_auc_score]
key_files:
  created:
    - src/features/win_feature_analysis.py
    - scripts/analyze_feature_importance.py
    - tests/test_win_feature_analysis.py
  modified:
    - src/models/two_stage_return_model.py
decisions:
  - LightGBM native pred_contrib=True を使用 (外部shapパッケージ不要)
  - ノイズ判定は mean_abs_shap < threshold AND gain <= threshold の両条件必須
  - ノイズ除外はFEATURE_COLSのみ、共有計算モジュールは変更しない
  - 初期コミットでは推測的な特徴量除外は行わず、分析インフラのみ実装
metrics:
  duration: 9m
  completed: "2026-05-02"
  tasks: 2
  files: 4
  tests: 18
  commits: 4
---

# Phase 1 Plan 01: 特徴量重要度分析・ノイズ除外 Summary

SHAP/gain特徴量重要度分析モジュールとCLI解析スクリプトを実装。LightGBMネイティブpred_contribでTreeSHAP値を取得し、ノイズ特徴量を特定・除外する基盤を構築。

## Completed Tasks

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | SHAP/gain特徴量重要度分析モジュールの実装 | `8c3235b` (RED), `fda8adb` (GREEN) | src/features/win_feature_analysis.py, tests/test_win_feature_analysis.py |
| 2 | 特徴量解析スクリプトの作成とノイズ除外によるFEATURE_COLS最適化 | `1ef3e24` (RED), `fa7487e` (GREEN) | scripts/analyze_feature_importance.py, src/models/two_stage_return_model.py, src/features/win_feature_analysis.py, tests/test_win_feature_analysis.py |

## Key Deliverables

### analyze_feature_importance (win_feature_analysis.py)
- LightGBM `feature_importance('gain')` + `predict(pred_contrib=True)` でSHAP/gainランキングを生成
- pred_contribのexpected value列を正しく除外 (shape `[n_samples, n_features+1]`)
- `top_n` パラメータで上位N件のみ返却可能

### identify_noise_features (win_feature_analysis.py)
- `mean_abs_shap < threshold AND gain <= threshold` でノイズ判定
- デフォルト閾値: SHAP=0.001, gain=0.0
- 両条件を満たす特徴量のみをノイズとして返す (AND条件)

### validate_noise_removal (win_feature_analysis.py)
- ノイズ除外前後のlogloss/AUCを比較する再学習検証関数
- logloss悪化0.5%超で警告ログを出力 (T-01-03 mitigation)
- 戻り値: `{original_logloss, new_logloss, original_auc, new_auc}`

### WinTwoStageModel.remove_noise_features (two_stage_return_model.py)
- クラスメソッドでFEATURE_COLSからノイズ特徴量を除外
- ログ出力あり (除外件数、除外前後の特徴量数)
- 存在しない特徴量名は無視 (エラーなし)

### scripts/analyze_feature_importance.py
- CLI解析エントリーポイント (`--model-dir`, `--shap-threshold`, `--gain-threshold`, `--output`, `--top-n`, `--auto-exclude`)
- CSVレポート出力 (columns: feature, gain, mean_abs_shap, is_noise)
- `--auto-exclude` フラグで自動除外 + 再学習検証を実行

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] テストのmock predict戻り値shape不一致**
- **Found during:** Task 2 GREEN phase
- **Issue:** テストのmock_model.predict()の戻り値配列長がDataFrame行数と不一致 (shape (6,) vs (2,))
- **Fix:** mockの戻り値をDataFrame行数(2行)に合わせて修正
- **Files modified:** tests/test_win_feature_analysis.py
- **Commit:** `fa7487e`

**2. [Rule 1 - Bug] numpy bool と Python bool のis比較非互換**
- **Found during:** Task 2 GREEN phase
- **Issue:** `np.True_ is True` が False を返す (numpy boolはPythonのis演算子と非互換)
- **Fix:** `is True` -> `== True` に変更
- **Files modified:** tests/test_win_feature_analysis.py
- **Commit:** `fa7487e`

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| LightGBM native pred_contrib 使用 | 外部shapパッケージ不要。依存関係を増やさずTreeSHAP値を取得可能 |
| ノイズ判定にAND条件を採用 | gain高だがSHAP低い特徴量 (gain偏向) とSHAP高だがgain低い特徴量 (分散型) を誤検出から保護 |
| 初期除外なし (分析インフラのみ) | 特徴量除外は実際の分析スクリプト実行後に判断すべき。推測的除外は危険 |

## TDD Gate Compliance

| Gate | Task 1 | Task 2 |
|------|--------|--------|
| RED (test failing) | `8c3235b` - 10 tests | `1ef3e24` - 8 tests |
| GREEN (all pass) | `fda8adb` - 10/10 passed | `fa7487e` - 18/18 passed |
| REFACTOR | Not needed | Not needed |

TDD gate sequence validated: RED -> GREEN commits exist in git log for both tasks.

## Test Results

```
18 passed, 2 warnings in 1.70s
```

- TestAnalyzeFeatureImportance: 5 tests
- TestIdentifyNoiseFeatures: 5 tests
- TestRemoveNoiseFeatures: 5 tests
- TestValidateNoiseRemoval: 2 tests
- TestCSVReportIsNoise: 1 test

## Verification Results

1. `python -m pytest tests/test_win_feature_analysis.py -v` -- 18/18 passed
2. `python scripts/analyze_feature_importance.py --help` -- usage表示確認
3. `analyze_feature_importance` in win_feature_analysis.py: 2 occurrences
4. `identify_noise_features` in win_feature_analysis.py: 1 occurrence
5. `pred_contrib` usage in win_feature_analysis.py: confirmed
6. `shap_matrix[:, :-1]` in win_feature_analysis.py: 1 occurrence
7. `remove_noise_features` in two_stage_return_model.py: 1 occurrence
8. `validate_noise_removal` in win_feature_analysis.py: 1 occurrence
9. `is_noise` in analyze_feature_importance.py: 3 occurrences

## Self-Check: PASSED

- All 5 files exist: src/features/win_feature_analysis.py, scripts/analyze_feature_importance.py, tests/test_win_feature_analysis.py, src/models/two_stage_return_model.py, 01-01-SUMMARY.md
- All 4 commits exist in git log: 8c3235b, fda8adb, 1ef3e24, fa7487e
- All 18 tests pass (0 failures)
