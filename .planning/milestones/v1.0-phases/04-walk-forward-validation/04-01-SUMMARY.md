---
phase: 04-walk-forward-validation
plan: 01
subsystem: validation
tags: [walk-forward, overfitting-detection, feature-importance, spearman, mlflow]
dependency_graph:
  requires: [src/models/walk_forward_cv.py, src/backtest/engine.py, src/pipelines/training_pipeline.py]
  provides: [src/models/walk_forward_cv.py::FoldResult, src/models/walk_forward_cv.py::WFValidationResult, scripts/run_wf_validation.py]
  affects: []
tech_stack:
  added: [scipy.stats.spearmanr, lightgbm.Booster.feature_importance]
  patterns: [dataclass-extensions, spearman-rank-correlation, mlflow-experiment-tracking]
key_files:
  created:
    - scripts/run_wf_validation.py
  modified:
    - src/models/walk_forward_cv.py
    - tests/test_walk_forward_cv.py
decisions:
  - AbilityModel.models(dict)の反復処理に変更(単一model属性ではなく複数Booster)
  - LightGBM feature_namesはDatasetコンストラクタで設定(lgb.trainには渡せない)
metrics:
  duration_min: 5
  completed: "2026-05-03"
---

# Phase 4 Plan 01: Walk-Forward Validation Infrastructure Summary

Walk-forward検証インフラを実装: FoldResult/WFValidationResultデータクラス、過学習検出ユーティリティ(extract_feature_ranking/compute_feature_stability/judge_overfitting)、2フォールドCLIスクリプト(run_wf_validation.py)

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | WFValidationResultデータクラスと過学習検出ユーティリティ | a140ada | src/models/walk_forward_cv.py, tests/test_walk_forward_cv.py |
| 2 | run_wf_validation.py CLIスクリプト | 4a2f854 | scripts/run_wf_validation.py |

## Key Changes

### Task 1: データクラスと過学習検出ユーティリティ

- `FoldResult`: 単一フォールド結果(ROI gap、ベット数、特徴量ランキング)
- `WFValidationResult`: 全体結果(プールROI、加重ROI、Spearman rho、3基準verdict)
- `extract_feature_ranking()`: LightGBM Boosterからtop-N特徴量順位を取得
- `compute_feature_stability()`: scipy.stats.spearmanrでフォールド間特徴量順位相関を計算
- `judge_overfitting()`: ROI gap(20%/30%閾値) + 一貫性 + 安定性(rho>=0.5)の3基準判定
- 既存のFold/CVResult/WalkForwardCVクラスは一切変更せず(追加のみ)
- テスト12件追加(全29テスト通過)

### Task 2: run_wf_validation.py CLIスクリプト

- FOLDS定数: 2フォールド(2020-2023->2024, 2021-2024->2025)
- 各フォールド: TrainingPipelineV5.run() + BacktestEngine(train期間/test期間別インスタンス)
- Feature importance: 芝/ダートstage1.models(dict) + win.hit_modelから統合抽出
- プールROI(総払戻/総投資) + ベット数加重ROI
- フォールド完了ごとに途中結果をJSONに書き出し(クラッシュ耐性)
- MLflow experiment "wf_validation" にパラメータ/メトリクス/tag記録
- 最終結果: data/backtest/wf_validation_result.json

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] LightGBM 4.6.0 lgb.train() API非互換**
- **Found during:** Task 1 テスト実行時
- **Issue:** `lgb.train(feature_names=[...])` が TypeError。LightGBM 4.6.0ではfeature_namesをDatasetコンストラクタに渡す必要がある
- **Fix:** `lgb.Dataset(data, label, feature_name=[...])` に変更
- **Files modified:** tests/test_walk_forward_cv.py
- **Commit:** a140ada

**2. [Rule 1 - Bug] AbilityModel.model属性が存在しない**
- **Found during:** Task 2 実装時のコード調査
- **Issue:** 計画では `sub.stage1.model` を参照していたが、実際は `sub.stage1.models` (dict[str, lgb.Booster]) で複数Boosterを保持
- **Fix:** `_extract_all_feature_rankings()` で `sub.stage1.models.items()` を反復処理
- **Files modified:** scripts/run_wf_validation.py
- **Commit:** 4a2f854

## Test Results

```
tests/test_walk_forward_cv.py: 29 passed (17 existing + 12 new)
```

## Self-Check: PASSED

- FOUND: src/models/walk_forward_cv.py
- FOUND: scripts/run_wf_validation.py
- FOUND: tests/test_walk_forward_cv.py
- FOUND: commit a140ada
- FOUND: commit 4a2f854
