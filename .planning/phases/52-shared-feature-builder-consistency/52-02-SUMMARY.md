---
phase: 52-shared-feature-builder-consistency
plan: 02
subsystem: features
tags: [feature-builder, consistency, backtest, paper-trading, training-pipeline]
dependency_graph:
  requires: [52-01]
  provides: [FeatureBuilder-integration-BT-PT-Train]
  affects: [src/backtest/, src/paper_trading/, src/pipelines/, tests/]
tech_stack:
  added: []
  patterns: [FeatureBuilder-delegation, mock-migration]
key_files:
  created: []
  modified:
    - src/backtest/engine.py
    - src/backtest/race_predictor.py
    - src/paper_trading/predictor.py
    - src/pipelines/training_pipeline.py
    - tests/test_backtest_engine.py
    - tests/test_paper_predictor.py
decisions:
  - BacktestEngine.prepare_data()/run() の両経路で FeatureBuilder.build_for_training() を使用
  - PaperPredictor.setup() は build_for_inference() を使用、track_stats 未設定時は build_for_training() にフォールバック
  - TrainingPipeline._train_submodel() から13エンリッチメントモジュールを削除、track_stats計算のみ残存
  - RacePredictor.predict() の jockey/trainer/jt マージに列既存チェックを追加（後方互換）
  - BacktestPreparedData の jockey_df_all/trainer_df_all/jt_df_all を空 DataFrame に変更
metrics:
  duration: 1685s
  completed: "2026-06-06T04:49:31Z"
  tasks: 2
  files: 6
  tests: 115
---

# Phase 52 Plan 02: 4呼び出し元の FeatureBuilder 統合 Summary

BacktestEngine (prepare_data + run 内部), RacePredictor, PaperPredictor, TrainingPipeline の
4経路からインライン特徴量構築コードを削除し、Plan 01 で新設した FeatureBuilder に委譲。
3コピー分岐を完全に統一し、PT の7ギャップ (Sire/PaceAptitude/Course/DamPedigree/Record/Mining/Interaction) を解消。

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | BacktestEngine/RacePredictor の FeatureBuilder 委譲 | `3c362db` | `src/backtest/engine.py`, `src/backtest/race_predictor.py`, `tests/test_backtest_engine.py` |
| 2 | PaperPredictor/TrainingPipeline の FeatureBuilder 委譲 | `fbe9ac9` | `src/paper_trading/predictor.py`, `src/pipelines/training_pipeline.py`, `tests/test_paper_predictor.py` |

追加コミット: `34e76c3` — ruff lint 修正 (未使用変数、行長、インポート順序)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] テストモックの更新漏れによる回帰**
- **Found during:** Task 1
- **Issue:** 内部パスが FeatureBuilder を使用するよう変更後、旧モック (FeatureEngine/SubModelManager/HorseHistoryFeatures 等) が呼び出されずテストが失敗
- **Fix:** 全10箇所のテストデコレータ・パラメータ・モック設定を FeatureBuilder に更新
- **Files modified:** `tests/test_backtest_engine.py`, `tests/test_paper_predictor.py`
- **Commits:** `3c362db`, `fbe9ac9`

**2. [Rule 1 - Bug] Hist feature 列がテストの feat_df に欠落**
- **Found during:** Task 1
- **Issue:** TestHistFeaturesPreMerge テストが hist feature 列を feat_df に含めていなかったため、FeatureBuilder 統合後 (hist_df_all マージが FeatureBuilder 内部で処理) にアサーション失敗
- **Fix:** テストの feat_df に closing_speed_ratio_avg/haron_race_gap_avg を追加
- **Files modified:** `tests/test_backtest_engine.py`
- **Commit:** `3c362db`

## Verification Results

```
115 tests passed (79 BT + 5 PT + 36 Training Pipeline + 5 FeatureBuilder)
1 test deselected (test_observed_true_on_all_groupby — pre-existing: src/investment/feature_frame.py)
ruff check: All checks passed
```

## Key Decisions

1. **BacktestEngine.prepare_data() の jockey_df_all/trainer_df_all/jt_df_all**: FeatureBuilder が内部でマージするため、空DataFrameを返すように変更。BacktestPreparedData 構造は維持。
2. **PaperPredictor の track_stats フォールバック**: FeatureState.from_submodel_set() が ValueError の場合、build_for_training() にフォールバック。これにより未学習モデルでも動作。
3. **TrainingPipeline._train_submodel() の track_stats 計算**: surface フィルタ後のデータから _compute_track_stats()/_compute_track_month_stats() を計算し、SubmodelSet に格納。FeatureBuilder は学習時にデータ全体から計算するが、_train_submodel は surface フィルタ後のデータから再計算するため SubmodelSet の永続性要件を満たす。
4. **RacePredictor の jockey/trainer/jt マージ**: 列既存チェックを追加。FeatureBuilder 統合済みの DF では列が既に存在するためスキップ、旧呼び出し元からの個別渡しにも対応。

## Threat Flags

なし。計画の `<threat_model>` 通りの対応:
- T-52-04: 既存 BT テスト (79件) が回帰カバレッジを提供
- T-52-05: FeatureBuilder により PT が BT/Train と同一特徴量を取得

## Deferred Items

- `test_observed_true_on_all_groupby` が検出する `src/investment/feature_frame.py` の12件の未観測 groupby — 事前からの問題、Phase 52 範囲外
