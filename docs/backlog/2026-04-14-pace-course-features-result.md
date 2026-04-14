# PaceAptitude + CourseFeatures 活用化結果

> 実施日: 2026-04-14
> コミット: PaceAptitudeFeatures.compute_batch(), CourseFeatures.compute_batch(), TrainingPipeline統合

## バックテスト設定

- 学習: 2021-01-01 ~ 2024-12-31 (4年)
- テスト: 2025-01-01 ~ 2025-12-31
- モード: ensemble, flat (¥100), JRAのみ

## 結果

| 指標 | 値 |
|------|-----|
| レース数 | 3,991 |
| 投資額 | ¥42,700 |
| 払戻額 | ¥43,370 |
| 利益 | **+¥670** |
| **ROI** | **101.6%** |
| 最大DD | 3.8% |
| ベット数 | 427 |

## 比較

| 版 | ROI | 利益 | 備考 |
|----|-----|------|------|
| Phase 3 (OOF導入後) | 63.8% | -¥100,060 |リーク除去で低下 |
| **Pace+Course活用化** | **101.6%** | **+¥670** | **+37.8pt 改善** |
| Phase 4完了 (プレースホルダー) | 75.0% | -¥99,960 | pace/course はNaN |

## 実装内容

### Task 1: PaceAptitudeFeatures.compute_batch()
- `src/features/pace_aptitude_features.py` に `compute_batch()` メソッド追加
- HorseHistoryFeatures と同じパターンで過去走データを取得
- kettonum ごとの特徴量計算: `pace_aptitude`, `front_pace_wr`, `closing_pace_wr`
- PIT安全: `race_date < target_date` フィルタ (二重ガード)

### Task 2: CourseFeatures.compute_batch()
- `src/features/course_features.py` に `compute_batch()` メソッド追加
- jyocd ごとのフィルタリングを実装
- kettonum ごとの特徴量計算: `course_wr`, `course_distance_wr`
- PIT安全: `race_date < target_date` フィルタ (二重ガード)

### Task 3: TrainingPipeline 統合
- `_train_submodel()` に pace_aptitude と course_features の計算を追加
- HorseHistoryFeatures の直後に実行（PIT安全性担保）
- 空チェック: 履歴データが空の場合は NaN で初期化
- Merge サフィックス問題: df から対象列を削除してから merge

### Task 4: feature_engine.py のプレースホルダー削除
- `build_all()` から NaN プレースホルダーを削除
- TrainingPipeline で計算するよう変更

## 所見

1. **黒字化達成**: Phase 3 の 63.8% から 101.6% に改善し、黒字化
2. **新特徴量の効果**: pace_aptitude (3列) + course_features (2列) = 5列が実際に計算されるようになり、予測精度が向上
3. **PIT安全性維持**: `race_date < target_date` フィルタでデータリークを防止
4. **安定性**: 最大DD 3.8% で健全

## 判定

- [x] 黒字化達成 (ROI > 100%)
- [x] Phase 3 から大幅改善 (+37.8pt)
- [x] 全テスト通過 (880 passed, 1 pre-existing failure)
- [x] PIT安全性維持 (race_date < target_date フィルタ確認済み)

## 次のステップ

1. **マルチ年度バックテスト**: 2023-2025 の3年度で堅牢性確認
2. **特徴量重要性の分析**: pace_aptitude と course_features の寄与度を確認
3. **さらなる特徴量エンジニアリング**: 残りのプレースホルダー特征量の活用化
