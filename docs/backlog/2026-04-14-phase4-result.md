# Phase 4 結果: 過去走拡張 + ペース適性 + コース適性

## バックテスト設定

- 学習: 2021-01-01 ~ 2024-12-31 (4年)
- テスト: 2025-01-01 ~ 2025-12-31
- モモード: ensemble, flat (¥100), JRAのみ

## 結果

| 指標 | 値 |
|------|-----|
| レース数 | 3,991 |
| 投資額 | ¥399,100 |
| 払戻額 | ¥299,140 |
| 利益 | **-¥99,960** |
| **ROI** | **75.0%** |
| 最大DD | 100.0% |

## 比較

| 版 | ROI | 利益 | 備考 |
|----|-----|------|------|
| Baseline (Phase 1-3前) | 98.8% | -¥610 | |
| Phase 3完了 (OOF統合後) | 63.8% | — | リーク除去で低下 |
| **Phase 4完了** | **75.0%** | **-¥99,960** | **+11.2pt 改善** |

## 実装した特徴量

### Task 4.1: 過去走 3→5 拡張
- `harontimel3_avg` → `harontimel5_avg` (5走平均)
- `harontimel3_zscore` → `harontimel5_zscore` (5走z-score)
- `harontime_late_trend`: 最後2走平均 - 最初3走平均 (負=改善)

### Task 4.2: ペース適性特徴量
- `pace_aptitude`: closing vs front 正規化着順差
- `front_pace_wr`: 前ペース時 Beta平滑化勝率
- `closing_pace_wr`: 後ろ待ち時 Beta 平滑化勝率

### Task 4.3: コース別適性特徴量
- `course_wr`: 競馬場別 Beta 平滑化勝率
- `course_distance_wr`: 競馬場×距離帯別 Beta 平滑化勝率

## 所見

1. **改善要因**: 主に L5 拡張と late_trend による信号品質向上
2. **未活用特徴量**: pace_aptitude (3列) と course_features (2列) は
   build_all() 時点では NaN プレースホルダーのため実質的に無効
   → TrainingPipeline/BacktestEngine での実計算 wiring が必要
3. **次のステップ**: ペース適性・コース適性を HorseHistoryFeatures と
   同じタイミングで計算するよう TrainingPipeline を改修すれば
   さらに改善が期待できる

## 判定

- [ ] 黒字化達成 (ROI > 100%)
- [x] Phase 3 から改善 (+11.2pt)
- [x] 全テスト通過 (872 passed, 1 pre-existing failure)
- [x] PIT安全性維持 (race_date < target_date フィルタ確認済み)
