# Phase 2: 種牡馬産駒特徴量 — 実装結果 (2026-04-13)

## 概要

PIT安全な種牡馬産駒累積統計に基づく5つの新特徴量を実装し、
学習・バックテストパイプラインに統合完了。

## 実装タスク

### Task 2.1: 事前計算スクリプト ✅
- **ファイル**: `scripts/precompute_sire_stats.py` (新規)
- **出力**: `data/raw/sire_career_stats.parquet`
- **データ量**: 161,408 rows / 1,073 種牡馬 / 処理時間 ~2.9s
- **PIT保証**: `shift(1).fillna(0).cumsum()` パターン (horse_career_stats.py と統一)
- **フィルタ**: JRAのみ (jyocd 1-10), NaN sire_id 除外

### Task 2.2: SireFeatures モジュール ✅
- **ファイル**: `src/features/sire_features.py` (新規)
- **クラス**: `SireFeatures` — compute() (単行) + compute_batch() (ベクトル化一括)
- **特徴量** (5列):
  - `sire_wr`: 種牡馬全体勝率 (Beta平滑化)
  - `sire_surface_wr`: サーフェス別勝率 (turf/dirt)
  - `sire_distance_wr`: 距離別勝率 (short ≤1600m / long >1600m)
  - `sire_prize_avg`: 平均賞金 (log1p変換)
  - `bms_wr`: 母父 (BMS) 産駒勝率
- **Beta平滑化**: `(alpha + wins) / (alpha + beta + starts)` where α=1, β=10
- **lookup方式**: groupby + searchsorted (merge_asof のソート要件問題を回避)
- **リーダー**: `src/db/readers.py` に `load_sire_stats()` 追加

### Task 2.3: モデル・パイプライン統合 ✅
- **Stage1 AbilityModel**: FEATURE_COLS 37 → 42 (+5 sire列)
- **PlaceAbilityModel**: FEATURE_COLS 38 → 43 (+5 sire列)
- **training_pipeline.py**: `_train_submodel()` に sire_features ブロック追加
  - horses.parquet から kettonum→sire_id/bms_id マッピング
  - `compute_batch()` ベクトル化計算
  - 出力を必要列のみフィルタ (sire_place_rate 除外)
- **backtest/engine.py**: 推論パスに同一の sire_features 計算を追加
  - 学習/推論の分布不一致 (train: 実値, test: NaN) を防止

### Task 2.4: バックテスト検証 ✅
- **設定**: 学習 2021-2024 (4年) / テスト 2025 / ensemble / flat
- **結果**:
  | 指標 | 値 |
  |------|-----|
  | ROI | **106.0%** |
  | 利益 | +¥2,470 |
  | 最大DD | 3.2% |
  | レース数 | 415 |
  | 投資額 | ¥41,500 |
  | 払戻額 | ¥43,970 |
  | 学習時間 | 461秒 |
  | テスト時間 | 424秒 |

## 技術的成果

### 解決した課題
1. **merge_asof のソート要件問題** (4回の失敗):
   - 原因: `by` 列 (sire_id) の NaN 値が pandas 内部ソート検証を破壊
   - 解決: groupby + searchsorted ループ方式に完全置換
   - コミット: `ac3acbf`

2. **型不一致エラー**:
   - 原因: sire_id が int64 (stats側) vs object (df側)
   - 解決: 両側を str にキャストして統一

3. **NA→int 変換エラー**:
   - 原因: trackcd の NaN が `.astype(int)` で失敗
   - 解決: `.fillna(False)` を挿入

### テストカバレッジ
- `test_sire_features.py`: 16 tests (BetaSmooth, SireWr, PlaceRate, PrizeAvg, MissingSire, PitSafety)
- `test_precompute_sire_stats.py`: 6 tests (PIT safety, multi-sire, turf/dirt, prize, columns, NaN exclusion)
- 既存テスト更新: stage1_ability_model, place_ability_model, training_pipeline (fixture 更新)
- **全 842 tests passing**

## 考察

### ROI 変動について
- **Phase 2 前 (JRA filter + odds alignment)**: ROI 216.6%
- **Phase 2 後 (sire features 追加)**: ROI 106.0%

ROI は低下したが黒字維持。考えられる要因:
1. 新特徴量がモデルの複雑性を増やし、過学習リスク上昇
2. 5特徴量の情報量が相対的に小さい (種牡馬統計は馬個人の能力よりマクロ)
3. ハイパラメータ再調整が必要 (n_estimators, learning_rate 等)
4. ベット数が 9,074 → 415 に激減 — スクリーニング条件との相互作用可能性

### 今後の改善方向
- Feature importance 確認: sire 特徴量が実際に寄与しているか
- アンサンブル重み再調整
- 複数年度バックテストで安定性確認
- 必要に応じて特徴量選択 (低 importance 列削除)

## コミット履歴

| コミット | 内容 |
|----------|------|
| (precompute) | scripts/precompute_sire_stats.py 新規作成 |
| (sire_features) | src/features/sire_features.py 新規作成 |
| (readers) | load_sire_stats() 追加 |
| (models) | Stage1/PlaceAbilityModel に +5 sire 列 |
| (pipeline) | training_pipeline.py に sire_features wiring |
| (engine) | backtest/engine.py に推論パス sire_features |
| `ce55d0b` | fix: NA→int エラー修正 (fillna False) |
| `053596e` | fix: merge_asof 型不一致修正 (str cast) |
| `b4420ac` | fix: merge_asof ソート順修正 |
| `ac3acbf` | fix: compute_batch → groupby+searchsorted 書き換え |
