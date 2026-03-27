# ROI改善 Phase 1: 馬固有特徴量追加 + 複勝確率モデル設計

**日付**: 2026-03-27
**ステータス**: Draft
**対象**: Stage1 AbilityModel の特徴量拡張 + PlaceAbilityModel 新規追加

---

## 背景

現在のホールドアウト結果: **ROI 63.8%** (2766ベット, 36.2%損失)。

根本原因分析により、以下の致命的問題を特定:

1. **Stage1 AbilityModel に馬固有特徴量が存在しない** — 7特徴量のうち5つはレース条件。同じレース内で馬を区別する情報がほぼゼロ。lambdarankは相対比較が前提のため、実質的に学習不能。
2. **`p_ability_place = p_ability_win * 3.0`** — 複勝的中率の粗い近似。順位確率は非線形（softmax構造）であり、線形スケーリングは理論的に破綻している。特に1番人気のplace確率を過大評価。
3. **RegimeDetector の10特徴量のうち3つがダミー値** — 常に0、常に0.1等。

本設計は問題1と2に対応する。問題3は後続イテレーションで扱う。

---

## 設計方針

- **最小インパクト狙い**: 最も効果が高い変更に集中する
- **既存パイプライン非破壊**: 新規モジュール追加方式。既存のFeatureEngine、Training Pipelineに呼び出しを追加するのみ
- **段階的改善**: 今回は3特徴量 + binary model。効果確認後に第2イテレーション（条件別騎手勝率、血統等）を実施

---

## 1. 新規特徴量: HorseHistoryFeatures

新規ファイル: `src/features/horse_history_features.py`

### 1-1. norm_finish_score (正規化着順スコア)

過去レースの着順を頭数で正規化し、高いほど好走を表すスコアに変換。

```python
score = 1 - (finish_pos - 1) / (field_size - 1)
```

- 直近3走の平均を特徴量として使用
- 1着/16頭 = 1.0、最下位/16頭 = 0.0
- 初出走馬: NaN（LightGBMが自動処理）
- 1-2走の場合: ある分だけの平均
- **リーク防止**: 当該レースは含めない（strictly past data）

### 1-2. jockey_surprise (騎手surprise)

騎手の「期待を上回る勝率」を定量化。強い馬に乗るバイアスを除去。

```python
expected_wins = sum(1 / odds for past races)
actual_wins = count(finish_pos == 1)
jockey_surprise = (actual_wins - expected_wins) / n_recent_races
```

- 直近N戦（N=100）に窓を絞る（非定常性対応）
- 最小サンプル数: 30レース未満なら NaN
- 正の値 = 期待以上、負の値 = 期待以下
- **リーク防止**: 当該レースは含めない

### 1-3. haron_time_zscore (ハロンタイムz-score)

末脚能力を距離・馬場条件で正規化して評価。

```python
group_key = (distance_bin, surface, baba_cd)
z = (haron_time_l3 - group_mean) / group_std
```

- 直近3走の平均を特徴量として使用
- グループ化: distance_bin x surface x baba_cd（馬場状態込み）
- NaN/0値のハロンタイムは除外
- グループ統計は全データから事前計算（リークなし: 過去レースの統計）
- **リーク防止**: 当該レースは含めない

### 1-4. レース内正規化特徴

各基本特徴量をレース内で相対化:

```python
for col in [norm_finish_score_avg, jockey_surprise, haron_time_zscore_avg]:
    race_mean = df.groupby("race_id")[col].transform("mean")
    race_std = df.groupby("race_id")[col].transform("std")
    race_std = max(race_std, 1e-6)  # std=0対策
    df[f"{col}_race_z"] = (df[col] - race_mean) / race_std
    df[f"{col}_race_rank"] = df.groupby("race_id")[col].rank(ascending=True)
    df[f"{col}_race_pct"] = df.groupby("race_id")[col].rank(pct=True)
```

これにより「このレースの中で相対的に強い馬」が定量化される。

---

## 2. PlaceAbilityModel (複勝binary model)

新規ファイル: `src/models/place_ability_model.py`

### 2-1. モデル仕様

| 項目 | 値 |
|------|-----|
| objective | binary |
| label | `finish_pos <= 3` (1 or 0) |
| 特徴量 | Stage1 FEATURE_COLS + 新規5特徴量 + レース内正規化 |
| 学習データ | Stage1と同じ期間 |
| クラス不均衡対応 | `scale_pos_weight = n_neg / n_pos`（自動計算）|

**クラス不均衡の重要性**:
- 複勝(top3)率は約 3/18 ≒ 17%
- `scale_pos_weight` 未設定だと全体確率が低く推定され、calibrationが崩壊

### 2-2. 確率校正 (Calibration)

LightGBMはランキングに強いが、確率出力は歪む。Isotonic regression で校正:

```python
from sklearn.calibration import CalibratedClassifierCV

# LightGBM model train後
calibrated = CalibratedClassifierCV(
    estimator=pretrained_lgb_model,
    method='isotonic',
    cv='prefit'
)
calibrated.fit(X_val, y_val)
p_ability_place = calibrated.predict_proba(X_new)[:, 1]
```

- Isotonic regressionはPlatt scalingより非線形に対応可能
- バリデーションデータで校正（学習データと分離必須）

### 2-3. 古い p_ability_place の置き換え

```python
# 旧 (stage1_ability_model.py L112)
df["p_ability_place"] = np.clip(df["p_ability_win"] * 3.0, 0, 1)

# 新 (training_pipeline.py で PlaceAbilityModel.train() 後)
df["p_ability_place"] = place_model.predict(df)
```

`add_ability_probs()` 内の `p_ability_place` 行を削除し、TrainingPipeline側で PlaceAbilityModel の出力を上書きする。

---

## 3. FeatureEngine への統合

ファイル: `src/features/feature_engine.py` の `build_all()`

```python
# build_all() の既存フローに追加
def build_all(self, race_df, entry_df, odds_df, odds_ts_df):
    ...
    # 既存: intra_race, odds_dynamics, market_bias, difficulty

    # NEW: 馬の過去成績特徴量
    hist_features = HorseHistoryFeatures()
    df = hist_features.compute(df, engine=self.engine)

    return df
```

HorseHistoryFeatures.compute() 内でDBクエリにより過去レースデータを取得するため、engine（DB接続）を渡す。

---

## 4. Stage1 AbilityModel の FEATURE_COLS 更新

ファイル: `src/models/stage1_ability_model.py`

```python
FEATURE_COLS = [
    # 既存 (7)
    "surface", "distance_bin", "track_condition_code",
    "grade_code", "field_size",
    "weight_diff_from_mean", "difficulty_score",
    # 新規: 馬の過去成績 (3)
    "norm_finish_score_avg", "jockey_surprise", "haron_time_zscore_avg",
    # 新規: レース内正規化 (3)
    "norm_finish_score_avg_race_z",
    "jockey_surprise_race_z",
    "haron_time_zscore_avg_race_z",
]
```

合計13特徴量（旧7 → 新13）。

---

## 5. Training Pipeline の変更

ファイル: `src/pipelines/training_pipeline.py`

### 5-1. 追加フロー（各サーフェス毎）

```
既存:
  MarketModel.train() → AbilityModel.train() → add_ability_probs()
  → WinTwoStage → EVCorrection → PlaceTwoStage → Wide

変更後:
  MarketModel.train() → AbilityModel.train() → add_ability_probs() [p_ability_winのみ]
  → PlaceAbilityModel.train() → predict()  [p_ability_placeをbinary modelで生成]
  → WinTwoStage → EVCorrection → PlaceTwoStage → Wide
```

PlaceAbilityModelはAbilityModelの直後に学習。AbilityModelと同じ特徴量 + レース内正規化特徴量を使用。

### 5-2. バリデーション分割

Isotonic calibration用に、学習データの末尾20%をバリデーションに使用:

```python
split_idx = int(len(train_df) * 0.8)
place_train = train_df.iloc[:split_idx]
place_calib = train_df.iloc[split_idx:]
```

---

## 6. 評価設計

### 6-1. 評価指標

| 指標 | 目的 | ツール |
|------|------|--------|
| ROI | 投資効率 | 既存バックテスト |
| Logloss | 確率予測精度 | `sklearn.metrics.log_loss` |
| Rank correlation | 順位予測の正しさ | `scipy.stats.spearmanr` |
| Calibration curve | 確率の校正確認 | `sklearn.calibration.calibration_curve` |

### 6-2. ベースライン比較

オッズ逆数（市場予測）をベースラインとして比較:

```python
p_baseline = (1 / odds) / sum(1 / odds_for_race)  # レース内で合計1
```

控除率補正: 生の `1/odds` は合計が1を超えるため、レース内正規化が必須。

### 6-3. 評価フロー

1. 特徴量追加後: `01b_feature_engineering.ipynb` で統計量・相関確認
2. モデル学習後: `12_time_series_cv.ipynb`（要作成）で時系列CV検証
3. CV各foldで: ROI, Logloss, Rank correlation, Calibration curve を記録
4. 全fold安定改善なら採用、ホールドアウトは温存

---

## 7. テスト設計

### 7-1. test_horse_history_features.py

- 正規化着順: 1着/16頭=1.0, 最下位=0.0 の境界値テスト
- 騎手surprise: 期待値計算の正確性テスト
- ハロンz-score: グループ統計の正確性テスト
- レース内z-score: **std=0 の場合にNaNにならない**テスト
- リーク防止: 当該レースが特徴量計算に含まれないテスト
- 欠損値: 初出走馬、最小サンプル未満騎手のNaNテスト

### 7-2. test_place_ability_model.py

- クラス不均衡: `scale_pos_weight` が正しく計算されるテスト
- 確率範囲: 出力が [0, 1] に収まるテスト
- Calibration: Isotonic校正後の確率が改善されるテスト
- FEATURE_COLS: 新規列が全て存在するテスト

---

## 8. データリーク対策

以下の境界でリークが起きやすい。テストで必ず検証:

- **同日複数レース**: 同一馬が同日に複数レースに出走しない（JRA仕様）ため問題なし
- **過去データの取得タイミング**: レース日付より前のデータのみ使用
- **特徴量生成のjoinタイミング**: HorseHistoryFeatures.compute() は build_all() 内で実行、当該レース除外

---

## 9. 変更ファイル一覧

| ファイル | 変更種別 | 概要 |
|---------|---------|------|
| `src/features/horse_history_features.py` | **新規** | 3基本特徴量 + レース内正規化 |
| `src/features/feature_engine.py` | 修正 | `build_all()`に呼び出し追加 |
| `src/models/stage1_ability_model.py` | 修正 | FEATURE_COLSに6列追加、`p_ability_place`行を削除 |
| `src/models/place_ability_model.py` | **新規** | binary model + calibration |
| `src/pipelines/training_pipeline.py` | 修正 | PlaceAbilityModel学習・予測追加 |
| `tests/test_horse_history_features.py` | **新規** | 特徴量計算のテスト |
| `tests/test_place_ability_model.py` | **新規** | モデル・calibrationのテスト |

---

## 10. スコープ外（第2イテレーション以降）

- RegimeDetector ダミー特徴量の修正
- 条件別騎手勝率（距離・馬場で分割）
- 血統情報
- 負担重量の絶対値
- Kelly基準による動的ベット額
- Win/Wide ベットのバックテスト有効化
- `12_time_series_cv.ipynb` の作成
