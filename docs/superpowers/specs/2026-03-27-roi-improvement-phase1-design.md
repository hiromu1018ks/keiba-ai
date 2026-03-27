# ROI改善 Phase 1: 馬固有特徴量追加 + 複勝確率モデル設計

**日付**: 2026-03-27
**ステータス**: Draft (Review #2 修正済み)
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
- **段階的改善**: 今回は3基本特徴量 + 3レース内z-score + binary model。効果確認後に第2イテレーション（条件別騎手勝率、血統等）を実施

---

## 1. 新規特徴量: HorseHistoryFeatures

新規ファイル: `src/features/horse_history_features.py`

### 1-0. データソースと計算方式

**データ取得元**: EveryDB2外部テーブル `n_uma_race` を直接クエリ（`raw.entries` はETLでロード済みの期間のみ）。

**計算場所**: FeatureEngine.build_all() ではなく TrainingPipeline 内で計算。

理由: FeatureEngine.build_all() は DataFrame-in/DataFrame-out の純粋関数であり、DB接続を持たない。過去レースデータの検索にはDB接続が必要なため、TrainingPipeline（DB接続を保持）内で HorseHistoryFeatures を呼び出し、結果のDataFrameを FeatureEngine.build_all() の出力に結合する。

```python
# TrainingPipelineV5._train_submodel() 内
from features.horse_history_features import HorseHistoryFeatures

hist = HorseHistoryFeatures(engine=self.engine)
hist_df = hist.compute(race_df, entry_df)  # DB から過去レースを検索

# build_all の出力に結合
df = feature_engine.build_all(race_df, entry_df, odds_df, odds_ts_df)
df = df.merge(hist_df, on=["race_id", "umaban"], how="left")
```

**クエリ仕様**:
- テーブル: EveryDB2 の `n_uma_race` JOIN `n_race`（field_size 取得用）
- ルックバック: 当該レース日付より前の全データ（制限なし、jockey_surprise は直近100戦に窓）
- フィルタ: 対象レースの全馬の `ketto_num` と `kisyu_code` で過去レースを検索

### 1-1. norm_finish_score_avg (正規化着順スコア平均)

過去レースの着順を頭数で正規化し、高いほど好走を表すスコアに変換。

```python
score = 1 - (finish_pos - 1) / (field_size - 1)
```

- 直近3走の平均を特徴量として使用
- 1着/16頭 = 1.0、最下位/16頭 = 0.0
- 初出走馬: NaN（LightGBMが自動処理）
- 1-2走の場合: ある分だけの平均
- **リーク防止**: 当該レース日付より前のレースのみ使用

### 1-2. jockey_surprise (騎手surprise)

騎手の「期待を上回る勝率」を定量化。強い馬に乗るバイアスを除去。

```python
# 騎手の直近N戦（N=100）の全過去レースを取得
expected_wins = sum(1 / odds for each past race)
actual_wins = count(finish_pos == 1)
jockey_surprise = (actual_wins - expected_wins) / n_recent_races
```

- 直近100戦に窓を絞る（非定常性対応）
- 最小サンプル数: 30レース未満なら NaN
- 正の値 = 期待以上、負の値 = 期待以下
- **リーク防止**: 当該レース日付より前のレースのみ使用

### 1-3. haron_time_zscore_avg (ハロンタイムz-score平均)

末脚能力を距離・馬場条件で正規化して評価。

```python
group_key = (distance_bin, surface, baba_cd)
z = (haron_time_l3 - group_mean) / group_std
```

- 直近3走の平均を特徴量として使用
- グループ化: distance_bin x surface x baba_cd（馬場状態込み）
- NaN/0値のハロンタイムは除外
- グループ統計はクエリ結果全体から計算（過去レースの統計）
- **リーク防止**: 当該レース日付より前のレースのみ使用

### 1-4. レース内z-score（FEATURE_COLSに含む3列のみ）

各基本特徴量をレース内でz-score化。Phase 1 では `_race_z` のみ FEATURE_COLS に含める。

```python
for col in [norm_finish_score_avg, jockey_surprise, haron_time_zscore_avg]:
    race_mean = df.groupby("race_id")[col].transform("mean")
    race_std = df.groupby("race_id")[col].transform("std")
    race_std = race_std.clip(lower=1e-6)  # std=0対策: 下限クリップ
    df[f"{col}_race_z"] = (df[col] - race_mean) / race_std
```

**Phase 1 で FEATURES_COLS に含む列**（計3列）:
- `norm_finish_score_avg_race_z`
- `jockey_surprise_race_z`
- `haron_time_zscore_avg_race_z`

**Phase 2 以降の候補**（今回は追加しない）:
- `_race_rank`, `_race_pct` — rank方向の定義（昇順/降順）を特徴量毎に適切に設定する必要があるため、Phase 2 で導入

### 1-5. 新規特徴量サマリ

**FEATURE_COLS に追加する全6列**:
1. `norm_finish_score_avg` — 正規化着順スコア平均
2. `jockey_surprise` — 騎手surprise
3. `haron_time_zscore_avg` — ハロンタイムz-score平均
4. `norm_finish_score_avg_race_z` — 着順スコアのレース内z-score
5. `jockey_surprise_race_z` — 騎手surpriseのレース内z-score
6. `haron_time_zscore_avg_race_z` — ハロンタイムのレース内z-score

---

## 2. PlaceAbilityModel (複勝binary model)

新規ファイル: `src/models/place_ability_model.py`

### 2-1. モデル仕様

| 項目 | 値 |
|------|-----|
| API | `lgb.LGBMClassifier`（sklearn互換。`CalibratedClassifierCV` で使用可能） |
| objective | binary |
| label | `finish_pos <= 3` (1 or 0) |
| 特徴量 | Stage1 FEATURE_COLS と同一（13列 = 既存7 + 新規6） |
| 学習データ | Stage1と同じ期間 |
| クラス不均衡対応 | `scale_pos_weight = n_neg / n_pos`（自動計算） |

**`LGBMClassifier` を使用する理由**:
- `lgb.train()` は sklearn の `fit()/predict_proba()` インターフェースを実装しない
- `CalibratedClassifierCV(estimator=..., cv='prefit')` は sklearn 互換 API を要求する
- `LGBMClassifier` は内部で `lgb.train()` を呼び出すが、sklearn 互換ラッパーとして動作する

**クラス不均衡の重要性**:
- 複勝(top3)率は約 3/18 ≒ 17%
- `scale_pos_weight` 未設定だと全体確率が低く推定され、calibrationが崩壊

### 2-2. 確率校正 (Calibration)

LightGBMはランキングに強いが、確率出力は歪む。Isotonic regression で校正:

```python
from sklearn.calibration import CalibratedClassifierCV

# LGBMClassifier で学習（sklearn API）
model = lgb.LGBMClassifier(
    objective="binary",
    scale_pos_weight=n_neg / n_pos,
    num_leaves=31,
    learning_rate=0.03,
    n_estimators=500,
)
model.fit(X_train, y_train)

# Isotonic calibration（別データで校正）
calibrated = CalibratedClassifierCV(
    estimator=model,
    method="isotonic",
    cv="prefit",
)
calibrated.fit(X_calib, y_calib)
p_ability_place = calibrated.predict_proba(X_new)[:, 1]
```

- Isotonic regressionはPlatt scalingより非線形に対応可能
- 校正用データは学習データと時系列的に分離（後述 5-2）

### 2-3. 古い p_ability_place の置き換え

```python
# 旧 (stage1_ability_model.py)
df["p_ability_place"] = np.clip(df["p_ability_win"] * 3.0, 0, 1)

# 新: この行を削除。TrainingPipeline で PlaceAbilityModel の出力を代入
```

`add_ability_probs()` 内の `p_ability_place` 行を削除し、TrainingPipeline側で PlaceAbilityModel の出力を設定する。

### 2-4. 推論時の統合

PlaceAbilityModel は `SubmodelSet` に格納し、バックテスト・推論時にも使用:

```python
# src/domain/models.py の SubmodelSet に追加
@dataclass(frozen=True)
class SubmodelSet:
    market: MarketModel
    stage1: AbilityModel
    place_ability: PlaceAbilityModel  # NEW
    win: WinTwoStageModel
    ev_corrector: EVCorrectionModel
    place: PlaceTwoStageModel
    wide: WideTwoStageModel
    confidence: RobustConfidenceEstimator
```

バックテスト・推論時のフロー:
1. FeatureEngine.build_all() → HorseHistoryFeatures を TrainingPipeline と同様に呼び出し
2. AbilityModel.add_ability_probs() → p_ability_win のみ設定
3. **PlaceAbilityModel.predict()** → p_ability_place を設定
4. WinTwoStage → EVCorrection → PlaceTwoStage → Wide（既存フロー）

推論時にDB接続が必要なため、BacktestEngine と BettingOrchestrator は engine を HorseHistoryFeatures に渡す。

---

## 3. FeatureEngine と TrainingPipeline の統合

**FeatureEngine.build_all() は変更しない**。HorseHistoryFeatures は TrainingPipeline 内で別途計算し、build_all() の出力に結合する。

### TrainingPipelineV5._train_submodel() の変更

```python
def _train_submodel(self, df, surface_key):
    # 既存: build_all → MarketModel → AbilityModel → ...

    # NEW: 馬の過去成績特徴量を計算（DB接続使用）
    hist = HorseHistoryFeatures(engine=self.engine)
    hist_df = hist.compute(race_df, entry_df)
    df = df.merge(hist_df, on=["race_id", "umaban"], how="left")

    # レース内z-score計算
    df = hist.add_race_z_scores(df)

    # 既存: MarketModel.train() → AbilityModel.train() → add_ability_probs()
    # ...

    # NEW: PlaceAbilityModel（AbilityModel直後）
    place_ability = PlaceAbilityModel(surface_key=surface_key)
    place_ability.train(train_df, calib_df)  # 時系列分割で学習+校正
    df = place_ability.predict(df)  # p_ability_place を設定

    # 既存: WinTwoStage → EVCorrection → PlaceTwoStage → Wide
    # ...
```

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
    # 新規: レース内z-score (3)
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
  [HorseHistoryFeatures.compute() → merge]  ← NEW
  MarketModel.train() → AbilityModel.train() → add_ability_probs() [p_ability_winのみ]
  [PlaceAbilityModel.train() → predict()]  ← NEW
  WinTwoStage → EVCorrection → PlaceTwoStage → Wide
```

### 5-2. 時系列バリデーション分割

Isotonic calibration用に、時系列で明示的に分割（行インデックスではなく日付基準）:

```python
# 日付でソート済み前提。最後の20%の期間を校正用に使用
dates = sorted(df["race_date"].unique())
split_date = dates[int(len(dates) * 0.8)]
place_train = df[df["race_date"] < split_date]
place_calib = df[df["race_date"] >= split_date]
```

これにより、学習データと校正データが時系列的に分離され、リークを防止。

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

ベースラインに対しても Logloss と Calibration curve を計算し、モデルと比較する。

### 6-3. 評価フロー

Phase 1 の評価は既存のバックテストエンジンとノートブックで行う:

1. 特徴量追加後: `01b_feature_engineering.ipynb` で統計量・相関確認
2. モデル学習後: 既存バックテストエンジンで ROI 検証
3. Logloss, Rank correlation, Calibration curve は `01b_feature_engineering.ipynb` または新規分析セルで計算
4. ホールドアウトは温存（時系列CVノートブックは Phase 2 で作成）

---

## 7. テスト設計

### 7-1. test_horse_history_features.py

- 正規化着順: 1着/16頭=1.0, 最下位=0.0 の境界値テスト
- 騎手surprise: 期待値計算の正確性テスト
- ハロンz-score: グループ統計の正確性テスト
- レース内z-score: **std=0 の場合にNaNにならない**テスト
- リーク防止: 当該レース日付より後のデータが特徴量計算に含まれないテスト
- 欠損値: 初出走馬、最小サンプル未満騎手のNaNテスト

### 7-2. test_place_ability_model.py

- クラス不均衡: `scale_pos_weight` が正しく計算されるテスト
- 確率範囲: 出力が [0, 1] に収まるテスト
- Calibration: Isotonic校正後の確率が改善されるテスト
- FEATURE_COLS: 新規6列が全て存在するテスト
- LGBMClassifier: sklearn互換APIで動作するテスト

### 7-3. test_training_pipeline.py (既存ファイルに追加)

- **統合テスト**: PlaceAbilityModel が AbilityModel 直後に学習されるテスト
- **p_ability_place 置き換え**: PlaceAbilityModel の出力が `df["p_ability_place"]` に正しく設定されるテスト
- **HorseHistoryFeatures 結合**: merge 後に全特徴量列が存在するテスト
- **SubmodelSet**: `place_ability` フィールドが正しく格納されるテスト

---

## 8. データリーク対策

以下の境界でリークが起きやすい。テストで必ず検証:

- **同日複数レース**: 同一馬が同日に複数レースに出走しない（JRA仕様）ため問題なし
- **過去データの取得タイミング**: レース日付より前（`< race_date`）のデータのみ使用。同日のレースも除外
- **特徴量生成のjoinタイミング**: HorseHistoryFeatures.compute() は TrainingPipeline 内で実行、当該レース除外
- **校正データの分離**: PlaceAbilityModel の校正データは学習データより未来の期間を使用（時系列分割）

---

## 9. 変更ファイル一覧

| ファイル | 変更種別 | 概要 |
|---------|---------|------|
| `src/features/horse_history_features.py` | **新規** | 3基本特徴量 + 3レース内z-score + DBクエリ |
| `src/features/feature_engine.py` | 変更なし | HorseHistoryFeatures は TrainingPipeline 側で呼び出し |
| `src/models/stage1_ability_model.py` | 修正 | FEATURE_COLSに6列追加、`p_ability_place`行を削除 |
| `src/models/place_ability_model.py` | **新規** | LGBMClassifier binary + Isotonic calibration |
| `src/domain/models.py` | 修正 | SubmodelSet に `place_ability` フィールド追加 |
| `src/pipelines/training_pipeline.py` | 修正 | HorseHistoryFeatures 呼び出し + PlaceAbilityModel 学習・予測追加 |
| `src/backtest/engine.py` | 修正 | HorseHistoryFeatures 呼び出し + PlaceAbilityModel 推論追加 |
| `tests/test_horse_history_features.py` | **新規** | 特徴量計算のテスト |
| `tests/test_place_ability_model.py` | **新規** | モデル・calibrationのテスト |
| `tests/test_training_pipeline.py` | 修正 | 統合テスト追加 |

---

## 10. スコープ外（第2イテレーション以降）

- RegimeDetector ダミー特徴量の修正
- 条件別騎手勝率（距離・馬場で分割）
- 血統情報
- 負担重量の絶対値
- Kelly基準による動的ベット額
- Win/Wide ベットのバックテスト有効化
- `12_time_series_cv.ipynb` の作成
- レース内 `_race_rank`, `_race_pct` 特徴量の追加
