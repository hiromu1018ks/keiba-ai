# ROI改善 包括的設計: Phase 1-5 カスケード方式

**日付**: 2026-03-28
**ステータス**: Draft
**対象**: 全5Phase によるROI包括改善（63.8% → 101%+）
**アプローチ**: 一括設計・段階実装（カスケード）

---

## 背景

現在のホールドアウト結果: **ROI 63.8%** (2766ベット, 36.2%損失)。

根本原因: Stage1 AbilityModel の特徴量不足（7列中5列がレース条件）。lambdarankがレース内の馬を区別できない。

本設計は5つのカスケードPhaseで改善する。各Phase間でバックテストを実行しROI寄与を測定。効果がないPhaseは見直す。

---

## 設計方針

- **予測精度ファースト**: モデル精度を最優先（Phase 1-3）、次にベッティング最適化（Phase 4）、最後に検証基盤（Phase 5）
- **既存パイプライン非破壊**: 新規モジュール追加方式。既存のFeatureEngine、TrainingPipelineに呼び出しを追加するのみ
- **段階的改善**: 各Phaseで効果確認後に次へ進む
- **崩れないモデル**: スムージング、正規化、fallback等で統計的安定性を確保

---

## Phase構成

```
Phase 1: 馬固有特徴量 + PlaceAbilityModel
    │ ROI最大インパクト。lambdarankが馬を区別可能に。
    ▼ バックテスト検証
Phase 2: 追加特徴量拡張
    │ 条件別騎手勝率、race_pct、負担重量絶対値
    ▼ バックテスト検証
Phase 3: RegimeDetector 実データ化
    │ flb_slope、odds_volatility、人気帯別回収率
    ▼ バックテスト検証
Phase 4: ベッティング戦略最適化
    │ ワイド活性化、Fractional Kelly、EV閾値チューニング
    ▼ バックテスト検証
Phase 5: 検証基盤強化
    Walk-forward CV、パラメータフリーズ、自動ホールドアウト検証
```

---

# Phase 1: 馬固有特徴量 + PlaceAbilityModel

新規ファイル: `src/features/horse_history_features.py`, `src/models/place_ability_model.py`

## 1-1. HorseHistoryFeatures

### データソースと計算方式

**データ取得元**: EveryDB2外部テーブル `n_uma_race` JOIN `n_race`（field_size取得用）。

**計算場所**: TrainingPipeline 内（FeatureEngine.build_all() は DB接続を持たない純粋関数のため）。

```python
# TrainingPipelineV5._train_submodel() 内
from features.horse_history_features import HorseHistoryFeatures

hist = HorseHistoryFeatures(engine=self.engine)
hist_df = hist.compute(race_df, entry_df)
df = df.merge(hist_df, on=["race_id", "umaban"], how="left")
df = hist.add_race_transforms(df)  # z-score + pct
```

**クエリ仕様**:
- テーブル: EveryDB2 の `n_uma_race` JOIN `n_race`
- ルックバック: 当該レース日付より前の全データ
- フィルタ: 対象レースの全馬の `ketto_num` と `kisyu_code` で過去レースを検索

### 1-1a. norm_finish_score_avg（logit変換・分散安定化）

過去レースの着順を頭数で正規化し、logit変換で分散安定化。

```python
# 頭数フィルタ: 8頭未満のレースは除外
if field_size < 8:
    return np.nan

score = 1 - (finish_pos - 1) / (field_size - 1)  # [0, 1]
score = np.clip(score, 0.05, 0.95)  # logit境界対策（極端値抑制）
logit_score = np.log(score / (1 - score))

# 直近3走のlogit平均を特徴量に
norm_finish_logit_avg = logit_scores[-3:].mean()
```

- 1着/16頭 = logit(0.95) ≈ 2.94、最下位/16頭 = logit(0.05) ≈ -2.94
- 初出走馬: NaN（LightGBM自動処理）
- **リーク防止**: 当該レース日付より前のレースのみ使用

### 1-1b. jockey_surprise（Beta事前分布スムージング・控除率補正）

騎手の「期待を上回る勝率」をBeta事前分布でスムージング。

```python
PAYOUT_RATE = 0.80  # JRA控除率20%

# expected_wins: 控除率補正済み確率
expected_wins = sum(PAYOUT_RATE / odds for each past race)
actual_wins = count(finish_pos == 1)

# Beta事前分布でスムージング (Empirical Bayes)
# 事前: Beta(alpha=1, beta=20) — 「20回に1回勝つ」という弱い事前信念
alpha_prior, beta_prior = 1.0, 20.0
alpha_post = alpha_prior + actual_wins
beta_post = beta_prior + n_recent_races - actual_wins

# 事後期待値と事前期待値の差 = surprise
smoothed_wr = alpha_post / (alpha_post + beta_post)
baseline_wr = alpha_prior / (alpha_prior + beta_prior)
jockey_surprise = smoothed_wr - baseline_wr
```

- 直近100戦に窓を絞る
- 最小サンプル数: 30レース未満なら NaN
- 正の値 = 期待以上、負の値 = 期待以下
- **控除率補正**: 生の `1/odds` ではなく `PAYOUT_RATE / odds` を使用
- **Beta事前分布**: n=30未満でも極端な値にならない
- **リーク防止**: 当該レース日付より前のレースのみ使用

### 1-1c. haron_time_zscore_avg（階層fallback）

末脚能力を距離・馬場条件で正規化。サンプル不足時に階層fallback。

```python
# fallback階層（細粒度 → 粗粒度）
FALLBACK_LEVELS = [
    # (key columns, min_samples)
    (["distance_bin", "surface", "baba_cd"], 50),  # Level 1
    (["distance_bin", "surface"], 30),              # Level 2
    (["distance_bin"], 20),                         # Level 3
    ([], 0),                                        # Level 4: 全体
]

def _get_group_stats(row, global_stats):
    for key_cols, min_n in FALLBACK_LEVELS:
        key = tuple(row[c] for c in key_cols) if key_cols else ("all",)
        group = global_stats.get(key)
        if group and group["n"] >= min_n:
            return group["mean"], group["std"]
    return global_stats[("all",)]["mean"], global_stats[("all",)]["std"]

z = (haron_time_l3 - group_mean) / group_std
```

- 直近3走の平均を特徴量
- NaN/0値のハロンタイムは除外
- **リーク防止**: 当該レース日付より前のレースのみ使用

### 1-1d. レース内変換（z-score + pct）

各基本特徴量をレース内でz-score化とパーセンタイル化。

```python
for col in [norm_finish_logit_avg, jockey_surprise, haron_time_zscore_avg]:
    # z-score
    race_mean = df.groupby("race_id")[col].transform("mean")
    race_std = df.groupby("race_id")[col].transform("std")
    race_std = race_std.clip(lower=1e-6)
    df[f"{col}_race_z"] = (df[col] - race_mean) / race_std

    # percentile
    df[f"{col}_race_pct"] = df.groupby("race_id")[col].rank(pct=True)
```

**Phase 1 FEATURE_COLS追加（計9列）**:
- `norm_finish_logit_avg` — logit着順スコア平均
- `jockey_surprise` — 騎手surprise
- `haron_time_zscore_avg` — ハロンタイムz-score平均
- `norm_finish_logit_avg_race_z` — 着順スコアのレース内z-score
- `jockey_surprise_race_z` — 騎手surpriseのレース内z-score
- `haron_time_zscore_avg_race_z` — ハロンタイムのレース内z-score
- `norm_finish_logit_avg_race_pct` — 着順スコアのレース内パーセンタイル
- `jockey_surprise_race_pct` — 騎手surpriseのレース内パーセンタイル
- `haron_time_zscore_avg_race_pct` — ハロンタイムのレース内パーセンタイル

**field_size特徴量（必須追加）**:
- `field_size` は既存FEATURE_COLSに含まれるため追加対応不要

## 1-2. PlaceAbilityModel

新規ファイル: `src/models/place_ability_model.py`

### モデル仕様

| 項目 | 値 |
|------|-----|
| API | `lgb.LGBMClassifier`（sklearn互換） |
| objective | binary |
| label | `finish_pos <= 3` (1 or 0) |
| 特徴量 | Stage1 FEATURE_COLS と同一 |
| 学習データ | Stage1と同じ期間 |
| クラス不均衡対応 | `scale_pos_weight = n_neg / n_pos` |
| 校正 | Isotonic regression (`CalibratedClassifierCV`) |

### LightGBM正則化パラメータ（過学習防止）

```python
model = lgb.LGBMClassifier(
    objective="binary",
    scale_pos_weight=n_neg / n_pos,
    num_leaves=31,
    max_depth=-1,
    min_data_in_leaf=100,
    feature_fraction=0.7,
    lambda_l2=1.0,
    learning_rate=0.03,
    n_estimators=500,
)
```

### 確率校正 + 温度スケーリング + 整合性制約

```python
from sklearn.calibration import CalibratedClassifierCV

# 1. Isotonic calibration（時系列分割で校正）
calibrated = CalibratedClassifierCV(
    estimator=model, method="isotonic", cv="prefit",
)
calibrated.fit(X_calib, y_calib)
raw_p = calibrated.predict_proba(X_new)[:, 1]
df["p_ability_place_raw"] = raw_p

# 2. 温度スケーリング（選別力維持）
T = 0.7  # <1 で尖らせる
scaled = raw_p ** (1 / T)

# 3. レース内正規化: sum(p_place) ≈ 3
race_sum = df.groupby("race_id")["p_ability_place_scaled"].transform("sum")
df["p_ability_place"] = scaled * (3.0 / race_sum.clip(lower=1e-6))

# 4. 整合性制約: p_place >= p_win
df["p_ability_place"] = np.maximum(df["p_ability_place"], df["p_ability_win"])

# 5. 再正規化（整合性強制後）
race_sum = df.groupby("race_id")["p_ability_place"].transform("sum")
df["p_ability_place"] = df["p_ability_place"] * (3.0 / race_sum.clip(lower=1e-6))
```

温度T=0.7により、強い馬はより強く、弱い馬はより弱く。過正規化（全馬p≈0.3への収束）を防止。

### 時系列バリデーション分割

```python
# 日付でソート済み前提。最後の20%を校正用に使用
dates = sorted(df["race_date"].unique())
split_date = dates[int(len(dates) * 0.8)]
place_train = df[df["race_date"] < split_date]
place_calib = df[df["race_date"] >= split_date]
```

### 古い p_ability_place の置き換え

`stage1_ability_model.py` の `add_ability_probs()` 内の `p_ability_place` 行を削除。

```python
# 旧 (削除)
df["p_ability_place"] = np.clip(df["p_ability_win"] * 3.0, 0, 1)
```

### SubmodelSet への追加

```python
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

## 1-3. 統合

### TrainingPipeline._train_submodel() 変更後フロー

```
[HorseHistoryFeatures.compute() → merge]         ← NEW
[hist.add_race_transforms(df)]                    ← NEW (z-score + pct)
MarketModel.train() → predict_and_calc_error()
AbilityModel.train() → add_ability_probs()        [p_ability_winのみ]
[PlaceAbilityModel.train() → predict()]           ← NEW
WinTwoStage → EVCorrection → PlaceTwoStage → Wide
```

### BacktestEngine 推論フロー

1. FeatureEngine.build_all() → HorseHistoryFeatures 呼び出し
2. AbilityModel.add_ability_probs() → p_ability_win のみ
3. PlaceAbilityModel.predict() → p_ability_place 設定（温度スケーリング + 正規化 + 整合性制約）
4. WinTwoStage → EVCorrection → PlaceTwoStage → Wide

## 1-4. Stage1 FEATURE_COLS 更新

```python
FEATURE_COLS = [
    # 既存 (7)
    "surface", "distance_bin", "track_condition_code",
    "grade_code", "field_size",
    "weight_diff_from_mean", "difficulty_score",
    # Phase 1: 馬の過去成績 (3)
    "norm_finish_logit_avg", "jockey_surprise", "haron_time_zscore_avg",
    # Phase 1: レース内z-score (3)
    "norm_finish_logit_avg_race_z",
    "jockey_surprise_race_z",
    "haron_time_zscore_avg_race_z",
    # Phase 1: レース内pct (3)
    "norm_finish_logit_avg_race_pct",
    "jockey_surprise_race_pct",
    "haron_time_zscore_avg_race_pct",
]
```

合計16特徴量（旧7 → 新16）。

---

# Phase 2: 追加特徴量拡張

## 2-1. 条件別騎手勝率（hierarchical smoothing）

`horse_history_features.py` に追加。

```python
# 距離bin × surface のみ（baba_cdは含めない → サンプル確保）
cond_wr = wins_in_cond / rides_in_cond
global_wr = total_wins / total_rides

# Empirical Bayes mixing
k = 25  # shrinking factor
w = rides_in_cond / (rides_in_cond + k)
smoothed_cond_wr = w * cond_wr + (1 - w) * global_wr
```

- 直近100戦を条件フィルタ
- 最小サンプル数: 10レース未満なら NaN → global_wr にshrink
- 特徴量名: `jockey_cond_wr`

### レース内変換

```python
# z-score + pct（Phase 1 と同じパターン）
df["jockey_cond_wr_race_z"] = ...
df["jockey_cond_wr_race_pct"] = ...
```

## 2-2. 負担重量絶対値

```python
# 馬の実際の負担重量（斤量）
df["weight_absolute"] = df["futan"]  # n_uma_race.futan
```

LGBMの非線形分割で処理（55kg vs 58kgの急激な悪化を自動捕捉）。

## 2-3. Phase 2 FEATURE_COLS最終

```python
FEATURE_COLS = [
    # 既存 (7)
    "surface", "distance_bin", "track_condition_code",
    "grade_code", "field_size",
    "weight_diff_from_mean", "difficulty_score",
    # Phase 1 (9)
    "norm_finish_logit_avg", "jockey_surprise", "haron_time_zscore_avg",
    "norm_finish_logit_avg_race_z", "jockey_surprise_race_z", "haron_time_zscore_avg_race_z",
    "norm_finish_logit_avg_race_pct", "jockey_surprise_race_pct", "haron_time_zscore_avg_race_pct",
    # Phase 2 (5)
    "jockey_cond_wr",
    "jockey_cond_wr_race_z",
    "jockey_cond_wr_race_pct",
    "weight_absolute",
]
```

合計21特徴量。

---

# Phase 3: RegimeDetector 実データ化

## 3-1. flb_slope（Favorite-Longshot Bias 傾き）

`features/market_bias_features.py` に新規関数 `compute_flb_slope()` を追加。

```python
def compute_flb_slope(race_feat_df):
    """過去200レースのオッズ-勝率関係からFLB傾きを推定。
    明示的に過去レースのみ使用（当該レースは含めない）。"""
    # 人気順位別の勝率を集計
    # オッズ逆数 vs 実勝率 の回帰傾き
    # 傾き < 1 → 人気薄が過小評価
    # 傾き > 1 → 人気薄が過大評価
```

## 3-2. odds_volatility_mean（時点固定）

```python
# 発走10分前 (t-10min) のオッズスナップショット時点の変動率のみ使用
# t-3min 直前データは late_money_filter 用なので混ぜない
from features.odds_dynamics_features import compute_rolling_volatility
stats["odds_volatility_mean"] = compute_rolling_volatility(race_feat_df)
```

## 3-3. 人気帯別回収率（EMA）

`rolling_roi_200` を3つの人気帯別EMA回収率に変更。

```python
POPULARITY_BANDS = {
    "favorite": (1, 3),   # 1-3人気
    "mid": (4, 7),        # 4-7人気
    "longshot": (8, 99),  # 8人気以降
}

# EMA (Exponential Moving Average) で時定数短縮
alpha = 0.07  # ≒50レースの時定数
for band_name, (lo, hi) in POPULARITY_BANDS.items():
    # 過去レースのみ（リーク防止）
    band_roi = compute_band_roi_ema(race_feat_df, lo, hi, alpha=alpha)
    stats[f"{band_name}_roi_ema"] = band_roi
```

**利点**: ほぼ定数になりにくい。市場状態の変化を捉えられる。EMAで遅延を最小化。

`training_pipeline.py` の `_build_regime_stats` でダミー値を置き換え:

```python
# 旧 (削除)
stats["flb_slope"] = 0.0
stats["odds_volatility_mean"] = 0.1
stats["rolling_roi_200"] = stats["rolling_roi"].fillna(0.5)

# 新
stats["flb_slope"] = compute_flb_slope(race_feat_df)
stats["odds_volatility_mean"] = compute_rolling_volatility(race_feat_df)
stats["favorite_roi_ema"] = ...
stats["mid_roi_ema"] = ...
stats["longshot_roi_ema"] = ...
```

RegimeDetector.FEATURE_COLS 更新:

```python
FEATURE_COLS = [
    "market_error_std",
    "market_error_mean",
    "market_entropy_mean",
    "overround_mean",
    "favorite_win_rate",
    "flb_slope",                # 実データ化
    "odds_volatility_mean",     # 実データ化
    "favorite_roi_ema",         # NEW: 人気帯別EMA
    "mid_roi_ema",              # NEW
    "longshot_roi_ema",         # NEW
    "field_size_mean",
]
```

---

# Phase 4: ベッティング戦略最適化

## 4-1. ワイド戦略改善（多様性 + 同一馬制約 + 相関制約）

```python
def select_bets(self, scored_pairs, ev_threshold, score_threshold, max_bets=3):
    candidates = [
        p for p in scored_pairs
        if p["ev_wide"] >= ev_threshold and p["wide_score_adj"] >= score_threshold
    ]
    candidates.sort(key=lambda x: x["wide_score_adj"], reverse=True)

    selected = []
    used_horses = set()
    used_bands = set()

    for pair in candidates:
        ha, hb = pair["umaban_a"], pair["umaban_b"]
        band_a = get_popularity_band(pair.get("popularity_rank_a", 0))
        band_b = get_popularity_band(pair.get("popularity_rank_b", 0))
        pair_band = (min(band_a, band_b), max(band_a, band_b))

        # 同一馬制約
        if ha in used_horses or hb in used_horses:
            continue
        # 同一人気帯制約
        if pair_band in used_bands:
            continue

        selected.append(pair)
        used_horses.update([ha, hb])
        used_bands.add(pair_band)

        if len(selected) >= max_bets:
            break

    return selected
```

## 4-2. Fractional Kelly サイジング

```python
FRACTIONAL_KELLY = 0.5  # Half Kelly（保守的）

def calc_stake(self, ev_lower_ci, odds, bankroll, bet_type):
    # CI下限ベースKelly
    kelly_fraction = max(0, (ev_lower_ci - 1.0) / (odds - 1.0))
    stake = kelly_fraction * FRACTIONAL_KELLY * bankroll
    # 2%キャップ
    stake = min(stake, bankroll * 0.02)
    # 100円単位切り下げ
    stake = int(stake // 100) * 100
    return max(stake, 0)
```

バックテストで `FRACTIONAL_KELLY` を 0.25-0.75 の感応度分析。

## 4-3. EV閾値チューニング

レジーム別EV閾値をウォークフォワードCV内で最適化。ホールドアウト期間では最適化済み閾値を固定（パラメータフリーズ）。

---

# Phase 5: 検証基盤強化

## 5-1. Walk-forward CV

新規ノートブック: `notebooks/12_time_series_cv.ipynb`

```
Window 1: 2018-2021 train → 2022 test
Window 2: 2019-2022 train → 2023 test
Window 3: 2020-2023 train → 2024 test
```

各ウィンドウで ROI, Max DD, ヒット率, Logloss を計測。

## 5-2. パラメータフリーズプロトコル

ホールドアウト期間中に以下が変更されていないことを検証:

1. モデルパラメータ (LightGBM booster)
2. 校正パラメータ (Isotonic calibration map)
3. Feature engineering 設定 (FEATURE_COLS の定義)
4. EV閾値、Kelly係数、レジーム閾値
5. **Feature engineering コード** (git commit hash で記録)

## 5-3. 自動ホールドアウト検証

`BacktestValidationSuite.run_all()` をノートブックから直接呼び出し可能に。

基準:
- Place ROI >= 100%
- Wide ROI >= 103%
- Overall ROI >= 101%
- Max DD <= 16%
- 36ヶ月中22ヶ月以上黒字

---

# テスト設計

## test_horse_history_features.py（新規）

- norm_finish_logit: 1着/16頭 ≈ 2.94、最下位 ≈ -2.94 の境界値テスト
- 8頭未満レース: NaNになるテスト
- clip幅: 0.05/0.95 で極端値が抑制されるテスト
- jockey_surprise: Beta事前分布スムージングの正確性テスト
- jockey_surprise: 控除率補正（PAYOUT_RATE）が適用されるテスト
- haron_z: 階層fallback (L1→L2→L3→L4) のテスト
- レース内z-score: std=0 の場合にNaNにならないテスト
- レース内pct: 正しいパーセンタイル計算テスト
- リーク防止: 当該レース日付より後のデータが含まれないテスト
- 欠損値: 初出走馬、最小サンプル未満騎手のNaNテスト

## test_place_ability_model.py（新規）

- クラス不均衡: scale_pos_weight が正しく計算されるテスト
- 確率範囲: 出力が [0, 1] に収まるテスト
- レース内正規化: sum(p_place) ≈ 3 になるテスト
- 温度スケーリング: T<1 で分布が尖るテスト
- 整合性制約: p_place >= p_win が満たされるテスト
- FEATURE_COLS: 新規列が全て存在するテスト
- 時系列分割: 校正データが学習データより未来の期間であるテスト

## test_training_pipeline.py（修正）

- 統合テスト: PlaceAbilityModel が AbilityModel 直後に学習されるテスト
- p_ability_place 置き換え: PlaceAbilityModel の出力が正しく設定されるテスト
- HorseHistoryFeatures 結合: merge 後に全特徴量列が存在するテスト
- SubmodelSet: place_ability フィールドが正しく格納されるテスト

---

# データリーク対策

- **同日複数レース**: JRA仕様で同一馬が同日に複数出走しないため問題なし
- **過去データ取得**: レース日付より前（`< race_date`）のデータのみ使用。同日のレースも除外
- **特徴量生成のjoin**: HorseHistoryFeatures.compute() は TrainingPipeline 内で実行、当該レース除外
- **校正データ分離**: PlaceAbilityModel の校正データは学習データより未来の期間（時系列分割）
- **flb_slope**: 過去レースのみで計算（未来情報禁止）
- **EV閾値最適化**: ウォークフォワード内でのみ実施（ホールドアウトでは固定）

---

# 変更ファイル一覧

| ファイル | Phase | 変更種別 | 概要 |
|---------|-------|---------|------|
| `src/features/horse_history_features.py` | 1,2 | **新規** | 3基本 + レース内変換(z+pct) + jockey_cond_wr |
| `src/models/place_ability_model.py` | 1 | **新規** | LGBMClassifier + Isotonic + 温度スケーリング |
| `src/models/stage1_ability_model.py` | 1 | 修正 | FEATURE_COLS 16列化、p_ability_place行削除 |
| `src/domain/models.py` | 1 | 修正 | SubmodelSet に place_ability 追加 |
| `src/pipelines/training_pipeline.py` | 1,2,3 | 修正 | HorseHistory + PlaceAbility + RegimeStats実データ化 |
| `src/backtest/engine.py` | 1 | 修正 | HorseHistoryFeatures + PlaceAbilityModel 推論追加 |
| `src/features/market_bias_features.py` | 3 | 修正 | compute_flb_slope() 追加 |
| `src/features/odds_dynamics_features.py` | 3 | 修正 | compute_rolling_volatility() 追加 |
| `src/betting/wide_strategy.py` | 4 | 修正 | 同一馬制約 + 人気帯多様性制約 |
| `src/betting/stake_calculator.py` | 4 | 修正 | Fractional Kelly + CI下限ベース |
| `src/backtest/validation_suite.py` | 5 | 修正 | run_all() のノートブック呼び出し対応 |
| `notebooks/12_time_series_cv.ipynb` | 5 | **新規** | Walk-forward CV 分析ノートブック |
| `tests/test_horse_history_features.py` | 1 | **新規** | 特徴量計算テスト |
| `tests/test_place_ability_model.py` | 1 | **新規** | モデル・calibrationテスト |
| `tests/test_training_pipeline.py` | 1 | 修正 | 統合テスト追加 |

---

# 評価設計

## 各Phase評価指標

| 指標 | 目的 | ツール |
|------|------|--------|
| ROI | 投資効率 | 既存バックテスト |
| Logloss | 確率予測精度 | `sklearn.metrics.log_loss` |
| Rank correlation | 順位予測の正しさ | `scipy.stats.spearmanr` |
| Calibration curve | 確率の校正確認 | `sklearn.calibration.calibration_curve` |

## ベースライン比較

オッズ逆数（市場予測）をベースラインとして比較:

```python
p_baseline = (1 / odds) / sum(1 / odds_for_race)  # レース内正規化
```

## 最終目標

- Place ROI >= 100%
- Wide ROI >= 103%
- Overall ROI >= 101%
- Max DD <= 16%
- 36ヶ月中22ヶ月以上黒字
