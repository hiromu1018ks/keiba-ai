# Feature Catalog

keiba-ai v5.5 の全モデルが使用する特徴量の完全カタログ。
最終更新: 2026-03-30

## モデル一覧と特徴量数

| モデル | 特徴量数 | ファイル | 役割 |
|--------|--------:|----------|------|
| AbilityModel (Stage1) | 30 | `src/models/stage1_ability_model.py` | 能力スコア推定 (lambdarank) |
| PlaceAbilityModel | 31 | `src/models/place_ability_model.py` | 複勝能力スコア推定 |
| MarketModel | 7 | `src/models/market_model.py` | 市場確率予測 |
| WinTwoStageModel | 16 | `src/models/two_stage_return_model.py` | 単勝EV推定 P(hit)×E(return) |
| PlaceTwoStageModel | 16 | `src/models/two_stage_return_model.py` | 複勝EV推定 (FEATURE_COLS = Win同一) |
| EVCorrectionModel | 19 | `src/models/ev_correction_model.py` | EV補正 (P補正+E補正) |
| WideTwoStageModel | 5 | `src/models/wide_two_stage_model.py` | ワイドEV推定 |
| RaceQualityScreener | 20 | `src/models/race_quality_screener.py` | レース品質スクリーニング |
| RegimeDetector | 11 | `src/models/regime_detector.py` | 市場レジーム検出 |

---

## 特徴量テーブル

### Group A: レース条件 (Race Condition)

FeatureEngine._map_basic_features() が生成。DB列からのリネーム/推導。

| # | 列名 | 型 | 生成元 | DBソース | 計算ロジック | 消費モデル | 時系列安全性 | 備考 |
|---|------|----|--------|----------|-------------|-----------|------------|------|
| 1 | `surface` | cat | _map_basic | `track_cd` | 10-22→turf, 23-29→dirt | S1,PA,MM,W2,P2,EV,W2S,RQS | SAFE | カテゴリ変数 |
| 2 | `distance_bin` | cat | _map_basic | `track_cd`+`kyori` | sprint(≤1400)/mile(1401-1700)/intermediate(1701-2100)/long(≥2101) | S1,PA,MM,W2,P2,EV,W2S,RQS | SAFE | surface+distanceから推導 |
| 3 | `track_condition_code` | cat | _map_basic | `baba_cd` | リネームのみ | S1,PA,MM,W2,P2,EV,W2S,RQS | SAFE | 馬場状態コード |
| 4 | `grade_code` | cat | _map_basic | `grade_cd` | リネームのみ | S1,PA,MM,W2,P2,RQS | SAFE | グレードコード |
| 5 | `field_size` | num | _map_basic | race_df | 出走頭数 | S1,PA,MM,W2,P2,EV,W2S,RQS,RD | SAFE | 16頭立等 |
| 6 | `popularity_rank` | num | _map_basic | `ninki` | リネームのみ | W2,P2,EV,RQS | SAFE | 人気順位 |

### Group B: レース内相対 (Intra-Race)

`src/features/intra_race_features.py` が生成。

| # | 列名 | 型 | 生成元 | DBソース | 計算ロジック | 消費モデル | 時系列安全性 | 備考 |
|---|------|----|--------|----------|-------------|-----------|------------|------|
| 7 | `weight_diff_from_mean` | num | intra_race | `ba_taijyu` | 馬体重 - レース平均 | S1,PA,MM | SAFE | レース内正規化 |
| 8 | `odds_rank` | num | intra_race | `win_odds` | rank(win_odds) within race | (未使用) | SAFE | 将来用 |

### Group C: レース難易度 (Race Difficulty)

`src/features/race_difficulty_model.py` が生成。

| # | 列名 | 型 | 生成元 | DBソース | 計算ロジック | 消費モデル | 時系列安全性 | 備考 |
|---|------|----|--------|----------|-------------|-----------|------------|------|
| 9 | `difficulty_score` | num | race_difficulty | grade+field_size+entropy | grade_weight × field_factor × entropy_norm, clipped [0,1] | S1,PA,MM,RQS | SAFE | 複合スコア |

### Group D: 過去走成績 (Past Performance)

`src/features/horse_history_features.py` が生成。searchsorted で過去データのみ使用 (last 3 runs)。

| # | 列名 | 型 | 生成元 | DBソース | 計算ロジック | 消費モデル | 時系列安全性 | 備考 |
|---|------|----|--------|----------|-------------|-----------|------------|------|
| 10 | `norm_finish_logit_avg` | num | HorseHistory | 出走馬SE | logit正規化着順の平均 (last 3) | S1,PA | SAFE (searchsorted) | |
| 11 | `haron_time_l3_avg` | num | HorseHistory | 出走馬SE | ハロンタイム平均 (last 3) | S1,PA | SAFE | |
| 12 | `haron_time_l3_zscore` | num | HorseHistory | 出走馬SE | 距離ビンz-score (last 3) | S1,PA | SAFE | std==0 → NaN |
| 13 | `time_diff_avg` | num | HorseHistory | 出走馬SE | 勝馬差時間平均 (last 3) | S1,PA | SAFE | |
| 14 | `corner_1c_avg` | num | HorseHistory | 出走馬SE | 1コーナー通過順位平均 (last 3) | S1,PA | SAFE | |
| 15 | `corner_4c_avg` | num | HorseHistory | 出走馬SE | 4コーナー通過順位平均 (last 3) | S1,PA | SAFE | |
| 16 | `closing_index_avg` | num | HorseHistory | 出走馬SE | (norm_4C - norm_finish) 平均 (last 3) | S1,PA | SAFE | しぶとさ指標 |
| 17 | `kyakusitu_cd` | cat | HorseHistory | 出走馬SE | 直近走の脚質コード (1=逃/2=先/3=差/4=追) | S1,PA | SAFE | カテゴリ変数 |

### Group E: 馬体 (Horse Body)

HorseHistoryFeatures.compute() 内で生成。

| # | 列名 | 型 | 生成元 | DBソース | 計算ロジック | 消費モデル | 時系列安全性 | 備考 |
|---|------|----|--------|----------|-------------|-----------|------------|------|
| 18 | `weight_absolute` | num | HorseHistory | `ba_taijyu` | 当レース馬体重 | S1,PA | SAFE | |

### Group F: レース内順位変換 (Intra-Race Rank)

HorseHistoryFeatures.add_race_transforms() が生成。レース内 percentile rank。

| # | 列名 | 型 | 生成元 | DBソース | 計算ロジック | 消費モデル | 時系列安全性 | 備考 |
|---|------|----|--------|----------|-------------|-----------|------------|------|
| 19 | `norm_finish_logit_avg_race_rank` | num | add_race_transforms | (D列) | pct_rank within race | S1,PA | SAFE | |
| 20 | `haron_time_l3_avg_race_rank` | num | add_race_transforms | (D列) | pct_rank within race | S1,PA | SAFE | |
| 21 | `time_diff_avg_race_rank` | num | add_race_transforms | (D列) | pct_rank within race | S1,PA | SAFE | |
| 22 | `corner_1c_avg_race_rank` | num | add_race_transforms | (D列) | pct_rank within race | S1,PA | SAFE | |
| 23 | `closing_index_avg_race_rank` | num | add_race_transforms | (D列) | pct_rank within race | S1,PA | SAFE | |

### Group G: 血統 (Bloodline)

`src/features/bloodline_features.py` が生成。x_UMA 産駎統計から Beta(1,10) 平滑化。

| # | 列名 | 型 | 生成元 | DBソース | 計算ロジック | 消費モデル | 時系列安全性 | 備考 |
|---|------|----|--------|----------|-------------|-----------|------------|------|
| 24 | `blood_surface_wr` | num | Bloodline | x_UMA `ba1chakukaisu*` | Beta(1,10) 芝/ダート勝率 | S1,PA | ⚠️ 注意 | 全期間集計 (時系列制約なし) |
| 25 | `blood_distance_wr` | num | Bloodline | x_UMA `kyori1chakukaisu*` | Beta(1,10) 距離別勝率 | S1,PA | ⚠️ 注意 | 同上 |
| 26 | `blood_condition_wr` | num | Bloodline | — | **Phase 2 プレースホルダー** | S1,PA | — | 常にNaN |
| 27 | `blood_total_wr` | num | Bloodline | x_UMA `chuochakukaisu*` | Beta(1,10) 全体勝率 | S1,PA | ⚠️ 注意 | 全期間集計 |
| 28 | `blood_prize_log` | num | Bloodline | x_UMA `ruikeihonsyoheichi` | log(1 + 累計賞金) | S1,PA | ⚠️ 注意 | 累計値 |
| 29 | `blood_keito_cd` | cat | Bloodline | — | **Phase 2 プレースホルダー** | S1,PA | — | 常にNaN |

### Group H: 交互作用 (Interaction)

`src/features/interaction_features.py` が生成。D列とA列の交差。

| # | 列名 | 型 | 生成元 | DBソース | 計算ロジック | 消費モデル | 時系列安全性 | 備考 |
|---|------|----|--------|----------|-------------|-----------|------------|------|
| 30 | `kyakusitu_x_distance` | cat | interaction | (D+F列) | str(kyakusitu) + "_" + str(distance_bin) | S1,PA | SAFE | カテゴリ変数 |
| 31 | `kyakusitu_x_surface` | cat | interaction | (D+F列) | str(kyakusitu) + "_" + str(surface) | S1,PA | SAFE | カテゴリ変数 |
| 32 | `weight_x_distance` | num | interaction | (E列) | weight_absolute × distance | S1,PA | SAFE | |

### Group I: 市場バイアス (Market Bias)

`src/features/market_bias_features.py` が生成。

| # | 列名 | 型 | 生成元 | DBソース | 計算ロジック | 消費モデル | 時系列安全性 | 備考 |
|---|------|----|--------|----------|-------------|-----------|------------|------|
| 33 | `p_market_win_adj` | num | market_bias | `tan_odds` | (1/odds) / Σ(1/odds) | MM (target) | SAFE | MarketModelのターゲット変数 |
| 34 | `market_entropy` | num | market_bias | `tan_odds` | Shannon entropy of p_market | W2,P2,EV,RQS,RD | SAFE | |
| 35 | `overround` | num | market_bias | `tan_odds` | Σ(1/odds) - 1.0 | W2,P2,EV,RQS,RD | SAFE | ブックメーカーマージン |

### Group J: オッズダイナミクス (Odds Dynamics)

`src/features/odds_dynamics_features.py` が生成。

| # | 列名 | 型 | 生成元 | DBソース | 計算ロジック | 消費モデル | 時系列安全性 | 備考 |
|---|------|----|--------|----------|-------------|-----------|------------|------|
| 36 | `odds_drop_rate_60_10` | num | odds_dynamics | 時系列オッズ | (odds@60 - odds@10) / odds@60 | W2,P2 | SAFE | 学習時NaN (メモリ節約) |
| 37 | `odds_drop_rate_30_10` | num | odds_dynamics | 時系列オッズ | (odds@30 - odds@10) / odds@30 | W2,P2 | SAFE | 同上 |
| 38 | `odds_velocity` | num | odds_dynamics | 時系列オッズ | 線形回帰スロープ | W2,P2 | SAFE | 同上 |
| 39 | `odds_volatility` | num | odds_dynamics | 時系列オッズ | 連続変動のstd | W2,P2 | SAFE | 同上 |
| 40 | `popularity_change_30_10` | num | odds_dynamics | 時系列オッズ | ninki(t-30) - ninki(t-10) | W2,P2 | SAFE | 同上 |

### Group K: 市場モデル出力 (Market Model Output)

`src/models/market_model.py` の predict_and_calc_error() が生成。

| # | 列名 | 型 | 生成元 | DBソース | 計算ロジック | 消費モデル | 時系列安全性 | 備考 |
|---|------|----|--------|----------|-------------|-----------|------------|------|
| 41 | `signed_log_error_win` | num | MarketModel | (I+M列) | log(p_market / p_pred) | W2,P2,EV | IN-SAMPLE | OOF未対応 (Phase 3) |
| 42 | `abs_log_error_win` | num | MarketModel | (I+M列) | abs(signed_log_error) | W2,P2,EV | IN-SAMPLE | 同上 |

### Group L: Stage1 出力 (Stage1 Prediction)

AbilityModel が生成。OOF対応済み (train_oof)。

| # | 列名 | 型 | 生成元 | DBソース | 計算ロジック | 消費モデル | 時系列安全性 | 備考 |
|---|------|----|--------|----------|-------------|-----------|------------|------|
| 43 | `p_ability_win` | num | AbilityModel | (A-H列) | LightGBM Ranker softmax出力 | PA,W2,P2 | OOF対応済 | K-fold expanding window |

### Group M: Stage2 出力 (Two-Stage Output)

WinTwoStageModel / PlaceTwoStageModel が生成。

| # | 列名 | 型 | 生成元 | DBソース | 計算ロジック | 消費モデル | 時系列安全性 | 備考 |
|---|------|----|--------|----------|-------------|-----------|------------|------|
| 44 | `p_win_pred` | num | WinTwoStage | (J+K+L列) | P(hit) LightGBM出力 | EV (init_score) | SAFE | init_score として使用 |
| 45 | `e_return_win_pred` | num | WinTwoStage | (J+K+L列) | E(return|hit) LightGBM出力 | EV | SAFE | |

### Group N: 騎手コンテキスト (Jockey Context)

`src/features/jockey_context_features.py` が生成。x_KISYU_SEISEKI から抽出。SetYear < race_year 制約。

| # | 列名 | 型 | 生成元 | DBソース | 計算ロジック | 消費モデル | 時系列安全性 | 備考 |
|---|------|----|--------|----------|-------------|-----------|------------|------|
| 46 | `jockey_wr_overall` | num | JockeyContext | x_KISYU_SEISEKI `heichichakukaisu*` | Beta(1,10) 全体勝率 | EV | SAFE | SetYear < race_year |
| 47 | `jockey_wr_distance` | num | JockeyContext | x_KISYU_SEISEKI `kyori1chakukaisu*` | Beta(1,10) 距離別勝率 | EV | SAFE | 同上 |
| 48 | `jockey_wr_venue` | num | JockeyContext | x_KISYU_SEISEKI `jyo5chakukaisu*` | Beta(1,10) 競馬場別勝率 | EV | SAFE | 同上 |
| 49 | `jockey_prize_log` | num | JockeyContext | x_KISYU_SEISEKI `honsyokinheichi` | log(1 + 賞金) | EV | SAFE | 同上 |

### Group O: 調教師コンテキスト (Trainer Context)

`src/features/trainer_context_features.py` が生成。x_CHOKYO_SEISEKI から抽出。SetYear < race_year 制約。

| # | 列名 | 型 | 生成元 | DBソース | 計算ロジック | 消費モデル | 時系列安全性 | 備考 |
|---|------|----|--------|----------|-------------|-----------|------------|------|
| 50 | `trainer_wr_overall` | num | TrainerContext | x_CHOKYO_SEISEKI `heichichakukaisu*` | Beta(1,10) 全体勝率 | EV | SAFE | SetYear < race_year |
| 51 | `trainer_wr_distance` | num | TrainerContext | x_CHOKYO_SEISEKI `kyori1chakukaisu*` | Beta(1,10) 距離別勝率 | EV | SAFE | 同上 |
| 52 | `trainer_wr_venue` | num | TrainerContext | x_CHOKYO_SEISEKI `jyo5chakukaisu*` | Beta(1,10) 競馬場別勝率 | EV | SAFE | 同上 |
| 53 | `trainer_prize_log` | num | TrainerContext | x_CHOKYO_SEISEKI `honsyokinheichi` | log(1 + 賞金) | EV | SAFE | 同上 |

### Group P: EV補正モデル交互作用 (EV Correction Interaction)

EVCorrectionModel._add_interaction_features() が生成。

| # | 列名 | 型 | 生成元 | DBソース | 計算ロジック | 消費モデル | 時系列安全性 | 備考 |
|---|------|----|--------|----------|-------------|-----------|------------|------|
| 54 | `p_x_e_interaction` | num | EVCorrection | (M列) | p_win_pred × e_return_win_pred | EV | SAFE | |
| 55 | `p_minus_e_gap` | num | EVCorrection | (M列) | abs(log(p) - log(E)) | EV | SAFE | |

### Group Q: 情報非対称性 (Info Asymmetry)

`src/features/info_asymmetry_features.py` が生成。expanding().shift(1) でリーク防止。

| # | 列名 | 型 | 生成元 | DBソース | 計算ロジック | 消費モデル | 時系列安全性 | 備考 |
|---|------|----|--------|----------|-------------|-----------|------------|------|
| 56 | `hist_hit_rate_topk` | num | info_asymmetry | 過去レース | expanding().mean().shift(1) | RQS | SAFE | レースレベル |
| 57 | `hist_roi_topk` | num | info_asymmetry | 過去レース | expanding().mean().shift(1) | RQS | SAFE | |
| 58 | `hist_positive_return_ratio` | num | info_asymmetry | 過去レース | expanding().mean().shift(1) | RQS | SAFE | |
| 59 | `hist_win_rate_same_condition` | num | info_asymmetry | 過去レース | GroupBy(surface+dist) expanding | RQS | SAFE | |
| 60 | `hist_market_entropy_avg` | num | info_asymmetry | 過去レース | GroupBy(surface+dist) expanding | RQS | SAFE | |

### Group R: レジーム検出 (Regime Detection)

パイプライン内で生成されるローリング統計量。

| # | 列名 | 型 | 生成元 | DBソース | 計算ロジック | 消費モデル | 時系列安全性 | 備考 |
|---|------|----|--------|----------|-------------|-----------|------------|------|
| 61 | `market_error_std` | num | pipeline | (K列) | 200レースローリングstd | RD | SAFE | |
| 62 | `market_error_mean` | num | pipeline | (K列) | 200レースローリングmean | RD | SAFE | |
| 63 | `market_entropy_mean` | num | pipeline | (I列) | 200レースローリングmean | RD | SAFE | |
| 64 | `overround_mean` | num | pipeline | (I列) | 200レースローリングmean | RD | SAFE | |
| 65 | `favorite_win_rate` | num | pipeline | finish_pos | 200レースローリング1番人気勝率 | RD | SAFE | |
| 66 | `flb_slope` | num | market_bias | オッズ | Favourite-Longshot Bias スロープ | RD | SAFE | |
| 67 | `odds_volatility_mean` | num | odds_dynamics | 時系列オッズ | ローリングボラティリティ平均 | RD | SAFE | |
| 68 | `favorite_roi_ema` | num | odds_dynamics | オッズ | 人気馬ROI EMA | RD | SAFE | |
| 69 | `mid_roi_ema` | num | odds_dynamics | オッズ | 中人気ROI EMA | RD | SAFE | |
| 70 | `longshot_roi_ema` | num | odds_dynamics | オッズ | 大穴ROI EMA | RD | SAFE | |
| 71 | `field_size_mean` | num | pipeline | race_df | ローリング平均頭数 | RD | SAFE | |

### Group S: 補助特徴量 (Auxiliary — 未使用)

SubModelManager.add_distance_band_features() が生成。現在どのモデルのFEATURE_COLSにも含まれない。

| # | 列名 | 型 | 生成元 | DBソース | 計算ロジック | 消費モデル | 時系列安全性 | 備考 |
|---|------|----|--------|----------|-------------|-----------|------------|------|
| 72 | `is_turf_sprint` | bool | SubModelManager | — | turf & dist ≤ 1400 | (なし) | SAFE | |
| 73 | `is_turf_mile` | bool | SubModelManager | — | turf & dist 1401-1700 | (なし) | SAFE | |
| 74 | `is_turf_intermediate` | bool | SubModelManager | — | turf & dist 1701-2100 | (なし) | SAFE | |
| 75 | `is_turf_long` | bool | SubModelManager | — | turf & dist ≥ 2101 | (なし) | SAFE | |
| 76 | `is_dirt_sprint` | bool | SubModelManager | — | dirt & dist ≤ 1400 | (なし) | SAFE | |
| 77 | `is_dirt_mile` | bool | SubModelManager | — | dirt & dist 1401-1700 | (なし) | SAFE | |
| 78 | `is_dirt_intermediate` | bool | SubModelManager | — | dirt & dist ≥ 1701 | (なし) | SAFE | |
| 79 | `is_good_track` | bool | SubModelManager | — | track_condition_code in [1,2] | (なし) | SAFE | |
| 80 | `is_soft_track` | bool | SubModelManager | — | track_condition_code in [3,4] | (なし) | SAFE | |

---

## 消費モデル略称

| 略称 | モデル | 特徴量数 |
|------|--------|--------:|
| S1 | AbilityModel (Stage1) | 30 |
| PA | PlaceAbilityModel | 31 |
| MM | MarketModel | 7 |
| W2 | WinTwoStageModel | 16 |
| P2 | PlaceTwoStageModel | 16 |
| EV | EVCorrectionModel | 19 |
| W2S | WideTwoStageModel | 5 |
| RQS | RaceQualityScreener | 20 |
| RD | RegimeDetector | 11 |

## 時系列安全性凡例

| マーク | 意味 |
|--------|------|
| SAFE | 過去データのみ使用。未来リークなし |
| OOF対応済 | in-sampleだったがOOF (train_oof) で修正済 |
| IN-SAMPLE | 同じ学習データで予測。既知のリーク (Phase 3対応予定) |
| ⚠️ 注意 | 全期間集計 (時系列制約なし)。影響小と推定 |
| — | プレースホルダー (常にNaN) |

## データフロー

```
FeatureEngine.build_all()
  │
  ├─→ Group A (レース条件) ─────────────────────────────────────────────┐
  ├─→ Group B (レース内相対) ───────────────────────────────────────────┤
  ├─→ Group C (レース難易度) ───────────────────────────────────────────┤
  ├─→ Group I (市場バイアス) ───────────────────────────────────────────┤
  ├─→ Group G (血統) ──────────────────────────────────────────────────┤
  ├─→ Group J (オッズダイナミクス) ─────────── NaN in training ────────┤
  └─→ Group S (補助) ──── 未使用 ─────────────────────────────────────┤
                                                                       │
TrainingPipeline._train_submodel()                                     │
  │                                                                    │
  ├─→ Group D (過去走成績) ──── searchsorted ─────────────────────────┤
  ├─→ Group E (馬体) ─────────────────────────────────────────────────┤
  ├─→ Group F (レース内順位) ─────────────────────────────────────────┤
  ├─→ Group H (交互作用) ─────────────────────────────────────────────┤
  │                                                                    │
  ├─→ MarketModel ──→ Group K (市場モデル出力) ────────────────────────┤
  │                                                                    │
  ├─→ AbilityModel.train_oof() ──→ Group L (Stage1出力) ──────────────┤
  │                                                                    │
  ├─→ WinTwoStageModel ──→ Group M (Stage2出力) ──────────────────────┤
  │                                                                    │
  ├─→ Group N (騎手コンテキスト) ──── SetYear < race_year ────────────┤
  ├─→ Group O (調教師コンテキスト) ── SetYear < race_year ────────────┤
  │                                                                    │
  └─→ EVCorrectionModel ──→ Group P (EV補正交互作用) ─────────────────┘
                                                                       │
BacktestEngine (推論時)                                                │
  │                                                                    │
  ├─→ Group Q (情報非対称性) ──── expanding.shift(1) ──→ RQS          │
  └─→ Group R (レジーム検出) ──── 200レースローリング ──→ RD          │
```
