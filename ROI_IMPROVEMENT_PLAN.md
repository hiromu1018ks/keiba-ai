# ROI 100%超えに向けた改善計画

作成日: 2026-05-27  
対象: 競馬AI予測システム v5.5 単勝/複勝/ワイド投資判断

## 1. 目的

本計画の目的は、2024/2025にだけ合う係数調整ではなく、本番運用でROI 100%超えを狙える投資AIへ構造的に改善すること。

単なるバックテスト最適化ではなく、以下を満たす改修を優先する。

- ベット数を過剰に減らしてROIを上げる方針は採用しない
- レジーム検出には依存しない
- 2024/2025の結果だけを見た係数合わせはしない
- OOF、ウォークフォワード、年別安定性で配備可否を判定する
- 最終的には単勝1頭固定にこだわらず、利益最大化のために複数頭、複数券種、配分最適化を扱える形にする

## 2. 現状認識

### 2.1 直近のバックテスト結果

主な結果は以下。

| Run | 主な内容 | 2024 ROI | 2025 ROI | 合計ROI |
|---|---|---:|---:|---:|
| #30 | surface別スコアリング後 | 90.9% | 91.1% | 91.0% |
| #31 | EV tail shrinkage改善後 | 90.9% | 92.3% | 91.6% |
| #32 | WinSegmentCalibrator導入 | 91.6% | 87.0% | 89.3% |
| #33 | WSC strict化後、旧コード | 88.5% | 87.2% | 87.8% |

#33の内訳:

| 年 | surface | bets | ROI | hit rate | avg odds | avg p |
|---|---|---:|---:|---:|---:|---:|
| 2024 | turf | 1,649 | 83.9% | 19.8% | 10.84 | 18.5% |
| 2024 | dirt | 1,678 | 93.0% | 23.2% | 7.87 | 22.6% |
| 2025 | turf | 1,689 | 89.5% | 24.6% | 5.30 | 27.7% |
| 2025 | dirt | 1,646 | 84.8% | 15.9% | 10.36 | 16.9% |

### 2.2 #33から分かったこと

単純にWSCを止めればよい、という結論ではない。

反実仮想再スコアでは以下。

| 条件 | 2024 ROI | 2025 ROI |
|---|---:|---:|
| #33実績 | 88.45% | 87.17% |
| WSCなし、同じポリシー | 88.55% | 84.13% |
| 芝ポリシーデフォルト、WSCあり | 90.01% | 87.17% |
| WSCなし、芝ポリシーデフォルト | 89.83% | 84.13% |

解釈:

- 2024の悪化は、WSCそのものより芝のWinSelectionPolicy過配備が大きい
- 2025芝ではWSCがむしろ効いている可能性が高い
- 2025ダートの悪化はWSC対象外なので、WSC停止では解決しない
- 現状は「確率推定の過信」「最終順位付けの不安定」「surface/year間の再現性不足」が主因

### 2.3 特徴量の棚卸し

`data/features/horse_features.parquet`:

- 138,153行
- 368列
- 期間: 2020-12-26から2023-12-28

`data/backtest/bt_2025_horse_features.parquet`:

- 46,160行
- 438列

カテゴリ別の概況:

| カテゴリ | 概数 | 評価 |
|---|---:|---|
| 市場/オッズ | 約36列 | 量はあるが、最終投資判断での利用がまだ薄い |
| late odds | 約7列 | 変化率中心。カーブ形状、安定性、時点別比較が不足 |
| 馬履歴/フォーム | 約28列 | 基礎はある |
| 騎手/調教師 | 約10列 | 基礎はあるが相互作用が薄い |
| 血統 | 約14列 | 基礎はあるが欠損が多い列あり |
| コース/馬場/距離 | 約44列 | 量は十分 |
| ペース/ラップ | 約47列 | 量は十分。ただし欠損列が多い |
| レース内相対特徴 | 約49列 | 重要。さらに投資用に強化余地あり |
| モデル出力 | 約6列 | 最終投資用には不足 |

欠損状況:

- 50%以上欠損: 39列
- 100%欠損または実質定数の列が複数存在
- `dm_time_*`, `dam_*`, `course_record_time`, `jyuni*` などは現状のままだと寄与が限定的

結論:

- ベース予測モデル用の特徴量の量は足りている
- 投資AIとしての最終判断用特徴量は不足している
- 特に「市場とのズレを説明する特徴量」「確率キャリブレーション用特徴量」「レース内ポートフォリオ用特徴量」が不足している

## 3. 根本課題

### 3.1 確率が投資判断に耐えていない

ROIが100%を超えない主因は、最終選定の係数ではなく、EV計算に使う確率の信用度にある。

現状は以下の問題がある。

- `p_win_final`がsurface、odds帯、prob rank帯、年度でズレる
- EVが平均的に過大評価される区間がある
- 高EV、高オッズ、低確率帯で分散が大きい
- 市場確率を強い事前分布として使い切れていない

投資AIでは、勝率の順位が少し良いだけでは不十分。オッズに対して過小評価されている馬を、確率として正しく見積もる必要がある。

### 3.2 最終セレクタが薄い特徴量で判断している

ベース特徴量は368から438列あるが、最終選定OOFは41列程度。

`win_selection_oof.parquet`の主な列:

- `p_win_oof`
- `p_win_pred`
- `p_win_corrected`
- `win_selection_prob`
- `win_selection_ev`
- `win_selection_edge`
- `win_market_logit_edge`
- `win_market_value_ratio`
- `win_market_selection_score`
- late odds関連

これは、最終投資判断がほぼ「モデル確率」「市場残差」「EV」「オッズ変化」だけで行われていることを意味する。

ベースモデルが外した時に、セレクタ側で復元する情報が足りない。

### 3.3 ROI直接最適化がノイズを拾いやすい

単勝は的中が疎で、特に高オッズ帯は数件の的中でROIが大きく動く。

そのため、以下は危険。

- 年度別の実ROIだけで係数を決める
- 高オッズ帯の偶然の的中を強化する
- 特定年度に効いたlate odds重みをそのまま配備する

配備条件は、ROIだけではなく以下の複合指標にする。

- logloss
- Brier
- calibration ECE
- odds帯別actual/pred
- surface別actual/pred
- CLV
- 年別の悪化有無

## 4. 改善方針

ROI 100%超えを狙うには、次の順で構造を作り直す。

1. OOFを投資判断用の完全な学習基盤にする
2. Market-awareな勝率キャリブレーションを新設する
3. 最終セレクタをrace-level rankerへ置き換える
4. 単勝1頭固定からrace-level portfolioへ拡張する
5. 単勝、複勝、ワイドの横断選択へ広げる

## 5. Phase 0: OOF/成果物健全性の再保証

### 5.1 目的

以後の全改善がOOFに依存するため、OOFが壊れた状態で学習や配備が進まないようにする。

### 5.2 実装内容

対象候補:

- `src/pipelines/training_pipeline.py`
- `src/models/*`
- `tests/test_training_pipeline.py`
- 新規 `src/validation/oof_health.py`

実装するチェック:

- `data/oof/oof_predictions.parquet` が0行なら保存禁止
- `data/oof/win_selection_oof.parquet` が0行なら保存禁止
- OOFに `is_oof=True` とfold列を必須化
- race_id単位でtrain/validの重複を検査
- OOF top1 hit rateが異常値なら停止
- OOF top1 ROIが異常値なら停止
- 同一race_idの複数fold混入を検査

異常判定の初期値:

- top1 hit rate > 35%なら警告または停止
- top1 ROI > 200%なら停止
- OOF行数 < 期待行数の70%なら停止
- fold数 < 3なら停止

### 5.3 成果物

- `data/oof/oof_health_report.json`
- `data/oof/win_selection_oof_health_report.json`
- `data/validation/oof_generation_manifest.json`

### 5.4 受け入れ基準

- テストがOOF成果物を空で上書きしない
- `run_backtest.py --years 2024 2025 --train-window 4 --ensemble --report` 後にOOFが10万行規模で存在する
- OOF healthがPASSしない限り、selection系モデルを配備しない

## 6. Phase 1: 投資判断用 FeatureFrame の新設

### 6.1 目的

ベースモデルの368から438列をそのまま後段に渡すのではなく、投資判断に必要な特徴量だけを整理した `InvestmentFeatureFrame` を作る。

### 6.2 新設候補

新規モジュール:

- `src/features/investment_features.py`

新規成果物:

- `data/features/investment_features.parquet`
- `data/oof/win_investment_oof.parquet`

### 6.3 最低限含める特徴量

識別:

- `race_id`
- `race_date`
- `surface`
- `umaban`
- `kakuteijyuni`
- `confirmed_odds`
- `tanodds`

モデル確率:

- `p_win_pred`
- `p_win_corrected`
- `p_win_final`
- `p_win_oof`
- `win_selection_prob`

市場確率:

- `p_market_win_raw`
- `p_market_win_norm`
- `logit_p_market`
- `market_rank`
- `market_share`
- `overround_proxy`

モデル対市場:

- `p_model_minus_p_market`
- `logit_p_model_minus_logit_p_market`
- `p_model_div_p_market`
- `rank_model_minus_rank_market`
- `rank_model_minus_popularity`
- `market_value_ratio`
- `market_residual`

レース内相対:

- `model_prob_rank`
- `market_prob_rank`
- `ev_rank`
- `edge_rank`
- `model_prob_gap_to_top`
- `model_prob_gap_to_runner_up`
- `market_prob_gap_to_top`
- `edge_gap_to_top`
- `field_size`
- `entropy_model_prob`
- `entropy_market_prob`
- `top1_market_concentration`
- `top3_market_concentration`

オッズ帯:

- `odds_band`
- `log_odds`
- `odds_rank`
- `is_favorite`
- `is_longshot`
- `favorite_odds`
- `favorite_gap`

late odds:

- `odds_drop_rate_60_10`
- `odds_drop_rate_30_10`
- `odds_velocity`
- `odds_volatility`
- `odds_acceleration`
- `popularity_change_30_10`
- `late_odds_drop_z`
- `late_odds_rank_change`
- `late_market_share_change`

能力/フォーム要約:

- `p_ability_win`
- `rel_p_ability_win_zscore`
- `rel_p_ability_win_rank`
- `form_trend`
- `form_consistency`
- `weighted_recent_form_finish`
- `weighted_recent_form_time`
- `days_since_last_race`
- `class_move`

コース/馬場/ペース要約:

- `surface`
- `distance_bin`
- `track_condition_code`
- `pace_scenario_fit`
- `pace_pressure`
- `pace_aptitude`
- `closing_speed_ratio_avg`
- `harontime_last3f_avg`

不確実性:

- `prob_calibration_bin`
- `odds_calibration_bin`
- `prediction_entropy`
- `model_market_disagreement`
- `conformal_width`
- `ev_uncertainty_proxy`

### 6.4 欠損対応

欠損率が高い列は以下に分ける。

- 常時欠損: 削除候補
- 一部期間だけ欠損: availability flagを追加
- レース条件で自然欠損: 欠損自体を情報として扱う

例:

- `dam_*` は欠損flagを追加
- `dm_time_*` は現状100%欠損なら投資FeatureFrameから除外
- `course_record_time` は利用可能になるまでは除外

### 6.5 受け入れ基準

- investment featureの列数は初期80から150列程度
- 全列の欠損率、nunique、期間別availabilityをレポート化
- 100%欠損列が含まれない
- race_id単位で全馬が揃っている

## 7. Phase 2: MarketAwareWinCalibrator の新設

### 7.1 目的

`p_win_final`を投資判断用の勝率として作り直す。

現在は最終選定がEVと市場残差を補正しているが、根本の確率がズレるとEVが壊れる。ここを市場込みで再構築する。

### 7.2 モデルの考え方

Benter型の考え方に寄せる。

基本形:

```text
logit(p_market_aware)
  = a * logit(p_model)
  + b * logit(p_market)
  + c * model_market_residual_features
  + d * surface/odds/prob_rank calibration features
```

重要なのは、市場を敵として扱うのではなく、強い事前分布として扱うこと。

### 7.3 新設候補

新規:

- `src/models/market_aware_win_calibrator.py`

既存改修:

- `src/pipelines/training_pipeline.py`
- `src/backtest/race_predictor.py`
- `src/db/model_loader.py`
- `src/domain/models.py`

### 7.4 学習データ

入力:

- `win_investment_oof.parquet`
- OOFの `p_win_oof`
- 市場確率
- 投資判断用特徴量

ラベル:

- `is_win = kakuteijyuni == 1`

Group:

- race_id単位
- 時系列fold

### 7.5 モデル候補

初期候補:

- LogisticRegression
- HistGradientBoostingClassifier
- LightGBM binary
- Isotonic regression by segment
- Beta calibration

まずは複雑にしすぎない。LightGBMを使う場合も、出力は確率であり、ROI直接最適化にはしない。

### 7.6 出力

- `p_win_market_aware`
- `p_win_market_aware_raw`
- `p_win_market_aware_calibrated`
- `market_aware_calibration_segment`
- `market_aware_uncertainty`

### 7.7 配備条件

配備はROIではなく、確率品質で決める。

必須:

- OOF Brierが既存 `p_win_final` 以下
- OOF loglossが既存以下
- ECEが既存以下
- odds帯別actual/predの最大乖離が改善
- surface別actual/predが改善
- どの学習年でもBrier/loglossが悪化しない、または悪化が極小

副次:

- CLVが悪化しない
- top1 ROIが大きく悪化しない

### 7.8 受け入れ基準

- `data/validation/market_aware_calibration_report.json` を出力
- 2024/2025両方で平均予測確率と実的中率の比率が改善
- `actual / predicted` がsurface別で0.95から1.05に近づく
- EV過大評価が縮む

## 8. Phase 3: Segment Calibration の作り直し

### 8.1 目的

surface、odds帯、prob rank帯、edge帯ごとの過信を補正する。ただしWSCのように順位を不安定に変える補正は配備条件を厳しくする。

### 8.2 現状の問題

WSCは年によって効果が反転する。

- 2025芝では効いている可能性が高い
- 2024では単体効果は小さい
- OOF上での単純な順位シミュレーションは2025芝の改善を見逃す可能性がある

そのため、WSCを即停止ではなく、役割を明確化する。

### 8.3 方針

WSCは以下のどちらかにする。

Option A:

- `p_win_market_aware` の後段で弱いsegment shrinkageだけ適用
- ただしEVには直接かけない

Option B:

- WSCを単独モデルとして使わず、MarketAwareWinCalibratorの特徴量に統合する

推奨はOption B。

### 8.4 実装

- `segment_actual_pred_ratio`
- `segment_sample_count`
- `segment_win_count`
- `segment_roi`
- `segment_shrinkage_factor`
- `segment_reliability_weight`

これらを特徴量として持たせ、最終確率モデルに判断させる。

### 8.5 配備条件

- segment補正単体で配備しない
- MarketAwareWinCalibratorの確率品質を悪化させる場合は使わない
- 年別actual/predを必ず見る

## 9. Phase 4: Race-Level Ranker の新設

### 9.1 目的

馬単位分類器ではなく、同一レース内でどの馬を買うべきかを学習する。

単純な「一番勝ちそうな馬」ではなく、「市場が過小評価している馬」を選ぶ。

### 9.2 新設候補

- `src/models/win_race_level_ranker.py`

### 9.3 入力特徴量

Phase 1のInvestmentFeatureFrameを使う。

特に重要:

- `p_win_market_aware`
- `p_market_win_norm`
- `logit_p_model_minus_logit_p_market`
- `model_rank_minus_market_rank`
- `odds_band`
- `surface`
- `late_odds_drop_z`
- `market_value_ratio`
- `model_market_disagreement`
- `prob_rank`
- `ev_rank`
- `edge_rank`
- `field_size`
- `top1_gap`
- `runner_up_gap`

### 9.4 ラベル

ROI直接ラベルはノイズが強いので、複数の目的を分ける。

候補:

- `is_win`
- `win_return_unit`
- `calibrated_ev`
- `positive_clv`
- `market_mispricing_score`

初期は以下の2段構成にする。

1. 勝率ranker: `is_win`を目的にする
2. value ranker: `calibrated_ev`, `market residual`, `CLV`を目的にする

最終score:

```text
investment_score =
    calibrated_log_ev
  + value_ranker_score
  + clv_score
  - uncertainty_penalty
```

### 9.5 モデル候補

- LightGBM ranker
- LambdaRank
- pairwise logistic ranker
- CatBoost ranking

最初はLightGBM rankerが現実的。

### 9.6 配備条件

ベット数は減らさない。

配備可否:

- OOF top1 ROIが現行より改善
- OOF top1 hit rateが異常に上がらない
- 年別で現行より大きく悪化しない
- surface別で片方だけ大きく壊れない
- odds帯別のactual/predが悪化しない

### 9.7 受け入れ基準

- `data/validation/win_ranker_report.json`
- 2024/2025ウォークフォワードで現行rankerを上回る
- bet countは同程度
- drawdownが悪化しない

## 10. Phase 5: Race-Level Portfolio への拡張

### 10.1 目的

単勝1レース1頭固定から、利益最大化を目的とした複数頭購入へ拡張する。

ユーザー方針として、必ず1レース1頭にこだわる必要はない。

### 10.2 前提

同一レースの単勝は排他的事象なので、通常の独立Kellyでは不適切。

必要なのは、同一レース内の候補群に対する制約付き配分。

### 10.3 実装候補

新規:

- `src/betting/win_portfolio_allocator.py`

入力:

- `p_win_market_aware`
- `confirmed_odds`
- `calibrated_ev`
- `uncertainty`
- `race_id`

制約:

- 1レースあたり最大投資額
- 1頭あたり最小100円
- 複数頭買いは最大3頭程度から開始
- レース単位の期待損失上限

### 10.4 アロケーション案

初期は3方式を比較。

1. Flat multi-bet
   - EV上位を100円ずつ
   - bet countは増えるが実装が単純

2. Fractional Kelly for mutually exclusive outcomes
   - レース内の排他的結果を考慮
   - 制約付き最適化でstakeを決める

3. Rank-weighted allocation
   - score差に応じて100円、200円、300円などに配分
   - 本番運用で説明しやすい

### 10.5 配備条件

ベット数を減らしてROIを上げるのではなく、同等以上のレースカバー率を維持する。

指標:

- レース参加率
- 総ベット数
- 総投資額
- ROI
- 利益
- 最大DD
- hit race rate
- race-level profit distribution

### 10.6 受け入れ基準

- 単勝1頭固定より合計利益が改善
- ROIだけでなく最大DDが悪化しない
- bet countが極端に減らない
- 2024/2025の片年だけの改善ではない

## 11. Phase 6: Win/Place/Wide 統合セレクタ

### 11.1 目的

単勝だけでROI 100%が難しい場合、同じ予測情報を使って券種を選ぶ。

単勝、複勝、ワイドは同じ馬評価を異なる市場で表現したもの。市場の歪みが単勝にない場合、複勝やワイドに出る可能性がある。

### 11.2 新設候補

- `src/models/bet_type_selector.py`
- `src/betting/multi_market_allocator.py`

### 11.3 入力

単勝:

- `p_win_market_aware`
- `win_ev`
- `win_uncertainty`

複勝:

- `p_place`
- `place_ev`
- `place_uncertainty`

ワイド:

- pair probability
- wide odds
- pair EV
- pair uncertainty

共通:

- CLV予測
- liquidity proxy
- odds volatility
- race uncertainty

### 11.4 出力

- bet type
- target horses
- stake
- reason
- expected value
- uncertainty

### 11.5 配備条件

- 券種を増やしても総リスクが増えすぎない
- 同一レース内で過剰投資しない
- 単勝だけより利益が改善
- 年別で極端に悪化しない

## 12. Phase 7: CLV予測モデル

### 12.1 目的

CLVは「市場より良い価格で買えているか」の指標。現状CLVはプラス傾向なのにROIが100%未満なので、CLVだけでは足りないが、選定補助としては重要。

### 12.2 新設候補

- `src/models/clv_predictor.py`

### 12.3 ラベル

```text
clv = closing_odds / bet_odds - 1
```

または既存の `clv` 列を使用。

### 12.4 入力

- late odds features
- market rank change
- model_market_disagreement
- field size
- surface
- odds band
- popularity band

### 12.5 使い方

CLV予測は購入可否ではなく、rankerの補助に使う。

```text
investment_score += clv_weight * predicted_clv
```

配備条件:

- CLV予測の方向精度がOOFで有意にプラス
- CLVを入れてもactual/pred calibrationが悪化しない

## 13. Phase 8: Feature Importance と削除候補の整理

### 13.1 目的

特徴量を増やす前に、死に特徴量、欠損過多、リーク疑い、重要度不安定な列を整理する。

### 13.2 実装

既存:

- `scripts/analyze_feature_importance.py`
- `scripts/prune_noise_features.py`
- `src/features/win_feature_analysis.py`

追加:

- `scripts/audit_investment_features.py`

出力:

- `data/validation/feature_audit_report.json`
- `data/validation/feature_audit_report.csv`

### 13.3 見る項目

- 欠損率
- nunique
- surface別availability
- 年別availability
- target leakage疑い
- model importance
- permutation importance
- OOF stability

### 13.4 削除候補

初期候補:

- 100%欠損列
- 実質定数列
- 本番推論時に使えない結果列
- レース後情報が混入する疑いのある列

注意:

- 低重要度でも正則化に効いている可能性はあるため、一括削除しない
- 投資FeatureFrameからは除外し、ベースモデルでは段階的に検証

## 14. Phase 9: 検証設計

### 14.1 必須検証

単年度だけでは判断しない。

最低限:

```bash
python scripts/run_backtest.py --years 2024 2025 --train-window 4 --ensemble --report
python scripts/run_wf_validation.py --ensemble
```

### 14.2 指標

ROI系:

- ROI
- 利益
- bet count
- race coverage
- max drawdown
- profit factor

確率系:

- Brier
- logloss
- ECE
- actual/pred
- odds帯別actual/pred
- surface別actual/pred
- prob_rank別actual/pred

市場系:

- CLV
- CLV hit rate
- closing odds ratio
- market rank drift

安定性:

- 年別ROI
- surface別ROI
- odds帯別ROI
- high odds dependence
- feature importance stability

### 14.3 配備判定

モデルや補正を配備する条件:

- 2024/2025合計で改善
- 片年だけで大きく悪化しない
- bet countが極端に減らない
- 確率品質が悪化しない
- 高オッズ少数的中だけで改善していない

禁止:

- bet countを大きく減らしてROIを上げる
- レジーム検出で説明する
- 2024/2025の係数を直接ハードコードする
- OOF health未通過の成果物を使う

## 15. 実装順序

推奨順序:

1. Phase 0: OOF健全性の再保証
2. Phase 1: InvestmentFeatureFrame
3. Phase 2: MarketAwareWinCalibrator
4. Phase 9の確率検証を先に実施
5. Phase 4: Race-Level Ranker
6. Phase 5: Race-Level Portfolio
7. Phase 6: Win/Place/Wide統合
8. Phase 8: Feature auditを継続運用へ組み込む

最初にやらない方がよいこと:

- WSCの係数だけをさらに詰める
- high oddsの閾値を2024/2025に合わせる
- bet countを減らすセレクタを作る
- ROI直接最大化のOptunaだけを回す

## 16. 最初の具体タスク

### Task 1: InvestmentFeatureFrame作成

作るもの:

- `src/features/investment_features.py`
- `tests/test_investment_features.py`
- `data/features/investment_features.parquet`

最低限の関数:

```python
build_win_investment_features(feature_df, prediction_df=None) -> pd.DataFrame
validate_investment_features(df) -> dict
```

### Task 2: MarketAwareWinCalibrator作成

作るもの:

- `src/models/market_aware_win_calibrator.py`
- `tests/test_market_aware_win_calibrator.py`

最低限のAPI:

```python
class MarketAwareWinCalibrator:
    def train(self, oof_df: pd.DataFrame) -> MarketAwareWinCalibrator: ...
    def apply(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> MarketAwareWinCalibrator: ...
```

出力列:

- `p_win_market_aware`
- `p_win_market_aware_raw`
- `market_aware_logit`
- `market_aware_segment`

### Task 3: calibration report作成

作るもの:

- `src/validation/calibration_report.py`
- `tests/test_calibration_report.py`

出力:

- `data/validation/market_aware_calibration_report.json`

### Task 4: RacePredictorへ接続

改修候補:

- `src/domain/models.py`
- `src/pipelines/training_pipeline.py`
- `src/db/model_loader.py`
- `src/backtest/race_predictor.py`

接続方針:

- `p_win_market_aware` が存在する場合、最終EV計算に使う
- なければ従来の `p_win_final` にフォールバック
- diagnosticに両方を出す

### Task 5: backtest比較

最低限:

```bash
python scripts/run_backtest.py --years 2024 2025 --train-window 4 --ensemble --report
```

比較対象:

- 現行main
- MarketAwareWinCalibratorあり
- RaceLevelRankerあり
- Portfolioあり

## 17. 成功の定義

短期:

- OOF健全性が保証される
- 確率キャリブレーションが改善する
- #31水準の91%台を安定して回復する

中期:

- 2024/2025合計ROI 95%以上
- surface別の大崩れを減らす
- EV過大評価を縮小する
- drawdownを改善する

長期:

- 2024/2025合計ROI 100%超え
- 年別片方だけに依存しない
- 単勝だけで難しい場合は複数券種統合で100%超えを狙う
- paper tradingでCLV、ROI、DDを継続監視できる

## 18. 重要な判断

現時点での判断:

- 特徴量の量は足りている
- 投資判断用特徴量は足りていない
- 最終選定の係数調整だけではROI 100%超えは難しい
- 次の本丸は `p_win_final` の再構築
- 市場確率を組み込んだ `MarketAwareWinCalibrator` を優先する
- その後にrace-level ranker、portfolio、multi-marketへ進む

## 19. 参考

- William Benter, "Computer Based Horse Race Handicapping and Wagering Systems"
- Favorite-longshot biasに関する競馬市場研究
- スポーツベッティングにおけるcalibration重視のモデル選択研究

本計画では、これらの一般原則を採用するが、実装上の配備可否は必ず本リポジトリのOOFとウォークフォワード検証で判定する。
