# 競馬AI ROI不振分析レポート

作成日: 2026-05-21  
作成目的: 現在の競馬AIバックテストでROIが100%を超えない原因を、保存済みの実データ・予測成果物・診断ログから特定し、改善案と次回保存すべき診断情報を整理する。

## 1. 結論

ROI不振は、ROI集計ミスではない。`data/backtest/multi_year_bet_history.json` から再計算したROIは、`multi_year_result.json` および `validation_report.json` の値と完全一致した。

最も確度の高い主因は、EVが高オッズ馬を過大評価していること。EVは `odds` と強く相関する一方、実払戻とは相関していない。

- 全体ROI: 89.08%
- `ev >= 3.0`: 75件、ROI 19.73%、的中1件
- `ev >= 5.0`: 50件、ROI 0.00%、的中0件
- EVと `odds` の相関: 0.80
- EVと `result` の相関: -0.04

ただし、`bt_2024_horse_diagnostics.csv` と `bt_2024_horse_features.parquet` が古い成果物で、最新の `multi_year_bet_history.json` / `predictions/2024.parquet` と時点がずれている。このため「なぜその馬を買ったか」の完全な因果追跡は不足している。

## 2. ファイル・スキーマ確認結果

対象ファイルは大半が存在したが、`data/models-backtest/2024/*.pkl` は存在しなかった。実際には `*.joblib` と `*.lgb` が保存されていた。

| ファイル | 状態 | 行 x 列 | 主な用途 |
|---|---:|---:|---|
| `data/features/horse_features.parquet` | 存在 | 138,153 x 518 | 学習特徴量、予測列、結果列確認 |
| `data/oof/oof_predictions.parquet` | 存在 | 138,153 x 518 | OOF評価候補。ただし `horse_features` と完全一致 |
| `data/models-backtest/2024/meta.json` | 存在 | JSON | 学習期間、surface、quality threshold確認 |
| `data/models-backtest/2024/*.json` | 存在 | 7件 | CQR/EV補正/メタ情報確認 |
| `data/models-backtest/2024/*.pkl` | 不存在 | 0件 | 解析不可 |
| `data/backtest/bt_2024_race_diagnostics.csv` | 存在 | 3,327 x 11 | レース単位の候補数・ベット数・regime |
| `data/backtest/bt_2024_horse_diagnostics.csv` | 存在 | 45,827 x 23 | 馬単位のplace系診断。ただし古い |
| `data/backtest/bt_2024_horse_features.parquet` | 存在 | 45,827 x 494 | 2024 BT特徴量・win/place予測。ただし古い |
| `data/backtest/predictions/2024.parquet` | 存在 | 45,827 x 39 | 実ベット行の `stake/result/ev` |
| `data/backtest/multi_year_result.json` | 存在 | JSON | レポートROI |
| `data/backtest/multi_year_bet_history.json` | 存在 | 2,139 x 28 | ROI再計算の主データ |
| `data/validation/validation_report.json` | 存在 | JSON | ROI・原因分析サマリ |
| `data/validation/multi_year_validation_report.json` | 存在 | JSON | multi-year validation。ただし2024単年 |

推定した主要列:

- ベット行: `stake.notna()` または `bet_type == "win"`
- 払戻: `result`
- 投資額: `stake`
- 単勝target: `kakuteijyuni == 1`
- 複勝圏target: `kakuteijyuni <= 3`
- EV: `multi_year_bet_history.ev` / `predictions.ev`
- オッズ: `odds`, `tanodds`, `confirmed_odds`, `fuku_odds_low`, `fukuoddslow`

## 3. ROI再計算結果

`multi_year_bet_history.json` 基準:

| 指標 | 値 |
|---|---:|
| bets | 2,139 |
| stake | 213,900 |
| return | 190,550 |
| profit | -23,350 |
| ROI | 89.08% |
| hit rate | 11.92% |
| 的中数 | 255 |

レポートとの差分:

| 比較項目 | 差分 |
|---|---:|
| stake | 0 |
| return | 0 |
| ROI | 0 |
| bets | 0 |

重複・混入確認:

- `multi_year_bet_history`: `race_id + umaban` 重複0件
- distinct race数: 2,139
- レース内複数購入は確認されず、各レース1点買い
- `predictions/2024.parquet` では `is_bet=True` と `stake.notna()` が一致しない
  - `stake.notna()`: 2,139行
  - `is_bet=True`: 2,224行
  - `stake.notna() & is_bet=False`: 1,848行
  - `is_bet=True & stake.isna()`: 1,933行

このため、実ベット判定には `is_bet` ではなく `stake/result` を使うべき。

## 4. 予測・EV・ベット選定の診断

### EV診断

EVは高いほどROIが上がる構造になっていない。

| 条件 | bets | ROI | hit rate |
|---|---:|---:|---:|
| 全体 | 2,139 | 89.08% | 11.92% |
| `ev >= 1.05` | 1,585 | 92.12% | 10.22% |
| `ev >= 1.10` | 1,224 | 94.13% | 9.07% |
| `ev >= 1.20` | 781 | 89.32% | 6.15% |
| `ev >= 1.30` | 548 | 82.23% | 4.20% |
| `ev >= 1.50` | 330 | 53.48% | 2.42% |
| `ev >= 2.00` | 127 | 36.22% | 1.57% |
| `ev >= 3.00` | 75 | 19.73% | 1.33% |
| `ev >= 5.00` | 50 | 0.00% | 0.00% |

EV上位の実体は超高オッズ馬への過大評価だった。EV上位20件は概ね単勝100倍超から400倍台の馬で、全て不的中だった。

### 予測確率診断

2024バックテスト特徴量上では、単勝確率モデル自体は完全には壊れていない。

| 予測列 | AUC | pred mean | actual rate |
|---|---:|---:|---:|
| `p_win_pred` | 0.835 | 0.06799 | 0.07273 |
| `p_win_corrected` | 0.834 | 0.07264 | 0.07273 |
| `p_win_final` | 0.835 | 0.07264 | 0.07273 |
| `p_ability_win` | 0.774 | 0.07264 | 0.07273 |

ただし、`p_win_final` 1位を全レース買う単純戦略はROI 75.49%。予測順位は的中率には効いているが、市場控除を超える単勝エッジにはなっていない。

### OOF診断

`data/oof/oof_predictions.parquet` は `data/features/horse_features.parquet` と行数・列数・内容が完全一致していた。真のOOF成果物とは判断できないため、OOF上の高いROIやAUCを根拠にするのは危険。

## 5. 損失源セグメント

損失が大きい条件:

| 条件 | bets | ROI | 備考 |
|---|---:|---:|---|
| odds 100+ | 45 | 0.00% | 全敗 |
| odds 50-100 | 80 | 75.63% | 低的中 |
| 人気 9-12 | 171 | 34.62% | 的中3件 |
| 距離 2001-2400 | 111 | 64.59% | 低ROI |
| 距離 2401+ | 45 | 47.56% | サンプル少 |
| Aggressive regime | 640 | 83.41% | Conservativeより悪い |

相対的に良い条件:

| 条件 | bets | ROI | 備考 |
|---|---:|---:|---|
| 距離 1601-2000 | 927 | 99.76% | ほぼ損益分岐 |
| odds 20-50 | 350 | 98.97% | 高オッズ内では良い |
| 人気 6-8 | 583 | 101.75% | プラス |
| 2024-02 | 189 | 155.77% | 月別。単月依存なので過信不可 |

月別では2月のみ大きくプラスで、7月・9月・10月が大きく悪化していた。ただし月別は季節性と偶然が混ざるため、改善ルールとして使うには追加年の検証が必要。

## 6. リーク・分布シフト疑い

### 成果物時点のズレ

以下の時点差があり、診断の整合性に問題がある。

- `bt_2024_horse_diagnostics.csv`: 2026-05-08
- `bt_2024_horse_features.parquet`: 2026-05-08
- `predictions/2024.parquet`: 2026-05-21
- `multi_year_bet_history.json`: 2026-05-21

このため、`bt_2024_horse_*` で見える候補選定状態と、最新の実ベット履歴が完全に対応していない。

### 欠損・分布シフト

trainと2024 BTの共通列は433列。train-onlyは89列、BT-onlyは65列。

主な欠損差:

- `deviation_rank`: train欠損0%、BT欠損100%
- `deviation_zscore`: train欠損0%、BT欠損100%
- `datakubun`: train欠損15.3%、BT欠損48.0%
- `jockey_surprise`: train欠損23.5%、BT欠損3.0%
- `harontime_late_trend`: train欠損64.3%、BT欠損43.9%

主な数値分布差:

- `sire_distance_wr`: train平均0.087、BT平均7.16。スケール異常疑いが強い。
- `jt_combo_starts`: train平均23.9、BT平均46.4
- `overround`: train平均0.2505、BT平均0.2641
- `pace_pressure`: train平均0.261、BT平均0.321

### リーク・時点安全性疑い

モデル特徴量に以下の列が含まれていた。即リークとは断定しないが、時点安全性の監査対象。

- `actual_pace_fit`
- `signed_log_error_win`
- `abs_log_error_win`
- `market_log_error_*`
- `hist_hit_rate_topk`
- `hist_positive_return_ratio`
- `tanodds`, `fukuoddslow`, `tanninki`

特に `actual_pace_fit` は名前上、結果後情報を含む可能性があるため、生成ロジックを確認すべき。

## 7. 改善案

| 優先度 | 改善案 | 根拠 | 期待効果 | 実装難易度 | 過剰最適化リスク | 検証方法 |
|---|---|---|---|---|---|---|
| 高 | 最新実行で馬別診断を必ず再生成し、`is_bet` と `stake` を一致させる | 現在の診断が古く、`is_bet` と実ベットが矛盾 | 原因分析の信頼性回復 | 中 | 低 | 同一timestamp、`stake.notna == is_actual_bet` 検査 |
| 高 | 高EV・高オッズ尾部を一旦禁止または強く縮小 | `ev>=5` が50件で的中0、ROI 0% | 即時ドローダウン低下 | 低 | 中 | odds>=50/100除外のwalk-forward検証 |
| 高 | EVをオッズ帯・人気帯別に再キャリブレーション | EVが結果ではなくオッズに強く相関 | EV閾値の意味を回復 | 中 | 中 | EV decile ROIの単調性確認 |
| 中 | `p_win_final` rank/probability gateを追加 | 確率AUCはあるがEV単独が暴走 | 長穴過大選定を抑制 | 低 | 中 | rank<=5/10、prob帯別ROIをWFで検証 |
| 中 | 真のOOFファイルを作り直す | OOFが特徴量ファイルと完全一致 | 過学習検出が可能 | 中 | 低 | `fold_id`, `is_oof` 付きでOOF AUC/ROI確認 |
| 中 | `deviation_*`, `sire_distance_wr`, `actual_pace_fit` を監査 | 欠損・スケール・時点安全性に疑い | 分布シフトとリーク低減 | 中 | 低 | feature ablation、train/BT分布テスト |
| 低 | Aggressive regime と長距離を縮小 | Aggressive ROI 83.4%、長距離低ROI | 小幅改善 | 低 | 高 | 事前固定ルールで複数年WF |

## 8. 次に保存すべき診断カラム

次回バックテストでは、各ベット行に以下を保存する。

- `is_actual_bet`
- `selection_reason`
- `excluded_reason`
- `filter_pass_flags`
- `candidate_count_before_filter`
- `candidate_count_after_filter`
- `selected_rank_by_p_win_final`
- `selected_rank_by_win_selection_ev`
- `p_win_pred`
- `p_win_corrected`
- `p_win_final`
- `e_return_win_pred`
- `e_return_win_corrected`
- `win_selection_ev`
- `win_selection_edge`
- `win_gate_score`
- `win_gate_pass`
- `pre_odds`
- `final_odds`
- `odds_source`
- `payout_return`
- `regime`
- `quality_score`
- `quality_passed`
- `max_bets_per_race`
- `stake`
- `result`

OOF成果物には以下を保存する。

- `fold_id`
- `train_start`
- `train_end`
- `valid_start`
- `valid_end`
- `is_oof`
- `model_version`
- `feature_manifest_hash`

## 9. 実務上の次アクション

1. `predictions/2024.parquet` と同じ実行時点で `bt_2024_horse_diagnostics.csv` / `bt_2024_horse_features.parquet` を再生成する。
2. `is_bet` を廃止または `is_actual_bet` に統一し、`stake.notna()` と必ず一致させる。
3. 暫定防御策として、`odds >= 100` または `ev >= 5` の単勝購入を停止してWF検証する。
4. EV補正をオッズ帯・人気帯別に再学習し、EV decileごとのROI単調性を検証する。
5. 真のOOF成果物を作り直し、現状の `oof_predictions.parquet` をOOF評価の根拠にしない。
