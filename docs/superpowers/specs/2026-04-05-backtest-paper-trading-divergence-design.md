# バックテスト vs ペーパートレード乖離調査・修正設計

**日付**: 2026-04-05
**ステータス**: 承認済み
**対象期間**: 学習 2020-01-01 ~ 2026-03-29、ペーパートレード 2026-04+

## 問題

| 指標 | バックテスト (2024テスト) | ペーパートレード (1日) |
|---|---|---|
| レース数 | ~2,592 | 24 |
| ベット数 | 2,312 | 45 |
| ベット/レース | ~0.89 | ~1.88 |
| ROI | 156.5% | 未計測 |

ペーパートレードのベット頻度がバックテストの約2倍。EV >= 1.50 が頻出。

## 調査結果

### 同一コードパスの確認

`run_paper_trading.py --mode predict` と `BacktestEngine.run()` は同じ `RacePredictor` を使用:

```
BacktestEngine.run()         _run_predict()
    ↓                            ↓
RacePredictor.predict()      RacePredictor.predict()      ← 同じ
RacePredictor.should_bet()   RacePredictor.should_bet()   ← 同じ
RacePredictor.select_bets()  RacePredictor.select_bets()  ← 同じ
```

- `PaperTradingConfig.ev_threshold = 1.0` は使用されていない (red herring)
- 閾値は常に `RegimeDetector.get_strategy_params()` から取得
- `select_bets()` は `ev_place` (生EV) で閾値判定

### 乖離の根本原因 (推定)

EV >= 1.50 が頻出 → モデルが体系的に高いEVを予測 → レジーム閾値に関係なく多くの馬がベット対象に。

原因候補:
1. **データソースの違い**: バックテスト=Parquet、ペーパートレード=EveryDB2直読み
2. **オッズ時系列データの差** (レビューで発見): ペーパートレードは `odds_ts_df` をEveryDB2から取得 (`run_paper_trading.py:239`)、バックテストは `odds_ts_df = None` (`engine.py:108`)。オッズ動的特徴量がペーパートレードのみに存在
3. **学習内データリーク**: TwoStageModel/EVCorrectionModel のランダム train/valid split がモデルの過信を引き起こす
4. **PlaceAbilityModel 温度スケーリング** (レビューで発見): `raw_p ** (1 / 0.7)` (`place_ability_model.py:164`) が確率を増幅しEVを押し上げる
5. **経年変化**: 学習データ (2020-2026/03) とペーパートレード (2026/04+) の市場環境差

### 既存のリーク (確認済み)

| 場所 | 種類 | 深刻度 |
|---|---|---|
| `two_stage_return_model.py:14-30` | ランダム train/valid split | 高 |
| `ev_correction_model.py:92-95, 139-141` | ランダム train/valid split | 高 |
| `stage1_ability_model.py:123-124` | 最終学習のランダム split | 中 |
| `horse_history_features.py:353-394` | global_stats に全期間データ含む | 低 |

## 設計: 3ステップ修正

### Step A: DiagnosticLogger

**新規ファイル**: `src/backtest/diagnostic_logger.py`

1レースごとに収集:

| 項目 | 型 | 説明 |
|---|---|---|
| `race_id` | str | レースID |
| `regime` | str | AGGRESSIVE / CONSERVATIVE / COLLAPSED |
| `ev_threshold` | float | 適用された閾値 |
| `quality_passed` | bool | RaceQualityScreener 判定 |
| `quality_score` | float | スクリーナー生スコア |
| `n_candidates` | int | ev_place >= threshold の馬数 |
| `n_bets` | int | 実際のベット数 |

1馬ごとに収集:

| 項目 | 型 | 説明 |
|---|---|---|
| `race_id` | str | レースID |
| `umaban` | int | 馬番 |
| `p_place_pred` | float | 複勝確率予測 |
| `e_return_place_pred` | float | 的中時払戻予測 |
| `ev_place` | float | 生EV (= p x e) |
| `fukuoddslow` | float | 実際の複勝オッズ |
| `is_bet` | bool | ベット対象か |

統合箇所:
- `BacktestEngine.run()` のレースループ内
- `run_paper_trading.py` の `_run_predict()` レースループ内

出力先:
- バックテスト: `data/backtest/diagnostics_bt.csv`
- ペーパートレード: `data/paper_trading/diagnostics_{YYYYMMDD}.csv`

### Step B: Parquet比較モード

**変更ファイル**: `scripts/run_paper_trading.py`

`--mode diagnose --start YYYYMMDD --end YYYYMMDD` オプションを追加:

1. EveryDB2をバイパスし、ParquetStoreからデータをロード
2. `_run_predict()` と同じ推論パイプラインを実行
3. 診断ログを出力
4. バックテストの診断ログ (`diagnostics_bt.csv`) と比較

期待される結果:
- ベット数が同じ → データソース (EveryDB2 vs Parquet) が乖離原因
- ベット数が違う → パイプライン自体に差がある (追加調査が必要)

**注意**: `odds_ts_df` の差異 (ペーパートレードはEveryDB2から取得、バックテストは `None`) もこのステップで捕捉される。オッズ動的特徴量の有無がEV推定に影響する可能性がある。

### Step C: リーク修正 + 再学習

**修正1: 時系列 train/valid split**

対象ファイル (3ファイル):

- `src/models/two_stage_return_model.py` 行 14-30 (`_train_valid_split`)
  - 現在: `np.random.RandomState(42).permutation(n)` でランダム分割
  - 修正後: `race_date` でソートして前80%/後20%

- `src/models/ev_correction_model.py` 行 92-95, 139-141
  - 現在: 同上のランダム分割
  - 修正後: 同上の時系列分割

- `src/models/stage1_ability_model.py` 行 123-124
  - 現在: 最終学習時のvalid splitがランダム
  - 修正後: 時系列split (OOF生成と同じ方式を適用)

**修正2: global_stats の expanding 化**

- `src/features/horse_history_features.py` 行 353-394
  - 現在: 全期間データから一括で mean/std を計算 (未来のハロンタイムがmean/stdに含まれる)
  - 修正後: `global_stats` をレース日ごとに累積計算。各馬の `race_date` 以前のデータのみから mean/std を算出。個別馬の過去レースフィルタリング (searchsorted, 行414-418) は既にリークフリーなので変更不要

**再学習**:

```bash
python scripts/run_train.py --start 20200101 --end 20260329
```

所要時間: ~7分

## 期待される結果

1. Step A で EV 分布の差異を可視化
2. Step B でデータソースの差を分離
3. Step C でリークを修正し、バックテストROIがより現実的な値に収束
4. 修正後のモデルでペーパートレードのベット頻度がバックテストと整合

## リスク

- **ROI低下**: リーク修正によりバックテストROIが低下する可能性 (160% → 100-120%?)
- **再学習コスト**: ~7分 (許容範囲)
- **Step Bで差が見つからない場合**: 追加の特徴量デバッグが必要
- **RegimeDetector/QualityScreener の再調整**: リーク修正後のEV分布に対応するため再学習パイプラインが自動再調整するが、新しい分布で正しく機能するか検証が必要

## テスト計画

- `_train_valid_split` が時系列分割を行うこと、行が `race_date` でソートされていること、後20%がvalidであることを確認する単体テスト
- `global_stats` が蓄積統計であり、未来日のデータを含まないことを確認する単体テスト
- `DiagnosticLogger` が期待される列を持つCSVを生成することを確認する単体テスト
- Step B の診断モードをmockでエンドツーエンド実行する統合テスト
