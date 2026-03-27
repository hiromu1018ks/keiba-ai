# ETL + 実データ検証 デザイン

## 概要

EveryDB2外部テーブル（`n_race`, `n_uma_race`, `n_harai`, `n_odds_*`, `n_jodds_tanpuku`）からプロジェクトスキーマ（`raw.*`, `odds_history.*`）へのETLを実装し、ノートブック駆動で段階的に実データ検証を行う。

## 背景

- Phase A〜Gの全ソースコードが実装済み（~5,400行、テスト43ファイル、ノートブック12冊）
- EveryDB2の蓄積データ取得が完了（2015年〜2026年）
- 実データでのパイプライン実行は未着手
- EveryDB2外部テーブルとプロジェクトスキーマ間にETL層が存在しない

## データ範囲

| テーブル | 期間 | 行数 |
|----------|------|------|
| `n_race` | 1986〜2026 | 70,733 |
| `n_uma_race` | 1986〜2026 | 865,657 |
| `n_harai` | 2015〜2026 | 38,788 |
| `n_odds_tanpuku` | 2015〜2026 | 543,048 |
| `n_odds_wide` | 2015〜2026 | 3,676,585 |
| `n_jodds_tanpuku` | 2015〜2026 | 83,312,154 |

**オッズ・払戻データは2015年以降のみ**のため、モデル学習期間は2015年〜とする。

## Step 0: ETLモジュール実装

### 新規ファイル: `src/db/etl.py`

EveryDB2外部テーブルからプロジェクトスキーマへのデータ変換・ロードを行う。

#### 関数一覧

| 関数 | 処理 |
|------|------|
| `create_project_schemas(engine)` | `schema.py`のDDLを実行して5スキーマを作成 |
| `etl_races(engine, start, end)` | `n_race` → `raw.races` |
| `etl_entries(engine, start, end)` | `n_uma_race` → `raw.entries` |
| `etl_payouts(engine, start, end)` | `n_harai` → `raw.payouts` |
| `etl_odds_snapshots(engine, start, end)` | `n_odds_tanpuku` → `odds_history.odds_snapshots` |
| `etl_wide_odds(engine, start, end)` | `n_odds_wide` → `odds_history.wide_odds` |
| `etl_odds_timeseries(engine, start, end)` | `n_jodds_tanpuku` → `odds_history.odds_time_series` |
| `run_full_etl(engine, start, end)` | 全ETLを一括実行 |

#### カラムマッピング: n_race → raw.races

| EveryDB2 | raw.races | 変換 |
|----------|-----------|------|
| `year` | `year` | VARCHAR→INT |
| `monthday` | `month_day` | リネーム |
| `jyocd` | `jyo_cd` | リネーム |
| `kaiji` | `kaiji` | そのまま |
| `nichiji` | `nichiji` | そのまま |
| `racenum` | `race_num` | リネーム |
| `trackcd` | `track_cd` | VARCHAR→INT |
| `kyori` | `distance` | VARCHAR→INT |
| `tenkocd` | `tenko_cd` | VARCHAR→INT |
| `sibababacd` / `dirtbabacd` | `baba_cd` | surfaceに応じて選択、VARCHAR→INT |
| `syubetucd` | `syubetu_cd` | リネーム |
| `jyokencd1` | `jyoken_cd` | リネーム |
| `gradecd` | `grade_cd` | NULL→`'_'` |
| `syussotosu` | `field_size` | VARCHAR→INT |

`raw.races` の `race_id`, `surface`, `distance_band` は GENERATED ALWAYS AS で自動生成される。

#### カラムマッピング: n_uma_race → raw.entries

| EveryDB2 | raw.entries | 変換 |
|----------|-------------|------|
| `umaban` | `umaban` | VARCHAR→INT |
| `kettonum` | `ketto_num` | リネーム |
| `kakuteijyuni` | `finish_pos` | VARCHAR→INT（0=出走取消等） |
| `time` | `finish_time` | VARCHAR→FLOAT（空文字→NULL） |
| `odds` | `win_odds` | VARCHAR→FLOAT |
| `ninki` | `ninki` | VARCHAR→INT |
| `bataijyu` | `ba_taijyu` | VARCHAR→FLOAT |
| `zogenfugo` | `zogen_fugo` | VARCHAR→INT |
| `zogensa` | `zogen_sa` | VARCHAR→FLOAT |
| `kisyucode` | `kisyu_code` | リネーム |
| `chokyosicode` | `chokyosi_code` | リネーム |
| `harontimel3` | `haron_time_l3` | リネーム、VARCHAR→FLOAT |
| `honsyokin` | `honsyokin` | VARCHAR→INT |
| `kyakusitukubun` | `kyakusitu` | リネーム、VARCHAR→INT |

FK: `raw.entries.race_id` は `(year || monthday || jyocd || kaiji || nichiji || racenum)` で生成し、`raw.races.race_id` に紐付ける。

#### カラムマッピング: n_harai → raw.payouts

| EveryDB2 | raw.payouts |
|----------|-------------|
| `paytansyoumaban1` | `tan_umaban` |
| `paytansyopay1` | `tan_pay` |
| `payfukusyoumaban{n}` | `fuku_umaban{n}` (n=1..5) |
| `payfukusyopay{n}` | `fuku_pay{n}` (n=1..5) |

FK: `n_harai` の複合キー `(year, monthday, jyocd, kaiji, nichiji, racenum)` でrace_idを生成。

#### 共通: レースキーによる race_id 生成

全EveryDB2テーブルはレース識別の6カラム `year`, `monthday`, `jyocd`, `kaiji`, `nichiji`, `racenum` を持つ。`raw.races` に対応するレースが存在する場合のみロードし、race_idは `year || monthday || jyocd || kaiji || nichiji || racenum` で生成する。

#### カラムマッピング: n_odds_tanpuku → odds_history.odds_snapshots

| EveryDB2 | odds_history.odds_snapshots | 備考 |
|----------|----------------------------|------|
| `year`, `monthday`, `jyocd`, `kaiji`, `nichiji`, `racenum` | `race_id` | 6カラム結合で生成 |
| `umaban` | `umaban` | VARCHAR→INT |
| `tanodds` | `tan_odds` | VARCHAR→FLOAT |
| `fukuoddslow` | `fuku_odds` | VARCHAR→FLOAT |

#### カラムマッピング: n_odds_wide → odds_history.wide_odds

| EveryDB2 | odds_history.wide_odds | 備考 |
|----------|----------------------|------|
| `year`, `monthday`, `jyocd`, `kaiji`, `nichiji`, `racenum` | `race_id` | 6カラム結合で生成 |
| `kumi` | `kumi` | そのまま（例: "3-7"） |
| `oddslow` | `odds_low` | VARCHAR→FLOAT |
| `oddshigh` | `odds_high` | VARCHAR→FLOAT |

#### カラムマッピング: n_jodds_tanpuku → odds_history.odds_time_series

| EveryDB2 | odds_history.odds_time_series | 備考 |
|----------|------------------------------|------|
| `year`, `monthday`, `jyocd`, `kaiji`, `nichiji`, `racenum` | `race_id` | 6カラム結合で生成 |
| `happyotime` | `happyo_time` | そのまま（MMDDHHmm形式） |
| `umaban` | `umaban` | VARCHAR→INT |
| `tanodds` | `tan_odds` | VARCHAR→FLOAT |
| `tanninki` | — | DBには保存せず、ETL関数の出力DataFrameに一時カラム `ninki` として追加。特徴量エンジン(`odds_dynamics_features.py`)が `if "ninki" in ts.columns` で参照する |
| `fukuoddslow` | `fuku_odds` | VARCHAR→FLOAT |

注意: `n_jodds_tanpuku` は8,300万行と非常に大きい。パフォーマンス対策は下記「大規模テーブルのETL戦略」を参照。

#### 大規模テーブルのETL戦略（n_jodds_tanpuku: 83M行）

1. **年次分割ロード**: `WHERE year = :year` で1年ずつ処理（1年あたり約5-7M行）
2. **バッチINSERT**: 50,000行ずつ `executemany()` でバッチ挿入
3. **インデックス制御**: ロード前にPKインデックスをDROP、ロード後に再作成
4. **Copyプロトコル**: pandasの `to_sql()` より `COPY` プロトコル（`psycopg2.extras.execute_values`）を使用して速度向上
5. **推定所要時間**: COPY + バッチで1年あたり数分、全期間で30-60分程度

#### ETLの冪等性（再実行対応）

- `raw.races`, `raw.entries`, `raw.payouts`: `INSERT ... ON CONFLICT DO NOTHING` で重複回避
- `odds_history.*`: 同様に `ON CONFLICT DO NOTHING` で対応（PK制約があるため）
- `run_full_etl()` は何度実行しても同じ結果になる（冪等性保証）

#### FK制約の設計意図

`odds_history.*` テーブルには `raw.races` へのFK制約を設定しない。理由:
- 83M行のテーブルにFK制約を置くとINSERT性能が大幅に低下
- オッズ時系列データは参照整合性より書き込み性能を優先
- アプリケーション層でrace_idの存在チェックを行う

#### baba_cd（馬場状態）の選択ロジック

```python
# track_cdに応じて芝/ダートの馬場状態を選択
CASE
    WHEN track_cd BETWEEN 10 AND 22 THEN sibababacd  -- 芝
    WHEN track_cd BETWEEN 23 AND 29 THEN dirtbabacd  -- ダート
END
```

#### 型変換ルール

- 全カラムが `character varying` のため、空文字→NULL変換が必要
- INT系: `NULLIF(col, '')::INTEGER`
- FLOAT系: `NULLIF(col, '')::FLOAT`
- 例外（異常値）はNULLにして警告ログを出力

## Step 1: DB接続 + スキーマ作成 + ETL実行

### 改修ファイル: `notebooks/00_setup.ipynb`

- DBヘルパー関数を修正: `raw.races` の `year||month_day` を参照する形式に変更
- ETL実行セルを追加: `run_full_etl(engine, "20150101", "20261231")`
- ロード後の検証セル: 行数確認、期間確認、NULL率確認

### 確認項目

- [ ] `raw.races` 行数が妥当（~35K件、2015年以降）
- [ ] `raw.entries` が `raw.races` に正しく紐付いている（orphanなし）
- [ ] `raw.payouts` が `raw.races` に紐付いている
- [ ] NULL率が許容範囲内（win_odds, ninki 等の主要カラム < 1%）

## Step 2: EDA

### 改修ファイル: `notebooks/01_eda.ipynb`

- `load_races()` のクエリを `connection.py` の実装に合わせる
- 実データでの年次推移グラフ
- 人気順位別勝率チャート
- 芝/ダート別統計

## Step 3: 特徴量エンジン検証

### 新規ファイル: `notebooks/01b_feature_engineering.ipynb`

- `FeatureEngine.build_all()` を実データで実行
- 各特徴量モジュールの出力分布を確認
- 欠損値・Infのチェック
- `validate_no_future_leakage()` の実行

### 確認項目

- [ ] 全特徴量がNaN/Infなし
- [ ] 分布が極端に偏っていない（std > 0）
- [ ] `validate_no_future_leakage()` がPASS

## Step 4: モデル学習 + バックテスト

### 改修ファイル: `notebooks/11_holdout_final_evaluation.ipynb`

- コメントアウトされたバックテストコードを実行可能にする
- `TrainingPipelineV5.run()` で学習
- `BacktestEngine` でバックテスト
- 受入基準の検証

### 受入基準（design.mdより）

| 指標 | 基準 |
|------|------|
| 複勝 ROI | >= 100% |
| ワイド ROI | >= 103% |
| 全体 ROI | >= 101% |
| 最大DD | <= 16% |
| 月次ROI | >= 100% が36ヶ月中22ヶ月以上 |

## Step 5: 設計ドキュメントノートブック更新

ノートブック03〜10（設計解説のみ）に、Step 3-4の実データ検証結果を追記する。

## 既存ノートブックの修正

### `notebooks/02_odds_dynamics.ipynb`

- `odds.odds_timeseries` → `odds_history.odds_time_series` に修正

## テスト

- `src/db/etl.py` に対するテストを `tests/test_etl.py` に追加
- モックを使用（DB不要）
