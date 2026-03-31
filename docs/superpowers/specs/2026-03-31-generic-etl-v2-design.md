# Generic ETL v2: EveryDB2 全テーブル Parquet 出力 + 差分更新

> 元設計: `docs/superpowers/specs/2026-03-29-generic-etl-design.md`
> 現在のコードベースの状態に合わせて改訂。

## 背景

EveryDB2 (JRA-VAN DataLab) の PostgreSQL 外部テーブルを全テーブル Parquet ファイルに出力する。
現在の `src/db/etl.py` は9テーブルのみ対応で、各テーブルごとにハードコードされたカラム選択・リネーム・型変換を持つ。
これをYAML設定駆動の汎用ETLエンジンに置き換える。

## 要件

1. PostgreSQL の全テーブル (n_ + s_) をすべて Parquet に出力
2. s_ テーブルを利用した差分更新（upsert/delete）に対応
3. **カラムは生のまま出力**（リネーム・型変換なし）。ただし `race_date` のみ計算列として追加
4. DataRepository に**一時的なリネームレイヤー**を追加し、MLパイプラインの互換性を維持
5. PostgreSQL内部スキーマへの書き込み関数は削除
6. `docs/everydb2/*.md` のテーブル定義と照合してYAML設定を作成

## フェーズ分割

### Phase 1 (今回): ETLエンジンの汎用化

- YAML設定駆動で全テーブルをParquetに出力 (生カラム名)
- `race_date` のみ計算列として追加 (pyarrow predicate pushdown用)
- `s_` テーブルの差分更新 (datakubun による upsert/delete)
- PostgreSQL書き込み系関数を削除
- DataRepositoryに一時リネームレイヤーを追加

### Phase 2 (次回): MLパイプラインの生カラム名移行

- MLパイプライン (feature_engine, horse_history, bloodline等) を生カラム名に修正
- DataRepositoryのリネームレイヤーを削除
- `schema.py` のPostgreSQL内部スキーマ定義を整理

## 制約

- **カラム名はすべて小文字**: `trackcd`, `jyocd`, `datakubun` 等。アンダースコアなし
- **`datakubun` は小文字**: PostgreSQL外部テーブルの実際のカラム名
- **全カラム varchar**: EveryDB2 はすべてのカラムを `character varying` で定義
- **EveryDB2Queriesは影響範囲外**: PaperTrading用の直接SQLクエリラッパーは変更しない

## 設計

### テーブル分類

| タイプ | 説明 | 日付フィルタ | 対象 |
|--------|------|-------------|------|
| `raced` | レースに紐づくデータ | `year + monthday` | race, uma_race, harai, odds系, hyosu系, jodds系 等 |
| `master` | マスターデータ | なし（フルダンプ） | uma, kisyu, chokyo, seisan, banusi等 |
| `delta` | s_ テーブル差分 | なし（前回取得からの差分） | s_race, s_uma_race 等 |

### データフロー (Phase 1後)

```
EveryDB2外部テーブル (PostgreSQL)
    |  SELECT * FROM n_xxx / s_xxx
    v
汎用ETLエンジン (etl.py)
    |  生カラム名 + race_date のみ計算
    v
ParquetStore
    |  data/raw/*.parquet, data/odds/*.parquet
    v
DataRepository (一時リネームレイヤー)
    |  trackcd→track_cd, kyori→distance, etc.
    v
MLパイプライン (変更なし)
```

### etl.py の新構成

削除する関数:
- `etl_races`, `etl_entries`, `etl_payouts`, `etl_odds_snapshots`, `etl_wide_odds`, `etl_odds_timeseries`
- `run_full_etl`
- `_insert_on_conflict`, `_to_int`, `_to_float`, `_to_odds`, `_make_race_id`, `_select_baba_cd`
- `create_project_schemas`
- `_etl_horses_to_parquet`, `_etl_jockey_stats_to_parquet`, `_etl_trainer_stats_to_parquet`
- `run_full_etl_to_parquet`

新規/残す関数:
```python
def load_table_config(path: str = ...) -> list[dict]: ...
def _read_db_table(engine: Engine, cfg: dict, start: str | None = None, end: str | None = None) -> pd.DataFrame: ...
def _add_race_date(df: pd.DataFrame) -> pd.DataFrame: ...
def run_full_load(store: ParquetStore, engine: Engine, config: list[dict], start: str, end: str, tables: list[str] | None = None) -> dict[str, int]: ...
def _merge_delta(existing: pd.DataFrame, delta: pd.DataFrame, pk: list[str]) -> pd.DataFrame: ...
def run_delta_update(store: ParquetStore, engine: Engine, config: list[dict]) -> dict[str, int]: ...
def _load_state() -> dict: ...
def _save_state(state: dict) -> None: ...
```

### DataRepository の一時リネームレイヤー

コアテーブル読み込み時に生カラム名を既存名にマッピング:
- `trackcd` → `track_cd`
- `kyori` → `distance`
- `monthday` → `month_day`
- `jyocd` → `jyo_cd`
- `racenum` → `race_num`
- `tenkocd` → `tenko_cd`
- `kakuteijyuni` → `finish_pos`
- `time` → `finish_time`
- `odds` → `win_odds` (÷10変換もここで)
- `kettonum` → `ketto_num`
- `harontimel3` → `haron_time_l3`
- `timediff` → `time_diff`
- `jyuni1c` → `corner_1c`
- `jyuni4c` → `corner_4c`
- `kisyucode` → `kisyu_code`
- `chokyosicode` → `chokyosi_code`
- `zogenfugo` → `zogen_fugo`
- `zogensa` → `zogen_sa`
- `bataijyu` → `ba_taijyu`
- `kyakusitukubun` → `kyakusitu`
- `ninki` → `ninki` (変更なし)

`race_id` は Repository 読み込み時に計算 (現在と同じロジック)。

### connection.py の変更

- `_compute_race_id()` を `etl.py` に移動
- `_compute_race_date()` を `etl.py` に移動
- `etl_to_parquet()` を新ETLの `run_full_load()` に委譲

### 設定ファイル (config/etl_tables.yaml)

`docs/everydb2/*.md` のテーブル定義と照合して作成。
各テーブルのフィールド:
- `db_table`: PostgreSQL テーブル名
- `parquet_key`: 出力先ファイル名
- `category`: ディレクトリ分類 (raw/odds)
- `type`: raced/master/delta
- `pk`: 主キーカラムリスト (差分マージ用)
- `partition_cols`: パーティション列 (任意)

### 差分マージロジック

1. 既存 Parquet を読み込み
2. `s_` テーブルを読み込み
3. `datakubun == '0'` → delete, それ以外 → upsert
4. PK ベースで既存データとマージ
5. `datakubun` 列は出力から除外
6. `race_date` を再計算 (type=raced の場合)
7. ParquetStore のアトミック書き込みで安全に出力

### CLI インターフェース

```bash
# フルロード
python scripts/run_etl.py --mode full --start 20140101 --end 20231231

# 差分更新
python scripts/run_etl.py --mode delta

# 特定テーブルのみ
python scripts/run_etl.py --mode full --tables races entries --start 20140101 --end 20231231
```

## 影響範囲

| ファイル | アクション | 内容 |
|----------|-----------|------|
| `src/db/etl.py` | リプレイス | 汎用ETLエンジン (~300行)。PostgreSQL書き込み系は全削除 |
| `config/etl_tables.yaml` | 新規 | 全テーブル定義 |
| `scripts/run_etl.py` | 更新 | `--mode full\|delta` 対応 |
| `src/db/connection.py` | 更新 | `_compute_race_id/date` を `etl.py` に移動、`etl_to_parquet` を新ETLに委譲 |
| `src/db/repository.py` | 更新 | 一時リネームレイヤー追加 |
| `src/db/schema.py` | そのまま | Phase 2で整理 |
| `src/db/parquet_store.py` | そのまま | 変更なし |
| `src/db/everydb2_queries.py` | そのまま | 影響範囲外 |
| `tests/test_etl.py` | リプレイス | 汎用ETLのテスト。PostgreSQL書き込み系テストは削除 |
| `tests/test_etl_new_tables.py` | 削除 | 不要 |
| `tests/test_db.py` | 更新 | `_compute_race_id/date` のインポート先変更 |
| `tests/test_repository.py` | 更新 | リネームレイヤーのテスト追加 |
| `data/etl_state.json` | 自動生成 | 差分管理状態 |

## テスト戦略

### 新規作成 (tests/test_etl.py)
- `TestLoadTableConfig` — YAML読み込み、ファイル不在エラー
- `TestReadDbTable` — raced typeの日付フィルタ、master typeのSELECT * のみ
- `TestAddRaceDate` — year+monthday → race_date 計算
- `TestMergeDelta` — upsert/delete/複合PK/datakubun除外
- `TestRunFullLoad` — raced+master処理、delta skip、table filter
- `TestRunDeltaUpdate` — 既存なしskip、マージ動作

### 削除
- `tests/test_etl_new_tables.py` 全ファイル
- `tests/test_etl.py` 内のPostgreSQL書き込み系テスト

### 更新
- `tests/test_db.py` — インポート先変更
- `tests/test_repository.py` — リネームレイヤーテスト

### 不变
- MLパイプラインのテストは一切変更しない
