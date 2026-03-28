# Parquet Migration Design

**Date:** 2026-03-28
**Status:** Approved (Rev 3 — レビュー2巡目指摘対応済み)
**Scope:** 全データ層のPostgreSQL → Parquet移行

## 1. 目的

MLパイプライン（学習・バックテスト・推論）のデータアクセスをPostgreSQLからParquetに移行し、I/Oを高速化する。

## 2. 決定事項

| 項目 | 決定 |
|---|---|
| 移行範囲 | 全データ層（ETL + 特徴量 + 予測 + 馬券） |
| PostgreSQLの役割 | EveryDB2外部テーブルからのETL入力のみ |
| DataFrameライブラリ | pandas（`pd.read_parquet()`） |
| ファイル構成 | テーブル単位。ただし `time_series` は年/月パーティション |
| ETL更新方法 | races/entries等は全上書き、time_seriesは増分更新 |
| アプローチ | B（役割ごとにクラスを分ける） |
| race_id生成 | Python/pandasで計算（PostgreSQLのGENERATED COLUMNに依存しない） |
| 日付フォーマット | API境界は `"YYYYMMDD"` 文字列。内部は `datetime64` に統一 |
| 将来拡張 | DataRepositoryを唯一のアクセス層とし、将来DuckDB/Polarsへの移行を妨げない |

## 3. アーキテクチャ

### 変更前

```
EveryDB2外部テーブル → PostgreSQL (raw.*, odds_history.*) → pd.read_sql() → MLパイプライン
```

### 変更後

```
EveryDB2外部テーブル → PostgreSQL (ETL用) → Parquetファイル群
                                              ↓
                         ParquetStore → DataRepository → MLパイプライン
```

## 4. ファイル構成

```
data/                          # .gitignore に追加
├── raw/
│   ├── races.parquet          # レース情報（race_date列を含む）
│   ├── entries.parquet        # 出走馬情報（race_date列を含む）
│   └── payouts.parquet        # 払戻情報（race_date列を含む）
├── odds/
│   ├── snapshots.parquet      # 最終オッズ（race_date列を含む）
│   ├── time_series/           # ★ パーティション（年/月）
│   │   ├── year=2020/month=01/
│   │   │   └── part-0.parquet
│   │   ├── year=2020/month=02/
│   │   │   └── part-0.parquet
│   │   └── ...
│   └── wide.parquet           # ワイドオッズ（race_date列を含む）
├── features/
│   └── horse_features.parquet # 馬の過去成績特徴量（キャッシュ）
├── predictions/
│   └── predictions.parquet    # 予測結果
└── bets/
    └── bets.parquet           # 馬券記録
```

### パーティション戦略

| テーブル | 粒度 | 理由 |
|---|---|---|
| races, entries, payouts | 1ファイル | 数十万〜数百万行。単一ファイルで十分 |
| odds/snapshots, wide | 1ファイル | 同上 |
| **odds/time_series** | **年/月パーティション** | 83M行。月単位でpyarrowがファイル単位スキップ可能に |
| features, predictions, bets | 1ファイル | 書き込み頻度低く、データ量も小さい |

全テーブルに `race_date` (datetime64) 列をETL時に永続化する。

## 5. クラス設計

### 5.1 ParquetStore（新規: `src/db/parquet_store.py`）

Parquetファイルの読み書きのみを担当。データの意味は知らない。

```python
class ParquetStore:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    def read(self, category: str, name: str, filters: list | None = None) -> pd.DataFrame:
        """例: store.read("raw", "races")
        filters: pyarrow述語プッシュダウン用 [(column, op, value), ...]
        例: [("race_date", ">=", "2020-01-01")]
        パーティションテーブルの場合は pyarrow.dataset で読み取り。
        """
        path = self.data_dir / category / name
        if path.is_dir():
            # パーティションテーブル → pyarrow.dataset
            import pyarrow.dataset as ds
            dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
            return dataset.to_table(filter=...).to_pandas()
        return pd.read_parquet(path.with_suffix(".parquet"), filters=filters)

    def write(self, category: str, name: str, df: pd.DataFrame, partition_cols: list[str] | None = None) -> None:
        """DataFrameをParquetにアトミック書き込み（temp → rename）
        partition_cols: 指定時はパーティション書き込み
        """
        path = self.data_dir / category / name
        path.parent.mkdir(parents=True, exist_ok=True)

        if partition_cols:
            # パーティション書き込み
            import pyarrow as pa
            table = pa.Table.from_pandas(df)
            pq.write_to_dataset(table, root_path=str(path), partition_cols=partition_cols)
        else:
            # 単一ファイル（アトミック）
            file_path = path.with_suffix(".parquet")
            tmp = file_path.with_suffix(".parquet.tmp")
            df.to_parquet(tmp, index=False)
            tmp.replace(file_path)

    def exists(self, category: str, name: str) -> bool:
        """ファイル or ディレクトリが存在するか"""
        path = self.data_dir / category / name
        return path.with_suffix(".parquet").exists() or path.is_dir()
```

**ポイント:**
- ディレクトリ（パーティション）か単一ファイルかを自動判定
- `filters` パラメータで pyarrow 述語プッシュダウン → 大きなファイルでも必要行だけ読める
- 単一ファイルの `write()` はテンポラリファイル → リネームで原子性担保

### 5.2 DatabaseConnection（変更: `src/db/connection.py`）

- **ETL専用**になる
- 既存の `load_*()` メソッドは削除
- 新規に `etl_to_parquet(store: ParquetStore, start: str, end: str)` メソッドを追加
  - `start`, `end` は `"YYYYMMDD"` 文字列
- PostgreSQL接続設定はそのまま保持

#### ETL内の race_id 生成

現在のPostgreSQL ETLは `raw.races` にJOINして `race_id` を取得している。
Parquet移行後は **pandasで `race_id` を計算** する:

```python
def _compute_race_id(df: pd.DataFrame) -> pd.DataFrame:
    """year + month_day + jyo_cd + kaiji + nichiji + race_num → race_id
    フォーマット契約: YYYY MMDD JJ KK NN RR (16桁)
    """
    df["race_id"] = (
        df["year"].astype(str).str.zfill(4)
        + df["month_day"].astype(str).str.zfill(4)
        + df["jyo_cd"].astype(str).str.zfill(2)
        + df["kaiji"].astype(str).str.zfill(2)
        + df["nichiji"].astype(str).str.zfill(2)
        + df["race_num"].astype(str).str.zfill(2)
    )
    return df

def _compute_race_date(df: pd.DataFrame) -> pd.DataFrame:
    """year + month_day → race_date (datetime64)"""
    date_int = df["year"] * 10000 + df["month_day"]
    df["race_date"] = pd.to_datetime(date_int.astype(str), format="%Y%m%d")
    return df
```

これにより、PostgreSQL内部スキーマへの書き込みは完全に不要になる。

#### ETL順序

1. `etl_races()` — EveryDB2 `n_race` → DataFrame → `_compute_race_id()` + `_compute_race_date()` → `ParquetStore.write("raw", "races")`
2. `etl_entries()` — EveryDB2 `n_uma_race` → DataFrame → `races.parquet` をメモリでJOIN → `ParquetStore.write("raw", "entries")`
3. 以降のETLも同様に、`races.parquet` をメモリでJOIN
4. `etl_odds_timeseries()` — EveryDB2 `n_jodds_tanpuku` → DataFrame → `ParquetStore.write("odds", "time_series", df, partition_cols=["year", "month"])`

**重要:** PostgreSQL内部スキーマ (raw.*, odds_history.*) への書き込みは完全に廃止。
ETLは EveryDB2 → DataFrame → Parquet のみ。

### 5.3 DataRepository（新規: `src/db/repository.py`）

MLパイプラインの唯一のデータアクセス窓口。
**将来DuckDB/Polarsへの移行を妨げないよう、この層が唯一のアクセス経路。**

**日付パラメータ:** `start`, `end` は `"YYYYMMDD"` 文字列（I/O境界）。
内部は `datetime` に変換し、pyarrow filters でプッシュダウン。
障害除外は専用メソッドで明示的に処理。

```python
from datetime import datetime

def _to_dt(yyyymmdd: str) -> datetime:
    return datetime.strptime(yyyymmdd, "%Y%m%d")

def _date_filters(start: str, end: str) -> list:
    """pyarrow述語プッシュダウン用フィルタを生成"""
    s, e = _to_dt(start), _to_dt(end)
    return [("race_date", ">=", s), ("race_date", "<=", e)]

def _exclude_steeple(df: pd.DataFrame) -> pd.DataFrame:
    """障害レース除外（track_cd 51-59）"""
    return df[~df["track_cd"].between(51, 59)].copy()


class DataRepository:
    def __init__(self, store: ParquetStore):
        self.store = store

    # --- 読み取り（pyarrow filtersでプッシュダウン） ---

    def load_races(self, start: str, end: str) -> pd.DataFrame:
        df = self.store.read("raw", "races", filters=_date_filters(start, end))
        return _exclude_steeple(df)

    def load_entries(self, start: str, end: str) -> pd.DataFrame:
        df = self.store.read("raw", "entries", filters=_date_filters(start, end))
        return _exclude_steeple(df)

    def load_odds_snapshots(self, start: str, end: str) -> pd.DataFrame:
        return self.store.read("odds", "snapshots", filters=_date_filters(start, end))

    def load_odds_time_series_range(self, start: str, end: str) -> pd.DataFrame:
        """オッズ時系列（日付範囲）— パーティションテーブル"""
        return self.store.read("odds", "time_series", filters=_date_filters(start, end))

    def load_odds_time_series(self, race_id: str) -> pd.DataFrame:
        """オッズ時系列（単一レース）"""
        return self.store.read("odds", "time_series",
            filters=[("race_id", "==", race_id)])

    def load_wide_odds(self, start: str, end: str) -> pd.DataFrame:
        return self.store.read("odds", "wide", filters=_date_filters(start, end))

    def load_payouts(self, start: str, end: str) -> pd.DataFrame:
        return self.store.read("raw", "payouts", filters=_date_filters(start, end))

    # --- 全履歴参照（HorseHistoryFeatures用） ---

    def load_history_entries(self, lookback_years: int = 5) -> pd.DataFrame:
        """過去N年のentriesをロード。HorseHistoryFeatures等の全履歴参照用。
        lookback_yearsでメモリ使用量を制御。
        """
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=lookback_years * 365)
        return self.store.read("raw", "entries",
            filters=[("race_date", ">=", cutoff)])

    def load_history_races(self, lookback_years: int = 5) -> pd.DataFrame:
        """過去N年のracesをロード"""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=lookback_years * 365)
        return self.store.read("raw", "races",
            filters=[("race_date", ">=", cutoff)])

    # --- 特徴量キャッシュ ---

    def load_features(self, start: str, end: str) -> pd.DataFrame | None:
        """特徴量キャッシュがあれば返す、なければNone"""
        if self.store.exists("features", "horse_features"):
            return self.store.read("features", "horse_features",
                filters=_date_filters(start, end))
        return None

    def save_features(self, df: pd.DataFrame) -> None:
        self.store.write("features", "horse_features", df)

    # --- 予測・馬券 ---

    def save_predictions(self, df: pd.DataFrame) -> None:
        self.store.write("predictions", "predictions", df)

    def save_bets(self, df: pd.DataFrame) -> None:
        self.store.write("bets", "bets", df)
```

### 5.4 etl.py → DatabaseConnection.etl_to_parquet() に統合

`src/db/etl.py` は削除し、そのロジックを `DatabaseConnection.etl_to_parquet()` に統合する。

ETLの流れ:
```
PostgreSQL (EveryDB2外部テーブル)
  → SQLで読み取り → DataFrame
  → _compute_race_id() + _compute_race_date() でrace_id/race_date付与
  → races: ParquetStore.write("raw", "races")
  → entries以降: races.parquet をメモリでJOIN → ParquetStore.write()
  → time_series: ParquetStore.write("odds", "time_series", df, partition_cols=["year", "month"])
```

### 5.5 HorseHistoryFeatures（変更: `src/features/horse_history_features.py`）

- 現在: 直接SQLで `raw.entries JOIN raw.races` をクエリ（馬ごと）
- 変更後: `DataRepository.load_history_entries(lookback_years=5)` + `load_history_races()` で過去データをロード
- `lookback_years` でメモリ使用量を制御（デフォルト5年）
- ロードしたデータはキャッシュして再利用（同じセッション内のバックテスト等で何度も呼ばれるため）

## 6. 影響を受けるファイル

| ファイル | 変更内容 |
|---|---|
| `src/db/connection.py` | `load_*()` 削除、`etl_to_parquet()` 追加、`_compute_race_id()` / `_compute_race_date()` 追加 |
| `src/db/etl.py` | `DatabaseConnection` に統合して **削除** |
| `src/db/parquet_store.py` | **新規** |
| `src/db/repository.py` | **新規** |
| `src/features/horse_history_features.py` | SQL直接アクセス → DataRepository 経由（lookback_years制御付き） |
| `src/features/feature_engine.py` | DataRepository からデータを受け取る |
| `src/pipelines/training_pipeline.py` | DatabaseConnection → DataRepository |
| `src/backtest/engine.py` | 同上。レースごとの再クエリ → メモリフィルタ |
| `src/ingestion/jvlink_fetcher.py` | DatabaseConnection.load_*() → DataRepository |
| `src/ingestion/odds_collector.py` | DatabaseConnection.save_predictions() → DataRepository |
| `src/backtest/validation_suite.py` | DatabaseConnection → DataRepository |
| `CLAUDE.md` | アーキテクチャ説明をParquetベースに更新 |
| `.gitignore` | `data/` ディレクトリを追加 |

### 影響を受けないファイル

- `src/domain/` — 変更なし
- `src/db/schema.py` — PostgreSQL DDLはEveryDB2外部テーブル参照用に残る
- `src/models/` — データソースを意識しない
- `config/settings.yaml` — `data_dir` パス設定を追加

## 7. バックテスト高速化

現在のバックテストエンジンはレースごとにDB再クエリしている。Parquet化により:

- データを `DataRepository` 経由でロード（pyarrow filtersで必要範囲のみ）
- レースごとは `df[df["race_id"] == target]` でフィルタ
- HorseHistoryFeatures は `load_history_entries(lookback_years=5)` で制御付きロード
- ネットワークI/Oがゼロになり、大幅な高速化が期待できる
- time_series（83M行）は 年/月パーティション + pyarrow で該当月のみ読み取り

## 8. CLAUDE.md更新

ArchitectureセクションをParquetベースの構成に更新。将来のセッションでLLMが一目でアーキテクチャを理解できるようにする。

## 9. 将来拡張

DataRepositoryを唯一のデータアクセス層とすることで、将来以下への移行を妨げない:

- **DuckDB**: ParquetStoreの内部実装をDuckDBに差し替え可能。SQLクエリによる高速JOIN/集計が可能。
- **Polars**: DataRepositoryの返り値をPolars DataFrameに変更可能。
- **クラウドストレージ**: ParquetStoreのパスをS3/GCSに変更可能。
