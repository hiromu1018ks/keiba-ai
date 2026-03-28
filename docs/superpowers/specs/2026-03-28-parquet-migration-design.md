# Parquet Migration Design

**Date:** 2026-03-28
**Status:** Approved (Rev 2 — レビュー指摘対応済み)
**Scope:** 全データ層のPostgreSQL → Parquet移行

## 1. 目的

MLパイプライン（学習・バックテスト・推論）のデータアクセスをPostgreSQLからParquetに移行し、I/Oを高速化する。

## 2. 決定事項

| 項目 | 決定 |
|---|---|
| 移行範囲 | 全データ層（ETL + 特徴量 + 予測 + 馬券） |
| PostgreSQLの役割 | EveryDB2外部テーブルからのETL入力のみ |
| DataFrameライブラリ | pandas（`pd.read_parquet()`） |
| ファイル構成 | テーブル単位（1テーブル = 1ファイル） |
| ETL更新方法 | 全上書き |
| アプローチ | B（役割ごとにクラスを分ける） |
| race_id生成 | Python/pandasで計算（PostgreSQLのGENERATED COLUMNに依存しない） |
| 日付フォーマット | `"YYYYMMDD"` 文字列（既存と同一）。Parquet内には `race_date` (datetime) 列を永続化 |

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
│   ├── time_series.parquet    # オッズ時系列（race_date列を含む）
│   └── wide.parquet           # ワイドオッズ（race_date列を含む）
├── features/
│   └── horse_features.parquet # 馬の過去成績特徴量（キャッシュ）
├── predictions/
│   └── predictions.parquet    # 予測結果
└── bets/
    └── bets.parquet           # 馬券記録
```

全テーブルに `race_date` (datetime64) 列をETL時に永続化する。
これにより `filter_by_date()` は `race_date` 列でシンプルにフィルタできる。

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
        """
        path = self.data_dir / category / f"{name}.parquet"
        return pd.read_parquet(path, filters=filters)

    def write(self, category: str, name: str, df: pd.DataFrame) -> None:
        """DataFrameをParquetにアトミック書き込み（temp → rename）"""
        path = self.data_dir / category / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(path)  # アトミックリネーム

    def exists(self, category: str, name: str) -> bool:
        """ファイルが存在するか"""
        return (self.data_dir / category / f"{name}.parquet").exists()
```

**ポイント:**
- `filters` パラメータで pyarrow 述語プッシュダウンを利用 → 大きなファイルでも必要行だけ読める
- `write()` はテンポラリファイルに書き出してからリネーム → クラッシュ時の破損防止

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
    """year + month_day + jyo_cd + kaiji + nichiji + race_num → race_id"""
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

**重要:** PostgreSQL内部スキーマ (raw.*, odds_history.*) への書き込みは完全に廃止。
ETLは EveryDB2 → DataFrame → Parquet のみ。

### 5.3 DataRepository（新規: `src/db/repository.py`）

MLパイプラインの唯一のデータアクセス窓口。

**日付パラメータ:** `start`, `end` は `"YYYYMMDD"` 文字列（既存コードとの互換）。
内部で `datetime` に変換し、`race_date` 列でフィルタ。
障害除外（`track_cd NOT BETWEEN 51 AND 59`）もここで処理。

```python
def _parse_date(d: str) -> datetime:
    return datetime.strptime(d, "%Y%m%d")

def _filter(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """race_date で日付フィルタ + 障害除外"""
    s, e = _parse_date(start), _parse_date(end)
    mask = (df["race_date"] >= s) & (df["race_date"] <= e)
    if "track_cd" in df.columns:
        mask &= ~df["track_cd"].between(51, 59)
    return df[mask].copy()


class DataRepository:
    def __init__(self, store: ParquetStore):
        self.store = store

    # --- 読み取り ---

    def load_races(self, start: str, end: str) -> pd.DataFrame:
        df = self.store.read("raw", "races")
        return _filter(df, start, end)

    def load_entries(self, start: str, end: str) -> pd.DataFrame:
        df = self.store.read("raw", "entries")
        return _filter(df, start, end)

    def load_odds_snapshots(self, start: str, end: str) -> pd.DataFrame:
        df = self.store.read("odds", "snapshots")
        return _filter(df, start, end)

    def load_odds_time_series_range(self, start: str, end: str) -> pd.DataFrame:
        """オッズ時系列（日付範囲）"""
        df = self.store.read("odds", "time_series")
        return _filter(df, start, end)

    def load_odds_time_series(self, race_id: str) -> pd.DataFrame:
        """オッズ時系列（単一レース）"""
        df = self.store.read("odds", "time_series")
        return df[df["race_id"] == race_id].copy()

    def load_wide_odds(self, start: str, end: str) -> pd.DataFrame:
        df = self.store.read("odds", "wide")
        return _filter(df, start, end)

    def load_payouts(self, start: str, end: str) -> pd.DataFrame:
        df = self.store.read("raw", "payouts")
        return _filter(df, start, end)

    def load_all_races(self) -> pd.DataFrame:
        """日付フィルタなし。HorseHistoryFeatures等の全履歴参照用"""
        return self.store.read("raw", "races")

    def load_all_entries(self) -> pd.DataFrame:
        """日付フィルタなし。HorseHistoryFeatures等の全履歴参照用"""
        return self.store.read("raw", "entries")

    # --- 特徴量キャッシュ ---

    def load_features(self, start: str, end: str) -> pd.DataFrame | None:
        """特徴量キャッシュがあれば返す、なければNone"""
        if self.store.exists("features", "horse_features"):
            return _filter(self.store.read("features", "horse_features"), start, end)
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
```

### 5.5 HorseHistoryFeatures（変更: `src/features/horse_history_features.py`）

- 現在: 直接SQLで `raw.entries JOIN raw.races` をクエリ（馬ごと）
- 変更後: `DataRepository.load_all_entries()` + `load_all_races()` で全データをメモリにロードし、pandasのフィルタで過去成績を検索
- 初回ロード後はキャッシュして再利用（同じセッション内のバックテスト等で何度も呼ばれるため）

## 6. 影響を受けるファイル

| ファイル | 変更内容 |
|---|---|
| `src/db/connection.py` | `load_*()` 削除、`etl_to_parquet()` 追加、`_compute_race_id()` / `_compute_race_date()` 追加 |
| `src/db/etl.py` | `DatabaseConnection` に統合して **削除** |
| `src/db/parquet_store.py` | **新規** |
| `src/db/repository.py` | **新規** |
| `src/features/horse_history_features.py` | SQL直接アクセス → DataRepository 経由（メモリフィルタ） |
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

- 全データをメモリに1回だけロード
- レースごとは `df[df["race_id"] == target]` でフィルタ
- HorseHistoryFeatures もメモリ上で過去データを検索
- ネットワークI/Oがゼロになり、大幅な高速化が期待できる
- オッズ時系列（83M行）は pyarrow 述語プッシュダウンで必要範囲のみ読み取り

## 8. CLAUDE.md更新

ArchitectureセクションをParquetベースの構成に更新。将来のセッションでLLMが一目でアーキテクチャを理解できるようにする。
