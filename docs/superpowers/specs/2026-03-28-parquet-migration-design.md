# Parquet Migration Design

**Date:** 2026-03-28
**Status:** Approved
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
data/
├── raw/
│   ├── races.parquet          # レース情報
│   ├── entries.parquet        # 出走馬情報
│   └── payouts.parquet        # 払戻情報
├── odds/
│   ├── snapshots.parquet      # 最終オッズ
│   ├── time_series.parquet    # オッズ時系列
│   └── wide.parquet           # ワイドオッズ
├── features/
│   └── horse_features.parquet # 馬の過去成績特徴量（キャッシュ）
├── predictions/
│   └── predictions.parquet    # 予測結果
└── bets/
    └── bets.parquet           # 馬券記録
```

## 5. クラス設計

### 5.1 ParquetStore（新規: `src/db/parquet_store.py`）

Parquetファイルの読み書きのみを担当。

```python
class ParquetStore:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    def read(self, category: str, name: str) -> pd.DataFrame:
        """例: store.read("raw", "races") → data/raw/races.parquet"""
        return pd.read_parquet(self.data_dir / category / f"{name}.parquet")

    def write(self, category: str, name: str, df: pd.DataFrame) -> None:
        """DataFrameをParquetに全上書き"""
        path = self.data_dir / category / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

    def exists(self, category: str, name: str) -> bool:
        """ファイルが存在するか"""
        return (self.data_dir / category / f"{name}.parquet").exists()
```

### 5.2 DatabaseConnection（変更: `src/db/connection.py`）

- **ETL専用**になる
- 既存の `load_*()` メソッドは削除
- 新規に `etl_to_parquet(store: ParquetStore)` メソッドを追加（`etl.py` の処理を統合）
- PostgreSQL接続設定はそのまま保持

### 5.3 DataRepository（新規: `src/db/repository.py`）

MLパイプラインの唯一のデータアクセス窓口。

```python
class DataRepository:
    def __init__(self, store: ParquetStore):
        self.store = store

    def load_races(self, start: str, end: str) -> pd.DataFrame:
        df = self.store.read("raw", "races")
        return filter_by_date(df, start, end)

    def load_entries(self, start: str, end: str) -> pd.DataFrame:
        ...

    def load_odds_snapshots(self, start: str, end: str) -> pd.DataFrame:
        ...

    def load_features(self, start: str, end: str) -> pd.DataFrame | None:
        """特徴量キャッシュがあれば返す、なければNone"""
        if self.store.exists("features", "horse_features"):
            return filter_by_date(self.store.read("features", "horse_features"), start, end)
        return None

    def save_features(self, df: pd.DataFrame) -> None:
        self.store.write("features", "horse_features", df)

    def save_predictions(self, df: pd.DataFrame) -> None:
        ...

    def save_bets(self, df: pd.DataFrame) -> None:
        ...
```

### 5.4 etl.py → DatabaseConnection.etl_to_parquet() に統合

ETLの流れ:
```
PostgreSQL (EveryDB2外部テーブル)
  → SQLで読み取り → DataFrame
  → ParquetStore.write() でファイル出力
  → PostgreSQL内部スキーマ (raw.*, odds_history.*) への書き込みは廃止
```

### 5.5 HorseHistoryFeatures（変更: `src/features/horse_history_features.py`）

- 現在: 直接SQLで `raw.entries JOIN raw.races` をクエリ
- 変更後: `DataRepository` からデータを取得（メモリ上でフィルタ）

## 6. 影響を受けるファイル

| ファイル | 変更内容 |
|---|---|
| `src/db/connection.py` | `load_*()` 削除、`etl_to_parquet()` 追加 |
| `src/db/etl.py` | `DatabaseConnection` に統合して削除 |
| `src/db/parquet_store.py` | **新規** |
| `src/db/repository.py` | **新規** |
| `src/features/horse_history_features.py` | SQL直接アクセス → DataRepository 経由 |
| `src/features/feature_engine.py` | DataRepository からデータを受け取る |
| `src/pipelines/training_pipeline.py` | DatabaseConnection → DataRepository |
| `src/backtest/engine.py` | 同上。レースごとの再クエリ → メモリフィルタ |
| `CLAUDE.md` | アーキテクチャ説明をParquetベースに更新 |

### 影響を受けないファイル

- `src/domain/` — 変更なし
- `src/db/schema.py` — PostgreSQL DDLはETL入力用に残る
- `src/models/` — データソースを意識しない
- `config/settings.yaml` — `data_dir` パス設定を追加

## 7. バックテスト高速化

現在のバックテストエンジンはレースごとにDB再クエリしている。Parquet化により:

- 全データをメモリに1回だけロード
- レースごとは `df[df["race_id"] == target]` でフィルタ
- HorseHistoryFeatures もメモリ上で過去データを検索
- ネットワークI/Oがゼロになり、大幅な高速化が期待できる

## 8. CLAUDE.md更新

ArchitectureセクションをParquetベースの構成に更新。将来のセッションでLLMが一目でアーキテクチャを理解できるようにする。
