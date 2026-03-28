# Parquet Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** PostgreSQL内部スキーマへの読み書きを廃止し、全データアクセスをParquetファイル経由に移行する。

**Architecture:** 3層データアクセス — `ParquetStore`（ファイルI/O）、`DataRepository`（ビジネスロジック窓口）、`DatabaseConnection`（ETL専用）。MLパイプラインは `DataRepository` のみを使用。

**Tech Stack:** pandas, pyarrow (Parquet + dataset API + predicate pushdown), pyarrow.parquet (partitioned writes)

**Spec:** `docs/superpowers/specs/2026-03-28-parquet-migration-design.md`

---

## File Structure

### New Files
| File | Responsibility |
|---|---|
| `src/db/parquet_store.py` | Parquetファイルの読み書き。単一ファイル + パーティション対応 |
| `src/db/repository.py` | MLパイプラインの唯一のデータアクセス窓口。日付フィルタ・障害除外・キャッシュ制御 |
| `tests/test_parquet_store.py` | ParquetStoreのテスト |
| `tests/test_repository.py` | DataRepositoryのテスト |

### Modified Files
| File | Change |
|---|---|
| `src/db/connection.py` | `load_*()` / `save_*()` 削除、`etl_to_parquet()` 追加、`_compute_race_id()` / `_compute_race_date()` 追加 |
| `src/db/etl.py` | **削除** — `connection.py` に統合 |
| `src/db/__init__.py` | exports更新 |
| `src/features/horse_history_features.py` | `Engine` → `DataRepository` 経由に変更 |
| `src/pipelines/training_pipeline.py` | `DatabaseConnection` → `DataRepository` |
| `src/backtest/engine.py` | 同上 |
| `src/ingestion/jvlink_fetcher.py` | 同上 |
| `src/ingestion/odds_collector.py` | 同上 |
| `src/backtest/validation_suite.py` | 同上 |
| `scripts/run_backtest.py` | 同上 |
| `CLAUDE.md` | アーキテクチャ説明をParquetベースに更新 |

### Unchanged Files
- `src/domain/` — データクラス・型定義
- `src/db/schema.py` — EveryDB2外部テーブル参照用DDL
- `src/features/feature_engine.py` — DataFrameを受け取るのみ
- `src/models/` — データソースを意識しない
- `.gitignore` — `data/` と `*.parquet` は既に登録済み
- `config/settings.yaml` — 既に `data_dir=data` を含む

---

## Task 0: Add pyarrow dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add pyarrow to dependencies**

```bash
pip install pyarrow
```

`pyproject.toml` の dependencies に `pyarrow>=14.0` を追加。

- [ ] **Step 2: Verify installation**

Run: `python -c "import pyarrow; print(pyarrow.__version__)"`
Expected: 14.x or higher

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: pyarrow依存関係追加（Parquet読み書き用）"
```

---

## Task 1: ParquetStore

**Files:**
- Create: `src/db/parquet_store.py`
- Create: `tests/test_parquet_store.py`

- [ ] **Step 1: Write failing tests for ParquetStore**

```python
# tests/test_parquet_store.py
"""ParquetStore のテスト — ファイルI/Oの読み書き・存在確認・パーティション対応"""
import pandas as pd
import pytest
from pathlib import Path

from db.parquet_store import ParquetStore


@pytest.fixture
def store(tmp_path: Path) -> ParquetStore:
    return ParquetStore(data_dir=str(tmp_path))


class TestParquetStoreWriteAndRead:
    def test_write_creates_parquet_file(self, store: ParquetStore, tmp_path: Path) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        store.write("raw", "races", df)
        assert (tmp_path / "raw" / "races.parquet").exists()

    def test_read_returns_written_dataframe(self, store: ParquetStore) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        store.write("raw", "races", df)
        result = store.read("raw", "races")
        pd.testing.assert_frame_equal(result, df)

    def test_write_overwrites_existing(self, store: ParquetStore) -> None:
        store.write("raw", "races", pd.DataFrame({"a": [1]}))
        store.write("raw", "races", pd.DataFrame({"a": [2]}))
        result = store.read("raw", "races")
        assert len(result) == 1
        assert result["a"].iloc[0] == 2

    def test_exists_returns_false_when_missing(self, store: ParquetStore) -> None:
        assert store.exists("raw", "races") is False

    def test_exists_returns_true_after_write(self, store: ParquetStore) -> None:
        store.write("raw", "races", pd.DataFrame({"a": [1]}))
        assert store.exists("raw", "races") is True

    def test_write_creates_subdirectories(self, store: ParquetStore, tmp_path: Path) -> None:
        store.write("odds", "snapshots", pd.DataFrame({"a": [1]}))
        assert (tmp_path / "odds").is_dir()

    def test_read_with_filters(self, store: ParquetStore) -> None:
        df = pd.DataFrame({"race_date": pd.to_datetime(["2020-01-01", "2020-06-15", "2021-01-01"]), "val": [1, 2, 3]})
        store.write("raw", "races", df)
        from datetime import datetime
        result = store.read("raw", "races",
            filters=[("race_date", ">=", datetime(2020, 6, 1)), ("race_date", "<=", datetime(2020, 12, 31))])
        assert len(result) == 1
        assert result["val"].iloc[0] == 2


class TestParquetStoreAtomicWrite:
    def test_no_tmp_file_remains_after_write(self, store: ParquetStore, tmp_path: Path) -> None:
        store.write("raw", "races", pd.DataFrame({"a": [1]}))
        assert not list(tmp_path.glob("**/*.tmp"))


class TestParquetStorePartitioned:
    def test_write_partitioned_creates_directory_structure(
        self, store: ParquetStore, tmp_path: Path
    ) -> None:
        df = pd.DataFrame({
            "year": [2020, 2020, 2021],
            "month": [1, 2, 1],
            "val": [10, 20, 30],
        })
        store.write("odds", "time_series", df, partition_cols=["year", "month"])
        assert (tmp_path / "odds" / "time_series").is_dir()
        # ディレクトリ構造があることを確認
        assert store.exists("odds", "time_series")

    def test_read_partitioned_with_filters(self, store: ParquetStore) -> None:
        df = pd.DataFrame({
            "race_date": pd.to_datetime(["2020-01-15", "2020-02-15", "2021-01-15"]),
            "val": [1, 2, 3],
        })
        # 年/月パーティションのためにカラム追加
        df["year"] = df["race_date"].dt.year
        df["month"] = df["race_date"].dt.month
        store.write("odds", "time_series", df, partition_cols=["year", "month"])

        from datetime import datetime
        result = store.read("odds", "time_series",
            filters=[("race_date", ">=", datetime(2020, 1, 1)), ("race_date", "<=", datetime(2020, 1, 31))])
        assert len(result) == 1
        assert result["val"].iloc[0] == 1

    def test_exists_returns_true_for_partitioned_dir(self, store: ParquetStore) -> None:
        df = pd.DataFrame({"year": [2020], "month": [1], "val": [1]})
        store.write("odds", "time_series", df, partition_cols=["year", "month"])
        assert store.exists("odds", "time_series") is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_parquet_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'db.parquet_store'`

- [ ] **Step 3: Implement ParquetStore**

```python
# src/db/parquet_store.py
"""Parquetファイルの読み書きを担当するクラス。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


class ParquetStore:
    """Parquetファイルの読み書きのみを担当。データの意味は知らない。"""

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = Path(data_dir)

    def read(
        self, category: str, name: str, filters: list[tuple] | None = None
    ) -> pd.DataFrame:
        """Parquetを読み取る。

        Args:
            category: カテゴリ (例: "raw", "odds")
            name: テーブル名 (例: "races")
            filters: pyarrow述語プッシュダウン用フィルタ
                     [(column, op, value), ...] 例: [("race_date", ">=", dt)]
        """
        path = self.data_dir / category / name
        if path.is_dir():
            dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
            if filters:
                mask = None
                for col, op, val in filters:
                    if op == ">=":
                        cond = ds.field(col) >= val
                    elif op == "<=":
                        cond = ds.field(col) <= val
                    elif op == "==":
                        cond = ds.field(col) == val
                    else:
                        raise ValueError(f"Unsupported filter operator: {op}")
                    mask = cond if mask is None else mask & cond
                table = dataset.to_table(filter=mask)
            else:
                table = dataset.to_table()
            return table.to_pandas()
        return pd.read_parquet(path.with_suffix(".parquet"), filters=filters)

    def write(
        self,
        category: str,
        name: str,
        df: pd.DataFrame,
        partition_cols: list[str] | None = None,
    ) -> None:
        """DataFrameをParquetに書き込む。

        partition_cols未指定時は単一ファイル（アトミック書き込み）。
        指定時はパーティション書き込み。
        """
        path = self.data_dir / category / name
        path.parent.mkdir(parents=True, exist_ok=True)

        if partition_cols:
            table = pa.Table.from_pandas(df)
            pq.write_to_dataset(table, root_path=str(path), partition_cols=partition_cols)
        else:
            file_path = path.with_suffix(".parquet")
            tmp = file_path.with_suffix(".parquet.tmp")
            df.to_parquet(tmp, index=False)
            tmp.replace(file_path)

    def exists(self, category: str, name: str) -> bool:
        """ファイル or パーティションディレクトリが存在するか。"""
        path = self.data_dir / category / name
        return path.with_suffix(".parquet").exists() or path.is_dir()
```

**注意:** `read()` のパーティション + filters 処理は複雑なので、実装後にテストで動作確認する。

- [ ] **Step 4: Run tests and fix until all pass**

Run: `python -m pytest tests/test_parquet_store.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/db/parquet_store.py tests/test_parquet_store.py
git commit -m "feat: ParquetStore — Parquetファイルの読み書きクラス"
```

---

## Task 2: DataRepository

**Files:**
- Create: `src/db/repository.py`
- Create: `tests/test_repository.py`

- [ ] **Step 1: Write failing tests for DataRepository**

```python
# tests/test_repository.py
"""DataRepository のテスト — 日付フィルタ・障害除外・pyarrowプッシュダウン"""
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from db.parquet_store import ParquetStore
from db.repository import DataRepository, _date_filters, _exclude_steeple, _to_dt


class TestHelpers:
    def test_to_dt(self) -> None:
        assert _to_dt("20200101") == datetime(2020, 1, 1)

    def test_date_filters(self) -> None:
        filters = _date_filters("20200101", "20201231")
        assert len(filters) == 2
        assert filters[0] == ("race_date", ">=", datetime(2020, 1, 1))
        assert filters[1] == ("race_date", "<=", datetime(2020, 12, 31))

    def test_exclude_steeple(self) -> None:
        df = pd.DataFrame({"track_cd": [30, 51, 55, 59, 10]})
        result = _exclude_steeple(df)
        assert list(result["track_cd"]) == [30, 10]

    def test_exclude_steeple_no_track_cd_column(self) -> None:
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(KeyError):
            _exclude_steeple(df)


@pytest.fixture
def mock_store() -> MagicMock:
    return MagicMock(spec=ParquetStore)


@pytest.fixture
def repo(mock_store: MagicMock) -> DataRepository:
    return DataRepository(store=mock_store)


class TestDataRepositoryLoadRaces:
    def test_calls_store_with_date_filters(self, repo: DataRepository, mock_store: MagicMock) -> None:
        mock_store.read.return_value = pd.DataFrame({
            "race_date": [datetime(2020, 6, 1)], "track_cd": [10]
        })
        repo.load_races("20200101", "20201231")
        call_args = mock_store.read.call_args
        assert call_args[0][:2] == ("raw", "races")
        filters = call_args[1].get("filters") or call_args[0][2]
        assert filters is not None

    def test_excludes_steeple(self, repo: DataRepository, mock_store: MagicMock) -> None:
        mock_store.read.return_value = pd.DataFrame({
            "race_date": [datetime(2020, 6, 1)] * 3,
            "track_cd": [10, 51, 55],
        })
        result = repo.load_races("20200101", "20201231")
        assert len(result) == 1
        assert result["track_cd"].iloc[0] == 10


class TestDataRepositoryLoadEntries:
    def test_calls_store_correctly(self, repo: DataRepository, mock_store: MagicMock) -> None:
        mock_store.read.return_value = pd.DataFrame({
            "race_date": [datetime(2020, 6, 1)], "track_cd": [10]
        })
        repo.load_entries("20200101", "20201231")
        call_args = mock_store.read.call_args
        assert call_args[0][:2] == ("raw", "entries")


class TestDataRepositoryLoadOddsTimeSeries:
    def test_range_calls_partitioned_table(self, repo: DataRepository, mock_store: MagicMock) -> None:
        mock_store.read.return_value = pd.DataFrame({"race_date": [datetime(2020, 6, 1)]})
        repo.load_odds_time_series_range("20200101", "20201231")
        call_args = mock_store.read.call_args
        assert call_args[0][:2] == ("odds", "time_series")

    def test_single_race_filters_by_race_id(self, repo: DataRepository, mock_store: MagicMock) -> None:
        mock_store.read.return_value = pd.DataFrame({"race_id": ["abc", "def"]})
        repo.load_odds_time_series("abc")
        call_args = mock_store.read.call_args
        assert call_args[0][:2] == ("odds", "time_series")
        filters = call_args[1].get("filters") or call_args[0][2]
        assert any(f[0] == "race_id" for f in filters)


class TestDataRepositoryLoadHistory:
    def test_load_history_entries_uses_lookback(self, repo: DataRepository, mock_store: MagicMock) -> None:
        mock_store.read.return_value = pd.DataFrame({"race_date": [datetime(2020, 1, 1)]})
        repo.load_history_entries(lookback_years=3)
        call_args = mock_store.read.call_args
        assert call_args[0][:2] == ("raw", "entries")
        filters = call_args[1].get("filters") or call_args[0][2]
        assert len(filters) == 1
        assert filters[0][0] == "race_date"

    def test_load_history_races_uses_lookback(self, repo: DataRepository, mock_store: MagicMock) -> None:
        mock_store.read.return_value = pd.DataFrame({"race_date": [datetime(2020, 1, 1)]})
        repo.load_history_races(lookback_years=5)
        call_args = mock_store.read.call_args
        assert call_args[0][:2] == ("raw", "races")


class TestDataRepositoryFeatures:
    def test_load_features_returns_none_when_missing(self, repo: DataRepository, mock_store: MagicMock) -> None:
        mock_store.exists.return_value = False
        result = repo.load_features("20200101", "20201231")
        assert result is None

    def test_load_features_returns_df_when_exists(self, repo: DataRepository, mock_store: MagicMock) -> None:
        mock_store.exists.return_value = True
        mock_store.read.return_value = pd.DataFrame({"race_date": [datetime(2020, 6, 1)]})
        result = repo.load_features("20200101", "20201231")
        assert result is not None


class TestDataRepositorySave:
    def test_save_features(self, repo: DataRepository, mock_store: MagicMock) -> None:
        df = pd.DataFrame({"a": [1]})
        repo.save_features(df)
        mock_store.write.assert_called_once_with("features", "horse_features", df)

    def test_save_predictions(self, repo: DataRepository, mock_store: MagicMock) -> None:
        df = pd.DataFrame({"a": [1]})
        repo.save_predictions(df)
        mock_store.write.assert_called_once_with("predictions", "predictions", df)

    def test_save_bets(self, repo: DataRepository, mock_store: MagicMock) -> None:
        df = pd.DataFrame({"a": [1]})
        repo.save_bets(df)
        mock_store.write.assert_called_once_with("bets", "bets", df)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_repository.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'db.repository'`

- [ ] **Step 3: Implement DataRepository**

```python
# src/db/repository.py
"""MLパイプラインの唯一のデータアクセス窓口。

将来DuckDB/Polarsへの移行を妨げないよう、この層が唯一のアクセス経路。
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from db.parquet_store import ParquetStore


def _to_dt(yyyymmdd: str) -> datetime:
    """'YYYYMMDD' 文字列 → datetime"""
    return datetime.strptime(yyyymmdd, "%Y%m%d")


def _date_filters(start: str, end: str) -> list[tuple]:
    """pyarrow述語プッシュダウン用フィルタを生成。"""
    s, e = _to_dt(start), _to_dt(end)
    return [("race_date", ">=", s), ("race_date", "<=", e)]


def _exclude_steeple(df: pd.DataFrame) -> pd.DataFrame:
    """障害レース除外（track_cd 51-59）。track_cd列が必須。"""
    return df[~df["track_cd"].between(51, 59)].copy()


class DataRepository:
    """MLパイプラインのデータアクセス窓口。"""

    def __init__(self, store: ParquetStore) -> None:
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
        """過去N年のentriesをロード。lookback_yearsでメモリ制御。"""
        cutoff = datetime.now() - timedelta(days=lookback_years * 365)
        return self.store.read("raw", "entries", filters=[("race_date", ">=", cutoff)])

    def load_history_races(self, lookback_years: int = 5) -> pd.DataFrame:
        """過去N年のracesをロード。"""
        cutoff = datetime.now() - timedelta(days=lookback_years * 365)
        return self.store.read("raw", "races", filters=[("race_date", ">=", cutoff)])

    # --- 特徴量キャッシュ ---

    def load_features(self, start: str, end: str) -> pd.DataFrame | None:
        """特徴量キャッシュがあれば返す、なければNone。"""
        if not self.store.exists("features", "horse_features"):
            return None
        return self.store.read("features", "horse_features", filters=_date_filters(start, end))

    def save_features(self, df: pd.DataFrame) -> None:
        self.store.write("features", "horse_features", df)

    # --- 予測・馬券 ---

    def save_predictions(self, df: pd.DataFrame) -> None:
        self.store.write("predictions", "predictions", df)

    def save_bets(self, df: pd.DataFrame) -> None:
        self.store.write("bets", "bets", df)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_repository.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/db/repository.py tests/test_repository.py
git commit -m "feat: DataRepository — MLパイプラインのデータアクセス窓口"
```

---

## Task 3: ETL Refactoring — DatabaseConnection.etl_to_parquet()

**Files:**
- Modify: `src/db/connection.py`
- Modify: `src/db/etl.py` → 統合後、削除
- Modify: `tests/test_etl.py`
- Modify: `tests/test_db.py`

**前提:** `etl.py` の既存のETL関数が EveryDB2外部テーブルからSQLで読み取っている部分は変更しない。変更点は:
1. `_compute_race_id()` / `_compute_race_date()` を connection.py に追加
2. `_insert_on_conflict()` 呼び出しを `ParquetStore.write()` に置換
3. `DatabaseConnection.etl_to_parquet(store, start, end)` を追加
4. 既存の `load_*()` / `save_*()` メソッドは **このタスクでは残す**（Task 5で消す）

- [ ] **Step 1: Add _compute_race_id() and _compute_race_date() to connection.py**

`src/db/connection.py` の `DatabaseConnection` クラスの前に追加:

```python
def _compute_race_id(df: pd.DataFrame) -> pd.DataFrame:
    """year + month_day + jyo_cd + kaiji + nichiji + race_num → race_id (16桁)"""
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
    """year + month_day → race_date (datetime64)

    注意: month_day は int (例: 101) または str (例: "0101") の両方に対応。
    ETL直後は int (101) → zfill で4桁に。
    """
    month_day_str = df["month_day"].astype(str).str.zfill(4)
    year_str = df["year"].astype(str).str.zfill(4)
    df["race_date"] = pd.to_datetime(year_str + month_day_str, format="%Y%m%d")
    return df
```

- [ ] **Step 2: Add etl_to_parquet() method to DatabaseConnection**

`DatabaseConnection` クラスに追加:

```python
def etl_to_parquet(self, store: "ParquetStore", start: str, end: str) -> dict[str, int]:
    """EveryDB2外部テーブル → Parquet にETL。"""
    from db.parquet_store import ParquetStore
    from db.etl import run_full_etl_to_parquet

    return run_full_etl_to_parquet(self.get_engine(), store, start, end)
```

- [ ] **Step 3: Add run_full_etl_to_parquet() to etl.py**

`src/db/etl.py` に追加（既存コードは残す）:

```python
def run_full_etl_to_parquet(
    engine: Engine, store: "ParquetStore", start: str, end: str
) -> dict[str, int]:
    """EveryDB2 → Parquet ETL。

    既存のSQL読み取り（EveryDB2外部テーブル）はそのまま使い、
    書き込み先をPostgreSQL内部スキーマ → Parquetに変更。
    """
    from db.connection import _compute_race_id, _compute_race_date

    counts: dict[str, int] = {}

    # 1. races
    sql = text("""SELECT ... FROM n_race WHERE ...""")  # 既存のetl_races SQL流用
    races_df = pd.read_sql(sql, engine, params={"start": int(start), "end": int(end)})
    # 既存の変換処理（列リネーム等）を適用
    races_df = _transform_races(races_df)  # 既存の変換ロジック
    _compute_race_id(races_df)
    _compute_race_date(races_df)
    store.write("raw", "races", races_df)
    counts["races"] = len(races_df)

    # 2. entries — races.parquetをJOIN
    sql = text("""SELECT ... FROM n_uma_race ...""")
    entries_df = pd.read_sql(sql, engine, params=...)
    entries_df = entries_df.merge(races_df[["race_id", "year", "month_day", "race_date"]],
                                   on=["year", "month_day", "jyo_cd", "kaiji", "nichiji", "race_num"],
                                   how="inner")
    _compute_race_date(entries_df)
    store.write("raw", "entries", entries_df)
    counts["entries"] = len(entries_df)

    # ... 同様に payouts, snapshots, wide, time_series ...
    # time_series は partition_cols=["year", "month"] で書き込み

    return counts
```

**重要 — SQLのJOIN依存解消 (C3):**

既存のETL関数（`etl_entries`, `etl_payouts`, `etl_odds_snapshots`, `etl_wide_odds`, `etl_odds_timeseries`）は全て `JOIN raw.races r ON ...` でPostgreSQL内部スキーマの `raw.races` に依存している。Parquet移行後はこのJOINが使えないため、以下のように対応する:

1. **races のSQL**: EveryDB2 `n_race` から直接読み取り（JOIN不要）→ `_compute_race_id()` + `_compute_race_date()` を適用
2. **entries以降**: EveryDB2から読み取った後、**メモリ上で `races_df` と pandas merge** して `race_id` / `race_date` を付与
3. 既存のSQLから `JOIN raw.races r ON ...` を **削除**し、EveryDB2テーブルのみからSELECTする

実装時は各ETL関数のSQLから `JOIN raw.races` を取り除き、代わりにpandas側でmergeする。

- [ ] **Step 4: Write test for _compute_race_id and _compute_race_date**

`tests/test_db.py` に追加:

```python
class TestComputeHelpers:
    def test_compute_race_id(self) -> None:
        from db.connection import _compute_race_id
        df = pd.DataFrame({
            "year": [2020], "month_day": [101], "jyo_cd": [5],
            "kaiji": [1], "nichiji": [1], "race_num": [11],
        })
        result = _compute_race_id(df)
        assert result["race_id"].iloc[0] == "2020010105010111"

    def test_compute_race_date(self) -> None:
        from db.connection import _compute_race_date
        df = pd.DataFrame({"year": [2020], "month_day": [315]})
        result = _compute_race_date(df)
        assert result["race_date"].iloc[0] == pd.Timestamp("2020-03-15")
```

- [ ] **Step 5: Run all DB-related tests**

Run: `python -m pytest tests/test_db.py tests/test_etl.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/db/connection.py src/db/etl.py tests/test_db.py tests/test_etl.py
git commit -m "feat: etl_to_parquet() — EveryDB2 → Parquet ETL追加"
```

---

## Task 4: HorseHistoryFeatures Migration

**Files:**
- Modify: `src/features/horse_history_features.py`
- Modify: `tests/test_horse_history_features.py`

**変更概要:** `HorseHistoryFeatures.__init__(engine)` → `__init__(repo: DataRepository)` に変更。
SQL直接クエリ → `repo.load_history_entries()` + `load_history_races()` でメモリフィルタ。

- [ ] **Step 1: Update HorseHistoryFeatures constructor and compute()**

```python
# src/features/horse_history_features.py
# 変更前:
#   def __init__(self, engine: Engine):
#       self.engine = engine
# 変更後:
#   def __init__(self, repo: "DataRepository"):
#       self.repo = repo
#       self._entries_cache: pd.DataFrame | None = None
#       self._races_cache: pd.DataFrame | None = None

# compute() 内の SQL クエリをパターン:
#   変更前: pd.read_sql(sql, self.engine, params={...})
#   変更後: self._get_history(ketto_nums, kisyu_codes)
#
# _get_history() は:
#   1. entries = self.repo.load_history_entries() — 初回のみロード、キャッシュ
#   2. races = self.repo.load_history_races() — 同上
#   3. pandas で ketto_num IN / kisyu_code IN にフィルタ
#   4. JOIN して返す
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_horse_history_features.py に追加

class TestHorseHistoryFeaturesWithRepo:
    """DataRepository経由のHorseHistoryFeaturesテスト"""

    @pytest.fixture
    def mock_repo(self) -> MagicMock:
        return MagicMock(spec=DataRepository)

    def test_constructor_accepts_repo(self, mock_repo: MagicMock) -> None:
        from features.horse_history_features import HorseHistoryFeatures
        hhf = HorseHistoryFeatures(repo=mock_repo)
        assert hhf.repo is mock_repo

    def test_compute_calls_load_history_entries(self, mock_repo: MagicMock) -> None:
        """compute() が load_history_entries() を呼ぶことを確認"""
        from features.horse_history_features import HorseHistoryFeatures
        # テスト用の最小DataFrame
        mock_repo.load_history_entries.return_value = pd.DataFrame({
            "race_id": ["r1"], "ketto_num": ["1234"], "kisyu_code": ["5678"],
            "finish_pos": [1], "win_odds": [2.0], "haron_time_l3": [34.5],
            "race_date": [pd.Timestamp("2020-01-01")], "field_size": [16],
        })
        mock_repo.load_history_races.return_value = pd.DataFrame({
            "race_id": ["r1"], "race_date": [pd.Timestamp("2020-01-01")],
            "field_size": [16],
        })
        hhf = HorseHistoryFeatures(repo=mock_repo)
        race_df = pd.DataFrame({"race_id": ["r2"], "ketto_num": ["1234"]})
        entry_df = pd.DataFrame({"race_id": ["r2"], "ketto_num": ["1234"], "kisyu_code": ["5678"]})
        result = hhf.compute(race_df, entry_df)
        mock_repo.load_history_entries.assert_called_once()

    def test_caching_prevents_repeated_loads(self, mock_repo: MagicMock) -> None:
        """2回目のcompute()はload_historyを再呼び出ししない（キャッシュ）"""
        from features.horse_history_features import HorseHistoryFeatures
        mock_repo.load_history_entries.return_value = pd.DataFrame({
            "race_id": ["r1"], "ketto_num": ["1234"], "kisyu_code": ["5678"],
            "finish_pos": [1], "win_odds": [2.0], "haron_time_l3": [34.5],
            "race_date": [pd.Timestamp("2020-01-01")], "field_size": [16],
        })
        mock_repo.load_history_races.return_value = pd.DataFrame({
            "race_id": ["r1"], "race_date": [pd.Timestamp("2020-01-01")],
        })
        hhf = HorseHistoryFeatures(repo=mock_repo)
        race_df = pd.DataFrame({"race_id": ["r2"], "ketto_num": ["1234"]})
        entry_df = pd.DataFrame({"race_id": ["r2"], "ketto_num": ["1234"], "kisyu_code": ["5678"]})
        hhf.compute(race_df, entry_df)
        hhf.compute(race_df, entry_df)
        # 1回しか呼ばれない（キャッシュ）
        assert mock_repo.load_history_entries.call_count == 1
```

**注意:** 既存の `TestLeakPrevention` は `pandas.read_sql` をパッチしている。このパッチは不要になるため削除する。
`_norm_finish_logit`, `_compute_jockey_surprise` 等の純粋関数テストは変更不要。

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_horse_history_features.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/features/horse_history_features.py tests/test_horse_history_features.py
git commit -m "refactor: HorseHistoryFeaturesをDataRepository経由に変更"
```

---

## Task 5: Consumer Migration

**Files:**
- Modify: `src/pipelines/training_pipeline.py`
- Modify: `src/backtest/engine.py`
- Modify: `src/ingestion/jvlink_fetcher.py`
- Modify: `src/ingestion/odds_collector.py`
- Modify: `src/backtest/validation_suite.py`
- Modify: `scripts/run_backtest.py`
- Modify: 対応する各テストファイル

**変更パターン（全ファイル共通）:**

```python
# 変更前:
#   from db.connection import DatabaseConnection
#   self.db = DatabaseConnection() or db_arg
#   self.db.load_races(start, end)
#
# 変更後:
#   from db.repository import DataRepository
#   from db.parquet_store import ParquetStore
#   self.repo = DataRepository(ParquetStore()) or repo_arg
#   self.repo.load_races(start, end)
```

**注意:** `load_entries_with_results()` → `load_entries()` にリネーム。

### 5a: TrainingPipelineV5

- [ ] **Step 1: Update constructor**

```python
# src/pipelines/training_pipeline.py
# __init__(self, db: DatabaseConnection | None = None, ...)
# → __init__(self, repo: DataRepository | None = None, db: DatabaseConnection | None = None, ...)
# db は etl_to_parquet 用に残す（後方互換）
```

- [ ] **Step 2: Update run() method**

```python
# 変更: self.db.load_*() → self.repo.load_*()
# 変更: self.db.load_entries_with_results() → self.repo.load_entries()
# 変更: self.db.load_wide_odds() → self.repo.load_wide_odds()
```

- [ ] **Step 3: Update _train_submodel()**

```python
# 変更: HorseHistoryFeatures(engine=self.db.get_engine())
# → HorseHistoryFeatures(repo=self.repo)
```

- [ ] **Step 4: Update tests**

```python
# tests/test_training_pipeline.py
# @patch("pipelines.training_pipeline.DatabaseConnection") を維持（db引数用）
# repo引数に DataRepository mock を渡すよう変更
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_training_pipeline.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/pipelines/training_pipeline.py tests/test_training_pipeline.py
git commit -m "refactor: TrainingPipelineV5をDataRepository経由に変更"
```

### 5b: BacktestEngine

- [ ] **Step 1-4: Same pattern as TrainingPipelineV5**

`src/backtest/engine.py`:
- `self.db` → `self.repo` (DataRepository)
- `self.db.load_*()` → `self.repo.load_*()`
- `HorseHistoryFeatures(engine=self.db.get_engine())` → `HorseHistoryFeatures(repo=self.repo)`

`tests/test_backtest_engine.py`: mock を `DatabaseConnection` → `DataRepository` に更新

- [ ] **Commit**

```bash
git add src/backtest/engine.py tests/test_backtest_engine.py
git commit -m "refactor: BacktestEngineをDataRepository経由に変更"
```

### 5c: JVLinkFetcher

- [ ] **Step 1-2: Update constructor and calls**

`src/ingestion/jvlink_fetcher.py`:
- `self.db: DatabaseConnection` → `self.repo: DataRepository`
- `self.db.load_races()` → `self.repo.load_races()`
- `self.db.load_entries_with_results()` → `self.repo.load_entries()`
- `self.db.load_odds_time_series()` → `self.repo.load_odds_time_series()`

`tests/test_jvlink_fetcher.py`: mock を更新

- [ ] **Commit**

```bash
git add src/ingestion/jvlink_fetcher.py tests/test_jvlink_fetcher.py
git commit -m "refactor: JVLinkFetcherをDataRepository経由に変更"
```

### 5d: OddsCollector

- [ ] **Step 1-2: Update constructor and calls**

`src/ingestion/odds_collector.py`:
- `self.db: Optional[DatabaseConnection]` → `self.repo: Optional[DataRepository]`
- `self.db.save_predictions()` → `self.repo.save_predictions()`
- **注意:** 現在 `store_snapshot()` (line 88) はオッズスナップショットを `save_predictions()` で保存している（セマンティクス不一致）。移行後もこの動作を維持するが、将来的に `save_odds_snapshot()` を DataRepository に追加すべき。

`tests/test_odds_collector.py`: mock を更新

- [ ] **Commit**

```bash
git add src/ingestion/odds_collector.py tests/test_odds_collector.py
git commit -m "refactor: OddsCollectorをDataRepository経由に変更"
```

### 5e: BacktestValidationSuite

- [ ] **Step 1-2: Update constructor and calls**

`src/backtest/validation_suite.py`:
- `self.db: DatabaseConnection | None` → `self.repo: DataRepository | None`
- **`run_walk_forward_cv()` (line 529)** 内の `TrainingPipelineV5(db=self.db)` (line 589) → `TrainingPipelineV5(repo=self.repo)`
- **`run_walk_forward_cv()` 内の `BacktestEngine(models=trained, db=self.db)` (line 597) → `BacktestEngine(models=trained, repo=self.repo)`

`tests/test_validation_suite.py`: mock を更新

- [ ] **Commit**

```bash
git add src/backtest/validation_suite.py tests/test_validation_suite.py
git commit -m "refactor: BacktestValidationSuiteをDataRepository経由に変更"
```

### 5f: scripts/run_backtest.py

- [ ] **Step 1: Update script**

```python
# 変更前:
# db = DatabaseConnection()
# pipeline = TrainingPipelineV5(db=db)
# engine = BacktestEngine(db=db, models=models)

# 変更後:
# from db.parquet_store import ParquetStore
# from db.repository import DataRepository
# db = DatabaseConnection()  # ETL用
# store = ParquetStore()
# repo = DataRepository(store)
# pipeline = TrainingPipelineV5(repo=repo)
# engine = BacktestEngine(repo=repo, models=models)
```

- [ ] **Commit**

```bash
git add scripts/run_backtest.py
git commit -m "refactor: run_backtest.pyをDataRepository経由に変更"
```

---

## Task 6: Cleanup — Remove Old Methods and etl.py

**Files:**
- Modify: `src/db/connection.py` — `load_*()`, `save_*()`, `get_engine()` の可視性変更
- Delete: `src/db/etl.py`
- Modify: `src/db/__init__.py` — exports更新
- Modify: `tests/test_etl.py` — 旧テスト削除・新テスト追加

**注意:** `get_engine()` は `etl_to_parquet()` 内部で使うため残すが、外部公開しない（`_get_engine()` にリネーム検討）。

- [ ] **Step 1: Remove load_*() and save_*() from DatabaseConnection**

```python
# src/db/connection.py
# 削除するメソッド:
#   load_races(), load_entries_with_results(), load_odds_snapshots(),
#   load_odds_time_series(), load_odds_time_series_range(),
#   load_wide_odds(), save_predictions(), save_bets()
# 残すメソッド:
#   get_engine(), create_schemas(), etl_to_parquet()
# 追加済み関数:
#   _compute_race_id(), _compute_race_date()
```

- [ ] **Step 2: Delete etl.py**

```bash
rm src/db/etl.py
```

ただし、`run_full_etl_to_parquet()` の内容は `connection.py` 内に移動するか、
`etl_to_parquet()` メソッド内に展開する。

- [ ] **Step 3: Update __init__.py**

```python
# src/db/__init__.py
from db.connection import DatabaseConnection
from db.parquet_store import ParquetStore
from db.repository import DataRepository
from db.schema import ALL_CREATE_STATEMENTS

__all__ = ["DatabaseConnection", "ParquetStore", "DataRepository", "ALL_CREATE_STATEMENTS"]
```

- [ ] **Step 4: Update tests**

- `tests/test_etl.py`: 旧ETLテストを削除。`run_full_etl_to_parquet` の新テストに置換（mock PostgreSQL read + ParquetStore.write 検証）
- `tests/test_db.py`: `load_*()` / `save_*()` テストを削除

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: 旧load/save系メソッド削除・etl.py統合"
```

---

## Task 7: CLAUDE.md Update

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update Architecture section**

CLAUDE.md の `## Architecture` セクションを以下に置換:

```markdown
## Architecture

### Data Layer (Parquet-based)

```
EveryDB2外部テーブル → PostgreSQL (ETL入力のみ) → Parquetファイル群
                                                      ↓
                         ParquetStore → DataRepository → MLパイプライン
```

### Class Structure

- **`ParquetStore`** (`src/db/parquet_store.py`) — Parquetファイルの読み書き。単一ファイル + 年/月パーティション対応。pyarrow述語プッシュダウン。
- **`DataRepository`** (`src/db/repository.py`) — MLパイプラインの唯一のデータアクセス窓口。日付フィルタ・障害除外・キャッシュ制御。
- **`DatabaseConnection`** (`src/db/connection.py`) — PostgreSQL ETL専用。EveryDB2 → Parquet への書き出し。

### Parquet Files

```
data/raw/races.parquet, entries.parquet, payouts.parquet
data/odds/snapshots.parquet, time_series/ (年/月パーティション), wide.parquet
data/features/horse_features.parquet  (特徴量キャッシュ)
data/predictions/predictions.parquet
data/bets/bets.parquet
```

全テーブルに `race_date` (datetime64) 列を含む。
`race_id` は `_compute_race_id()` でpandas計算（PostgreSQL GENERATED COLUMN不使用）。

### Key Dependencies

- `src/db/parquet_store.py` — pyarrow, pandas
- `src/db/repository.py` — ParquetStore, pandas
- `src/db/connection.py` — SQLAlchemy Core, ParquetStore, pandas

### Consumer Migration

全MLパイプラインコンポーネントは `DataRepository` を使用:
`TrainingPipelineV5`, `BacktestEngine`, `JVLinkFetcher`, `OddsCollector`, `BacktestValidationSuite`

### import path

- `from db.repository import DataRepository` — MLパイプライン用
- `from db.parquet_store import ParquetStore` — 低レベルI/O
- `from db.connection import DatabaseConnection` — ETL専用
```

- [ ] **Step 2: Update "Current State" to reflect Parquet**

`### Current State (Phase A: Foundation)` の記述を更新し、Parquet-based であることを反映。

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: CLAUDE.md — Parquetベースのアーキテクチャに更新"
```

---

## Final Verification

- [ ] **Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Run linting**

Run: `ruff check src/ tests/`
Expected: No errors

Run: `ruff format --check src/ tests/`
Expected: No errors

- [ ] **Run type checking**

Run: `mypy src/`
Expected: No errors

- [ ] **Final commit if any fixes needed**
