"""ParquetStore のテスト — ファイルI/Oの読み書き・存在確認・パーティション対応"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

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
        df = pd.DataFrame(
            {
                "race_date": pd.to_datetime(["2020-01-01", "2020-06-15", "2021-01-01"]),
                "val": [1, 2, 3],
            }
        )
        store.write("raw", "races", df)
        result = store.read(
            "raw",
            "races",
            filters=[
                ("race_date", ">=", datetime(2020, 6, 1)),
                ("race_date", "<=", datetime(2020, 12, 31)),
            ],
        )
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
        df = pd.DataFrame(
            {
                "year": [2020, 2020, 2021],
                "month": [1, 2, 1],
                "val": [10, 20, 30],
            }
        )
        store.write("odds", "time_series", df, partition_cols=["year", "month"])
        assert (tmp_path / "odds" / "time_series").is_dir()
        # ディレクトリ構造があることを確認
        assert store.exists("odds", "time_series")

    def test_read_partitioned_with_filters(self, store: ParquetStore) -> None:
        df = pd.DataFrame(
            {
                "race_date": pd.to_datetime(["2020-01-15", "2020-02-15", "2021-01-15"]),
                "val": [1, 2, 3],
            }
        )
        # 年/月パーティションのためにカラム追加
        df["year"] = df["race_date"].dt.year
        df["month"] = df["race_date"].dt.month
        store.write("odds", "time_series", df, partition_cols=["year", "month"])

        result = store.read(
            "odds",
            "time_series",
            filters=[
                ("race_date", ">=", datetime(2020, 1, 1)),
                ("race_date", "<=", datetime(2020, 1, 31)),
            ],
        )
        assert len(result) == 1
        assert result["val"].iloc[0] == 1

    def test_exists_returns_true_for_partitioned_dir(self, store: ParquetStore) -> None:
        df = pd.DataFrame({"year": [2020], "month": [1], "val": [1]})
        store.write("odds", "time_series", df, partition_cols=["year", "month"])
        assert store.exists("odds", "time_series") is True
