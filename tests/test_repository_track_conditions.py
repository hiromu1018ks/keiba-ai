"""DataRepository.load_track_conditions のテスト (ETL-03)。"""

from unittest.mock import MagicMock, patch

import pandas as pd

from db.repository import DataRepository


class TestLoadTrackConditions:
    """DataRepository.load_track_conditions のユニットテスト。"""

    def _make_store_mock(
        self, exists_return: bool, read_return: pd.DataFrame | None = None
    ) -> MagicMock:
        store = MagicMock()
        store.exists.return_value = exists_return
        if read_return is not None:
            store.read.return_value = read_return
        return store

    def test_returns_dataframe_when_parquet_exists(self):
        """parquetが存在する場合、read結果をcoerce_typesして返す。"""
        expected_df = pd.DataFrame({
            "race_id": ["202001010101"],
            "race_date": pd.to_datetime(["2020-01-01"]),
            "dirt_moisture": [5.0],
            "turf_cushion": [3.0],
        })
        store = self._make_store_mock(True, expected_df)
        repo = DataRepository(store)

        result = repo.load_track_conditions("20200101", "20201231")

        assert len(result) == 1
        assert list(result.columns) == ["race_id", "race_date", "dirt_moisture", "turf_cushion"]
        store.exists.assert_called_once_with("raw", "track_conditions")

    def test_passes_date_filters_to_read(self):
        """date_filters(start, end)がstore.readに渡される。"""
        store = self._make_store_mock(True, pd.DataFrame({"race_id": []}))
        repo = DataRepository(store)

        repo.load_track_conditions("20200101", "20201231")

        store.read.assert_called_once()
        call_args = store.read.call_args
        assert call_args[0][0] == "raw"
        assert call_args[0][1] == "track_conditions"
        # filters are passed as keyword arg
        filters = call_args[1].get("filters") or call_args[0][2]
        assert len(filters) == 2
        assert filters[0][0] == "race_date"
        assert filters[0][1] == ">="
        assert filters[1][0] == "race_date"
        assert filters[1][1] == "<="

    def test_returns_empty_dataframe_when_parquet_missing(self):
        """parquetが存在しない場合、空DataFrameを返す。"""
        store = self._make_store_mock(False)
        repo = DataRepository(store)

        result = repo.load_track_conditions("20200101", "20201231")

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        store.read.assert_not_called()

    def test_coerce_types_applied(self):
        """coerce_typesが呼び出されることで型変換が適用される。"""
        raw_df = pd.DataFrame({
            "race_id": ["202001010101"],
            "race_date": ["2020-01-01"],  # string, not datetime
            "dirt_moisture": [5.0],
            "turf_cushion": [3.0],
        })
        store = self._make_store_mock(True, raw_df)
        repo = DataRepository(store)

        result = repo.load_track_conditions("20200101", "20201231")

        # coerce_types converts string race_date to datetime
        assert pd.api.types.is_datetime64_any_dtype(result["race_date"])
