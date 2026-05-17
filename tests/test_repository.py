"""DataRepository のテスト — DB不要 (全mock)"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from db.repository import DataRepository


@pytest.fixture
def mock_store():
    """DataRepositoryに注入するモックParquetStore。"""
    store = MagicMock()
    store.read.return_value = pd.DataFrame(
        {
            "race_date": pd.to_datetime(["2024-01-01"]),
            "kumi": ["0102"],
            "odds": [10.5],
            "ninki": [3],
        }
    )
    return store


# --- Init ---


class TestInit:
    def test_default_store_created(self):
        """引数なしの場合、内部で ParquetStore() が生成される。"""
        with patch("db.repository.ParquetStore") as mock_ps:
            repo = DataRepository()
            mock_ps.assert_called_once()
            assert repo._store is mock_ps.return_value

    def test_explicit_store_used(self, mock_store):
        """明示的に渡した ParquetStore が使われる。"""
        repo = DataRepository(store=mock_store)
        assert repo._store is mock_store


# --- load_trio_odds ---


class TestLoadTrioOdds:
    def test_calls_store_with_correct_args(self, mock_store):
        """store.read が ("odds", "odds_sanren") で呼ばれる。"""
        repo = DataRepository(store=mock_store)
        repo.load_trio_odds("20240101", "20241231")
        mock_store.read.assert_called_once()
        args, kwargs = mock_store.read.call_args
        assert args == ("odds", "odds_sanren")

    def test_passes_date_filters(self, mock_store):
        """filters キーワード引数に _date_filters の結果が渡る。"""
        repo = DataRepository(store=mock_store)
        repo.load_trio_odds("20240101", "20241231")
        args, kwargs = mock_store.read.call_args
        filters = kwargs["filters"]
        assert len(filters) == 2
        assert filters[0][0] == "race_date"
        assert filters[0][1] == ">="
        assert filters[1][0] == "race_date"
        assert filters[1][1] == "<="

    def test_returns_dataframe(self, mock_store):
        """戻り値が DataFrame である。"""
        repo = DataRepository(store=mock_store)
        result = repo.load_trio_odds("20240101", "20241231")
        assert isinstance(result, pd.DataFrame)


# --- load_exacta_odds ---


class TestLoadExactaOdds:
    def test_calls_store_with_correct_args(self, mock_store):
        """store.read が ("odds", "odds_umaren") で呼ばれる。"""
        repo = DataRepository(store=mock_store)
        repo.load_exacta_odds("20240101", "20241231")
        mock_store.read.assert_called_once()
        args, kwargs = mock_store.read.call_args
        assert args == ("odds", "odds_umaren")

    def test_passes_date_filters(self, mock_store):
        """filters キーワード引数に _date_filters の結果が渡る。"""
        repo = DataRepository(store=mock_store)
        repo.load_exacta_odds("20240101", "20241231")
        args, kwargs = mock_store.read.call_args
        filters = kwargs["filters"]
        assert len(filters) == 2

    def test_returns_dataframe(self, mock_store):
        """戻り値が DataFrame である。"""
        repo = DataRepository(store=mock_store)
        result = repo.load_exacta_odds("20240101", "20241231")
        assert isinstance(result, pd.DataFrame)


# --- load_trifecta_odds ---


class TestLoadTrifectaOdds:
    def test_calls_store_with_correct_args(self, mock_store):
        """store.read が ("odds", "odds_sanrentan") で呼ばれる。"""
        repo = DataRepository(store=mock_store)
        repo.load_trifecta_odds("20240101", "20241231")
        mock_store.read.assert_called_once()
        args, kwargs = mock_store.read.call_args
        assert args == ("odds", "odds_sanrentan")

    def test_passes_date_filters(self, mock_store):
        """filters キーワード引数に _date_filters の結果が渡る。"""
        repo = DataRepository(store=mock_store)
        repo.load_trifecta_odds("20240101", "20241231")
        args, kwargs = mock_store.read.call_args
        filters = kwargs["filters"]
        assert len(filters) == 2

    def test_returns_dataframe(self, mock_store):
        """戻り値が DataFrame である。"""
        repo = DataRepository(store=mock_store)
        result = repo.load_trifecta_odds("20240101", "20241231")
        assert isinstance(result, pd.DataFrame)
