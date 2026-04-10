import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from db.parquet_store import ParquetStore
from features.jockey_trainer_combo import JockeyTrainerComboFeatures, FEATURE_COLS


def _make_entries_hist() -> pd.DataFrame:
    """騎手-調教師コンビの過去出走履歴"""
    return pd.DataFrame({
        "race_id": ["R001", "R002", "R003", "R004"],
        "race_date": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"]),
        "kisyucode": ["K01", "K01", "K01", "K02"],
        "chokyosicode": ["T01", "T01", "T02", "T01"],
        "kakuteijyuni": [1, 3, 5, 2],
        "umaban": [1, 1, 1, 1],
    })


def _make_entry_df() -> pd.DataFrame:
    """現在の出走データ"""
    return pd.DataFrame({
        "race_id": ["R005", "R005"],
        "umaban": [1, 2],
        "kisyucode": ["K01", "K02"],
        "chokyosicode": ["T01", "T01"],
        "race_date": pd.to_datetime(["2023-06-01", "2023-06-01"]),
    })


class TestJockeyTrainerCombo:
    def test_known_combo_stats(self):
        """K01+T01 コンビ: 2走中1勝 → wr = (1+1)/(2+11) ≈ 0.154"""
        mock_store = MagicMock(spec=ParquetStore)
        combo = JockeyTrainerComboFeatures(store=mock_store)
        combo._cache = _make_entries_hist()

        result = combo.compute(_make_entry_df())
        row = result[result["umaban"] == 1].iloc[0]
        # Beta(1,10): wr = (1+1)/(2+11) = 2/13 ≈ 0.154
        assert abs(row["jt_combo_wr"] - 2 / 13) < 1e-6
        # place_rate = (2+1)/(2+11) = 3/13 (1着+3着=2複勝)
        assert abs(row["jt_combo_place_rate"] - 3 / 13) < 1e-6
        assert row["jt_combo_starts"] == 2

    def test_unknown_combo_nan(self):
        """存在しないコンビ → NaN"""
        mock_store = MagicMock(spec=ParquetStore)
        combo = JockeyTrainerComboFeatures(store=mock_store)
        combo._cache = _make_entries_hist()

        entry = _make_entry_df().copy()
        entry["chokyosicode"] = ["T99", "T99"]  # 存在しない調教師
        result = combo.compute(entry)
        assert np.isnan(result.iloc[0]["jt_combo_wr"])

    def test_no_chokyosicode_column(self):
        """chokyosicode 列なし → 全 NaN"""
        mock_store = MagicMock(spec=ParquetStore)
        combo = JockeyTrainerComboFeatures(store=mock_store)
        combo._cache = pd.DataFrame()

        entry = _make_entry_df().drop(columns=["chokyosicode"])
        result = combo.compute(entry)
        for col in FEATURE_COLS:
            assert col in result.columns
            assert np.isnan(result.iloc[0][col])

    def test_feature_cols(self):
        assert FEATURE_COLS == [
            "jt_combo_wr", "jt_combo_place_rate",
            "jt_combo_starts", "jt_combo_prize_log",
        ]
