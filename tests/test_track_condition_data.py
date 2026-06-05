"""test_track_condition_data.py — track_condition_data モジュールのテスト

CSV→Parquet変換パイプラインの各関数を単体テストする。
DB不要、mock ParquetStore 使用。
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from features.track_condition_data import (
    aggregate_to_race_level,
    convert_track_conditions,
    parse_track_condition_csv,
    validate_physical_range,
)


# ── parse_track_condition_csv ──────────────────────────────────────


class TestParseTrackConditionCsv:
    """parse_track_condition_csv のテスト。"""

    def test_basic_parse(self, tmp_path: Path) -> None:
        """18桁IDの分割、日付導出、数値変換が正しいこと。"""
        csv_content = "202009120604010301,11.2\n202009120604010302,12.5\n"
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        result = parse_track_condition_csv(csv_file, "dirt_moisture")

        assert list(result.columns) == ["entry_id", "race_id", "umaban", "race_date", "dirt_moisture"]
        assert len(result) == 2
        assert result.iloc[0]["race_id"] == "2020091206040103"
        assert result.iloc[0]["umaban"] == "01"
        assert result.iloc[1]["umaban"] == "02"
        assert result.iloc[0]["race_date"] == pd.Timestamp("2020-09-12")
        assert result.iloc[0]["dirt_moisture"] == 11.2
        assert result.iloc[1]["dirt_moisture"] == 12.5

    def test_non_numeric_coerced_to_nan(self, tmp_path: Path) -> None:
        """非数値がNaNに変換されること。"""
        csv_content = "202009120604010301,11.2\n202009120604010302,abc\n"
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        result = parse_track_condition_csv(csv_file, "value")

        assert result.iloc[0]["value"] == 11.2
        assert pd.isna(result.iloc[1]["value"])

    def test_invalid_entry_id_length_raises(self, tmp_path: Path) -> None:
        """entry_idが18桁でない場合ValueErrorが発生すること。"""
        csv_content = "12345,11.2\n"
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        with pytest.raises(ValueError, match="entry_id must be 18 digits"):
            parse_track_condition_csv(csv_file, "value")


# ── aggregate_to_race_level ────────────────────────────────────────


class TestAggregateToRaceLevel:
    """aggregate_to_race_level のテスト。"""

    def test_identical_values_ok(self) -> None:
        """同一race_id内で値が全て同じ場合、その値が返ること。"""
        df = pd.DataFrame({
            "race_id": ["2020091206040103"] * 3,
            "race_date": [pd.Timestamp("2020-09-12")] * 3,
            "value": [2.8, 2.8, 2.8],
        })
        result = aggregate_to_race_level(df, "value")

        assert len(result) == 1
        assert result.iloc[0]["race_id"] == "2020091206040103"
        assert result.iloc[0]["value"] == 2.8

    def test_nan_non_nan_mix_picks_non_nan(self) -> None:
        """NaNと非NaNの混在時に非NaN値が採用されること。"""
        df = pd.DataFrame({
            "race_id": ["2020091206040103"] * 3,
            "race_date": [pd.Timestamp("2020-09-12")] * 3,
            "value": [np.nan, 2.8, np.nan],
        })
        result = aggregate_to_race_level(df, "value")

        assert len(result) == 1
        assert result.iloc[0]["value"] == 2.8

    def test_all_nan_returns_nan(self) -> None:
        """全てNaNの場合はNaNが返ること。"""
        df = pd.DataFrame({
            "race_id": ["2020091206040103"] * 3,
            "race_date": [pd.Timestamp("2020-09-12")] * 3,
            "value": [np.nan, np.nan, np.nan],
        })
        result = aggregate_to_race_level(df, "value")

        assert len(result) == 1
        assert pd.isna(result.iloc[0]["value"])

    def test_different_values_raises(self) -> None:
        """同一race_id内に異なる非NaN値がある場合ValueErrorが発生すること。"""
        df = pd.DataFrame({
            "race_id": ["2020091206040103"] * 3,
            "race_date": [pd.Timestamp("2020-09-12")] * 3,
            "value": [2.8, 3.1, 2.8],
        })
        with pytest.raises(ValueError, match="Multiple distinct non-NaN values"):
            aggregate_to_race_level(df, "value")

    def test_multiple_races(self) -> None:
        """複数race_idがそれぞれ正しく集約されること。"""
        df = pd.DataFrame({
            "race_id": ["2020091206040103"] * 2 + ["2020091306040103"] * 2,
            "race_date": [
                pd.Timestamp("2020-09-12"),
                pd.Timestamp("2020-09-12"),
                pd.Timestamp("2020-09-13"),
                pd.Timestamp("2020-09-13"),
            ],
            "value": [2.8, 2.8, 5.0, 5.0],
        })
        result = aggregate_to_race_level(df, "value")

        assert len(result) == 2
        assert set(result["race_id"]) == {"2020091206040103", "2020091306040103"}

    def test_empty_df(self) -> None:
        """空DataFrameでもエラーにならないこと。"""
        df = pd.DataFrame(columns=["race_id", "race_date", "value"])
        result = aggregate_to_race_level(df, "value")
        assert len(result) == 0
        assert list(result.columns) == ["race_id", "race_date", "value"]


# ── validate_physical_range ────────────────────────────────────────


class TestValidatePhysicalRange:
    """validate_physical_range のテスト。"""

    def test_out_of_range_replaced(self) -> None:
        """範囲外の値がNaNに置換されること。"""
        df = pd.DataFrame({"value": [50.0, -1.0, 101.0, 25.0, 0.0, 100.0]})
        result, count = validate_physical_range(df, "value", low=0.0, high=100.0)

        assert count == 4  # -1.0, 101.0, 0.0, 100.0 are all outside (0, 100) exclusive
        assert result.iloc[0]["value"] == 50.0  # in range
        assert pd.isna(result.iloc[1]["value"])  # -1.0
        assert pd.isna(result.iloc[2]["value"])  # 101.0
        assert result.iloc[3]["value"] == 25.0  # in range
        assert pd.isna(result.iloc[4]["value"])  # 0.0 (boundary, exclusive)
        assert pd.isna(result.iloc[5]["value"])  # 100.0 (boundary, exclusive)

    def test_valid_values_pass(self) -> None:
        """範囲内の値がそのまま通過すること。"""
        df = pd.DataFrame({"value": [1.0, 50.0, 99.0]})
        result, count = validate_physical_range(df, "value", low=0.0, high=100.0)

        assert count == 0
        assert list(result["value"]) == [1.0, 50.0, 99.0]

    def test_nan_preserved(self) -> None:
        """NaN値がNaNのまま保持されること。"""
        df = pd.DataFrame({"value": [50.0, np.nan, 25.0]})
        result, count = validate_physical_range(df, "value", low=0.0, high=100.0)

        assert count == 0
        assert result.iloc[0]["value"] == 50.0
        assert pd.isna(result.iloc[1]["value"])
        assert result.iloc[2]["value"] == 25.0

    def test_inf_high_for_positive_only(self) -> None:
        """high=np.infで正の値のみ許容すること。"""
        df = pd.DataFrame({"value": [5.0, -1.0, 0.0, 1000.0]})
        result, count = validate_physical_range(df, "value", low=0.0, high=np.inf)

        assert count == 2  # -1.0 and 0.0
        assert result.iloc[0]["value"] == 5.0
        assert pd.isna(result.iloc[1]["value"])  # -1.0
        assert pd.isna(result.iloc[2]["value"])  # 0.0 (boundary)
        assert result.iloc[3]["value"] == 1000.0  # any positive OK


# ── convert_track_conditions ───────────────────────────────────────


class TestConvertTrackConditions:
    """convert_track_conditions のテスト (end-to-end)。"""

    def _make_csv(self, tmp_path: Path, filename: str, lines: list[str]) -> Path:
        csv_file = tmp_path / filename
        csv_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return csv_file

    def test_end_to_end(self, tmp_path: Path) -> None:
        """両CSVからrace-level表が生成され、ParquetStore.writeが呼ばれること。"""
        dirt_csv = self._make_csv(tmp_path, "dirt.csv", [
            "202009120604010301,2.8",
            "202009120604010302,2.8",
            "202009130604010301,5.0",
        ])
        cushion_csv = self._make_csv(tmp_path, "cushion.csv", [
            "202009120604010301,11.2",
            "202009120604010302,11.2",
            "202009140604010301,9.5",
        ])

        mock_store = MagicMock(spec=["write"])
        mock_store.write = MagicMock()

        result = convert_track_conditions(dirt_csv, cushion_csv, mock_store)

        # 結果の検証
        assert set(result.columns) == {"race_id", "race_date", "dirt_moisture", "turf_cushion"}
        assert len(result) == 3  # 3 unique race_ids

        # race_idの確認
        race_ids = set(result["race_id"])
        assert "2020091206040103" in race_ids
        assert "2020091306040103" in race_ids
        assert "2020091406040103" in race_ids

        # 値の確認
        row_0912 = result[result["race_id"] == "2020091206040103"].iloc[0]
        assert row_0912["dirt_moisture"] == 2.8
        assert row_0912["turf_cushion"] == 11.2

        row_0913 = result[result["race_id"] == "2020091306040103"].iloc[0]
        assert row_0913["dirt_moisture"] == 5.0
        assert pd.isna(row_0913["turf_cushion"])

        row_0914 = result[result["race_id"] == "2020091406040103"].iloc[0]
        assert pd.isna(row_0914["dirt_moisture"])
        assert row_0914["turf_cushion"] == 9.5

        # ParquetStore.writeが呼ばれたこと
        mock_store.write.assert_called_once()
        call_args = mock_store.write.call_args
        assert call_args[0][0] == "raw"
        assert call_args[0][1] == "track_conditions"

    def test_with_races_df_cross_validation(self, tmp_path: Path) -> None:
        """races_dfとの交差検証が実行されること (ログ出力のみ、エラーなし)。"""
        dirt_csv = self._make_csv(tmp_path, "dirt.csv", [
            "202009120604010301,2.8",
        ])
        cushion_csv = self._make_csv(tmp_path, "cushion.csv", [
            "202009120604010301,11.2",
        ])

        mock_store = MagicMock(spec=["write"])

        # races_dfに含まれないrace_id
        races_df = pd.DataFrame({"race_id": ["9999999999999999"]})

        # エラーにならないこと (ログ警告のみ)
        result = convert_track_conditions(dirt_csv, cushion_csv, mock_store, races_df=races_df)
        assert len(result) == 1

    def test_physical_validation_in_pipeline(self, tmp_path: Path) -> None:
        """物理的異常値がパイプライン内でNaN化されること。"""
        dirt_csv = self._make_csv(tmp_path, "dirt.csv", [
            "202009120604010301,2.8",
            "202009120604010302,-5.0",  # 異常値
        ])
        cushion_csv = self._make_csv(tmp_path, "cushion.csv", [
            "202009120604010301,11.2",
        ])

        mock_store = MagicMock(spec=["write"])

        # dirtに2つの異なる値(non-NaN: 2.8 and -5.0)があるが、
        # -5.0はvalidate_physical_rangeでNaN化されるので同じ値2.8のみ残る
        result = convert_track_conditions(dirt_csv, cushion_csv, mock_store)

        row = result.iloc[0]
        assert row["dirt_moisture"] == 2.8
        assert row["turf_cushion"] == 11.2
