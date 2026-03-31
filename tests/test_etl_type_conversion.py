"""ETL型変換のテスト"""

import math

import pandas as pd

from db.etl import (
    _apply_type_conversions,
    _compute_surface,
    _compute_track_condition_code,
)


class TestApplyTypeConversions:
    def test_entries_int_conversion(self):
        df = pd.DataFrame({"umaban": ["1", "2", ""], "kakuteijyuni": ["1", "0", ""]})
        result = _apply_type_conversions(df, "entries")
        assert result["umaban"].tolist() == [1, 2, pd.NA]
        assert result["kakuteijyuni"].tolist() == [1, 0, pd.NA]

    def test_entries_float_conversion(self):
        df = pd.DataFrame({"time": ["65.3", "", "N/A"], "bataijyu": ["500", "", ""]})
        result = _apply_type_conversions(df, "entries")
        vals = result["time"].tolist()
        assert vals[0] == 65.3
        assert math.isnan(vals[1])
        assert math.isnan(vals[2])
        vals = result["bataijyu"].tolist()
        assert vals[0] == 500.0
        assert math.isnan(vals[1])
        assert math.isnan(vals[2])

    def test_entries_odds10_conversion(self):
        df = pd.DataFrame({"odds": ["0054", "0100", ""]})
        result = _apply_type_conversions(df, "entries")
        vals = result["odds"].tolist()
        assert vals[0] == 5.4
        assert vals[1] == 10.0
        assert math.isnan(vals[2])

    def test_odds_wide_odds100_conversion(self):
        df = pd.DataFrame({"oddslow": ["00150", "00200"], "oddshigh": ["00500", ""]})
        result = _apply_type_conversions(df, "odds_wide")
        assert result["oddslow"].tolist() == [1.50, 2.00]
        vals = result["oddshigh"].tolist()
        assert vals[0] == 5.00
        assert math.isnan(vals[1])

    def test_unknown_table_key(self):
        df = pd.DataFrame({"col": ["1"]})
        result = _apply_type_conversions(df, "nonexistent")
        assert result["col"].tolist() == ["1"]  # unchanged

    def test_missing_columns(self):
        df = pd.DataFrame({"other": ["1"]})
        result = _apply_type_conversions(df, "entries")
        assert "umaban" not in result.columns


class TestComputeSurface:
    def test_turf(self):
        df = pd.DataFrame({"trackcd": [10, 22, 23, 29, 51]})
        result = _compute_surface(df)
        assert result["surface"].tolist() == ["turf", "turf", "dirt", "dirt", "other"]

    def test_no_trackcd(self):
        df = pd.DataFrame({"col": [1]})
        result = _compute_surface(df)
        assert "surface" not in result.columns


class TestComputeTrackConditionCode:
    def test_turf_uses_sibababacd(self):
        df = pd.DataFrame(
            {
                "trackcd": [10, 22, 23, 29],
                "sibababacd": ["2", "3", "1", "2"],
                "dirtbabacd": ["1", "2", "3", "4"],
            }
        )
        result = _compute_track_condition_code(df)
        # np.where preserves string dtype from source columns
        assert result["track_condition_code"].tolist() == ["2", "3", "3", "4"]

    def test_missing_columns(self):
        df = pd.DataFrame({"trackcd": [10]})
        result = _compute_track_condition_code(df)
        assert "track_condition_code" not in result.columns
