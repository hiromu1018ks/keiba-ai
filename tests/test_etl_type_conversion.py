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


class TestSentinelRules:
    """sentinel_float / sentinel_int rule tests (TDD RED -> GREEN)."""

    def test_sentinel_float_replaces_sentinels(self):
        """000/999 sentinels become NaN, valid values become float64."""
        from db.etl import _TABLE_TYPE_RULES, _apply_type_conversions

        df = pd.DataFrame({"harontimel3": ["000", "999", "345", ""]})
        result = _apply_type_conversions(df, "entries")
        vals = result["harontimel3"].tolist()
        assert math.isnan(vals[0])
        assert math.isnan(vals[1])
        assert vals[2] == 345.0
        assert math.isnan(vals[3])

    def test_sentinel_float_with_divisor(self):
        """LapTime: 000 -> NaN, valid values divided by 10."""
        df = pd.DataFrame({"laptime1": ["000", "345", "120", ""]})
        result = _apply_type_conversions(df, "races")
        vals = result["laptime1"].tolist()
        assert math.isnan(vals[0])
        assert vals[1] == 34.5
        assert vals[2] == 12.0
        assert math.isnan(vals[3])

    def test_sentinel_float_missing_columns(self):
        """Missing target columns do not cause errors."""
        df = pd.DataFrame({"other": ["1"]})
        result = _apply_type_conversions(df, "races")
        assert result["other"].tolist() == ["1"]

    def test_haron_timel3_migrated_from_float(self):
        """harontimel3 removed from entries.float, now in entries.sentinel_float."""
        from db.etl import _TABLE_TYPE_RULES

        entries_float = _TABLE_TYPE_RULES["entries"].get("float", [])
        assert "harontimel3" not in entries_float

        sentinel_rule = _TABLE_TYPE_RULES["entries"].get("sentinel_float", {})
        assert isinstance(sentinel_rule, dict)
        assert "harontimel3" in sentinel_rule.get("columns", [])

    def test_sentinel_float_no_double_processing(self):
        """harontimel3 in sentinel_float produces correct value, not NaN from double processing."""
        df = pd.DataFrame({"harontimel3": ["345"]})
        result = _apply_type_conversions(df, "entries")
        assert result["harontimel3"].iloc[0] == 345.0

    def test_races_harontime_sentinel(self):
        """RA table HaronTimeL3/L4 sentinels become NaN, valid values as float64."""
        df = pd.DataFrame({
            "harontimel3": ["000", "999", "350", ""],
            "harontimel4": ["000", "999", "470", ""],
        })
        result = _apply_type_conversions(df, "races")
        l3 = result["harontimel3"].tolist()
        l4 = result["harontimel4"].tolist()
        assert all(math.isnan(v) for v in l3[:2])
        assert l3[2] == 350.0
        assert math.isnan(l3[3])
        assert all(math.isnan(v) for v in l4[:2])
        assert l4[2] == 470.0
        assert math.isnan(l4[3])


class TestReadersCompat:
    """readers.py _INT_COLS/_FLOAT_COLS backward compatibility."""

    def test_float_cols_includes_harontimel4_and_laptimes(self):
        from db.readers import _FLOAT_COLS

        assert "harontimel4" in _FLOAT_COLS
        for i in range(1, 26):
            assert f"laptime{i}" in _FLOAT_COLS

    def test_int_cols_includes_jyuni23c(self):
        from db.readers import _INT_COLS

        assert "jyuni2c" in _INT_COLS
        assert "jyuni3c" in _INT_COLS


class TestPostRaceCols:
    """ETL-04: 含水率・クッション値がPOST_RACE_COLSに含まれないことをCI検証。"""

    def test_dirt_moisture_not_in_post_race_cols(self):
        from domain.types import POST_RACE_COLS

        assert "dirt_moisture" not in POST_RACE_COLS

    def test_turf_cushion_not_in_post_race_cols(self):
        from domain.types import POST_RACE_COLS

        assert "turf_cushion" not in POST_RACE_COLS
