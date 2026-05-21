"""test_hlf_features.py — HLF (Haron/Lap Feature) computation tests

Tests for:
- HLF-01: HaronTime L4 history stats (avg/zscore/trend)
- HLF-01: harontime_last3f unified column (distance-based auto-selection)
- HLF-02: HaronTime race-rank extensions
- HLF-03: LapTime pace features (pace_ratio_avg/zscore/trend, segment avgs)
- HLF-04: Model FEATURE_COLS registration
- HLF-05: POST_RACE safety
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry_row(race_id: str = "202401010101", umaban: int = 1,
                    kettonum: str = "12345", kisyucode: str = "01001",
                    **kwargs) -> dict:
    """Create a single entry row dict."""
    row = {
        "race_id": race_id,
        "umaban": umaban,
        "kettonum": kettonum,
        "kisyucode": kisyucode,
    }
    row.update(kwargs)
    return row


def _make_race_row(race_id: str = "202401010101", race_date: str = "2024-01-01",
                   kyori: int = 2000, surface: str = "turf", **kwargs) -> dict:
    """Create a single race row dict."""
    row = {
        "race_id": race_id,
        "race_date": pd.Timestamp(race_date),
        "kyori": kyori,
        "surface": surface,
        "track_condition_code": 1,
        "gradecd": "A",
        "jyokencd1": 5.0,
        "syussotosu": 16,
        "trackcd": 1,
    }
    row.update(kwargs)
    return row


def _build_history_df(entries_list: list[dict], races_list: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build entries_hist and races_hist DataFrames from lists of row dicts."""
    entries_df = pd.DataFrame(entries_list)
    races_df = pd.DataFrame(races_list)
    return entries_df, races_df


def _setup_hhf_mock(store: MagicMock, entries_hist: pd.DataFrame,
                    races_hist: pd.DataFrame) -> None:
    """Patch HorseHistoryFeatures._get_history to return given data."""
    from features.horse_history_features import HorseHistoryFeatures
    HorseHistoryFeatures.clear_class_cache()
    instance = HorseHistoryFeatures(store)
    instance._entries_cache = entries_hist
    instance._races_cache = races_hist
    return instance


def _create_hhf_with_history(
    past_entries: list[dict],
    past_races: list[dict],
    current_entries: list[dict],
    current_races: list[dict],
) -> tuple:
    """Create a HorseHistoryFeatures instance with mock history data."""
    from features.horse_history_features import HorseHistoryFeatures

    store = MagicMock()
    HorseHistoryFeatures.clear_class_cache()
    hhf = HorseHistoryFeatures(store)

    # Build history data
    entries_hist = pd.DataFrame(past_entries)
    races_hist = pd.DataFrame(past_races)

    hhf._entries_cache = entries_hist
    hhf._races_cache = races_hist

    # Build current data
    entry_df = pd.DataFrame(current_entries)
    race_df = pd.DataFrame(current_races)

    return hhf, entry_df, race_df


# ---------------------------------------------------------------------------
# D-08: closing_speed_ratio (replaces HaronTime L4 direct stats)
# ---------------------------------------------------------------------------

class TestHaronTimeL4Stats:
    """Tests for closing_speed_ratio features (replaces harontimel4 direct stats)."""

    def test_closing_speed_ratio_avg_column_exists(self):
        """D-08: compute() returns closing_speed_ratio_avg column (replaces harontimel4_avg)."""
        ketto = "12345"
        past_entries = [
            _make_entry_row("202306010101", 1, ketto, race_date=pd.Timestamp("2023-06-01"),
                           kakuteijyuni=3, odds=10.0,
                           harontimel3=35.0, harontimel4=47.0),
            _make_entry_row("202307010101", 1, ketto, race_date=pd.Timestamp("2023-07-01"),
                           kakuteijyuni=2, odds=8.0,
                           harontimel3=34.5, harontimel4=46.5),
            _make_entry_row("202308010101", 1, ketto, race_date=pd.Timestamp("2023-08-01"),
                           kakuteijyuni=1, odds=5.0,
                           harontimel3=34.0, harontimel4=46.0),
        ]
        past_races = [
            _make_race_row("202306010101", "2023-06-01"),
            _make_race_row("202307010101", "2023-07-01"),
            _make_race_row("202308010101", "2023-08-01"),
        ]
        current_entries = [
            _make_entry_row("202401010101", 1, ketto),
        ]
        current_races = [
            _make_race_row("202401010101", "2024-01-01", kyori=2000),
        ]

        hhf, entry_df, race_df = _create_hhf_with_history(
            past_entries, past_races, current_entries, current_races
        )
        result = hhf.compute(race_df, entry_df)

        # Column name changed from harontimel4_avg to closing_speed_ratio_avg
        assert "closing_speed_ratio_avg" in result.columns
        assert "harontimel4_avg" not in result.columns

    def test_closing_speed_ratio_zscore_exists(self):
        """D-08: closing_speed_ratio_zscore column exists (NaN with insufficient data)."""
        ketto = "12345"
        past_entries = [
            _make_entry_row("202312010101", 1, ketto, race_date=pd.Timestamp("2023-12-01"),
                           kakuteijyuni=1, odds=5.0,
                           harontimel3=35.0, harontimel4=47.0),
        ]
        past_races = [
            _make_race_row("202312010101", "2023-12-01"),
        ]
        current_entries = [_make_entry_row("202401010101", 1, ketto)]
        current_races = [_make_race_row("202401010101", "2024-01-01", kyori=2000)]

        hhf, entry_df, race_df = _create_hhf_with_history(
            past_entries, past_races, current_entries, current_races
        )
        result = hhf.compute(race_df, entry_df)

        assert "closing_speed_ratio_zscore" in result.columns
        assert "harontimel4_zscore" not in result.columns

    def test_closing_speed_ratio_trend_column_exists(self):
        """D-08: closing_speed_ratio_trend column exists (may be NaN with insufficient data)."""
        ketto = "12345"
        past_entries = [
            _make_entry_row("202310010101", 1, ketto, race_date=pd.Timestamp("2023-10-01"),
                           kakuteijyuni=3, odds=10.0,
                           harontimel3=35.5, harontimel4=48.0),
            _make_entry_row("202311010101", 1, ketto, race_date=pd.Timestamp("2023-11-01"),
                           kakuteijyuni=2, odds=8.0,
                           harontimel3=34.5, harontimel4=46.0),
            _make_entry_row("202312010101", 1, ketto, race_date=pd.Timestamp("2023-12-01"),
                           kakuteijyuni=1, odds=5.0,
                           harontimel3=34.0, harontimel4=45.0),
        ]
        past_races = [
            _make_race_row("202310010101", "2023-10-01"),
            _make_race_row("202311010101", "2023-11-01"),
            _make_race_row("202312010101", "2023-12-01"),
        ]
        current_entries = [_make_entry_row("202401010101", 1, ketto)]
        current_races = [_make_race_row("202401010101", "2024-01-01", kyori=2000)]

        hhf, entry_df, race_df = _create_hhf_with_history(
            past_entries, past_races, current_entries, current_races
        )
        result = hhf.compute(race_df, entry_df)

        assert "closing_speed_ratio_trend" in result.columns
        assert "harontimel4_trend" not in result.columns

    def test_closing_speed_ratio_backward_compat_nan_when_column_absent(self):
        """D-08: closing_speed_ratio values are NaN when harontimel4 column absent from data."""
        ketto = "12345"
        past_entries = [
            _make_entry_row("202312010101", 1, ketto, race_date=pd.Timestamp("2023-12-01"),
                           kakuteijyuni=1, odds=5.0,
                           harontimel3=35.0),
            # Note: NO harontimel4 column
        ]
        past_races = [
            _make_race_row("202312010101", "2023-12-01"),
        ]
        current_entries = [_make_entry_row("202401010101", 1, ketto)]
        current_races = [_make_race_row("202401010101", "2024-01-01", kyori=2000)]

        hhf, entry_df, race_df = _create_hhf_with_history(
            past_entries, past_races, current_entries, current_races
        )
        result = hhf.compute(race_df, entry_df)

        assert "closing_speed_ratio_avg" in result.columns
        assert pd.isna(result["closing_speed_ratio_avg"].iloc[0])
        assert pd.isna(result["closing_speed_ratio_zscore"].iloc[0])
        assert pd.isna(result["closing_speed_ratio_trend"].iloc[0])


# ---------------------------------------------------------------------------
# D-07: harontime_last3f L3-only (distance-based split removed)
# ---------------------------------------------------------------------------

class TestHaronTimeLast3fUnified:
    """Tests for harontime_last3f L3-only (D-07: distance-based split removed)."""

    def test_last3f_always_uses_l3_even_for_long_distance(self):
        """D-07: harontime_last3f_avg uses L3 even for kyori >= 2000 (old: selected L4)."""
        ketto = "12345"
        past_entries = [
            _make_entry_row("202310010101", 1, ketto, race_date=pd.Timestamp("2023-10-01"),
                           kakuteijyuni=1, odds=5.0,
                           harontimel3=35.0, harontimel4=47.0),
            _make_entry_row("202311010101", 1, ketto, race_date=pd.Timestamp("2023-11-01"),
                           kakuteijyuni=2, odds=8.0,
                           harontimel3=34.0, harontimel4=46.0),
        ]
        past_races = [
            _make_race_row("202310010101", "2023-10-01"),
            _make_race_row("202311010101", "2023-11-01"),
        ]
        current_entries = [_make_entry_row("202401010101", 1, ketto)]
        current_races = [_make_race_row("202401010101", "2024-01-01", kyori=2000)]

        hhf, entry_df, race_df = _create_hhf_with_history(
            past_entries, past_races, current_entries, current_races
        )
        result = hhf.compute(race_df, entry_df)

        assert "harontime_last3f_avg" in result.columns
        val = result["harontime_last3f_avg"].iloc[0]
        assert pd.notna(val)
        # D-07: Always uses L3 (35.0, 34.0), NOT L4 (47.0, 46.0)
        assert 32.0 < val < 37.0

    def test_last3f_uses_l3_for_short_distance(self):
        """D-07: harontime_last3f_avg uses L3 for short distance (unchanged behavior)."""
        ketto = "12345"
        past_entries = [
            _make_entry_row("202310010101", 1, ketto, race_date=pd.Timestamp("2023-10-01"),
                           kakuteijyuni=1, odds=5.0,
                           harontimel3=35.0, harontimel4=47.0),
            _make_entry_row("202311010101", 1, ketto, race_date=pd.Timestamp("2023-11-01"),
                           kakuteijyuni=2, odds=8.0,
                           harontimel3=34.0, harontimel4=46.0),
        ]
        past_races = [
            _make_race_row("202310010101", "2023-10-01"),
            _make_race_row("202311010101", "2023-11-01"),
        ]
        current_entries = [_make_entry_row("202401010101", 1, ketto)]
        current_races = [_make_race_row("202401010101", "2024-01-01", kyori=1600)]

        hhf, entry_df, race_df = _create_hhf_with_history(
            past_entries, past_races, current_entries, current_races
        )
        result = hhf.compute(race_df, entry_df)

        val = result["harontime_last3f_avg"].iloc[0]
        assert pd.notna(val)
        # Uses L3 values (35.0, 34.0)
        assert 32.0 < val < 37.0

    def test_last3f_nan_when_l3_is_nan(self):
        """D-07: harontime_last3f is NaN when L3 is NaN (no L4 fallback)."""
        ketto = "12345"
        past_entries = [
            _make_entry_row("202310010101", 1, ketto, race_date=pd.Timestamp("2023-10-01"),
                           kakuteijyuni=1, odds=5.0,
                           harontimel3=float("nan"), harontimel4=47.0),
            _make_entry_row("202311010101", 1, ketto, race_date=pd.Timestamp("2023-11-01"),
                           kakuteijyuni=2, odds=8.0,
                           harontimel3=float("nan"), harontimel4=46.0),
        ]
        past_races = [
            _make_race_row("202310010101", "2023-10-01"),
            _make_race_row("202311010101", "2023-11-01"),
        ]
        current_entries = [_make_entry_row("202401010101", 1, ketto)]
        current_races = [_make_race_row("202401010101", "2024-01-01", kyori=2000)]

        hhf, entry_df, race_df = _create_hhf_with_history(
            past_entries, past_races, current_entries, current_races
        )
        result = hhf.compute(race_df, entry_df)

        # D-07: L3-only, so NaN when L3 is NaN (no L4 fallback)
        assert pd.isna(result["harontime_last3f_avg"].iloc[0])


# ---------------------------------------------------------------------------
# D-08: closing_speed_ratio race-rank extensions
# ---------------------------------------------------------------------------

class TestHaronTimeRaceRank:
    """Tests for closing_speed_ratio_avg and harontime_last3f_avg race_rank."""

    def test_race_transforms_produces_closing_speed_ratio_avg_race_rank(self):
        """D-08: add_race_transforms produces closing_speed_ratio_avg_race_rank."""
        from features.horse_history_features import HorseHistoryFeatures

        df = pd.DataFrame({
            "race_id": ["R1", "R1", "R1"],
            "closing_speed_ratio_avg": [0.75, 0.78, 0.74],
        })
        result = HorseHistoryFeatures.add_race_transforms(df)

        assert "closing_speed_ratio_avg_race_rank" in result.columns
        ranks = result["closing_speed_ratio_avg_race_rank"].values
        assert all(pd.notna(r) for r in ranks)

    def test_race_transforms_produces_harontime_last3f_avg_race_rank(self):
        """D-08: add_race_transforms produces harontime_last3f_avg_race_rank."""
        from features.horse_history_features import HorseHistoryFeatures

        df = pd.DataFrame({
            "race_id": ["R1", "R1", "R1"],
            "harontime_last3f_avg": [35.0, 36.0, 34.0],
        })
        result = HorseHistoryFeatures.add_race_transforms(df)

        assert "harontime_last3f_avg_race_rank" in result.columns

    def test_race_transforms_skips_when_cols_missing(self):
        """D-08: add_race_transforms gracefully skips when HLF cols are missing."""
        from features.horse_history_features import HorseHistoryFeatures

        df = pd.DataFrame({
            "race_id": ["R1", "R1"],
            "norm_finish_logit_avg": [1.0, 2.0],
        })
        result = HorseHistoryFeatures.add_race_transforms(df)
        # Should not crash, and closing_speed_ratio_avg_race_rank should NOT be present
        assert "closing_speed_ratio_avg_race_rank" not in result.columns

    def test_race_predictor_mirrors_hlf_race_rank_cols(self):
        """D-08: RacePredictor._race_rank_cols includes closing_speed_ratio_avg."""
        from backtest.race_predictor import RacePredictor

        import inspect
        source = inspect.getsource(RacePredictor.predict)
        assert "closing_speed_ratio_avg" in source
        assert "harontime_last3f_avg" in source


# ---------------------------------------------------------------------------
# HLF-03: LapTime Pace Features
# ---------------------------------------------------------------------------

class TestLapTimePaceFeatures:
    """Tests for LapTime pace_ratio features."""

    def test_pace_ratio_nan_when_no_laptime_data(self):
        """HLF-03: pace_ratio features are NaN when no LapTime data available."""
        ketto = "12345"
        past_entries = [
            _make_entry_row("202310010101", 1, ketto, race_date=pd.Timestamp("2023-10-01"),
                           kakuteijyuni=1, odds=5.0,
                           harontimel3=35.0),
        ]
        past_races = [
            _make_race_row("202310010101", "2023-10-01"),
            # No laptime columns
        ]
        current_entries = [_make_entry_row("202401010101", 1, ketto)]
        current_races = [_make_race_row("202401010101", "2024-01-01")]

        hhf, entry_df, race_df = _create_hhf_with_history(
            past_entries, past_races, current_entries, current_races
        )
        result = hhf.compute(race_df, entry_df)

        for col in ["pace_ratio_avg", "pace_ratio_zscore", "pace_ratio_trend",
                     "pace_early_avg", "pace_mid_avg", "pace_late_avg"]:
            assert col in result.columns, f"Missing column: {col}"
            assert pd.isna(result[col].iloc[0]), f"{col} should be NaN when no laptime data"

    def test_pace_ratio_valid_with_laptime_data(self):
        """HLF-03: pace_ratio_avg is valid when past races have LapTime data."""
        ketto = "12345"
        # 2400m race = 12 laps, split into 3 segments of 4
        lap_values = [12.0, 12.1, 12.2, 12.3, 12.5, 12.6, 12.7, 12.8,
                      13.0, 13.1, 13.2, 13.3]
        race1_laps = {f"laptime{i+1}": lap_values[i] for i in range(12)}
        # Fill remaining laptime cols with NaN
        for i in range(13, 26):
            race1_laps[f"laptime{i}"] = float("nan")

        past_entries = [
            _make_entry_row("202310010101", 1, ketto, race_date=pd.Timestamp("2023-10-01"),
                           kakuteijyuni=1, odds=5.0,
                           harontimel3=35.0, harontimel4=47.0),
            _make_entry_row("202311010101", 1, ketto, race_date=pd.Timestamp("2023-11-01"),
                           kakuteijyuni=2, odds=8.0,
                           harontimel3=34.0, harontimel4=46.0),
        ]
        past_races = [
            _make_race_row("202310010101", "2023-10-01", kyori=2400, **race1_laps),
            _make_race_row("202311010101", "2023-11-01", kyori=2400, **race1_laps),
        ]
        current_entries = [_make_entry_row("202401010101", 1, ketto)]
        current_races = [_make_race_row("202401010101", "2024-01-01", kyori=2400)]

        hhf, entry_df, race_df = _create_hhf_with_history(
            past_entries, past_races, current_entries, current_races
        )
        result = hhf.compute(race_df, entry_df)

        assert "pace_ratio_avg" in result.columns
        val = result["pace_ratio_avg"].iloc[0]
        assert pd.notna(val), "pace_ratio_avg should be valid with LapTime data"
        # pace_ratio = late_avg / early_avg
        # early: [12.0, 12.1, 12.2, 12.3] -> avg ~12.15
        # mid: [12.5, 12.6, 12.7, 12.8] -> avg ~12.65
        # late: [13.0, 13.1, 13.2, 13.3] -> avg ~13.15
        # pace_ratio = 13.15 / 12.15 ≈ 1.082
        assert 1.0 < val < 1.2

    def test_pace_ratio_computed_correctly(self):
        """HLF-03: pace_ratio = late segment avg / early segment avg."""
        ketto = "12345"
        # 1600m = 8 laps -> segments: 3/3/2 via np.array_split
        lap_values = [12.0, 12.0, 12.0, 13.0, 13.0, 13.0, 14.0, 14.0]
        race_laps = {f"laptime{i+1}": lap_values[i] for i in range(8)}
        for i in range(9, 26):
            race_laps[f"laptime{i}"] = float("nan")

        past_entries = [
            _make_entry_row("202310010101", 1, ketto, race_date=pd.Timestamp("2023-10-01"),
                           kakuteijyuni=1, odds=5.0,
                           harontimel3=35.0, harontimel4=47.0),
        ]
        past_races = [
            _make_race_row("202310010101", "2023-10-01", kyori=1600, **race_laps),
        ]
        current_entries = [_make_entry_row("202401010101", 1, ketto)]
        current_races = [_make_race_row("202401010101", "2024-01-01", kyori=1600)]

        hhf, entry_df, race_df = _create_hhf_with_history(
            past_entries, past_races, current_entries, current_races
        )
        result = hhf.compute(race_df, entry_df)

        val = result["pace_ratio_avg"].iloc[0]
        assert pd.notna(val)
        # np.array_split([12,12,12,13,13,13,14,14], 3) -> [12,12,12], [13,13,13], [14,14]
        # early_avg=12.0, late_avg=14.0, pace_ratio=14.0/12.0 ≈ 1.167
        expected = 14.0 / 12.0
        assert abs(val - expected) < 0.01


# ---------------------------------------------------------------------------
# HLF-04: Model FEATURE_COLS Registration
# ---------------------------------------------------------------------------

# All model classes that should have HLF features registered
MODEL_CLASSES = [
    ("AbilityModel", "src.models.stage1_ability_model", "FEATURE_COLS"),
    ("WinTwoStageModel", "src.models.two_stage_return_model", "FEATURE_COLS"),
    ("EVCorrectionModel", "src.models.ev_correction_model", "FEATURE_COLS"),
    ("PlaceEVCorrectionModel", "src.models.ev_correction_model", "FEATURE_COLS"),
    ("ConformalEVModel", "src.models.conformal_ev_model", "FEATURE_COLS"),
    ("PlaceAbilityModel", "src.models.place_ability_model", "FEATURE_COLS"),
    ("RegimeDetector", "src.models.regime_detector", "FEATURE_COLS"),
    ("WideTwoStageModel", "src.models.wide_two_stage_model", "SHARED_FEATURE_COLS"),
]

# HLF features that MUST be in all model FEATURE_COLS
HLF_HARON_FEATURES = [
    "closing_speed_ratio_avg",
    "closing_speed_ratio_zscore",
    "closing_speed_ratio_trend",
    "harontime_last3f_avg",
    "harontime_last3f_zscore",
    "harontime_last3f_trend",
    # D-02: haron_race_gap (new in Phase 36.1)
    "haron_race_gap_avg",
    "haron_race_gap_zscore",
    "haron_race_gap_trend",
    # D-03: pace_adj_finish (new in Phase 36.1)
    "pace_adj_finish_avg",
]

HLF_LAP_FEATURES = [
    "pace_ratio_avg",
    "pace_ratio_zscore",
    "pace_ratio_trend",
    "pace_early_avg",
    "pace_mid_avg",
    "pace_late_avg",
]

HLF_RACE_RANK_FEATURES = [
    "closing_speed_ratio_avg_race_rank",
    "harontime_last3f_avg_race_rank",
]

ALL_HLF_FEATURES = HLF_HARON_FEATURES + HLF_LAP_FEATURES + HLF_RACE_RANK_FEATURES


@pytest.mark.parametrize("model_name,module_path,cols_attr", MODEL_CLASSES)
@pytest.mark.parametrize("feature_name", ALL_HLF_FEATURES)
def test_model_has_hlf_feature(model_name: str, module_path: str,
                                cols_attr: str, feature_name: str):
    """HLF-04: All 12 model FEATURE_COLS contain all HLF features."""
    import importlib
    module = importlib.import_module(module_path)
    model_cls = getattr(module, model_name)
    feature_cols = getattr(model_cls, cols_attr)
    assert feature_name in feature_cols, (
        f"{model_name}.{cols_attr} missing HLF feature: {feature_name}"
    )


# ---------------------------------------------------------------------------
# HLF-05: POST_RACE Safety
# ---------------------------------------------------------------------------

class TestPostRaceSafety:
    """Ensure HLF feature names are NOT in POST_RACE_COLS."""

    def test_hlf_features_not_in_post_race_cols(self):
        """HLF-05: No HLF feature name appears in POST_RACE_COLS."""
        from domain.types import POST_RACE_COLS

        for feature in ALL_HLF_FEATURES:
            assert feature not in POST_RACE_COLS, (
                f"HLF feature '{feature}' must NOT be in POST_RACE_COLS"
            )

    def test_base_cols_includes_all_hlf_features(self):
        """HLF-05: All HLF feature names are in HorseHistoryFeatures.BASE_COLS."""
        from features.horse_history_features import HorseHistoryFeatures

        for feature in HLF_HARON_FEATURES + HLF_LAP_FEATURES:
            assert feature in HorseHistoryFeatures.BASE_COLS, (
                f"HLF feature '{feature}' missing from BASE_COLS"
            )

    def test_distance_threshold_removed_or_unused(self):
        """D-07: DISTANCE_THRESHOLD should not exist or not be used in last3f calculation."""
        import features.horse_history_features as hhf_mod
        # DISTANCE_THRESHOLD should not be defined (removed)
        assert not hasattr(hhf_mod, "DISTANCE_THRESHOLD"), (
            "DISTANCE_THRESHOLD should be removed per D-07"
        )


class TestD09DataSourceFix:
    """D-09: race_cols_all must include harontimel4 for race-level L4 data."""

    def test_race_cols_all_includes_harontimel4(self):
        """D-09: race_cols_all has harontimel4 so races side L4 is merged."""
        from features.horse_history_features import HorseHistoryFeatures
        # We need to check that race_cols_all (used in compute()) includes harontimel4.
        # Since race_cols_all is a local variable, we verify by checking the source.
        import inspect
        source = inspect.getsource(HorseHistoryFeatures.compute)
        # Check that harontimel4 is in the race_cols_all list within the source
        assert '"harontimel4"' in source or "'harontimel4'" in source, (
            "race_cols_all must include 'harontimel4' per D-09"
        )


class TestD07Last3fL3Only:
    """D-07: harontime_last3f uses only L3 data, no DISTANCE_THRESHOLD branching."""

    def test_last3f_always_uses_l3_regardless_of_distance(self):
        """D-07: harontime_last3f_avg should use L3 values even for long distance races."""
        ketto = "12345"
        past_entries = [
            _make_entry_row("202310010101", 1, ketto, race_date=pd.Timestamp("2023-10-01"),
                           kakuteijyuni=1, odds=5.0,
                           harontimel3=35.0, harontimel4=47.0),
            _make_entry_row("202311010101", 1, ketto, race_date=pd.Timestamp("2023-11-01"),
                           kakuteijyuni=2, odds=8.0,
                           harontimel3=34.0, harontimel4=46.0),
        ]
        past_races = [
            _make_race_row("202310010101", "2023-10-01", kyori=2400),
            _make_race_row("202311010101", "2023-11-01", kyori=2400),
        ]
        current_entries = [_make_entry_row("202401010101", 1, ketto)]
        # kyori=2400 (long distance): OLD behavior would select L4 (47,46)
        # NEW behavior (D-07): always uses L3 (35,34)
        current_races = [_make_race_row("202401010101", "2024-01-01", kyori=2400)]

        hhf, entry_df, race_df = _create_hhf_with_history(
            past_entries, past_races, current_entries, current_races
        )
        result = hhf.compute(race_df, entry_df)

        val = result["harontime_last3f_avg"].iloc[0]
        assert pd.notna(val)
        # Should be based on L3 values (35.0, 34.0) = ~34.5 range, NOT L4 (46-47)
        assert 32.0 < val < 37.0, (
            f"Expected L3-based value (~34-35), got {val}. "
            "harontime_last3f should always use L3 per D-07"
        )

    def test_last3f_ignores_l4_fully(self):
        """D-07: When L3 is present but L4 is NaN, last3f should still work with L3."""
        ketto = "12345"
        past_entries = [
            _make_entry_row("202310010101", 1, ketto, race_date=pd.Timestamp("2023-10-01"),
                           kakuteijyuni=1, odds=5.0,
                           harontimel3=35.0, harontimel4=float("nan")),
            _make_entry_row("202311010101", 1, ketto, race_date=pd.Timestamp("2023-11-01"),
                           kakuteijyuni=2, odds=8.0,
                           harontimel3=34.0, harontimel4=float("nan")),
        ]
        past_races = [
            _make_race_row("202310010101", "2023-10-01"),
            _make_race_row("202311010101", "2023-11-01"),
        ]
        current_entries = [_make_entry_row("202401010101", 1, ketto)]
        current_races = [_make_race_row("202401010101", "2024-01-01", kyori=2400)]

        hhf, entry_df, race_df = _create_hhf_with_history(
            past_entries, past_races, current_entries, current_races
        )
        result = hhf.compute(race_df, entry_df)

        val = result["harontime_last3f_avg"].iloc[0]
        assert pd.notna(val)
        # Uses L3 (35.0, 34.0)
        assert 32.0 < val < 37.0


class TestD08HarontimeL4RenamedToClosingSpeedRatio:
    """D-08: harontimel4_avg/zscore/trend renamed to closing_speed_ratio_avg/zscore/trend."""

    def test_base_cols_has_closing_speed_ratio(self):
        """D-08: BASE_COLS contains closing_speed_ratio_avg/zscore/trend."""
        from features.horse_history_features import HorseHistoryFeatures
        for col in ["closing_speed_ratio_avg", "closing_speed_ratio_zscore",
                     "closing_speed_ratio_trend"]:
            assert col in HorseHistoryFeatures.BASE_COLS, (
                f"BASE_COLS missing: {col}"
            )

    def test_base_cols_no_harontimel4_avg(self):
        """D-08: BASE_COLS does NOT contain harontimel4_avg/zscore/trend."""
        from features.horse_history_features import HorseHistoryFeatures
        for col in ["harontimel4_avg", "harontimel4_zscore", "harontimel4_trend"]:
            assert col not in HorseHistoryFeatures.BASE_COLS, (
                f"BASE_COLS should NOT contain: {col} (replaced by closing_speed_ratio)"
            )

    def test_results_dict_keys_renamed(self):
        """D-08: compute() result has closing_speed_ratio_* keys, not harontimel4_*."""
        ketto = "12345"
        past_entries = [
            _make_entry_row("202310010101", 1, ketto, race_date=pd.Timestamp("2023-10-01"),
                           kakuteijyuni=1, odds=5.0,
                           harontimel3=35.0, harontimel4=47.0),
        ]
        past_races = [
            _make_race_row("202310010101", "2023-10-01", harontimel4=47.0),
        ]
        current_entries = [_make_entry_row("202401010101", 1, ketto)]
        current_races = [_make_race_row("202401010101", "2024-01-01", kyori=2000)]

        hhf, entry_df, race_df = _create_hhf_with_history(
            past_entries, past_races, current_entries, current_races
        )
        result = hhf.compute(race_df, entry_df)

        # New keys should exist
        assert "closing_speed_ratio_avg" in result.columns
        assert "closing_speed_ratio_zscore" in result.columns
        assert "closing_speed_ratio_trend" in result.columns
        # Old keys should NOT exist
        assert "harontimel4_avg" not in result.columns
        assert "harontimel4_zscore" not in result.columns
        assert "harontimel4_trend" not in result.columns

    def test_race_predictor_has_closing_speed_ratio(self):
        """D-08: RacePredictor._race_rank_cols uses closing_speed_ratio_avg."""
        from backtest.race_predictor import RacePredictor
        import inspect
        source = inspect.getsource(RacePredictor.predict)
        assert "closing_speed_ratio_avg" in source, (
            "RacePredictor._race_rank_cols should contain 'closing_speed_ratio_avg'"
        )
        # Check code lines only (excluding comments)
        lines = source.split('\n')
        code_lines = [l for l in lines if not l.strip().startswith('#')]
        code_source = '\n'.join(code_lines)
        assert '"harontimel4_avg"' not in code_source, (
            "RacePredictor should use closing_speed_ratio_avg, not harontimel4_avg"
        )

    def test_race_transforms_has_closing_speed_ratio(self):
        """D-08: add_race_transforms uses closing_speed_ratio_avg for race_rank."""
        from features.horse_history_features import HorseHistoryFeatures
        import inspect
        source = inspect.getsource(HorseHistoryFeatures.add_race_transforms)
        assert "closing_speed_ratio_avg" in source, (
            "add_race_transforms should contain 'closing_speed_ratio_avg' in race_rank_cols"
        )
        # Check that harontimel4_avg is not used as an actual column reference
        # (it may appear in comments like "replaces harontimel4_avg")
        lines = source.split('\n')
        code_lines = [l for l in lines if not l.strip().startswith('#')]
        code_source = '\n'.join(code_lines)
        assert '"harontimel4_avg"' not in code_source, (
            "add_race_transforms code should NOT reference 'harontimel4_avg' as a column"
        )


# ---------------------------------------------------------------------------
# Task 2: D-01 closing_speed_ratio + D-02 haron_race_gap + D-03 pace_adj_finish
# ---------------------------------------------------------------------------

class TestClosingSpeedRatio:
    """D-01: closing_speed_ratio = L3 / L4 computation tests."""

    def test_closing_speed_ratio_correct_calculation(self):
        """D-01: closing_speed_ratio = L3/L4 is correctly computed (L3=34.5, L4=46.0 -> ratio=0.750)."""
        ketto = "12345"
        past_entries = [
            _make_entry_row("202310010101", 1, ketto, race_date=pd.Timestamp("2023-10-01"),
                           kakuteijyuni=1, odds=5.0,
                           harontimel3=34.5),
        ]
        past_races = [
            _make_race_row("202310010101", "2023-10-01", harontimel4=46.0),
        ]
        current_entries = [_make_entry_row("202401010101", 1, ketto)]
        current_races = [_make_race_row("202401010101", "2024-01-01", kyori=2000)]

        hhf, entry_df, race_df = _create_hhf_with_history(
            past_entries, past_races, current_entries, current_races
        )
        result = hhf.compute(race_df, entry_df)

        val = result["closing_speed_ratio_avg"].iloc[0]
        assert pd.notna(val)
        # 34.5 / 46.0 = 0.75 (single value, EMA = value itself)
        assert abs(val - 0.75) < 0.01, f"Expected ~0.75, got {val}"

    def test_closing_speed_ratio_nan_when_l3_nan(self):
        """D-01: closing_speed_ratio is NaN when L3 is NaN."""
        ketto = "12345"
        past_entries = [
            _make_entry_row("202310010101", 1, ketto, race_date=pd.Timestamp("2023-10-01"),
                           kakuteijyuni=1, odds=5.0,
                           harontimel3=float("nan"), harontimel4=46.0),
        ]
        past_races = [
            _make_race_row("202310010101", "2023-10-01"),
        ]
        current_entries = [_make_entry_row("202401010101", 1, ketto)]
        current_races = [_make_race_row("202401010101", "2024-01-01", kyori=2000)]

        hhf, entry_df, race_df = _create_hhf_with_history(
            past_entries, past_races, current_entries, current_races
        )
        result = hhf.compute(race_df, entry_df)

        assert pd.isna(result["closing_speed_ratio_avg"].iloc[0])

    def test_closing_speed_ratio_nan_when_l4_nan(self):
        """D-01: closing_speed_ratio is NaN when L4 is NaN."""
        ketto = "12345"
        past_entries = [
            _make_entry_row("202310010101", 1, ketto, race_date=pd.Timestamp("2023-10-01"),
                           kakuteijyuni=1, odds=5.0,
                           harontimel3=35.0, harontimel4=float("nan")),
        ]
        past_races = [
            _make_race_row("202310010101", "2023-10-01"),
        ]
        current_entries = [_make_entry_row("202401010101", 1, ketto)]
        current_races = [_make_race_row("202401010101", "2024-01-01", kyori=2000)]

        hhf, entry_df, race_df = _create_hhf_with_history(
            past_entries, past_races, current_entries, current_races
        )
        result = hhf.compute(race_df, entry_df)

        assert pd.isna(result["closing_speed_ratio_avg"].iloc[0])

    def test_closing_speed_ratio_ema_avg(self):
        """D-01: closing_speed_ratio_avg uses EMA(halflife=3)."""
        ketto = "12345"
        # 3 races with increasing closing speed ratio
        past_entries = [
            _make_entry_row("202310010101", 1, ketto, race_date=pd.Timestamp("2023-10-01"),
                           kakuteijyuni=3, odds=10.0,
                           harontimel3=33.0),  # ratio = 0.6875 (L4=48.0 from race)
            _make_entry_row("202311010101", 1, ketto, race_date=pd.Timestamp("2023-11-01"),
                           kakuteijyuni=2, odds=8.0,
                           harontimel3=34.5),  # ratio = 0.75 (L4=46.0 from race)
            _make_entry_row("202312010101", 1, ketto, race_date=pd.Timestamp("2023-12-01"),
                           kakuteijyuni=1, odds=5.0,
                           harontimel3=35.5),  # ratio = 0.807 (L4=44.0 from race)
        ]
        past_races = [
            _make_race_row("202310010101", "2023-10-01", harontimel4=48.0),
            _make_race_row("202311010101", "2023-11-01", harontimel4=46.0),
            _make_race_row("202312010101", "2023-12-01", harontimel4=44.0),
        ]
        current_entries = [_make_entry_row("202401010101", 1, ketto)]
        current_races = [_make_race_row("202401010101", "2024-01-01", kyori=2000)]

        hhf, entry_df, race_df = _create_hhf_with_history(
            past_entries, past_races, current_entries, current_races
        )
        result = hhf.compute(race_df, entry_df)

        val = result["closing_speed_ratio_avg"].iloc[0]
        assert pd.notna(val)
        # Simple avg = 0.748, EMA weights newest more -> should be closer to 0.807
        assert val > 0.748, f"EMA avg should be > simple avg, got {val}"

    def test_closing_speed_ratio_trend(self):
        """D-01: closing_speed_ratio_trend is computed as linear regression slope."""
        ketto = "12345"
        past_entries = [
            _make_entry_row("202310010101", 1, ketto, race_date=pd.Timestamp("2023-10-01"),
                           kakuteijyuni=3, odds=10.0,
                           harontimel3=33.0),  # ratio = 0.6875 (L4=48.0)
            _make_entry_row("202311010101", 1, ketto, race_date=pd.Timestamp("2023-11-01"),
                           kakuteijyuni=2, odds=8.0,
                           harontimel3=34.0),  # ratio = 0.723 (L4=47.0)
            _make_entry_row("202312010101", 1, ketto, race_date=pd.Timestamp("2023-12-01"),
                           kakuteijyuni=1, odds=5.0,
                           harontimel3=35.0),  # ratio = 0.761 (L4=46.0)
        ]
        past_races = [
            _make_race_row("202310010101", "2023-10-01", harontimel4=48.0),
            _make_race_row("202311010101", "2023-11-01", harontimel4=47.0),
            _make_race_row("202312010101", "2023-12-01", harontimel4=46.0),
        ]
        current_entries = [_make_entry_row("202401010101", 1, ketto)]
        current_races = [_make_race_row("202401010101", "2024-01-01", kyori=2000)]

        hhf, entry_df, race_df = _create_hhf_with_history(
            past_entries, past_races, current_entries, current_races
        )
        result = hhf.compute(race_df, entry_df)

        val = result["closing_speed_ratio_trend"].iloc[0]
        assert pd.notna(val)
        # Ratios increasing: 0.6875 -> 0.723 -> 0.761 -> positive trend
        assert val > 0


class TestHaronRaceGap:
    """D-02: haron_race_gap = L3 - L4*0.75 computation tests."""

    def test_haron_race_gap_correct_calculation(self):
        """D-02: haron_race_gap = L3 - L4*0.75 (L3=34.5, L4=46.0 -> gap=0.0)."""
        ketto = "12345"
        past_entries = [
            _make_entry_row("202310010101", 1, ketto, race_date=pd.Timestamp("2023-10-01"),
                           kakuteijyuni=1, odds=5.0,
                           harontimel3=34.5),
        ]
        past_races = [
            _make_race_row("202310010101", "2023-10-01", harontimel4=46.0),
        ]
        current_entries = [_make_entry_row("202401010101", 1, ketto)]
        current_races = [_make_race_row("202401010101", "2024-01-01", kyori=2000)]

        hhf, entry_df, race_df = _create_hhf_with_history(
            past_entries, past_races, current_entries, current_races
        )
        result = hhf.compute(race_df, entry_df)

        val = result["haron_race_gap_avg"].iloc[0]
        assert pd.notna(val)
        # 34.5 - 46.0*0.75 = 34.5 - 34.5 = 0.0
        assert abs(val - 0.0) < 0.01, f"Expected ~0.0, got {val}"

    def test_haron_race_gap_negative_when_fast_closing(self):
        """D-02: Negative gap means fast closing (strong kick)."""
        ketto = "12345"
        past_entries = [
            _make_entry_row("202310010101", 1, ketto, race_date=pd.Timestamp("2023-10-01"),
                           kakuteijyuni=1, odds=5.0,
                           harontimel3=33.0),
        ]
        past_races = [
            _make_race_row("202310010101", "2023-10-01", harontimel4=46.0),
        ]
        current_entries = [_make_entry_row("202401010101", 1, ketto)]
        current_races = [_make_race_row("202401010101", "2024-01-01", kyori=2000)]

        hhf, entry_df, race_df = _create_hhf_with_history(
            past_entries, past_races, current_entries, current_races
        )
        result = hhf.compute(race_df, entry_df)

        val = result["haron_race_gap_avg"].iloc[0]
        assert pd.notna(val)
        # 33.0 - 46.0*0.75 = 33.0 - 34.5 = -1.5
        assert val < 0, f"Expected negative (fast closing), got {val}"

    def test_haron_race_gap_in_base_cols(self):
        """D-02: haron_race_gap_avg/zscore/trend are in BASE_COLS."""
        from features.horse_history_features import HorseHistoryFeatures
        for col in ["haron_race_gap_avg", "haron_race_gap_zscore", "haron_race_gap_trend"]:
            assert col in HorseHistoryFeatures.BASE_COLS, f"Missing: {col}"


class TestPaceAdjFinish:
    """D-03: pace_adj_finish_avg = norm_finish * pace_ratio past average."""

    def test_pace_adj_finish_in_base_cols(self):
        """D-03: pace_adj_finish_avg is in BASE_COLS."""
        from features.horse_history_features import HorseHistoryFeatures
        assert "pace_adj_finish_avg" in HorseHistoryFeatures.BASE_COLS

    def test_pace_adj_finish_computed_with_laptime(self):
        """D-03: pace_adj_finish_avg is computed when laptime data is available."""
        ketto = "12345"
        # 2400m = 12 laps, split into 3 segments of 4
        lap_values = [12.0, 12.1, 12.2, 12.3, 12.5, 12.6, 12.7, 12.8,
                      13.0, 13.1, 13.2, 13.3]
        race1_laps = {f"laptime{i+1}": lap_values[i] for i in range(12)}
        for i in range(13, 26):
            race1_laps[f"laptime{i}"] = float("nan")

        past_entries = [
            _make_entry_row("202310010101", 1, ketto, race_date=pd.Timestamp("2023-10-01"),
                           kakuteijyuni=1, odds=5.0,
                           harontimel3=35.0),
        ]
        past_races = [
            _make_race_row("202310010101", "2023-10-01", kyori=2400, syussotosu=12,
                           harontimel4=47.0, **race1_laps),
        ]
        current_entries = [_make_entry_row("202401010101", 1, ketto)]
        current_races = [_make_race_row("202401010101", "2024-01-01", kyori=2400)]

        hhf, entry_df, race_df = _create_hhf_with_history(
            past_entries, past_races, current_entries, current_races
        )
        result = hhf.compute(race_df, entry_df)

        assert "pace_adj_finish_avg" in result.columns


class TestBaseColsCount:
    """BASE_COLS total count after Task 2 additions."""

    def test_base_cols_count_after_task2(self):
        """BASE_COLS should be 66 after Task 2 (old 62 - 3 + 7 new = 66)."""
        from features.horse_history_features import HorseHistoryFeatures
        assert len(HorseHistoryFeatures.BASE_COLS) == 66, (
            f"BASE_COLS should be 66, got {len(HorseHistoryFeatures.BASE_COLS)}"
        )
