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
# HLF-01: HaronTime L4 History Stats
# ---------------------------------------------------------------------------

class TestHaronTimeL4Stats:
    """Tests for HaronTime L4 avg/zscore/trend features."""

    def test_harontimel4_avg_ema_weighted_values(self):
        """HLF-01: compute() returns harontimel4_avg with correct EMA-weighted values."""
        ketto = "12345"
        # Horse has 3 past races with harontimel4 values
        # Note: syussotosu comes from races_hist, not entries_hist
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

        assert "harontimel4_avg" in result.columns
        val = result["harontimel4_avg"].iloc[0]
        assert pd.notna(val)
        # EMA halflife=3: newest (46.0) gets highest weight
        # Expected: weights are (1-decay)^i reversed, so newest has highest weight
        # 46.0 should be closer to result than 47.0
        assert val < 47.0  # Newest (46.0) pulls down from simple mean 46.5

    def test_harontimel4_zscore_nan_when_no_expanding_stats(self):
        """HLF-01: compute() returns harontimel4_zscore as NaN when insufficient data."""
        ketto = "12345"
        # Only 1 past race - not enough for expanding_stats
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

        assert "harontimel4_zscore" in result.columns
        # With only 1 past race, expanding_stats won't have enough data
        # so zscore should be NaN
        assert pd.isna(result["harontimel4_zscore"].iloc[0])

    def test_harontimel4_trend_linear_regression(self):
        """HLF-01: compute() returns harontimel4_trend as linear regression slope."""
        ketto = "12345"
        # 3 past races with decreasing L4 times (improving)
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

        assert "harontimel4_trend" in result.columns
        val = result["harontimel4_trend"].iloc[0]
        assert pd.notna(val)
        # Times are decreasing (48.0 -> 46.0 -> 45.0), so trend should be negative
        assert val < 0

    def test_harontimel4_backward_compat_nan_when_column_absent(self):
        """HLF-01: harontimel4 values default to NaN when column absent from data."""
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

        assert "harontimel4_avg" in result.columns
        assert pd.isna(result["harontimel4_avg"].iloc[0])
        assert pd.isna(result["harontimel4_zscore"].iloc[0])
        assert pd.isna(result["harontimel4_trend"].iloc[0])


# ---------------------------------------------------------------------------
# HLF-01: harontime_last3f Unified Column
# ---------------------------------------------------------------------------

class TestHaronTimeLast3fUnified:
    """Tests for harontime_last3f distance-based auto-selection."""

    def test_unified_selects_l4_for_long_distance(self):
        """HLF-01: harontime_last3f_avg selects L4 for kyori >= 2000."""
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
        # kyori=2000: should select L4 (distance >= DISTANCE_THRESHOLD=2000)
        current_races = [_make_race_row("202401010101", "2024-01-01", kyori=2000)]

        hhf, entry_df, race_df = _create_hhf_with_history(
            past_entries, past_races, current_entries, current_races
        )
        result = hhf.compute(race_df, entry_df)

        assert "harontime_last3f_avg" in result.columns
        val = result["harontime_last3f_avg"].iloc[0]
        assert pd.notna(val)
        # Value should be based on L4 values (47.0, 46.0), not L3 (35.0, 34.0)
        # L4 values are in the ~46-47 range
        assert 44.0 < val < 48.0

    def test_unified_selects_l3_for_short_distance(self):
        """HLF-01: harontime_last3f_avg selects L3 for kyori < 2000."""
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
        # kyori=1600: should select L3 (distance < DISTANCE_THRESHOLD=2000)
        current_races = [_make_race_row("202401010101", "2024-01-01", kyori=1600)]

        hhf, entry_df, race_df = _create_hhf_with_history(
            past_entries, past_races, current_entries, current_races
        )
        result = hhf.compute(race_df, entry_df)

        val = result["harontime_last3f_avg"].iloc[0]
        assert pd.notna(val)
        # Value should be based on L3 values (35.0, 34.0), not L4 (47.0, 46.0)
        # L3 values are in the ~34-35 range
        assert 32.0 < val < 37.0

    def test_unified_fallback_when_preferred_is_nan(self):
        """HLF-01: harontime_last3f falls back when preferred column is NaN."""
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
        # kyori=2000: prefers L4, but L4 is all NaN -> should fallback to L3
        current_races = [_make_race_row("202401010101", "2024-01-01", kyori=2000)]

        hhf, entry_df, race_df = _create_hhf_with_history(
            past_entries, past_races, current_entries, current_races
        )
        result = hhf.compute(race_df, entry_df)

        val = result["harontime_last3f_avg"].iloc[0]
        assert pd.notna(val)
        # Should use L3 fallback values (35.0, 34.0)
        assert 32.0 < val < 37.0


# ---------------------------------------------------------------------------
# HLF-02: HaronTime race-rank extensions
# ---------------------------------------------------------------------------

class TestHaronTimeRaceRank:
    """Tests for harontimel4_avg and harontime_last3f_avg race_rank."""

    def test_race_transforms_produces_harontimel4_avg_race_rank(self):
        """HLF-02: add_race_transforms produces harontimel4_avg_race_rank."""
        from features.horse_history_features import HorseHistoryFeatures

        df = pd.DataFrame({
            "race_id": ["R1", "R1", "R1"],
            "harontimel4_avg": [46.0, 47.0, 45.0],
        })
        result = HorseHistoryFeatures.add_race_transforms(df)

        assert "harontimel4_avg_race_rank" in result.columns
        # rank(pct=True): 46.0=0.5, 47.0=1.0, 45.0=0.333...  (average method)
        ranks = result["harontimel4_avg_race_rank"].values
        assert all(pd.notna(r) for r in ranks)

    def test_race_transforms_produces_harontime_last3f_avg_race_rank(self):
        """HLF-02: add_race_transforms produces harontime_last3f_avg_race_rank."""
        from features.horse_history_features import HorseHistoryFeatures

        df = pd.DataFrame({
            "race_id": ["R1", "R1", "R1"],
            "harontime_last3f_avg": [35.0, 36.0, 34.0],
        })
        result = HorseHistoryFeatures.add_race_transforms(df)

        assert "harontime_last3f_avg_race_rank" in result.columns

    def test_race_transforms_skips_when_cols_missing(self):
        """HLF-02: add_race_transforms gracefully skips when HLF cols are missing."""
        from features.horse_history_features import HorseHistoryFeatures

        df = pd.DataFrame({
            "race_id": ["R1", "R1"],
            "norm_finish_logit_avg": [1.0, 2.0],
        })
        result = HorseHistoryFeatures.add_race_transforms(df)
        # Should not crash, and harontimel4_avg_race_rank should NOT be present
        assert "harontimel4_avg_race_rank" not in result.columns

    def test_race_predictor_mirrors_hlf_race_rank_cols(self):
        """HLF-05: RacePredictor._race_rank_cols includes HLF source columns."""
        from backtest.race_predictor import RacePredictor

        # _race_rank_cols should include the new HLF columns
        # We check by inspecting the predict method's _race_rank_cols list
        # (it's a local variable, so we check by examining source)
        import inspect
        source = inspect.getsource(RacePredictor.predict)
        assert "harontimel4_avg" in source
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
    ("MarketModel", "src.models.market_model", "FEATURE_COLS"),
    ("PlaceAbilityModel", "src.models.place_ability_model", "FEATURE_COLS"),
    ("RaceQualityScreener", "src.models.race_quality_screener", "FEATURE_COLS"),
    ("RegimeDetector", "src.models.regime_detector", "FEATURE_COLS"),
    ("WideTwoStageModel", "src.models.wide_two_stage_model", "SHARED_FEATURE_COLS"),
]

# HLF features that MUST be in all model FEATURE_COLS
HLF_HARON_FEATURES = [
    "harontimel4_avg",
    "harontimel4_zscore",
    "harontimel4_trend",
    "harontime_last3f_avg",
    "harontime_last3f_zscore",
    "harontime_last3f_trend",
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
    "harontimel4_avg_race_rank",
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

    def test_distance_threshold_constant(self):
        """HLF-01: DISTANCE_THRESHOLD = 2000 constant exists at module level."""
        from features.horse_history_features import DISTANCE_THRESHOLD
        assert DISTANCE_THRESHOLD == 2000
