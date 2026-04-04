"""test_jockey_context_features.py — JockeyContextFeatures の単体テスト"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from features.jockey_context_features import FEATURE_COLS, JockeyContextFeatures

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_store(jockey_stats_df: pd.DataFrame) -> MagicMock:
    store = MagicMock()
    store.read.return_value = jockey_stats_df
    return store


def _make_jockey_stats_row(
    kisyucode: str = "JK001",
    setyear: int = 2023,
    wins: int = 50,
    total_starts: int = 400,
    ky1_wins: int = 20,
    ky1_total: int = 150,
    j5_wins: int = 10,
    j5_total: int = 80,
    honsyokinheichi: float = 100000.0,
) -> dict:
    """Build one row of x_KISYU_SEISEKI data.

    heichichakukaisu1-6: overall performance (1=win, 2-5=places, 6=unplaced)
    kyori1chakukaisu1-6: distance category 1 performance
    jyo5chakukaisu1-6: venue 5 performance
    """
    non_wins = total_starts - wins
    rest = non_wins
    row: dict = {
        "kisyucode": kisyucode,
        "setyear": setyear,
        "heichichakukaisu1": wins,
        "honsyokinheichi": honsyokinheichi,
    }
    for i in range(2, 7):
        val = rest // 5 if i < 6 else rest - 4 * (rest // 5)
        row[f"heichichakukaisu{i}"] = val

    ky_non = ky1_total - ky1_wins
    row["kyori1chakukaisu1"] = ky1_wins
    for i in range(2, 7):
        val = ky_non // 5 if i < 6 else ky_non - 4 * (ky_non // 5)
        row[f"kyori1chakukaisu{i}"] = val

    j5_non = j5_total - j5_wins
    row["jyo5chakukaisu1"] = j5_wins
    for i in range(2, 7):
        val = j5_non // 5 if i < 6 else j5_non - 4 * (j5_non // 5)
        row[f"jyo5chakukaisu{i}"] = val

    return row


def _make_entry(
    n: int = 1,
    kisyu_codes: list[str] | None = None,
    race_date: str = "2024-06-01",
) -> pd.DataFrame:
    if kisyu_codes is None:
        kisyu_codes = ["JK001"] * n
    return pd.DataFrame(
        {
            "race_id": ["r1"] * n,
            "umaban": list(range(1, n + 1)),
            "kisyucode": kisyu_codes,
            "race_date": [race_date] * n,
        }
    )


# ===========================================================================
# Tests: _smoothed_wr
# ===========================================================================


class TestSmoothedWr:
    """Beta(1,10) smoothing: (wins+1)/(total+11)"""

    def test_basic(self):
        """wins=5, total=50 -> (5+1)/(50+11) = 6/61"""
        result = JockeyContextFeatures._smoothed_wr(5, 50)
        assert abs(result - 6 / 61) < 1e-10

    def test_zero_total(self):
        """total=0 -> NaN"""
        result = JockeyContextFeatures._smoothed_wr(3, 0)
        assert np.isnan(result)

    def test_zero_wins(self):
        """wins=0, total=100 -> (0+1)/(100+11) = 1/111"""
        result = JockeyContextFeatures._smoothed_wr(0, 100)
        assert abs(result - 1 / 111) < 1e-10


# ===========================================================================
# Tests: jockey_wr_overall
# ===========================================================================


class TestJockeyWrOverall:
    """jockey_wr_overall: SetYear < race_year の最新年を使用"""

    def test_wr_overall(self):
        """wins=50, total=400 -> (50+1)/(400+11) = 51/411"""
        stats = pd.DataFrame([_make_jockey_stats_row(wins=50, total_starts=400)])
        store = _make_store(stats)
        feat = JockeyContextFeatures(store)
        entry = _make_entry()
        result = feat.compute(entry)
        expected = (50 + 1) / (400 + 11)
        assert abs(result["jockey_wr_overall"].iloc[0] - expected) < 1e-10

    def test_latest_year_used(self):
        """最新年のみ使用: 2022と2023の両方がある場合2023を使用"""
        stats = pd.DataFrame(
            [
                _make_jockey_stats_row(setyear=2022, wins=10, total_starts=100),
                _make_jockey_stats_row(setyear=2023, wins=50, total_starts=400),
            ]
        )
        store = _make_store(stats)
        feat = JockeyContextFeatures(store)
        entry = _make_entry(race_date="2024-06-01")
        result = feat.compute(entry)
        expected = (50 + 1) / (400 + 11)
        assert abs(result["jockey_wr_overall"].iloc[0] - expected) < 1e-10


# ===========================================================================
# Tests: jockey_prize_log
# ===========================================================================


class TestJockeyPrizeLog:
    """jockey_prize_log: log1p(honsyokinheichi)"""

    def test_prize_log(self):
        """honsyokinheichi=100000 -> log1p(100000)"""
        stats = pd.DataFrame([_make_jockey_stats_row(honsyokinheichi=100000.0)])
        store = _make_store(stats)
        feat = JockeyContextFeatures(store)
        entry = _make_entry()
        result = feat.compute(entry)
        expected = float(np.log1p(100000.0))
        assert abs(result["jockey_prize_log"].iloc[0] - expected) < 1e-10

    def test_prize_log_zero(self):
        """honsyokinheichi=0 -> log1p(0) = 0.0"""
        stats = pd.DataFrame([_make_jockey_stats_row(honsyokinheichi=0.0)])
        store = _make_store(stats)
        feat = JockeyContextFeatures(store)
        entry = _make_entry()
        result = feat.compute(entry)
        expected = float(np.log1p(0.0))
        assert abs(result["jockey_prize_log"].iloc[0] - expected) < 1e-10


# ===========================================================================
# Tests: year boundary
# ===========================================================================


class TestYearBoundary:
    """SetYear < race_year: 2024年のレースは2023年以前の成績のみ使用"""

    def test_2024_race_uses_2023_stats(self):
        """race_year=2024 → setyear=2023使用、setyear=2024は除外"""
        stats = pd.DataFrame(
            [
                _make_jockey_stats_row(setyear=2023, wins=30, total_starts=200),
                _make_jockey_stats_row(setyear=2024, wins=60, total_starts=400),
            ]
        )
        store = _make_store(stats)
        feat = JockeyContextFeatures(store)
        entry = _make_entry(race_date="2024-06-01")
        result = feat.compute(entry)
        expected = (30 + 1) / (200 + 11)
        assert abs(result["jockey_wr_overall"].iloc[0] - expected) < 1e-10

    def test_2023_race_uses_2022_stats(self):
        """race_year=2023 → setyear=2022使用"""
        stats = pd.DataFrame(
            [
                _make_jockey_stats_row(setyear=2022, wins=20, total_starts=150),
                _make_jockey_stats_row(setyear=2023, wins=40, total_starts=300),
            ]
        )
        store = _make_store(stats)
        feat = JockeyContextFeatures(store)
        entry = _make_entry(race_date="2023-06-01")
        result = feat.compute(entry)
        expected = (20 + 1) / (150 + 11)
        assert abs(result["jockey_wr_overall"].iloc[0] - expected) < 1e-10


# ===========================================================================
# Tests: no stats / missing jockey
# ===========================================================================


class TestNoStats:
    """空のstats → 全てNaN"""

    def test_empty_stats(self):
        stats = pd.DataFrame()
        store = _make_store(stats)
        feat = JockeyContextFeatures(store)
        entry = _make_entry()
        result = feat.compute(entry)
        for col in FEATURE_COLS:
            assert np.isnan(result[col].iloc[0]), f"Expected NaN for {col}"


class TestMissingJockey:
    """kisyucode not in stats → NaN"""

    def test_missing_jockey(self):
        stats = pd.DataFrame([_make_jockey_stats_row(kisyucode="JK999")])
        store = _make_store(stats)
        feat = JockeyContextFeatures(store)
        entry = _make_entry(kisyu_codes=["JK001"])
        result = feat.compute(entry)
        for col in FEATURE_COLS:
            assert np.isnan(result[col].iloc[0]), f"Expected NaN for {col}"


# ===========================================================================
# Tests: distance and venue win rates
# ===========================================================================


class TestDistanceAndVenue:
    """jockey_wr_distance, jockey_wr_venue の計算"""

    def test_wr_distance(self):
        """ky1_wins=20, ky1_total=150 -> (20+1)/(150+11) = 21/161"""
        stats = pd.DataFrame([_make_jockey_stats_row(ky1_wins=20, ky1_total=150)])
        store = _make_store(stats)
        feat = JockeyContextFeatures(store)
        entry = _make_entry()
        result = feat.compute(entry)
        expected = (20 + 1) / (150 + 11)
        assert abs(result["jockey_wr_distance"].iloc[0] - expected) < 1e-10

    def test_wr_venue(self):
        """j5_wins=10, j5_total=80 -> (10+1)/(80+11) = 11/91"""
        stats = pd.DataFrame([_make_jockey_stats_row(j5_wins=10, j5_total=80)])
        store = _make_store(stats)
        feat = JockeyContextFeatures(store)
        entry = _make_entry()
        result = feat.compute(entry)
        expected = (10 + 1) / (80 + 11)
        assert abs(result["jockey_wr_venue"].iloc[0] - expected) < 1e-10


# ===========================================================================
# Tests: result structure
# ===========================================================================


class TestResultStructure:
    """出力DataFrameの構造検証"""

    def test_columns(self):
        """結果に race_id, umaban + FEATURE_COLS が含まれる"""
        stats = pd.DataFrame([_make_jockey_stats_row()])
        store = _make_store(stats)
        feat = JockeyContextFeatures(store)
        entry = _make_entry()
        result = feat.compute(entry)
        expected_cols = ["race_id", "umaban"] + FEATURE_COLS
        assert list(result.columns) == expected_cols

    def test_multiple_jockeys(self):
        """複数騎手のエントリーでそれぞれ正しい値が返る"""
        stats = pd.DataFrame(
            [
                _make_jockey_stats_row(kisyucode="JK001", wins=50, total_starts=400),
                _make_jockey_stats_row(kisyucode="JK002", wins=10, total_starts=100),
            ]
        )
        store = _make_store(stats)
        feat = JockeyContextFeatures(store)
        entry = _make_entry(n=2, kisyu_codes=["JK001", "JK002"])
        result = feat.compute(entry)
        assert len(result) == 2
        expected_k1 = (50 + 1) / (400 + 11)
        expected_k2 = (10 + 1) / (100 + 11)
        assert abs(result["jockey_wr_overall"].iloc[0] - expected_k1) < 1e-10
        assert abs(result["jockey_wr_overall"].iloc[1] - expected_k2) < 1e-10
