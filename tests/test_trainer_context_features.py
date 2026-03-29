"""test_trainer_context_features.py — TrainerContextFeatures の単体テスト"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from features.trainer_context_features import FEATURE_COLS, TrainerContextFeatures


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_repo(trainer_stats_df: pd.DataFrame) -> MagicMock:
    repo = MagicMock()
    repo.load_trainer_stats.return_value = trainer_stats_df
    return repo


def _make_trainer_stats_row(
    chokyosicode: str = "TR001",
    setyear: int = 2023,
    wins: int = 40,
    total_starts: int = 300,
    ky1_wins: int = 15,
    ky1_total: int = 100,
    j5_wins: int = 8,
    j5_total: int = 60,
    honsyokinheichi: float = 80000.0,
) -> dict:
    """Build one row of x_CHOKYO_SEISEKI data.

    Same column naming convention as jockey but with chokyosicode.
    """
    non_wins = total_starts - wins
    rest = non_wins
    row: dict = {
        "chokyosicode": chokyosicode,
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
    chokyosi_codes: list[str] | None = None,
    race_date: str = "2024-06-01",
) -> pd.DataFrame:
    if chokyosi_codes is None:
        chokyosi_codes = ["TR001"] * n
    return pd.DataFrame(
        {
            "race_id": ["r1"] * n,
            "umaban": list(range(1, n + 1)),
            "chokyosi_code": chokyosi_codes,
            "race_date": [race_date] * n,
        }
    )


# ===========================================================================
# Tests: _smoothed_wr
# ===========================================================================


class TestSmoothedWr:
    """Beta(1,10) smoothing: (wins+1)/(total+11)"""

    def test_basic(self):
        result = TrainerContextFeatures._smoothed_wr(5, 50)
        assert abs(result - 6 / 61) < 1e-10

    def test_zero_total(self):
        result = TrainerContextFeatures._smoothed_wr(3, 0)
        assert np.isnan(result)


# ===========================================================================
# Tests: trainer_wr_overall
# ===========================================================================


class TestTrainerWrOverall:
    """trainer_wr_overall: SetYear < race_year の最新年を使用"""

    def test_wr_overall(self):
        """wins=40, total=300 -> (40+1)/(300+11) = 41/311"""
        stats = pd.DataFrame([_make_trainer_stats_row(wins=40, total_starts=300)])
        repo = _make_repo(stats)
        feat = TrainerContextFeatures(repo)
        entry = _make_entry()
        result = feat.compute(entry)
        expected = (40 + 1) / (300 + 11)
        assert abs(result["trainer_wr_overall"].iloc[0] - expected) < 1e-10

    def test_latest_year_used(self):
        stats = pd.DataFrame([
            _make_trainer_stats_row(setyear=2022, wins=10, total_starts=100),
            _make_trainer_stats_row(setyear=2023, wins=40, total_starts=300),
        ])
        repo = _make_repo(stats)
        feat = TrainerContextFeatures(repo)
        entry = _make_entry(race_date="2024-06-01")
        result = feat.compute(entry)
        expected = (40 + 1) / (300 + 11)
        assert abs(result["trainer_wr_overall"].iloc[0] - expected) < 1e-10


# ===========================================================================
# Tests: trainer_prize_log
# ===========================================================================


class TestTrainerPrizeLog:
    """trainer_prize_log: log1p(honsyokinheichi)"""

    def test_prize_log(self):
        stats = pd.DataFrame([_make_trainer_stats_row(honsyokinheichi=80000.0)])
        repo = _make_repo(stats)
        feat = TrainerContextFeatures(repo)
        entry = _make_entry()
        result = feat.compute(entry)
        expected = float(np.log1p(80000.0))
        assert abs(result["trainer_prize_log"].iloc[0] - expected) < 1e-10

    def test_prize_log_zero(self):
        stats = pd.DataFrame([_make_trainer_stats_row(honsyokinheichi=0.0)])
        repo = _make_repo(stats)
        feat = TrainerContextFeatures(repo)
        entry = _make_entry()
        result = feat.compute(entry)
        expected = float(np.log1p(0.0))
        assert abs(result["trainer_prize_log"].iloc[0] - expected) < 1e-10


# ===========================================================================
# Tests: year boundary
# ===========================================================================


class TestYearBoundary:
    def test_2024_race_uses_2023_stats(self):
        stats = pd.DataFrame([
            _make_trainer_stats_row(setyear=2023, wins=25, total_starts=200),
            _make_trainer_stats_row(setyear=2024, wins=50, total_starts=400),
        ])
        repo = _make_repo(stats)
        feat = TrainerContextFeatures(repo)
        entry = _make_entry(race_date="2024-06-01")
        result = feat.compute(entry)
        expected = (25 + 1) / (200 + 11)
        assert abs(result["trainer_wr_overall"].iloc[0] - expected) < 1e-10


# ===========================================================================
# Tests: no stats / missing trainer
# ===========================================================================


class TestNoStats:
    def test_empty_stats(self):
        stats = pd.DataFrame()
        repo = _make_repo(stats)
        feat = TrainerContextFeatures(repo)
        entry = _make_entry()
        result = feat.compute(entry)
        for col in FEATURE_COLS:
            assert np.isnan(result[col].iloc[0]), f"Expected NaN for {col}"


class TestMissingTrainer:
    def test_missing_trainer(self):
        stats = pd.DataFrame([_make_trainer_stats_row(chokyosicode="TR999")])
        repo = _make_repo(stats)
        feat = TrainerContextFeatures(repo)
        entry = _make_entry(chokyosi_codes=["TR001"])
        result = feat.compute(entry)
        for col in FEATURE_COLS:
            assert np.isnan(result[col].iloc[0]), f"Expected NaN for {col}"


# ===========================================================================
# Tests: distance and venue
# ===========================================================================


class TestDistanceAndVenue:
    def test_wr_distance(self):
        stats = pd.DataFrame([_make_trainer_stats_row(ky1_wins=15, ky1_total=100)])
        repo = _make_repo(stats)
        feat = TrainerContextFeatures(repo)
        entry = _make_entry()
        result = feat.compute(entry)
        expected = (15 + 1) / (100 + 11)
        assert abs(result["trainer_wr_distance"].iloc[0] - expected) < 1e-10

    def test_wr_venue(self):
        stats = pd.DataFrame([_make_trainer_stats_row(j5_wins=8, j5_total=60)])
        repo = _make_repo(stats)
        feat = TrainerContextFeatures(repo)
        entry = _make_entry()
        result = feat.compute(entry)
        expected = (8 + 1) / (60 + 11)
        assert abs(result["trainer_wr_venue"].iloc[0] - expected) < 1e-10


# ===========================================================================
# Tests: result structure
# ===========================================================================


class TestResultStructure:
    def test_columns(self):
        stats = pd.DataFrame([_make_trainer_stats_row()])
        repo = _make_repo(stats)
        feat = TrainerContextFeatures(repo)
        entry = _make_entry()
        result = feat.compute(entry)
        expected_cols = ["race_id", "umaban"] + FEATURE_COLS
        assert list(result.columns) == expected_cols

    def test_multiple_trainers(self):
        stats = pd.DataFrame([
            _make_trainer_stats_row(chokyosicode="TR001", wins=40, total_starts=300),
            _make_trainer_stats_row(chokyosicode="TR002", wins=8, total_starts=80),
        ])
        repo = _make_repo(stats)
        feat = TrainerContextFeatures(repo)
        entry = _make_entry(n=2, chokyosi_codes=["TR001", "TR002"])
        result = feat.compute(entry)
        assert len(result) == 2
        expected_t1 = (40 + 1) / (300 + 11)
        expected_t2 = (8 + 1) / (80 + 11)
        assert abs(result["trainer_wr_overall"].iloc[0] - expected_t1) < 1e-10
        assert abs(result["trainer_wr_overall"].iloc[1] - expected_t2) < 1e-10
