"""test_sire_features.py — SireFeatures の単体テスト"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.sire_features import SireFeatures, _beta_smooth


class TestBetaSmooth:
    def test_basic(self):
        assert _beta_smooth(6, 50) == 7 / 61

    def test_zero_starts(self):
        """starts=0 -> prior only"""
        assert _beta_smooth(0, 0) == 1 / 11


class TestSireWr:
    """sire_wr: Beta 平滑化全体勝率"""

    def test_sire_wr_beta_smoothed(self):
        """sire_wr が Beta 平滑化勝率を返す"""
        sire_stats = pd.DataFrame({
            "sire_id": ["SIRE_A", "SIRE_A"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-06-01"]),
            "sire_starts": [0, 50],
            "sire_wins": [0, 6],
            "sire_places": [0, 15],
            "sire_turf_starts": [0, 30], "sire_turf_wins": [0, 4],
            "sire_dirt_starts": [0, 20], "sire_dirt_wins": [0, 2],
            "sire_short_starts": [0, 25], "sire_short_wins": [0, 4],
            "sire_long_starts": [0, 25], "sire_long_wins": [0, 2],
            "sire_prize_total": [0.0, 500000.0],
        })

        feat = SireFeatures(sire_stats)
        # 2024-06-01 時点: sire_starts=50, sire_wins=6
        row = feat.compute(sire_id="SIRE_A", race_date="2024-06-01",
                           surface="turf", kyori=1600)
        # Beta(1+6, 1+10+50-6) = 7/61 ≈ 0.115
        assert abs(row["sire_wr"] - 7 / 61) < 0.001

    def test_sire_surface_wr_turf(self):
        """surface=turf → sire_turf_wr = (1+4)/(1+10+30) = 5/41"""
        sire_stats = pd.DataFrame({
            "sire_id": ["SIRE_A", "SIRE_A"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-06-01"]),
            "sire_starts": [0, 50],
            "sire_wins": [0, 6],
            "sire_places": [0, 15],
            "sire_turf_starts": [0, 30], "sire_turf_wins": [0, 4],
            "sire_dirt_starts": [0, 20], "sire_dirt_wins": [0, 2],
            "sire_short_starts": [0, 25], "sire_short_wins": [0, 4],
            "sire_long_starts": [0, 25], "sire_long_wins": [0, 2],
            "sire_prize_total": [0.0, 500000.0],
        })

        feat = SireFeatures(sire_stats)
        row = feat.compute(sire_id="SIRE_A", race_date="2024-06-01",
                           surface="turf", kyori=1600)
        # _beta_smooth(4, 30) = (1+4)/(1+10+30) = 5/41
        assert abs(row["sire_surface_wr"] - 5 / 41) < 0.001

    def test_sire_distance_wr_short(self):
        """kyori=1600 (short) → sire_short_wr = (1+4)/(1+10+25) = 5/36"""
        sire_stats = pd.DataFrame({
            "sire_id": ["SIRE_A", "SIRE_A"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-06-01"]),
            "sire_starts": [0, 50],
            "sire_wins": [0, 6],
            "sire_places": [0, 15],
            "sire_turf_starts": [0, 30], "sire_turf_wins": [0, 4],
            "sire_dirt_starts": [0, 20], "sire_dirt_wins": [0, 2],
            "sire_short_starts": [0, 25], "sire_short_wins": [0, 4],
            "sire_long_starts": [0, 25], "sire_long_wins": [0, 2],
            "sire_prize_total": [0.0, 500000.0],
        })

        feat = SireFeatures(sire_stats)
        row = feat.compute(sire_id="SIRE_A", race_date="2024-06-01",
                           surface="turf", kyori=1600)
        # _beta_smooth(4, 25) = (1+4)/(1+10+25) = 5/36
        assert abs(row["sire_distance_wr"] - 5 / 36) < 0.001

    def test_sire_distance_wr_long(self):
        """kyori=2000 (long) → sire_long_wr を使用"""
        sire_stats = pd.DataFrame({
            "sire_id": ["SIRE_A", "SIRE_A"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-06-01"]),
            "sire_starts": [0, 50],
            "sire_wins": [0, 6],
            "sire_places": [0, 15],
            "sire_turf_starts": [0, 30], "sire_turf_wins": [0, 4],
            "sire_dirt_starts": [0, 20], "sire_dirt_wins": [0, 2],
            "sire_short_starts": [0, 25], "sire_short_wins": [0, 4],
            "sire_long_starts": [0, 25], "sire_long_wins": [0, 2],
            "sire_prize_total": [0.0, 500000.0],
        })

        feat = SireFeatures(sire_stats)
        row = feat.compute(sire_id="SIRE_A", race_date="2024-06-01",
                           surface="turf", kyori=2000)
        # _beta_smooth(2, 25) = (1+2)/(1+10+25) = 3/36
        assert abs(row["sire_distance_wr"] - 3 / 36) < 0.001

    def test_sire_surface_wr_dirt(self):
        """surface=dirt → sire_dirt_wr を使用"""
        sire_stats = pd.DataFrame({
            "sire_id": ["SIRE_A", "SIRE_A"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-06-01"]),
            "sire_starts": [0, 50],
            "sire_wins": [0, 6],
            "sire_places": [0, 15],
            "sire_turf_starts": [0, 30], "sire_turf_wins": [0, 4],
            "sire_dirt_starts": [0, 20], "sire_dirt_wins": [0, 2],
            "sire_short_starts": [0, 25], "sire_short_wins": [0, 4],
            "sire_long_starts": [0, 25], "sire_long_wins": [0, 2],
            "sire_prize_total": [0.0, 500000.0],
        })

        feat = SireFeatures(sire_stats)
        row = feat.compute(sire_id="SIRE_A", race_date="2024-06-01",
                           surface="dirt", kyori=1700)
        # _beta_smooth(2, 20) = (1+2)/(1+10+20) = 3/31
        assert abs(row["sire_surface_wr"] - 3 / 31) < 0.001


class TestSirePlaceRate:
    """sire_place_rate: Beta 平滑化複勝率"""

    def test_sire_place_rate(self):
        sire_stats = pd.DataFrame({
            "sire_id": ["SIRE_A", "SIRE_A"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-06-01"]),
            "sire_starts": [0, 50],
            "sire_wins": [0, 6],
            "sire_places": [0, 15],
            "sire_turf_starts": [0, 30], "sire_turf_wins": [0, 4],
            "sire_dirt_starts": [0, 20], "sire_dirt_wins": [0, 2],
            "sire_short_starts": [0, 25], "sire_short_wins": [0, 4],
            "sire_long_starts": [0, 25], "sire_long_wins": [0, 2],
            "sire_prize_total": [0.0, 500000.0],
        })

        feat = SireFeatures(sire_stats)
        row = feat.compute(sire_id="SIRE_A", race_date="2024-06-01",
                           surface="turf", kyori=1600)
        # place_rate: (1+15)/(1+10+50) = 16/61
        assert abs(row["sire_place_rate"] - 16 / 61) < 0.001


class TestSirePrizeAvg:
    """sire_prize_avg: log1p(賞金/starts)"""

    def test_sire_prize_avg(self):
        sire_stats = pd.DataFrame({
            "sire_id": ["SIRE_A", "SIRE_A"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-06-01"]),
            "sire_starts": [0, 50],
            "sire_wins": [0, 6],
            "sire_places": [0, 15],
            "sire_turf_starts": [0, 30], "sire_turf_wins": [0, 4],
            "sire_dirt_starts": [0, 20], "sire_dirt_wins": [0, 2],
            "sire_short_starts": [0, 25], "sire_short_wins": [0, 4],
            "sire_long_starts": [0, 25], "sire_long_wins": [0, 2],
            "sire_prize_total": [0.0, 500000.0],
        })

        feat = SireFeatures(sire_stats)
        row = feat.compute(sire_id="SIRE_A", race_date="2024-06-01",
                           surface="turf", kyori=1600)
        expected = float(np.log1p(500000.0 / 50))
        assert abs(row["sire_prize_avg"] - expected) < 0.001


class TestMissingSire:
    """未知の種牡馬・エッジケース"""

    def test_missing_sire_returns_nan(self):
        """未知の種牡馬はNaNを返す"""
        feat = SireFeatures(pd.DataFrame(columns=["sire_id", "race_date"]))
        row = feat.compute(sire_id="UNKNOWN", race_date="2024-01-01",
                           surface="turf", kyori=1600)
        assert pd.isna(row["sire_wr"])

    def test_none_sire_id_returns_nan(self):
        """sire_id=None はNaNを返す"""
        sire_stats = pd.DataFrame({
            "sire_id": ["SIRE_A"],
            "race_date": pd.to_datetime(["2024-06-01"]),
            "sire_starts": [50],
            "sire_wins": [6],
            "sire_places": [15],
            "sire_turf_starts": [30], "sire_turf_wins": [4],
            "sire_dirt_starts": [20], "sire_dirt_wins": [2],
            "sire_short_starts": [25], "sire_short_wins": [4],
            "sire_long_starts": [25], "sire_long_wins": [2],
            "sire_prize_total": [500000.0],
        })
        feat = SireFeatures(sire_stats)
        row = feat.compute(sire_id=None, race_date="2024-06-01",
                           surface="turf", kyori=1600)
        assert pd.isna(row["sire_wr"])

    def test_empty_stats_returns_nan(self):
        """空の統計DataFrameはNaNを返す"""
        feat = SireFeatures(pd.DataFrame())
        row = feat.compute(sire_id="SIRE_A", race_date="2024-01-01",
                           surface="turf", kyori=1600)
        assert pd.isna(row["sire_wr"])

    def test_no_data_before_race_date_uses_prior(self):
        """レース日以前にデータがない場合、事前分布を使用"""
        sire_stats = pd.DataFrame({
            "sire_id": ["SIRE_A"],
            "race_date": pd.to_datetime("2024-12-01"),
            "sire_starts": [50],
            "sire_wins": [6],
            "sire_places": [15],
            "sire_turf_starts": [30], "sire_turf_wins": [4],
            "sire_dirt_starts": [20], "sire_dirt_wins": [2],
            "sire_short_starts": [25], "sire_short_wins": [4],
            "sire_long_starts": [25], "sire_long_wins": [2],
            "sire_prize_total": [500000.0],
        })

        feat = SireFeatures(sire_stats)
        row = feat.compute(sire_id="SIRE_A", race_date="2024-06-01",
                           surface="turf", kyori=1600)
        # Prior: _beta_smooth(0, 0) = 1/11
        assert abs(row["sire_wr"] - 1 / 11) < 0.001
        assert row["sire_prize_avg"] == 0.0

    def test_all_nan_keys_present(self):
        """未知の種牡馬ですべてのキーがNaNを返す"""
        feat = SireFeatures(pd.DataFrame(columns=["sire_id", "race_date"]))
        row = feat.compute(sire_id="UNKNOWN", race_date="2024-01-01",
                           surface="turf", kyori=1600)
        for col in ["sire_wr", "sire_place_rate", "sire_surface_wr",
                    "sire_distance_wr", "sire_prize_avg"]:
            assert pd.isna(row[col]), f"Expected NaN for {col}"


class TestPitSafety:
    """Point-in-Time 安全性: searchsorted で過去データのみ参照"""

    def test_selects_latest_before_race_date(self):
        """race_date以前の最新行を選択する"""
        sire_stats = pd.DataFrame({
            "sire_id": ["SIRE_A", "SIRE_A", "SIRE_A"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-06-01", "2024-12-01"]),
            "sire_starts": [10, 30, 60],
            "sire_wins": [1, 3, 8],
            "sire_places": [3, 9, 18],
            "sire_turf_starts": [6, 18, 36],
            "sire_turf_wins": [1, 2, 5],
            "sire_dirt_starts": [4, 12, 24],
            "sire_dirt_wins": [0, 1, 3],
            "sire_short_starts": [5, 15, 30],
            "sire_short_wins": [1, 2, 4],
            "sire_long_starts": [5, 15, 30],
            "sire_long_wins": [0, 1, 4],
            "sire_prize_total": [10000.0, 30000.0, 70000.0],
        })

        feat = SireFeatures(sire_stats)
        # 2024-09-01 の時点では 2024-06-01 の行が最新
        row = feat.compute(sire_id="SIRE_A", race_date="2024-09-01",
                           surface="turf", kyori=1400)
        # starts=30, wins=3 -> (1+3)/(11+30) = 4/41
        assert abs(row["sire_wr"] - 4 / 41) < 0.001

    def test_does_not_leak_future_data(self):
        """未来のデータを参照しない"""
        sire_stats = pd.DataFrame({
            "sire_id": ["SIRE_A", "SIRE_A"],
            "race_date": pd.to_datetime(["2024-06-01", "2024-12-01"]),
            "sire_starts": [10, 100],
            "sire_wins": [1, 20],
            "sire_places": [3, 40],
            "sire_turf_starts": [6, 60],
            "sire_turf_wins": [1, 12],
            "sire_dirt_starts": [4, 40],
            "sire_dirt_wins": [0, 8],
            "sire_short_starts": [5, 50],
            "sire_short_wins": [1, 10],
            "sire_long_starts": [5, 50],
            "sire_long_wins": [0, 10],
            "sire_prize_total": [10000.0, 100000.0],
        })

        feat = SireFeatures(sire_stats)
        # 2024-03-01 の時点ではまだデータがない -> prior
        row = feat.compute(sire_id="SIRE_A", race_date="2024-03-01",
                           surface="turf", kyori=1400)
        assert abs(row["sire_wr"] - 1 / 11) < 0.001
        # 2024-09-01 の時点では 2024-06-01 の行のみ
        row2 = feat.compute(sire_id="SIRE_A", race_date="2024-09-01",
                            surface="turf", kyori=1400)
        assert abs(row2["sire_wr"] - (1 + 1) / (11 + 10)) < 0.001
