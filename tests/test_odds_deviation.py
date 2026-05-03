"""ODDS-01: odds deviation features + EV interval tests"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestOddsDeviationFeatures:
    """compute_odds_deviation_features() の単体テスト"""

    def test_returns_deviation_rank_and_zscore(self) -> None:
        """odds_to_ability_ratioとrace_idからdeviation_rank, deviation_zscoreが計算される"""
        from features.odds_deviation_features import compute_odds_deviation_features

        df = pd.DataFrame(
            {
                "race_id": ["R1", "R1", "R1", "R2", "R2"],
                "odds_to_ability_ratio": [1.5, 0.8, 2.0, 1.0, 1.2],
            }
        )
        result = compute_odds_deviation_features(df)

        assert "deviation_rank" in result.columns
        assert "deviation_zscore" in result.columns
        assert len(result) == 5

    def test_single_horse_race_edge_case(self) -> None:
        """1頭立てレース: deviation_rank=1.0, deviation_zscore=NaN (std=0)"""
        from features.odds_deviation_features import compute_odds_deviation_features

        df = pd.DataFrame(
            {
                "race_id": ["R1"],
                "odds_to_ability_ratio": [1.5],
            }
        )
        result = compute_odds_deviation_features(df)

        assert result["deviation_rank"].iloc[0] == pytest.approx(1.0)
        assert pd.isna(result["deviation_zscore"].iloc[0])

    def test_missing_odds_to_ability_ratio(self) -> None:
        """odds_to_ability_ratio列がない場合: NaN列が追加され、例外は発生しない"""
        from features.odds_deviation_features import compute_odds_deviation_features

        df = pd.DataFrame(
            {
                "race_id": ["R1", "R1"],
                "umaban": [1, 2],
            }
        )
        result = compute_odds_deviation_features(df)

        assert "deviation_rank" in result.columns
        assert "deviation_zscore" in result.columns
        assert result["deviation_rank"].isna().all()
        assert result["deviation_zscore"].isna().all()

    def test_zscore_clipped_to_range(self) -> None:
        """deviation_zscoreが[-5.0, 5.0]にクリップされる"""
        from features.odds_deviation_features import compute_odds_deviation_features

        # 極端な比率のレースで z-score が大きくなるケース
        df = pd.DataFrame(
            {
                "race_id": ["R1"] * 10,
                "odds_to_ability_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 10.0],
            }
        )
        result = compute_odds_deviation_features(df)

        assert result["deviation_zscore"].min() >= -5.0
        assert result["deviation_zscore"].max() <= 5.0

    def test_rank_ascending_false(self) -> None:
        """deviation_rankは降順(ratio大=過小評価=高いrank)"""
        from features.odds_deviation_features import compute_odds_deviation_features

        df = pd.DataFrame(
            {
                "race_id": ["R1", "R1", "R1"],
                "odds_to_ability_ratio": [1.0, 2.0, 0.5],
            }
        )
        result = compute_odds_deviation_features(df)

        # ratio 2.0 (最高) -> rank 1, ratio 1.0 -> rank 2, ratio 0.5 (最低) -> rank 3
        ranks = result["deviation_rank"].values
        assert ranks[0] == pytest.approx(2.0)  # ratio=1.0
        assert ranks[1] == pytest.approx(1.0)  # ratio=2.0 (highest -> rank 1)
        assert ranks[2] == pytest.approx(3.0)  # ratio=0.5 (lowest -> rank 3)

    def test_numerical_consistency_known_values(self) -> None:
        """3頭レースで既知の比率から正確なrankとzscoreを検証"""
        from features.odds_deviation_features import compute_odds_deviation_features

        # race R1: 3頭, ratios = [1.0, 2.0, 0.5]
        # mean = 1.1667, std = 0.7638
        # rank(ascending=False): 2.0->1, 1.0->2, 0.5->3
        # zscore(2.0) = (2.0 - 1.1667) / 0.7638 = 1.0911
        # zscore(1.0) = (1.0 - 1.1667) / 0.7638 = -0.2182
        # zscore(0.5) = (0.5 - 1.1667) / 0.7638 = -0.8729
        df = pd.DataFrame(
            {
                "race_id": ["R1", "R1", "R1"],
                "odds_to_ability_ratio": [1.0, 2.0, 0.5],
            }
        )
        result = compute_odds_deviation_features(df)

        # Rank verification
        assert result["deviation_rank"].iloc[0] == pytest.approx(2.0)
        assert result["deviation_rank"].iloc[1] == pytest.approx(1.0)
        assert result["deviation_rank"].iloc[2] == pytest.approx(3.0)

        # Z-score verification
        ratios = np.array([1.0, 2.0, 0.5])
        mean = ratios.mean()
        std = ratios.std(ddof=1)
        expected_zscores = (ratios - mean) / std

        for i in range(3):
            assert result["deviation_zscore"].iloc[i] == pytest.approx(
                expected_zscores[i], rel=1e-6
            )
