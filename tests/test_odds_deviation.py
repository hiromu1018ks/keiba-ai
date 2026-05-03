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


class TestOddsDeviationNumericalConsistency:
    """ODDS-02: 数値的一貫性チェック (EV区間順序性、NaN率、スコア範囲)"""

    def test_ev_interval_ordering(self) -> None:
        """EV区間: lower < ev < upper (predict_interval mockデータ)"""
        from models.robust_confidence_estimator import RobustConfidenceEstimator

        np.random.seed(42)
        n = 20
        cal_win = pd.DataFrame(
            {
                "race_id": [f"R{i}" for i in range(n)],
                "ev_win_corrected": np.random.uniform(0.5, 2.0, n),
                "actual_ev_win": np.random.uniform(0.0, 3.0, n),
            }
        )
        cal_place = pd.DataFrame(
            {
                "race_id": [f"R{i}" for i in range(n)],
                "ev_place_corrected": np.random.uniform(0.5, 2.0, n),
                "actual_ev_place": np.random.uniform(0.0, 3.0, n),
            }
        )

        estimator = RobustConfidenceEstimator()
        estimator.calibrate(cal_win, cal_place)

        test_win = pd.DataFrame(
            {
                "race_id": ["T1"] * 5,
                "ev_win_corrected": [1.0, 1.5, 0.8, 2.0, 1.2],
            }
        )
        test_place = pd.DataFrame(
            {
                "race_id": ["T1"] * 5,
                "ev_place_corrected": [1.0, 1.3, 0.9, 1.6, 1.1],
            }
        )

        win_result, _ = estimator.predict_interval(test_win, test_place)
        for i in range(len(win_result)):
            lower = win_result["EV_lower_win_corrected"].iloc[i]
            ev = win_result["ev_win_corrected"].iloc[i]
            upper = win_result["EV_upper_win_corrected"].iloc[i]
            assert lower <= ev + 1e-10, f"Row {i}: lower={lower} > ev={ev}"
            assert ev <= upper + 1e-10, f"Row {i}: ev={ev} > upper={upper}"

    def test_conformal_confidence_score_range(self) -> None:
        """conformal_confidence_score は [0, inf) の範囲"""
        from models.robust_confidence_estimator import RobustConfidenceEstimator

        np.random.seed(42)
        n = 20
        cal_win = pd.DataFrame(
            {
                "race_id": [f"R{i}" for i in range(n)],
                "ev_win_corrected": np.random.uniform(0.5, 2.0, n),
                "actual_ev_win": np.random.uniform(0.0, 3.0, n),
            }
        )
        cal_place = pd.DataFrame(
            {
                "race_id": [f"R{i}" for i in range(n)],
                "ev_place_corrected": np.random.uniform(0.5, 2.0, n),
                "actual_ev_place": np.random.uniform(0.0, 3.0, n),
            }
        )

        estimator = RobustConfidenceEstimator()
        estimator.calibrate(cal_win, cal_place)

        test_win = pd.DataFrame(
            {
                "race_id": ["T1"] * 5,
                "ev_win_corrected": [1.0, 1.5, 0.8, 2.0, 1.2],
            }
        )
        test_place = pd.DataFrame(
            {
                "race_id": ["T1"] * 5,
                "ev_place_corrected": [1.0, 1.3, 0.9, 1.6, 1.1],
            }
        )

        win_result, _ = estimator.predict_interval(test_win, test_place)
        assert (win_result["conformal_confidence_score"] >= 0).all()
        assert np.isfinite(win_result["conformal_confidence_score"]).all()

    def test_deviation_features_nan_rate_bounded(self) -> None:
        """odds_to_ability_ratioが存在する場合、NaN率は0%である"""
        from features.odds_deviation_features import compute_odds_deviation_features

        df = pd.DataFrame(
            {
                "race_id": ["R1"] * 5 + ["R2"] * 5,
                "odds_to_ability_ratio": [1.0, 1.5, 2.0, 0.8, 1.2] * 2,
            }
        )
        result = compute_odds_deviation_features(df)

        # deviation_rank は NaN でないはず
        assert result["deviation_rank"].notna().all()
        # deviation_zscore は複数頭レースでは NaN でない
        assert result["deviation_zscore"].notna().all()
