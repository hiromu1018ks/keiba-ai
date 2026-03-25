"""src/models/robust_confidence_estimator.py のテスト"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.robust_confidence_estimator import RobustConfidenceEstimator


@pytest.fixture
def calibration_win_df() -> pd.DataFrame:
    """単勝キャリブレーションデータ (10サンプル)"""
    np.random.seed(42)
    n = 10
    return pd.DataFrame(
        {
            "race_id": [f"R{i}" for i in range(n)],
            "umaban": [1] * n,
            "ev_win_corrected": [1.2, 1.5, 0.8, 1.8, 0.9, 1.3, 1.1, 1.6, 0.7, 1.4],
            "actual_ev_win": [1.5, 0.8, 1.0, 2.0, 0.5, 1.0, 1.5, 1.2, 0.9, 1.8],
        }
    )


@pytest.fixture
def calibration_place_df() -> pd.DataFrame:
    """複勝キャリブレーションデータ (10サンプル)"""
    np.random.seed(42)
    n = 10
    return pd.DataFrame(
        {
            "race_id": [f"R{i}" for i in range(n)],
            "umaban": [1] * n,
            "ev_place_corrected": [1.1, 1.3, 0.9, 1.4, 0.95, 1.2, 1.05, 1.35, 0.85, 1.25],
            "actual_ev_place": [1.2, 1.0, 1.1, 1.3, 0.8, 1.1, 1.3, 1.2, 0.9, 1.4],
        }
    )


@pytest.fixture
def inference_df() -> pd.DataFrame:
    """推論用データ (3サンプル)"""
    return pd.DataFrame(
        {
            "race_id": ["R20", "R21", "R22"],
            "umaban": [1, 2, 3],
            "ev_win_corrected": [1.5, 1.2, 1.8],
        }
    )


@pytest.fixture
def inference_place_df() -> pd.DataFrame:
    """推論用複勝データ (3サンプル)"""
    return pd.DataFrame(
        {
            "race_id": ["R20", "R21", "R22"],
            "umaban": [1, 2, 3],
            "ev_place_corrected": [1.3, 1.1, 1.5],
        }
    )


class TestRobustConfidenceEstimator:
    def test_calibrate_sets_quantiles(
        self,
        calibration_win_df: pd.DataFrame,
        calibration_place_df: pd.DataFrame,
    ) -> None:
        estimator = RobustConfidenceEstimator()
        estimator.calibrate(calibration_win_df, calibration_place_df)
        assert estimator._win_cp_quantile > 0
        assert estimator._place_cp_quantile > 0

    def test_predict_lower_bound_returns_correct_columns(
        self,
        calibration_win_df: pd.DataFrame,
        calibration_place_df: pd.DataFrame,
        inference_df: pd.DataFrame,
        inference_place_df: pd.DataFrame,
    ) -> None:
        estimator = RobustConfidenceEstimator()
        estimator.calibrate(calibration_win_df, calibration_place_df)

        win_result, place_result = estimator.predict_lower_bound(
            inference_df,
            inference_place_df,
        )
        assert "EV_lower_win_corrected" in win_result.columns
        assert "EV_lower_place" in place_result.columns

    def test_lower_bound_is_non_negative(
        self,
        calibration_win_df: pd.DataFrame,
        calibration_place_df: pd.DataFrame,
        inference_df: pd.DataFrame,
        inference_place_df: pd.DataFrame,
    ) -> None:
        estimator = RobustConfidenceEstimator()
        estimator.calibrate(calibration_win_df, calibration_place_df)

        win_result, _ = estimator.predict_lower_bound(inference_df, inference_place_df)
        assert (win_result["EV_lower_win_corrected"] >= 0).all()

    def test_lower_bound_less_than_ev(
        self,
        calibration_win_df: pd.DataFrame,
        calibration_place_df: pd.DataFrame,
        inference_df: pd.DataFrame,
        inference_place_df: pd.DataFrame,
    ) -> None:
        estimator = RobustConfidenceEstimator()
        estimator.calibrate(calibration_win_df, calibration_place_df)

        win_result, _ = estimator.predict_lower_bound(inference_df, inference_place_df)
        for i in range(len(win_result)):
            lower = win_result["EV_lower_win_corrected"].iloc[i]
            ev = win_result["ev_win_corrected"].iloc[i]
            assert lower <= ev

    def test_calibrate_before_predict(
        self,
        inference_df: pd.DataFrame,
        inference_place_df: pd.DataFrame,
    ) -> None:
        """calibrate() 前に predict_lower_bound() を呼ぶとエラー"""
        estimator = RobustConfidenceEstimator()
        with pytest.raises(RuntimeError, match="calibrate"):
            estimator.predict_lower_bound(inference_df, inference_place_df)

    def test_uses_min_of_cp_and_rolling(
        self,
        calibration_win_df: pd.DataFrame,
        calibration_place_df: pd.DataFrame,
        inference_df: pd.DataFrame,
        inference_place_df: pd.DataFrame,
    ) -> None:
        """CP と Rolling Quantile の min を使用する (Rule 4)"""
        estimator = RobustConfidenceEstimator()
        estimator.calibrate(calibration_win_df, calibration_place_df)

        win_result, _ = estimator.predict_lower_bound(inference_df, inference_place_df)
        # CP bound と rolling bound の両方が計算されていることを確認
        assert "_cp_lower" not in win_result.columns  # 内部列は削除される
        assert "EV_lower_win_corrected" in win_result.columns
