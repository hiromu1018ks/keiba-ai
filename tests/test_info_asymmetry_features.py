"""src/features/info_asymmetry_features.py のテスト"""

import pandas as pd
import pytest

from features.info_asymmetry_features import compute_hist_features


@pytest.fixture
def historical_df() -> pd.DataFrame:
    """時系列ソート済みの履歴DataFrame（5レース分）

    race_date でソート済み。expanding().shift(1) により
    各行は自分より前のデータのみから計算される。
    """
    return pd.DataFrame({
        "race_id": ["R1", "R2", "R3", "R4", "R5"],
        "race_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03",
                                     "2024-01-04", "2024-01-05"]),
        "surface": ["turf", "turf", "dirt", "turf", "turf"],
        "distance_band": ["mile", "mile", "sprint", "mile", "mile"],
        "market_entropy": [2.5, 2.7, 2.0, 2.6, 2.8],
        "topk_hit": [1, 0, 1, 1, 0],
        "topk_roi": [1.5, -0.5, 2.0, 1.2, -0.3],
        "positive_return": [True, False, True, True, False],
        "is_winner": [1, 0, 0, 1, 0],
    })


class TestHistFeatures:
    def test_first_row_is_nan(self, historical_df: pd.DataFrame):
        """最初の行は履歴データがないため NaN"""
        result = compute_hist_features(historical_df)
        assert pd.isna(result.iloc[0]["hist_hit_rate_topk"])
        assert pd.isna(result.iloc[0]["hist_roi_topk"])
        assert pd.isna(result.iloc[0]["hist_positive_return_ratio"])
        assert pd.isna(result.iloc[0]["hist_win_rate_same_condition"])
        assert pd.isna(result.iloc[0]["hist_market_entropy_avg"])

    def test_second_row_uses_first_only(self, historical_df: pd.DataFrame):
        """2行目は1行目のデータのみから計算（未来情報なし）"""
        result = compute_hist_features(historical_df)
        assert abs(result.iloc[1]["hist_hit_rate_topk"] - 1.0) < 1e-10
        assert abs(result.iloc[1]["hist_roi_topk"] - 1.5) < 1e-10
        assert abs(result.iloc[1]["hist_positive_return_ratio"] - 1.0) < 1e-10

    def test_third_row_excludes_future(self, historical_df: pd.DataFrame):
        """3行目は1-2行目のデータのみから計算"""
        result = compute_hist_features(historical_df)
        assert abs(result.iloc[2]["hist_hit_rate_topk"] - 0.5) < 1e-10
        assert abs(result.iloc[2]["hist_roi_topk"] - 0.5) < 1e-10

    def test_no_future_leakage(self, historical_df: pd.DataFrame):
        """expanding().shift(1) により未来情報が含まれないことを検証"""
        result = compute_hist_features(historical_df)
        for i in range(len(result)):
            past = historical_df.iloc[:i]
            if len(past) == 0:
                assert pd.isna(result.iloc[i]["hist_hit_rate_topk"])
            else:
                expected_hit = past["topk_hit"].mean()
                actual_hit = result.iloc[i]["hist_hit_rate_topk"]
                assert abs(actual_hit - expected_hit) < 1e-10, (
                    f"行{i}: hist_hit_rate_topk に未来情報リークの疑い"
                )

    def test_same_condition_filtering(self, historical_df: pd.DataFrame):
        """同条件（surface + distance_band）で絞り込んで計算"""
        result = compute_hist_features(historical_df)
        # R4 (turf/mile): 同条件は R1, R2 (turf/mile)
        assert abs(result.iloc[3]["hist_win_rate_same_condition"] - 0.5) < 1e-10
        assert abs(result.iloc[3]["hist_market_entropy_avg"] - 2.6) < 1e-10

    def test_different_condition_excluded(self, historical_df: pd.DataFrame):
        """異なる条件のレースは同条件計算に含まれない"""
        result = compute_hist_features(historical_df)
        # R3 (dirt/sprint): 同条件の過去レースなし → NaN
        assert pd.isna(result.iloc[2]["hist_win_rate_same_condition"])
        assert pd.isna(result.iloc[2]["hist_market_entropy_avg"])

    def test_preserves_columns(self, historical_df: pd.DataFrame):
        """既存列を保持する"""
        result = compute_hist_features(historical_df)
        assert "race_id" in result.columns
        assert "race_date" in result.columns
        assert "surface" in result.columns
