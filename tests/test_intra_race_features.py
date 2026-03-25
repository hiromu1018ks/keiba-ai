"""src/features/intra_race_features.py のテスト"""

import pandas as pd
import pytest

from features.intra_race_features import compute_intra_race_features


@pytest.fixture
def merged_df() -> pd.DataFrame:
    """feature_engine.build_all() 出力を模擬したマージ済みDataFrame"""
    return pd.DataFrame({
        "race_id": ["2024032405030208"] * 5,
        "umaban": [1, 2, 3, 4, 5],
        "win_odds": [2.0, 5.0, 3.0, 10.0, 8.0],
        "ninki": [1, 3, 2, 5, 4],
        "ba_taijyu": [480.0, 470.0, 490.0, 460.0, 500.0],
        "popularity_rank": [1, 3, 2, 5, 4],
    })


@pytest.fixture
def multi_race_df() -> pd.DataFrame:
    """複数レースを含むDataFrame（グループ処理の確認用）"""
    return pd.DataFrame({
        "race_id": ["R1"] * 3 + ["R2"] * 4,
        "umaban": [1, 2, 3, 1, 2, 3, 4],
        "win_odds": [2.0, 5.0, 8.0, 3.0, 4.0, 10.0, 15.0],
        "ninki": [1, 2, 3, 1, 2, 3, 4],
        "ba_taijyu": [480.0, 470.0, 490.0, 485.0, 475.0, 465.0, 495.0],
        "popularity_rank": [1, 2, 3, 1, 2, 3, 4],
    })


class TestIntraRaceFeatures:
    def test_weight_diff_from_mean(self, merged_df: pd.DataFrame):
        """馬体重とレース平均との差を計算"""
        result = compute_intra_race_features(merged_df)
        # mean = (480+470+490+460+500)/5 = 480.0
        expected = [0.0, -10.0, 10.0, -20.0, 20.0]
        for i, exp in enumerate(expected):
            assert abs(result.iloc[i]["weight_diff_from_mean"] - exp) < 1e-10

    def test_weight_diff_from_mean_multi_race(self, multi_race_df: pd.DataFrame):
        """複数レースでそれぞれ独立に平均を計算"""
        result = compute_intra_race_features(multi_race_df)
        r1_rows = result[result["race_id"] == "R1"]
        r2_rows = result[result["race_id"] == "R2"]
        # R1: mean = (480+470+490)/3 = 480.0
        assert abs(r1_rows.iloc[0]["weight_diff_from_mean"] - 0.0) < 1e-10
        assert abs(r1_rows.iloc[1]["weight_diff_from_mean"] - (-10.0)) < 1e-10
        # R2: mean = (485+475+465+495)/4 = 480.0
        assert abs(r2_rows.iloc[0]["weight_diff_from_mean"] - 5.0) < 1e-10

    def test_odds_rank(self, merged_df: pd.DataFrame):
        """単勝オッズのレース内順位（低い順=1位）"""
        result = compute_intra_race_features(merged_df)
        # win_odds: 2.0(rank1), 5.0(rank3), 3.0(rank2), 10.0(rank5), 8.0(rank4)
        odds_rank_map = {1: 1, 2: 3, 3: 2, 4: 5, 5: 4}
        for _, row in result.iterrows():
            umaban = int(row["umaban"])
            assert row["odds_rank"] == odds_rank_map[umaban]

    def test_odds_rank_multi_race(self, multi_race_df: pd.DataFrame):
        """複数レースでそれぞれ独立に順位を計算"""
        result = compute_intra_race_features(multi_race_df)
        r1_rows = result[result["race_id"] == "R1"]
        # R1 odds: 2.0(1), 5.0(2), 8.0(3)
        assert r1_rows.iloc[0]["odds_rank"] == 1
        assert r1_rows.iloc[1]["odds_rank"] == 2
        assert r1_rows.iloc[2]["odds_rank"] == 3

    def test_preserves_existing_columns(self, merged_df: pd.DataFrame):
        """既存列を保持する"""
        result = compute_intra_race_features(merged_df)
        assert "race_id" in result.columns
        assert "umaban" in result.columns
        assert "win_odds" in result.columns

    def test_returns_new_dataframe(self, merged_df: pd.DataFrame):
        """入力DataFrameを変更しない"""
        original_cols = set(merged_df.columns)
        _ = compute_intra_race_features(merged_df)
        assert set(merged_df.columns) == original_cols
