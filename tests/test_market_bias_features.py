"""src/features/market_bias_features.py のテスト"""

import math

import pandas as pd
import pytest

from features.market_bias_features import compute_flb_slope, compute_market_bias


@pytest.fixture
def simple_odds_df() -> pd.DataFrame:
    """単純なオッズDataFrame（全馬均等オッズ）— 生カラム名"""
    return pd.DataFrame(
        {
            "race_id": ["R1"] * 4,
            "umaban": [1, 2, 3, 4],
            "tanodds": [4.0, 4.0, 4.0, 4.0],
        }
    )


@pytest.fixture
def skewed_odds_df() -> pd.DataFrame:
    """歪んだオッズDataFrame（人気馬+穴馬）— 生カラム名"""
    return pd.DataFrame(
        {
            "race_id": ["R1"] * 3,
            "umaban": [1, 2, 3],
            "tanodds": [2.0, 5.0, 10.0],
        }
    )


@pytest.fixture
def multi_race_df() -> pd.DataFrame:
    """複数レース — 生カラム名"""
    return pd.DataFrame(
        {
            "race_id": ["R1", "R1", "R2", "R2"],
            "umaban": [1, 2, 1, 2],
            "tanodds": [2.0, 2.0, 3.0, 3.0],
        }
    )


class TestMarketBiasFeatures:
    def test_p_market_win_adj_sums_to_one(self, simple_odds_df: pd.DataFrame):
        """正規化確率の合計が1になる"""
        result = compute_market_bias(simple_odds_df)
        p_sum = result.groupby("race_id")["p_market_win_adj"].sum()
        assert abs(p_sum.iloc[0] - 1.0) < 1e-10

    def test_p_market_win_adj_equal_odds(self, simple_odds_df: pd.DataFrame):
        """均等オッズ(4頭@4.0)の場合、各馬の確率は0.25"""
        result = compute_market_bias(simple_odds_df)
        for _, row in result.iterrows():
            assert abs(row["p_market_win_adj"] - 0.25) < 1e-10

    def test_overround(self, simple_odds_df: pd.DataFrame):
        """均等オッズの overround = 0 (sum(1/odds)=1.0)"""
        result = compute_market_bias(simple_odds_df)
        # 4頭@4.0 → sum(1/4.0) = 1.0 → overround = 0.0
        assert abs(result.iloc[0]["overround"]) < 1e-10

    def test_overround_skewed(self, skewed_odds_df: pd.DataFrame):
        """歪んだオッズの overround > 0"""
        result = compute_market_bias(skewed_odds_df)
        # sum(1/2 + 1/5 + 1/10) = 0.5 + 0.2 + 0.1 = 0.8 → overround = -0.2
        expected_overround = 0.5 + 0.2 + 0.1 - 1.0
        assert abs(result.iloc[0]["overround"] - expected_overround) < 1e-10

    def test_market_entropy_equal(self, simple_odds_df: pd.DataFrame):
        """均等確率のエントロピーは最大 (= ln(n))"""
        result = compute_market_bias(simple_odds_df)
        max_entropy = math.log(4)
        assert abs(result.iloc[0]["market_entropy"] - max_entropy) < 1e-10

    def test_market_entropy_skewed(self, skewed_odds_df: pd.DataFrame):
        """歪んだ確率のエントロピーは最大より小さい"""
        result = compute_market_bias(skewed_odds_df)
        max_entropy = math.log(3)
        assert result.iloc[0]["market_entropy"] < max_entropy

    def test_multi_race_independent(self, multi_race_df: pd.DataFrame):
        """複数レースで独立に計算される"""
        result = compute_market_bias(multi_race_df)
        for rid in ["R1", "R2"]:
            p_sum = result[result["race_id"] == rid]["p_market_win_adj"].sum()
            assert abs(p_sum - 1.0) < 1e-10

    def test_market_entropy_formula(self, skewed_odds_df: pd.DataFrame):
        """エントロピー公式: H = -sum(p_i * ln(p_i))"""
        result = compute_market_bias(skewed_odds_df)
        p_values = result["p_market_win_adj"].values
        expected = -sum(p * math.log(p) for p in p_values)
        assert abs(result.iloc[0]["market_entropy"] - expected) < 1e-10

    def test_preserves_existing_columns(self, skewed_odds_df: pd.DataFrame):
        """既存列を保持する"""
        result = compute_market_bias(skewed_odds_df)
        assert "race_id" in result.columns
        assert "umaban" in result.columns
        assert "tanodds" in result.columns


class TestComputeOddsShape:
    def test_returns_dataframe_with_two_columns(self) -> None:
        """odds_skewness と implied_prob_hhi の2列を持つ DataFrame を返す"""
        df = pd.DataFrame({"race_id": ["R1"] * 3, "umaban": [1, 2, 3],
            "tanodds": [2.0, 5.0, 10.0]})
        result = compute_flb_slope(df)
        assert isinstance(result, pd.DataFrame)
        assert "odds_skewness" in result.columns
        assert "implied_prob_hhi" in result.columns

    def test_equal_odds_zero_skewness(self) -> None:
        """均等オッズの歪度は0に近い"""
        df = pd.DataFrame({"race_id": ["R1"] * 4, "umaban": [1, 2, 3, 4],
            "tanodds": [4.0, 4.0, 4.0, 4.0]})
        result = compute_flb_slope(df)
        assert abs(result["odds_skewness"].iloc[0]) < 1e-10

    def test_skewed_odds_positive_skewness(self) -> None:
        """オッズのばらつきが大きいと正の歪度になる"""
        df = pd.DataFrame({"race_id": ["R1"] * 3, "umaban": [1, 2, 3],
            "tanodds": [2.0, 5.0, 100.0]})
        result = compute_flb_slope(df)
        assert result["odds_skewness"].iloc[0] > 0.0

    def test_hhi_dominant_favorite(self) -> None:
        """圧倒的1番人気のHHIが高い"""
        df_dom = pd.DataFrame({"race_id": ["R1"] * 3, "umaban": [1, 2, 3],
            "tanodds": [1.1, 20.0, 50.0]})
        df_eq = pd.DataFrame({"race_id": ["R2"] * 3, "umaban": [1, 2, 3],
            "tanodds": [5.0, 5.0, 5.0]})
        assert compute_flb_slope(df_dom)["implied_prob_hhi"].iloc[0] > \
               compute_flb_slope(df_eq)["implied_prob_hhi"].iloc[0]

    def test_missing_tanodds_returns_zeros(self) -> None:
        df = pd.DataFrame({"race_id": ["R1", "R1"], "umaban": [1, 2]})
        result = compute_flb_slope(df)
        assert (result["odds_skewness"] == 0.0).all()

    def test_multi_race_independent(self) -> None:
        df = pd.DataFrame({"race_id": ["R1"]*3 + ["R2"]*3, "umaban": [1,2,3,1,2,3],
            "tanodds": [2.0, 5.0, 10.0, 3.0, 3.0, 3.0]})
        result = compute_flb_slope(df)
        assert abs(result.iloc[3]["odds_skewness"]) < abs(result.iloc[0]["odds_skewness"])

    def test_single_race_same_values(self) -> None:
        df = pd.DataFrame({"race_id": ["R1"]*5, "umaban": [1,2,3,4,5],
            "tanodds": [2.0, 3.0, 5.0, 10.0, 20.0]})
        result = compute_flb_slope(df)
        assert result["odds_skewness"].nunique() == 1
