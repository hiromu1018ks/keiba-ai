"""src/features/market_bias_features.py のテスト"""

import math

import pandas as pd
import pytest

from features.market_bias_features import compute_market_bias


@pytest.fixture
def simple_odds_df() -> pd.DataFrame:
    """単純なオッズDataFrame（全馬均等オッズ）"""
    return pd.DataFrame({
        "race_id": ["R1"] * 4,
        "umaban": [1, 2, 3, 4],
        "tan_odds": [4.0, 4.0, 4.0, 4.0],
    })


@pytest.fixture
def skewed_odds_df() -> pd.DataFrame:
    """歪んだオッズDataFrame（人気馬+穴馬）"""
    return pd.DataFrame({
        "race_id": ["R1"] * 3,
        "umaban": [1, 2, 3],
        "tan_odds": [2.0, 5.0, 10.0],
    })


@pytest.fixture
def multi_race_df() -> pd.DataFrame:
    """複数レース"""
    return pd.DataFrame({
        "race_id": ["R1", "R1", "R2", "R2"],
        "umaban": [1, 2, 1, 2],
        "tan_odds": [2.0, 2.0, 3.0, 3.0],
    })


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
        assert "tan_odds" in result.columns
