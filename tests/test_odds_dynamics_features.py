"""src/features/odds_dynamics_features.py のテスト"""

import numpy as np
import pandas as pd
import pytest

from features.odds_dynamics_features import (
    compute_odds_dynamics,
    compute_roi_ema,
    compute_rolling_volatility,
)


@pytest.fixture
def odds_ts_df() -> pd.DataFrame:
    """3頭のオッズ時系列データ（6時点: t-60, t-50, t-40, t-30, t-20, t-10）

    umaban=1: オッズ上昇（人気が落ちる） 3.0 → 5.5
    umaban=2: オッズ下降（人気が上がる） 10.0 → 5.0
    umaban=3: 安定 5.0 → 5.0
    """
    times = ["03241000", "03241010", "03241020", "03241030", "03241040", "03241050"]
    data = []
    for t_idx, time in enumerate(times):
        data.append(
            {
                "race_id": "R1",
                "happyotime": time,
                "umaban": 1,
                "tanodds": 3.0 + t_idx * 0.5,
                "fukuoddslow": 1.5,
                "ninki": 1 + t_idx,
            }
        )
        data.append(
            {
                "race_id": "R1",
                "happyotime": time,
                "umaban": 2,
                "tanodds": 10.0 - t_idx * 1.0,
                "fukuoddslow": 3.0,
                "ninki": 6 - t_idx,
            }
        )
        data.append(
            {
                "race_id": "R1",
                "happyotime": time,
                "umaban": 3,
                "tanodds": 5.0,
                "fukuoddslow": 2.0,
                "ninki": 3,
            }
        )
    return pd.DataFrame(data)


@pytest.fixture
def base_df() -> pd.DataFrame:
    """feature_engine出力を模擬したベースDataFrame"""
    return pd.DataFrame(
        {
            "race_id": ["R1"] * 3,
            "umaban": [1, 2, 3],
        }
    )


class TestOddsDynamicsFeatures:
    def test_odds_drop_rate_60_10(self, base_df: pd.DataFrame, odds_ts_df: pd.DataFrame):
        """t-60→t-10 のオッズ変化率を正しく計算"""
        result = compute_odds_dynamics(base_df, odds_ts_df)
        # umaban=1: idx0=3.0, idx5=5.5 → (3.0-5.5)/3.0 = -0.833
        assert abs(result.iloc[0]["odds_drop_rate_60_10"] - (-2.5 / 3.0)) < 1e-10
        # umaban=2: idx0=10.0, idx5=5.0 → (10.0-5.0)/10.0 = 0.5
        assert abs(result.iloc[1]["odds_drop_rate_60_10"] - 0.5) < 1e-10
        # umaban=3: 変化なし → 0.0
        assert abs(result.iloc[2]["odds_drop_rate_60_10"] - 0.0) < 1e-10

    def test_odds_drop_rate_30_10(self, base_df: pd.DataFrame, odds_ts_df: pd.DataFrame):
        """t-30→t-10 のオッズ変化率を正しく計算"""
        result = compute_odds_dynamics(base_df, odds_ts_df)
        # umaban=1: idx3=4.5, idx5=5.5 → (4.5-5.5)/4.5 = -0.222
        assert abs(result.iloc[0]["odds_drop_rate_30_10"] - (-1.0 / 4.5)) < 1e-10
        # umaban=2: idx3=7.0, idx5=5.0 → (7.0-5.0)/7.0 = 0.286
        assert abs(result.iloc[1]["odds_drop_rate_30_10"] - (2.0 / 7.0)) < 1e-10

    def test_odds_velocity(self, base_df: pd.DataFrame, odds_ts_df: pd.DataFrame):
        """オッズ変化速度（線形回帰の傾き）を計算"""
        result = compute_odds_dynamics(base_df, odds_ts_df)
        # umaban=1: 傾き=0.5 (等間隔で+0.5ずつ)
        assert abs(result.iloc[0]["odds_velocity"] - 0.5) < 1e-10
        # umaban=2: 傾き=-1.0
        assert abs(result.iloc[1]["odds_velocity"] - (-1.0)) < 1e-10
        # umaban=3: 傾き=0.0
        assert abs(result.iloc[2]["odds_velocity"] - 0.0) < 1e-10

    def test_odds_volatility(self, base_df: pd.DataFrame, odds_ts_df: pd.DataFrame):
        """オッズボラティリティ（変化量の標準偏差）を計算"""
        result = compute_odds_dynamics(base_df, odds_ts_df)
        # umaban=1: changes=[0.5, 0.5, 0.5, 0.5, 0.5] → std=0.0
        assert abs(result.iloc[0]["odds_volatility"]) < 1e-10
        # umaban=2: changes=[-1.0, -1.0, -1.0, -1.0, -1.0] → std=0.0
        assert abs(result.iloc[1]["odds_volatility"]) < 1e-10

    def test_popularity_change_30_10(self, base_df: pd.DataFrame, odds_ts_df: pd.DataFrame):
        """t-30→t-10 の人気順位変化を計算"""
        result = compute_odds_dynamics(base_df, odds_ts_df)
        # umaban=1: ninki at idx3=4, idx5=6 → change = 4-6 = -2 (人気が落ちた)
        assert result.iloc[0]["popularity_change_30_10"] == -2
        # umaban=2: ninki at idx3=3, idx5=1 → change = 3-1 = 2 (人気が上がった)
        assert result.iloc[1]["popularity_change_30_10"] == 2
        # umaban=3: ninki 安定 → change = 0
        assert result.iloc[2]["popularity_change_30_10"] == 0

    def test_no_time_series_returns_nan(self, base_df: pd.DataFrame):
        """時系列データがない場合は NaN を返す"""
        result = compute_odds_dynamics(base_df, pd.DataFrame())
        for col in [
            "odds_drop_rate_60_10",
            "odds_drop_rate_30_10",
            "odds_velocity",
            "odds_volatility",
            "popularity_change_30_10",
        ]:
            assert result[col].isna().all(), f"{col} should be NaN"

    def test_none_time_series_returns_nan(self, base_df: pd.DataFrame):
        """時系列データがNoneの場合は NaN を返す"""
        result = compute_odds_dynamics(base_df, None)
        assert result["odds_drop_rate_60_10"].isna().all()

    def test_missing_horse_in_ts(self, base_df: pd.DataFrame):
        """時系列データに存在しない馬は NaN"""
        odds_ts = pd.DataFrame(
            {
                "race_id": ["R1", "R1", "R1", "R1", "R1", "R1"],
                "happyotime": [
                    "03241000",
                    "03241010",
                    "03241020",
                    "03241030",
                    "03241040",
                    "03241050",
                ],
                "umaban": [1, 1, 1, 1, 1, 1],
                "tanodds": [3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
                "fukuoddslow": [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
                "ninki": [1, 2, 3, 4, 5, 6],
            }
        )
        result = compute_odds_dynamics(base_df, odds_ts)
        assert not np.isnan(result.iloc[0]["odds_drop_rate_60_10"])
        assert np.isnan(result.iloc[1]["odds_drop_rate_60_10"])
        assert np.isnan(result.iloc[2]["odds_drop_rate_60_10"])

    def test_out_of_range_tanodds_produces_nan_features(self, base_df: pd.DataFrame):
        """tanodds が範囲外 (1.0未満, 999.9超) の場合、特徴量が NaN になる。"""
        odds_ts = pd.DataFrame(
            {
                "race_id": ["R1", "R1"],
                "umaban": [1, 1],
                "happyotime": [1, 2],
                "tanodds": [0.5, 1500.0],
            }
        )
        result = compute_odds_dynamics(base_df, odds_ts)
        assert pd.isna(result["odds_drop_rate_60_10"].iloc[0])

    def test_valid_range_tanodds_computes_normally(self, base_df: pd.DataFrame):
        """tanodds が範囲内 (1.0-999.9) の場合、特徴量が正常に計算される。"""
        odds_ts = pd.DataFrame(
            {
                "race_id": ["R1", "R1"],
                "umaban": [1, 1],
                "happyotime": [1, 2],
                "tanodds": [5.0, 3.0],
            }
        )
        result = compute_odds_dynamics(base_df, odds_ts)
        assert not pd.isna(result["odds_drop_rate_60_10"].iloc[0])

    def test_preserves_existing_columns(self, base_df: pd.DataFrame, odds_ts_df: pd.DataFrame):
        """既存列を保持する"""
        result = compute_odds_dynamics(base_df, odds_ts_df)
        assert "race_id" in result.columns
        assert "umaban" in result.columns


class TestComputeRollingVolatility:
    def test_returns_series(self) -> None:
        """Series を返す"""
        df = pd.DataFrame(
            {
                "race_id": ["R1", "R1", "R2", "R2"],
                "odds_volatility": [0.1, 0.2, 0.3, 0.4],
            }
        )
        result = compute_rolling_volatility(df)
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)

    def test_missing_column_returns_nan(self) -> None:
        """odds_volatility 列がない場合は NaN"""
        df = pd.DataFrame({"race_id": ["R1", "R1"], "umaban": [1, 2]})
        result = compute_rolling_volatility(df)
        assert result.isna().all()

    def test_rolling_window_produces_nans_for_short_data(self) -> None:
        """min_periods=50 の場合、短いデータは NaN"""
        df = pd.DataFrame(
            {
                "race_id": [f"R{i}" for i in range(10)] * 2,
                "odds_volatility": np.random.uniform(0, 0.5, 20),
            }
        )
        result = compute_rolling_volatility(df, window=200, min_periods=50)
        # データが10レースしかないので rolling 結果は NaN
        assert result.isna().all()


class TestComputeOddsEma:
    def test_returns_dataframe_with_ema_columns(self) -> None:
        """3つのオッズ EMA 列を含む DataFrame を返す"""
        np.random.seed(42)
        rows = []
        for r in range(60):
            for h in range(10):
                rows.append({"race_id": f"R{r:04d}", "umaban": h + 1,
                    "tanodds": np.random.uniform(1.5, 100.0), "popularity_rank": h + 1})
        df = pd.DataFrame(rows)
        result = compute_roi_ema(df, span=20, min_periods=10)
        assert "favorite_implied_prob_ema" in result.columns
        assert "overround_ema" in result.columns
        assert "entropy_ema" in result.columns

    def test_missing_columns_returns_zeros(self) -> None:
        df = pd.DataFrame({"race_id": ["R1", "R1"], "umaban": [1, 2]})
        result = compute_roi_ema(df)
        assert (result["favorite_implied_prob_ema"] == 0.0).all()

    def test_no_kakuteijyuni_used(self) -> None:
        """kakuteijyuni 列がなくても正常に計算される"""
        np.random.seed(42)
        rows = []
        for r in range(60):
            for h in range(10):
                rows.append({"race_id": f"R{r:04d}", "umaban": h + 1,
                    "tanodds": np.random.uniform(1.5, 30.0), "popularity_rank": h + 1})
        df = pd.DataFrame(rows)
        assert "kakuteijyuni" not in df.columns
        result = compute_roi_ema(df, span=10, min_periods=5)
        assert "favorite_implied_prob_ema" in result.columns

    def test_overround_ema_computed(self) -> None:
        """overround_ema が計算される (NaN ではない)"""
        np.random.seed(42)
        rows = []
        for r in range(60):
            for h in range(10):
                rows.append({"race_id": f"R{r:04d}", "umaban": h + 1,
                    "tanodds": np.random.uniform(1.5, 30.0), "popularity_rank": h + 1})
        df = pd.DataFrame(rows)
        result = compute_roi_ema(df, span=10, min_periods=5)
        last = result[result["race_id"] == "R0059"]
        assert not last["overround_ema"].isna().any()

    def test_single_race_returns_same_value(self) -> None:
        """同一レース内の全行が同じ EMA 値を持つ"""
        np.random.seed(42)
        rows = [{"race_id": "R0001", "umaban": h + 1, "tanodds": np.random.uniform(1.5, 30.0),
            "popularity_rank": h + 1} for h in range(10)]
        df = pd.DataFrame(rows)
        result = compute_roi_ema(df, span=20, min_periods=1)
        assert result["favorite_implied_prob_ema"].nunique() == 1
