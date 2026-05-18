"""race_level_features.py の単体テスト

6つの rl_* 特徴量の計算を検証する:
- rl_log_odds_entropy: インプライド確率のシャノンエントロピー
- rl_odds_dispersion: レース内tanoddsの標準偏差
- rl_top3_odds_gap: 1番人気と3番人気のtanodds差
- rl_top1_odds: 1番人気のtanodds値を全馬にブロードキャスト
- rl_favorite_rank_gap: 1番人気と2番人気の対数オッズ差
- rl_n_horses: 出走頭数
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_3horse_df() -> pd.DataFrame:
    """3頭立てレース (race_idあり) のテストデータ"""
    return pd.DataFrame(
        {
            "race_id": ["R001", "R001", "R001"],
            "umaban": [1, 2, 3],
            "tanodds": [2.0, 5.0, 10.0],
            "field_size": [3, 3, 3],
        }
    )


def _make_2horse_df() -> pd.DataFrame:
    """2頭立てレースのテストデータ"""
    return pd.DataFrame(
        {
            "race_id": ["R002", "R002"],
            "umaban": [1, 2],
            "tanodds": [1.5, 4.0],
            "field_size": [2, 2],
        }
    )


def _make_no_tanodds_df() -> pd.DataFrame:
    """tanodds列なしのテストデータ"""
    return pd.DataFrame(
        {
            "race_id": ["R003", "R003", "R003"],
            "umaban": [1, 2, 3],
            "field_size": [3, 3, 3],
        }
    )


def _make_no_race_id_df() -> pd.DataFrame:
    """race_id列なしの単一レースDataFrame"""
    return pd.DataFrame(
        {
            "umaban": [1, 2, 3],
            "tanodds": [2.0, 5.0, 10.0],
            "field_size": [3, 3, 3],
        }
    )


def _make_tanodds_with_zero_nan_df() -> pd.DataFrame:
    """tanoddsに0とNaNが含まれるテストデータ"""
    return pd.DataFrame(
        {
            "race_id": ["R004", "R004", "R004", "R004"],
            "umaban": [1, 2, 3, 4],
            "tanodds": [2.0, 0.0, np.nan, 8.0],
            "field_size": [4, 4, 4, 4],
        }
    )


class TestComputeRaceLevelFeatures:
    """compute_race_level_features() のテスト"""

    def test_3horse_race_all_features_computed(self) -> None:
        """Test 1: 3頭立てレースで全 rl_* 特徴量が正しく計算される"""
        from features.race_level_features import compute_race_level_features

        df = _make_3horse_df()
        result = compute_race_level_features(df)

        # 全6特徴量が追加されている
        expected_cols = [
            "rl_log_odds_entropy",
            "rl_odds_dispersion",
            "rl_top3_odds_gap",
            "rl_top1_odds",
            "rl_favorite_rank_gap",
            "rl_n_horses",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

        # 全行に同じ値がブロードキャストされている (同一race_id)
        for col in expected_cols:
            assert result[col].notna().all(), f"{col} has NaN in 3-horse race"

        # rl_log_odds_entropy: p = [1/2, 1/5, 1/10] normalized = [10/17, 4/17, 2/17]
        # H = -sum(p * log(p))
        p = np.array([1 / 2, 1 / 5, 1 / 10])
        p_norm = p / p.sum()
        expected_entropy = -np.sum(p_norm * np.log(p_norm))
        np.testing.assert_allclose(
            result["rl_log_odds_entropy"].iloc[0], expected_entropy, rtol=1e-6
        )

        # rl_odds_dispersion: std([2.0, 5.0, 10.0])
        expected_std = np.std([2.0, 5.0, 10.0], ddof=1)
        np.testing.assert_allclose(
            result["rl_odds_dispersion"].iloc[0], expected_std, rtol=1e-6
        )

        # rl_top3_odds_gap: 3番人気(10.0) - 1番人気(2.0) = 8.0
        np.testing.assert_allclose(
            result["rl_top3_odds_gap"].iloc[0], 8.0, rtol=1e-6
        )

        # rl_top1_odds: 1番人気のオッズ = 2.0
        np.testing.assert_allclose(
            result["rl_top1_odds"].iloc[0], 2.0, rtol=1e-6
        )

        # rl_favorite_rank_gap: log(5.0 / 2.0)
        expected_gap = np.log(5.0 / 2.0)
        np.testing.assert_allclose(
            result["rl_favorite_rank_gap"].iloc[0], expected_gap, rtol=1e-6
        )

        # rl_n_horses: 3
        assert result["rl_n_horses"].iloc[0] == 3

    def test_no_tanodds_returns_nan(self) -> None:
        """Test 2: tanodds列なしの場合、全 rl_* が NaN"""
        from features.race_level_features import compute_race_level_features

        df = _make_no_tanodds_df()
        result = compute_race_level_features(df)

        rl_cols = [
            "rl_log_odds_entropy",
            "rl_odds_dispersion",
            "rl_top3_odds_gap",
            "rl_top1_odds",
            "rl_favorite_rank_gap",
            "rl_n_horses",
        ]
        for col in rl_cols:
            assert col in result.columns, f"Missing column: {col}"
            assert result[col].isna().all(), f"{col} should be all NaN when tanodds missing"

    def test_tanodds_with_zero_and_nan(self) -> None:
        """Test 3: tanodds に 0 と NaN が含まれてもエラーにならず安全にフォールバック"""
        from features.race_level_features import compute_race_level_features

        df = _make_tanodds_with_zero_nan_df()
        # エラーが発生しないことを確認
        result = compute_race_level_features(df)

        # 有効なオッズは [2.0, 8.0] の2頭のみ → rl_n_horses = 4 (field_size)
        # ただし entropy や std の計算は有効オッズのみで行われる
        assert "rl_log_odds_entropy" in result.columns
        # 0 と NaN が除外されてもエラーにならない
        assert not np.isinf(result["rl_log_odds_entropy"]).any()

    def test_2horse_top3_gap_is_nan(self) -> None:
        """Test 4: 2頭立ての場合 rl_top3_odds_gap は NaN、rl_favorite_rank_gap は計算される"""
        from features.race_level_features import compute_race_level_features

        df = _make_2horse_df()
        result = compute_race_level_features(df)

        # rl_top3_odds_gap: 3番人気がいないので NaN
        assert result["rl_top3_odds_gap"].isna().all()

        # rl_favorite_rank_gap: log(4.0 / 1.5) は計算される
        expected_gap = np.log(4.0 / 1.5)
        np.testing.assert_allclose(
            result["rl_favorite_rank_gap"].iloc[0], expected_gap, rtol=1e-6
        )

        # rl_n_horses: 2
        assert result["rl_n_horses"].iloc[0] == 2

    def test_no_race_id_single_race(self) -> None:
        """Test 5: race_idなしの単一DataFrameでも動作する"""
        from features.race_level_features import compute_race_level_features

        df = _make_no_race_id_df()
        result = compute_race_level_features(df)

        # 全特徴量が計算される (groupbyなしで全体を1レースとして処理)
        expected_cols = [
            "rl_log_odds_entropy",
            "rl_odds_dispersion",
            "rl_top3_odds_gap",
            "rl_top1_odds",
            "rl_favorite_rank_gap",
            "rl_n_horses",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"
            assert result[col].notna().all(), f"{col} should be computed without race_id"

        # 値の妥当性チェック (3頭 [2.0, 5.0, 10.0])
        np.testing.assert_allclose(result["rl_top1_odds"].iloc[0], 2.0, rtol=1e-6)
        assert result["rl_n_horses"].iloc[0] == 3

    def test_no_post_race_columns_used(self) -> None:
        """Test 6: rl_* 計算に POST_RACE_COLS が使われていない (import確認)"""
        from domain.types import POST_RACE_COLS
        from features.race_level_features import compute_race_level_features

        # POST_RACE列を含むDataFrameで実行しても、rl_*の値がPOST_RACE列に依存しない
        df_with_post_race = pd.DataFrame(
            {
                "race_id": ["R005", "R005", "R005"],
                "umaban": [1, 2, 3],
                "tanodds": [3.0, 6.0, 12.0],
                "field_size": [3, 3, 3],
                # POST_RACE cols — これらが計算に影響を与えてはならない
                "kakuteijyuni": [1, 2, 3],
                "confirmed_odds": [2.8, 5.5, 11.0],
                "ninki": [1, 2, 3],
            }
        )
        result_with = compute_race_level_features(df_with_post_race)

        # POST_RACE列なしの同じデータで計算
        df_without_post_race = pd.DataFrame(
            {
                "race_id": ["R005", "R005", "R005"],
                "umaban": [1, 2, 3],
                "tanodds": [3.0, 6.0, 12.0],
                "field_size": [3, 3, 3],
            }
        )
        result_without = compute_race_level_features(df_without_post_race)

        rl_cols = [
            "rl_log_odds_entropy",
            "rl_odds_dispersion",
            "rl_top3_odds_gap",
            "rl_top1_odds",
            "rl_favorite_rank_gap",
            "rl_n_horses",
        ]
        for col in rl_cols:
            np.testing.assert_allclose(
                result_with[col].values,
                result_without[col].values,
                rtol=1e-10,
                err_msg=f"{col} differs with POST_RACE cols present",
            )

    def test_multi_race_different_values(self) -> None:
        """複数レースが混在するDataFrameで各レースに正しい値が計算される"""
        from features.race_level_features import compute_race_level_features

        df = pd.DataFrame(
            {
                "race_id": ["R001", "R001", "R001", "R002", "R002"],
                "umaban": [1, 2, 3, 1, 2],
                "tanodds": [2.0, 5.0, 10.0, 1.5, 4.0],
                "field_size": [3, 3, 3, 2, 2],
            }
        )
        result = compute_race_level_features(df)

        # R001: rl_top1_odds = 2.0
        r001_mask = result["race_id"] == "R001"
        np.testing.assert_allclose(
            result.loc[r001_mask, "rl_top1_odds"].unique(), [2.0], rtol=1e-6
        )

        # R002: rl_top1_odds = 1.5
        r002_mask = result["race_id"] == "R002"
        np.testing.assert_allclose(
            result.loc[r002_mask, "rl_top1_odds"].unique(), [1.5], rtol=1e-6
        )

        # R002: rl_n_horses = 2 (field_size)
        assert result.loc[r002_mask, "rl_n_horses"].iloc[0] == 2

    def test_does_not_modify_input(self) -> None:
        """入力DataFrameが変更されないことを確認"""
        from features.race_level_features import compute_race_level_features

        df = _make_3horse_df()
        original_cols = set(df.columns)
        original_values = df.values.copy()

        compute_race_level_features(df)

        # 入力DataFrameは変更されていない
        assert set(df.columns) == original_cols
        np.testing.assert_array_equal(df.values, original_values)
