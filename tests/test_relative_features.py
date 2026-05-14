"""src/features/relative_features.py のテスト

レース内相対比較特徴量 (7特徴量) のテスト。
全テスト mock 使用 (DB不要) -- プロジェクト規約に従う。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.relative_features import RELATIVE_FEATURE_COLS, compute_relative_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def race_df() -> pd.DataFrame:
    """2レース、各4頭のテストデータ。

    Race 1: horses with distinct values
      norm_finish_logit_avg: [0.8, 0.5, 0.3, 0.1]
      harontimel5_avg:       [12.0, 12.5, NaN, 13.5]
      timediff_avg:          [0.2, 0.5, 0.8, 1.2]
      blood_total_wr:        [0.20, 0.15, 0.10, 0.05]
      sire_wr:               [0.18, 0.12, 0.08, 0.04]
      weight_zscore:         [0.5, -0.3, 1.2, -0.4]
      closing_index_avg:     [0.1, 0.3, 0.5, 0.8]

    Race 2: all identical values (tests std=0 fallback)
      norm_finish_logit_avg: [0.7, 0.7, 0.7, 0.7]
      harontimel5_avg:       [12.0, 12.0, 12.0, 12.0]
      timediff_avg:          [0.5, 0.5, 0.5, 0.5]
      blood_total_wr:        [0.10, 0.10, 0.10, 0.10]
      sire_wr:               [0.10, 0.10, 0.10, 0.10]
      weight_zscore:         [0.0, 0.0, 0.0, 0.0]
      closing_index_avg:     [0.4, 0.4, 0.4, 0.4]
    """
    return pd.DataFrame(
        {
            "race_id": ["R1"] * 4 + ["R2"] * 4,
            "umaban": [1, 2, 3, 4, 1, 2, 3, 4],
            "norm_finish_logit_avg": [0.8, 0.5, 0.3, 0.1, 0.7, 0.7, 0.7, 0.7],
            "harontimel5_avg": [12.0, 12.5, float("nan"), 13.5, 12.0, 12.0, 12.0, 12.0],
            "timediff_avg": [0.2, 0.5, 0.8, 1.2, 0.5, 0.5, 0.5, 0.5],
            "blood_total_wr": [0.20, 0.15, 0.10, 0.05, 0.10, 0.10, 0.10, 0.10],
            "sire_wr": [0.18, 0.12, 0.08, 0.04, 0.10, 0.10, 0.10, 0.10],
            "weight_zscore": [0.5, -0.3, 1.2, -0.4, 0.0, 0.0, 0.0, 0.0],
            "closing_index_avg": [0.1, 0.3, 0.5, 0.8, 0.4, 0.4, 0.4, 0.4],
        }
    )


# ---------------------------------------------------------------------------
# Test 1: All 7 columns produced
# ---------------------------------------------------------------------------


class TestRelativeFeatureCols:
    """RELATIVE_FEATURE_COLS and column production tests."""

    def test_relative_feature_cols_has_7_entries(self) -> None:
        """RELATIVE_FEATURE_COLS has exactly 7 entries."""
        assert len(RELATIVE_FEATURE_COLS) == 7

    def test_relative_feature_cols_no_duplicates(self) -> None:
        """RELATIVE_FEATURE_COLS has no duplicate entries."""
        assert len(RELATIVE_FEATURE_COLS) == len(set(RELATIVE_FEATURE_COLS))

    def test_all_7_columns_produced(self, race_df: pd.DataFrame) -> None:
        """compute_relative_features() produces all 7 RELATIVE_FEATURE_COLS columns."""
        result = compute_relative_features(race_df)
        for col in RELATIVE_FEATURE_COLS:
            assert col in result.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Test 2: rel_norm_finish_zscore = (value - race_mean) / race_std
# ---------------------------------------------------------------------------


class TestZscoreNormFinish:
    """z-score of norm_finish_logit_avg within race."""

    def test_zscore_calculation_race1(self, race_df: pd.DataFrame) -> None:
        """Race 1: z-score = (value - mean) / std for norm_finish_logit_avg."""
        result = compute_relative_features(race_df)
        r1 = result[result["race_id"] == "R1"]
        values = r1["norm_finish_logit_avg"].values
        mean = np.mean(values)
        std = np.std(values, ddof=1)  # pandas groupby std uses ddof=1
        expected = (values - mean) / std
        np.testing.assert_allclose(r1["rel_norm_finish_zscore"].values, expected, atol=1e-10)

    def test_zscore_fallback_std0(self, race_df: pd.DataFrame) -> None:
        """Race 2 (all identical): z-score outputs 0.0 (std=0 fallback)."""
        result = compute_relative_features(race_df)
        r2 = result[result["race_id"] == "R2"]
        np.testing.assert_allclose(r2["rel_norm_finish_zscore"].values, [0.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Test 3: rel_haron_vs_mean = value - race_mean
# ---------------------------------------------------------------------------


class TestVsMeanHaron:
    """vs_mean of harontimel5_avg within race."""

    def test_vs_mean_calculation(self, race_df: pd.DataFrame) -> None:
        """rel_haron_vs_mean = harontimel5_avg - race_mean."""
        result = compute_relative_features(race_df)
        r1 = result[result["race_id"] == "R1"]
        # NaN row (umaban=3) should produce NaN
        nan_mask = r1["harontimel5_avg"].isna()
        assert r1.loc[nan_mask, "rel_haron_vs_mean"].isna().all()

        # Non-NaN rows: value - race_mean (mean computed over non-NaN values)
        valid = r1.dropna(subset=["harontimel5_avg"])
        mean_val = valid["harontimel5_avg"].mean()
        expected = valid["harontimel5_avg"] - mean_val
        np.testing.assert_allclose(valid["rel_haron_vs_mean"].values, expected.values, atol=1e-10)

    def test_vs_mean_std0_fallback(self, race_df: pd.DataFrame) -> None:
        """Race 2 (all identical): vs_mean = 0.0."""
        result = compute_relative_features(race_df)
        r2 = result[result["race_id"] == "R2"]
        np.testing.assert_allclose(r2["rel_haron_vs_mean"].values, [0.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Test 4: rel_timediff_rank = rank ascending (lower timediff = rank 1)
# ---------------------------------------------------------------------------


class TestRankTimediff:
    """rank of timediff_avg within race (ascending)."""

    def test_rank_ascending(self, race_df: pd.DataFrame) -> None:
        """rel_timediff_rank: lower timediff = rank 1."""
        result = compute_relative_features(race_df)
        r1 = result[result["race_id"] == "R1"]
        # timediff_avg: [0.2, 0.5, 0.8, 1.2] -> rank [1, 2, 3, 4]
        np.testing.assert_array_equal(r1["rel_timediff_rank"].values, [1.0, 2.0, 3.0, 4.0])

    def test_rank_ties(self, race_df: pd.DataFrame) -> None:
        """Race 2 (all identical): all get rank 1 (method=min, ascending)."""
        result = compute_relative_features(race_df)
        r2 = result[result["race_id"] == "R2"]
        # All same value -> all rank 1
        np.testing.assert_array_equal(r2["rel_timediff_rank"].values, [1.0, 1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# Test 5: rel_blood_quality_rank = rank descending (higher blood_total_wr = rank 1)
# ---------------------------------------------------------------------------


class TestRankBloodQuality:
    """rank of blood_total_wr within race (descending)."""

    def test_rank_descending(self, race_df: pd.DataFrame) -> None:
        """rel_blood_quality_rank: higher blood_total_wr = rank 1."""
        result = compute_relative_features(race_df)
        r1 = result[result["race_id"] == "R1"]
        # blood_total_wr: [0.20, 0.15, 0.10, 0.05] -> desc rank [1, 2, 3, 4]
        np.testing.assert_array_equal(r1["rel_blood_quality_rank"].values, [1.0, 2.0, 3.0, 4.0])


# ---------------------------------------------------------------------------
# Test 6: rel_sire_quality_rank = rank descending (higher sire_wr = rank 1)
# ---------------------------------------------------------------------------


class TestRankSireQuality:
    """rank of sire_wr within race (descending)."""

    def test_rank_descending(self, race_df: pd.DataFrame) -> None:
        """rel_sire_quality_rank: higher sire_wr = rank 1."""
        result = compute_relative_features(race_df)
        r1 = result[result["race_id"] == "R1"]
        # sire_wr: [0.18, 0.12, 0.08, 0.04] -> desc rank [1, 2, 3, 4]
        np.testing.assert_array_equal(r1["rel_sire_quality_rank"].values, [1.0, 2.0, 3.0, 4.0])


# ---------------------------------------------------------------------------
# Test 7: rel_weight_zscore = z-score of weight_zscore within race
# ---------------------------------------------------------------------------


class TestZscoreWeight:
    """z-score of weight_zscore within race."""

    def test_zscore_calculation_race1(self, race_df: pd.DataFrame) -> None:
        """rel_weight_zscore: z-score of weight_zscore in Race 1."""
        result = compute_relative_features(race_df)
        r1 = result[result["race_id"] == "R1"]
        values = r1["weight_zscore"].values
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        expected = (values - mean) / std
        np.testing.assert_allclose(r1["rel_weight_zscore"].values, expected, atol=1e-10)

    def test_zscore_fallback_std0(self, race_df: pd.DataFrame) -> None:
        """Race 2 (all identical): rel_weight_zscore = 0.0."""
        result = compute_relative_features(race_df)
        r2 = result[result["race_id"] == "R2"]
        np.testing.assert_allclose(r2["rel_weight_zscore"].values, [0.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Test 8: rel_closing_index_rank = rank ascending (lower closing index = rank 1
# for closing power -- lower means closer to front in closing)
# ---------------------------------------------------------------------------


class TestRankClosingIndex:
    """rank of closing_index_avg within race (ascending)."""

    def test_rank_ascending(self, race_df: pd.DataFrame) -> None:
        """rel_closing_index_rank: lower closing_index_avg = rank 1."""
        result = compute_relative_features(race_df)
        r1 = result[result["race_id"] == "R1"]
        # closing_index_avg: [0.1, 0.3, 0.5, 0.8] -> asc rank [1, 2, 3, 4]
        np.testing.assert_array_equal(r1["rel_closing_index_rank"].values, [1.0, 2.0, 3.0, 4.0])


# ---------------------------------------------------------------------------
# Test 9: std=0 fallback: z-score outputs 0.0 (not NaN/inf)
# ---------------------------------------------------------------------------


class TestStd0Fallback:
    """When all values in race are identical, z-score outputs 0.0."""

    def test_zscore_no_nan_when_std0(self, race_df: pd.DataFrame) -> None:
        """z-score features produce 0.0 (not NaN/inf) when std=0."""
        result = compute_relative_features(race_df)
        r2 = result[result["race_id"] == "R2"]
        for col in ["rel_norm_finish_zscore", "rel_weight_zscore"]:
            assert not r2[col].isna().any(), f"{col} has NaN values"
            assert np.isfinite(r2[col]).all(), f"{col} has non-finite values"


# ---------------------------------------------------------------------------
# Test 10: Missing base feature column is skipped silently
# ---------------------------------------------------------------------------


class TestMissingColumnSkipped:
    """Missing base feature column is skipped silently."""

    def test_missing_norm_finish_skipped(self) -> None:
        """When norm_finish_logit_avg is missing, rel_norm_finish_zscore is not added."""
        df = pd.DataFrame(
            {
                "race_id": ["R1", "R1"],
                "umaban": [1, 2],
                "harontimel5_avg": [12.0, 13.0],
                "timediff_avg": [0.2, 0.5],
                "blood_total_wr": [0.20, 0.15],
                "sire_wr": [0.18, 0.12],
                "weight_zscore": [0.5, -0.3],
                "closing_index_avg": [0.1, 0.3],
            }
        )
        result = compute_relative_features(df)
        assert "rel_norm_finish_zscore" not in result.columns

    def test_no_error_when_all_base_missing(self) -> None:
        """When no base features are present, only original columns remain."""
        df = pd.DataFrame({"race_id": ["R1", "R1"], "umaban": [1, 2]})
        result = compute_relative_features(df)
        assert "rel_norm_finish_zscore" not in result.columns
        assert "race_id" in result.columns


# ---------------------------------------------------------------------------
# Test 11: NaN base values produce NaN relative features
# ---------------------------------------------------------------------------


class TestNaNPropagation:
    """NaN base values produce NaN relative features."""

    def test_nan_haron_produces_nan_vs_mean(self, race_df: pd.DataFrame) -> None:
        """NaN in harontimel5_avg produces NaN in rel_haron_vs_mean."""
        result = compute_relative_features(race_df)
        r1 = result[result["race_id"] == "R1"]
        nan_row = r1[r1["harontimel5_avg"].isna()]
        assert nan_row["rel_haron_vs_mean"].isna().all()

    def test_nan_does_not_affect_group_mean(self, race_df: pd.DataFrame) -> None:
        """NaN values are excluded from group mean/std calculation."""
        result = compute_relative_features(race_df)
        r1 = result[result["race_id"] == "R1"]
        valid = r1.dropna(subset=["harontimel5_avg"])
        mean_val = valid["harontimel5_avg"].mean()
        # Mean should be computed from non-NaN values only: (12.0 + 12.5 + 13.5) / 3
        assert abs(mean_val - 12.666667) < 0.01


# ---------------------------------------------------------------------------
# Test 12: RELATIVE_FEATURE_COLS exactly 7 entries, no duplicates
# ---------------------------------------------------------------------------


class TestRelativeFeatureColsIntegrity:
    """RELATIVE_FEATURE_COLS list integrity tests."""

    def test_exactly_7_entries(self) -> None:
        """RELATIVE_FEATURE_COLS has exactly 7 entries."""
        assert len(RELATIVE_FEATURE_COLS) == 7

    def test_no_duplicates(self) -> None:
        """RELATIVE_FEATURE_COLS has no duplicate entries."""
        assert len(RELATIVE_FEATURE_COLS) == len(set(RELATIVE_FEATURE_COLS))

    def test_expected_names(self) -> None:
        """RELATIVE_FEATURE_COLS contains the expected feature names."""
        expected = [
            "rel_norm_finish_zscore",
            "rel_haron_vs_mean",
            "rel_timediff_rank",
            "rel_blood_quality_rank",
            "rel_sire_quality_rank",
            "rel_weight_zscore",
            "rel_closing_index_rank",
        ]
        assert sorted(RELATIVE_FEATURE_COLS) == sorted(expected)
