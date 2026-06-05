"""Surface-aware NaN CI tests for track condition features (REG-02, D-05).

Verifies that dirt features are NaN on turf rows, turf features are NaN
on dirt rows, and cross-surface features are available on both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.track_condition_features import (
    compute_track_condition_features,
)


def _make_mixed_df() -> pd.DataFrame:
    """Create a mixed surface DataFrame with track condition columns.

    3 turf rows: have turf_cushion, no dirt_moisture
    3 dirt rows: have dirt_moisture, no turf_cushion
    """
    return pd.DataFrame(
        {
            "race_id": ["R001"] * 6,
            "surface": ["turf"] * 3 + ["dirt"] * 3,
            "trackcd": [11, 11, 11, 23, 23, 23],
            "kyori": [1600] * 6,
            "kyakusitukubun_cd": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "sire_id": ["S001", "S002", "S003", "S004", "S005", "S006"],
            "frame_number": [1, 2, 3, 4, 5, 6],
            "bataijyu": [480.0, 470.0, 490.0, 460.0, 475.0, 485.0],
            "barei": [4.0, 5.0, 6.0, 3.0, 7.0, 4.0],
            # Turf rows get turf_cushion, dirt rows get NaN
            "turf_cushion": [9.5, 8.0, 10.5, np.nan, np.nan, np.nan],
            # Dirt rows get dirt_moisture, turf rows get NaN
            "dirt_moisture": [np.nan, np.nan, np.nan, 8.0, 15.0, 2.0],
        }
    )


class TestSurfaceAwareNaN:
    """D-05: Surface-aware NaN verification for track condition features."""

    @pytest.fixture()
    def mixed_result(self) -> pd.DataFrame:
        """Compute track condition features on mixed surface DataFrame."""
        df = _make_mixed_df()
        return compute_track_condition_features(df)

    def test_dirt_features_nan_on_turf_rows(self, mixed_result: pd.DataFrame) -> None:
        """Per D-05: dirt features are NaN on turf rows (no dirt_moisture)."""
        turf_mask = mixed_result["surface"] == "turf"
        dirt_features = [
            "dirt_moisture_x_kyakusitu",
            "dirt_moisture_x_barrier_pos",
            "dirt_moisture_high_flag",
            "dirt_moisture_dry_flag",
        ]
        for feat in dirt_features:
            if feat in mixed_result.columns:
                turf_vals = mixed_result.loc[turf_mask, feat]
                assert turf_vals.isna().all(), (
                    f"{feat} should be all-NaN on turf rows, "
                    f"got {turf_vals.notna().sum()} non-NaN"
                )

    def test_turf_features_nan_on_dirt_rows(self, mixed_result: pd.DataFrame) -> None:
        """Per D-05: turf features are NaN on dirt rows (no turf_cushion)."""
        dirt_mask = mixed_result["surface"] == "dirt"
        turf_features = [
            "turf_cushion_track_relative",
            "turf_cushion_track_zscore",
            "turf_cushion_x_kyakusitu",
        ]
        for feat in turf_features:
            if feat in mixed_result.columns:
                dirt_vals = mixed_result.loc[dirt_mask, feat]
                assert dirt_vals.isna().all(), (
                    f"{feat} should be all-NaN on dirt rows, "
                    f"got {dirt_vals.notna().sum()} non-NaN"
                )

    def test_cross_surface_features_available(self, mixed_result: pd.DataFrame) -> None:
        """Per D-05: track_front_bias_score and kickback_risk_score are NOT NaN on both surfaces."""
        cross_features = ["track_front_bias_score", "kickback_risk_score"]
        for feat in cross_features:
            assert feat in mixed_result.columns, f"Missing cross-surface feature: {feat}"
            # Should be non-NaN on both turf and dirt
            turf_mask = mixed_result["surface"] == "turf"
            dirt_mask = mixed_result["surface"] == "dirt"
            assert mixed_result.loc[turf_mask, feat].notna().all(), (
                f"{feat} should be non-NaN on turf rows"
            )
            assert mixed_result.loc[dirt_mask, feat].notna().all(), (
                f"{feat} should be non-NaN on dirt rows"
            )

    def test_sire_x_cushion_band_nan_on_dirt(self, mixed_result: pd.DataFrame) -> None:
        """Per D-05: sire_x_cushion_band is NaN on dirt rows (no turf_cushion to bin)."""
        if "sire_x_cushion_band" not in mixed_result.columns:
            pytest.skip("sire_x_cushion_band not computed (missing prerequisites)")
        dirt_mask = mixed_result["surface"] == "dirt"
        dirt_vals = mixed_result.loc[dirt_mask, "sire_x_cushion_band"]
        # Categorical NaN check
        assert dirt_vals.isna().all(), (
            f"sire_x_cushion_band should be all-NaN on dirt rows, "
            f"got {dirt_vals.notna().sum()} non-NaN"
        )

    def test_expected_pace_class_available_both_surfaces(
        self, mixed_result: pd.DataFrame
    ) -> None:
        """expected_pace_class should be non-NaN on both surfaces."""
        feat = "expected_pace_class"
        assert feat in mixed_result.columns, f"Missing feature: {feat}"
        turf_mask = mixed_result["surface"] == "turf"
        dirt_mask = mixed_result["surface"] == "dirt"
        assert mixed_result.loc[turf_mask, feat].notna().all(), (
            f"{feat} should be non-NaN on turf rows"
        )
        assert mixed_result.loc[dirt_mask, feat].notna().all(), (
            f"{feat} should be non-NaN on dirt rows"
        )
