"""WF Fold0 NaN rate verification tests (VLD-03, D-10 through D-14).

Tests the NaN rate threshold logic and surface-aware denominator computation
used by the validate_track_condition_nan.py diagnostic script.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _compute_nan_rate_with_thresholds(
    df: pd.DataFrame,
    feature_cols: list[str],
    surface_col: str = "surface",
) -> dict[str, dict[str, object]]:
    """Compute surface-aware NaN rates with 3-tier verdict (D-11).

    Per D-10: dirt_* features denominator = dirt rows only.
              turf_* features denominator = turf rows only.
              cross-surface features denominator = all rows.

    Per D-11: NaN rate < 30% -> PASS
              30% <= NaN rate < 50% -> WARN
              NaN rate >= 50% -> FAIL
    """
    total_rows = len(df)
    turf_rows = int((df[surface_col] == "turf").sum())
    dirt_rows = int((df[surface_col] == "dirt").sum())

    results: dict[str, dict[str, object]] = {}

    for col in feature_cols:
        if col not in df.columns:
            continue

        # Determine denominator based on feature name prefix (D-10)
        col_lower = col.lower()
        if col_lower.startswith("dirt_") or "moisture" in col_lower:
            if "track_front_bias" in col_lower or "kickback" in col_lower:
                denominator = total_rows
            else:
                denominator = dirt_rows
        elif col_lower.startswith("turf_") or col_lower.startswith("cushion_"):
            if col_lower in (
                "cushion_season_deviation",
                "cushion_anomaly_flag",
                "cushion_x_distance",
                "cushion_x_age",
            ):
                denominator = turf_rows
            else:
                denominator = turf_rows
        else:
            # Cross-surface features (track_front_bias_score, etc.)
            denominator = total_rows

        if denominator == 0:
            nan_rate = 1.0
        else:
            nan_count = int(df[col].isna().sum())
            nan_rate = nan_count / denominator

        # 3-tier threshold (D-11)
        if nan_rate < 0.30:
            verdict = "PASS"
        elif nan_rate < 0.50:
            verdict = "WARN"
        else:
            verdict = "FAIL"

        results[col] = {
            "nan_count": int(df[col].isna().sum()),
            "denominator": denominator,
            "nan_rate": nan_rate,
            "verdict": verdict,
        }

    return results


class TestWFold0NaNRate:
    """VLD-03: WF Fold0 NaN rate threshold logic tests."""

    def test_nan_rate_thresholds_applied_correctly(self) -> None:
        """Per D-11: < 30% PASS, 30-50% WARN, >= 50% FAIL."""
        df = pd.DataFrame(
            {
                "surface": ["turf"] * 100 + ["dirt"] * 100,
                # cross-surface features: denominator = 200
                # 25 NaN / 200 = 0.125 -> PASS
                "feature_pass": [np.nan] * 15 + [1.0] * 85 + [np.nan] * 10 + [1.0] * 90,
                # 75 NaN / 200 = 0.375 -> WARN
                "feature_warn": [np.nan] * 40 + [1.0] * 60 + [np.nan] * 35 + [1.0] * 65,
                # 110 NaN / 200 = 0.55 -> FAIL
                "feature_fail": [np.nan] * 60 + [1.0] * 40 + [np.nan] * 50 + [1.0] * 50,
            }
        )

        results = _compute_nan_rate_with_thresholds(
            df,
            ["feature_pass", "feature_warn", "feature_fail"],
        )

        assert results["feature_pass"]["verdict"] == "PASS"
        assert results["feature_pass"]["nan_rate"] == pytest.approx(0.125)

        assert results["feature_warn"]["verdict"] == "WARN"
        assert results["feature_warn"]["nan_rate"] == pytest.approx(0.375)

        assert results["feature_fail"]["verdict"] == "FAIL"
        assert results["feature_fail"]["nan_rate"] == pytest.approx(0.55)

    def test_nan_rate_warn_threshold(self) -> None:
        """30-50% NaN rate produces WARN verdict."""
        # 100 turf rows, 35 NaN = 35% -> WARN for turf feature
        df = pd.DataFrame(
            {
                "surface": ["turf"] * 100 + ["dirt"] * 100,
                "turf_cushion_track_relative": [np.nan] * 35 + [1.0] * 65 + [1.0] * 100,
            }
        )
        results = _compute_nan_rate_with_thresholds(
            df, ["turf_cushion_track_relative"]
        )
        assert results["turf_cushion_track_relative"]["verdict"] == "WARN"
        assert results["turf_cushion_track_relative"]["nan_rate"] == pytest.approx(0.35)
        assert results["turf_cushion_track_relative"]["denominator"] == 100

    def test_nan_rate_fail_threshold(self) -> None:
        """>= 50% NaN rate produces FAIL verdict."""
        # 100 dirt rows, 55 NaN = 55% -> FAIL for dirt feature
        df = pd.DataFrame(
            {
                "surface": ["turf"] * 100 + ["dirt"] * 100,
                "dirt_moisture_x_kyakusitu": [1.0] * 100 + [np.nan] * 55 + [1.0] * 45,
            }
        )
        results = _compute_nan_rate_with_thresholds(
            df, ["dirt_moisture_x_kyakusitu"]
        )
        assert results["dirt_moisture_x_kyakusitu"]["verdict"] == "FAIL"
        assert results["dirt_moisture_x_kyakusitu"]["nan_rate"] == pytest.approx(0.55)
        assert results["dirt_moisture_x_kyakusitu"]["denominator"] == 100

    def test_surface_aware_denominator(self) -> None:
        """Per D-10: turf features use turf-only denominator, not total rows."""
        # 100 turf rows with 25 NaN for turf_cushion_track_relative
        # 100 dirt rows (should be excluded from denominator)
        # NaN rate = 25/100 = 25%, NOT 25/200 = 12.5%
        df = pd.DataFrame(
            {
                "surface": ["turf"] * 100 + ["dirt"] * 100,
                "turf_cushion_track_relative": [np.nan] * 25 + [1.0] * 75 + [1.0] * 100,
            }
        )
        results = _compute_nan_rate_with_thresholds(
            df, ["turf_cushion_track_relative"]
        )
        assert results["turf_cushion_track_relative"]["denominator"] == 100
        assert results["turf_cushion_track_relative"]["nan_rate"] == pytest.approx(0.25)
        assert results["turf_cushion_track_relative"]["verdict"] == "PASS"

    def test_cross_surface_denominator_uses_all_rows(self) -> None:
        """Cross-surface features use total rows as denominator."""
        df = pd.DataFrame(
            {
                "surface": ["turf"] * 100 + ["dirt"] * 100,
                "track_front_bias_score": [np.nan] * 30 + [1.0] * 70 + [1.0] * 100,
            }
        )
        results = _compute_nan_rate_with_thresholds(
            df, ["track_front_bias_score"]
        )
        assert results["track_front_bias_score"]["denominator"] == 200
        assert results["track_front_bias_score"]["nan_rate"] == pytest.approx(0.15)
        assert results["track_front_bias_score"]["verdict"] == "PASS"
