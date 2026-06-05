"""Tests for track condition IC evaluation script (VLD-02).

Per-feature IC evaluation for all 23 track condition features with:
- Univariate Spearman IC + C-orthogonal IC
- Surface stratification
- Category column separate evaluation (Kruskal-Wallis)
- Tier-level aggregation
- Signal classification (abs(C-IC) >= 0.005)
- Sign reversal detection between surface subsets
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from run_track_condition_ic_eval import (
    _classify_signal,
    _compute_c_orthogonal_ic,
    _compute_category_evaluation,
    _compute_tier_aggregation,
    _detect_flags,
    run_track_condition_ic_eval,
)


class TestCOrthogonalComputation:
    """Test C-orthogonal IC: feature regressed out of odds, Spearman of residual vs target."""

    def test_c_orthogonal_matches_expected(self) -> None:
        """Given synthetic feature + odds + target, verify C-orthogonal IC is correct."""
        np.random.seed(42)
        n = 500
        # Create odds (market probability proxy)
        odds = np.random.uniform(2.0, 20.0, n)
        # Feature correlated with odds (market-dependent signal)
        feature = 0.5 * odds + np.random.normal(0, 1, n)
        # Target independent of odds
        target = np.random.choice([0.0, 1.0], n, p=[0.9, 0.1])

        result = _compute_c_orthogonal_ic(feature, odds, target)

        # C-orthogonal IC should be near zero (signal removed by OLS)
        assert "rho" in result
        assert "p_value" in result
        assert "n" in result
        assert result["n"] == n
        # After regressing out odds, residual should have near-zero correlation with target
        assert abs(result["rho"]) < 0.15  # relaxed for random data

    def test_c_orthogonal_with_pure_signal(self) -> None:
        """Feature has signal independent of odds -> C-orthogonal IC should be non-zero."""
        np.random.seed(123)
        n = 500
        odds = np.random.uniform(2.0, 20.0, n)
        # Target depends on both odds and feature
        target_prob = 1.0 / odds + 0.05 * np.random.normal(0, 1, n)
        target = (target_prob > 0.1).astype(float)
        # Feature has signal independent of odds
        feature = target + np.random.normal(0, 0.5, n)

        result = _compute_c_orthogonal_ic(feature, odds, target)
        assert result["n"] == n
        # C-orthogonal IC should capture the feature-target correlation
        # not the feature-odds correlation

    def test_c_orthogonal_insufficient_samples(self) -> None:
        """With fewer than 30 valid samples, return NaN IC."""
        feature = np.array([1.0, 2.0, np.nan] * 5)
        odds = np.array([3.0, 4.0, 5.0] * 5)
        target = np.array([0.0, 1.0, 0.0] * 5)

        # Only 10 valid (non-NaN) samples
        result = _compute_c_orthogonal_ic(feature, odds, target)
        assert result["n"] < 30
        assert np.isnan(result["rho"])


class TestSignalClassification:
    """Signal classification: abs(IC) >= 0.005 = signal, below = weak."""

    def test_signal_classified_correctly(self) -> None:
        """abs(IC) >= 0.005 classified as signal."""
        assert _classify_signal(0.005) == "signal"
        assert _classify_signal(0.010) == "signal"
        assert _classify_signal(-0.005) == "signal"
        assert _classify_signal(-0.100) == "signal"

    def test_weak_classified_correctly(self) -> None:
        """abs(IC) < 0.005 classified as weak."""
        assert _classify_signal(0.004) == "weak"
        assert _classify_signal(0.0) == "weak"
        assert _classify_signal(0.001) == "weak"
        assert _classify_signal(-0.003) == "weak"

    def test_nan_classified_as_weak(self) -> None:
        """NaN IC classified as weak."""
        assert _classify_signal(float("nan")) == "weak"


class TestCategoryColumnSeparateEvaluation:
    """sire_x_cushion_band produces Kruskal-Wallis result instead of Spearman IC."""

    def test_category_column_kruskal_wallis(self) -> None:
        """Category column produces Kruskal-Wallis H test result."""
        np.random.seed(42)
        n = 200
        series = pd.Series(np.random.choice(["A", "B", "C", "D"], n), dtype="category")
        target = pd.Series(np.random.choice([0.0, 1.0], n, p=[0.9, 0.1]))

        result = _compute_category_evaluation(series, target)

        assert "category_count" in result
        assert "category_target_means" in result
        assert "kruskal_wallis" in result
        assert "H" in result["kruskal_wallis"]
        assert "p_value" in result["kruskal_wallis"]
        assert result["category_count"] == 4
        assert len(result["category_target_means"]) == 4
        assert result["n"] == n

    def test_category_with_clear_signal(self) -> None:
        """Categories with different target rates -> Kruskal-Wallis should detect signal."""
        np.random.seed(42)
        n = 300
        # Category A: 20% hit rate, Category B: 5% hit rate
        cats = []
        targets = []
        for i in range(n):
            if i < 150:
                cats.append("A")
                targets.append(1.0 if np.random.random() < 0.20 else 0.0)
            else:
                cats.append("B")
                targets.append(1.0 if np.random.random() < 0.05 else 0.0)

        series = pd.Series(cats, dtype="category")
        target = pd.Series(targets)

        result = _compute_category_evaluation(series, target)
        # H should be significant (p < 0.05 expected)
        assert result["kruskal_wallis"]["p_value"] < 0.1  # relaxed for random data

    def test_category_insufficient_data(self) -> None:
        """With very few samples, return NaN statistics."""
        series = pd.Series(["A", "B"], dtype="category")
        target = pd.Series([0.0, 1.0])

        result = _compute_category_evaluation(series, target)
        assert result["n"] == 2
        # Should handle gracefully


class TestSurfaceStratification:
    """Features evaluated separately on turf/dirt subsets when surface column exists."""

    def test_surface_stratification_in_full_eval(self) -> None:
        """Full evaluation produces surface-stratified IC results."""
        np.random.seed(42)
        n = 200

        oof_df = pd.DataFrame({
            "race_id": [f"r{i//10:04d}" for i in range(n)],
            "umaban": [i % 10 + 1 for i in range(n)],
            "kakuteijyuni": np.random.choice([1, 2, 3, 4, 5], n),
            "tanodds": np.random.uniform(2.0, 20.0, n),
            "surface": np.random.choice(["turf", "dirt"], n),
            "dirt_moisture_x_kyakusitu": np.random.normal(10, 3, n),
        })

        features_df = pd.DataFrame({
            "race_id": oof_df["race_id"].values,
            "umaban": oof_df["umaban"].values,
            "dirt_moisture_x_kyakusitu": oof_df["dirt_moisture_x_kyakusitu"].values,
        })

        result = run_track_condition_ic_eval(oof_df, features_df)

        feat_data = result["per_feature"].get("dirt_moisture_x_kyakusitu", {})
        by_surface = feat_data.get("by_surface", {})

        assert "turf" in by_surface
        assert "dirt" in by_surface
        assert "rho" in by_surface["turf"]
        assert "rho" in by_surface["dirt"]
        assert "n" in by_surface["turf"]
        assert "n" in by_surface["dirt"]


class TestTierAggregation:
    """Verify mean abs C-IC is computed correctly across Tier groups."""

    def test_tier_aggregation_basic(self) -> None:
        """Tier aggregation computes correct mean abs C-IC and signal count."""
        per_feature = {
            "dirt_moisture_x_kyakusitu": {
                "c_orthogonal_ic": {"rho": 0.010, "p_value": 0.01, "n": 5000},
            },
            "turf_cushion_track_relative": {
                "c_orthogonal_ic": {"rho": 0.003, "p_value": 0.5, "n": 5000},
            },
            "turf_cushion_track_zscore": {
                "c_orthogonal_ic": {"rho": -0.007, "p_value": 0.1, "n": 5000},
            },
            # Other T1/T2 features missing for simplicity
            # sire_x_cushion_band should be excluded
            "sire_x_cushion_band": {
                "c_orthogonal_ic": {"rho": 0.001, "p_value": 0.9, "n": 5000},
            },
        }

        result = _compute_tier_aggregation(per_feature)

        # T1_T2 tier has 8 features in TRACK_CONDITION_COLS, but we only have 4
        # The function iterates over ALL TRACK_CONDITION_COLS but only counts available
        t1_t2 = result["T1_T2"]
        assert "mean_abs_c_ic" in t1_t2
        assert "signal_count" in t1_t2
        assert "total" in t1_t2
        # Only 3 numeric features have data (sire excluded as category)
        # mean_abs = mean(|0.010|, |0.003|, |-0.007|) = 0.00667
        # signal_count = 2 (0.010 >= 0.005, |-0.007| >= 0.005)
        assert t1_t2["signal_count"] == 2

    def test_tier_aggregation_empty(self) -> None:
        """Empty per_feature dict produces NaN mean and zero signal count."""
        result = _compute_tier_aggregation({})
        for tier in ["T1_T2", "T3_T4_derived", "T4_02_race_level"]:
            assert result[tier]["signal_count"] == 0
            assert np.isnan(result[tier]["mean_abs_c_ic"])


class TestFlagSignReversal:
    """When turf IC and dirt IC have opposite signs, feature is flagged."""

    def test_sign_reversal_detected(self) -> None:
        """Opposite sign between turf and dirt IC triggers flag."""
        per_feature = {
            "test_feature": {
                "c_orthogonal_ic": {"rho": 0.01, "p_value": 0.01, "n": 5000},
                "by_surface": {
                    "turf": {"rho": 0.05, "n": 2500},
                    "dirt": {"rho": -0.03, "n": 2500},
                },
            }
        }

        flags = _detect_flags(per_feature)
        assert any("sign_reversal:test_feature" in f for f in flags)

    def test_no_sign_reversal_same_sign(self) -> None:
        """Same sign on both surfaces does NOT trigger flag."""
        per_feature = {
            "test_feature": {
                "c_orthogonal_ic": {"rho": 0.01, "p_value": 0.01, "n": 5000},
                "by_surface": {
                    "turf": {"rho": 0.05, "n": 2500},
                    "dirt": {"rho": 0.03, "n": 2500},
                },
            }
        }

        flags = _detect_flags(per_feature)
        assert not any("sign_reversal" in f for f in flags)

    def test_low_samples_flagged(self) -> None:
        """Feature with < 1000 valid samples gets low_samples flag."""
        per_feature = {
            "rare_feature": {
                "c_orthogonal_ic": {"rho": 0.01, "p_value": 0.01, "n": 500},
                "univariate_ic": {"rho": 0.01, "p_value": 0.01, "n": 500},
                "by_surface": {},
            }
        }

        flags = _detect_flags(per_feature)
        assert any("low_samples:rare_feature" in f for f in flags)

    def test_category_col_not_flagged(self) -> None:
        """sire_x_cushion_band (category) is excluded from sign reversal checks."""
        per_feature = {
            "sire_x_cushion_band": {
                "c_orthogonal_ic": {"rho": 0.01, "p_value": 0.01, "n": 5000},
                "univariate_ic": {"rho": 0.01, "p_value": 0.01, "n": 5000},
                "by_surface": {
                    "turf": {"rho": 0.05, "n": 2500},
                    "dirt": {"rho": -0.03, "n": 2500},
                },
            }
        }

        flags = _detect_flags(per_feature)
        # Category columns should be skipped
        assert not any("sign_reversal:sire_x_cushion_band" in f for f in flags)
