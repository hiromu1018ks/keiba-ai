"""Feature Routing Audit tests (SAF-01).

Fail-fast unit tests + diff tests ensuring calibrator (51 features) and
ranker (28 features) never leak into MarketModel or RaceQualityScreener.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from audit.feature_routing_registry import (
    ADVISORY_TARGET_MODELS,
    CRITICAL_TARGET_MODELS,
    FORBIDDEN_CALIBRATOR_FEATURES,
    FORBIDDEN_RANKER_FEATURES,
    REGISTRY_VERSION,
    run_feature_audit,
)
from models.market_model import MarketModel
from models.race_quality_screener import RaceQualityScreener
from models.market_aware_win_calibrator import MarketAwareWinCalibrator
from models.race_level_ranker import RaceLevelRanker


class TestFeatureRoutingAudit:
    """SAF-01: Feature routing audit — fail-fast + diff tests."""

    # --- Count tests ---

    def test_calibration_features_count(self) -> None:
        """FORBIDDEN_CALIBRATOR_FEATURES has exactly 51 elements."""
        assert len(FORBIDDEN_CALIBRATOR_FEATURES) == 51

    def test_ranker_features_count(self) -> None:
        """FORBIDDEN_RANKER_FEATURES has exactly 28 unique elements."""
        assert len(FORBIDDEN_RANKER_FEATURES) == 28

    def test_critical_targets_count(self) -> None:
        """CRITICAL_TARGET_MODELS contains exactly 2 entries."""
        assert len(CRITICAL_TARGET_MODELS) == 2

    # --- Fail-fast intersection tests ---

    def test_market_model_no_calibrator_leak(self) -> None:
        """MarketModel.FEATURE_COLS has zero intersection with FORBIDDEN_CALIBRATOR_FEATURES."""
        model_features = set(MarketModel.FEATURE_COLS)
        intersection = model_features & FORBIDDEN_CALIBRATOR_FEATURES
        assert intersection == set(), (
            f"MarketModel contains forbidden calibrator features: {intersection}"
        )

    def test_market_model_no_ranker_leak(self) -> None:
        """MarketModel.FEATURE_COLS has zero intersection with FORBIDDEN_RANKER_FEATURES."""
        model_features = set(MarketModel.FEATURE_COLS)
        intersection = model_features & FORBIDDEN_RANKER_FEATURES
        assert intersection == set(), (
            f"MarketModel contains forbidden ranker features: {intersection}"
        )

    def test_race_quality_screener_no_calibrator_leak(self) -> None:
        """RaceQualityScreener.FEATURE_COLS has zero intersection with FORBIDDEN_CALIBRATOR_FEATURES."""
        model_features = set(RaceQualityScreener.FEATURE_COLS)
        intersection = model_features & FORBIDDEN_CALIBRATOR_FEATURES
        assert intersection == set(), (
            f"RaceQualityScreener contains forbidden calibrator features: {intersection}"
        )

    def test_race_quality_screener_no_ranker_leak(self) -> None:
        """RaceQualityScreener.FEATURE_COLS has zero intersection with FORBIDDEN_RANKER_FEATURES."""
        model_features = set(RaceQualityScreener.FEATURE_COLS)
        intersection = model_features & FORBIDDEN_RANKER_FEATURES
        assert intersection == set(), (
            f"RaceQualityScreener contains forbidden ranker features: {intersection}"
        )

    # --- Diff tests (catch stale registry) ---

    def test_calibrator_features_match_build_feature_matrix(self) -> None:
        """FORBIDDEN_CALIBRATOR_FEATURES matches actual build_feature_matrix() output."""
        # Construct minimal DataFrame with required input columns
        df = pd.DataFrame({
            "p_model": [0.3, 0.2, 0.15],
            "p_market": [0.25, 0.18, 0.12],
            "tanodds": [4.0, 5.5, 8.0],
            "popularity_rank": [2, 3, 5],
            "field_size": [12, 12, 12],
            "p_win_race_rank_pct": [0.9, 0.7, 0.3],
            "race_id": ["r1", "r1", "r1"],
        })
        calibrator = MarketAwareWinCalibrator()
        _, feature_names = calibrator.build_feature_matrix(df)
        actual_features = set(feature_names)
        assert actual_features == FORBIDDEN_CALIBRATOR_FEATURES, (
            f"Registry mismatch: extra={actual_features - FORBIDDEN_CALIBRATOR_FEATURES}, "
            f"missing={FORBIDDEN_CALIBRATOR_FEATURES - actual_features}"
        )

    def test_ranker_features_match_class_attributes(self) -> None:
        """FORBIDDEN_RANKER_FEATURES matches union of RLR feature class attributes."""
        expected = (
            set(RaceLevelRanker.RELEVANCE_FEATURES)
            | set(RaceLevelRanker.VALUE_FEATURES)
            | set(RaceLevelRanker.DERIVED_VALUE_FEATURES)
        )
        assert expected == FORBIDDEN_RANKER_FEATURES, (
            f"Registry mismatch: extra={expected - FORBIDDEN_RANKER_FEATURES}, "
            f"missing={FORBIDDEN_RANKER_FEATURES - expected}"
        )

    # --- Advisory targets ---

    def test_advisory_targets_exist(self) -> None:
        """ADVISORY_TARGET_MODELS has at least 5 entries."""
        assert len(ADVISORY_TARGET_MODELS) >= 5

    def test_registry_version_is_string(self) -> None:
        """REGISTRY_VERSION is a non-empty string."""
        assert isinstance(REGISTRY_VERSION, str) and len(REGISTRY_VERSION) > 0

    # --- run_feature_audit integration ---

    def test_run_feature_audit_passes(self) -> None:
        """run_feature_audit() returns all critical targets as PASS."""
        results = run_feature_audit()
        for model_result in results["critical_models"]:
            assert model_result["status"] == "PASS", (
                f"Critical model {model_result['model_name']} status: "
                f"{model_result['status']}, "
                f"forbidden_intersections={model_result['forbidden_intersections']}"
            )
        for model_result in results["advisory_models"]:
            assert model_result["status"] in ("PASS", "WARN"), (
                f"Advisory model {model_result['model_name']} status: "
                f"{model_result['status']}"
            )
