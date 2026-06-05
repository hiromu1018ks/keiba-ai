"""Feature Routing Audit tests (SAF-01).

Fail-fast unit tests + diff tests ensuring calibrator (50 features) and
ranker (28 features) never leak into MarketModel or RaceQualityScreener.
Plus track condition feature routing verification (REG-02).
"""

from __future__ import annotations

import pandas as pd

from audit.feature_routing_registry import (
    ADVISORY_TARGET_MODELS,
    CALIBRATOR_EXCLUDED_RAW_INPUTS,
    CRITICAL_TARGET_MODELS,
    FORBIDDEN_CALIBRATOR_FEATURES,
    FORBIDDEN_RANKER_FEATURES,
    REGISTRY_VERSION,
    run_feature_audit,
)
from features.track_condition_features import (
    RACE_CONDITION_COLS,
    TRACK_CONDITION_COLS,
    TRACK_DERIVED_COLS,
)
from models.conformal_ev_model import ConformalEVModel
from models.ev_correction_model import EVCorrectionModel, PlaceEVCorrectionModel
from models.market_aware_win_calibrator import MarketAwareWinCalibrator
from models.market_model import MarketModel
from models.place_ability_model import PlaceAbilityModel
from models.race_level_ranker import RaceLevelRanker
from models.race_quality_screener import RaceQualityScreener
from models.regime_detector import RegimeDetector
from models.stage1_ability_model import AbilityModel
from models.two_stage_return_model import PlaceTwoStageModel, WinTwoStageModel

# Union of all 23 track condition features (T1/T2 + T3/T4 + T4-02)
ALL_TRACK_CONDITION_COLS: list[str] = (
    TRACK_CONDITION_COLS + TRACK_DERIVED_COLS + RACE_CONDITION_COLS
)


class TestFeatureRoutingAudit:
    """SAF-01: Feature routing audit — fail-fast + diff tests."""

    # --- Count tests ---

    def test_calibration_features_count(self) -> None:
        """FORBIDDEN_CALIBRATOR_FEATURES has exactly 50 elements (51 outputs minus field_size)."""
        assert len(FORBIDDEN_CALIBRATOR_FEATURES) == 50

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
        """RaceQualityScreener.FEATURE_COLS ∩ FORBIDDEN_CALIBRATOR_FEATURES is empty."""
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
        """FORBIDDEN_CALIBRATOR_FEATURES matches build_feature_matrix() output minus raw inputs."""
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
        # build_feature_matrix() outputs 51 features; we exclude raw inputs
        # (field_size) per Pitfall 3, yielding 50 forbidden features.
        actual_forbidden = set(feature_names) - CALIBRATOR_EXCLUDED_RAW_INPUTS
        assert actual_forbidden == FORBIDDEN_CALIBRATOR_FEATURES, (
            f"Registry mismatch: extra={actual_forbidden - FORBIDDEN_CALIBRATOR_FEATURES}, "
            f"missing={FORBIDDEN_CALIBRATOR_FEATURES - actual_forbidden}"
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


class TestTrackConditionRouting:
    """REG-02: Track condition feature routing verification.

    Verifies surgical routing of 23 track condition features:
    - 4 excluded models: MarketModel, RaceQualityScreener, RegimeDetector, ConformalEVModel
    - 7 included models: AbilityModel, WinTwoStageModel, PlaceTwoStageModel,
      EVCorrectionModel, PlaceEVCorrectionModel, PlaceAbilityModel, WideTwoStageModel
    Per D-06: Phase 48/49 surgical routing maintained.
    """

    # -- Excluded models (4) --

    def test_excluded_models_no_track_condition_features(self) -> None:
        """Per D-06: 4 excluded models have zero track condition features."""
        excluded_models = [
            ("MarketModel", MarketModel.FEATURE_COLS),
            ("RaceQualityScreener", RaceQualityScreener.FEATURE_COLS),
            ("RegimeDetector", RegimeDetector.FEATURE_COLS),
            ("ConformalEVModel", ConformalEVModel.FEATURE_COLS),
        ]
        tc_set = set(ALL_TRACK_CONDITION_COLS)
        for model_name, feature_cols in excluded_models:
            intersection = set(feature_cols) & tc_set
            assert intersection == set(), (
                f"{model_name} contains track condition features: {intersection}"
            )

    # -- Included models (7) --

    def test_included_models_have_track_condition_features(self) -> None:
        """Per D-04: All 7 included models have all 23 track condition features."""
        included_models = [
            ("AbilityModel", AbilityModel.FEATURE_COLS),
            ("WinTwoStageModel", WinTwoStageModel.FEATURE_COLS),
            ("PlaceTwoStageModel.HIT", PlaceTwoStageModel.HIT_FEATURE_COLS),
            ("PlaceTwoStageModel.RETURN", PlaceTwoStageModel.RETURN_FEATURE_COLS),
            ("EVCorrectionModel", EVCorrectionModel.FEATURE_COLS),
            ("PlaceEVCorrectionModel", PlaceEVCorrectionModel.FEATURE_COLS),
            ("PlaceAbilityModel", PlaceAbilityModel.FEATURE_COLS),
        ]
        tc_set = set(ALL_TRACK_CONDITION_COLS)
        for model_name, feature_cols in included_models:
            feature_set = set(feature_cols)
            missing = tc_set - feature_set
            assert not missing, (
                f"{model_name} missing track condition features: {missing}"
            )

    def test_wide_two_stage_has_track_condition_features(self) -> None:
        """WideTwoStageModel.SHARED_FEATURE_COLS has all 23 track condition features."""
        from models.wide_two_stage_model import WideTwoStageModel

        tc_set = set(ALL_TRACK_CONDITION_COLS)
        feature_set = set(WideTwoStageModel.SHARED_FEATURE_COLS)
        missing = tc_set - feature_set
        assert not missing, (
            f"WideTwoStageModel.SHARED_FEATURE_COLS missing track condition features: "
            f"{missing}"
        )

    # -- run_feature_audit integration --

    def test_audit_still_passes(self) -> None:
        """run_feature_audit() still returns overall_status='PASS'."""
        results = run_feature_audit()
        assert results["overall_status"] == "PASS"
