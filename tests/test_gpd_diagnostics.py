"""Tests for Gain per Depth diagnostic module.

TDD RED phase: Tests for FEATURE_CATEGORY_MAP completeness, Booster extraction,
depth-gain computation, MDR/FAD metrics, full pipeline, and edge cases.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Test 1: FEATURE_CATEGORY_MAP completeness
# ---------------------------------------------------------------------------


class TestFeatureCategoryMapCompleteness:
    """Verify every feature from every model's FEATURE_COLS is in FEATURE_CATEGORY_MAP."""

    def _collect_all_feature_cols(self) -> set[str]:
        """Collect the union of all FEATURE_COLS from all model classes."""
        from models.conformal_ev_model import ConformalEVModel
        from models.ev_correction_model import EVCorrectionModel, PlaceEVCorrectionModel
        from models.market_model import MarketModel
        from models.place_ability_model import PlaceAbilityModel
        from models.stage1_ability_model import AbilityModel
        from models.two_stage_return_model import PlaceTwoStageModel, WinTwoStageModel
        from models.wide_two_stage_model import WideTwoStageModel

        all_features: set[str] = set()
        # AbilityModel
        all_features.update(AbilityModel.FEATURE_COLS)
        # WinTwoStageModel
        all_features.update(WinTwoStageModel.FEATURE_COLS)
        # PlaceTwoStageModel: union of HIT + RETURN
        all_features.update(PlaceTwoStageModel.HIT_FEATURE_COLS)
        all_features.update(PlaceTwoStageModel.RETURN_FEATURE_COLS)
        # MarketModel
        all_features.update(MarketModel.FEATURE_COLS)
        # EVCorrectionModel
        all_features.update(EVCorrectionModel.FEATURE_COLS)
        # PlaceEVCorrectionModel
        all_features.update(PlaceEVCorrectionModel.FEATURE_COLS)
        # WideTwoStageModel
        all_features.update(WideTwoStageModel.SHARED_FEATURE_COLS)
        # ConformalEVModel
        all_features.update(ConformalEVModel.FEATURE_COLS)
        # PlaceAbilityModel
        all_features.update(PlaceAbilityModel.FEATURE_COLS)
        return all_features

    def test_all_features_registered(self) -> None:
        """Every feature from all model FEATURE_COLS must exist in FEATURE_CATEGORY_MAP."""
        from models.gpd_diagnostics import FEATURE_CATEGORY_MAP

        all_features = self._collect_all_feature_cols()
        unregistered = all_features - set(FEATURE_CATEGORY_MAP.keys())
        assert unregistered == set(), (
            f"FEATURE_CATEGORY_MAP is missing {len(unregistered)} features: "
            f"{sorted(unregistered)}"
        )

    def test_all_values_are_valid_categories(self) -> None:
        """Every value in FEATURE_CATEGORY_MAP must be market/fundamental/categorical."""
        from models.gpd_diagnostics import FEATURE_CATEGORY_MAP

        valid = {"market", "fundamental", "categorical"}
        invalid = {k: v for k, v in FEATURE_CATEGORY_MAP.items() if v not in valid}
        assert not invalid, (
            f"FEATURE_CATEGORY_MAP has {len(invalid)} invalid categories: {invalid}"
        )

    def test_no_extra_features_in_map(self) -> None:
        """FEATURE_CATEGORY_MAP should not have features not in any model (optional check)."""
        from models.gpd_diagnostics import FEATURE_CATEGORY_MAP

        all_features = self._collect_all_feature_cols()
        extra = set(FEATURE_CATEGORY_MAP.keys()) - all_features
        # This is informational, not an error -- the map may legitimately include
        # features from StackedEnsemble that match WinTwoStageModel features
        # Just assert no wild extras (tolerance for small known extras)
        # StackedEnsemble uses the same feature space as WinTwoStageModel
        assert len(extra) == 0, (
            f"FEATURE_CATEGORY_MAP has {len(extra)} extra features not in any model: "
            f"{sorted(extra)}"
        )


# ---------------------------------------------------------------------------
# Test 2: Booster extraction
# ---------------------------------------------------------------------------


class TestBoosterExtraction:
    """Verify _extract_boosters() correctly extracts all LightGBM Boosters from
    TrainedModelsV5."""

    def _make_mock_booster(self) -> MagicMock:
        """Create a mock lgb.Booster."""
        booster = MagicMock()
        booster.trees_to_dataframe.return_value = pd.DataFrame({
            "tree_index": [0],
            "node_depth": [1],
            "node_index": ["0-S"],
            "left_child": [None],
            "right_child": [None],
            "parent_index": [None],
            "split_feature": ["odds"],
            "split_gain": [1.0],
            "threshold": [5.0],
            "decision_type": ["<="],
            "missing_direction": ["left"],
            "missing_type": ["None"],
            "value": [0.5],
            "weight": [100.0],
            "count": [100],
        })
        return booster

    def _make_mock_models(self) -> MagicMock:
        """Create a mock TrainedModelsV5 with typical SubmodelSet structure."""
        import lightgbm as lgb

        mock_booster = self._make_mock_booster()

        # Mock StackedEnsemble
        mock_ensemble = MagicMock()
        mock_ensemble.lgbm_model = mock_booster
        type(mock_ensemble).best_iteration = PropertyMock(return_value=100)

        # SubmodelSet
        sub = MagicMock()
        # stage1: AbilityModel with dict of per-surface boosters
        sub.stage1.models = {"turf": mock_booster, "dirt": mock_booster}
        # win: hit_model (StackedEnsemble) + return_model (Booster)
        sub.win.hit_model = mock_ensemble  # StackedEnsemble wrapper
        sub.win.return_model = mock_booster
        # market
        sub.market.model = mock_booster
        # ev_corrector
        sub.ev_corrector.p_correction_model = mock_booster
        sub.ev_corrector.e_correction_model = mock_booster
        # place (optional)
        sub.place.hit_model = mock_booster
        sub.place.return_model = mock_booster
        # place_ev_corrector (optional)
        sub.place_ev_corrector.p_correction_model = mock_booster
        sub.place_ev_corrector.e_correction_model = mock_booster
        # wide (optional)
        sub.wide.hit_model = mock_booster
        sub.wide.return_model = mock_booster
        # conformal_ev_model (optional)
        sub.conformal_ev_model.q_low_model = mock_booster
        sub.conformal_ev_model.q_high_model = mock_booster
        # place_ability
        sub.place_ability._model = MagicMock()
        sub.place_ability._model.booster_ = mock_booster

        # TrainedModelsV5
        models = MagicMock()
        models.submodels = {"turf": sub, "dirt": sub}
        return models

    def test_extract_primary_models(self) -> None:
        """Primary tier models (stage1, win, market) are extracted."""
        from models.gpd_diagnostics import _extract_boosters

        models = self._make_mock_models()
        boosters = _extract_boosters(models)

        # Check primary tier
        assert "stage1_turf" in boosters
        assert "stage1_dirt" in boosters
        assert "win_hit_turf" in boosters or "ensemble_lgbm_turf" in boosters
        assert "win_ret_turf" in boosters
        assert "win_ret_dirt" in boosters
        assert "market_turf" in boosters
        assert "market_dirt" in boosters

    def test_extract_detailed_models(self) -> None:
        """Detailed tier models (ev_corr, place, wide, cqr) are extracted."""
        from models.gpd_diagnostics import _extract_boosters

        models = self._make_mock_models()
        boosters = _extract_boosters(models)

        # EV correction
        assert "ev_corr_p_turf" in boosters
        assert "ev_corr_e_turf" in boosters
        # Place
        assert "place_hit_turf" in boosters
        assert "place_ret_turf" in boosters
        # Place EV correction
        assert "place_ev_corr_p_turf" in boosters
        assert "place_ev_corr_e_turf" in boosters
        # Wide
        assert "wide_hit_turf" in boosters
        assert "wide_ret_turf" in boosters
        # Conformal EV / CQR
        assert "cqr_q_low_turf" in boosters
        assert "cqr_q_high_turf" in boosters

    def test_stacked_ensemble_unwrapped(self) -> None:
        """StackedEnsemble is unwrapped to .lgbm_model for Booster access."""
        from models.gpd_diagnostics import _extract_boosters

        models = self._make_mock_models()
        boosters = _extract_boosters(models)

        # The win hit model was a StackedEnsemble, so it should be unwrapped
        assert "ensemble_lgbm_turf" in boosters
        assert "ensemble_lgbm_dirt" in boosters

    def test_extract_boosters_with_none_optionals(self) -> None:
        """Optional models (place, wide, cqr) as None should not cause errors."""
        from models.gpd_diagnostics import _extract_boosters

        mock_booster = self._make_mock_booster()
        sub = MagicMock()
        sub.stage1.models = {"turf": mock_booster}
        sub.win.hit_model = mock_booster
        sub.win.return_model = mock_booster
        sub.market.model = mock_booster
        sub.ev_corrector.p_correction_model = mock_booster
        sub.ev_corrector.e_correction_model = mock_booster
        # All optional models are None
        sub.place = None
        sub.place_ev_corrector = None
        sub.wide = None
        sub.conformal_ev_model = None
        sub.place_ability = None

        models = MagicMock()
        models.submodels = {"turf": sub}
        boosters = _extract_boosters(models)

        # Primary tier should still work
        assert "stage1_turf" in boosters
        assert "win_hit_turf" in boosters
        # Optional tier should be absent
        assert "place_hit_turf" not in boosters
        assert "wide_hit_turf" not in boosters
        assert "cqr_q_low_turf" not in boosters


# ---------------------------------------------------------------------------
# Test 3: Depth-gain computation
# ---------------------------------------------------------------------------


class TestDepthGainComputation:
    """Verify _compute_depth_gains() correctly groups gain by depth and category."""

    def _make_tree_df(
        self,
        rows: list[dict[str, object]],
    ) -> pd.DataFrame:
        """Build a trees_to_dataframe() output DataFrame."""
        defaults: dict[str, object] = {
            "tree_index": 0,
            "node_depth": 1,
            "node_index": "0-S",
            "left_child": None,
            "right_child": None,
            "parent_index": None,
            "split_feature": "odds",
            "split_gain": 1.0,
            "threshold": 5.0,
            "decision_type": "<=",
            "missing_direction": "left",
            "missing_type": "None",
            "value": 0.5,
            "weight": 100.0,
            "count": 100,
        }
        data = []
        for row in rows:
            merged = {**defaults, **row}
            data.append(merged)
        return pd.DataFrame(data)

    def test_gain_grouped_by_depth_and_category(self) -> None:
        """Gain is correctly grouped by depth and feature category."""
        from models.gpd_diagnostics import FEATURE_CATEGORY_MAP, _compute_depth_gains

        # Create tree data with known features at known depths
        tree_df = self._make_tree_df([
            {"node_depth": 1, "split_feature": "odds", "split_gain": 10.0},
            {"node_depth": 1, "split_feature": "odds", "split_gain": 5.0},
            {"node_depth": 2, "split_feature": "sire_wr", "split_gain": 3.0},
            {"node_depth": 3, "split_feature": "surface", "split_gain": 2.0},
        ])

        result = _compute_depth_gains(tree_df)

        # Verify structure
        assert "depths" in result
        assert "categories" in result
        assert "gains" in result
        assert "num_trees" in result
        assert "max_depth" in result
        assert "total_gain" in result

        # Market features (odds) at depth 1 should sum to 15.0
        depth1_market_gain = 0.0
        for i, d in enumerate(result["depths"]):
            if d == 1:
                cat = result["categories"][i]
                if cat == "market":
                    depth1_market_gain += result["gains"][i]

        assert depth1_market_gain == pytest.approx(15.0)

    def test_leaf_nodes_excluded(self) -> None:
        """Leaf nodes (split_feature=None) are excluded from gain aggregation."""
        from models.gpd_diagnostics import _compute_depth_gains

        tree_df = self._make_tree_df([
            {"node_depth": 1, "split_feature": "odds", "split_gain": 10.0},
            {"node_depth": 2, "split_feature": None, "split_gain": 0.0},  # leaf
        ])

        result = _compute_depth_gains(tree_df)

        # Only depth 1 should have entries; depth 2 leaf should be excluded
        depth2_entries = [
            i for i, d in enumerate(result["depths"]) if d == 2
        ]
        assert len(depth2_entries) == 0

    def test_nan_split_gain_filled_to_zero(self) -> None:
        """NaN split_gain values are filled with 0 before aggregation."""
        from models.gpd_diagnostics import _compute_depth_gains

        tree_df = self._make_tree_df([
            {"node_depth": 1, "split_feature": "odds", "split_gain": np.nan},
            {"node_depth": 1, "split_feature": "odds", "split_gain": 5.0},
        ])

        result = _compute_depth_gains(tree_df)

        # NaN should become 0, so total gain at depth 1 for market = 5.0
        depth1_market_gain = 0.0
        for i, d in enumerate(result["depths"]):
            if d == 1 and result["categories"][i] == "market":
                depth1_market_gain += result["gains"][i]

        assert depth1_market_gain == pytest.approx(5.0)

    def test_summary_statistics_correct(self) -> None:
        """num_trees, max_depth, total_gain are correctly computed."""
        from models.gpd_diagnostics import _compute_depth_gains

        tree_df = self._make_tree_df([
            {"tree_index": 0, "node_depth": 1, "split_feature": "odds", "split_gain": 10.0},
            {"tree_index": 0, "node_depth": 2, "split_feature": "sire_wr", "split_gain": 5.0},
            {"tree_index": 1, "node_depth": 1, "split_feature": "blood_keito_cd", "split_gain": 8.0},
        ])

        result = _compute_depth_gains(tree_df)

        assert result["num_trees"] == 2
        assert result["max_depth"] == 2
        assert result["total_gain"] == pytest.approx(23.0)


# ---------------------------------------------------------------------------
# Test 4: MDR computation
# ---------------------------------------------------------------------------


class TestMarketDominanceRatio:
    """Verify Market Dominance Ratio computation."""

    def _make_depth_gains(
        self,
        depth_category_gains: list[tuple[int, str, float]],
    ) -> dict:
        """Create a depth_gains dict structure from (depth, category, gain) tuples."""
        depths = [t[0] for t in depth_category_gains]
        categories = [t[1] for t in depth_category_gains]
        gains = [t[2] for t in depth_category_gains]
        return {
            "depths": depths,
            "categories": categories,
            "gains": gains,
            "num_trees": 5,
            "max_depth": max(depths) if depths else 0,
            "total_gain": sum(gains),
        }

    def test_positive_mdr_when_market_dominates_shallow(self) -> None:
        """MDR > 0 when Market dominates at shallow depths."""
        from models.gpd_diagnostics import _compute_market_dominance_ratio

        depth_gains = self._make_depth_gains([
            (1, "market", 100.0),
            (2, "market", 80.0),
            (3, "market", 60.0),
            (4, "fundamental", 90.0),
            (5, "fundamental", 80.0),
        ])

        mdr = _compute_market_dominance_ratio(depth_gains)
        assert mdr is not None
        assert mdr > 0  # Market dominates shallow, Fundamental dominates deep

    def test_negative_mdr_when_market_dominates_deep(self) -> None:
        """MDR < 0 when Market dominates at deeper depths."""
        from models.gpd_diagnostics import _compute_market_dominance_ratio

        depth_gains = self._make_depth_gains([
            (1, "fundamental", 100.0),
            (2, "fundamental", 80.0),
            (3, "fundamental", 60.0),
            (4, "market", 90.0),
            (5, "market", 80.0),
        ])

        mdr = _compute_market_dominance_ratio(depth_gains)
        assert mdr is not None
        assert mdr < 0

    def test_mdr_none_when_no_shallow_gain(self) -> None:
        """MDR returns None when total gain at depth 1-3 is zero."""
        from models.gpd_diagnostics import _compute_market_dominance_ratio

        depth_gains = self._make_depth_gains([
            (4, "market", 90.0),
            (5, "fundamental", 80.0),
        ])

        mdr = _compute_market_dominance_ratio(depth_gains)
        assert mdr is None


# ---------------------------------------------------------------------------
# Test 5: FAD computation
# ---------------------------------------------------------------------------


class TestFundamentalActivationDepth:
    """Verify Fundamental Activation Depth computation."""

    def _make_depth_gains(
        self,
        depth_category_gains: list[tuple[int, str, float]],
    ) -> dict:
        """Create a depth_gains dict structure."""
        depths = [t[0] for t in depth_category_gains]
        categories = [t[1] for t in depth_category_gains]
        gains = [t[2] for t in depth_category_gains]
        return {
            "depths": depths,
            "categories": categories,
            "gains": gains,
            "num_trees": 5,
            "max_depth": max(depths) if depths else 0,
            "total_gain": sum(gains),
        }

    def test_fad_returns_correct_depth(self) -> None:
        """FAD returns the depth where Fundamental first exceeds Market."""
        from models.gpd_diagnostics import _compute_fundamental_activation_depth

        depth_gains = self._make_depth_gains([
            (1, "market", 100.0),
            (2, "market", 60.0),
            (2, "fundamental", 40.0),
            (3, "fundamental", 80.0),
            (3, "market", 20.0),
        ])

        fad = _compute_fundamental_activation_depth(depth_gains)
        # At depth 1: Market=100, Fund=0 => Market > Fund
        # At depth 2: Market=60, Fund=40 => Market > Fund
        # At depth 3: Market=20, Fund=80 => Fund > Market => FAD = 3
        assert fad == 3

    def test_fad_none_when_market_always_dominates(self) -> None:
        """FAD returns None when Market dominates at all depths."""
        from models.gpd_diagnostics import _compute_fundamental_activation_depth

        depth_gains = self._make_depth_gains([
            (1, "market", 100.0),
            (2, "market", 80.0),
            (3, "market", 60.0),
        ])

        fad = _compute_fundamental_activation_depth(depth_gains)
        assert fad is None


# ---------------------------------------------------------------------------
# Test 6: Full pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """Verify compute_gpd_diagnostics() end-to-end with mock models."""

    def _make_mock_booster(self) -> MagicMock:
        """Create a mock lgb.Booster with controlled tree structure."""
        booster = MagicMock()
        booster.trees_to_dataframe.return_value = pd.DataFrame({
            "tree_index": [0, 0, 1],
            "node_depth": [1, 2, 1],
            "node_index": ["0-S0", "0-S1", "1-S0"],
            "left_child": ["0-S1", None, None],
            "right_child": ["0-S2", None, None],
            "parent_index": [None, "0-S0", None],
            "split_feature": ["odds", "sire_wr", "blood_keito_cd"],
            "split_gain": [10.0, 5.0, 3.0],
            "threshold": [5.0, 0.5, 2.0],
            "decision_type": ["<=", "<=", "<="],
            "missing_direction": ["left", "left", "left"],
            "missing_type": ["None", "None", "None"],
            "value": [0.5, 0.3, 0.2],
            "weight": [100.0, 50.0, 80.0],
            "count": [100, 50, 80],
        })
        return booster

    def _make_mock_models(self) -> MagicMock:
        """Create a minimal mock TrainedModelsV5."""
        mock_booster = self._make_mock_booster()

        sub = MagicMock()
        sub.stage1.models = {"turf": mock_booster}
        sub.win.hit_model = mock_booster
        sub.win.return_model = mock_booster
        sub.market.model = mock_booster
        sub.ev_corrector.p_correction_model = mock_booster
        sub.ev_corrector.e_correction_model = mock_booster
        sub.place = None
        sub.place_ev_corrector = None
        sub.wide = None
        sub.conformal_ev_model = None
        sub.place_ability = None

        models = MagicMock()
        models.submodels = {"turf": sub}
        return models

    def test_compute_gpd_diagnostics_returns_valid_dict(self) -> None:
        """compute_gpd_diagnostics() returns a dict with expected keys."""
        from models.gpd_diagnostics import compute_gpd_diagnostics

        models = self._make_mock_models()
        result = compute_gpd_diagnostics(models)

        assert isinstance(result, dict)
        assert "models" in result
        assert "metadata" in result

        # At least stage1_turf should be in the result
        assert len(result["models"]) > 0

        # Each model should have depth_gains, mdr, fad
        for model_name, model_data in result["models"].items():
            assert "depth_gains" in model_data, f"{model_name} missing depth_gains"
            assert "market_dominance_ratio" in model_data, f"{model_name} missing mdr"
            assert "fundamental_activation_depth" in model_data, f"{model_name} missing fad"
            assert "tier" in model_data, f"{model_name} missing tier"

    def test_json_output_written(self, tmp_path: Path) -> None:
        """JSON output file is written when output_dir is provided."""
        import json

        from models.gpd_diagnostics import compute_gpd_diagnostics

        models = self._make_mock_models()
        result = compute_gpd_diagnostics(models, output_dir=tmp_path)

        json_path = tmp_path / "gpd_report.json"
        assert json_path.exists()

        with open(json_path) as f:
            written = json.load(f)

        assert "models" in written

    def test_console_summary_runs_without_error(self) -> None:
        """console_summary() runs without error on valid result."""
        from models.gpd_diagnostics import compute_gpd_diagnostics, console_summary

        models = self._make_mock_models()
        result = compute_gpd_diagnostics(models)
        # Should not raise
        console_summary(result)
