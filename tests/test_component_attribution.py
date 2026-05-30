"""ComponentAttribution unit tests (BISECT-01/02)

Tests the 4-step sequential attribution engine that reads Phase 41/43/42 artifacts
and attributes each DeploymentGate FAIL to a specific component (MAWC / Ranker / OBF).
Also tests coefficient analysis for MAWC (51-dim) and Ranker (Ridge) models.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backtest.component_attribution import (
    CoefficientAnalysisResult,
    ComponentAttribution,
    ComponentAttributionResult,
)

# ---------------------------------------------------------------------------
# Constants matching actual artifact structure
# ---------------------------------------------------------------------------

_BASELINE = "baseline"
_SHADOW = "ridge_shadow"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_horse_df(
    *,
    n_races: int = 10,
    horses_per_race: int = 8,
    shadow_name: str = _SHADOW,
) -> pd.DataFrame:
    """Synthetic horse_diff DataFrame matching shadow_horse_diff.parquet schema (18 cols)."""
    rows: list[dict] = []
    np.random.seed(42)

    for r in range(n_races):
        race_id = f"2024{r + 1:02d}010101"
        for h in range(horses_per_race):
            umaban = h + 1
            bl_p = round(max(0.01, min(0.99, np.random.uniform(0.02, 0.45))), 4)
            # shadow p_win slightly different (MAWC effect)
            sh_p = round(max(0.01, min(0.99, bl_p + np.random.uniform(-0.08, 0.08))), 4)
            rows.append({
                "race_id": race_id,
                "umaban": umaban,
                f"{_BASELINE}_p_win_final": bl_p,
                f"{_BASELINE}_investment_score": round(np.random.uniform(0.1, 0.9), 4),
                f"{_BASELINE}_stake": 100.0 if h < 2 else 0.0,
                f"{_BASELINE}_win_market_selection_score": round(np.random.uniform(0.2, 0.8), 4),
                f"{_BASELINE}_selected": h < 2,
                f"{shadow_name}_p_win_final": sh_p,
                f"{shadow_name}_investment_score": round(np.random.uniform(0.1, 0.9), 4),
                f"{shadow_name}_stake": 100.0 if h == 0 else 0.0,
                f"{shadow_name}_win_market_selection_score": round(np.random.uniform(0.2, 0.8), 4),
                f"{shadow_name}_selected": h == 0,
                "kakuteijyuni": h + 1,
                "surface": "turf" if r % 2 == 0 else "dirt",
                "tanodds": round(1.5 + h * 2.5, 1),
                "closing_win_odds": round(1.6 + h * 2.5, 1),
                "popularity": h + 1,
                "fold_year": 2024,
            })

    return pd.DataFrame(rows)


def _make_race_df(
    *,
    n_races: int = 10,
) -> pd.DataFrame:
    """Synthetic race_diff DataFrame matching shadow_race_diff.parquet schema (21 cols)."""
    rows: list[dict] = []
    for r in range(n_races):
        race_id = f"2024{r + 1:02d}010101"
        changed = r < 5  # first 5 races are changed
        rows.append({
            "race_id": race_id,
            "baseline_selected_umaban": 1,
            "shadow_selected_umaban": 2 if changed else 1,
            "selected_changed": changed,
            "baseline_result": 150.0 if r == 0 else 0.0,
            "shadow_result": 100.0 if r == 1 else 0.0,
            "baseline_stake": 100.0,
            "shadow_stake": 100.0,
            "baseline_tanodds": 3.5,
            "shadow_tanodds": 5.2,
            "baseline_p_win_final": 0.25,
            "shadow_p_win_final": 0.20,
            "baseline_win_selection_ev": 0.8,
            "shadow_win_selection_ev": 0.7,
            "baseline_win_market_selection_score": 0.6,
            "shadow_win_market_selection_score": 0.5,
            "baseline_closing_win_odds": 3.6,
            "shadow_closing_win_odds": 5.3,
            "baseline_investment_score": 0.7,
            "shadow_investment_score": 0.65,
            "fold_year": 2024,
        })
    return pd.DataFrame(rows)


def _make_diagnosis_result() -> dict:
    """Minimal shadow_diagnosis_result.json."""
    return {
        "generated_at": "2026-05-30T00:00:00Z",
        "step1_probability_quality": {
            "baseline": {"ece": 0.0093, "actual_predicted_ratio": 0.928},
            "shadow": {"ece": 0.0156, "actual_predicted_ratio": 1.154},
        },
        "step3_calibration": {
            "segments": [
                {
                    "segment_name": "odds_band",
                    "segment_value": "1-3",
                    "baseline_ece": 0.0447,
                    "shadow_ece": 0.1444,
                },
            ],
        },
    }


def _make_gate_result() -> dict:
    """Minimal deployment_gate_result.json."""
    return {
        "gate_results": [
            {"gate_id": "ece_fold_2025", "status": "FAIL"},
            {"gate_id": "bet_count_preservation_fold_2024", "status": "FAIL"},
            {"gate_id": "bet_count_preservation_fold_2025", "status": "FAIL"},
        ],
        "baseline_metrics": {
            "2024": {"bet_count": 3327},
            "2025": {"bet_count": 3335},
        },
        "shadow_metrics": {
            "2024": {"bet_count": 2580},
            "2025": {"bet_count": 2550},
        },
    }


def _setup_input_dir(
    tmp_path: Path,
    *,
    horse_df: pd.DataFrame | None = None,
    race_df: pd.DataFrame | None = None,
    diagnosis: dict | None = None,
    gate_result: dict | None = None,
) -> Path:
    """Set up input directory with fixture data."""
    input_dir = tmp_path / "shadow_input"
    input_dir.mkdir(parents=True, exist_ok=True)

    if horse_df is None:
        horse_df = _make_horse_df()
    if race_df is None:
        race_df = _make_race_df()
    if diagnosis is None:
        diagnosis = _make_diagnosis_result()
    if gate_result is None:
        gate_result = _make_gate_result()

    horse_df.to_parquet(input_dir / "shadow_horse_diff.parquet", index=False)
    race_df.to_parquet(input_dir / "shadow_race_diff.parquet", index=False)

    diag_dir = input_dir / "diagnosis"
    diag_dir.mkdir(parents=True, exist_ok=True)
    (diag_dir / "shadow_diagnosis_result.json").write_text(
        json.dumps(diagnosis, indent=2), encoding="utf-8"
    )

    gates_dir = input_dir / "gates"
    gates_dir.mkdir(parents=True, exist_ok=True)
    (gates_dir / "deployment_gate_result.json").write_text(
        json.dumps(gate_result, indent=2), encoding="utf-8"
    )

    return input_dir


def _make_mawc_state() -> dict:
    """Synthetic MAWC model state for mocking joblib.load."""
    # 51 features: 6 main + 15 one-hot + 30 interactions
    feature_names = [
        "logit_model", "logit_market", "log_odds",
        "popularity_rank_pct", "p_win_race_rank_pct", "field_size",
        # 7 odds_band
        "1-2", "2-3", "3-5", "5-10", "10-30", "30-100", "100+",
        # 5 pop_bucket
        "pop_1", "pop_2_3", "pop_4_6", "pop_7_9", "pop_10_plus",
        # 3 p_rank
        "top_25", "mid_25_75", "bottom_25",
        # 15 logit_model interactions
        *["logit_model_x_1-2", "logit_model_x_2-3", "logit_model_x_3-5",
          "logit_model_x_5-10", "logit_model_x_10-30", "logit_model_x_30-100",
          "logit_model_x_100+", "logit_model_x_pop_1", "logit_model_x_pop_2_3",
          "logit_model_x_pop_4_6", "logit_model_x_pop_7_9",
          "logit_model_x_pop_10_plus", "logit_model_x_top_25",
          "logit_model_x_mid_25_75", "logit_model_x_bottom_25"],
        # 15 logit_market interactions
        *["logit_market_x_1-2", "logit_market_x_2-3", "logit_market_x_3-5",
          "logit_market_x_5-10", "logit_market_x_10-30", "logit_market_x_30-100",
          "logit_market_x_100+", "logit_market_x_pop_1", "logit_market_x_pop_2_3",
          "logit_market_x_pop_4_6", "logit_market_x_pop_7_9",
          "logit_market_x_pop_10_plus", "logit_market_x_top_25",
          "logit_market_x_mid_25_75", "logit_market_x_bottom_25"],
    ]
    assert len(feature_names) == 51, f"Expected 51 features, got {len(feature_names)}"

    # Simulate actual coefficient magnitudes
    coef = np.zeros(51)
    coef[0] = 0.0435   # logit_model
    coef[1] = 0.3911   # logit_market (dominant)
    coef[2] = -0.3571  # log_odds
    # odds_band 1-2 interaction with market
    coef[21 + 7 + 0] = 0.05  # logit_market_x_1-2

    mock_calibrator = MagicMock()
    mock_calibrator.coef_ = coef.reshape(1, -1)
    mock_calibrator.intercept_ = np.array([-2.5])

    return {
        "calibrator": mock_calibrator,
        "feature_names": feature_names,
        "training_summary": {
            "beta_market_contribution": 0.90,
            "best_c": 0.03,
        },
    }


def _make_ranker_state() -> dict:
    """Synthetic Ranker model state for mocking joblib.load."""
    rel_features = [
        "if_p_win_final", "if_p_win_race_rank", "if_p_ability_win",
        "rel_p_ability_win_rank", "if_norm_finish_avg", "if_closing_index",
        "if_weighted_recent_form", "if_jockey_wr", "if_trainer_wr",
        "if_blood_surface_wr", "if_class_level", "if_surface",
        "if_distance_bin", "if_grade_code", "if_n_horses",
    ]
    val_features = [
        "if_logit_gap", "if_edge_win", "if_ev_calibrated",
        "if_odds_log", "if_odds_band_id", "if_odds_drop_60_10",
        "if_odds_drop_30_10", "if_overround", "if_market_entropy",
        "if_conformal_width", "if_ev_uncertainty_ratio", "if_p_win_race_rank",
        "if_n_horses", "if_odds_rank", "if_abs_logit_gap",
    ]

    rel_coef = np.zeros(15)
    rel_coef[0] = 0.80  # if_p_win_final dominant
    rel_coef[2] = 0.27  # if_p_ability_win

    val_coef = np.zeros(15)
    val_coef[2] = 0.83  # if_ev_calibrated dominant
    val_coef[7] = -0.74  # if_overround

    mock_rel = MagicMock()
    mock_rel.coef_ = rel_coef.reshape(1, -1)
    mock_rel.intercept_ = np.array([0.1])

    mock_val = MagicMock()
    mock_val.coef_ = val_coef.reshape(1, -1)
    mock_val.intercept_ = np.array([-0.2])

    return {
        "relevance_scorer_turf": mock_rel,
        "relevance_scorer_dirt": None,
        "value_scorer_turf": mock_val,
        "value_scorer_dirt": None,
        "relevance_feature_names": rel_features,
        "value_feature_names": val_features,
        "training_summary": {"deployment_status": "shadow_only"},
    }


# ---------------------------------------------------------------------------
# Test 1: Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    """Test ComponentAttribution loads artifacts and initializes."""

    def test_loads_all_artifacts_without_error(self, tmp_path: Path) -> None:
        input_dir = _setup_input_dir(tmp_path)
        ca = ComponentAttribution(input_dir=input_dir)

        assert ca.horse_diff is not None
        assert ca.race_diff is not None
        assert ca.diagnosis is not None
        assert ca.gate_result is not None
        assert len(ca.horse_diff) > 0
        assert len(ca.race_diff) > 0

    def test_resolves_variant_names(self, tmp_path: Path) -> None:
        input_dir = _setup_input_dir(tmp_path)
        ca = ComponentAttribution(input_dir=input_dir)

        assert ca.baseline_name == _BASELINE
        assert ca.shadow_name == _SHADOW


# ---------------------------------------------------------------------------
# Test 2: ECE Degradation Attribution
# ---------------------------------------------------------------------------


class TestECEDegradationAttribution:
    """Test attribute_ece_degradation() per-segment analysis."""

    def test_returns_per_segment_attribution(self, tmp_path: Path) -> None:
        input_dir = _setup_input_dir(tmp_path)
        ca = ComponentAttribution(input_dir=input_dir)
        result = ca.attribute_ece_degradation()

        assert "segments" in result
        segments = result["segments"]
        assert len(segments) > 0

        # Each segment should have baseline_ece, shadow_ece, delta_ece
        for seg in segments:
            assert "segment_name" in seg
            assert "segment_value" in seg
            assert "baseline_ece" in seg
            assert "shadow_ece" in seg
            assert "delta_ece" in seg
            assert "mean_p_win_shift" in seg

    def test_identifies_mawc_direct_effect(self, tmp_path: Path) -> None:
        """When ECE worsens AND p_win shift is significant, attribute to MAWC."""
        input_dir = _setup_input_dir(tmp_path)
        ca = ComponentAttribution(input_dir=input_dir)
        result = ca.attribute_ece_degradation()

        assert "attribution" in result
        # Should contain MAWC direct effect or selection population effect
        assert any(
            "MAWC" in a or "selection" in a.lower()
            for a in result["attribution"]
        )

    def test_includes_actual_predicted_ratio(self, tmp_path: Path) -> None:
        input_dir = _setup_input_dir(tmp_path)
        ca = ComponentAttribution(input_dir=input_dir)
        result = ca.attribute_ece_degradation()

        for seg in result["segments"]:
            assert "baseline_apr" in seg
            assert "shadow_apr" in seg


# ---------------------------------------------------------------------------
# Test 3: APR Deviation Attribution
# ---------------------------------------------------------------------------


class TestAPRDeviationAttribution:
    """Test attribute_apr_deviation() all-horse vs selected-horse separation."""

    def test_separates_all_horse_and_selected_horse_apr(self, tmp_path: Path) -> None:
        input_dir = _setup_input_dir(tmp_path)
        ca = ComponentAttribution(input_dir=input_dir)
        result = ca.attribute_apr_deviation()

        assert "all_horse_apr" in result
        assert "selected_horse_apr" in result

        all_horse = result["all_horse_apr"]
        assert "baseline_apr" in all_horse
        assert "shadow_apr" in all_horse
        assert "delta_apr" in all_horse

        selected = result["selected_horse_apr"]
        assert "baseline_apr" in selected
        assert "shadow_apr" in selected

    def test_attributes_apr_correctly(self, tmp_path: Path) -> None:
        """Should attribute APR to MAWC probability level or Ranker selection."""
        input_dir = _setup_input_dir(tmp_path)
        ca = ComponentAttribution(input_dir=input_dir)
        result = ca.attribute_apr_deviation()

        assert "attribution" in result
        assert len(result["attribution"]) > 0


# ---------------------------------------------------------------------------
# Test 4: Bet Count Loss Attribution
# ---------------------------------------------------------------------------


class TestBetCountLossAttribution:
    """Test attribute_bet_count_loss() decomposition."""

    def test_decomposes_bet_count_gap(self, tmp_path: Path) -> None:
        input_dir = _setup_input_dir(tmp_path)
        ca = ComponentAttribution(input_dir=input_dir)
        result = ca.attribute_bet_count_loss()

        assert "baseline_bet_count" in result
        assert "shadow_bet_count" in result
        assert "gap" in result
        assert "ranker_exclusion" in result
        assert "selection_changed_count" in result

    def test_quantifies_ranker_exclusion(self, tmp_path: Path) -> None:
        """Count races where selection changed = Ranker-driven exclusions."""
        input_dir = _setup_input_dir(tmp_path)
        ca = ComponentAttribution(input_dir=input_dir)
        result = ca.attribute_bet_count_loss()

        # Our fixture has 5 changed races out of 10
        assert result["selection_changed_count"] == 5

    def test_includes_obf_analysis(self, tmp_path: Path) -> None:
        """Per D-04, OBF analysis integrated into bet_count step."""
        input_dir = _setup_input_dir(tmp_path)
        ca = ComponentAttribution(input_dir=input_dir)
        result = ca.attribute_bet_count_loss()

        assert "obf_analysis" in result


# ---------------------------------------------------------------------------
# Test 5: MAWC Coefficient Analysis
# ---------------------------------------------------------------------------


class TestMAWCCoefficientAnalysis:
    """Test analyze_mawc_coefficients() 51-dim extraction."""

    @patch("backtest.component_attribution.joblib.load")
    def test_extracts_51_dim_coefficients(self, mock_load, tmp_path: Path) -> None:
        mock_load.return_value = _make_mawc_state()

        input_dir = _setup_input_dir(tmp_path)
        ca = ComponentAttribution(input_dir=input_dir)
        result = ca.analyze_mawc_coefficients()

        assert "feature_coefficients" in result
        assert len(result["feature_coefficients"]) == 51

    @patch("backtest.component_attribution.joblib.load")
    def test_identifies_dominant_features(self, mock_load, tmp_path: Path) -> None:
        mock_load.return_value = _make_mawc_state()

        input_dir = _setup_input_dir(tmp_path)
        ca = ComponentAttribution(input_dir=input_dir)
        result = ca.analyze_mawc_coefficients()

        # logit_market should be identified as dominant
        assert "top_features" in result
        top_names = [f["feature"] for f in result["top_features"]]
        assert "logit_market" in top_names

    @patch("backtest.component_attribution.joblib.load")
    def test_computes_per_segment_contribution(self, mock_load, tmp_path: Path) -> None:
        mock_load.return_value = _make_mawc_state()

        input_dir = _setup_input_dir(tmp_path)
        ca = ComponentAttribution(input_dir=input_dir)
        result = ca.analyze_mawc_coefficients()

        assert "segment_contributions" in result


# ---------------------------------------------------------------------------
# Test 6: Ranker Coefficient Analysis
# ---------------------------------------------------------------------------


class TestRankerCoefficientAnalysis:
    """Test analyze_ranker_coefficients() Ridge coef extraction."""

    @patch("backtest.component_attribution.joblib.load")
    def test_extracts_relevance_and_value_coefficients(
        self, mock_load, tmp_path: Path
    ) -> None:
        mock_load.return_value = _make_ranker_state()

        input_dir = _setup_input_dir(tmp_path)
        ca = ComponentAttribution(input_dir=input_dir)
        result = ca.analyze_ranker_coefficients()

        assert "relevance_coefficients" in result
        assert "value_coefficients" in result

    @patch("backtest.component_attribution.joblib.load")
    def test_identifies_dominant_features(self, mock_load, tmp_path: Path) -> None:
        mock_load.return_value = _make_ranker_state()

        input_dir = _setup_input_dir(tmp_path)
        ca = ComponentAttribution(input_dir=input_dir)
        result = ca.analyze_ranker_coefficients()

        # if_p_win_final should dominate relevance, if_ev_calibrated should dominate value
        rel_top = result["relevance_top_features"]
        val_top = result["value_top_features"]

        rel_names = [f["feature"] for f in rel_top]
        val_names = [f["feature"] for f in val_top]

        assert "if_p_win_final" in rel_names
        assert "if_ev_calibrated" in val_names


# ---------------------------------------------------------------------------
# Test 7: Segment Coefficient Contribution Comparison
# ---------------------------------------------------------------------------


class TestSegmentCoefficientContribution:
    """Test analyze_segment_coefficient_contribution() changed/dropped/retained."""

    @patch("backtest.component_attribution.joblib.load")
    def test_splits_into_groups(self, mock_load, tmp_path: Path) -> None:
        mock_load.side_effect = lambda p: (
            _make_mawc_state() if "calibrator" in str(p) else _make_ranker_state()
        )

        input_dir = _setup_input_dir(tmp_path)
        ca = ComponentAttribution(input_dir=input_dir)
        result = ca.analyze_segment_coefficient_contribution()

        assert "changed" in result
        assert "dropped" in result
        assert "retained" in result

    @patch("backtest.component_attribution.joblib.load")
    def test_compares_mawc_and_ranker_contributions(
        self, mock_load, tmp_path: Path
    ) -> None:
        mock_load.side_effect = lambda p: (
            _make_mawc_state() if "calibrator" in str(p) else _make_ranker_state()
        )

        input_dir = _setup_input_dir(tmp_path)
        ca = ComponentAttribution(input_dir=input_dir)
        result = ca.analyze_segment_coefficient_contribution()

        for group in ["changed", "dropped", "retained"]:
            assert "mean_p_win_delta" in result[group]
            assert "n_races" in result[group]


# ---------------------------------------------------------------------------
# Test 8: Full Attribution Sequence
# ---------------------------------------------------------------------------


class TestFullAttribution:
    """Test run_full_attribution() ECE -> APR -> bet_count -> OBF sequence."""

    @patch("backtest.component_attribution.joblib.load")
    def test_returns_complete_result(self, mock_load, tmp_path: Path) -> None:
        mock_load.side_effect = lambda p: (
            _make_mawc_state() if "calibrator" in str(p) else _make_ranker_state()
        )

        input_dir = _setup_input_dir(tmp_path)
        ca = ComponentAttribution(input_dir=input_dir)
        result = ca.run_full_attribution()

        assert isinstance(result, ComponentAttributionResult)
        assert result.ece_attribution is not None
        assert result.apr_attribution is not None
        assert result.bet_count_attribution is not None
        assert result.coefficient_analysis is not None
        assert isinstance(result.upstream_anomaly_check, str)
        assert isinstance(result.recommendations, list)

    @patch("backtest.component_attribution.joblib.load")
    def test_includes_upstream_anomaly_check(self, mock_load, tmp_path: Path) -> None:
        mock_load.side_effect = lambda p: (
            _make_mawc_state() if "calibrator" in str(p) else _make_ranker_state()
        )

        input_dir = _setup_input_dir(tmp_path)
        ca = ComponentAttribution(input_dir=input_dir)
        result = ca.run_full_attribution()

        # Should have documented upstream anomaly check result
        assert result.upstream_anomaly_check != ""
