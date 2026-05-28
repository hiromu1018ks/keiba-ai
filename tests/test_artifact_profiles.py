"""Tests for OOF artifact health profiles (CalibratorArtifactProfile, RankerArtifactProfile).

SAF-02: Phase 39/40 artifact-specific OOF validation profiles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from validation.artifact_profiles import (
    CalibratorArtifactProfile,
    DEFAULT_CALIBRATOR_PROFILE,
    DEFAULT_RANKER_PROFILE,
    PROFILES,
    RankerArtifactProfile,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_calibrator_oof() -> pd.DataFrame:
    """Build a minimal valid MAWC OOF DataFrame (2 races, 3 horses each)."""
    rows = []
    for race_id in ["R001", "R002"]:
        for i in range(3):
            rows.append({
                "race_id": race_id,
                "p_win_combined": 0.1 + i * 0.1,  # Will be overridden per race
                "p_win_final": 0.1 + i * 0.1,
                "fold": 0,
                "edge_win": 0.05,
            })
    df = pd.DataFrame(rows)
    # Fix probabilities to sum to 1.0 per race
    for race_id in ["R001", "R002"]:
        mask = df["race_id"] == race_id
        probs = [0.5, 0.3, 0.2]
        df.loc[mask, "p_win_combined"] = probs
        df.loc[mask, "p_win_final"] = probs
    return df


def _make_valid_ranker_oof() -> pd.DataFrame:
    """Build a minimal valid Ranker OOF DataFrame (2 races, 3 horses each)."""
    rows = []
    for race_id in ["R001", "R002"]:
        for i in range(3):
            rows.append({
                "race_id": race_id,
                "investment_score": 0.8 - i * 0.2,
                "relevance_score": 0.7 - i * 0.15,
                "value_score": 0.6 - i * 0.1,
                "fold": 0,
            })
    return pd.DataFrame(rows)


# ===========================================================================
# CalibratorArtifactProfile tests
# ===========================================================================


class TestCalibratorArtifactProfile:
    """Tests for CalibratorArtifactProfile.validate()."""

    def test_valid_oof_returns_no_failures(self) -> None:
        """Test 1: Valid MAWC OOF DataFrame returns empty failures list."""
        df = _make_valid_calibrator_oof()
        profile = CalibratorArtifactProfile()
        failures = profile.validate(df)
        assert failures == []

    def test_nan_in_p_win_combined(self) -> None:
        """Test 2: NaN in p_win_combined produces failure with 'NaN' and column name."""
        df = _make_valid_calibrator_oof()
        df.loc[0, "p_win_combined"] = np.nan
        profile = CalibratorArtifactProfile()
        failures = profile.validate(df)
        assert len(failures) > 0
        assert any("NaN" in f and "p_win_combined" in f for f in failures)

    def test_inf_in_p_win_final(self) -> None:
        """Test 3: inf in p_win_final produces failure with 'inf' and column name."""
        df = _make_valid_calibrator_oof()
        df.loc[0, "p_win_final"] = np.inf
        profile = CalibratorArtifactProfile()
        failures = profile.validate(df)
        assert len(failures) > 0
        assert any("inf" in f and "p_win_final" in f for f in failures)

    def test_probability_above_one(self) -> None:
        """Test 4: Probability > 1.0 in p_win_combined produces failure with 'range'."""
        df = _make_valid_calibrator_oof()
        df.loc[0, "p_win_combined"] = 1.5
        profile = CalibratorArtifactProfile()
        failures = profile.validate(df)
        assert len(failures) > 0
        assert any("range" in f.lower() and "p_win_combined" in f for f in failures)

    def test_probability_below_zero(self) -> None:
        """Test 5: Probability < 0.0 in p_win_final produces failure with 'range'."""
        df = _make_valid_calibrator_oof()
        df.loc[0, "p_win_final"] = -0.1
        profile = CalibratorArtifactProfile()
        failures = profile.validate(df)
        assert len(failures) > 0
        assert any("range" in f.lower() and "p_win_final" in f for f in failures)

    def test_sum_to_one_violation(self) -> None:
        """Test 6: Sum-to-1.0 violation per race_id produces failure with 'sum-to-1'."""
        df = _make_valid_calibrator_oof()
        # Make first race sum to 2.0
        mask = df["race_id"] == "R001"
        df.loc[mask, "p_win_combined"] = [0.8, 0.7, 0.5]  # sum = 2.0
        profile = CalibratorArtifactProfile()
        failures = profile.validate(df)
        assert len(failures) > 0
        assert any("sum-to-1" in f.lower() for f in failures)

    def test_forbidden_p_win_pred_column(self) -> None:
        """Test 7: p_win_pred column present produces failure with 'p_win_pred' and 'forbidden'."""
        df = _make_valid_calibrator_oof()
        df["p_win_pred"] = 0.5
        profile = CalibratorArtifactProfile()
        failures = profile.validate(df)
        assert len(failures) > 0
        assert any("p_win_pred" in f and "forbidden" in f.lower() for f in failures)

    def test_missing_fold_column(self) -> None:
        """Test 8: Missing fold column produces failure with 'fold' and 'required'."""
        df = _make_valid_calibrator_oof()
        df = df.drop(columns=["fold"])
        profile = CalibratorArtifactProfile()
        failures = profile.validate(df)
        assert len(failures) > 0
        assert any("fold" in f.lower() and "required" in f.lower() for f in failures)

    def test_missing_race_id_column(self) -> None:
        """Test 9: Missing race_id column produces failure with 'race_id' and 'required'."""
        df = _make_valid_calibrator_oof()
        df = df.drop(columns=["race_id"])
        profile = CalibratorArtifactProfile()
        failures = profile.validate(df)
        assert len(failures) > 0
        assert any("race_id" in f and "required" in f.lower() for f in failures)


# ===========================================================================
# RankerArtifactProfile tests
# ===========================================================================


class TestRankerArtifactProfile:
    """Tests for RankerArtifactProfile.validate()."""

    def test_valid_oof_returns_no_failures(self) -> None:
        """Test 10: Valid ranker OOF DataFrame returns empty failures list."""
        df = _make_valid_ranker_oof()
        profile = RankerArtifactProfile()
        failures = profile.validate(df)
        assert failures == []

    def test_nan_in_investment_score(self) -> None:
        """Test 11: NaN in investment_score produces failure with 'NaN'."""
        df = _make_valid_ranker_oof()
        df.loc[0, "investment_score"] = np.nan
        profile = RankerArtifactProfile()
        failures = profile.validate(df)
        assert len(failures) > 0
        assert any("NaN" in f and "investment_score" in f for f in failures)

    def test_inf_in_relevance_score(self) -> None:
        """Test 12: inf in relevance_score produces failure with 'inf'."""
        df = _make_valid_ranker_oof()
        df.loc[0, "relevance_score"] = np.inf
        profile = RankerArtifactProfile()
        failures = profile.validate(df)
        assert len(failures) > 0
        assert any("inf" in f and "relevance_score" in f for f in failures)

    def test_nan_in_value_score(self) -> None:
        """Test 13: NaN in value_score produces failure with 'NaN'."""
        df = _make_valid_ranker_oof()
        df.loc[0, "value_score"] = np.nan
        profile = RankerArtifactProfile()
        failures = profile.validate(df)
        assert len(failures) > 0
        assert any("NaN" in f and "value_score" in f for f in failures)

    def test_non_deterministic_race_ranks_warning(self) -> None:
        """Test 14: Non-deterministic race-level ranks produces warning string."""
        df = _make_valid_ranker_oof()
        # Make all investment_scores identical in race R001
        mask = df["race_id"] == "R001"
        df.loc[mask, "investment_score"] = 0.5
        profile = RankerArtifactProfile()
        failures = profile.validate(df)
        # This should produce a WARNING about rank determinism
        assert any("WARNING" in f and ("determinis" in f.lower() or "rank" in f.lower()) for f in failures)

    def test_missing_fold_column(self) -> None:
        """Test 15: Missing fold column produces failure with 'fold' and 'required'."""
        df = _make_valid_ranker_oof()
        df = df.drop(columns=["fold"])
        profile = RankerArtifactProfile()
        failures = profile.validate(df)
        assert len(failures) > 0
        assert any("fold" in f.lower() and "required" in f.lower() for f in failures)

    def test_missing_race_id_column(self) -> None:
        """Test 16: Missing race_id column produces failure with 'race_id' and 'required'."""
        df = _make_valid_ranker_oof()
        df = df.drop(columns=["race_id"])
        profile = RankerArtifactProfile()
        failures = profile.validate(df)
        assert len(failures) > 0
        assert any("race_id" in f and "required" in f.lower() for f in failures)


# ===========================================================================
# PROFILES registry tests
# ===========================================================================


class TestProfilesRegistry:
    """Tests for PROFILES registry dict."""

    def test_calibrator_in_registry(self) -> None:
        """PROFILES['calibrator'] is CalibratorArtifactProfile."""
        assert PROFILES["calibrator"] is CalibratorArtifactProfile

    def test_ranker_in_registry(self) -> None:
        """PROFILES['ranker'] is RankerArtifactProfile."""
        assert PROFILES["ranker"] is RankerArtifactProfile

    def test_default_instances(self) -> None:
        """Default instances are correct types."""
        assert isinstance(DEFAULT_CALIBRATOR_PROFILE, CalibratorArtifactProfile)
        assert isinstance(DEFAULT_RANKER_PROFILE, RankerArtifactProfile)
