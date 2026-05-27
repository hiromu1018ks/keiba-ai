"""Leakage detection module tests (38-01 Task 2).

TDD RED: 以下テストは実装前に実行すると全て失敗する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from investment.schema_registry import InvestmentFeatureSpec


class TestValidateNoPostRaceLeakage:
    """Tests for validate_no_post_race_leakage()."""

    def test_raises_on_post_race_column(self) -> None:
        """Test 1: raises ValueError mentioning 'kakuteijyuni'."""
        from investment.leakage import validate_no_post_race_leakage

        with pytest.raises(ValueError, match="kakuteijyuni"):
            validate_no_post_race_leakage(["race_id", "if_p_win", "kakuteijyuni"])

    def test_passes_on_clean_columns(self) -> None:
        """Test 2: passes without error when no POST_RACE columns."""
        from investment.leakage import validate_no_post_race_leakage

        # Should not raise
        validate_no_post_race_leakage(["race_id", "if_p_win"])


class TestValidateOofSafeSources:
    """Tests for validate_oof_safe_sources()."""

    def test_detects_in_sample_only_sources(self) -> None:
        """Test 3: returns violation for spec with in-sample-only train_sources."""
        from investment.leakage import validate_oof_safe_sources

        specs = {
            "if_bad": InvestmentFeatureSpec(
                name="if_bad",
                category="model_prob",
                dtype="float64",
                train_sources=("p_win_pred",),
                infer_sources=("p_win_pred",),
                required=True,
                default_value=None,
                missing_indicator=None,
                leakage_class="safe",
                description="bad spec with in-sample source",
            ),
        }
        violations = validate_oof_safe_sources(specs)
        assert len(violations) == 1
        assert "if_bad" in violations[0]

    def test_returns_empty_for_safe_sources(self) -> None:
        """Test 4: returns empty list when all sources are OOF-safe."""
        from investment.leakage import validate_oof_safe_sources

        specs = {
            "if_safe": InvestmentFeatureSpec(
                name="if_safe",
                category="model_prob",
                dtype="float64",
                train_sources=("p_win_oof",),
                infer_sources=("p_win_pred",),
                required=True,
                default_value=None,
                missing_indicator=None,
                leakage_class="safe",
                description="safe spec",
            ),
        }
        violations = validate_oof_safe_sources(specs)
        assert violations == []


class TestValidateSchemaIdentity:
    """Tests for validate_schema_identity()."""

    def test_passes_on_matching_schema(self) -> None:
        """Test 5: passes when columns and dtypes match."""
        from investment.leakage import validate_schema_identity

        train_df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        infer_df = pd.DataFrame({"a": [3.0], "b": [4.0]})
        # Should not raise
        validate_schema_identity(train_df, infer_df)

    def test_raises_on_different_column_order(self) -> None:
        """Test 6: raises AssertionError on different column order."""
        from investment.leakage import validate_schema_identity

        train_df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        infer_df = pd.DataFrame({"b": [4.0], "a": [3.0]})
        with pytest.raises(AssertionError, match="[Cc]olumn"):
            validate_schema_identity(train_df, infer_df)

    def test_raises_on_different_dtype(self) -> None:
        """Test 7: raises AssertionError on different dtype."""
        from investment.leakage import validate_schema_identity

        train_df = pd.DataFrame({"a": [1.0], "b": [2]})
        infer_df = pd.DataFrame({"a": [3.0], "b": [4.0]})
        with pytest.raises(AssertionError, match="[Dd]type|dtype"):
            validate_schema_identity(train_df, infer_df)


class TestInSampleOnlyCols:
    """Test 8: IN_SAMPLE_ONLY_COLS constant."""

    def test_contains_expected_cols(self) -> None:
        from investment.leakage import IN_SAMPLE_ONLY_COLS

        expected = {"p_win_pred", "ev_win", "ev_win_calibrated", "p_win_final", "edge_win"}
        assert expected.issubset(IN_SAMPLE_ONLY_COLS)

    def test_is_frozenset(self) -> None:
        from investment.leakage import IN_SAMPLE_ONLY_COLS

        assert isinstance(IN_SAMPLE_ONLY_COLS, frozenset)
