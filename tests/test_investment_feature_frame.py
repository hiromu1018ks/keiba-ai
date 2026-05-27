"""Tests for investment feature frame builder (feature_frame.py).

TDD RED phase: tests for InvestmentFeatureFrameBuilder per D-10~D-19,
IFF-01, IFF-02, IFF-03.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from investment.feature_frame import (
    InvestmentFeatureFrameBuilder,
    build_frame,
)
from investment.schema_registry import ALL_IF_COLUMNS, CATEGORY_ORDER, FEATURE_SPECS


# ---------------------------------------------------------------------------
# Helper: build a DataFrame with all required source columns
# ---------------------------------------------------------------------------


def _make_train_source_df(n: int = 20) -> pd.DataFrame:
    """Build a DataFrame with all train-mode source columns populated."""
    np.random.seed(42)
    data: dict[str, list[float] | list[str] | list[int]] = {
        "race_id": [f"R{i:04d}" for i in range(n)],
        "umaban": list(range(1, n + 1)),
    }
    # Collect all unique train source columns
    train_sources: set[str] = set()
    for spec in FEATURE_SPECS.values():
        train_sources.update(spec.train_sources)
    for col in sorted(train_sources):
        data[col] = list(np.random.randn(n))
    return pd.DataFrame(data)


def _make_infer_source_df(n: int = 20) -> pd.DataFrame:
    """Build a DataFrame with all infer-mode source columns populated."""
    np.random.seed(43)
    data: dict[str, list[float] | list[str] | list[int]] = {
        "race_id": [f"R{i:04d}" for i in range(n)],
        "umaban": list(range(1, n + 1)),
    }
    # Collect all unique infer source columns
    infer_sources: set[str] = set()
    for spec in FEATURE_SPECS.values():
        infer_sources.update(spec.infer_sources)
    for col in sorted(infer_sources):
        data[col] = list(np.random.randn(n))
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildFrameTrainMode:
    """Tests for build_frame(df, mode='train')."""

    def test_produces_if_columns_in_range(self) -> None:
        """Test 1: train mode produces 90-130 'if_*' columns (excluding _missing)."""
        df = _make_train_source_df()
        builder = InvestmentFeatureFrameBuilder()
        result = builder.build_frame(df, mode="train")

        # Count feature columns (exclude _missing indicators)
        if_cols = [
            c for c in result.columns
            if c.startswith("if_") and not c.endswith("_missing")
        ]
        assert 90 <= len(if_cols) <= 130, (
            f"Expected 90-130 if_* feature columns, got {len(if_cols)}"
        )

    def test_output_contains_identity_columns(self) -> None:
        """Output should contain race_id and umaban."""
        df = _make_train_source_df()
        builder = InvestmentFeatureFrameBuilder()
        result = builder.build_frame(df, mode="train")
        assert "race_id" in result.columns
        assert "umaban" in result.columns


class TestBuildFrameInferMode:
    """Tests for build_frame(df, mode='infer')."""

    def test_produces_identical_schema_as_train(self) -> None:
        """Test 2: infer mode returns identical column names and order as train."""
        train_df = _make_train_source_df()
        infer_df = _make_infer_source_df()
        builder = InvestmentFeatureFrameBuilder()

        train_result = builder.build_frame(train_df, mode="train")
        infer_result = builder.build_frame(infer_df, mode="infer")

        assert list(train_result.columns) == list(infer_result.columns), (
            f"Column mismatch:\n"
            f"train-only: {set(train_result.columns) - set(infer_result.columns)}\n"
            f"infer-only: {set(infer_result.columns) - set(train_result.columns)}"
        )

    def test_identical_dtypes_as_train(self) -> None:
        """Test 2b: infer mode returns identical dtypes as train."""
        train_df = _make_train_source_df()
        infer_df = _make_infer_source_df()
        builder = InvestmentFeatureFrameBuilder()

        train_result = builder.build_frame(train_df, mode="train")
        infer_result = builder.build_frame(infer_df, mode="infer")

        for col in train_result.columns:
            assert train_result[col].dtype == infer_result[col].dtype, (
                f"dtype mismatch for '{col}': "
                f"train={train_result[col].dtype}, infer={infer_result[col].dtype}"
            )


class TestBuildFrameValidation:
    """Tests for build_frame validation and error handling."""

    def test_invalid_mode_raises_value_error(self) -> None:
        """Test 3: build_frame raises ValueError for invalid mode."""
        df = _make_train_source_df()
        builder = InvestmentFeatureFrameBuilder()
        with pytest.raises(ValueError, match="mode must be"):
            builder.build_frame(df, mode="invalid")

    def test_train_mode_missing_required_raises_value_error(self) -> None:
        """Test 4: train mode raises ValueError when required source missing."""
        df = pd.DataFrame({"race_id": ["R001"], "umaban": [1]})
        builder = InvestmentFeatureFrameBuilder()
        with pytest.raises(ValueError, match="Required feature.*no source"):
            builder.build_frame(df, mode="train")

    def test_infer_mode_missing_required_raises_value_error(self) -> None:
        """Test 5: infer mode raises ValueError when required source missing."""
        df = pd.DataFrame({"race_id": ["R001"], "umaban": [1]})
        builder = InvestmentFeatureFrameBuilder()
        with pytest.raises(ValueError, match="Required feature.*no source"):
            builder.build_frame(df, mode="infer")


class TestOptionalFeatureMissing:
    """Tests for optional feature graceful degradation."""

    def test_optional_missing_produces_nan_and_indicator(self) -> None:
        """Test 6: Optional feature missing produces NaN + missing_indicator=1."""
        # Build df with only required train sources
        required_sources: set[str] = set()
        for spec in FEATURE_SPECS.values():
            if spec.required:
                required_sources.update(spec.train_sources)

        np.random.seed(42)
        data: dict[str, list] = {
            "race_id": ["R001", "R002"],
            "umaban": [1, 2],
        }
        for col in sorted(required_sources):
            data[col] = list(np.random.randn(2))
        df = pd.DataFrame(data)

        builder = InvestmentFeatureFrameBuilder()
        result = builder.build_frame(df, mode="train")

        # Find an optional spec with missing_indicator
        optional_spec = None
        for spec in FEATURE_SPECS.values():
            if not spec.required and spec.missing_indicator is not None:
                # Check that its source is not in the required set
                sources = set(spec.train_sources)
                if not sources.issubset(required_sources) and len(sources) > 0:
                    optional_spec = spec
                    break

        if optional_spec is not None:
            # The feature should exist but be NaN
            assert optional_spec.name in result.columns
            assert result[optional_spec.name].isna().all()
            # Missing indicator should be 1
            assert optional_spec.missing_indicator in result.columns
            assert (result[optional_spec.missing_indicator] == 1).all()

    def test_optional_present_produces_values_and_zero_indicator(self) -> None:
        """Test 7: Optional feature present produces values + missing_indicator=0."""
        df = _make_train_source_df()
        builder = InvestmentFeatureFrameBuilder()
        result = builder.build_frame(df, mode="train")

        # Find optional specs that have sources present
        for spec in FEATURE_SPECS.values():
            if not spec.required and spec.missing_indicator is not None:
                if len(spec.train_sources) > 0:
                    source_present = all(
                        s in df.columns for s in spec.train_sources
                    )
                    if source_present:
                        assert spec.missing_indicator in result.columns
                        assert (result[spec.missing_indicator] == 0).all(), (
                            f"{spec.missing_indicator} should be 0 when source present"
                        )
                        break


class TestDerivedFeatures:
    """Tests for derived feature computation."""

    def test_ev_raw_computed_correctly(self) -> None:
        """Test 8a: if_ev_raw = if_p_win * if_e_return."""
        df = _make_train_source_df()
        builder = InvestmentFeatureFrameBuilder()
        result = builder.build_frame(df, mode="train")

        if "if_ev_raw" in result.columns and "if_p_win" in result.columns:
            expected = result["if_p_win"] * result["if_e_return"]
            np.testing.assert_allclose(
                result["if_ev_raw"].values,
                expected.values,
                rtol=1e-10,
            )

    def test_logit_gap_computed(self) -> None:
        """Test 8b: if_logit_gap = logit(if_p_win) - logit(if_implied_prob)."""
        df = _make_train_source_df()
        builder = InvestmentFeatureFrameBuilder()
        result = builder.build_frame(df, mode="train")

        if "if_logit_gap" in result.columns:
            assert not result["if_logit_gap"].isna().all()


class TestColumnOrder:
    """Tests for output column ordering."""

    def test_output_column_order_matches_category_order(self) -> None:
        """Test 9: Output column order matches CATEGORY_ORDER then spec order."""
        df = _make_train_source_df()
        builder = InvestmentFeatureFrameBuilder()
        result = builder.build_frame(df, mode="train")

        # Identity cols come first
        identity_cols = ["race_id", "umaban"]
        for i, col in enumerate(identity_cols):
            assert result.columns[i] == col, f"Expected {col} at position {i}"

        # Then if_* columns follow CATEGORY_ORDER
        if_cols = [c for c in result.columns if c.startswith("if_") and not c.endswith("_missing")]
        prev_cat_idx = -1
        for col in if_cols:
            if col in FEATURE_SPECS:
                cat = FEATURE_SPECS[col].category
                cat_idx = CATEGORY_ORDER.index(cat)
                assert cat_idx >= prev_cat_idx, (
                    f"Column {col} (cat={cat}) out of order after "
                    f"category index {prev_cat_idx}"
                )
                prev_cat_idx = cat_idx

        # Missing indicators come last
        missing_cols = [c for c in result.columns if c.endswith("_missing")]
        non_missing = [c for c in result.columns if not c.endswith("_missing")]
        missing_start = len(result.columns) - len(missing_cols)
        for col in missing_cols:
            assert list(result.columns).index(col) >= missing_start


class TestConvenienceWrappers:
    """Tests for build_train_frame and build_inference_frame."""

    def test_build_train_frame_calls_with_train_mode(self) -> None:
        """Test 10: build_train_frame() calls build_frame with mode='train'."""
        df = _make_train_source_df()
        builder = InvestmentFeatureFrameBuilder()
        result = builder.build_train_frame(df)
        # Should not raise -- implies mode="train" was used

        if_cols = [c for c in result.columns if c.startswith("if_")]
        assert len(if_cols) > 0

    def test_build_inference_frame_calls_with_infer_mode(self) -> None:
        """Test 11: build_inference_frame() calls build_frame with mode='infer'."""
        df = _make_infer_source_df()
        builder = InvestmentFeatureFrameBuilder()
        result = builder.build_inference_frame(df)

        if_cols = [c for c in result.columns if c.startswith("if_")]
        assert len(if_cols) > 0


class TestLeakageValidation:
    """Tests for leakage detection integration."""

    def test_validate_no_post_race_leakage_called(self) -> None:
        """Test 12: validate_no_post_race_leakage is called during build_frame."""
        # If build_frame works normally, leakage validation passed
        df = _make_train_source_df()
        builder = InvestmentFeatureFrameBuilder()
        result = builder.build_frame(df, mode="train")

        # Output should not contain any POST_RACE columns
        from domain.types import POST_RACE_COLS

        overlap = set(result.columns) & set(POST_RACE_COLS)
        assert len(overlap) == 0, f"POST_RACE columns found in output: {overlap}"


class TestDeterminism:
    """Tests for deterministic output."""

    def test_same_input_produces_identical_output(self) -> None:
        """Test 13: Same input produces byte-identical output."""
        df = _make_train_source_df()
        builder = InvestmentFeatureFrameBuilder()

        result1 = builder.build_frame(df, mode="train")
        result2 = builder.build_frame(df, mode="train")

        pd.testing.assert_frame_equal(result1, result2)


class TestModuleLevelBuildFrame:
    """Tests for module-level build_frame function."""

    def test_module_level_build_frame_works(self) -> None:
        """Module-level build_frame function should work."""
        df = _make_train_source_df()
        result = build_frame(df, mode="train")
        if_cols = [c for c in result.columns if c.startswith("if_")]
        assert len(if_cols) > 0
