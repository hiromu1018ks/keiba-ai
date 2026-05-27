"""End-to-end integration tests for InvestmentFeatureFrame pipeline.

Validates the complete pipeline: build_frame(train) -> build_frame(infer) ->
schema identity -> manifest generation -> leakage audit.
Covers IFF-01~07 + VAL-01 integration requirements.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from domain.types import POST_RACE_COLS
from investment import (
    ALL_IF_COLUMNS,
    CATEGORY_ORDER,
    FEATURE_SPECS,
    IN_SAMPLE_ONLY_COLS,
    InvestmentFeatureFrameBuilder,
    compute_investment_schema_hash,
    generate_investment_manifest,
    validate_oof_safe_sources,
    validate_schema_identity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_train_source_df(n: int = 20, n_races: int = 4) -> pd.DataFrame:
    """Build a DataFrame with all train-mode source columns populated.

    Distributes n entries across n_races so groupby operations work.
    """
    np.random.seed(42)
    horses_per_race = n // n_races
    race_ids: list[str] = []
    umabans: list[int] = []
    for r in range(n_races):
        for h in range(horses_per_race):
            race_ids.append(f"R{r:04d}")
            umabans.append(h + 1)

    data: dict[str, list[float] | list[str] | list[int]] = {
        "race_id": race_ids,
        "umaban": umabans,
    }
    # Collect all unique train source columns
    train_sources: set[str] = set()
    for spec in FEATURE_SPECS.values():
        train_sources.update(spec.train_sources)
    for col in sorted(train_sources):
        data[col] = list(np.random.randn(len(race_ids)))
    return pd.DataFrame(data)


def _make_infer_source_df(n: int = 20, n_races: int = 4) -> pd.DataFrame:
    """Build a DataFrame with all infer-mode source columns populated."""
    np.random.seed(43)
    horses_per_race = n // n_races
    race_ids: list[str] = []
    umabans: list[int] = []
    for r in range(n_races):
        for h in range(horses_per_race):
            race_ids.append(f"R{r:04d}")
            umabans.append(h + 1)

    data: dict[str, list[float] | list[str] | list[int]] = {
        "race_id": race_ids,
        "umaban": umabans,
    }
    # Collect all unique infer source columns
    infer_sources: set[str] = set()
    for spec in FEATURE_SPECS.values():
        infer_sources.update(spec.infer_sources)
    for col in sorted(infer_sources):
        data[col] = list(np.random.randn(len(race_ids)))
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# 1. End-to-end pipeline test
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Full pipeline: build_frame -> schema identity -> manifest."""

    def test_full_pipeline_train_to_infer_to_manifest(self) -> None:
        """Build train frame, infer frame, verify identity, generate manifest."""
        train_df = _make_train_source_df()
        infer_df = _make_infer_source_df()
        builder = InvestmentFeatureFrameBuilder()

        # Step 1: Build train frame
        train_result = builder.build_frame(train_df, mode="train")
        assert "race_id" in train_result.columns
        assert "if_p_win" in train_result.columns

        # Step 2: Build infer frame
        infer_result = builder.build_frame(infer_df, mode="infer")

        # Step 3: Validate schema identity (IFF-03)
        validate_schema_identity(train_result, infer_result)

        # Step 4: Generate manifest (IFF-07, D-30)
        manifest = generate_investment_manifest(
            train_result,
            feature_version="v2.0",
            builder_version="1.0.0",
            mode="train",
            source_artifact_hash="abc123",
        )
        assert manifest["artifact_name"] == "investment_feature_frame"
        assert manifest["row_count"] == len(train_result)
        assert manifest["column_count"] == len(train_result.columns)

    def test_column_count_in_range(self) -> None:
        """Feature columns (excluding _missing indicators) in 90-130 range."""
        df = _make_train_source_df()
        builder = InvestmentFeatureFrameBuilder()
        result = builder.build_frame(df, mode="train")

        if_cols = [
            c for c in result.columns
            if c.startswith("if_") and not c.endswith("_missing")
        ]
        assert 90 <= len(if_cols) <= 130, (
            f"Expected 90-130 if_* feature columns, got {len(if_cols)}"
        )

    def test_no_post_race_columns_in_output(self) -> None:
        """Output contains zero POST_RACE columns (IFF-05, VAL-01)."""
        df = _make_train_source_df()
        builder = InvestmentFeatureFrameBuilder()
        result = builder.build_frame(df, mode="train")

        overlap = set(result.columns) & set(POST_RACE_COLS)
        assert len(overlap) == 0, (
            f"POST_RACE columns found in output: {overlap}"
        )

    def test_train_mode_zero_in_sample_only_columns(self) -> None:
        """Train mode output contains zero in-sample-only source columns (IFF-02)."""
        df = _make_train_source_df()
        builder = InvestmentFeatureFrameBuilder()
        result = builder.build_frame(df, mode="train")

        # Output columns should not contain any IN_SAMPLE_ONLY_COLS names
        in_sample_in_output = set(result.columns) & IN_SAMPLE_ONLY_COLS
        assert len(in_sample_in_output) == 0, (
            f"In-sample-only columns in train output: {in_sample_in_output}"
        )


# ---------------------------------------------------------------------------
# 2. Leakage audit across all specs
# ---------------------------------------------------------------------------


class TestLeakageAudit:
    """Comprehensive leakage audit across all FEATURE_SPECS (D-20, VAL-01)."""

    def test_all_specs_oof_safe(self) -> None:
        """All FEATURE_SPECS train_sources have zero IN_SAMPLE_ONLY overlap (D-20)."""
        violations = validate_oof_safe_sources(FEATURE_SPECS)
        assert violations == [], (
            f"OOF-unsafe train_sources found: {violations}"
        )

    def test_all_spec_names_no_post_race_overlap(self) -> None:
        """No spec name overlaps with POST_RACE_COLS (VAL-01)."""
        post_race_set = set(POST_RACE_COLS)
        for spec in FEATURE_SPECS.values():
            assert spec.name not in post_race_set, (
                f"Spec '{spec.name}' overlaps with POST_RACE_COLS"
            )

    def test_all_spec_train_sources_no_post_race(self) -> None:
        """No spec's train_sources contain any POST_RACE column."""
        post_race_set = set(POST_RACE_COLS)
        for spec in FEATURE_SPECS.values():
            overlap = set(spec.train_sources) & post_race_set
            assert len(overlap) == 0, (
                f"Spec '{spec.name}' train_sources contain POST_RACE: {overlap}"
            )

    def test_all_spec_infer_sources_no_post_race(self) -> None:
        """No spec's infer_sources contain any POST_RACE column."""
        post_race_set = set(POST_RACE_COLS)
        for spec in FEATURE_SPECS.values():
            overlap = set(spec.infer_sources) & post_race_set
            assert len(overlap) == 0, (
                f"Spec '{spec.name}' infer_sources contain POST_RACE: {overlap}"
            )

    def test_all_spec_leakage_class_is_safe_or_oof_only(self) -> None:
        """All specs have leakage_class 'safe' or 'oof_only' (never 'post_race')."""
        for spec in FEATURE_SPECS.values():
            assert spec.leakage_class in ("safe", "oof_only"), (
                f"Spec '{spec.name}' has leakage_class='{spec.leakage_class}'"
            )


# ---------------------------------------------------------------------------
# 3. Required/Optional behavior verification
# ---------------------------------------------------------------------------


class TestRequiredOptionalBehavior:
    """Verify required/optional feature handling per D-19."""

    def test_required_missing_raises_value_error(self) -> None:
        """Required feature source missing causes ValueError."""
        df = pd.DataFrame({"race_id": ["R001"], "umaban": [1]})
        builder = InvestmentFeatureFrameBuilder()
        with pytest.raises(ValueError, match="Required feature.*no source"):
            builder.build_frame(df, mode="train")

    def test_optional_missing_produces_nan_with_indicator(self) -> None:
        """Optional feature source missing produces NaN + missing_indicator=1."""
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

        # Find an optional spec whose source is not in the required set
        for spec in FEATURE_SPECS.values():
            if (
                not spec.required
                and spec.missing_indicator is not None
                and len(spec.train_sources) > 0
                and not set(spec.train_sources).issubset(required_sources)
            ):
                assert spec.name in result.columns
                assert result[spec.name].isna().all(), (
                    f"Optional '{spec.name}' should be NaN when source missing"
                )
                assert spec.missing_indicator in result.columns
                assert (result[spec.missing_indicator] == 1).all(), (
                    f"'{spec.missing_indicator}' should be 1 when source missing"
                )
                return  # Found and verified one

    def test_optional_present_has_zero_indicator(self) -> None:
        """Optional feature with source present has missing_indicator=0."""
        df = _make_train_source_df()
        builder = InvestmentFeatureFrameBuilder()
        result = builder.build_frame(df, mode="train")

        for spec in FEATURE_SPECS.values():
            if (
                not spec.required
                and spec.missing_indicator is not None
                and len(spec.train_sources) > 0
                and all(s in df.columns for s in spec.train_sources)
            ):
                assert spec.missing_indicator in result.columns
                assert (result[spec.missing_indicator] == 0).all(), (
                    f"'{spec.missing_indicator}' should be 0 when source present"
                )


# ---------------------------------------------------------------------------
# 4. Column count per category
# ---------------------------------------------------------------------------


class TestColumnCountPerCategory:
    """Verify column counts per category match D-05 ranges."""

    EXPECTED_RANGES: dict[str, tuple[int, int]] = {
        "model_prob": (8, 12),
        "market_prob": (6, 10),
        "model_market_gap": (10, 16),
        "race_relative": (12, 18),
        "odds_band": (6, 10),
        "late_odds": (8, 12),
        "ability_form": (15, 25),
        "course_pace": (10, 18),
        "uncertainty": (10, 16),
    }

    def test_spec_count_per_category(self) -> None:
        """Each category has specs within D-05 range."""
        cat_counts: dict[str, int] = {}
        for spec in FEATURE_SPECS.values():
            cat_counts[spec.category] = cat_counts.get(spec.category, 0) + 1

        for cat, (lo, hi) in self.EXPECTED_RANGES.items():
            count = cat_counts.get(cat, 0)
            assert lo <= count <= hi, (
                f"Category '{cat}' has {count} specs, expected {lo}-{hi}"
            )

    def test_total_spec_count(self) -> None:
        """Total spec count is in 90-130 range."""
        total = len(FEATURE_SPECS)
        assert 90 <= total <= 130, f"Total specs: {total}, expected 90-130"


# ---------------------------------------------------------------------------
# 5. Determinism and schema consistency
# ---------------------------------------------------------------------------


class TestDeterminismAndSchema:
    """Verify deterministic behavior and schema consistency."""

    def test_schema_hash_deterministic(self) -> None:
        """Same output produces same schema hash."""
        df = _make_train_source_df()
        builder = InvestmentFeatureFrameBuilder()
        result1 = builder.build_frame(df, mode="train")
        result2 = builder.build_frame(df, mode="train")

        hash1 = compute_investment_schema_hash(result1)
        hash2 = compute_investment_schema_hash(result2)
        assert hash1 == hash2

    def test_train_infer_produce_same_schema_hash(self) -> None:
        """Train and infer mode produce identical schema hash."""
        train_df = _make_train_source_df()
        infer_df = _make_infer_source_df()
        builder = InvestmentFeatureFrameBuilder()

        train_result = builder.build_frame(train_df, mode="train")
        infer_result = builder.build_frame(infer_df, mode="infer")

        train_hash = compute_investment_schema_hash(train_result)
        infer_hash = compute_investment_schema_hash(infer_result)
        assert train_hash == infer_hash, (
            f"Train schema hash != Infer schema hash: {train_hash} != {infer_hash}"
        )

    def test_all_if_columns_matches_output(self) -> None:
        """ALL_IF_COLUMNS matches the feature columns in build_frame output."""
        df = _make_train_source_df()
        builder = InvestmentFeatureFrameBuilder()
        result = builder.build_frame(df, mode="train")

        output_if_cols = [
            c for c in result.columns
            if c.startswith("if_") and not c.endswith("_missing")
        ]
        assert set(output_if_cols) == set(ALL_IF_COLUMNS), (
            f"Output if_* columns don't match ALL_IF_COLUMNS.\n"
            f"Extra in output: {set(output_if_cols) - set(ALL_IF_COLUMNS)}\n"
            f"Missing from output: {set(ALL_IF_COLUMNS) - set(output_if_cols)}"
        )

    def test_all_9_categories_in_output(self) -> None:
        """Output contains features from all 9 categories."""
        df = _make_train_source_df()
        builder = InvestmentFeatureFrameBuilder()
        result = builder.build_frame(df, mode="train")

        output_cats = set()
        for col in result.columns:
            if col in FEATURE_SPECS:
                output_cats.add(FEATURE_SPECS[col].category)

        for cat in CATEGORY_ORDER:
            assert cat in output_cats, (
                f"No features from category '{cat}' in output"
            )
