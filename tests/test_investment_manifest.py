"""Tests for investment manifest module (manifest.py).

TDD RED phase: tests for compute_investment_schema_hash and
generate_investment_manifest per D-30, IFF-07.
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd
import pytest

from investment.manifest import (
    compute_investment_schema_hash,
    generate_investment_manifest,
)


# ---------------------------------------------------------------------------
# compute_investment_schema_hash
# ---------------------------------------------------------------------------


class TestComputeInvestmentSchemaHash:
    """Tests for compute_investment_schema_hash."""

    def test_deterministic_hash_for_same_columns(self) -> None:
        """Test 1: Same columns+dtypes produce identical hash."""
        df = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
        hash1 = compute_investment_schema_hash(df)
        hash2 = compute_investment_schema_hash(df)
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hexdigest

    def test_different_columns_produce_different_hash(self) -> None:
        """Test 2: Different columns produce different hash."""
        df1 = pd.DataFrame({"a": [1.0], "b": [2.0]})
        df2 = pd.DataFrame({"x": [1.0], "y": [2.0]})
        hash1 = compute_investment_schema_hash(df1)
        hash2 = compute_investment_schema_hash(df2)
        assert hash1 != hash2

    def test_hash_independent_of_column_order(self) -> None:
        """Hash should be same regardless of column order (sorted)."""
        df1 = pd.DataFrame({"z": [1.0], "a": [2.0], "m": [3.0]})
        df2 = pd.DataFrame({"a": [2.0], "m": [3.0], "z": [1.0]})
        hash1 = compute_investment_schema_hash(df1)
        hash2 = compute_investment_schema_hash(df2)
        assert hash1 == hash2

    def test_hash_matches_manual_sha256(self) -> None:
        """Hash should match manual SHA256 of sorted columns."""
        df = pd.DataFrame({"col_b": [1.0], "col_a": [2.0]})
        expected_cols = sorted(df.columns.tolist())
        expected_hash = hashlib.sha256(
            json.dumps(expected_cols).encode()
        ).hexdigest()
        actual_hash = compute_investment_schema_hash(df)
        assert actual_hash == expected_hash


# ---------------------------------------------------------------------------
# generate_investment_manifest
# ---------------------------------------------------------------------------


class TestGenerateInvestmentManifest:
    """Tests for generate_investment_manifest."""

    @pytest.fixture()
    def sample_df(self) -> pd.DataFrame:
        """Create a sample DataFrame for manifest tests."""
        return pd.DataFrame(
            {
                "race_id": ["R001", "R002"],
                "umaban": [1, 2],
                "if_p_win": [0.3, 0.5],
                "if_ev_raw": [1.2, 2.0],
            }
        )

    def test_manifest_contains_all_required_keys(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Test 3: Manifest has all D-30 required keys."""
        manifest = generate_investment_manifest(
            sample_df,
            feature_version="v2.0",
            builder_version="1.0.0",
            mode="train",
            source_artifact_hash="abc123",
        )
        required_keys = {
            "artifact_name",
            "builder_version",
            "feature_version",
            "generated_at",
            "mode",
            "row_count",
            "schema_hash",
            "schema_dtype_hash",
            "source_artifact_hash",
            "source_oof_manifest_path",
            "column_count",
        }
        assert required_keys.issubset(set(manifest.keys())), (
            f"Missing keys: {required_keys - set(manifest.keys())}"
        )

    def test_generated_at_is_iso8601_utc(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Test 4: generated_at is ISO 8601 UTC format."""
        manifest = generate_investment_manifest(
            sample_df,
            feature_version="v2.0",
            builder_version="1.0.0",
            mode="train",
            source_artifact_hash="abc123",
        )
        generated_at = manifest["generated_at"]
        # ISO 8601 with timezone: ends with +00:00 or Z
        assert re.match(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[\.\d]*[+-]\d{2}:\d{2}|Z",
            generated_at,
        ), f"generated_at '{generated_at}' is not ISO 8601 UTC"

    def test_schema_hash_matches_compute_hash(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Test 5: schema_hash matches compute_investment_schema_hash output."""
        manifest = generate_investment_manifest(
            sample_df,
            feature_version="v2.0",
            builder_version="1.0.0",
            mode="train",
            source_artifact_hash="abc123",
        )
        expected_hash = compute_investment_schema_hash(sample_df)
        assert manifest["schema_hash"] == expected_hash

    def test_row_count_matches(self, sample_df: pd.DataFrame) -> None:
        """row_count should match DataFrame length."""
        manifest = generate_investment_manifest(
            sample_df,
            feature_version="v2.0",
            builder_version="1.0.0",
            mode="train",
            source_artifact_hash="abc123",
        )
        assert manifest["row_count"] == len(sample_df)

    def test_column_count_matches(self, sample_df: pd.DataFrame) -> None:
        """column_count should match DataFrame column count."""
        manifest = generate_investment_manifest(
            sample_df,
            feature_version="v2.0",
            builder_version="1.0.0",
            mode="train",
            source_artifact_hash="abc123",
        )
        assert manifest["column_count"] == len(sample_df.columns)

    def test_artifact_name_is_investment_feature_frame(
        self, sample_df: pd.DataFrame
    ) -> None:
        """artifact_name should be 'investment_feature_frame'."""
        manifest = generate_investment_manifest(
            sample_df,
            feature_version="v2.0",
            builder_version="1.0.0",
            mode="train",
            source_artifact_hash="abc123",
        )
        assert manifest["artifact_name"] == "investment_feature_frame"

    def test_mode_passed_through(self, sample_df: pd.DataFrame) -> None:
        """mode should be passed through to manifest."""
        manifest = generate_investment_manifest(
            sample_df,
            feature_version="v2.0",
            builder_version="1.0.0",
            mode="infer",
            source_artifact_hash="abc123",
        )
        assert manifest["mode"] == "infer"

    def test_source_oof_manifest_path_defaults_none(
        self, sample_df: pd.DataFrame
    ) -> None:
        """source_oof_manifest_path should default to None."""
        manifest = generate_investment_manifest(
            sample_df,
            feature_version="v2.0",
            builder_version="1.0.0",
            mode="train",
            source_artifact_hash="abc123",
        )
        assert manifest["source_oof_manifest_path"] is None

    def test_source_oof_manifest_path_set(
        self, sample_df: pd.DataFrame
    ) -> None:
        """source_oof_manifest_path should be set when provided."""
        manifest = generate_investment_manifest(
            sample_df,
            feature_version="v2.0",
            builder_version="1.0.0",
            mode="train",
            source_artifact_hash="abc123",
            source_oof_manifest_path="data/oof/manifests/win_predictions.json",
        )
        assert manifest["source_oof_manifest_path"] == (
            "data/oof/manifests/win_predictions.json"
        )

    def test_schema_dtype_hash_computed(
        self, sample_df: pd.DataFrame
    ) -> None:
        """schema_dtype_hash should be computed and non-empty."""
        manifest = generate_investment_manifest(
            sample_df,
            feature_version="v2.0",
            builder_version="1.0.0",
            mode="train",
            source_artifact_hash="abc123",
        )
        assert isinstance(manifest["schema_dtype_hash"], str)
        assert len(manifest["schema_dtype_hash"]) == 64

    def test_versions_passed_through(
        self, sample_df: pd.DataFrame
    ) -> None:
        """builder_version and feature_version should be passed through."""
        manifest = generate_investment_manifest(
            sample_df,
            feature_version="v3.1",
            builder_version="2.0.0",
            mode="train",
            source_artifact_hash="def456",
        )
        assert manifest["feature_version"] == "v3.1"
        assert manifest["builder_version"] == "2.0.0"
        assert manifest["source_artifact_hash"] == "def456"
