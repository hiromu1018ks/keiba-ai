"""Tests for investment cache module (cache.py).

TDD RED phase: tests for InvestmentFrameCache per D-21~D-27, IFF-06.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from investment.cache import InvestmentFrameCache


# ---------------------------------------------------------------------------
# _compute_cache_key
# ---------------------------------------------------------------------------


class TestComputeCacheKey:
    """Tests for InvestmentFrameCache._compute_cache_key."""

    def test_deterministic_for_same_inputs(self) -> None:
        """Test 1: Same inputs produce identical cache key."""
        cache = InvestmentFrameCache()
        key1 = cache._compute_cache_key(
            mode="train",
            feature_version="v2.0",
            source_artifact_hash="abc123",
            schema_hash="def456",
            builder_version="1.0.0",
        )
        key2 = cache._compute_cache_key(
            mode="train",
            feature_version="v2.0",
            source_artifact_hash="abc123",
            schema_hash="def456",
            builder_version="1.0.0",
        )
        assert key1 == key2

    def test_different_mode_produces_different_key(self) -> None:
        """Test 2: Different mode produces different key."""
        cache = InvestmentFrameCache()
        key_train = cache._compute_cache_key(
            mode="train",
            feature_version="v2.0",
            source_artifact_hash="abc",
            schema_hash="def",
            builder_version="1.0.0",
        )
        key_infer = cache._compute_cache_key(
            mode="infer",
            feature_version="v2.0",
            source_artifact_hash="abc",
            schema_hash="def",
            builder_version="1.0.0",
        )
        assert key_train != key_infer

    def test_different_source_artifact_hash_produces_different_key(self) -> None:
        """Test 3: Different source_artifact_hash produces different key."""
        cache = InvestmentFrameCache()
        key1 = cache._compute_cache_key(
            mode="train",
            feature_version="v2.0",
            source_artifact_hash="hash1",
            schema_hash="def",
            builder_version="1.0.0",
        )
        key2 = cache._compute_cache_key(
            mode="train",
            feature_version="v2.0",
            source_artifact_hash="hash2",
            schema_hash="def",
            builder_version="1.0.0",
        )
        assert key1 != key2

    def test_key_includes_mode_prefix(self) -> None:
        """Cache key path should start with mode directory."""
        cache = InvestmentFrameCache()
        key = cache._compute_cache_key(
            mode="train",
            feature_version="v2.0",
            source_artifact_hash="abc",
            schema_hash="def",
            builder_version="1.0.0",
        )
        assert key.startswith("train/")

    def test_key_ends_with_parquet(self) -> None:
        """Cache key path should end with .parquet."""
        cache = InvestmentFrameCache()
        key = cache._compute_cache_key(
            mode="infer",
            feature_version="v2.0",
            source_artifact_hash="abc",
            schema_hash="def",
            builder_version="1.0.0",
        )
        assert key.endswith(".parquet")


# ---------------------------------------------------------------------------
# load_cached / save
# ---------------------------------------------------------------------------


class TestCacheHitMiss:
    """Tests for InvestmentFrameCache load/save behavior."""

    @pytest.fixture()
    def cache_dir(self, tmp_path: Path) -> str:
        """Create temporary cache directory."""
        cache_path = tmp_path / "investment_cache"
        cache_path.mkdir()
        return str(cache_path)

    @pytest.fixture()
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame for cache tests."""
        return pd.DataFrame(
            {
                "race_id": ["R001", "R002"],
                "umaban": [1, 2],
                "if_p_win": [0.3, 0.5],
            }
        )

    @pytest.fixture()
    def sample_manifest(self) -> dict[str, Any]:
        """Create sample manifest for cache tests."""
        return {
            "artifact_name": "investment_feature_frame",
            "schema_hash": "abc123",
            "mode": "train",
            "feature_version": "v2.0",
        }

    def test_cache_miss_returns_none(
        self,
        cache_dir: str,
    ) -> None:
        """Test 4: Cache miss (no parquet file) returns None."""
        cache = InvestmentFrameCache(cache_dir=cache_dir)
        result = cache.load_cached("train/nonexistent.parquet", "abc123")
        assert result is None

    def test_cache_hit_returns_df_and_manifest(
        self,
        cache_dir: str,
        sample_df: pd.DataFrame,
        sample_manifest: dict[str, Any],
    ) -> None:
        """Test 5: Cache hit returns DataFrame and manifest dict."""
        cache = InvestmentFrameCache(cache_dir=cache_dir)
        cache_key = "train/test_cache.parquet"
        cache.save(cache_key, sample_df, sample_manifest)

        result = cache.load_cached(cache_key, sample_manifest["schema_hash"])
        assert result is not None
        loaded_df, loaded_manifest = result

        # DataFrame should match original data
        pd.testing.assert_frame_equal(
            loaded_df.reset_index(drop=True),
            sample_df.reset_index(drop=True),
        )
        # Manifest should match original
        assert loaded_manifest["schema_hash"] == sample_manifest["schema_hash"]

    def test_save_creates_parquet_and_sidecar(
        self,
        cache_dir: str,
        sample_df: pd.DataFrame,
        sample_manifest: dict[str, Any],
    ) -> None:
        """Test 6: Save creates .parquet + .manifest.json sidecar."""
        cache = InvestmentFrameCache(cache_dir=cache_dir)
        cache_key = "train/test_save.parquet"
        cache.save(cache_key, sample_df, sample_manifest)

        # Check parquet file exists
        parquet_path = Path(cache_dir) / cache_key
        assert parquet_path.exists()

        # Check sidecar manifest JSON exists
        manifest_path = parquet_path.with_suffix(".manifest.json")
        assert manifest_path.exists()

        # Verify manifest JSON is valid
        with open(manifest_path, encoding="utf-8") as f:
            loaded_manifest = json.load(f)
        assert loaded_manifest["schema_hash"] == sample_manifest["schema_hash"]

    def test_corrupted_schema_hash_triggers_cache_miss(
        self,
        cache_dir: str,
        sample_df: pd.DataFrame,
        sample_manifest: dict[str, Any],
    ) -> None:
        """Test 7: Wrong schema_hash in sidecar triggers cache miss."""
        cache = InvestmentFrameCache(cache_dir=cache_dir)
        cache_key = "train/corrupted.parquet"
        cache.save(cache_key, sample_df, sample_manifest)

        # Load with a different expected schema hash
        result = cache.load_cached(cache_key, "wrong_hash_value")
        assert result is None

    def test_load_or_compute_with_cache_miss(
        self,
        cache_dir: str,
        sample_df: pd.DataFrame,
    ) -> None:
        """load_or_compute calls compute_fn on cache miss."""
        cache = InvestmentFrameCache(cache_dir=cache_dir)

        compute_called = False

        def compute_fn(df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, dict]:
            nonlocal compute_called
            compute_called = True
            result_df = df.copy()
            manifest = {"schema_hash": "computed_hash", "mode": mode}
            return result_df, manifest

        result_df, result_manifest = cache.load_or_compute(
            df=sample_df,
            mode="train",
            feature_version="v2.0",
            source_artifact_hash="abc",
            builder_version="1.0.0",
            compute_fn=compute_fn,
        )
        assert compute_called
        assert len(result_df) == 2

    def test_load_or_compute_with_cache_hit(
        self,
        cache_dir: str,
        sample_df: pd.DataFrame,
        sample_manifest: dict[str, Any],
    ) -> None:
        """load_or_compute returns cached result on hit."""
        cache = InvestmentFrameCache(cache_dir=cache_dir)

        # Pre-populate cache
        cache_key = cache._compute_cache_key(
            mode="train",
            feature_version="v2.0",
            source_artifact_hash="abc",
            schema_hash=sample_manifest["schema_hash"],
            builder_version="1.0.0",
        )
        cache.save(cache_key, sample_df, sample_manifest)

        compute_called = False

        def compute_fn(df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, dict]:
            nonlocal compute_called
            compute_called = True
            return df, {}

        result_df, _ = cache.load_or_compute(
            df=sample_df,
            mode="train",
            feature_version="v2.0",
            source_artifact_hash="abc",
            builder_version="1.0.0",
            compute_fn=compute_fn,
        )
        # compute_fn should NOT be called on cache hit
        assert not compute_called
        pd.testing.assert_frame_equal(
            result_df.reset_index(drop=True),
            sample_df.reset_index(drop=True),
        )
