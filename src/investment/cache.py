"""Investment feature frame cache module.

Provides InvestmentFrameCache for Parquet + sidecar manifest JSON caching
per D-21~D-27, IFF-06. Follows ParquetStore I/O patterns.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

__all__ = ["InvestmentFrameCache"]


class InvestmentFrameCache:
    """Cache for investment feature frames using Parquet + sidecar manifest JSON.

    Per D-21: Parquet cache with sidecar manifest JSON for integrity.
    Per D-22: Cache path pattern: {mode}/{feature_version}_{hash_prefix}.parquet
    Per D-23: Cache key includes mode, feature_version, source_artifact_hash,
              schema_hash, builder_version.
    Per D-24: Cache load verifies sidecar manifest schema_hash.
    """

    def __init__(self, cache_dir: str = "data/features/investment_frame") -> None:
        """Initialize cache with directory path.

        Args:
            cache_dir: Root directory for cache storage.
        """
        self._cache_dir = Path(cache_dir)

    def _compute_cache_key(
        self,
        mode: str,
        feature_version: str,
        source_artifact_hash: str,
        schema_hash: str,
        builder_version: str,
    ) -> str:
        """Compute deterministic cache key path (D-22, D-23).

        Combines all key components into a SHA256 hash and formats
        as {mode}/{feature_version}_{hash_prefix}.parquet.

        Args:
            mode: "train" or "infer".
            feature_version: Feature schema version.
            source_artifact_hash: Hash of source artifact.
            schema_hash: Output schema hash.
            builder_version: Builder code version.

        Returns:
            Relative path string for cache file.
        """
        key_input = (
            f"{mode}|{feature_version}|"
            f"{source_artifact_hash}|{schema_hash}|{builder_version}"
        )
        hash_val = hashlib.sha256(key_input.encode()).hexdigest()[:16]
        return f"{mode}/{feature_version}_{hash_val}.parquet"

    def load_cached(
        self, cache_key: str, expected_schema_hash: str
    ) -> tuple[pd.DataFrame, dict[str, Any]] | None:
        """Load cached DataFrame and manifest if schema_hash matches (D-24).

        Args:
            cache_key: Relative path for cache file.
            expected_schema_hash: Schema hash to verify against.

        Returns:
            Tuple of (DataFrame, manifest dict) on hit, None on miss.
        """
        parquet_path = self._cache_dir / cache_key
        manifest_path = parquet_path.with_suffix(".manifest.json")

        if not parquet_path.exists() or not manifest_path.exists():
            return None

        # Load and verify sidecar manifest
        with open(manifest_path, encoding="utf-8") as f:
            manifest: dict[str, Any] = json.load(f)

        stored_hash = manifest.get("schema_hash", "")
        if stored_hash != expected_schema_hash:
            return None

        # Load parquet
        df = pd.read_parquet(parquet_path)
        return df, manifest

    def save(
        self,
        cache_key: str,
        df: pd.DataFrame,
        manifest: dict[str, Any],
    ) -> None:
        """Save DataFrame and manifest to cache (D-21).

        Writes parquet file and sidecar manifest JSON with deterministic
        formatting (json.dumps sort_keys=True, indent=2).

        Args:
            cache_key: Relative path for cache file.
            df: DataFrame to cache.
            manifest: Manifest dict to save as sidecar.
        """
        parquet_path = self._cache_dir / cache_key
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        # Write parquet
        df.to_parquet(parquet_path, engine="pyarrow", index=False)

        # Write sidecar manifest JSON
        manifest_path = parquet_path.with_suffix(".manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, sort_keys=True, indent=2)

    def load_or_compute(
        self,
        *,
        df: pd.DataFrame,
        mode: str,
        feature_version: str,
        source_artifact_hash: str,
        builder_version: str,
        compute_fn: Callable[..., tuple[pd.DataFrame, dict[str, Any]]],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Load from cache or compute and cache the result.

        Computes the schema_hash from the input df to build a cache key.
        On cache hit, returns the cached result. On miss, calls compute_fn,
        saves the result, and returns it.

        **Important:** ``source_artifact_hash`` is the only guard against stale
        cache hits when source *data* changes but the column schema stays the
        same.  Callers MUST ensure ``source_artifact_hash`` changes whenever
        the content of the source artifact changes; otherwise cached results
        from a previous run may be returned for new data.

        Args:
            df: Input DataFrame.
            mode: "train" or "infer".
            feature_version: Feature schema version.
            source_artifact_hash: Hash of source artifact.
            builder_version: Builder code version.
            compute_fn: Callable(df, mode) -> (DataFrame, manifest_dict).

        Returns:
            Tuple of (DataFrame, manifest dict).
        """
        # First compute a preliminary result to get schema_hash for cache key
        # Or use a pre-computation approach: compute first, then check cache
        # The simpler approach: compute schema hash from df columns
        from investment.manifest import compute_investment_schema_hash

        schema_hash = compute_investment_schema_hash(df)

        cache_key = self._compute_cache_key(
            mode=mode,
            feature_version=feature_version,
            source_artifact_hash=source_artifact_hash,
            schema_hash=schema_hash,
            builder_version=builder_version,
        )

        # Try cache load
        cached = self.load_cached(cache_key, schema_hash)
        if cached is not None:
            return cached

        # Compute
        result_df, result_manifest = compute_fn(df, mode)

        # Save to cache
        self.save(cache_key, result_df, result_manifest)

        return result_df, result_manifest
