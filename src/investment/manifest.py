"""Investment feature frame manifest generation.

Provides schema hash computation and manifest generation per D-30, IFF-07.
Follows OOFHealthValidator._compute_schema_hashes pattern for deterministic
hashing (hashlib.sha256 + json.dumps(sort_keys=True)).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd

__all__ = [
    "compute_investment_schema_hash",
    "generate_investment_manifest",
]


def compute_investment_schema_hash(df: pd.DataFrame) -> str:
    """Compute deterministic SHA256 hash from DataFrame column names.

    Follows OOFHealthValidator._compute_schema_hashes pattern:
    sorted columns -> JSON string -> SHA256 hexdigest.

    Args:
        df: DataFrame whose column names to hash.

    Returns:
        SHA256 hexdigest string (64 chars).
    """
    cols_sorted = sorted(df.columns.tolist())
    return hashlib.sha256(json.dumps(cols_sorted).encode()).hexdigest()


def _compute_schema_dtype_hash(df: pd.DataFrame) -> str:
    """Compute SHA256 hash from column-dtype pairs.

    Args:
        df: DataFrame whose column+dtype pairs to hash.

    Returns:
        SHA256 hexdigest string (64 chars).
    """
    dtype_pairs = sorted(f"{col}:{df[col].dtype}" for col in df.columns)
    return hashlib.sha256(json.dumps(dtype_pairs).encode()).hexdigest()


def generate_investment_manifest(
    df: pd.DataFrame,
    *,
    feature_version: str,
    builder_version: str,
    mode: str,
    source_artifact_hash: str,
    source_oof_manifest_path: str | None = None,
) -> dict[str, Any]:
    """Generate investment feature frame artifact manifest (D-30, IFF-07).

    Produces a dict with all D-30 required fields for traceability.

    Args:
        df: The output investment feature DataFrame.
        feature_version: Feature schema version string.
        builder_version: Builder code version string.
        mode: "train" or "infer".
        source_artifact_hash: Hash of the source artifact used.
        source_oof_manifest_path: Path to OOF health manifest (train mode).

    Returns:
        Dict with manifest fields per D-30 specification.
    """
    schema_hash = compute_investment_schema_hash(df)
    schema_dtype_hash = _compute_schema_dtype_hash(df)

    return {
        "artifact_name": "investment_feature_frame",
        "builder_version": builder_version,
        "feature_version": feature_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "row_count": len(df),
        "schema_hash": schema_hash,
        "schema_dtype_hash": schema_dtype_hash,
        "source_artifact_hash": source_artifact_hash,
        "source_oof_manifest_path": source_oof_manifest_path,
        "column_count": len(df.columns),
    }
