"""Leakage detection functions for InvestmentFeatureFrame.

Provides three validation functions and one constant for ensuring:
1. No POST_RACE columns leak into output (IFF-05, VAL-01)
2. Train mode sources are OOF-safe, not in-sample-only (D-13, D-29)
3. Train/infer output schema identity (IFF-03, D-15)
"""

from __future__ import annotations

import pandas as pd

from domain.types import POST_RACE_COLS
from investment.schema_registry import InvestmentFeatureSpec

# D-13: columns that must NOT appear in train_sources (in-sample-only)
IN_SAMPLE_ONLY_COLS: frozenset[str] = frozenset({
    "p_win_pred",
    "ev_win",
    "ev_win_calibrated",
    "p_win_final",
    "edge_win",
})


def validate_no_post_race_leakage(output_columns: list[str]) -> None:
    """Validate that output columns contain no POST_RACE columns.

    Raises ValueError if any POST_RACE column is found in output_columns.
    Follows test_post_race_leakage.py Layer 2 pattern (IFF-05, VAL-01).

    Args:
        output_columns: list of column names in the output DataFrame.

    Raises:
        ValueError: with details of overlapping POST_RACE columns.
    """
    overlap = set(output_columns) & set(POST_RACE_COLS)
    if overlap:
        raise ValueError(
            f"InvestmentFeatureFrame output contains POST_RACE columns: {overlap}"
        )


def validate_oof_safe_sources(
    specs: dict[str, InvestmentFeatureSpec],
) -> list[str]:
    """Validate that all specs have OOF-safe train_sources.

    Checks each spec's train_sources against IN_SAMPLE_ONLY_COLS.
    Returns a list of violation strings (empty if no violations).
    Does NOT raise -- caller decides how to handle violations (D-29).

    Args:
        specs: dict of spec_name -> InvestmentFeatureSpec.

    Returns:
        List of violation strings in "spec_name: {overlap}" format.
    """
    violations: list[str] = []
    for spec in specs.values():
        overlap = set(spec.train_sources) & IN_SAMPLE_ONLY_COLS
        if overlap:
            violations.append(f"{spec.name}: {overlap}")
    return violations


def validate_schema_identity(
    train_df: pd.DataFrame,
    infer_df: pd.DataFrame,
) -> None:
    """Validate that train and infer DataFrames have identical schemas.

    Checks:
    (a) Column list identity (name AND order, per IFF-03)
    (b) Per-column dtype identity

    Args:
        train_df: train mode output DataFrame.
        infer_df: infer mode output DataFrame.

    Raises:
        AssertionError: with diff information on mismatch.
    """
    train_cols = list(train_df.columns)
    infer_cols = list(infer_df.columns)

    assert train_cols == infer_cols, (
        f"Column mismatch: train has {len(train_cols)} columns, "
        f"infer has {len(infer_cols)} columns. "
        f"Train-only: {set(train_cols) - set(infer_cols)}, "
        f"Infer-only: {set(infer_cols) - set(train_cols)}"
    )

    for col in train_cols:
        assert train_df[col].dtype == infer_df[col].dtype, (
            f"dtype mismatch for '{col}': "
            f"train={train_df[col].dtype}, infer={infer_df[col].dtype}"
        )
