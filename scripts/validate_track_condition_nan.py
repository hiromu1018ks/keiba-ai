"""validate_track_condition_nan.py — NaN rate diagnostic for track condition features.

VLD-03 diagnostic script that loads horse_features.parquet and computes
surface-aware NaN rates for all track condition features with 3-tier verdict.

Per D-10: Surface-aware denominator (dirt_* -> dirt rows, turf_* -> turf rows).
Per D-11: 3-tier threshold (< 30% PASS, 30-50% WARN, >= 50% FAIL).
Per D-12: NaN cause separation (raw vs derived).
Per D-13: WARN = logged only. FAIL = detailed cause report.
Per D-14: Training start date NOT modified (report only).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from features.track_condition_features import (
    RACE_CONDITION_COLS,
    TRACK_CONDITION_COLS,
    TRACK_DERIVED_COLS,
)

logger = logging.getLogger(__name__)

# All 23 track condition features
ALL_TC_FEATURES: list[str] = TRACK_CONDITION_COLS + TRACK_DERIVED_COLS + RACE_CONDITION_COLS

# Raw columns that are the source of NaN
RAW_COLUMNS: set[str] = {"dirt_moisture", "turf_cushion"}

# Classification prefixes for surface-aware denominator
_DIRT_PREFIXES: tuple[str, ...] = ("dirt_moisture_", "moisture_")
_TURF_PREFIXES: tuple[str, ...] = ("turf_cushion_", "cushion_")
_CROSS_SURFACE_NAMES: frozenset[str] = frozenset({
    "track_front_bias_score",
    "kickback_risk_score",
    "expected_pace_class",
    "surface_condition_transition",
    "race_condition_match_score",
    "race_condition_match_max",
    "race_condition_match_ratio",
    "race_field_front_bias",
    "sire_x_cushion_band",
})


def _classify_feature(col: str) -> str:
    """Classify a feature as 'dirt', 'turf', or 'cross' for denominator selection."""
    if col in _CROSS_SURFACE_NAMES:
        return "cross"
    col_lower = col.lower()
    for prefix in _DIRT_PREFIXES:
        if col_lower.startswith(prefix):
            return "dirt"
    for prefix in _TURF_PREFIXES:
        if col_lower.startswith(prefix):
            return "turf"
    return "cross"


def _compute_nan_verdict(nan_rate: float) -> str:
    """Apply 3-tier threshold per D-11."""
    if nan_rate < 0.30:
        return "PASS"
    elif nan_rate < 0.50:
        return "WARN"
    else:
        return "FAIL"


def compute_surface_aware_nan_rates(
    df: pd.DataFrame,
    features: list[str],
) -> dict[str, dict[str, object]]:
    """Compute surface-aware NaN rates with cause separation.

    Per D-10: dirt_* features -> dirt rows only as denominator.
              turf_* features -> turf rows only.
              cross-surface features -> all rows.

    Per D-12: NaN cause separation distinguishes raw data NaN from derived
              processing NaN.
    """
    surface_col = "surface"
    has_surface = surface_col in df.columns

    if has_surface:
        turf_mask = df[surface_col] == "turf"
        dirt_mask = df[surface_col] == "dirt"
        n_turf = int(turf_mask.sum())
        n_dirt = int(dirt_mask.sum())
    else:
        turf_mask = pd.Series(True, index=df.index)
        dirt_mask = pd.Series(True, index=df.index)
        n_turf = len(df)
        n_dirt = len(df)

    total_rows = len(df)
    results: dict[str, dict[str, object]] = {}

    for col in features:
        if col not in df.columns:
            results[col] = {
                "nan_count": 0,
                "denominator": 0,
                "nan_rate": 1.0,
                "verdict": "FAIL",
                "cause_separation": {"raw_cause_pct": 1.0, "derived_cause_pct": 0.0},
            }
            continue

        classification = _classify_feature(col)

        if classification == "dirt":
            denominator = n_dirt
            feature_series = df.loc[dirt_mask, col]
            raw_col = "dirt_moisture"
            raw_in_scope = df.loc[dirt_mask, raw_col] if raw_col in df.columns else pd.Series(
                np.nan, index=df.loc[dirt_mask].index
            )
        elif classification == "turf":
            denominator = n_turf
            feature_series = df.loc[turf_mask, col]
            raw_col = "turf_cushion"
            raw_in_scope = df.loc[turf_mask, raw_col] if raw_col in df.columns else pd.Series(
                np.nan, index=df.loc[turf_mask].index
            )
        else:
            denominator = total_rows
            feature_series = df[col]
            raw_in_scope = None

        if denominator == 0:
            nan_rate = 1.0
            nan_count = 0
        else:
            nan_count = int(feature_series.isna().sum())
            nan_rate = nan_count / denominator

        verdict = _compute_nan_verdict(nan_rate)

        # Per D-12: Cause separation
        cause_separation = _compute_cause_separation(
            feature_series, raw_in_scope
        )

        results[col] = {
            "nan_count": nan_count,
            "denominator": denominator,
            "nan_rate": round(nan_rate, 4),
            "verdict": verdict,
            "cause_separation": cause_separation,
        }

    return results


def _compute_cause_separation(
    feature_series: pd.Series,
    raw_series: pd.Series | None,
) -> dict[str, float]:
    """Per D-12: Separate NaN caused by raw data missing vs derived processing.

    If raw column is NaN and derived feature is also NaN -> raw_cause.
    If raw column is NOT NaN but derived is NaN -> derived_cause.
    """
    if raw_series is None:
        # Cross-surface features: no raw source to compare
        return {"raw_cause_pct": 0.0, "derived_cause_pct": 1.0}

    feature_nan = feature_series.isna()
    raw_nan = raw_series.reindex(feature_series.index).isna()

    total_nan = int(feature_nan.sum())
    if total_nan == 0:
        return {"raw_cause_pct": 0.0, "derived_cause_pct": 0.0}

    raw_cause = int((feature_nan & raw_nan).sum())
    derived_cause = int((feature_nan & ~raw_nan).sum())

    return {
        "raw_cause_pct": round(raw_cause / total_nan, 4) if total_nan > 0 else 0.0,
        "derived_cause_pct": round(derived_cause / total_nan, 4) if total_nan > 0 else 0.0,
    }


def main() -> None:
    """CLI entry point for NaN rate diagnostic."""
    parser = argparse.ArgumentParser(
        description="Validate track condition feature NaN rates (VLD-03)"
    )
    parser.add_argument(
        "--features-path",
        default="data/features/horse_features.parquet",
        help="Path to horse_features.parquet",
    )
    parser.add_argument(
        "--start",
        default="20200101",
        help="Start date for WF Fold0 period (YYYYMMDD)",
    )
    parser.add_argument(
        "--end",
        default="20231231",
        help="End date for WF Fold0 period (YYYYMMDD)",
    )
    parser.add_argument(
        "--output",
        default="data/audit/track_condition_nan_report.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    features_path = Path(args.features_path)
    if not features_path.exists():
        logger.error("Features file not found: %s", features_path)
        sys.exit(1)

    # Load features
    logger.info("Loading features from %s", features_path)
    df = pd.read_parquet(features_path)

    # Filter by date range
    if "race_date" in df.columns:
        start_date = pd.Timestamp(args.start)
        end_date = pd.Timestamp(args.end)
        df = df[(df["race_date"] >= start_date) & (df["race_date"] <= end_date)]
        logger.info("Filtered to %d rows (%s to %s)", len(df), args.start, args.end)

    # Compute surface-aware NaN rates
    results = compute_surface_aware_nan_rates(df, ALL_TC_FEATURES)

    # Overall verdict
    verdicts = [r["verdict"] for r in results.values()]
    if any(v == "FAIL" for v in verdicts):
        overall_verdict = "FAIL"
    elif any(v == "WARN" for v in verdicts):
        overall_verdict = "WARN"
    else:
        overall_verdict = "PASS"

    # Count rows by surface
    surface_col = "surface"
    n_turf = int((df[surface_col] == "turf").sum()) if surface_col in df.columns else 0
    n_dirt = int((df[surface_col] == "dirt").sum()) if surface_col in df.columns else 0

    report = {
        "fold0_period": {"start": args.start, "end": args.end},
        "total_rows": len(df),
        "turf_rows": n_turf,
        "dirt_rows": n_dirt,
        "features": results,
        "overall_verdict": overall_verdict,
    }

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print("Track Condition NaN Rate Report (VLD-03)")
    print(f"{'='*60}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Total rows: {len(df)} (turf: {n_turf}, dirt: {n_dirt})")
    print(f"{'='*60}")

    for feat, info in results.items():
        verdict_str = info["verdict"]
        nan_rate = info["nan_rate"]
        print(f"  {feat:40s}  NaN rate: {nan_rate:6.2%}  [{verdict_str}]")

    print(f"{'='*60}")
    print(f"Overall verdict: {overall_verdict}")
    print(f"Output: {output_path}")

    # Per D-13: WARN = logged only. FAIL = detailed cause report.
    for feat, info in results.items():
        if info["verdict"] == "FAIL":
            cause = info["cause_separation"]
            print(f"\nFAIL detail for {feat}:")
            print(f"  NaN rate: {info['nan_rate']:.2%}")
            print(f"  Raw data cause: {cause['raw_cause_pct']:.1%}")
            print(f"  Derived cause: {cause['derived_cause_pct']:.1%}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
