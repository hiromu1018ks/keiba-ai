#!/usr/bin/env python3
"""Per-feature IC evaluation for track condition features (VLD-02, D-07 through D-09).

Evaluates all 23 track condition features with:
- Univariate Spearman IC: spearmanr(feature_value, kakuteijyuni)
- C-orthogonal IC: Spearman(resid, kakuteijyuni) where resid = feat - OLS(feat | odds)
- Surface-stratified: turf and dirt subsets
- Tier-level aggregation: T1/T2, T3/T4 derived, T4-02 race-level
- Category column (sire_x_cushion_band): Kruskal-Wallis instead of Spearman per D-08
- Signal classification: abs(C-orthogonal IC) >= 0.005 = signal per D-09
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(Path(ROOT) / "src"))

from features.track_condition_features import (  # noqa: E402
    RACE_CONDITION_COLS,
    TRACK_CONDITION_COLS,
    TRACK_DERIVED_COLS,
)

logger = logging.getLogger(__name__)

# All 23 track condition features
ALL_TC_FEATURES: list[str] = TRACK_CONDITION_COLS + TRACK_DERIVED_COLS + RACE_CONDITION_COLS

# Category column handled separately per D-08
CATEGORY_COLS: frozenset[str] = frozenset({"sire_x_cushion_band"})

# Signal threshold per D-09
SIGNAL_THRESHOLD: float = 0.005

# Minimum sample for valid IC
MIN_SAMPLE_SIZE: int = 1000

# Target column
TARGET_COL: str = "kakuteijyuni"

# Market odds column for C-orthogonal
MARKET_ODDS_COL: str = "tanodds"


def _compute_c_orthogonal_ic(
    feature: np.ndarray,
    odds: np.ndarray,
    target: np.ndarray,
) -> dict[str, object]:
    """C-orthogonal IC: Spearman(residual, target) where residual = feature - OLS(feature | odds).

    Per D-07: Measures signal independent of market odds.
    """
    valid = np.isfinite(feature) & np.isfinite(odds) & np.isfinite(target)
    n = int(valid.sum())
    if n < 30:
        return {"rho": float("nan"), "p_value": float("nan"), "n": n}

    f_valid = feature[valid]
    o_valid = odds[valid]
    t_valid = target[valid]

    # OLS: feature = a + b * odds
    x_with_intercept = np.column_stack([np.ones(len(o_valid)), o_valid])
    coeffs, _, _, _ = np.linalg.lstsq(x_with_intercept, f_valid, rcond=None)
    residuals = f_valid - x_with_intercept @ coeffs

    rho, p_value = spearmanr(residuals, t_valid)
    return {"rho": float(rho), "p_value": float(p_value), "n": n}


def _compute_univariate_ic(
    feature: np.ndarray,
    target: np.ndarray,
) -> dict[str, object]:
    """Univariate Spearman IC between feature and target."""
    valid = np.isfinite(feature) & np.isfinite(target)
    n = int(valid.sum())
    if n < 30:
        return {"rho": float("nan"), "p_value": float("nan"), "n": n}

    rho, p_value = spearmanr(feature[valid], target[valid])
    return {"rho": float(rho), "p_value": float(p_value), "n": n}


def _classify_signal(c_orth_rho: float) -> str:
    """Classify signal strength per D-09."""
    if not np.isfinite(c_orth_rho):
        return "weak"
    if abs(c_orth_rho) >= SIGNAL_THRESHOLD:
        return "signal"
    return "weak"


def _compute_category_evaluation(
    series: pd.Series,
    target: pd.Series,
) -> dict[str, object]:
    """Per D-08: Evaluate category column via Kruskal-Wallis, not Spearman."""
    valid = series.notna() & target.notna()
    if valid.sum() < 30:
        return {
            "category_count": 0,
            "category_target_means": {},
            "kruskal_wallis": {"H": float("nan"), "p_value": float("nan")},
            "n": int(valid.sum()),
        }

    cats = series[valid]
    tgt = target[valid]

    # Per-category target mean
    category_means: dict[str, float] = {}
    groups: list[np.ndarray] = []
    for cat_val, grp in pd.DataFrame({"cat": cats, "target": tgt}).groupby(
        "cat", observed=True
    ):
        category_means[str(cat_val)] = float(grp["target"].mean())
        groups.append(grp["target"].values)

    # Kruskal-Wallis H test
    if len(groups) >= 2:
        h_stat, p_value = kruskal(*groups)
    else:
        h_stat, p_value = float("nan"), float("nan")

    return {
        "category_count": len(category_means),
        "category_target_means": category_means,
        "kruskal_wallis": {"H": float(h_stat), "p_value": float(p_value)},
        "n": int(valid.sum()),
    }


def _compute_tier_aggregation(
    per_feature: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Per D-07: Aggregate IC by Tier groups."""
    tier_groups = {
        "T1_T2": TRACK_CONDITION_COLS,
        "T3_T4_derived": TRACK_DERIVED_COLS,
        "T4_02_race_level": RACE_CONDITION_COLS,
    }

    result: dict[str, dict[str, object]] = {}
    for tier_name, cols in tier_groups.items():
        c_ic_values: list[float] = []
        signal_count = 0
        total = 0
        for col in cols:
            if col in CATEGORY_COLS:
                continue  # Category cols excluded from numeric IC aggregation
            feat_data = per_feature.get(col, {})
            c_orth = feat_data.get("c_orthogonal_ic", {})
            rho = c_orth.get("rho", float("nan"))
            if isinstance(rho, (int, float)) and np.isfinite(rho):
                c_ic_values.append(abs(rho))
                if abs(rho) >= SIGNAL_THRESHOLD:
                    signal_count += 1
            total += 1

        mean_abs = float(np.mean(c_ic_values)) if c_ic_values else float("nan")
        result[tier_name] = {
            "mean_abs_c_ic": round(mean_abs, 6),
            "signal_count": signal_count,
            "total": total,
        }

    return result


def _compute_level_aggregation(
    per_feature: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Per D-07: Aggregate by horse-level vs race-level features."""
    race_level_cols = set(RACE_CONDITION_COLS)
    horse_c_ic: list[float] = []
    horse_signal = 0
    race_c_ic: list[float] = []
    race_signal = 0

    for col, feat_data in per_feature.items():
        if col in CATEGORY_COLS:
            continue
        c_orth = feat_data.get("c_orthogonal_ic", {})
        rho = c_orth.get("rho", float("nan"))
        if not (isinstance(rho, (int, float)) and np.isfinite(rho)):
            continue

        if col in race_level_cols:
            race_c_ic.append(abs(rho))
            if abs(rho) >= SIGNAL_THRESHOLD:
                race_signal += 1
        else:
            horse_c_ic.append(abs(rho))
            if abs(rho) >= SIGNAL_THRESHOLD:
                horse_signal += 1

    return {
        "horse_level": {
            "mean_abs_c_ic": round(float(np.mean(horse_c_ic)), 6) if horse_c_ic else float("nan"),
            "signal_count": horse_signal,
        },
        "race_level": {
            "mean_abs_c_ic": round(float(np.mean(race_c_ic)), 6) if race_c_ic else float("nan"),
            "signal_count": race_signal,
        },
    }


def _detect_flags(
    per_feature: dict[str, dict[str, object]],
) -> list[str]:
    """Flag features with sign reversal between surfaces or low samples."""
    flags: list[str] = []

    for col, feat_data in per_feature.items():
        if col in CATEGORY_COLS:
            continue

        # Sign reversal: turf and dirt IC have opposite signs
        by_surface = feat_data.get("by_surface", {})
        turf_rho = by_surface.get("turf", {}).get("rho", float("nan"))
        dirt_rho = by_surface.get("dirt", {}).get("rho", float("nan"))
        if (
            isinstance(turf_rho, (int, float))
            and isinstance(dirt_rho, (int, float))
            and np.isfinite(turf_rho)
            and np.isfinite(dirt_rho)
            and turf_rho * dirt_rho < 0
        ):
            flags.append(f"sign_reversal:{col}")

        # Low sample warning
        uni_ic = feat_data.get("univariate_ic", {})
        n = uni_ic.get("n", 0)
        if isinstance(n, (int, float)) and n < MIN_SAMPLE_SIZE:
            flags.append(f"low_samples:{col}")

    return flags


def run_track_condition_ic_eval(
    oof_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> dict[str, object]:
    """Run per-feature IC evaluation for all track condition features.

    Args:
        oof_df: OOF predictions DataFrame with kakuteijyuni, tanodds, surface, race_id.
        features_df: horse_features DataFrame with track condition feature columns.

    Returns:
        IC evaluation report dict.
    """
    # Merge OOF with features on common index/key
    # Both should have race_id + horse-level identifier
    # Use intersection of columns to avoid duplicates
    merge_cols = ["race_id"]
    if "umaban" in oof_df.columns and "umaban" in features_df.columns:
        merge_cols.append("umaban")

    # Determine which TC features are available
    available_features = [f for f in ALL_TC_FEATURES if f in features_df.columns]
    if not available_features:
        logger.warning(
            "No track condition features found in features DataFrame. "
            "Retraining with track condition features is required."
        )
        # Return empty report with note
        return {
            "per_feature": {},
            "sire_x_cushion_band": {},
            "tier_aggregation": _compute_tier_aggregation({}),
            "level_aggregation": _compute_level_aggregation({}),
            "flags": [],
            "note": "No track condition features in feature data. Retraining required.",
        }

    # Select merge subset
    feature_subset = features_df[merge_cols + available_features].copy()

    # Check if target columns exist
    if TARGET_COL not in oof_df.columns:
        logger.error("Target column '%s' not found in OOF DataFrame", TARGET_COL)
        return {"error": f"Missing {TARGET_COL} column"}

    if MARKET_ODDS_COL not in oof_df.columns:
        logger.error("Market odds column '%s' not found in OOF DataFrame", MARKET_ODDS_COL)
        return {"error": f"Missing {MARKET_ODDS_COL} column"}

    # Merge
    merged = oof_df.merge(feature_subset, on=merge_cols, how="left", suffixes=("", "_feat"))
    logger.info("Merged OOF + features: %d rows, %d columns", len(merged), len(merged.columns))

    # Target: kakuteijyuni is 1=1st, 2=2nd, etc.
    # For win prediction, we use binary target: 1 if 1st place, 0 otherwise
    target = (pd.to_numeric(merged[TARGET_COL], errors="coerce") == 1).astype(float)
    odds = pd.to_numeric(merged[MARKET_ODDS_COL], errors="coerce").values

    has_surface = "surface" in merged.columns
    if has_surface:
        turf_mask = merged["surface"] == "turf"
        dirt_mask = merged["surface"] == "dirt"
    else:
        turf_mask = pd.Series(True, index=merged.index)
        dirt_mask = pd.Series(True, index=merged.index)

    per_feature: dict[str, dict[str, object]] = {}

    for col in available_features:
        feature_values = pd.to_numeric(merged[col], errors="coerce").values

        # Univariate IC
        uni_ic = _compute_univariate_ic(feature_values, target.values)

        # C-orthogonal IC
        c_orth_ic = _compute_c_orthogonal_ic(feature_values, odds, target.values)

        # NaN rate
        nan_rate = float(pd.isna(merged[col]).mean())

        # Signal classification
        signal_class = _classify_signal(c_orth_ic.get("rho", float("nan")))

        # Surface stratification
        by_surface: dict[str, dict[str, object]] = {}
        if has_surface:
            for surf_name, surf_mask in [("turf", turf_mask), ("dirt", dirt_mask)]:
                surf_feature = feature_values[surf_mask.values]
                surf_target = target.values[surf_mask.values]
                n_valid = int(np.isfinite(surf_feature).sum())

                if n_valid >= 30:
                    rho, p_val = spearmanr(
                        surf_feature[np.isfinite(surf_feature)],
                        surf_target[np.isfinite(surf_feature)],
                    )
                    by_surface[surf_name] = {"rho": float(rho), "n": n_valid}
                else:
                    by_surface[surf_name] = {"rho": float("nan"), "n": n_valid}

        # Category column: separate evaluation per D-08
        if col in CATEGORY_COLS:
            cat_eval = _compute_category_evaluation(merged[col], target)
            per_feature[col] = {
                "univariate_ic": uni_ic,
                "c_orthogonal_ic": c_orth_ic,
                "nan_rate": round(nan_rate, 4),
                "signal_classification": signal_class,
                "by_surface": by_surface,
                "category_evaluation": cat_eval,
            }
        else:
            per_feature[col] = {
                "univariate_ic": uni_ic,
                "c_orthogonal_ic": c_orth_ic,
                "nan_rate": round(nan_rate, 4),
                "signal_classification": signal_class,
                "by_surface": by_surface,
            }

    # sire_x_cushion_band separate section if present
    sire_eval: dict[str, object] = {}
    if "sire_x_cushion_band" in per_feature:
        sire_eval = per_feature["sire_x_cushion_band"].pop("category_evaluation", {})

    # Tier aggregation
    tier_aggregation = _compute_tier_aggregation(per_feature)

    # Level aggregation
    level_aggregation = _compute_level_aggregation(per_feature)

    # Flags
    flags = _detect_flags(per_feature)

    return {
        "per_feature": per_feature,
        "sire_x_cushion_band": sire_eval,
        "tier_aggregation": tier_aggregation,
        "level_aggregation": level_aggregation,
        "flags": flags,
    }


def main() -> None:
    """CLI entry point for track condition IC evaluation."""
    parser = argparse.ArgumentParser(
        description="Per-feature IC evaluation for track condition features (VLD-02)"
    )
    parser.add_argument(
        "--oof-path",
        default="data/oof/oof_predictions.parquet",
        help="Path to OOF predictions parquet",
    )
    parser.add_argument(
        "--features-path",
        default="data/features/horse_features.parquet",
        help="Path to horse_features parquet",
    )
    parser.add_argument(
        "--output",
        default="data/audit/track_condition_ic_report.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    oof_path = Path(args.oof_path)
    features_path = Path(args.features_path)
    output_path = Path(args.output)

    # Load data
    logger.info("Loading OOF predictions from %s", oof_path)
    if not oof_path.exists():
        logger.error("OOF file not found: %s", oof_path)
        sys.exit(1)
    oof_df = pd.read_parquet(oof_path)
    logger.info("OOF: %d rows, %d columns", len(oof_df), len(oof_df.columns))

    logger.info("Loading features from %s", features_path)
    if not features_path.exists():
        logger.error("Features file not found: %s", features_path)
        sys.exit(1)
    features_df = pd.read_parquet(features_path)
    logger.info("Features: %d rows, %d columns", len(features_df), len(features_df.columns))

    # Run evaluation
    report = run_track_condition_ic_eval(oof_df, features_df)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=_json_default)

    # Print summary
    print(f"\n{'='*70}")
    print("Track Condition IC Evaluation Report (VLD-02)")
    print(f"{'='*70}")

    per_feature = report.get("per_feature", {})
    if not per_feature:
        note = report.get("note", "No features evaluated")
        print(f"  NOTE: {note}")
    else:
        for feat, data in per_feature.items():
            c_orth = data.get("c_orthogonal_ic", {})
            rho = c_orth.get("rho", float("nan"))
            classification = data.get("signal_classification", "?")
            nan_rate = data.get("nan_rate", 0)
            if isinstance(rho, (int, float)) and np.isfinite(rho):
                print(
                    f"  {feat:40s}  C-IC: {rho:+.6f}  "
                    f"NaN: {nan_rate:5.1%}  [{classification}]"
                )
            else:
                print(
                    f"  {feat:40s}  C-IC:       NaN  "
                    f"NaN: {nan_rate:5.1%}  [{classification}]"
                )

    # Tier aggregation
    tier_agg = report.get("tier_aggregation", {})
    print(f"\n{'='*70}")
    print("Tier Aggregation:")
    for tier, agg in tier_agg.items():
        mean_c = agg.get("mean_abs_c_ic", float("nan"))
        sig = agg.get("signal_count", 0)
        total = agg.get("total", 0)
        mean_str = (
            f"{mean_c:.6f}"
            if isinstance(mean_c, (int, float)) and np.isfinite(mean_c)
            else "NaN"
        )
        print(f"  {tier:20s}  mean|C-IC|: {mean_str}  signal: {sig}/{total}")

    # Flags
    flags = report.get("flags", [])
    if flags:
        print(f"\n{'='*70}")
        print("Flags:")
        for flag in flags:
            print(f"  - {flag}")

    print(f"\nOutput: {output_path}")
    print(f"{'='*70}")


def _json_default(obj: object) -> object:
    """JSON non-serializable type fallback."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
