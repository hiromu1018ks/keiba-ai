"""v1.7 vs current backtest diff diagnostic script.

Compares baseline (v1.7) and current backtest result CSVs to identify
same-horse / different-horse bets in common races, compute ROI breakdowns,
and generate a Phase36 contribution decomposition report.

Usage:
    python scripts/diagnose_phase36_diff.py --baseline <path> --current <path> [--output <json_path>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def load_backtest_results(path: str) -> pd.DataFrame:
    """Read backtest CSV and standardize columns.

    - Lowercase all column names
    - Ensure race_id and umaban columns exist
    - Filter to win bets only if bet_type column is present
    - Return DataFrame (may be empty)
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path, dtype=str)

    # Standardize column names to lowercase
    df.columns = [col.lower().strip() for col in df.columns]

    # Convert numeric columns
    numeric_cols = ["stake", "odds", "result", "ev", "edge"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Ensure race_id is string
    if "race_id" in df.columns:
        df["race_id"] = df["race_id"].astype(str).str.strip()
    else:
        df["race_id"] = ""

    # Ensure umaban is int for matching
    if "umaban" in df.columns:
        df["umaban"] = pd.to_numeric(df["umaban"], errors="coerce").fillna(0).astype(int)

    # Filter to win bets if bet_type column exists
    if "bet_type" in df.columns:
        df = df[df["bet_type"].str.lower().str.strip() == "win"].copy()

    # Ensure quality_passed is boolean if present
    if "quality_passed" in df.columns:
        df["quality_passed"] = df["quality_passed"].astype(str).str.lower().str.strip() == "true"

    return df.reset_index(drop=True)


def find_common_races(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> tuple[set[str], set[str], set[str]]:
    """Identify intersection of race_ids between two DataFrames.

    Returns:
        (common_race_ids, baseline_only_race_ids, current_only_race_ids)
    """
    if baseline_df.empty:
        base_ids: set[str] = set()
    else:
        base_ids = set(baseline_df["race_id"].unique())

    if current_df.empty:
        curr_ids: set[str] = set()
    else:
        curr_ids = set(current_df["race_id"].unique())

    common = base_ids & curr_ids
    baseline_only = base_ids - curr_ids
    current_only = curr_ids - base_ids

    return common, baseline_only, current_only


def compute_horse_overlap(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    common_race_ids: set[str],
) -> dict[str, Any]:
    """Classify same-horse vs different-horse bets in common races.

    For each common race, identify:
    - same_horse: voted in both baseline and current
    - baseline_only: voted only in baseline
    - current_only: voted only in current

    Returns dict with counts and DataFrames for each category.
    """
    if not common_race_ids or baseline_df.empty or current_df.empty:
        return {
            "same_horse_df": pd.DataFrame(),
            "baseline_only_horse_df": pd.DataFrame(),
            "current_only_horse_df": pd.DataFrame(),
            "n_same_horse": 0,
            "n_baseline_only": 0,
            "n_current_only": 0,
        }

    # Filter to common races only
    base_common = baseline_df[baseline_df["race_id"].isin(common_race_ids)].copy()
    curr_common = current_df[current_df["race_id"].isin(common_race_ids)].copy()

    # Merge on (race_id, umaban) with indicator
    merged = base_common.merge(
        curr_common[["race_id", "umaban"]],
        on=["race_id", "umaban"],
        how="outer",
        indicator=True,
    )

    same = merged[merged["_merge"] == "both"].copy()
    bl_only = merged[merged["_merge"] == "left_only"].copy()
    cur_only = merged[merged["_merge"] == "right_only"].copy()

    # Drop merge indicator
    for df in [same, bl_only, cur_only]:
        df.drop(columns=["_merge"], inplace=True, errors="ignore")

    return {
        "same_horse_df": same,
        "baseline_only_horse_df": bl_only,
        "current_only_horse_df": cur_only,
        "n_same_horse": len(same),
        "n_baseline_only": len(bl_only),
        "n_current_only": len(cur_only),
    }


def compute_roi_breakdown(df: pd.DataFrame) -> dict[str, float]:
    """Compute ROI breakdown for a bet DataFrame.

    Returns dict with: n_bets, total_stake, total_return, roi, win_rate.
    Handles empty DataFrame (returns zeros).
    """
    if df.empty or "stake" not in df.columns:
        return {
            "n_bets": 0,
            "total_stake": 0.0,
            "total_return": 0.0,
            "roi": 0.0,
            "win_rate": 0.0,
        }

    n_bets = len(df)
    total_stake = float(df["stake"].sum())
    total_return = float(df["result"].sum()) if "result" in df.columns else 0.0
    roi = total_return / total_stake if total_stake > 0 else 0.0

    n_wins = int((df["result"] > 0).sum()) if "result" in df.columns else 0
    win_rate = n_wins / n_bets if n_bets > 0 else 0.0

    return {
        "n_bets": n_bets,
        "total_stake": total_stake,
        "total_return": total_return,
        "roi": roi,
        "win_rate": win_rate,
    }


def generate_report(baseline_path: str, current_path: str) -> dict[str, Any]:
    """Generate full comparison report between baseline and current backtest results.

    Returns dict with:
    - n_common_races, n_baseline_only_races, n_current_only_races
    - same_horse: ROI breakdown
    - baseline_only_horse: ROI breakdown
    - current_only_horse: ROI breakdown
    - phase36_contribution: decomposition sub-dict
    """
    base_df = load_backtest_results(baseline_path)
    curr_df = load_backtest_results(current_path)

    common, base_only, curr_only = find_common_races(base_df, curr_df)
    overlap = compute_horse_overlap(base_df, curr_df, common)

    # ROI for same-horse bets (need original df rows for stake/result)
    # Build same-horse ROI from baseline data (matches are the same bets)
    same_df = _extract_matching_rows(
        base_df, overlap["same_horse_df"], "race_id", "umaban"
    )
    bl_only_df = _extract_matching_rows(
        base_df, overlap["baseline_only_horse_df"], "race_id", "umaban"
    )
    cur_only_df = _extract_matching_rows(
        curr_df, overlap["current_only_horse_df"], "race_id", "umaban"
    )

    same_roi = compute_roi_breakdown(same_df)
    bl_only_roi = compute_roi_breakdown(bl_only_df)
    cur_only_roi = compute_roi_breakdown(cur_only_df)

    # Phase36 contribution decomposition
    phase36 = _compute_phase36_contribution(
        same_roi, cur_only_roi, bl_only_roi, cur_only_df, overlap, base_df, curr_df
    )

    return {
        "n_common_races": len(common),
        "n_baseline_only_races": len(base_only),
        "n_current_only_races": len(curr_only),
        "same_horse": same_roi,
        "baseline_only_horse": bl_only_roi,
        "current_only_horse": cur_only_roi,
        "phase36_contribution": phase36,
    }


def _extract_matching_rows(
    source_df: pd.DataFrame,
    match_df: pd.DataFrame,
    key1: str,
    key2: str,
) -> pd.DataFrame:
    """Extract rows from source_df that match (key1, key2) pairs in match_df."""
    if match_df.empty or source_df.empty:
        return pd.DataFrame()

    if key1 not in match_df.columns or key2 not in match_df.columns:
        return pd.DataFrame()

    if key1 not in source_df.columns or key2 not in source_df.columns:
        return pd.DataFrame()

    # Create merge keys
    match_pairs = set(
        zip(match_df[key1].astype(str), match_df[key2].astype(str))
    )
    mask = source_df.apply(
        lambda row: (str(row[key1]), str(row[key2])) in match_pairs,
        axis=1,
    )
    return source_df[mask].copy()


def _compute_phase36_contribution(
    same_roi: dict[str, float],
    cur_only_roi: dict[str, float],
    bl_only_roi: dict[str, float],
    cur_only_df: pd.DataFrame,
    overlap: dict[str, Any],
    base_df: pd.DataFrame,
    curr_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute Phase36 contribution decomposition.

    Sub-keys:
    - roi_same: ROI for same-horse group (shared baseline)
    - roi_new: ROI for current_only_horse group (added/changed by fix)
    - roi_removed: ROI for baseline_only_horse group (removed by fix)
    - n_new_horses: count of new horses in current
    - n_removed_horses: count of removed horses from baseline
    - net_contribution_pct: net Phase36 contribution percentage
    - screener_fix_count: races where quality_passed changed from False to True
    - ev_tail_count: current_only_horse bets where EV >= 1.5
    """
    roi_same = same_roi["roi"]
    roi_new = cur_only_roi["roi"]
    roi_removed = bl_only_roi["roi"]

    n_new = overlap["n_current_only"]
    n_removed = overlap["n_baseline_only"]

    # Net contribution: (new ROI * new stake - removed ROI * removed stake) / total stake
    total_stake_common = same_roi["total_stake"] + cur_only_roi["total_stake"] + bl_only_roi["total_stake"]
    if total_stake_common > 0:
        net_contribution = (
            (roi_new * cur_only_roi["total_stake"])
            - (roi_removed * bl_only_roi["total_stake"])
        ) / total_stake_common
    else:
        net_contribution = 0.0

    # Screener fix count: races where quality_passed changed from False (baseline) to True (current)
    screener_fix_count = 0
    if not base_df.empty and not curr_df.empty:
        if "quality_passed" in base_df.columns and "quality_passed" in curr_df.columns:
            base_race_quality = (
                base_df.groupby("race_id")["quality_passed"].any()
            )
            curr_race_quality = (
                curr_df.groupby("race_id")["quality_passed"].any()
            )
            for rid in base_race_quality.index:
                if rid in curr_race_quality.index:
                    if not base_race_quality[rid] and curr_race_quality[rid]:
                        screener_fix_count += 1

    # EV tail count: current_only_horse bets with EV >= 1.5
    ev_tail_count = 0
    if not cur_only_df.empty and "ev" in cur_only_df.columns:
        ev_tail_count = int((cur_only_df["ev"] >= 1.5).sum())

    return {
        "roi_same": roi_same,
        "roi_new": roi_new,
        "roi_removed": roi_removed,
        "n_new_horses": n_new,
        "n_removed_horses": n_removed,
        "net_contribution_pct": net_contribution,
        "screener_fix_count": screener_fix_count,
        "ev_tail_count": ev_tail_count,
    }


def main() -> None:
    """CLI entry point for diff diagnostic."""
    parser = argparse.ArgumentParser(
        description="Compare v1.7 baseline and current backtest results"
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to baseline (v1.7) backtest CSV file",
    )
    parser.add_argument(
        "--current",
        required=True,
        help="Path to current backtest CSV file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for JSON report output",
    )

    args = parser.parse_args()

    report = generate_report(args.baseline, args.current)

    # Print summary to stdout
    print("=" * 60)
    print("  Phase 36/36.1.1 Diff Diagnostic Report")
    print("=" * 60)
    print()
    print(f"Common races:      {report['n_common_races']}")
    print(f"Baseline-only:     {report['n_baseline_only_races']}")
    print(f"Current-only:      {report['n_current_only_races']}")
    print()
    print("--- Same Horse (both versions) ---")
    _print_roi(report["same_horse"])
    print()
    print("--- Baseline-Only Horse ---")
    _print_roi(report["baseline_only_horse"])
    print()
    print("--- Current-Only Horse ---")
    _print_roi(report["current_only_horse"])
    print()
    print("--- Phase36 Contribution ---")
    p36 = report["phase36_contribution"]
    print(f"  ROI same:        {p36['roi_same']:.4f}")
    print(f"  ROI new:         {p36['roi_new']:.4f}")
    print(f"  ROI removed:     {p36['roi_removed']:.4f}")
    print(f"  New horses:      {p36['n_new_horses']}")
    print(f"  Removed horses:  {p36['n_removed_horses']}")
    print(f"  Net contribution: {p36['net_contribution_pct']:.4f}")
    print(f"  Screener fixes:  {p36['screener_fix_count']}")
    print(f"  EV tail bets:    {p36['ev_tail_count']}")

    # Write JSON report if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to: {args.output}")


def _print_roi(breakdown: dict[str, float]) -> None:
    """Print ROI breakdown to stdout."""
    print(f"  Bets:   {breakdown['n_bets']}")
    print(f"  Stake:  {breakdown['total_stake']:.0f}")
    print(f"  Return: {breakdown['total_return']:.0f}")
    print(f"  ROI:    {breakdown['roi']:.4f}")
    print(f"  Win%:   {breakdown['win_rate']:.2%}")


if __name__ == "__main__":
    main()
