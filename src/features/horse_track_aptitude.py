"""horse_track_aptitude.py — PIT-safe horse track condition aptitude statistics.

Precomputes per-horse historical performance under different track conditions
(wet/dry dirt, hard/soft turf) using expanding window + shift(1) pattern.

Follows horse_career_stats.parquet pattern: each (kettonum, race_id) row
reflects only past race results, ensuring no lookahead bias.

Output keyed on race_id + kettonum for FeatureEngine.build_all() merge.

Known limitation: horse_condition_type and horse_condition_versatility use only
dirt metrics (wet/dry). Turf-only horses are always "unknown" with NaN versatility.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Track condition thresholds per D-03
_DIRT_WET_THRESHOLD = 12.0  # dirt_moisture >= 12 -> wet
_DIRT_DRY_THRESHOLD = 3.0   # dirt_moisture < 3 -> dry
_TURF_HARD_THRESHOLD = 10.0  # turf_cushion >= 10 -> hard
_TURF_SOFT_THRESHOLD = 8.0   # turf_cushion < 8 -> soft

# Classification thresholds per D-05
_MIN_STARTS = 3
_HIT_RATE_THRESHOLD = 0.3

# Surface classification from horse_career_stats.py
_TURF_TRACKCD_RANGE = (10, 22)  # trackcd 10-22
_DIRT_TRACKCD_RANGE = (23, 29)  # trackcd 23-29

# Output column names
APTITUDE_COLS: list[str] = [
    # Keys
    "race_id",
    "kettonum",
    # Hit rates (4)
    "horse_dirt_wet_hit_rate",
    "horse_dirt_dry_hit_rate",
    "horse_cushion_hard_hit_rate",
    "horse_cushion_soft_hit_rate",
    # Starts counts (4)
    "horse_dirt_wet_starts_count",
    "horse_dirt_dry_starts_count",
    "horse_cushion_hard_starts_count",
    "horse_cushion_soft_starts_count",
    # Versatility (1)
    "horse_condition_versatility",
    # Category (1)
    "horse_condition_type",
    # Previous race conditions (2)
    "prev_dirt_moisture",
    "prev_turf_cushion",
]


def _classify_surface(trackcd: pd.Series) -> pd.Series:
    """Classify surface from trackcd. Returns 'turf', 'dirt', or 'other'."""
    trackcd_num = pd.to_numeric(trackcd, errors="coerce")
    is_turf = trackcd_num.between(*_TURF_TRACKCD_RANGE).fillna(False)
    is_dirt = trackcd_num.between(*_DIRT_TRACKCD_RANGE).fillna(False)
    return pd.Series(
        np.where(is_turf, "turf", np.where(is_dirt, "dirt", "other")),
        index=trackcd.index,
    )


def precompute_track_aptitude(
    entries_df: pd.DataFrame,
    races_df: pd.DataFrame,
    track_conditions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Precompute PIT-safe horse track condition aptitude statistics.

    Each (kettonum, race_id) row reflects only past race results via
    expanding window + shift(1) pattern (same as horse_career_stats.py).

    Args:
        entries_df: Race entries with columns: race_id, kettonum, kakuteijyuni, race_date
        races_df: Race info with columns: race_id, trackcd
        track_conditions_df: Track conditions with columns: race_id, dirt_moisture, turf_cushion

    Returns:
        DataFrame with 14 columns (APTITUDE_COLS), keyed on (race_id, kettonum).
    """
    if entries_df.empty:
        return pd.DataFrame(columns=APTITUDE_COLS)

    ent = entries_df.copy()

    # Dedup (same pattern as horse_career_stats.py)
    n_before = len(ent)
    ent = ent.drop_duplicates(subset=["race_id", "kettonum"], keep="last")
    n_dropped = n_before - len(ent)
    if n_dropped > 0:
        logger.warning(
            "Dropped %d duplicate (race_id,kettonum) entries (%d -> %d)",
            n_dropped, n_before, len(ent),
        )

    if ent.empty:
        return pd.DataFrame(columns=APTITUDE_COLS)

    # Merge race info
    race_cols = [c for c in ["race_id", "trackcd"] if c in races_df.columns]
    ent = ent.merge(races_df[race_cols], on="race_id", how="left")

    # Merge track conditions
    tc_cols = [c for c in ["race_id", "dirt_moisture", "turf_cushion"]
               if c in track_conditions_df.columns]
    ent = ent.merge(track_conditions_df[tc_cols], on="race_id", how="left")

    # Classify surface
    ent["surface"] = _classify_surface(ent["trackcd"])

    # Numeric conversions
    ent["kakuteijyuni_int"] = pd.to_numeric(ent["kakuteijyuni"], errors="coerce")
    ent["dirt_moisture_num"] = pd.to_numeric(ent["dirt_moisture"], errors="coerce")
    ent["turf_cushion_num"] = pd.to_numeric(ent["turf_cushion"], errors="coerce")

    # Hit: kakuteijyuni <= 3 per D-04
    ent["is_hit"] = (ent["kakuteijyuni_int"] <= 3).astype(int)
    # Valid start: kakuteijyuni > 0 and not NaN (excluded from denominator per D-04)
    ent["is_valid_start"] = (
        ent["kakuteijyuni_int"].notna() & (ent["kakuteijyuni_int"] > 0)
    ).astype(int)

    # Condition classification flags
    is_dirt = ent["surface"] == "dirt"
    is_turf = ent["surface"] == "turf"

    moisture = ent["dirt_moisture_num"]
    cushion = ent["turf_cushion_num"]

    # Dirt condition flags per D-03
    is_dirt_wet = is_dirt & moisture.notna() & (moisture >= _DIRT_WET_THRESHOLD)
    is_dirt_dry = is_dirt & moisture.notna() & (moisture < _DIRT_DRY_THRESHOLD)

    # Turf condition flags per D-03
    is_turf_hard = is_turf & cushion.notna() & (cushion >= _TURF_HARD_THRESHOLD)
    is_turf_soft = is_turf & cushion.notna() & (cushion < _TURF_SOFT_THRESHOLD)

    # Valid starts per condition (hit only counted for valid starts)
    is_valid = ent["is_valid_start"] == 1
    is_hit_flag = ent["is_hit"] == 1
    ent["valid_dirt_wet"] = (is_dirt_wet & is_valid).astype(int)
    ent["hit_dirt_wet"] = (is_dirt_wet & is_hit_flag & is_valid).astype(int)
    ent["valid_dirt_dry"] = (is_dirt_dry & is_valid).astype(int)
    ent["hit_dirt_dry"] = (is_dirt_dry & is_hit_flag & is_valid).astype(int)
    ent["valid_turf_hard"] = (is_turf_hard & is_valid).astype(int)
    ent["hit_turf_hard"] = (is_turf_hard & is_hit_flag & is_valid).astype(int)
    ent["valid_turf_soft"] = (is_turf_soft & is_valid).astype(int)
    ent["hit_turf_soft"] = (is_turf_soft & is_hit_flag & is_valid).astype(int)

    # Sort by kettonum + race_date
    if "race_date" not in ent.columns and "race_date" not in entries_df.columns:
        # Fallback: use race_id as proxy for date ordering
        logger.warning(
            "race_date not found; using race_id as date proxy (may be incorrect)"
        )
        ent = ent.sort_values(["kettonum", "race_id"]).reset_index(drop=True)
    else:
        ent = ent.sort_values(["kettonum", "race_date", "race_id"]).reset_index(drop=True)

    # PIT-safe cumulative sums: shift(1) + cumsum (excludes current race)
    cum_cols = {}
    for prefix, valid_col, hit_col in [
        ("dirt_wet", "valid_dirt_wet", "hit_dirt_wet"),
        ("dirt_dry", "valid_dirt_dry", "hit_dirt_dry"),
        ("turf_hard", "valid_turf_hard", "hit_turf_hard"),
        ("turf_soft", "valid_turf_soft", "hit_turf_soft"),
    ]:
        cum_cols[f"cum_valid_{prefix}"] = (
            ent.groupby("kettonum", observed=True)[valid_col]
            .transform(lambda x: x.shift(1).fillna(0).cumsum())
        )
        cum_cols[f"cum_hit_{prefix}"] = (
            ent.groupby("kettonum", observed=True)[hit_col]
            .transform(lambda x: x.shift(1).fillna(0).cumsum())
        )

    for col_name, series in cum_cols.items():
        ent[col_name] = series

    # Compute hit rates: hits / starts where starts > 0, else NaN
    rate_cols = {}
    for prefix in ["dirt_wet", "dirt_dry", "turf_hard", "turf_soft"]:
        starts = ent[f"cum_valid_{prefix}"]
        hits = ent[f"cum_hit_{prefix}"]
        rate = np.where(starts > 0, hits / starts, np.nan)
        rate_cols[f"horse_{prefix}_hit_rate"] = pd.Series(rate, index=ent.index)

    # Assign rate columns with proper naming
    ent["horse_dirt_wet_hit_rate"] = rate_cols["horse_dirt_wet_hit_rate"]
    ent["horse_dirt_dry_hit_rate"] = rate_cols["horse_dirt_dry_hit_rate"]
    ent["horse_cushion_hard_hit_rate"] = rate_cols["horse_turf_hard_hit_rate"]
    ent["horse_cushion_soft_hit_rate"] = rate_cols["horse_turf_soft_hit_rate"]

    # Starts counts
    ent["horse_dirt_wet_starts_count"] = ent["cum_valid_dirt_wet"]
    ent["horse_dirt_dry_starts_count"] = ent["cum_valid_dirt_dry"]
    ent["horse_cushion_hard_starts_count"] = ent["cum_valid_turf_hard"]
    ent["horse_cushion_soft_starts_count"] = ent["cum_valid_turf_soft"]

    # --- horse_condition_type (D-05) ---
    # NOTE (WR-01): Classification uses only dirt metrics (wet/dry hit rates).
    # Turf-only horses (no dirt starts) are always classified as "unknown".
    # Similarly, horse_condition_versatility will be NaN for turf-only horses.
    # This is a known limitation; extending with turf-specific metrics (hard/soft)
    # would improve coverage for the majority of JRA races (turf > dirt).
    wet_rate = ent["horse_dirt_wet_hit_rate"]
    dry_rate = ent["horse_dirt_dry_hit_rate"]
    wet_starts = ent["horse_dirt_wet_starts_count"]
    dry_starts = ent["horse_dirt_dry_starts_count"]

    wet_sufficient = (wet_starts >= _MIN_STARTS) & wet_rate.notna()
    dry_sufficient = (dry_starts >= _MIN_STARTS) & dry_rate.notna()

    condition_type = pd.Series("unknown", index=ent.index, dtype="object")

    # wet_good: wet_rate >= threshold AND dry_rate < threshold (dry must be classified too)
    is_wet_good = (
        wet_sufficient & (wet_rate >= _HIT_RATE_THRESHOLD)
        & dry_sufficient & (dry_rate < _HIT_RATE_THRESHOLD)
    )
    condition_type = condition_type.where(~is_wet_good, "wet_good")

    # dry_good: reverse
    is_dry_good = (
        dry_sufficient & (dry_rate >= _HIT_RATE_THRESHOLD)
        & wet_sufficient & (wet_rate < _HIT_RATE_THRESHOLD)
    )
    condition_type = condition_type.where(~is_dry_good, "dry_good")

    # balanced: both >= threshold
    is_balanced = (
        wet_sufficient & (wet_rate >= _HIT_RATE_THRESHOLD)
        & dry_sufficient & (dry_rate >= _HIT_RATE_THRESHOLD)
    )
    condition_type = condition_type.where(~is_balanced, "balanced")

    ent["horse_condition_type"] = condition_type

    # --- horse_condition_versatility (D-06) ---
    mean_rate = (wet_rate + dry_rate) / 2
    abs_diff = (wet_rate - dry_rate).abs()
    versatility = mean_rate * (1 - abs_diff)
    # NaN when either component rate is NaN
    versatility = versatility.where(wet_rate.notna() & dry_rate.notna(), other=np.nan)
    ent["horse_condition_versatility"] = versatility

    # --- prev_dirt_moisture / prev_turf_cushion (D-07) ---
    ent["prev_dirt_moisture"] = ent.groupby("kettonum", observed=True)[
        "dirt_moisture_num"
    ].transform(lambda x: x.shift(1))
    ent["prev_turf_cushion"] = ent.groupby("kettonum", observed=True)[
        "turf_cushion_num"
    ].transform(lambda x: x.shift(1))

    # Build output
    result = ent[APTITUDE_COLS].copy()

    logger.info(
        "Track aptitude: %d entries, %d horses",
        len(result),
        result["kettonum"].nunique(),
    )
    return result
