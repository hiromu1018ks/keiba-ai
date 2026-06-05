"""Group F: 馬場状態トラック条件特徴量 (含水率/クッション値)

Phase 48: Tier 1+2 interaction features from track_conditions.parquet.
Phase 49: Tier 3-04 season deviation, Tier 4-01 bias/pace, T4-03 anomaly,
          T4-04 interactions, T4-02 race-level aggregation.

Raw values (dirt_moisture, turf_cushion) are merged in FeatureEngine.build_all().
Interaction features are computed here after HorseHistoryFeatures provides kyakusitukubun_cd.

Surface-aware design:
- dirt_moisture features are naturally NaN for turf races (dirt_moisture is NaN)
- turf_cushion features are naturally NaN for dirt races (turf_cushion is NaN)
- LightGBM handles NaN natively
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 8 track condition interaction features (T1: 3, T2: 5)
TRACK_CONDITION_COLS: list[str] = [
    # T1-01: ダート含水率 × 脚質 (数値積)
    "dirt_moisture_x_kyakusitu",
    # T1-02: 芝クッション値 競馬場別相対値・zscore
    "turf_cushion_track_relative",
    "turf_cushion_track_zscore",
    # T2-01: ダート含水率 × 枠位置 + フラグ
    "dirt_moisture_x_barrier_pos",
    "dirt_moisture_high_flag",
    "dirt_moisture_dry_flag",
    # T2-02: 芝クッション値 × 脚質 (数値積)
    "turf_cushion_x_kyakusitu",
    # T2-03: 種牡馬 × クッション値ビン (カテゴリ積)
    "sire_x_cushion_band",
]

# 11 derived/higher-order track condition features (T3-04 + T4-01 + T4-03 + T4-04)
TRACK_DERIVED_COLS: list[str] = [
    # T4-01: ペース・バイアススコア
    "track_front_bias_score",
    "kickback_risk_score",
    "expected_pace_class",
    # T3-04: 季節偏差
    "cushion_season_deviation",
    "moisture_season_deviation",
    # T4-03: 異常値フラグ
    "cushion_anomaly_flag",
    "moisture_extreme_flag",
    # T4-04: 既存特徴量インタラクション (3 products + 1 transition)
    "cushion_x_distance",
    "moisture_x_weight",
    "cushion_x_age",
    "surface_condition_transition",
]

# 4 race-level track condition features (T4-02)
RACE_CONDITION_COLS: list[str] = [
    "race_condition_match_score",
    "race_condition_match_max",
    "race_condition_match_ratio",
    "race_field_front_bias",
]


def _compute_track_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute trackcd-level mean/std for turf_cushion from training data.

    Called in _train_submodel() to generate statistics stored on SubmodelSet.
    Used by compute_track_condition_features() at inference time.

    Args:
        df: Training DataFrame with 'turf_cushion' and 'trackcd' columns.

    Returns:
        Dict mapping trackcd -> {"mean": float, "std": float}.
    """
    if "turf_cushion" not in df.columns or "trackcd" not in df.columns:
        return {}

    valid = df[["trackcd", "turf_cushion"]].dropna(subset=["turf_cushion"])
    if valid.empty:
        return {}

    stats: dict[str, dict[str, float]] = {}
    for trackcd, group in valid.groupby("trackcd", observed=True):
        cushion_vals = pd.to_numeric(group["turf_cushion"], errors="coerce").dropna()
        if len(cushion_vals) >= 2:
            stats[str(trackcd)] = {
                "mean": float(cushion_vals.mean()),
                "std": float(cushion_vals.std()),
            }
    return stats


def _compute_track_month_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute trackcd x month level mean/std for cushion and moisture from training data.

    Extends _compute_track_stats pattern with month dimension for T3-04 season deviation.
    Called in _train_submodel() and stored on SubmodelSet.track_month_stats.

    Args:
        df: Training DataFrame with turf_cushion, dirt_moisture, trackcd, and race_date columns.

    Returns:
        Dict mapping "trackcd_month" -> {"cushion_mean", "cushion_std",
        "moisture_mean", "moisture_std"}.
    """
    has_cushion = "turf_cushion" in df.columns
    has_moisture = "dirt_moisture" in df.columns
    has_trackcd = "trackcd" in df.columns

    if (not has_cushion and not has_moisture) or not has_trackcd:
        return {}

    # Derive month from race_date if not present
    if "month" not in df.columns:
        if "race_date" not in df.columns:
            return {}
        df = df.copy()
        df["month"] = pd.to_datetime(df["race_date"], errors="coerce").dt.month

    valid = df[["trackcd", "month"]].copy()
    if has_cushion:
        valid["turf_cushion"] = pd.to_numeric(df["turf_cushion"], errors="coerce")
    if has_moisture:
        valid["dirt_moisture"] = pd.to_numeric(df["dirt_moisture"], errors="coerce")

    stats: dict[str, dict[str, float]] = {}
    for (trackcd, month), group in valid.groupby(["trackcd", "month"], observed=True):
        entry: dict[str, float] = {}
        if has_cushion:
            cushion_vals = group["turf_cushion"].dropna()
            if len(cushion_vals) >= 2:
                entry["cushion_mean"] = float(cushion_vals.mean())
                entry["cushion_std"] = float(cushion_vals.std())
        if has_moisture:
            moisture_vals = group["dirt_moisture"].dropna()
            if len(moisture_vals) >= 2:
                entry["moisture_mean"] = float(moisture_vals.mean())
                entry["moisture_std"] = float(moisture_vals.std())
        if entry:
            stats[f"{trackcd}_{int(month)}"] = entry

    return stats


def compute_track_condition_features(
    df: pd.DataFrame,
    *,
    track_stats: dict[str, dict[str, float]] | None = None,
    track_month_stats: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Compute track condition interaction features.

    Each feature is guarded by column existence checks and propagates NaN
    via .where() for numeric products. Surface-aware: dirt_moisture features
    are naturally NaN for turf races, turf_cushion features for dirt races.

    Args:
        df: Input DataFrame with track condition columns merged by FeatureEngine.
        track_stats: Training-period trackcd statistics for T1-02 features.
            If None, relative/zscore features are skipped.
        track_month_stats: Training-period trackcd x month statistics for T3-04.
            If None, season deviation features are skipped.

    Returns:
        Copy of df with new feature columns appended.
    """
    df = df.copy()

    # --- T1-01: dirt_moisture_x_kyakusitu (numeric product) ---
    if "dirt_moisture" in df.columns and "kyakusitukubun_cd" in df.columns:
        moisture = pd.to_numeric(df["dirt_moisture"], errors="coerce")
        kyakusitu = pd.to_numeric(df["kyakusitukubun_cd"], errors="coerce")
        df["dirt_moisture_x_kyakusitu"] = (moisture * kyakusitu).where(
            moisture.notna() & kyakusitu.notna(),
            other=float("nan"),
        )

    # --- T1-02: turf_cushion_track_relative / turf_cushion_track_zscore ---
    if (
        "turf_cushion" in df.columns
        and "trackcd" in df.columns
        and track_stats is not None
        and len(track_stats) > 0
    ):
        cushion = pd.to_numeric(df["turf_cushion"], errors="coerce")
        trackcd_str = df["trackcd"].astype(str)

        mean_map = {k: v["mean"] for k, v in track_stats.items()}
        std_map = {k: v["std"] for k, v in track_stats.items()}

        track_mean = trackcd_str.map(mean_map)
        track_std = trackcd_str.map(std_map)

        # relative = cushion - track_mean (NaN-safe)
        relative = (cushion - track_mean).where(
            cushion.notna() & track_mean.notna(),
            other=float("nan"),
        )
        df["turf_cushion_track_relative"] = relative

        # zscore = relative / track_std (std==0 or NaN produces NaN)
        zscore = (relative / track_std).where(
            relative.notna() & track_std.notna() & (track_std > 0),
            other=float("nan"),
        )
        df["turf_cushion_track_zscore"] = zscore

    # --- T2-01: dirt_moisture_x_barrier_pos + flags ---
    if "dirt_moisture" in df.columns:
        moisture = pd.to_numeric(df["dirt_moisture"], errors="coerce")

        if "frame_number" in df.columns:
            frame = pd.to_numeric(df["frame_number"], errors="coerce")
            df["dirt_moisture_x_barrier_pos"] = (moisture * frame).where(
                moisture.notna() & frame.notna(),
                other=float("nan"),
            )

        # High moisture flag (>12)
        df["dirt_moisture_high_flag"] = (moisture > 12).astype(float).where(
            moisture.notna(),
            other=float("nan"),
        )

        # Dry flag (<3)
        df["dirt_moisture_dry_flag"] = (moisture < 3).astype(float).where(
            moisture.notna(),
            other=float("nan"),
        )

    # --- T2-02: turf_cushion_x_kyakusitu (numeric product) ---
    if "turf_cushion" in df.columns and "kyakusitukubun_cd" in df.columns:
        cushion = pd.to_numeric(df["turf_cushion"], errors="coerce")
        kyakusitu = pd.to_numeric(df["kyakusitukubun_cd"], errors="coerce")
        df["turf_cushion_x_kyakusitu"] = (cushion * kyakusitu).where(
            cushion.notna() & kyakusitu.notna(),
            other=float("nan"),
        )

    # --- T2-03: sire_x_cushion_band (category interaction) ---
    if "sire_id" in df.columns and "turf_cushion" in df.columns:
        cushion = pd.to_numeric(df["turf_cushion"], errors="coerce")
        # Fixed 5-bin: [0, 7, 8, 9, 10, inf] per D-12
        cushion_band = pd.cut(
            cushion,
            bins=[0, 7, 8, 9, 10, float("inf")],
            labels=["very_soft", "soft", "standard", "firm", "very_firm"],
            right=True,
        )
        # String concatenation + category type per D-13
        interaction = (
            df["sire_id"].astype(str) + "_" + cushion_band.astype(str)
        )
        # Rows where either is NaN produce NaN
        interaction = interaction.where(
            df["sire_id"].notna() & cushion.notna(),
            other=pd.NA,
        )
        df["sire_x_cushion_band"] = interaction.astype("category")

    # === Phase 49 additions below ===

    # --- T4-01: track_front_bias_score / kickback_risk_score / expected_pace_class ---
    has_dirt_moisture = "dirt_moisture" in df.columns
    has_turf_cushion = "turf_cushion" in df.columns

    front_bias = pd.Series(np.nan, index=df.index, dtype=float)
    kickback = pd.Series(np.nan, index=df.index, dtype=float)

    if has_dirt_moisture:
        moisture = pd.to_numeric(df["dirt_moisture"], errors="coerce")
        # D-09: dirt -> clip((moisture - 3) / 9, 0, 1)
        dirt_bias = ((moisture - 3.0) / 9.0).clip(0.0, 1.0)
        dirt_kickback = ((12.0 - moisture) / 9.0).clip(0.0, 1.0)
        front_bias = front_bias.where(moisture.isna(), other=dirt_bias)
        kickback = kickback.where(moisture.isna(), other=dirt_kickback)

    if has_turf_cushion:
        cushion = pd.to_numeric(df["turf_cushion"], errors="coerce")
        # D-09: turf -> clip((cushion - 8) / 2, 0, 1)
        turf_bias = ((cushion - 8.0) / 2.0).clip(0.0, 1.0)
        turf_kickback = ((10.0 - cushion) / 2.0).clip(0.0, 1.0)
        # Only fill where front_bias is still NaN (i.e. dirt wasn't available)
        front_bias = front_bias.where(
            front_bias.notna() | cushion.isna(), other=turf_bias
        )
        kickback = kickback.where(
            kickback.notna() | cushion.isna(), other=turf_kickback
        )

    df["track_front_bias_score"] = front_bias
    df["kickback_risk_score"] = kickback

    # expected_pace_class: 0=slow, 1=neutral, 2=fast (D-10)
    pace = pd.Series(1.0, index=df.index, dtype=float)  # default neutral
    pace = pace.where(front_bias.notna() & kickback.notna(), other=np.nan)
    # slow: front_bias > 0.6 AND kickback < 0.4
    is_slow = (front_bias > 0.6) & (kickback < 0.4)
    pace = pace.where(~is_slow, other=0.0)
    # fast: front_bias < 0.4 AND kickback > 0.6
    is_fast = (front_bias < 0.4) & (kickback > 0.6)
    pace = pace.where(~is_fast, other=2.0)
    df["expected_pace_class"] = pace

    # --- T3-04: season deviation (track_month_stats lookup) ---
    if (
        track_month_stats is not None
        and len(track_month_stats) > 0
        and "trackcd" in df.columns
    ):
        # Derive month from race_date
        if "month" not in df.columns and "race_date" in df.columns:
            month = pd.to_datetime(df["race_date"], errors="coerce").dt.month
        elif "month" in df.columns:
            month = df["month"]
        else:
            month = None

        if month is not None:
            trackcd_str = df["trackcd"].astype(str)
            month_key = trackcd_str + "_" + month.astype(int).astype(str)

            if has_turf_cushion:
                cushion = pd.to_numeric(df["turf_cushion"], errors="coerce")
                cushion_mean_map = {}
                cushion_std_map = {}
                for k, v in track_month_stats.items():
                    if "cushion_mean" in v:
                        cushion_mean_map[k] = v["cushion_mean"]
                    if "cushion_std" in v:
                        cushion_std_map[k] = v["cushion_std"]

                cushion_month_mean = month_key.map(cushion_mean_map)
                cushion_month_std = month_key.map(cushion_std_map)
                cushion_deviation = (
                    (cushion - cushion_month_mean) / cushion_month_std
                ).where(
                    cushion.notna()
                    & cushion_month_mean.notna()
                    & cushion_month_std.notna()
                    & (cushion_month_std > 0),
                    other=float("nan"),
                )
                df["cushion_season_deviation"] = cushion_deviation
            else:
                df["cushion_season_deviation"] = float("nan")

            if has_dirt_moisture:
                moisture = pd.to_numeric(df["dirt_moisture"], errors="coerce")
                moisture_mean_map = {}
                moisture_std_map = {}
                for k, v in track_month_stats.items():
                    if "moisture_mean" in v:
                        moisture_mean_map[k] = v["moisture_mean"]
                    if "moisture_std" in v:
                        moisture_std_map[k] = v["moisture_std"]

                moisture_month_mean = month_key.map(moisture_mean_map)
                moisture_month_std = month_key.map(moisture_std_map)
                moisture_deviation = (
                    (moisture - moisture_month_mean) / moisture_month_std
                ).where(
                    moisture.notna()
                    & moisture_month_mean.notna()
                    & moisture_month_std.notna()
                    & (moisture_month_std > 0),
                    other=float("nan"),
                )
                df["moisture_season_deviation"] = moisture_deviation
            else:
                df["moisture_season_deviation"] = float("nan")
    else:
        # No track_month_stats -> create season deviation columns as NaN
        # so downstream T4-03 anomaly flags are always produced
        if "cushion_season_deviation" not in df.columns:
            df["cushion_season_deviation"] = float("nan")
        if "moisture_season_deviation" not in df.columns:
            df["moisture_season_deviation"] = float("nan")

    # --- T4-03: anomaly flags (D-15) ---
    if "cushion_season_deviation" in df.columns:
        dev = df["cushion_season_deviation"]
        df["cushion_anomaly_flag"] = (dev.abs() > 2.0).astype(float).where(
            dev.notna(), other=float("nan")
        )
    if "moisture_season_deviation" in df.columns:
        dev = df["moisture_season_deviation"]
        df["moisture_extreme_flag"] = (dev.abs() > 2.0).astype(float).where(
            dev.notna(), other=float("nan")
        )

    # --- T4-04: numeric interactions (D-16) ---
    if has_turf_cushion and "kyori" in df.columns:
        cushion = pd.to_numeric(df["turf_cushion"], errors="coerce")
        kyori = pd.to_numeric(df["kyori"], errors="coerce")
        df["cushion_x_distance"] = (cushion * kyori).where(
            cushion.notna() & kyori.notna(),
            other=float("nan"),
        )

    if has_dirt_moisture and "bataijyu" in df.columns:
        moisture = pd.to_numeric(df["dirt_moisture"], errors="coerce")
        bataijyu = pd.to_numeric(df["bataijyu"], errors="coerce")
        df["moisture_x_weight"] = (moisture * bataijyu).where(
            moisture.notna() & bataijyu.notna(),
            other=float("nan"),
        )

    if has_turf_cushion and "barei" in df.columns:
        cushion = pd.to_numeric(df["turf_cushion"], errors="coerce")
        barei = pd.to_numeric(df["barei"], errors="coerce")
        df["cushion_x_age"] = (cushion * barei).where(
            cushion.notna() & barei.notna(),
            other=float("nan"),
        )

    # T4-04: surface_condition_transition (D-17)
    # dirt: dirt_moisture - prev_dirt_moisture; turf: turf_cushion - prev_turf_cushion
    transition = pd.Series(np.nan, index=df.index, dtype=float)

    if has_dirt_moisture and "prev_dirt_moisture" in df.columns:
        moisture = pd.to_numeric(df["dirt_moisture"], errors="coerce")
        prev_moisture = pd.to_numeric(df["prev_dirt_moisture"], errors="coerce")
        dirt_transition = (moisture - prev_moisture).where(
            moisture.notna() & prev_moisture.notna(),
            other=float("nan"),
        )
        transition = transition.where(dirt_transition.isna(), other=dirt_transition)

    if has_turf_cushion and "prev_turf_cushion" in df.columns:
        cushion = pd.to_numeric(df["turf_cushion"], errors="coerce")
        prev_cushion = pd.to_numeric(df["prev_turf_cushion"], errors="coerce")
        turf_transition = (cushion - prev_cushion).where(
            cushion.notna() & prev_cushion.notna(),
            other=float("nan"),
        )
        # Only fill where transition is still NaN
        transition = transition.where(
            transition.notna() | turf_transition.isna(),
            other=turf_transition,
        )

    df["surface_condition_transition"] = transition

    # Ensure all TRACK_DERIVED_COLS exist (NaN fallback for missing prerequisites)
    for col in TRACK_DERIVED_COLS:
        if col not in df.columns:
            df[col] = float("nan")

    return df


def compute_race_condition_features(
    df: pd.DataFrame,
    *,
    track_month_stats: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Compute race-level track condition aggregation features (T4-02).

    Aggregates per-horse aptitude data to race-level scores.
    Must be called after compute_track_condition_features() which produces
    track_front_bias_score.

    Args:
        df: DataFrame with T3 aptitude columns, dirt_moisture, turf_cushion,
            kyakusitukubun_cd, track_front_bias_score.
        track_month_stats: Unused placeholder for API symmetry.

    Returns:
        Copy of df with 4 race-level feature columns added (broadcast to all entries).
    """
    if df.empty:
        return df

    df = df.copy()

    # Thresholds per D-03 / D-11
    dirt_wet_threshold = 12.0
    dirt_dry_threshold = 3.0
    turf_hard_threshold = 10.0
    turf_soft_threshold = 8.0
    hit_rate_threshold = 0.3
    min_starts = 3

    # --- race_condition_match_score/max/ratio ---
    # Determine current condition per race and select matching aptitude rate
    has_dirt = "dirt_moisture" in df.columns
    has_turf = "turf_cushion" in df.columns

    if "race_id" in df.columns and (has_dirt or has_turf):
        # For each entry, pick the matching aptitude rate
        match_rate = pd.Series(np.nan, index=df.index, dtype=float)
        starts_count = pd.Series(np.nan, index=df.index, dtype=float)

        if has_dirt:
            moisture = pd.to_numeric(df["dirt_moisture"], errors="coerce")
            is_wet = moisture.notna() & (moisture >= dirt_wet_threshold)
            is_dry = moisture.notna() & (moisture < dirt_dry_threshold)

            wet_rate = (
                pd.to_numeric(df["horse_dirt_wet_hit_rate"], errors="coerce")
                if "horse_dirt_wet_hit_rate" in df.columns
                else pd.Series(np.nan, index=df.index)
            )
            dry_rate = (
                pd.to_numeric(df["horse_dirt_dry_hit_rate"], errors="coerce")
                if "horse_dirt_dry_hit_rate" in df.columns
                else pd.Series(np.nan, index=df.index)
            )
            wet_starts = (
                pd.to_numeric(df["horse_dirt_wet_starts_count"], errors="coerce")
                if "horse_dirt_wet_starts_count" in df.columns
                else pd.Series(np.nan, index=df.index)
            )
            dry_starts = (
                pd.to_numeric(df["horse_dirt_dry_starts_count"], errors="coerce")
                if "horse_dirt_dry_starts_count" in df.columns
                else pd.Series(np.nan, index=df.index)
            )

            # Wet dirt: use wet_rate
            match_rate = match_rate.where(~is_wet, other=wet_rate)
            starts_count = starts_count.where(~is_wet, other=wet_starts)
            # Dry dirt: use dry_rate
            match_rate = match_rate.where(~is_dry, other=dry_rate)
            starts_count = starts_count.where(~is_dry, other=dry_starts)
            # Middle dirt: mean of both
            is_dirt_middle = moisture.notna() & ~is_wet & ~is_dry
            mid_rate = (wet_rate.fillna(0) + dry_rate.fillna(0)) / 2
            # Use NaN where both rates are NaN
            either_valid = wet_rate.notna() | dry_rate.notna()
            mid_rate = mid_rate.where(either_valid, other=np.nan)
            match_rate = match_rate.where(~is_dirt_middle, other=mid_rate)
            mid_starts = (wet_starts.fillna(0) + dry_starts.fillna(0))
            mid_starts = mid_starts.where(either_valid, other=np.nan)
            starts_count = starts_count.where(~is_dirt_middle, other=mid_starts)

        if has_turf:
            cushion = pd.to_numeric(df["turf_cushion"], errors="coerce")
            is_hard = cushion.notna() & (cushion >= turf_hard_threshold)
            is_soft = cushion.notna() & (cushion < turf_soft_threshold)

            hard_rate = (
                pd.to_numeric(df["horse_cushion_hard_hit_rate"], errors="coerce")
                if "horse_cushion_hard_hit_rate" in df.columns
                else pd.Series(np.nan, index=df.index)
            )
            soft_rate = (
                pd.to_numeric(df["horse_cushion_soft_hit_rate"], errors="coerce")
                if "horse_cushion_soft_hit_rate" in df.columns
                else pd.Series(np.nan, index=df.index)
            )
            hard_starts = (
                pd.to_numeric(df["horse_cushion_hard_starts_count"], errors="coerce")
                if "horse_cushion_hard_starts_count" in df.columns
                else pd.Series(np.nan, index=df.index)
            )
            soft_starts = (
                pd.to_numeric(df["horse_cushion_soft_starts_count"], errors="coerce")
                if "horse_cushion_soft_starts_count" in df.columns
                else pd.Series(np.nan, index=df.index)
            )

            # Hard turf: use hard_rate
            match_rate = match_rate.where(~is_hard, other=hard_rate)
            starts_count = starts_count.where(~is_hard, other=hard_starts)
            # Soft turf: use soft_rate
            match_rate = match_rate.where(~is_soft, other=soft_rate)
            starts_count = starts_count.where(~is_soft, other=soft_starts)
            # Middle turf: mean of both
            is_turf_middle = cushion.notna() & ~is_hard & ~is_soft
            mid_rate = (hard_rate.fillna(0) + soft_rate.fillna(0)) / 2
            either_valid = hard_rate.notna() | soft_rate.notna()
            mid_rate = mid_rate.where(either_valid, other=np.nan)
            match_rate = match_rate.where(~is_turf_middle, other=mid_rate)
            mid_starts = (hard_starts.fillna(0) + soft_starts.fillna(0))
            mid_starts = mid_starts.where(either_valid, other=np.nan)
            starts_count = starts_count.where(~is_turf_middle, other=mid_starts)

        # Broadcast: groupby race_id
        valid_mask = match_rate.notna()
        n_valid = valid_mask.groupby(df["race_id"], observed=True).transform("sum")

        # race_condition_match_score = mean of matching rate per race
        race_mean = (
            match_rate.where(valid_mask)
            .groupby(df["race_id"], observed=True)
            .transform("mean")
        )
        df["race_condition_match_score"] = race_mean

        # race_condition_match_max = max of matching rate per race
        race_max = (
            match_rate.where(valid_mask)
            .groupby(df["race_id"], observed=True)
            .transform("max")
        )
        df["race_condition_match_max"] = race_max

        # race_condition_match_ratio = count(rate >= threshold AND starts >= min_starts) / n_valid
        qualified = (match_rate >= hit_rate_threshold) & (starts_count >= min_starts)
        qualified_count = qualified.groupby(df["race_id"], observed=True).transform("sum")
        ratio = (qualified_count / n_valid).where(n_valid > 0, other=float("nan"))
        df["race_condition_match_ratio"] = ratio

    # --- race_field_front_bias (D-14) ---
    if (
        "race_id" in df.columns
        and "kyakusitukubun_cd" in df.columns
        and "track_front_bias_score" in df.columns
    ):
        kyakusitu = pd.to_numeric(df["kyakusitukubun_cd"], errors="coerce")
        # front_runner: kyakusitukubun_cd in [1, 2] (escape/precede)
        is_front_runner = kyakusitu.notna() & kyakusitu.isin([1.0, 2.0])
        valid_entries = kyakusitu.notna()
        n_valid = valid_entries.groupby(df["race_id"], observed=True).transform("sum")
        n_front = is_front_runner.groupby(df["race_id"], observed=True).transform("sum")
        front_runner_ratio = (n_front / n_valid).where(n_valid > 0, other=float("nan"))

        bias = pd.to_numeric(df["track_front_bias_score"], errors="coerce")
        df["race_field_front_bias"] = (front_runner_ratio * bias).where(
            front_runner_ratio.notna() & bias.notna(),
            other=float("nan"),
        )

    # Ensure all RACE_CONDITION_COLS exist (NaN fallback for missing prerequisites)
    for col in RACE_CONDITION_COLS:
        if col not in df.columns:
            df[col] = float("nan")

    return df
