"""Group F: 馬場状態トラック条件特徴量 (含水率/クッション値)

Phase 48: Tier 1+2 interaction features from track_conditions.parquet.
Raw values (dirt_moisture, turf_cushion) are merged in FeatureEngine.build_all().
Interaction features are computed here after HorseHistoryFeatures provides kyakusitukubun_cd.

Surface-aware design:
- dirt_moisture features are naturally NaN for turf races (dirt_moisture is NaN)
- turf_cushion features are naturally NaN for dirt races (turf_cushion is NaN)
- LightGBM handles NaN natively
"""

from __future__ import annotations

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


def compute_track_condition_features(
    df: pd.DataFrame,
    *,
    track_stats: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Compute track condition interaction features.

    Each feature is guarded by column existence checks and propagates NaN
    via .where() for numeric products. Surface-aware: dirt_moisture features
    are naturally NaN for turf races, turf_cushion features for dirt races.

    Args:
        df: Input DataFrame with track condition columns merged by FeatureEngine.
        track_stats: Training-period trackcd statistics for T1-02 features.
            If None, relative/zscore features are skipped.

    Returns:
        Copy of df with 8 new feature columns appended.
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

    return df
