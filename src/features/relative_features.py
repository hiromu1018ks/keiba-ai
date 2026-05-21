"""カテゴリG: レース内相対比較特徴量

各馬のper-horse特徴量をgroupby("race_id")で相対化し、
z-score/rank/vs_meanの3種の変換を適用する。
Stage1 OOF前に計算可能な特徴量 (compute_relative_features) と
Stage1 OOF後の特徴量 (compute_stage2_relative_features) の2段階構成。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RELATIVE_FEATURE_COLS: list[str] = [
    "rel_norm_finish_zscore",    # z-score of norm_finish_logit_avg within race
    "rel_haron_vs_mean",         # (harontimel5_avg - race_mean) for late speed
    "rel_timediff_rank",         # rank of timediff_avg within race (ascending)
    "rel_blood_quality_rank",    # rank of blood_total_wr within race (descending)
    "rel_sire_quality_rank",     # rank of sire_wr within race (descending)
    "rel_bms_quality_rank",      # rank of bms_wr within race (descending)
    "rel_bms_surface_quality_rank",
    "rel_weight_zscore",         # z-score of weight_zscore within race
    "rel_closing_index_rank",    # rank of closing_index_avg within race (ascending)
    "rel_fuku_odds_zscore",      # z-score of fukuoddslow within race
    "rel_popularity_rank_zscore",  # z-score of popularity_rank within race
]

STAGE2_RELATIVE_FEATURE_COLS: list[str] = [
    "rel_p_ability_win_zscore",    # z-score of p_ability_win within race
    "rel_p_ability_win_rank",      # rank of p_ability_win within race (descending)
    "rel_odds_ability_deviation",  # z-score of odds_to_ability_ratio within race
]

_BASE_FEATURES: list[dict[str, str]] = [
    {"base": "norm_finish_logit_avg", "output": "rel_norm_finish_zscore", "transform": "zscore"},
    {"base": "harontimel5_avg", "output": "rel_haron_vs_mean", "transform": "vs_mean"},
    {"base": "timediff_avg", "output": "rel_timediff_rank", "transform": "rank_asc"},
    {"base": "blood_total_wr", "output": "rel_blood_quality_rank", "transform": "rank_desc"},
    {"base": "sire_wr", "output": "rel_sire_quality_rank", "transform": "rank_desc"},
    {"base": "bms_wr", "output": "rel_bms_quality_rank", "transform": "rank_desc"},
    {"base": "bms_surface_wr", "output": "rel_bms_surface_quality_rank", "transform": "rank_desc"},
    {"base": "weight_zscore", "output": "rel_weight_zscore", "transform": "zscore"},
    {"base": "closing_index_avg", "output": "rel_closing_index_rank", "transform": "rank_asc"},
    {"base": "fukuoddslow", "output": "rel_fuku_odds_zscore", "transform": "zscore"},
    {"base": "popularity_rank", "output": "rel_popularity_rank_zscore", "transform": "zscore"},
]


def compute_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """レース内相対比較特徴量を計算。

    各base特徴量をgroupby("race_id")で相対化し、z-score/vs_mean/rank変換を適用。
    base特徴量が存在しない場合はスキップ (エラーなし)。
    NaNはそのまま伝播し、group統計には含まれない。

    Args:
        df: race_id, および各base特徴量列を含むDataFrame

    Returns:
        相対特徴量列が追加されたDataFrame
    """
    df = df.copy()
    for spec in _BASE_FEATURES:
        base = spec["base"]
        output = spec["output"]
        transform = spec["transform"]
        if base not in df.columns:
            continue
        grp = df.groupby("race_id", observed=True)[base]
        if transform == "zscore":
            mean = grp.transform("mean")
            std = grp.transform("std").fillna(0)
            std = std.replace(0, 1)  # fallback: std=0 -> zscore=0
            df[output] = (df[base] - mean) / std
        elif transform == "vs_mean":
            mean = grp.transform("mean")
            df[output] = df[base] - mean
        elif transform == "rank_asc":
            df[output] = grp.rank(method="min", ascending=True, na_option="keep")
        elif transform == "rank_desc":
            df[output] = grp.rank(method="min", ascending=False, na_option="keep")
    return df


def compute_stage2_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stage1 OOF後のレース内相対特徴量を計算。

    p_ability_win (Stage1出力) と odds_to_ability_ratio (市場/能力比) から
    レース内相対特徴量を生成する。
    これらの特徴量はStage1 OOF予測に依存するため、
    compute_relative_features() とは別関数で呼び出す。

    base列が存在しない場合はNaN列を生成 (エラーなし)。

    Args:
        df: race_id, p_ability_win, odds_to_ability_ratio列を含むDataFrame

    Returns:
        Stage2相対特徴量列が追加されたDataFrame
    """
    df = df.copy()

    # p_ability_win: z-score + rank (descending)
    if "p_ability_win" in df.columns:
        grp = df.groupby("race_id", observed=True)["p_ability_win"]
        mean = grp.transform("mean")
        std = grp.transform("std").fillna(0)
        std = std.replace(0, 1)
        df["rel_p_ability_win_zscore"] = (df["p_ability_win"] - mean) / std
        df["rel_p_ability_win_rank"] = grp.rank(
            method="min", ascending=False, na_option="keep"
        )
    else:
        df["rel_p_ability_win_zscore"] = np.nan
        df["rel_p_ability_win_rank"] = np.nan

    # odds_to_ability_ratio: z-score
    if "odds_to_ability_ratio" in df.columns:
        grp = df.groupby("race_id", observed=True)["odds_to_ability_ratio"]
        mean = grp.transform("mean")
        std = grp.transform("std").fillna(0)
        std = std.replace(0, 1)
        df["rel_odds_ability_deviation"] = (df["odds_to_ability_ratio"] - mean) / std
    else:
        df["rel_odds_ability_deviation"] = np.nan

    return df
