"""カテゴリG: レース内相対比較特徴量

各馬のper-horse特徴量をgroupby("race_id")で相対化し、
z-score/rank/vs_meanの3種の変換を適用する。
"""

from __future__ import annotations

import pandas as pd

RELATIVE_FEATURE_COLS: list[str] = [
    "rel_norm_finish_zscore",    # z-score of norm_finish_logit_avg within race
    "rel_haron_vs_mean",         # (harontimel5_avg - race_mean) for late speed
    "rel_timediff_rank",         # rank of timediff_avg within race (ascending)
    "rel_blood_quality_rank",    # rank of blood_total_wr within race (descending)
    "rel_sire_quality_rank",     # rank of sire_wr within race (descending)
    "rel_weight_zscore",         # z-score of weight_zscore within race
    "rel_closing_index_rank",    # rank of closing_index_avg within race (ascending)
]

_BASE_FEATURES: list[dict[str, str]] = [
    {"base": "norm_finish_logit_avg", "output": "rel_norm_finish_zscore", "transform": "zscore"},
    {"base": "harontimel5_avg", "output": "rel_haron_vs_mean", "transform": "vs_mean"},
    {"base": "timediff_avg", "output": "rel_timediff_rank", "transform": "rank_asc"},
    {"base": "blood_total_wr", "output": "rel_blood_quality_rank", "transform": "rank_desc"},
    {"base": "sire_wr", "output": "rel_sire_quality_rank", "transform": "rank_desc"},
    {"base": "weight_zscore", "output": "rel_weight_zscore", "transform": "zscore"},
    {"base": "closing_index_avg", "output": "rel_closing_index_rank", "transform": "rank_asc"},
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
