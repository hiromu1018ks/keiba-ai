"""カテゴリB: レース内相対特徴量

各馬のレース内での相対的な位置づけを表す特徴量を計算する。
"""

from __future__ import annotations

import pandas as pd


def compute_intra_race_features(df: pd.DataFrame) -> pd.DataFrame:
    """レース内相対特徴量を計算

    Args:
        df: race_id, umaban, odds, bataijyu を含むDataFrame

    Returns:
        weight_diff_from_mean, odds_rank 列が追加されたDataFrame
    """
    df = df.copy()

    if "bataijyu" in df.columns:
        weight_mean = df.groupby("race_id", observed=True)["bataijyu"].transform("mean")
        df["weight_diff_from_mean"] = df["bataijyu"] - weight_mean

    if "odds" in df.columns:
        df["odds_rank"] = (
            df.groupby("race_id", observed=True)["odds"]
            .rank(method="min", ascending=True)
            .astype("Int64")
        )

    return df
