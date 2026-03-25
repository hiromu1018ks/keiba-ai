"""カテゴリB: レース内相対特徴量

各馬のレース内での相対的な位置づけを表す特徴量を計算する。
"""

from __future__ import annotations

import pandas as pd


def compute_intra_race_features(df: pd.DataFrame) -> pd.DataFrame:
    """レース内相対特徴量を計算

    Args:
        df: race_id, umaban, win_odds, ba_taijyu を含むDataFrame

    Returns:
        weight_diff_from_mean, odds_rank 列が追加されたDataFrame
    """
    df = df.copy()

    if "ba_taijyu" in df.columns:
        weight_mean = df.groupby("race_id")["ba_taijyu"].transform("mean")
        df["weight_diff_from_mean"] = df["ba_taijyu"] - weight_mean

    if "win_odds" in df.columns:
        df["odds_rank"] = df.groupby("race_id")["win_odds"].rank(
            method="min", ascending=True
        ).astype(int)

    return df
