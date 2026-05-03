"""ODDS-01: モデル予測確率と市場オッズの乖離を特徴量化"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_odds_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    """odds_to_ability_ratioからレース内相対特徴量を計算。

    入力前提: odds_to_ability_ratio列が既に計算済み(training_pipeline)または
    WinTwoStageModel._prepare_features()で計算される(race_predictor)。

    Args:
        df: race_id, odds_to_ability_ratio列を含むDataFrame

    Returns:
        deviation_rank, deviation_zscore列が追加されたDataFrame
    """
    df = df.copy()

    ratio = df.get("odds_to_ability_ratio")
    if ratio is None:
        df["deviation_rank"] = pd.Series(np.nan, index=df.index, dtype=float)
        df["deviation_zscore"] = pd.Series(np.nan, index=df.index, dtype=float)
        return df

    ratio = pd.to_numeric(ratio, errors="coerce")

    # レース内ランク (descending: ratio大=過小評価=高いrank)
    df["deviation_rank"] = (
        ratio.groupby(df["race_id"])
        .rank(method="first", ascending=False)
        .astype("Float64")
    )

    # レース内z-score標準化
    race_mean = ratio.groupby(df["race_id"]).transform("mean")
    race_std = ratio.groupby(df["race_id"]).transform("std").replace(0, np.nan)
    df["deviation_zscore"] = ((ratio - race_mean) / race_std).clip(-5.0, 5.0)

    return df
