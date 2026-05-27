"""Walk-forward race-level split utility.

Shared by MarketAwareWinCalibrator, WinBenterGate, and TrainingPipelineV5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def walk_forward_race_splits(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    min_train_races: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """race_id単位のexpanding walk-forward splitを返す。

    Chronological WF splits on race_id/race_date that never split a race
    across train/validation boundaries.
    """
    if df.empty or "race_id" not in df.columns:
        return []
    race_cols = ["race_id"]
    if "race_date" in df.columns:
        race_cols.append("race_date")
    race_order = df[race_cols].drop_duplicates()
    if "race_date" in race_order.columns:
        race_order = race_order.sort_values(["race_date", "race_id"])
    else:
        race_order = race_order.sort_values("race_id")
    race_order = race_order.reset_index(drop=True)
    n_races = len(race_order)
    if n_races <= min_train_races + 1:
        return []

    effective_splits = min(n_splits, max(2, n_races // max(1, min_train_races)))
    if n_races <= effective_splits:
        return []

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    splitter = TimeSeriesSplit(n_splits=effective_splits)
    for train_race_idx, val_race_idx in splitter.split(race_order):
        if len(train_race_idx) < min_train_races or len(val_race_idx) == 0:
            continue
        train_races = set(race_order.iloc[train_race_idx]["race_id"])
        val_races = set(race_order.iloc[val_race_idx]["race_id"])
        train_idx = df.index[df["race_id"].isin(train_races)].to_numpy()
        val_idx = df.index[df["race_id"].isin(val_races)].to_numpy()
        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))
    return splits
