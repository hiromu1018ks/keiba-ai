"""Group C: 騎手コンテキスト特徴量 (Stage2のみ)

x_KISYU_SEISEKI から騎手年度別特徴量を生成。
SetYear < race_year の最新年を使用し、リークを防止。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from db.parquet_store import ParquetStore

FEATURE_COLS: list[str] = [
    "jockey_wr_overall",
    "jockey_wr_distance",
    "jockey_wr_venue",
    "jockey_prize_log",
]


class JockeyContextFeatures:
    """x_KISYU_SEISEKI から騎手年度別特徴量を生成。SetYear < race_year。"""

    def __init__(self, store: ParquetStore) -> None:
        self.store = store
        self._stats_cache: pd.DataFrame | None = None

    def _load_stats(self) -> pd.DataFrame:
        if self._stats_cache is None:
            from db.readers import load_jockey_stats

            self._stats_cache = load_jockey_stats(self.store)
        return self._stats_cache

    @staticmethod
    def _smoothed_wr(wins: int, total: int) -> float:
        """Beta(1,10) smoothing: (wins+1)/(total+11). total=0 -> NaN"""
        if total == 0:
            return float("nan")
        return (wins + 1) / (total + 11)

    def compute(self, entry_df: pd.DataFrame) -> pd.DataFrame:
        """entry_df (race_id, umaban, kisyucode, race_date) -> 騎手特徴量DataFrame

        SetYear < race_year の最新年を使用。
        """
        stats_df = self._load_stats()
        if stats_df.empty:
            return entry_df[["race_id", "umaban"]].assign(**{c: float("nan") for c in FEATURE_COLS})

        entry_df = entry_df.copy()
        entry_df["race_year"] = pd.to_datetime(entry_df["race_date"]).dt.year

        # kisyucode の型を統一 (entry: float64, stats: object の不一致を回避)
        entry_df["kisyucode"] = entry_df["kisyucode"].astype(str)
        stats_df = stats_df.copy()
        stats_df["kisyucode"] = stats_df["kisyucode"].astype(str)

        merged = entry_df[["race_id", "umaban", "kisyucode", "race_year"]].merge(
            stats_df, on="kisyucode", how="left"
        )
        merged = merged[merged["setyear"] < merged["race_year"]]
        if merged.empty:
            return entry_df[["race_id", "umaban"]].assign(**{c: float("nan") for c in FEATURE_COLS})

        latest = (
            merged.sort_values("setyear")
            .groupby(["race_id", "umaban", "kisyucode"], observed=True)
            .last()
            .reset_index()
        )

        # Vectorized feature computation
        heichi_cols = [f"heichichakukaisu{i}" for i in range(1, 7)]
        heichi_data = latest[heichi_cols].fillna(0).astype(float)
        wins = heichi_data["heichichakukaisu1"]
        total = heichi_data[heichi_cols].sum(axis=1)
        result = latest[["race_id", "umaban"]].copy()
        result["jockey_wr_overall"] = np.where(total == 0, np.nan, (wins + 1) / (total + 11))

        ky_cols = [f"kyori1chakukaisu{i}" for i in range(1, 7)]
        ky_data = latest[ky_cols].fillna(0).astype(float)
        ky1_w = ky_data["kyori1chakukaisu1"]
        ky1_t = ky_data[ky_cols].sum(axis=1)
        result["jockey_wr_distance"] = np.where(ky1_t == 0, np.nan, (ky1_w + 1) / (ky1_t + 11))

        j5_cols = [f"jyo5chakukaisu{i}" for i in range(1, 7)]
        j5_data = latest[j5_cols].fillna(0).astype(float)
        j5_w = j5_data["jyo5chakukaisu1"]
        j5_t = j5_data[j5_cols].sum(axis=1)
        result["jockey_wr_venue"] = np.where(j5_t == 0, np.nan, (j5_w + 1) / (j5_t + 11))

        prize = pd.to_numeric(latest["honsyokinheichi"], errors="coerce")
        result["jockey_prize_log"] = np.log1p(prize.fillna(0))

        return result[["race_id", "umaban"] + FEATURE_COLS]
