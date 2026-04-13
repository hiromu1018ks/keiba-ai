"""種牡馬産駒特徴量 — PIT安全な累積統計ベース"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _beta_smooth(wins: int, starts: int, alpha: int = 1, beta: int = 10) -> float:
    """Beta 平滑化勝率: (alpha + wins) / (alpha + beta + starts)"""
    return (alpha + wins) / (alpha + beta + starts)


class SireFeatures:
    """種牡馬産駒特徴量の計算 (PIT安全)"""

    def __init__(self, sire_stats_df: pd.DataFrame) -> None:
        self._stats = sire_stats_df
        if not self._stats.empty:
            self._stats = self._stats.sort_values(["sire_id", "race_date"])
            self._stats["_sire_date_key"] = (
                self._stats["sire_id"].astype(str) + "_" +
                self._stats["race_date"].astype(str)
            )

    def compute(
        self,
        sire_id: str | None,
        race_date: str | pd.Timestamp,
        surface: str,
        kyori: int,
    ) -> dict[str, float]:
        """1頭分の種牡馬特徴量を計算"""
        result: dict[str, float] = {}

        if sire_id is None or pd.isna(sire_id) or self._stats.empty:
            for col in ["sire_wr", "sire_place_rate", "sire_surface_wr",
                        "sire_distance_wr", "sire_prize_avg"]:
                result[col] = np.nan
            return result

        # searchsorted で該当日以前の最新行を取得
        mask = self._stats["sire_id"] == sire_id
        subset = self._stats[mask]
        if subset.empty:
            for col in ["sire_wr", "sire_place_rate", "sire_surface_wr",
                        "sire_distance_wr", "sire_prize_avg"]:
                result[col] = np.nan
            return result

        ts = pd.Timestamp(race_date)
        idx = subset["race_date"].searchsorted(ts, side="right") - 1
        if idx < 0:
            # 当日以前のデータなし → Beta(1,10) 事前分布
            prior = _beta_smooth(0, 0)
            result["sire_wr"] = prior
            result["sire_place_rate"] = prior
            result["sire_surface_wr"] = prior
            result["sire_distance_wr"] = prior
            result["sire_prize_avg"] = 0.0
            return result

        row = subset.iloc[idx]

        # 全体勝率
        result["sire_wr"] = _beta_smooth(int(row.get("sire_wins", 0)),
                                          int(row.get("sire_starts", 0)))
        # 複勝率
        result["sire_place_rate"] = _beta_smooth(int(row.get("sire_places", 0)),
                                                   int(row.get("sire_starts", 0)))
        # サーフェス別勝率
        if surface == "turf":
            result["sire_surface_wr"] = _beta_smooth(
                int(row.get("sire_turf_wins", 0)),
                int(row.get("sire_turf_starts", 0)))
        else:
            result["sire_surface_wr"] = _beta_smooth(
                int(row.get("sire_dirt_wins", 0)),
                int(row.get("sire_dirt_starts", 0)))

        # 距離別勝率
        if kyori <= 1600:
            result["sire_distance_wr"] = _beta_smooth(
                int(row.get("sire_short_wins", 0)),
                int(row.get("sire_short_starts", 0)))
        else:
            result["sire_distance_wr"] = _beta_smooth(
                int(row.get("sire_long_wins", 0)),
                int(row.get("sire_long_starts", 0)))

        # 平均賞金
        starts = max(1, int(row.get("sire_starts", 0)))
        result["sire_prize_avg"] = float(np.log1p(row.get("sire_prize_total", 0) / starts))

        return result
