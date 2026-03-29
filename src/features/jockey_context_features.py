"""Group C: 騎手コンテキスト特徴量 (Stage2のみ)

x_KISYU_SEISEKI から騎手年度別特徴量を生成。
SetYear < race_year の最新年を使用し、リークを防止。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from db.repository import DataRepository

FEATURE_COLS: list[str] = [
    "jockey_wr_overall",
    "jockey_wr_distance",
    "jockey_wr_venue",
    "jockey_prize_log",
]


class JockeyContextFeatures:
    """x_KISYU_SEISEKI から騎手年度別特徴量を生成。SetYear < race_year。"""

    def __init__(self, repo: DataRepository) -> None:
        self.repo = repo
        self._stats_cache: pd.DataFrame | None = None

    def _load_stats(self) -> pd.DataFrame:
        if self._stats_cache is None:
            self._stats_cache = self.repo.load_jockey_stats()
        return self._stats_cache

    @staticmethod
    def _smoothed_wr(wins: int, total: int) -> float:
        """Beta(1,10) smoothing: (wins+1)/(total+11). total=0 -> NaN"""
        if total == 0:
            return float("nan")
        return (wins + 1) / (total + 11)

    def compute(self, entry_df: pd.DataFrame) -> pd.DataFrame:
        """entry_df (race_id, umaban, kisyu_code, race_date) -> 騎手特徴量DataFrame

        SetYear < race_year の最新年を使用。
        """
        stats_df = self._load_stats()
        if stats_df.empty:
            return entry_df[["race_id", "umaban"]].assign(
                **{c: float("nan") for c in FEATURE_COLS}
            )

        entry_df = entry_df.copy()
        entry_df["race_year"] = pd.to_datetime(entry_df["race_date"]).dt.year

        merged = entry_df[["race_id", "umaban", "kisyu_code", "race_year"]].merge(
            stats_df, left_on="kisyu_code", right_on="kisyucode", how="left"
        )
        merged = merged[merged["setyear"] < merged["race_year"]]
        if merged.empty:
            return entry_df[["race_id", "umaban"]].assign(
                **{c: float("nan") for c in FEATURE_COLS}
            )

        latest = (
            merged.sort_values("setyear")
            .groupby(["race_id", "umaban", "kisyu_code"])
            .last()
            .reset_index()
        )

        rows: list[dict] = []
        for _, r in latest.iterrows():
            feats: dict = {}

            # Overall win rate
            wins = int(r.get("heichichakukaisu1", 0) or 0)
            total = sum(int(r.get(f"heichichakukaisu{i}", 0) or 0) for i in range(1, 7))
            feats["jockey_wr_overall"] = self._smoothed_wr(wins, total)

            # Distance win rate (short = kyori1)
            ky1_w = int(r.get("kyori1chakukaisu1", 0) or 0)
            ky1_t = sum(int(r.get(f"kyori1chakukaisu{i}", 0) or 0) for i in range(1, 7))
            feats["jockey_wr_distance"] = self._smoothed_wr(ky1_w, ky1_t)

            # Venue win rate (Tokyo = jyo5)
            j5_w = int(r.get("jyo5chakukaisu1", 0) or 0)
            j5_t = sum(int(r.get(f"jyo5chakukaisu{i}", 0) or 0) for i in range(1, 7))
            feats["jockey_wr_venue"] = self._smoothed_wr(j5_w, j5_t)

            # Prize log
            prize = r.get("honsyokinheichi")
            feats["jockey_prize_log"] = float(np.log1p(float(prize or 0)))

            feats["race_id"] = r["race_id"]
            feats["umaban"] = r["umaban"]
            rows.append(feats)

        result = pd.DataFrame(rows)
        if result.empty:
            return entry_df[["race_id", "umaban"]].assign(
                **{c: float("nan") for c in FEATURE_COLS}
            )
        return result[["race_id", "umaban"] + FEATURE_COLS]
