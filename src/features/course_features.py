"""コース別適性特徴量 — 競馬場×距離帯の過去勝率"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _beta_smooth(wins: int, starts: int, alpha: int = 1, beta: int = 10) -> float:
    return (alpha + wins) / (alpha + beta + starts)


class CourseFeatures:
    """競馬場別・距離帯別の過去勝率を計算"""

    def compute(
        self,
        history: pd.DataFrame,
        jyocd: str,
        distance_bin: str,
        target_date: str | pd.Timestamp,
    ) -> dict[str, float]:
        """1頭分のコース適性特徴量を計算

        Args:
            history: 過去出走DataFrame
                (race_date, jyocd, kakuteijyuni, distance_bin, syussotosu 列が必要)
            jyocd: 競馬場コード (e.g., "01" = 京都, "02" = 中山, etc.)
            distance_bin: 距離帯 ("sprint", "mile", "intermediate", "long")
            target_date: 対象レース日付

        Returns:
            dict with keys: course_wr, course_distance_wr
        """
        result: dict[str, float] = {"course_wr": np.nan, "course_distance_wr": np.nan}

        if history.empty:
            return result

        ts = pd.Timestamp(target_date)
        # *** PIT CRITICAL: 当日以前のみ使用 ***
        past = history[history["race_date"] < ts]
        if past.empty:
            return result

        # 競馬場別勝率
        venue_races = past[past["jyocd"] == jyocd]
        if len(venue_races) > 0:
            wins = int((venue_races["kakuteijyuni"] == 1).sum())
            result["course_wr"] = _beta_smooth(wins, len(venue_races))
        else:
            result["course_wr"] = _beta_smooth(0, 0)

        # 競馬場×距離帯別勝率
        vd_races = venue_races[venue_races["distance_bin"] == distance_bin]
        if len(vd_races) > 0:
            wins = int((vd_races["kakuteijyuni"] == 1).sum())
            result["course_distance_wr"] = _beta_smooth(wins, len(vd_races))
        else:
            result["course_distance_wr"] = _beta_smooth(0, 0)

        return result
