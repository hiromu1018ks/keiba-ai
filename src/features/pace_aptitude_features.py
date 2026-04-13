"""ペース適性特徴量 — 角通過順位から推定"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _beta_smooth(wins: int, starts: int, alpha: int = 1, beta: int = 10) -> float:
    return (alpha + wins) / (alpha + beta + starts)


class PaceAptitudeFeatures:
    """過去走の jyuni1c/jyuni4c からペース適性を計算"""

    def compute(self, history: pd.DataFrame, target_date: str | pd.Timestamp) -> dict[str, float]:
        """1頭分のペース適性特徴量を計算

        Args:
            history: 過去出走DataFrame (race_date, kakuteijyuni, jyuni1c,
                jyuni4c, syussotosu 列が必要)
            target_date: 対象レース日付

        Returns:
            dict with keys: pace_aptitude, front_pace_wr, closing_pace_wr
        """
        result: dict[str, float] = {
            "pace_aptitude": np.nan,
            "front_pace_wr": np.nan,
            "closing_pace_wr": np.nan,
        }
        if history.empty:
            return result

        ts = pd.Timestamp(target_date)
        # *** PIT CRITICAL: 当日以前のみ使用 ***
        past = history[history["race_date"] < ts]
        if len(past) < 2:
            return result

        # 正規化着順と正規化1C通過位置
        norm_finish = past["kakuteijyuni"] / past["syussotosu"]
        norm_1c = past["jyuni1c"] / past["syussotosu"]

        front_mask = norm_1c <= 0.33  # 前ペース (1Cが上位1/3)
        closing_mask = norm_1c > 0.66  # 後ペース (1Cが下位1/3)

        # front pace での勝率 (Beta平滑化)
        front_races = past[front_mask]
        if len(front_races) > 0:
            result["front_pace_wr"] = _beta_smooth(
                int((front_races["kakuteijyuni"] == 1).sum()), len(front_races)
            )
        else:
            result["front_pace_wr"] = _beta_smooth(0, 0)

        # closing pace での勝率
        closing_races = past[closing_mask]
        if len(closing_races) > 0:
            result["closing_pace_wr"] = _beta_smooth(
                int((closing_races["kakuteijyuni"] == 1).sum()), len(closing_races)
            )
        else:
            result["closing_pace_wr"] = _beta_smooth(0, 0)

        # ペース適性スコア: closingでの正規化着順 - frontでの正規化着順
        # 正値 = 後ろ待ちの方が好成績, 負値 = 逃げ/先行の方が好成績
        front_avg = norm_finish[front_mask].mean() if front_mask.any() else np.nan
        closing_avg = norm_finish[closing_mask].mean() if closing_mask.any() else np.nan
        if pd.notna(front_avg) and pd.notna(closing_avg):
            result["pace_aptitude"] = float(closing_avg - front_avg)

        return result
