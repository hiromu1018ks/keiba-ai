"""EV Tail Calibration — feature family合意度による高EV候補スケーリング

EV >= 1.5の高EV長穴候補を5つのfeature family合意度で評価し、
単一family跳ねを縮小、複数family合意を拡大する。
"""

from __future__ import annotations

import math

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ZSCORE_THRESHOLD: float = 1.0
SINGLE_FAMILY_FACTOR: float = 0.85
MULTI_FAMILY_FACTOR: float = 1.05
NO_FAMILY_FACTOR: float = 0.70
EV_THRESHOLD: float = 1.5

# Feature family → column names
FAMILY_FEATURES: dict[str, list[str]] = {
    "trf": [
        "form_trend_race_rank",
        "blood_total_wr_race_rank",
        "blood_surface_wr_race_rank",
    ],
    "int": [
        "grade_x_form_trend",
        "distance_x_closing_index",
        "grade_x_blood_prize_log",
    ],
    "hlf": [
        "closing_speed_ratio_avg",
        "harontime_last3f_avg",
        "haron_race_gap_avg",
        "pace_ratio_avg",
    ],
    "market": [
        "implied_prob_hhi",
        "odds_skewness",
        "overround",
        "market_entropy",
    ],
    "ability": [
        "p_win_pred",
    ],
}


class EVTtailCalibrator:
    """Feature family合意度で高EV候補のEVをスケーリングする。"""

    def calibrate(
        self,
        horse_row: pd.Series,
        race_df: pd.DataFrame,
        ev_value: float,
    ) -> float:
        """高EV候補のEVをfeature family合意度でスケーリングする。

        Args:
            horse_row: 対象馬の特徴量 (pd.Series)
            race_df: レース全馬の特徴量 (pd.DataFrame)
            ev_value: 現在のEV値

        Returns:
            スケーリング後のEV値。EV < 1.5の場合はそのまま返す。
        """
        if ev_value < EV_THRESHOLD:
            return ev_value

        agreeing_families = 0

        for family_cols in FAMILY_FEATURES.values():
            # レースdfにある列のみ対象
            available = [c for c in family_cols if c in race_df.columns]
            if not available:
                continue

            family_agrees = False
            for col in available:
                if col not in horse_row.index:
                    continue
                horse_val = horse_row[col]
                if pd.isna(horse_val):
                    continue

                race_vals = pd.to_numeric(race_df[col], errors="coerce")
                race_mean = float(race_vals.mean())
                race_std = float(race_vals.std())

                if race_std == 0.0 or math.isnan(race_std):
                    # 全値同一 → z=0 → 同意なし
                    continue

                z_score = (float(horse_val) - race_mean) / race_std
                if z_score > ZSCORE_THRESHOLD:
                    family_agrees = True
                    break  # 1つでも > threshold ならfamily同意

            if family_agrees:
                agreeing_families += 1

        # スケーリング係数の選択
        if agreeing_families == 0:
            factor = NO_FAMILY_FACTOR
        elif agreeing_families == 1:
            factor = SINGLE_FAMILY_FACTOR
        else:
            factor = MULTI_FAMILY_FACTOR

        return ev_value * factor
