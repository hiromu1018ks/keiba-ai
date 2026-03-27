"""horse_history_features.py — 馬の過去成績ベース特徴量

主な特徴量:
  - norm_finish_logit_avg: 着順をログット変換したスコアの平均
  - jockey_surprise: Beta事前分布でスムージングした騎手勝率サプライズ
  - haron_time_zscore_avg: 階層fallback付きハロンタイムz-score平均
"""

from __future__ import annotations

import math

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAYOUT_RATE: float = 0.80  # JRA控除率20%
CLIP_LO: float = 0.05
CLIP_HI: float = 0.95

# ---------------------------------------------------------------------------
# Helper: normalised finish-logit
# ---------------------------------------------------------------------------


def _norm_finish_logit(finish_pos: int, field_size: int) -> float:
    """着順をフィールドサイズで正規化し、クリップ→logit 変換する。

    score = 1 - (finish_pos - 1) / (field_size - 1)  → [0, 1]
    field_size < 8 の場合は NaN を返す。
    """
    if field_size < 8:
        return float("nan")

    score = 1.0 - (finish_pos - 1) / (field_size - 1)
    score = max(CLIP_LO, min(CLIP_HI, score))
    return math.log(score / (1.0 - score))


# ---------------------------------------------------------------------------
# Helper: jockey surprise
# ---------------------------------------------------------------------------


def _compute_jockey_surprise(actual_wins: int, n_races: int, expected_wins: float) -> float:
    """Beta事前分布でスムージングした騎手勝率のサプライズ値を返す。

    n_races < 30 の場合は NaN を返す。
    """
    if n_races < 30:
        return float("nan")

    alpha_prior: float = 1.0
    beta_prior: float = 20.0
    alpha_post = alpha_prior + actual_wins
    beta_post = beta_prior + n_races - actual_wins

    smoothed_wr = alpha_post / (alpha_post + beta_post)
    baseline_wr = alpha_prior / (alpha_prior + beta_prior)

    return smoothed_wr - baseline_wr


# ---------------------------------------------------------------------------
# Helper: hierarchical fallback for haron-time z-score
# ---------------------------------------------------------------------------

FALLBACK_LEVELS: list[tuple[list[str], int]] = [
    (["distance_bin", "surface", "baba_cd"], 50),  # L1: full condition, min 50
    (["distance_bin", "surface"], 30),  # L2: distance + surface, min 30
    (["distance_bin"], 20),  # L3: distance only, min 20
    ([], 0),  # L4: global fallback
]


def _get_group_stats(
    distance_bin: str,
    surface: str,
    baba_cd: str,
    global_stats: dict[tuple, dict],
) -> tuple[float, float]:
    """階層fallbackで (mean, std) を返す。

    FALLBACK_LEVELS を上から順に試し、global_stats に key が存在し
    n >= min_n を満たす最初のレベルの統計量を返す。
    最終的に ("all",) キーの統計量にfallbackする。
    """
    values: dict[str, str] = {
        "distance_bin": distance_bin,
        "surface": surface,
        "baba_cd": baba_cd,
    }

    for cols, min_n in FALLBACK_LEVELS:
        key = tuple(values[c] for c in cols) if cols else ("all",)
        if key in global_stats and global_stats[key].get("n", 0) >= min_n:
            stats = global_stats[key]
            return float(stats["mean"]), float(stats["std"])

    # final fallback
    stats = global_stats[("all",)]
    return float(stats["mean"]), float(stats["std"])


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class HorseHistoryFeatures:
    """馬の過去成績特徴量を計算・管理するクラス。"""

    BASE_COLS: list[str] = [
        "norm_finish_logit_avg",
        "jockey_surprise",
        "haron_time_zscore_avg",
    ]

    def __init__(self, engine: object) -> None:
        self.engine = engine

    def compute(
        self,
        race_df: pd.DataFrame,
        entry_df: pd.DataFrame,
        target_race_ids: object = None,
    ) -> pd.DataFrame:
        """特徴量を計算してDataFrameを返す (stub)。"""
        return pd.DataFrame(columns=["race_id", "umaban"] + self.BASE_COLS)

    @staticmethod
    def add_race_transforms(df: pd.DataFrame) -> pd.DataFrame:
        """BASE_COLS の各列についてレース内 z-score と rank percentile を追加する。"""
        df = df.copy()
        for col in HorseHistoryFeatures.BASE_COLS:
            if col not in df.columns:
                continue
            race_mean = df.groupby("race_id")[col].transform("mean")
            race_std = df.groupby("race_id")[col].transform("std")
            race_std = race_std.clip(lower=1e-6).fillna(1e-6)
            df[f"{col}_race_z"] = (df[col] - race_mean) / race_std
            df[f"{col}_race_pct"] = df.groupby("race_id")[col].rank(pct=True)
        return df
