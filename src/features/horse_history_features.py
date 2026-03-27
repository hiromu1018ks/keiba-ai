"""horse_history_features.py — 馬の過去成績ベース特徴量

主な特徴量:
  - norm_finish_logit_avg: 着順をログット変換したスコアの平均
  - jockey_surprise: Beta事前分布でスムージングした騎手勝率サプライズ
  - haron_time_zscore_avg: 階層fallback付きハロンタイムz-score平均
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAYOUT_RATE: float = 0.80  # JRA控除率20%
CLIP_LO: float = 0.05
CLIP_HI: float = 0.95

# Beta prior parameters for jockey surprise smoothing
ALPHA_PRIOR: float = 1.0
BETA_PRIOR: float = 20.0

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


def _compute_jockey_surprise(
    actual_wins: int,
    n_races: int,
    expected_wins: float,  # noqa: ARG001 — signature matches spec; used by caller
) -> float:
    """Beta事前分布でスムージングした騎手勝率のサプライズ値を返す。

    n_races < 30 の場合は NaN を返す。
    """
    if n_races < 30:
        return float("nan")

    alpha_post = ALPHA_PRIOR + actual_wins
    beta_post = BETA_PRIOR + n_races - actual_wins

    smoothed_wr = alpha_post / (alpha_post + beta_post)
    baseline_wr = ALPHA_PRIOR / (ALPHA_PRIOR + BETA_PRIOR)

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
    fallback = global_stats.get(("all",))
    if fallback is None:
        return float("nan"), float("nan")
    return float(fallback["mean"]), float(fallback["std"])


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

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def compute(
        self,
        race_df: pd.DataFrame,
        entry_df: pd.DataFrame,
        target_race_ids: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """過去成績特徴量を計算"""
        if target_race_ids is not None:
            entry_df = entry_df[entry_df["race_id"].isin(target_race_ids)]

        # 対象レースの馬・騎手リスト
        horses = entry_df[["race_id", "umaban", "ketto_num", "kisyu_code"]].copy()
        if "race_date" not in horses.columns:
            date_map = race_df.set_index("race_id")["race_date"]
            horses["race_date"] = horses["race_id"].map(date_map)

        unique_ketto = horses["ketto_num"].unique().tolist()
        unique_kisyu = horses["kisyu_code"].unique().tolist()

        if not unique_ketto:
            return pd.DataFrame(columns=["race_id", "umaban"] + self.BASE_COLS)

        # SQL: 過去レースデータを一括取得
        sql = text("""
            SELECT
                ur.race_id AS past_race_id,
                r.year, r.month_day,
                ur.ketto_num, ur.kisyu_code, ur.umaban,
                ur.kakutei_jyuni AS finish_pos,
                r.torosu AS field_size,
                ur.tansyo_odds AS win_odds,
                ur.harontimelong3 AS haron_time_l3,
                CASE WHEN r.torosu >= 8 THEN 1 ELSE 0 END AS valid_field
            FROM n_uma_race ur
            JOIN n_race r ON ur.year = r.year
                AND ur.monthday = r.month_day
                AND ur.jyocd = r.jyocd
                AND ur.kaiji = r.kaiji
                AND ur.nichiji = r.nichiji
                AND ur.racenum = r.racenum
            WHERE ur.ketto_num IN :ketto_nums
               OR ur.kisyu_code IN :kisyu_codes
            ORDER BY r.year, r.month_day
        """)

        past_df = pd.read_sql(
            sql,
            self.engine,
            params={
                "ketto_nums": tuple(unique_ketto),
                "kisyu_codes": tuple(unique_kisyu),
            },
        )

        if past_df.empty:
            return pd.DataFrame(columns=["race_id", "umaban"] + self.BASE_COLS)

        # race_date 生成
        past_df["race_date"] = pd.to_datetime(
            past_df["year"].astype(str) + past_df["month_day"].astype(str),
            format="%Y%m%d",
        )

        # 馬ごとに特徴量計算
        results: list[dict] = []
        for _, row in horses.iterrows():
            race_date = row["race_date"]
            ketto = row["ketto_num"]
            kisyu = row["kisyu_code"]

            # norm_finish_logit_avg: 同じ馬の過去3走
            horse_past = past_df[
                (past_df["ketto_num"] == ketto)
                & (past_df["race_date"] < race_date)
                & (past_df["valid_field"] == 1)
                & (past_df["finish_pos"] > 0)
            ].tail(3)

            if len(horse_past) > 0:
                logits = horse_past.apply(
                    lambda r: _norm_finish_logit(r["finish_pos"], r["field_size"]),
                    axis=1,
                )
                norm_finish_logit_avg: float = logits.mean()
            else:
                norm_finish_logit_avg = float("nan")

            # jockey_surprise: 騎手の過去100戦
            jockey_past = past_df[
                (past_df["kisyu_code"] == kisyu)
                & (past_df["race_date"] < race_date)
                & (past_df["finish_pos"] > 0)
                & (past_df["win_odds"] > 0)
            ].tail(100)

            if len(jockey_past) >= 30:
                expected = (PAYOUT_RATE / jockey_past["win_odds"].clip(lower=1.1)).sum()
                actual = int((jockey_past["finish_pos"] == 1).sum())
                jockey_surprise: float = _compute_jockey_surprise(
                    actual, len(jockey_past), expected
                )
            else:
                jockey_surprise = float("nan")

            # haron_time_zscore_avg: 過去3走 (Phase 1: simplified, uses nan for now)
            haron_time_zscore_avg: float = float("nan")

            results.append(
                {
                    "race_id": row["race_id"],
                    "umaban": row["umaban"],
                    "norm_finish_logit_avg": norm_finish_logit_avg,
                    "jockey_surprise": jockey_surprise,
                    "haron_time_zscore_avg": haron_time_zscore_avg,
                }
            )

        return pd.DataFrame(results)

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
