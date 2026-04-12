"""horse_career_stats.py — Point-in-Time 累積キャリア統計の事前計算

entries.parquet (出走履歴) + races.parquet (レース条件) から、
各 (kettonum, race_id) ごとの「レース開催前時点」での累積成績を計算する。

これにより、horses.parquet (x_UMA) の ETL 時点累積値に含まれる
ルックアヘッドバイアス (未来のレース結果が特徴量に混入) を排除する。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# JRA 競走条件
_TURF_TRACKCD_RANGE = (10, 22)  # 芝 (trackcd 10-22)
_DIRT_TRACKCD_RANGE = (23, 29)  # ダート (trackcd 23-29)
_SHORT_DISTANCE_MAX = 1600  # 芝1600M以下 (x_UMA kyori1 の定義)


def _classify_surface(trackcd: pd.Series) -> pd.Series:
    """trackcd から surface を分類。"""
    trackcd_num = pd.to_numeric(trackcd, errors="coerce")
    return np.where(
        trackcd_num.between(*_TURF_TRACKCD_RANGE),
        "turf",
        np.where(trackcd_num.between(*_DIRT_TRACKCD_RANGE), "dirt", "other"),
    )


def _compute_cumulative_before(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
) -> pd.Series:
    """グループごとに「現在行より前」の累積和を計算。

    shift(1) で現在行を除外してから cumsum を取る。
    """
    return df.groupby(group_col)[value_col].transform(lambda x: x.shift(1).fillna(0).cumsum())


def precompute_career_stats(
    entries_df: pd.DataFrame,
    races_df: pd.DataFrame,
) -> pd.DataFrame:
    """Point-in-Time 累積キャリア統計を計算。

    Args:
        entries_df: 出走履歴 (race_id, kettonum, kakuteijyuni, honsyokin, race_date, jyocd)
        races_df: レース情報 (race_id, trackcd, kyori)

    Returns:
        DataFrame with columns:
            race_id, kettonum, race_date,
            cum_starts, cum_wins, cum_prize,
            cum_turf_starts, cum_turf_wins,
            cum_dirt_starts, cum_dirt_wins,
            cum_short_starts, cum_short_wins
    """
    # JRA レースのみ (jyocd 1-10)
    jyocd_num = pd.to_numeric(entries_df["jyocd"], errors="coerce")
    jra_mask = jyocd_num.between(1, 10)
    ent = entries_df[jra_mask].copy()

    if ent.empty:
        return pd.DataFrame()

    # レース条件をマージ
    race_info = races_df[["race_id", "trackcd", "kyori"]].copy()
    ent = ent.merge(race_info, on="race_id", how="left")

    # surface / short distance 判定
    ent["surface"] = _classify_surface(ent["trackcd"])
    ent["is_turf"] = (ent["surface"] == "turf").astype(int)
    ent["is_dirt"] = (ent["surface"] == "dirt").astype(int)
    ent["is_short"] = (
        (ent["surface"] == "turf")
        & (pd.to_numeric(ent["kyori"], errors="coerce") <= _SHORT_DISTANCE_MAX)
    ).astype(int)

    # 着順・賞金の数値化 (Nullable Int64 → float64 で NA を処理)
    ent["kakuteijyuni_int"] = pd.to_numeric(ent["kakuteijyuni"], errors="coerce")
    ent["is_win"] = (ent["kakuteijyuni_int"] == 1).astype("Int64").fillna(0).astype(int)
    ent["honsyokin_num"] = pd.to_numeric(ent["honsyokin"], errors="coerce").fillna(0)
    ent["is_turf_win"] = (ent["is_turf"] & ent["is_win"]).astype(int)
    ent["is_dirt_win"] = (ent["is_dirt"] & ent["is_win"]).astype(int)
    ent["is_short_win"] = (ent["is_short"] & ent["is_win"]).astype(int)

    # 馬ごとに日付順でソート
    ent = ent.sort_values(["kettonum", "race_date", "race_id"]).reset_index(drop=True)

    # 累積和 (現在行を除外 = shift(1) → cumsum)
    ent["one"] = 1
    ent["cum_starts"] = _compute_cumulative_before(ent, "kettonum", "one")
    ent["cum_wins"] = _compute_cumulative_before(ent, "kettonum", "is_win")
    ent["cum_prize"] = _compute_cumulative_before(ent, "kettonum", "honsyokin_num")
    ent["cum_turf_starts"] = _compute_cumulative_before(ent, "kettonum", "is_turf")
    ent["cum_turf_wins"] = _compute_cumulative_before(ent, "kettonum", "is_turf_win")
    ent["cum_dirt_starts"] = _compute_cumulative_before(ent, "kettonum", "is_dirt")
    ent["cum_dirt_wins"] = _compute_cumulative_before(ent, "kettonum", "is_dirt_win")
    ent["cum_short_starts"] = _compute_cumulative_before(ent, "kettonum", "is_short")
    ent["cum_short_wins"] = _compute_cumulative_before(ent, "kettonum", "is_short_win")

    result = ent[
        [
            "race_id",
            "kettonum",
            "race_date",
            "cum_starts",
            "cum_wins",
            "cum_prize",
            "cum_turf_starts",
            "cum_turf_wins",
            "cum_dirt_starts",
            "cum_dirt_wins",
            "cum_short_starts",
            "cum_short_wins",
        ]
    ].copy()

    logger.info(
        "Career stats: %d entries, %d horses",
        len(result),
        result["kettonum"].nunique(),
    )
    return result
