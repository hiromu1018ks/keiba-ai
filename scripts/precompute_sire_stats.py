#!/usr/bin/env python3
"""種牡馬産駒累積統計の事前計算スクリプト。

PIT保証: shift(1).fillna(0).cumsum() により当日の結果を含まない。
         horse_career_stats.py の _compute_cumulative_before パターンに統一。
出力: data/raw/sire_career_stats.parquet
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_TURF_TRACKCD_RANGE = (10, 22)
_SHORT_DISTANCE_MAX = 1600


def _cumulative_before(series: pd.Series, group: pd.Series) -> pd.Series:
    """PIT安全な累積: shift(1)->fillna(0)->cumsum (horse_career_stats.py と同じ)."""
    return series.groupby(group).transform(lambda x: x.shift(1).fillna(0).cumsum())


def compute_sire_stats(
    entries_df: pd.DataFrame,
    horses_df: pd.DataFrame,
) -> pd.DataFrame:
    """種牡馬ごとの日次産駒成績を累積計算する。

    Args:
        entries_df: entries.parquet (kettonum, race_date, race_id, kakuteijyuni, ...)
        horses_df: horses.parquet (kettonum, ketto3infohansyokunum1, ...)

    Returns:
        sire_career_stats: (sire_id, race_date) ごとの累積統計
    """
    # entries -> horses -> sire_id を結合
    sire_map = horses_df.set_index("kettonum")["ketto3infohansyokunum1"]
    ent = entries_df.copy()
    ent["sire_id"] = ent["kettonum"].map(sire_map)

    # フラグ列 (horse_career_stats.py と同じパターン)
    trackcd_num = pd.to_numeric(ent["trackcd"], errors="coerce")
    ent["is_turf"] = trackcd_num.between(*_TURF_TRACKCD_RANGE).fillna(False).astype(int)
    ent["is_dirt"] = (~trackcd_num.between(*_TURF_TRACKCD_RANGE)).astype(int)
    kyori_num = pd.to_numeric(ent["kyori"], errors="coerce")
    ent["is_short"] = (kyori_num <= _SHORT_DISTANCE_MAX).fillna(False).astype(int)
    ent["is_long"] = (kyori_num > _SHORT_DISTANCE_MAX).fillna(False).astype(int)

    jyuni_num = pd.to_numeric(ent["kakuteijyuni"], errors="coerce")
    ent["is_win"] = (jyuni_num == 1).astype(int)
    ent["is_place"] = jyuni_num.between(1, 3).astype(int)

    # 複合フラグ (lambda なし — 外部DataFrame参照バグを回避)
    ent["is_turf_win"] = ent["is_turf"] * ent["is_win"]
    ent["is_dirt_win"] = ent["is_dirt"] * ent["is_win"]
    ent["is_short_win"] = ent["is_short"] * ent["is_win"]
    ent["is_long_win"] = ent["is_long"] * ent["is_win"]

    # 賞金の数値化
    ent["honsyokin_num"] = pd.to_numeric(ent["honsyokin"], errors="coerce").fillna(0)

    # (sire_id, race_date) で日次集計
    daily = (
        ent.groupby(["sire_id", "race_date"])
        .agg(
            daily_starts=("kakuteijyuni", "count"),
            daily_wins=("is_win", "sum"),
            daily_places=("is_place", "sum"),
            daily_turf_starts=("is_turf", "sum"),
            daily_turf_wins=("is_turf_win", "sum"),
            daily_dirt_starts=("is_dirt", "sum"),
            daily_dirt_wins=("is_dirt_win", "sum"),
            daily_short_starts=("is_short", "sum"),
            daily_short_wins=("is_short_win", "sum"),
            daily_long_starts=("is_long", "sum"),
            daily_long_wins=("is_long_win", "sum"),
            daily_prize_total=("honsyokin_num", "sum"),
        )
        .reset_index()
    )

    # 日付順でソート (cumsum の正確性のため)
    daily = daily.sort_values(["sire_id", "race_date"]).reset_index(drop=True)

    # PIT安全な累積: shift(1).fillna(0).cumsum() (horse_career_stats.py と統一)
    cum_cols = [c for c in daily.columns if c.startswith("daily_")]
    for col in cum_cols:
        prefix = col.replace("daily_", "sire_")
        daily[prefix] = _cumulative_before(daily[col], daily["sire_id"])

    # 不要な daily_* 列を削除
    result = daily.drop(columns=cum_cols)

    # NaN の sire_id を除外
    result = result.dropna(subset=["sire_id"])

    logger.info(
        "Sire stats: %d rows, %d sires",
        len(result),
        result["sire_id"].nunique(),
    )
    return result


def main() -> None:
    from db.parquet_store import ParquetStore

    store = ParquetStore()

    logger.info("Loading entries.parquet...")
    t0 = time.time()
    entries = store.read("raw", "entries")
    logger.info("  %d rows (%.1fs)", len(entries), time.time() - t0)

    logger.info("Loading horses.parquet...")
    horses = store.read("raw", "horses")

    logger.info("Loading races.parquet...")
    races = store.read("raw", "races")

    # entries に race 条件をマージ
    race_info = races[["race_id", "trackcd", "kyori"]].copy()
    entries = entries.merge(race_info, on="race_id", how="left")

    logger.info("Computing sire stats...")
    t0 = time.time()
    stats = compute_sire_stats(entries, horses)
    logger.info("  %d rows (%.1fs)", len(stats), time.time() - t0)

    out_path = _PROJECT_ROOT / "data" / "raw" / "sire_career_stats.parquet"
    stats.to_parquet(out_path, index=False)
    logger.info("Saved %d rows to %s", len(stats), out_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
