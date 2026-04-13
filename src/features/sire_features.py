"""種牡馬産駒特徴量 — PIT安全な累積統計ベース"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _beta_smooth(wins: int, starts: int, alpha: int = 1, beta: int = 10) -> float:
    """Beta 平滑化勝率: (alpha + wins) / (alpha + beta + starts)"""
    return (alpha + wins) / (alpha + beta + starts)


def _beta_smooth_vec(wins: pd.Series, starts: pd.Series, alpha: int = 1, beta: int = 10) -> pd.Series:
    """Beta 平滑化勝率のベクトル版 (Series 入出力)"""
    w = wins.fillna(0).astype(float)
    s = starts.fillna(0).astype(float)
    return (alpha + w) / (alpha + beta + s)


class SireFeatures:
    """種牡馬産駒特徴量の計算 (PIT安全)"""

    def __init__(self, sire_stats_df: pd.DataFrame) -> None:
        self._stats = sire_stats_df
        if not self._stats.empty:
            self._stats = self._stats.sort_values(["sire_id", "race_date"])

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

    def compute_batch(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """ベクトル化された一括計算。DataFrame を受け取り、特徴量 DataFrame を返す。

        Args:
            df: kettonum, race_date, surface, kyori, sire_id, bms_id 列を持つ DataFrame

        Returns:
            sire_wr, sire_surface_wr, sire_distance_wr, sire_prize_avg, bms_wr 列を持つ DataFrame
        """
        if self._stats.empty or df.empty:
            result = pd.DataFrame(index=df.index)
            for col in ["sire_wr", "sire_surface_wr", "sire_distance_wr", "sire_prize_avg", "bms_wr"]:
                result[col] = np.nan
            return result

        # 各行の sire_id ごとに最新の累積統計をマージ
        # merge_asof で race_date 以前の最新行を取得 (PIT安全)
        # sire_id の型を統一 (object ← ketto3infohansyokunum1 は文字列)
        _stats_str = self._stats.copy()
        _stats_str["sire_id"] = _stats_str["sire_id"].astype(str)
        left = df[["sire_id", "race_date"]].copy()
        left["sire_id"] = left["sire_id"].astype(str)

        # merge_asof は by 列が両側でソートされている必要がある
        _right = _stats_str[["sire_id", "race_date", "sire_starts", "sire_wins",
                             "sire_places", "sire_turf_starts", "sire_turf_wins",
                             "sire_dirt_starts", "sire_dirt_wins",
                             "sire_short_starts", "sire_short_wins",
                             "sire_long_starts", "sire_long_wins",
                             "sire_prize_total"]].sort_values(["sire_id", "race_date"])

        merged = pd.merge_asof(
            left.sort_values(["sire_id", "race_date"]),
            _right,
            on="race_date",
            by="sire_id",
            direction="backward",
        )

        result = pd.DataFrame(index=df.index)

        # 全体勝率
        result["sire_wr"] = _beta_smooth_vec(merged["sire_wins"], merged["sire_starts"])

        # サーフェス別勝率
        is_turf = df["surface"].astype(str) == "turf"
        turf_mask = is_turf.reindex(result.index)
        dirt_mask = ~turf_mask

        result["sire_surface_wr"] = np.where(
            turf_mask,
            _beta_smooth_vec(merged["sire_turf_wins"], merged["sire_turf_starts"]),
            _beta_smooth_vec(merged["sire_dirt_wins"], merged["sire_dirt_starts"]),
        )

        # 距離別勝率
        kyori_num = pd.to_numeric(df["kyori"], errors="coerce").reindex(result.index)
        is_short = kyori_num <= 1600

        result["sire_distance_wr"] = np.where(
            is_short,
            _beta_smooth_vec(merged["sire_short_wins"], merged["sire_short_starts"]),
            _beta_smooth_vec(merged["sire_long_wins"], merged["sire_long_starts"]),
        )

        # 平均賞金
        starts_safe = merged["sire_starts"].fillna(0).clip(lower=1).astype(float)
        result["sire_prize_avg"] = np.log1p(merged["sire_prize_total"].fillna(0) / starts_safe)

        # bms_wr: 母父の産駒勝率 (bms_id で同様にマージ)
        bms_left = df[["bms_id", "race_date"]].rename(columns={"bms_id": "sire_id"}).copy()
        bms_left["sire_id"] = bms_left["sire_id"].astype(str)

        _right_bms = _stats_str[["sire_id", "race_date", "sire_starts", "sire_wins"]].sort_values(["sire_id", "race_date"])

        bms_merged = pd.merge_asof(
            bms_left.sort_values(["sire_id", "race_date"]),
            _right_bms,
            on="race_date",
            by="sire_id",
            direction="backward",
        )
        result["bms_wr"] = _beta_smooth_vec(bms_merged["sire_wins"], bms_merged["sire_starts"])

        return result
