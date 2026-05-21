"""種牡馬産駒特徴量 — PIT安全な累積統計ベース"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def normalize_bloodline_id(value: object) -> str | None:
    """Normalize JV bloodline IDs to 10-digit strings."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        numeric = float(text)
        if np.isfinite(numeric) and numeric.is_integer():
            text = str(int(numeric))
    except ValueError:
        pass
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return text
    return digits[-10:].zfill(10)


def _beta_smooth(wins: int, starts: int, alpha: int = 1, beta: int = 10) -> float:
    """Beta 平滑化勝率: (alpha + wins) / (alpha + beta + starts)"""
    return (alpha + wins) / (alpha + beta + starts)


def _beta_smooth_vec(
    wins: pd.Series, starts: pd.Series, alpha: int = 1, beta: int = 10,
) -> pd.Series:
    """Beta 平滑化勝率のベクトル版 (Series 入出力)"""
    w = wins.fillna(0).astype(float)
    s = starts.fillna(0).astype(float)
    return (alpha + w) / (alpha + beta + s)


class SireFeatures:
    """種牡馬産駒特徴量の計算 (PIT安全)"""

    FEATURE_COLUMNS: tuple[str, ...] = (
        "sire_wr",
        "sire_place_rate",
        "sire_surface_wr",
        "sire_distance_wr",
        "sire_prize_avg",
        "bms_wr",
        "bms_surface_wr",
        "bms_distance_wr",
        "bms_has_history",
        "bms_starts_log",
        "bms_surface_starts_log",
        "bms_distance_starts_log",
    )
    SIRE_FEATURE_COLUMNS: tuple[str, ...] = (
        "sire_wr",
        "sire_place_rate",
        "sire_surface_wr",
        "sire_distance_wr",
        "sire_prize_avg",
    )

    def __init__(self, sire_stats_df: pd.DataFrame) -> None:
        self._stats = sire_stats_df
        if not self._stats.empty:
            self._stats = self._stats.copy()
            self._stats["sire_id"] = self._stats["sire_id"].map(normalize_bloodline_id)
            self._stats = self._stats[self._stats["sire_id"].notna()]
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
        normalized_sire_id = normalize_bloodline_id(sire_id)

        if normalized_sire_id is None or self._stats.empty:
            for col in self.SIRE_FEATURE_COLUMNS:
                result[col] = np.nan
            return result

        # searchsorted で該当日以前の最新行を取得
        mask = self._stats["sire_id"] == normalized_sire_id
        subset = self._stats[mask]
        if subset.empty:
            for col in self.SIRE_FEATURE_COLUMNS:
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
            for col in self.FEATURE_COLUMNS:
                result[col] = np.nan
            return result

        # sire_id を10桁文字列に統一
        stats = self._stats.copy()
        stats["sire_id"] = stats["sire_id"].map(normalize_bloodline_id)
        stats = stats[stats["sire_id"].notna()]

        result = pd.DataFrame(index=df.index)
        n = len(df)

        # 各列を初期化
        for col in ["sire_wins", "sire_starts", "sire_places",
                    "sire_turf_wins", "sire_turf_starts",
                    "sire_dirt_wins", "sire_dirt_starts",
                    "sire_short_wins", "sire_short_starts",
                    "sire_long_wins", "sire_long_starts",
                    "sire_prize_total"]:
            result[col] = np.nan

        # sire_id ごとに groupby で lookup (merge_asof のソート要件を回避)
        sire_ids = df["sire_id"].map(normalize_bloodline_id).values
        race_dates = pd.to_datetime(df["race_date"]).values
        sire_ids_unique = np.array([sid for sid in pd.unique(sire_ids) if sid is not None])

        for sid in sire_ids_unique:
            mask = sire_ids == sid
            subset = stats[stats["sire_id"] == sid]
            if subset.empty:
                continue
            # searchsorted で該当日以前の最新行
            idx_arr = subset["race_date"].searchsorted(
                pd.DatetimeIndex(race_dates[mask]), side="right"
            ) - 1
            valid = idx_arr >= 0
            if not valid.any():
                continue
            # Per-entry lookup: 各エントリが自身のrace_dateに対応する累積行を取得
            positions = np.where(mask)[0]
            for col in ["sire_wins", "sire_starts", "sire_places",
                        "sire_turf_wins", "sire_turf_starts",
                        "sire_dirt_wins", "sire_dirt_starts",
                        "sire_short_wins", "sire_short_starts",
                        "sire_long_wins", "sire_long_starts",
                        "sire_prize_total"]:
                if col not in subset.columns:
                    continue
                vals = subset[col].values
                result.iloc[positions[valid], result.columns.get_loc(col)] = vals[idx_arr[valid]]

        # Beta 平滑化
        result["sire_wr"] = _beta_smooth_vec(result["sire_wins"], result["sire_starts"])
        result["sire_place_rate"] = _beta_smooth_vec(result["sire_places"], result["sire_starts"])

        # サーフェス別勝率
        is_turf = df["surface"].astype(str).values == "turf"
        result["sire_surface_wr"] = np.where(
            is_turf,
            _beta_smooth_vec(result["sire_turf_wins"], result["sire_turf_starts"]),
            _beta_smooth_vec(result["sire_dirt_wins"], result["sire_dirt_starts"]),
        )

        # 距離別勝率
        kyori_num = pd.to_numeric(df["kyori"], errors="coerce").values
        is_short = kyori_num <= 1600
        result["sire_distance_wr"] = np.where(
            is_short,
            _beta_smooth_vec(result["sire_short_wins"], result["sire_short_starts"]),
            _beta_smooth_vec(result["sire_long_wins"], result["sire_long_starts"]),
        )

        # 勝率値の範囲検証
        for wr_col in ["sire_wr", "sire_surface_wr", "sire_distance_wr", "sire_place_rate"]:
            if wr_col in result.columns:
                out_of_range = (result[wr_col] > 1.5) | (result[wr_col] < 0)
                if out_of_range.any():
                    logger.warning(
                        "sire_features: %d values out of [0, 1.5] for %s, clamping",
                        out_of_range.sum(), wr_col,
                    )
                    result.loc[out_of_range, wr_col] = result.loc[out_of_range, wr_col].clip(0, 1.5)

        # 平均賞金
        starts_safe = result["sire_starts"].fillna(0).clip(lower=1).astype(float)
        result["sire_prize_avg"] = np.log1p(result["sire_prize_total"].fillna(0) / starts_safe)

        # bms_wr + BMS拡張: 母父の産駒勝率 + surface/distance 別 — 同じロジックで bms_id を lookup
        bms_ids = df["bms_id"].map(normalize_bloodline_id).values
        bms_wins = np.full(n, np.nan)
        bms_starts = np.full(n, np.nan)
        bms_turf_wins = np.full(n, np.nan)
        bms_turf_starts = np.full(n, np.nan)
        bms_dirt_wins = np.full(n, np.nan)
        bms_dirt_starts = np.full(n, np.nan)
        bms_short_wins = np.full(n, np.nan)
        bms_short_starts = np.full(n, np.nan)
        bms_long_wins = np.full(n, np.nan)
        bms_long_starts = np.full(n, np.nan)

        bms_unique = np.array([bid for bid in pd.unique(bms_ids) if bid is not None])
        for bid in bms_unique:
            mask = bms_ids == bid
            subset = stats[stats["sire_id"] == bid]
            if subset.empty:
                continue
            idx_arr = subset["race_date"].searchsorted(
                pd.DatetimeIndex(race_dates[mask]), side="right"
            ) - 1
            valid = idx_arr >= 0
            if valid.any():
                positions = np.where(mask)[0]
                valid_positions = positions[valid]
                vals = subset["sire_wins"].values
                bms_wins[valid_positions] = vals[idx_arr[valid]]
                vals = subset["sire_starts"].values
                bms_starts[valid_positions] = vals[idx_arr[valid]]
                for arr, col_name in [
                    (bms_turf_wins, "sire_turf_wins"),
                    (bms_turf_starts, "sire_turf_starts"),
                    (bms_dirt_wins, "sire_dirt_wins"),
                    (bms_dirt_starts, "sire_dirt_starts"),
                    (bms_short_wins, "sire_short_wins"),
                    (bms_short_starts, "sire_short_starts"),
                    (bms_long_wins, "sire_long_wins"),
                    (bms_long_starts, "sire_long_starts"),
                ]:
                    if col_name in subset.columns:
                        arr[valid_positions] = subset[col_name].values[idx_arr[valid]]

        result["bms_wr"] = _beta_smooth_vec(pd.Series(bms_wins), pd.Series(bms_starts)).values
        result["bms_has_history"] = pd.Series(bms_starts).notna().astype(float).values
        result["bms_starts_log"] = np.log1p(pd.Series(bms_starts).fillna(0).astype(float)).values

        # BMS拡張: surface/distance 別勝率
        result["bms_surface_wr"] = np.where(
            is_turf,
            _beta_smooth_vec(pd.Series(bms_turf_wins), pd.Series(bms_turf_starts)),
            _beta_smooth_vec(pd.Series(bms_dirt_wins), pd.Series(bms_dirt_starts)),
        )
        result["bms_distance_wr"] = np.where(
            is_short,
            _beta_smooth_vec(pd.Series(bms_short_wins), pd.Series(bms_short_starts)),
            _beta_smooth_vec(pd.Series(bms_long_wins), pd.Series(bms_long_starts)),
        )
        bms_surface_starts = np.where(is_turf, bms_turf_starts, bms_dirt_starts)
        bms_distance_starts = np.where(is_short, bms_short_starts, bms_long_starts)
        result["bms_surface_starts_log"] = np.log1p(
            pd.Series(bms_surface_starts).fillna(0).astype(float)
        ).values
        result["bms_distance_starts_log"] = np.log1p(
            pd.Series(bms_distance_starts).fillna(0).astype(float)
        ).values

        # BMS 勝率値の範囲検証
        for wr_col in ["bms_wr", "bms_surface_wr", "bms_distance_wr"]:
            if wr_col in result.columns:
                out_of_range = (result[wr_col] > 1.5) | (result[wr_col] < 0)
                if out_of_range.any():
                    logger.warning(
                        "sire_features: %d values out of [0, 1.5] for %s, clamping",
                        out_of_range.sum(), wr_col,
                    )
                    result.loc[out_of_range, wr_col] = result.loc[out_of_range, wr_col].clip(0, 1.5)

        return result
