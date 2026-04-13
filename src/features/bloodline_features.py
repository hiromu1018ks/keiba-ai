"""bloodline_features.py — Group B: 血統・産駒成績特徴量 (Point-in-Time)

主な特徴量:
  - blood_surface_wr:  芝別勝率 Beta平滑化 (entries+races から再構成)
  - blood_distance_wr: 芝1600M以下勝率 Beta平滑化 (entries+races から再構成)
  - blood_condition_wr: 馬場状態別勝率 (Phase 2, 現在NaN)
  - blood_total_wr:    総合成績勝率 Beta平滑化 (entries から再構成)
  - blood_prize_log:   log(1 + 累計賞金)
  - blood_keito_cd:    系統コード (種牡馬系統, e.g. SS=サンデーサイレンス系)

ルックアヘッドバイアス修正:
  従来は horses.parquet (x_UMA ETL時点の累積値) を使用しており、
  BT で未来のレース結果が特徴量に混入していた。
  修正後は horse_career_stats.parquet (各レース時点での事前累積値) を使用。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from db.parquet_store import ParquetStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Beta prior parameters for win-rate smoothing
ALPHA_PRIOR: int = 1
BETA_PRIOR: int = 10
TOTAL_OFFSET: int = ALPHA_PRIOR + BETA_PRIOR  # = 11

FEATURE_COLS: list[str] = [
    "blood_surface_wr",
    "blood_distance_wr",
    "blood_condition_wr",
    "blood_total_wr",
    "blood_prize_log",
    "blood_keito_cd",
]


class BloodlineFeatures:
    """Point-in-Time 血統特徴量を生成。

    horse_career_stats.parquet から各レース時点での累積成績を読み込み、
    Beta 平滑化勝率を計算する。
    """

    def __init__(self, store: ParquetStore) -> None:
        self.store = store
        self._career_cache: pd.DataFrame | None = None
        self._keito_cache: dict[str, str] | None = None

    def _load_career_stats(self) -> pd.DataFrame:
        if self._career_cache is None:
            from db.readers import load_career_stats

            self._career_cache = load_career_stats(self.store)
        return self._career_cache

    def _load_keito_map(self) -> dict[str, str]:
        """sire_id -> keitousystemcd のマッピングを構築。

        JOIN: horses.ketto3infohansyokunum1 -> keito.keitoucode -> keito.keitousystemcd
        """
        if self._keito_cache is None:
            from db.readers import load_keito

            keito = load_keito(self.store)
            horses = (
                self.store.read("raw", "horses")
                if self.store.exists("raw", "horses")
                else pd.DataFrame()
            )
            if keito.empty or horses.empty:
                self._keito_cache = {}
            else:
                sire_col = "ketto3infohansyokunum1"
                code_col = (
                    "keitousystemcd"
                    if "keitousystemcd" in keito.columns
                    else keito.columns[1]
                )
                if sire_col not in horses.columns or code_col not in keito.columns:
                    self._keito_cache = {}
                else:
                    merged = horses[[sire_col]].merge(
                        keito, left_on=sire_col, right_on="keitoucode", how="left"
                    )
                    self._keito_cache = dict(
                        zip(horses[sire_col], merged[code_col].fillna("unknown"))
                    )
        return self._keito_cache

    def compute(self, entry_df: pd.DataFrame) -> pd.DataFrame:
        """entry_df (race_id, umaban, kettonum) -> 血統特徴量 DataFrame。

        horse_career_stats.parquet から point-in-time 累積成績を取得し、
        Beta 平滑化勝率を計算する。
        """
        career = self._load_career_stats()

        if "kettonum" not in entry_df.columns or career.empty:
            return entry_df[["race_id", "umaban"]].assign(**{c: float("nan") for c in FEATURE_COLS})

        # entry_df と career_stats を (race_id, kettonum) で結合
        merge_keys = ["race_id", "kettonum"]
        merged = entry_df[["race_id", "umaban", "kettonum"]].merge(
            career, on=merge_keys, how="left"
        )

        result = merged[["race_id", "umaban"]].copy()

        # --- 総合成績勝率 ---
        result["blood_total_wr"] = np.where(
            merged["cum_starts"].fillna(0) == 0,
            np.nan,
            (merged["cum_wins"].fillna(0) + ALPHA_PRIOR)
            / (merged["cum_starts"].fillna(0) + TOTAL_OFFSET),
        )

        # --- 累計賞金 (log変換) ---
        prize = merged["cum_prize"].fillna(0)
        result["blood_prize_log"] = np.where(prize > 0, np.log1p(prize), np.nan)

        # --- 芝別勝率 (全芝 = ba1chakukaisu の近似) ---
        result["blood_surface_wr"] = np.where(
            merged["cum_turf_starts"].fillna(0) == 0,
            np.nan,
            (merged["cum_turf_wins"].fillna(0) + ALPHA_PRIOR)
            / (merged["cum_turf_starts"].fillna(0) + TOTAL_OFFSET),
        )

        # --- 芝1600以下勝率 (kyori1chakukaisu の近似) ---
        result["blood_distance_wr"] = np.where(
            merged["cum_short_starts"].fillna(0) == 0,
            np.nan,
            (merged["cum_short_wins"].fillna(0) + ALPHA_PRIOR)
            / (merged["cum_short_starts"].fillna(0) + TOTAL_OFFSET),
        )

        # --- 馬場状態別勝率 — Phase 2 ---
        result["blood_condition_wr"] = np.nan

        # --- 系統コード ---
        keito_map = self._load_keito_map()
        if keito_map and "kettonum" in merged.columns:
            horses = (
                self.store.read("raw", "horses")
                if self.store.exists("raw", "horses")
                else pd.DataFrame()
            )
            if (
                not horses.empty
                and "ketto3infohansyokunum1" in horses.columns
            ):
                sire_map = horses.set_index("kettonum")["ketto3infohansyokunum1"]
                sire_ids = merged["kettonum"].map(sire_map)
                result["blood_keito_cd"] = sire_ids.map(keito_map).fillna("unknown")
            else:
                result["blood_keito_cd"] = "unknown"
        else:
            result["blood_keito_cd"] = "unknown"

        return result[["race_id", "umaban"] + FEATURE_COLS]
