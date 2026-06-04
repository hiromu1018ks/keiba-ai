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

        JOIN: horses.ketto3infohansyokunum1 -> keito.hansyokunum -> keito.keitoname
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
                # Parquet列名: hansyokunum (JOIN key), keitoname (系統名)
                join_col = "hansyokunum"
                code_col = "keitoname"
                if sire_col not in horses.columns or code_col not in keito.columns:
                    self._keito_cache = {}
                else:
                    # coerce_types で hansyokunum が int64 になるため str に揃える
                    keito[join_col] = keito[join_col].astype(str)
                    merged = horses[[sire_col]].merge(
                        keito, left_on=sire_col, right_on=join_col, how="left"
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
        # datakubun違い等で同一(race_id, umaban)が複数行ある場合に備え dedup
        entry_df = entry_df.drop_duplicates(subset=["race_id", "umaban"], keep="first")

        career = self._load_career_stats()

        if "kettonum" not in entry_df.columns or career.empty:
            return entry_df[["race_id", "umaban"]].assign(**{c: float("nan") for c in FEATURE_COLS})

        # entry_df と career_stats を (race_id, kettonum) で結合
        # career 側に重複がある場合 cross-join で行数爆発するため dedup
        # （PIT安全: cum_start 最小の行＝当日結果を含まない）
        career = career.drop_duplicates(subset=["race_id", "kettonum"], keep="first")
        merge_keys = ["race_id", "kettonum"]
        # surface/track_condition_code も merged に含めて全計算を merged 上で行う
        _sel = ["race_id", "umaban", "kettonum"]
        for _c in ("surface", "track_condition_code"):
            if _c in entry_df.columns:
                _sel.append(_c)
        merged = entry_df[_sel].merge(
            career, on=merge_keys, how="left"
        )

        # cross-join 防御: merged 自体を (race_id, umaban) で dedup
        # これにより merged, result, entry_df の行数が全て一致する
        merged = merged.drop_duplicates(subset=["race_id", "umaban"], keep="first")

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

        # --- 馬場状態別勝率 ---
        _cond_cols = [
            "cum_turf_good_starts", "cum_turf_good_wins",
            "cum_turf_heavy_starts", "cum_turf_heavy_wins",
            "cum_dirt_good_starts", "cum_dirt_good_wins",
            "cum_dirt_heavy_starts", "cum_dirt_heavy_wins",
        ]
        if all(c in merged.columns for c in _cond_cols):
            baba = pd.to_numeric(
                merged.get("track_condition_code", pd.Series(0, index=merged.index)),
                errors="coerce",
            )
            is_good = baba.isin([1, 2])
            is_heavy = baba.isin([3, 4])
            is_turf_race = merged.get("surface", pd.Series("", index=merged.index)) == "turf"

            # 芝良
            cond_turf_good = is_turf_race & is_good
            result["blood_condition_wr"] = np.where(
                cond_turf_good & (merged["cum_turf_good_starts"].fillna(0) > 0),
                (merged["cum_turf_good_wins"].fillna(0) + ALPHA_PRIOR)
                / (merged["cum_turf_good_starts"].fillna(0) + TOTAL_OFFSET),
                np.nan,
            )
            # 芝重
            cond_turf_heavy = is_turf_race & is_heavy
            result["blood_condition_wr"] = np.where(
                cond_turf_heavy & (merged["cum_turf_heavy_starts"].fillna(0) > 0),
                (merged["cum_turf_heavy_wins"].fillna(0) + ALPHA_PRIOR)
                / (merged["cum_turf_heavy_starts"].fillna(0) + TOTAL_OFFSET),
                result["blood_condition_wr"],
            )
            # ダート良
            cond_dirt_good = ~is_turf_race & is_good
            result["blood_condition_wr"] = np.where(
                cond_dirt_good & (merged["cum_dirt_good_starts"].fillna(0) > 0),
                (merged["cum_dirt_good_wins"].fillna(0) + ALPHA_PRIOR)
                / (merged["cum_dirt_good_starts"].fillna(0) + TOTAL_OFFSET),
                result["blood_condition_wr"],
            )
            # ダート重
            cond_dirt_heavy = ~is_turf_race & is_heavy
            result["blood_condition_wr"] = np.where(
                cond_dirt_heavy & (merged["cum_dirt_heavy_starts"].fillna(0) > 0),
                (merged["cum_dirt_heavy_wins"].fillna(0) + ALPHA_PRIOR)
                / (merged["cum_dirt_heavy_starts"].fillna(0) + TOTAL_OFFSET),
                result["blood_condition_wr"],
            )
        else:
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
