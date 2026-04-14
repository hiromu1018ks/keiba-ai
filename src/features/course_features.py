"""コース別適性特徴量 — 競馬場×距離帯の過去勝率（ベクトル化版）"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from db.parquet_store import ParquetStore


def _beta_smooth(wins: int, starts: int, alpha: int = 1, beta: int = 10) -> float:
    return (alpha + wins) / (alpha + beta + starts)


class CourseFeatures:
    """競馬場別・距離帯別の過去勝率を計算"""

    def __init__(self, store: ParquetStore) -> None:
        self.store = store

    def compute(
        self,
        history: pd.DataFrame,
        jyocd: str,
        distance_bin: str,
        target_date: str | pd.Timestamp,
    ) -> dict[str, float]:
        """1頭分のコース適性特徴量を計算"""
        result: dict[str, float] = {"course_wr": np.nan, "course_distance_wr": np.nan}

        if history.empty:
            return result

        ts = pd.Timestamp(target_date)
        past = history[history["race_date"] < ts]
        if past.empty:
            return result

        venue_races = past[past["jyocd"] == jyocd]
        if len(venue_races) > 0:
            wins = int((venue_races["kakuteijyuni"] == 1).sum())
            result["course_wr"] = _beta_smooth(wins, len(venue_races))
        else:
            result["course_wr"] = _beta_smooth(0, 0)

        vd_races = venue_races[venue_races["distance_bin"] == distance_bin]
        if len(vd_races) > 0:
            wins = int((vd_races["kakuteijyuni"] == 1).sum())
            result["course_distance_wr"] = _beta_smooth(wins, len(vd_races))
        else:
            result["course_distance_wr"] = _beta_smooth(0, 0)

        return result

    def compute_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """全馬のコース適性特徴量を一括計算する（完全ベクトル化）

        Pythonレベルのループは kettonum のユニーク値1回のみ。
        """
        from db.readers import load_history_entries, load_history_races

        result_cols = ["course_wr", "course_distance_wr"]

        entries_hist = load_history_entries(self.store)
        races_hist = load_history_races(self.store)

        # 空チェック
        if (hasattr(entries_hist, "empty") and entries_hist.empty) or \
           (hasattr(races_hist, "empty") and races_hist.empty) or \
           df.empty:
            out = df[["kettonum", "race_id"]].copy()
            for c in result_cols:
                out[c] = np.nan
            return out

        # --- 過去走データ準備 ---
        # 重要: entries_hist にも jyocd があるため、races_hist からは除外
        # （merge 時の _x/_y サフィックス衝突を回避）
        race_cols = ["race_id", "trackcd", "kyori", "surface", "track_condition_code"]
        if "syussotosu" in races_hist.columns:
            race_cols.append("syussotosu")
        # syssotosu は races 由来を使うため、entries 側から削除（_x/_y 衝突回避）
        entries_for_merge = entries_hist.drop(columns=["syussotosu"], errors="ignore")
        races_subset = races_hist[races_hist["race_id"].isin(entries_for_merge["race_id"])]
        past_df = entries_for_merge.merge(
            races_subset[race_cols].drop_duplicates("race_id"),
            on="race_id",
            how="left",
        )

        # distance_bin マッピング
        if "distance_bin" not in past_df.columns and "kyori" in past_df.columns:
            is_turf = past_df["surface"] == "turf"
            dist = past_df["kyori"]
            past_df["distance_bin"] = "unknown"
            past_df.loc[is_turf & (dist > 2100), "distance_bin"] = "long"
            past_df.loc[is_turf & (dist <= 2100), "distance_bin"] = "intermediate"
            past_df.loc[is_turf & (dist <= 1700), "distance_bin"] = "mile"
            past_df.loc[is_turf & (dist <= 1400), "distance_bin"] = "sprint"
            past_df.loc[~is_turf & (dist > 1700), "distance_bin"] = "intermediate"
            past_df.loc[~is_turf & (dist <= 1700), "distance_bin"] = "mile"
            past_df.loc[~is_turf & (dist <= 1400), "distance_bin"] = "sprint"

        # jyocd を文字列化（entries_hist 由来）
        if "jyocd" in past_df.columns:
            past_df["jyocd"] = past_df["jyocd"].astype(str).str.zfill(2)

        # syussotosu >= 8 のみ有効
        syussotosu_numeric = pd.to_numeric(past_df["syussotosu"], errors="coerce").fillna(-1)
        past_df = past_df[syussotosu_numeric >= 8].copy()

        if past_df.empty:
            out = df[["kettonum", "race_id"]].copy()
            for c in result_cols:
                out[c] = np.nan
            return out

        # --- ターゲット馬に絞る ---
        target_kettons = set(df["kettonum"].unique())
        hist = past_df[past_df["kettonum"].isin(target_kettons)].copy()
        if hist.empty:
            out = df[["kettonum", "race_id"]].copy()
            for c in result_cols:
                out[c] = np.nan
            return out

        # --- 完全ベクトル化 ---
        hist = hist.sort_values(["kettonum", "race_date"]).reset_index(drop=True)

        # 各馬のインデックス範囲
        horse_keys, horse_starts, horse_ends = np.unique(
            hist["kettonum"].values, return_index=True, return_counts=True
        )
        horse_ends = horse_starts + horse_ends

        # 各馬のデータ配列
        h_dates = hist["race_date"].values
        h_jyocd = hist["jyocd"].values.astype(str)
        h_dist_bin = hist["distance_bin"].values.astype(str)
        h_is_win = (hist["kakuteijyuni"] == 1).values.astype(np.int64)

        # --- ターゲット行処理 ---
        targets = df[["kettonum", "race_id", "race_date", "jyocd", "distance_bin"]].copy().reset_index(drop=True)
        targets["jyocd_str"] = targets["jyocd"].astype(str).str.zfill(2)
        targets["db_str"] = targets["distance_bin"].astype(str)

        kt_to_idx = {kt: i for i, kt in enumerate(horse_keys)}

        results = {
            "course_wr": np.full(len(targets), np.nan),
            "course_distance_wr": np.full(len(targets), np.nan),
        }

        for kt in targets["kettonum"].unique():
            if kt not in kt_to_idx:
                continue

            hi = kt_to_idx[kt]
            hs = horse_starts[hi]
            he = horse_ends[hi]

            mask_target = targets["kettonum"] == kt
            target_dates = targets.loc[mask_target, "race_date"].values
            target_jyocds = targets.loc[mask_target, "jyocd_str"].values
            target_dbs = targets.loc[mask_target, "db_str"].values
            target_indices = targets.loc[mask_target].index.values

            # PIT: side='left' で target_date と同日のレースを除外 (厳密な過去のみ)
            cutoffs = np.searchsorted(h_dates[hs:he], target_dates, side='left')
            base = hs

            for j, (ti, c, tjc, tdb) in enumerate(zip(target_indices, cutoffs, target_jyocds, target_dbs)):
                if c < 1:
                    continue
                pos = base + c - 1

                jc_slice = h_jyocd[base:base + c]
                iw_slice = h_is_win[base:base + c]
                db_slice = h_dist_bin[base:base + c]

                venue_mask = jc_slice == tjc
                vn = int(venue_mask.sum())
                vw = int(iw_slice[venue_mask].sum()) if vn > 0 else 0

                vd_mask = venue_mask & (db_slice == tdb)
                vdn = int(vd_mask.sum())
                vdw = int(iw_slice[vd_mask].sum()) if vdn > 0 else 0

                results["course_wr"][ti] = _beta_smooth(vw, vn)
                results["course_distance_wr"][ti] = _beta_smooth(vdw, vdn)

        out = df[["kettonum", "race_id"]].copy()
        for c in result_cols:
            out[c] = results[c]
        return out
