"""ペース適性特徴量 — 角通過順位から推定（ベクトル化版）"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from db.parquet_store import ParquetStore


def _beta_smooth(wins: int, starts: int, alpha: int = 1, beta: int = 10) -> float:
    return (alpha + wins) / (alpha + beta + starts)


class PaceAptitudeFeatures:
    """過去走の jyuni1c/jyuni4c からペース適性を計算"""

    def __init__(self, store: ParquetStore) -> None:
        self.store = store

    def compute(self, history: pd.DataFrame, target_date: str | pd.Timestamp) -> dict[str, float]:
        """1頭分のペース適性特徴量を計算"""
        result: dict[str, float] = {
            "pace_aptitude": np.nan,
            "front_pace_wr": np.nan,
            "closing_pace_wr": np.nan,
        }
        if history.empty:
            return result

        ts = pd.Timestamp(target_date)
        past = history[history["race_date"] < ts]
        if len(past) < 2:
            return result

        norm_finish = past["kakuteijyuni"] / past["syussotosu"]
        norm_1c = past["jyuni1c"] / past["syussotosu"]

        front_mask = norm_1c <= 0.33
        closing_mask = norm_1c > 0.66

        front_races = past[front_mask]
        if len(front_races) > 0:
            result["front_pace_wr"] = _beta_smooth(
                int((front_races["kakuteijyuni"] == 1).sum()), len(front_races)
            )
        else:
            result["front_pace_wr"] = _beta_smooth(0, 0)

        closing_races = past[closing_mask]
        if len(closing_races) > 0:
            result["closing_pace_wr"] = _beta_smooth(
                int((closing_races["kakuteijyuni"] == 1).sum()), len(closing_races)
            )
        else:
            result["closing_pace_wr"] = _beta_smooth(0, 0)

        front_avg = norm_finish[front_mask].mean() if front_mask.any() else np.nan
        closing_avg = norm_finish[closing_mask].mean() if closing_mask.any() else np.nan
        if pd.notna(front_avg) and pd.notna(closing_avg):
            result["pace_aptitude"] = float(closing_avg - front_avg)

        return result

    def compute_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """全馬のペース適性特徴量を一括計算する（完全ベクトル化）

        HorseHistoryFeatures と同じパターン: 事前ソート + searchsorted + numpy 集計
        Pythonレベルのループは kettonum のユニーク値1回のみ。
        """
        from db.readers import load_history_entries, load_history_races

        result_cols = ["pace_aptitude", "front_pace_wr", "closing_pace_wr"]

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
        race_cols = ["race_id", "trackcd", "kyori", "surface", "track_condition_code"]
        if "syussotosu" in races_hist.columns:
            race_cols.append("syussotosu")
        # jyocd は entries_hist に既にあるので除外（_x/_y 衝突回避）
        # syssotosu は races 由来を使うため、entries 側から削除（_x/_y 衝突回避）
        entries_for_merge = entries_hist.drop(columns=["syussotosu"], errors="ignore")
        races_subset = races_hist[races_hist["race_id"].isin(entries_for_merge["race_id"])]
        past_df = entries_for_merge.merge(
            races_subset[race_cols].drop_duplicates("race_id"),
            on="race_id",
            how="left",
        )
        if "syussotosu" not in past_df.columns and "syussotosu" in entries_hist.columns:
            past_df["syussotosu"] = entries_hist.set_index("race_id").loc[past_df["race_id"], "syussotosu"].values

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

        # --- 正規化列を事前計算 ---
        hist["norm_finish"] = hist["kakuteijyuni"].astype(float) / hist["syussotosu"].astype(float)
        hist["norm_1c"] = hist["jyuni1c"].astype(float) / hist["syussotosu"].astype(float)
        hist["is_front"] = hist["norm_1c"] <= 0.33
        hist["is_closing"] = hist["norm_1c"] > 0.66
        hist["is_win"] = hist["kakuteijyuni"] == 1

        # --- 完全ベクトル化: 各馬の過去走をソート済み配列として保持 ---
        hist = hist.sort_values(["kettonum", "race_date"]).reset_index(drop=True)

        # 各馬のインデックス範囲を取得
        horse_keys, horse_starts, horse_ends = np.unique(
            hist["kettonum"].values, return_index=True, return_counts=True
        )
        horse_ends = horse_starts + horse_ends

        # 各馬の日付・特徴量配列
        h_dates = hist["race_date"].values  # ソート済み
        h_norm_finish = hist["norm_finish"].values
        h_is_front = hist["is_front"].values
        h_is_closing = hist["is_closing"].values
        h_is_win = hist["is_win"].values

        # 累積和（各馬内での累積値）
        cum_front_count = np.zeros(len(hist), dtype=np.int64)
        cum_closing_count = np.zeros(len(hist), dtype=np.int64)
        cum_front_win = np.zeros(len(hist), dtype=np.int64)
        cum_closing_win = np.zeros(len(hist), dtype=np.int64)
        cum_front_nf_sum = np.zeros(len(hist), dtype=np.float64)
        cum_closing_nf_sum = np.zeros(len(hist), dtype=np.float64)
        cum_total_nf_sum = np.zeros(len(hist), dtype=np.float64)
        cum_total_count = np.zeros(len(hist), dtype=np.int64)

        # 各馬ごとの累積和を計算
        for i, (s, e) in enumerate(zip(horse_starts, horse_ends)):
            fc = h_is_front[s:e]
            cc = h_is_closing[s:e]
            iw = h_is_win[s:e]
            nf = h_norm_finish[s:e]

            cum_front_count[s:e] = np.cumsum(fc.astype(np.int64))
            cum_closing_count[s:e] = np.cumsum(cc.astype(np.int64))
            cum_front_win[s:e] = np.cumsum((fc & iw).astype(np.int64))
            cum_closing_win[s:e] = np.cumsum((cc & iw).astype(np.int64))

            # front/closing の norm_finish 累積和（条件付き）
            nf_front = nf.copy(); nf_front[~fc] = 0
            nf_closing = nf.copy(); nf_closing[~cc] = 0
            cum_front_nf_sum[s:e] = np.cumsum(nf_front)
            cum_closing_nf_sum[s:e] = np.cumsum(nf_closing)
            cum_total_nf_sum[s:e] = np.cumsum(nf)
            cum_total_count[s:e] = np.arange(1, e - s + 1, dtype=np.int64)

        # --- ターゲット行の処理 ---
        targets = df[["kettonum", "race_id", "race_date"]].copy().reset_index(drop=True)

        # kettonum → hist内インデックスマッピング
        kt_to_idx = {kt: i for i, kt in enumerate(horse_keys)}

        results = {
            "pace_aptitude": np.full(len(targets), np.nan),
            "front_pace_wr": np.full(len(targets), np.nan),
            "closing_pace_wr": np.full(len(targets), np.nan),
        }

        # 各ターゲット馬について一括処理
        for kt in targets["kettonum"].unique():
            if kt not in kt_to_idx:
                continue

            hi = kt_to_idx[kt]  # horse_keys 内のインデックス
            hs = horse_starts[hi]
            he = horse_ends[hi]

            mask_target = targets["kettonum"] == kt
            target_dates = targets.loc[mask_target, "race_date"].values
            target_indices = targets.loc[mask_target].index.values

            # 各ターゲット日付に対して searchsorted
            # PIT: side='left' で target_date と同日のレースを除外 (厳密な過去のみ)
            cutoffs = np.searchsorted(h_dates[hs:he], target_dates, side='left')

            # 各カットオフ位置の累積値を参照
            base = hs  # hist 配列内のベースオフセット
            for j, (ti, c) in enumerate(zip(target_indices, cutoffs)):
                if c < 2:
                    continue  # データ不足 → NaN のまま
                pos = base + c - 1  # 最後の有効な過去走のインデックス

                fc_n = int(cum_front_count[pos])
                cc_n = int(cum_closing_count[pos])
                fw_n = int(cum_front_win[pos])
                cw_n = int(cum_closing_win[pos])

                f_wr = _beta_smooth(fw_n, fc_n) if fc_n > 0 else _beta_smooth(0, 0)
                c_wr = _beta_smooth(cw_n, cc_n) if cc_n > 0 else _beta_smooth(0, 0)

                f_avg = (cum_front_nf_sum[pos] / fc_n) if fc_n > 0 else np.nan
                c_avg = (cum_closing_nf_sum[pos] / cc_n) if cc_n > 0 else np.nan

                if not np.isnan(f_avg) and not np.isnan(c_avg):
                    pa = float(c_avg - f_avg)
                else:
                    pa = np.nan

                results["pace_aptitude"][ti] = pa
                results["front_pace_wr"][ti] = f_wr
                results["closing_pace_wr"][ti] = c_wr

        out = df[["kettonum", "race_id"]].copy()
        for c in result_cols:
            out[c] = results[c]
        return out
