"""ペース適性特徴量 — 角通過順位から推定"""

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
        """1頭分のペース適性特徴量を計算

        Args:
            history: 過去出走DataFrame (race_date, kakuteijyuni, jyuni1c,
                jyuni4c, syussotosu 列が必要)
            target_date: 対象レース日付

        Returns:
            dict with keys: pace_aptitude, front_pace_wr, closing_pace_wr
        """
        result: dict[str, float] = {
            "pace_aptitude": np.nan,
            "front_pace_wr": np.nan,
            "closing_pace_wr": np.nan,
        }
        if history.empty:
            return result

        ts = pd.Timestamp(target_date)
        # *** PIT CRITICAL: 当日以前のみ使用 ***
        past = history[history["race_date"] < ts]
        if len(past) < 2:
            return result

        # 正規化着順と正規化1C通過位置
        norm_finish = past["kakuteijyuni"] / past["syussotosu"]
        norm_1c = past["jyuni1c"] / past["syussotosu"]

        front_mask = norm_1c <= 0.33  # 前ペース (1Cが上位1/3)
        closing_mask = norm_1c > 0.66  # 後ペース (1Cが下位1/3)

        # front pace での勝率 (Beta平滑化)
        front_races = past[front_mask]
        if len(front_races) > 0:
            result["front_pace_wr"] = _beta_smooth(
                int((front_races["kakuteijyuni"] == 1).sum()), len(front_races)
            )
        else:
            result["front_pace_wr"] = _beta_smooth(0, 0)

        # closing pace での勝率
        closing_races = past[closing_mask]
        if len(closing_races) > 0:
            result["closing_pace_wr"] = _beta_smooth(
                int((closing_races["kakuteijyuni"] == 1).sum()), len(closing_races)
            )
        else:
            result["closing_pace_wr"] = _beta_smooth(0, 0)

        # ペース適性スコア: closingでの正規化着順 - frontでの正規化着順
        # 正値 = 後ろ待ちの方が好成績, 負値 = 逃げ/先行の方が好成績
        front_avg = norm_finish[front_mask].mean() if front_mask.any() else np.nan
        closing_avg = norm_finish[closing_mask].mean() if closing_mask.any() else np.nan
        if pd.notna(front_avg) and pd.notna(closing_avg):
            result["pace_aptitude"] = float(closing_avg - front_avg)

        return result

    def compute_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """全馬のペース適性特徴量を一括計算する

        Args:
            df: kettonum, race_id, race_date, surface, distance_bin, jyocd 列を持つ DataFrame

        Returns:
            pace_aptitude, front_pace_wr, closing_pace_wr 列を持つ DataFrame
        """
        from db.readers import load_history_entries, load_history_races

        # 過去走データをロード（HorseHistoryFeatures と同じパターン）
        entries_hist = load_history_entries(self.store)
        races_hist = load_history_races(self.store)

        # 結果列を初期化
        result_cols = ["pace_aptitude", "front_pace_wr", "closing_pace_wr"]
        for col in result_cols:
            df[col] = np.nan

        # 空チェック (テスト環境では mock が空 DataFrame を返す)
        # hasattr で DataFrame かどうかを確認
        if (hasattr(entries_hist, "empty") and entries_hist.empty) or \
           (hasattr(races_hist, "empty") and races_hist.empty) or \
           df.empty:
            return df[["kettonum", "race_id"] + result_cols].copy()

        # 必要列の結合
        race_cols = ["race_id", "trackcd", "kyori", "surface", "track_condition_code"]
        races_subset = races_hist[races_hist["race_id"].isin(entries_hist["race_id"])]
        past_df = entries_hist.merge(
            races_subset[race_cols].drop_duplicates("race_id"),
            on="race_id",
            how="left",
        )

        # distance_bin 追加 (FeatureEngine と同じマッピング)
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

        # syussotosu >= 8 のみ有効な出走のみ対象
        # syussotosu を数値に変換 (テスト環境での MagicMock 対策)
        syussotosu_numeric = pd.to_numeric(past_df["syussotosu"], errors="coerce").fillna(-1)
        valid_mask = syussotosu_numeric >= 8
        past_df = past_df[valid_mask].copy()

        # kettonum ごとの特徴量計算
        for kettonum in df["kettonum"].unique():
            target_rows = df[df["kettonum"] == kettonum]

            # 該当馬の過去走を抽出
            horse_past = past_df[past_df["kettonum"] == kettonum].copy()
            if horse_past.empty:
                continue

            # 各対象レースの特徴量を計算
            for _, target_row in target_rows.iterrows():
                target_id = target_row["race_id"]
                target_date = target_row["race_date"]

                # *** PIT CRITICAL: 対象レースより前のデータのみ使用 ***
                past_before_target = horse_past[horse_past["race_date"] < target_date]

                # compute() を呼び出し
                feat_dict = self.compute(past_before_target, target_date)

                # 結果を保存
                row_mask = (df["kettonum"] == kettonum) & (df["race_id"] == target_id)
                for col, val in feat_dict.items():
                    df.loc[row_mask, col] = val

        return df[["kettonum", "race_id"] + result_cols].copy()
