"""コース別適性特徴量 — 競馬場×距離帯の過去勝率"""

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
        """1頭分のコース適性特徴量を計算

        Args:
            history: 過去出走DataFrame
                (race_date, jyocd, kakuteijyuni, distance_bin, syussotosu 列が必要)
            jyocd: 競馬場コード (e.g., "01" = 京都, "02" = 中山, etc.)
            distance_bin: 距離帯 ("sprint", "mile", "intermediate", "long")
            target_date: 対象レース日付

        Returns:
            dict with keys: course_wr, course_distance_wr
        """
        result: dict[str, float] = {"course_wr": np.nan, "course_distance_wr": np.nan}

        if history.empty:
            return result

        ts = pd.Timestamp(target_date)
        # *** PIT CRITICAL: 当日以前のみ使用 ***
        past = history[history["race_date"] < ts]
        if past.empty:
            return result

        # 競馬場別勝率
        venue_races = past[past["jyocd"] == jyocd]
        if len(venue_races) > 0:
            wins = int((venue_races["kakuteijyuni"] == 1).sum())
            result["course_wr"] = _beta_smooth(wins, len(venue_races))
        else:
            result["course_wr"] = _beta_smooth(0, 0)

        # 競馬場×距離帯別勝率
        vd_races = venue_races[venue_races["distance_bin"] == distance_bin]
        if len(vd_races) > 0:
            wins = int((vd_races["kakuteijyuni"] == 1).sum())
            result["course_distance_wr"] = _beta_smooth(wins, len(vd_races))
        else:
            result["course_distance_wr"] = _beta_smooth(0, 0)

        return result

    def compute_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """全馬のコース適性特徴量を一括計算する

        Args:
            df: kettonum, race_id, race_date, surface, distance_bin, jyocd 列を持つ DataFrame

        Returns:
            course_wr, course_distance_wr 列を持つ DataFrame
        """
        from db.readers import load_history_entries, load_history_races

        # 過去走データをロード
        entries_hist = load_history_entries(self.store)
        races_hist = load_history_races(self.store)

        # 結果列を初期化
        result_cols = ["course_wr", "course_distance_wr"]
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
        if "jyocd" in races_hist.columns:
            race_cols.append("jyocd")
        races_subset = races_hist[races_hist["race_id"].isin(entries_hist["race_id"])]
        past_df = entries_hist.merge(
            races_subset[race_cols].drop_duplicates("race_id"),
            on="race_id",
            how="left",
        )

        # distance_bin 追加
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

        # jyocd を文字列化（2桁ゼロ埋め）
        if "jyocd" in past_df.columns:
            past_df["jyocd"] = past_df["jyocd"].astype(str).str.zfill(2)

        # kettonum ごとの特徴量計算
        df["course_wr"] = np.nan
        df["course_distance_wr"] = np.nan

        for kettonum in df["kettonum"].unique():
            target_races = df[df["kettonum"] == kettonum]["race_id"].unique()

            # 該当馬の過去走を抽出 (有効な出走のみ)
            # syussotosu を数値に変換 (テスト環境での MagicMock 対策)
            syussotosu_numeric = pd.to_numeric(past_df["syussotosu"], errors="coerce").fillna(-1)
            horse_past = past_df[
                (past_df["kettonum"] == kettonum) & (syussotosu_numeric >= 8)
            ].copy()

            # 各対象レースの特徴量を計算
            for target_id in target_races:
                target_date = df[df["race_id"] == target_id]["race_date"].values[0]

                # *** PIT CRITICAL: 対象レースより前のデータのみ ***
                past_before_target = horse_past[horse_past["race_date"] < target_date]

                # race_df から jyocd, distance_bin を取得
                race_row = df[df["race_id"] == target_id].iloc[0]
                jyocd = str(race_row.get("jyocd", ""))
                distance_bin = str(race_row.get("distance_bin", ""))

                # compute() を呼び出し
                feat_dict = self.compute(past_before_target, jyocd, distance_bin, target_date)

                # 結果を保存
                row_mask = (df["kettonum"] == kettonum) & (df["race_id"] == target_id)
                for col, val in feat_dict.items():
                    df.loc[row_mask, col] = val

        # 結果列のみを返す
        return df[["kettonum", "race_id", "course_wr", "course_distance_wr"]].copy()
