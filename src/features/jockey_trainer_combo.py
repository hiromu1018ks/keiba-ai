"""騎手-調教師コンビ特徴量 (Stage2)

過去出走データから特定の騎手-調教師コンビの実績を計算。
リーク防止: compute() に渡された race_date 以前のデータのみ使用。
Beta(1,10) smoothing で小サンプルを安定化。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from db.parquet_store import ParquetStore

FEATURE_COLS: list[str] = [
    "jt_combo_wr",
    "jt_combo_place_rate",
    "jt_combo_starts",
    "jt_combo_prize_log",
]


class JockeyTrainerComboFeatures:
    """騎手-調教師コンビの過去実績特徴量を生成。"""

    def __init__(self, store: ParquetStore) -> None:
        self.store = store
        self._cache: pd.DataFrame | None = None

    def _load_history(self) -> pd.DataFrame:
        if self._cache is None:
            from db.readers import load_history_entries

            entries = load_history_entries(self.store)
            if "chokyosicode" not in entries.columns:
                self._cache = pd.DataFrame()
                return self._cache
            self._cache = entries[entries["kakuteijyuni"] > 0].copy()
        return self._cache

    def compute(self, entry_df: pd.DataFrame) -> pd.DataFrame:
        """コンビ特徴量を計算。

        Args:
            entry_df: (race_id, umaban, kisyucode, chokyosicode, race_date) を含む DataFrame
        """
        result = entry_df[["race_id", "umaban"]].copy()
        nan_cols = {c: float("nan") for c in FEATURE_COLS}

        hist = self._load_history()
        if hist.empty or "chokyosicode" not in entry_df.columns:
            return result.assign(**nan_cols)

        # リーク防止: entry_df のレース日以前のデータのみ使用
        if not hist.empty and "race_date" in hist.columns and "race_date" in entry_df.columns:
            max_date = entry_df["race_date"].max()
            hist = hist[hist["race_date"] < max_date]

        entry = entry_df.copy()
        entry["jt_combo"] = entry["kisyucode"].astype(str) + "_" + entry["chokyosicode"].astype(str)

        hist = hist.copy()
        hist["jt_combo"] = hist["kisyucode"].astype(str) + "_" + hist["chokyosicode"].astype(str)

        # コンビ別集計
        grouped = hist.groupby("jt_combo")
        stats = pd.DataFrame({
            "jt_starts": grouped["kakuteijyuni"].count(),
            "jt_wins": grouped["kakuteijyuni"].apply(lambda x: (x == 1).sum()),
            "jt_places": grouped["kakuteijyuni"].apply(lambda x: (x <= 3).sum()),
        })

        # Beta(1,10) smoothing
        stats["jt_combo_wr"] = (stats["jt_wins"] + 1) / (stats["jt_starts"] + 11)
        stats["jt_combo_place_rate"] = (stats["jt_places"] + 1) / (stats["jt_starts"] + 11)
        stats["jt_combo_starts"] = stats["jt_starts"]
        stats["jt_combo_prize_log"] = np.log1p(stats["jt_starts"] * 10)  # 賞金列が無い場合の代替

        # 賞金列が存在する場合はそちらを使用
        if "honsyokin" in hist.columns:
            prize_sum = grouped["honsyokin"].apply(
                lambda x: pd.to_numeric(x, errors="coerce").fillna(0).sum()
            )
            stats["jt_combo_prize_log"] = np.log1p(prize_sum)

        # マージ
        result["jt_combo"] = entry["jt_combo"].values
        result = result.merge(
            stats[FEATURE_COLS].reset_index(),
            on="jt_combo",
            how="left",
        )

        return result[["race_id", "umaban"] + FEATURE_COLS]
