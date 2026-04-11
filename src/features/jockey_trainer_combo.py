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

        hist = hist.copy()
        hist["jt_combo"] = hist["kisyucode"].astype(str) + "_" + hist["chokyosicode"].astype(str)

        # Sort by combo + date, build numpy arrays per combo
        hist_sorted = hist.sort_values(["jt_combo", "race_date"]).reset_index(drop=True)
        grouped_hist = {k: g.reset_index(drop=True) for k, g in hist_sorted.groupby("jt_combo")}

        combo_arrays: dict[str, dict[str, np.ndarray]] = {}
        for k, g in grouped_hist.items():
            combo_arrays[k] = {
                "race_date": g["race_date"].values.astype("datetime64[ns]"),
                "kakuteijyuni": g["kakuteijyuni"].values.astype(float),
            }
            if "honsyokin" in g.columns:
                combo_arrays[k]["honsyokin"] = (
                    pd.to_numeric(g["honsyokin"], errors="coerce").fillna(0).values
                )

        n_rows = len(entry_df)
        jt_combo_wr = np.full(n_rows, np.nan)
        jt_combo_place_rate = np.full(n_rows, np.nan)
        jt_combo_starts = np.full(n_rows, np.nan)
        jt_combo_prize_log = np.full(n_rows, np.nan)

        for i, row in enumerate(entry_df.itertuples(index=False)):
            key = f"{row.kisyucode}_{row.chokyosicode}"
            arrs = combo_arrays.get(key)
            if arrs is None or len(arrs["race_date"]) == 0:
                continue

            target_date_np = np.datetime64(row.race_date, "ns")
            dates = arrs["race_date"]
            idx = int(dates.searchsorted(target_date_np, side="left"))

            if idx == 0:
                continue

            past_jyuni = arrs["kakuteijyuni"][:idx]
            n = len(past_jyuni)
            wins = float((past_jyuni == 1).sum())
            places = float((past_jyuni <= 3).sum())

            jt_combo_wr[i] = (wins + 1) / (n + 11)
            jt_combo_place_rate[i] = (places + 1) / (n + 11)
            jt_combo_starts[i] = n

            if "honsyokin" in arrs:
                prize_sum = float(arrs["honsyokin"][:idx].sum())
            else:
                prize_sum = float(n) * 10.0
            jt_combo_prize_log[i] = np.log1p(prize_sum)

        result["jt_combo_wr"] = jt_combo_wr
        result["jt_combo_place_rate"] = jt_combo_place_rate
        result["jt_combo_starts"] = jt_combo_starts
        result["jt_combo_prize_log"] = jt_combo_prize_log

        return result[["race_id", "umaban"] + FEATURE_COLS]
