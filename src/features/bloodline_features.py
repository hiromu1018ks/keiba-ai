"""bloodline_features.py — Group B: 血統・産駒成績特徴量 (x_UMA 静的)

主な特徴量:
  - blood_surface_wr:  血統馬場別勝率 (芝=ba1) Beta平滑化
  - blood_distance_wr: 血統距離別勝率 (短距離=kyori1) Beta平滑化
  - blood_condition_wr: 血統馬場状態別勝率 (Phase 2, 現在NaN)
  - blood_total_wr:    血統総合成績勝率 (中央=chuo) Beta平滑化
  - blood_prize_log:   log(1 + 累計賞金)
  - blood_keito_cd:    系統コード (Phase 2, 現在NaN)
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
    """x_UMA の産駒成績から血統特徴量を生成。静的 (馬ごとに1回計算)。"""

    @staticmethod
    def _safe_int(val: object) -> int:
        """NaN-safe int conversion for pandas Series values from left joins."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 0
        try:
            return int(val)
        except (ValueError, TypeError):
            return 0

    def __init__(self, store: ParquetStore) -> None:
        self.store = store
        self._horses_cache: pd.DataFrame | None = None

    def _load_horses(self) -> pd.DataFrame:
        if self._horses_cache is None:
            from db.readers import load_horses

            self._horses_cache = load_horses(self.store)
        return self._horses_cache

    @staticmethod
    def _smoothed_wr(wins: int, total: int) -> float:
        """Beta(alpha, beta) 平滑化勝率: (wins+1)/(total+11)。

        total=0 の場合は NaN を返す (未出走カテゴリ)。
        """
        if total == 0:
            return float("nan")
        return (wins + ALPHA_PRIOR) / (total + TOTAL_OFFSET)

    def compute(self, entry_df: pd.DataFrame) -> pd.DataFrame:
        """entry_df (race_id, umaban, kettonum) -> 血統特徴量 DataFrame。

        x_UMA の産駒成績列 (Ba*/Kyori*/ChuoChakukaisu*/RuikeiHonsyo*) を使用。
        blood_condition_wr and blood_keito_cd are Phase 2 (currently NaN).
        """
        horses_df = self._load_horses()

        if "kettonum" not in entry_df.columns or horses_df.empty:
            return entry_df[["race_id", "umaban"]].assign(**{c: float("nan") for c in FEATURE_COLS})

        merged = entry_df[["race_id", "umaban", "kettonum"]].merge(
            horses_df, on="kettonum", how="left"
        )

        result = merged[["race_id", "umaban"]].copy()

        # --- 馬場別勝率 (芝 = ba1) ---
        ba_cols = [f"ba1chakukaisu{i}" for i in range(1, 7)]
        ba_data = merged[ba_cols].fillna(0).astype(float)
        ba1_wins = ba_data["ba1chakukaisu1"]
        ba1_total = ba_data[ba_cols].sum(axis=1)
        result["blood_surface_wr"] = np.where(
            ba1_total == 0, np.nan, (ba1_wins + ALPHA_PRIOR) / (ba1_total + TOTAL_OFFSET)
        )

        # --- 距離別勝率 (短距離 = kyori1) ---
        ky_cols = [f"kyori1chakukaisu{i}" for i in range(1, 7)]
        ky_data = merged[ky_cols].fillna(0).astype(float)
        ky1_wins = ky_data["kyori1chakukaisu1"]
        ky1_total = ky_data[ky_cols].sum(axis=1)
        result["blood_distance_wr"] = np.where(
            ky1_total == 0, np.nan, (ky1_wins + ALPHA_PRIOR) / (ky1_total + TOTAL_OFFSET)
        )

        # --- 馬場状態別勝率 — Phase 2 ---
        result["blood_condition_wr"] = np.nan

        # --- 総合成績勝率 (中央 = chuo) ---
        ch_cols = [f"chuochakukaisu{i}" for i in range(1, 7)]
        ch_data = merged[ch_cols].fillna(0).astype(float)
        ch_wins = ch_data["chuochakukaisu1"]
        ch_total = ch_data[ch_cols].sum(axis=1)
        result["blood_total_wr"] = np.where(
            ch_total == 0, np.nan, (ch_wins + ALPHA_PRIOR) / (ch_total + TOTAL_OFFSET)
        )

        # --- 累計賞金 (log変換) ---
        prize = pd.to_numeric(merged["ruikeihonsyoheiti"], errors="coerce")
        result["blood_prize_log"] = np.where(prize.fillna(0) > 0, np.log1p(prize.fillna(0)), np.nan)

        # --- 系統コード — Phase 2 ---
        result["blood_keito_cd"] = np.nan

        return result[["race_id", "umaban"] + FEATURE_COLS]
