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
    from db.repository import DataRepository

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

    def __init__(self, repo: DataRepository) -> None:
        self.repo = repo
        self._horses_cache: pd.DataFrame | None = None

    def _load_horses(self) -> pd.DataFrame:
        if self._horses_cache is None:
            self._horses_cache = self.repo.load_horses()
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
        """entry_df (race_id, umaban, ketto_num) -> 血統特徴量 DataFrame。

        x_UMA の産駒成績列 (Ba*/Kyori*/ChuoChakukaisu*/RuikeiHonsyo*) を使用。
        blood_condition_wr and blood_keito_cd are Phase 2 (currently NaN).
        """
        horses_df = self._load_horses()

        # ketto_num で join (entry -> horses)
        if "ketto_num" not in entry_df.columns or horses_df.empty:
            return entry_df[["race_id", "umaban"]].assign(
                **{c: float("nan") for c in FEATURE_COLS}
            )

        merged = entry_df[["race_id", "umaban", "ketto_num"]].merge(
            horses_df, left_on="ketto_num", right_on="kettonum", how="left"
        )

        rows: list[dict[str, float | int | str]] = []
        for _, row in merged.iterrows():
            feats: dict[str, float | int | str] = {}

            # --- 馬場別勝率 (芝 = ba1) ---
            ba1_wins = self._safe_int(row.get("ba1chakukaisu1", 0))
            ba1_total = sum(
                self._safe_int(row.get(f"ba1chakukaisu{i}", 0)) for i in range(1, 7)
            )
            feats["blood_surface_wr"] = self._smoothed_wr(ba1_wins, ba1_total)

            # --- 距離別勝率 (短距離 = kyori1) ---
            ky1_wins = self._safe_int(row.get("kyori1chakukaisu1", 0))
            ky1_total = sum(
                self._safe_int(row.get(f"kyori1chakukaisu{i}", 0)) for i in range(1, 7)
            )
            feats["blood_distance_wr"] = self._smoothed_wr(ky1_wins, ky1_total)

            # --- 馬場状態別勝率 — Phase 2 ---
            feats["blood_condition_wr"] = float("nan")

            # --- 総合成績勝率 (中央 = chuo) ---
            chuo_wins = self._safe_int(row.get("chuochakukaisu1", 0))
            chuo_total = sum(
                self._safe_int(row.get(f"chuochakukaisu{i}", 0)) for i in range(1, 7)
            )
            feats["blood_total_wr"] = self._smoothed_wr(chuo_wins, chuo_total)

            # --- 累計賞金 (log変換) ---
            prize = row.get("ruikeihonsyoheichi")
            if pd.notna(prize) and float(prize) > 0:
                feats["blood_prize_log"] = float(np.log1p(float(prize)))
            else:
                feats["blood_prize_log"] = float("nan")

            # --- 系統コード — Phase 2 (x_KEITO join needed) ---
            feats["blood_keito_cd"] = float("nan")

            feats["race_id"] = row["race_id"]
            feats["umaban"] = row["umaban"]
            rows.append(feats)

        result = pd.DataFrame(rows)
        if result.empty:
            return entry_df[["race_id", "umaban"]].assign(
                **{c: float("nan") for c in FEATURE_COLS}
            )
        return result[["race_id", "umaban"] + FEATURE_COLS]
