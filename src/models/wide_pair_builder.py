"""ワイド馬券ペアビルダー — C(n,2) ペアを構築する"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class WideJointPairBuilder:
    """レース内の全馬ペア (C(n,2)) を構築する

    WideTwoStageModel の学習・推論の前提。
    各ペアに joint_hit ラベル, wide_odds, popularity_sum, running_style_combo を付与。

    必要列 (入力DF):
      race_id, umaban, surface, distance_bin, track_condition_code,
      grade_code, field_size, finish_pos, popularity_rank, running_style,
      wide_odds_{a}_{b} (全ペア分)
    """

    def build(self, entry_df: pd.DataFrame) -> pd.DataFrame:
        """全レースの馬ペアを構築

        Args:
            entry_df: 馬レベルの特徴量DataFrame (1行=1馬)

        Returns:
            ペアDataFrame (1行=1ペア)
        """
        if entry_df.empty:
            return pd.DataFrame()

        all_pairs: list[dict[str, Any]] = []

        for _, group in entry_df.groupby("race_id"):
            horses = group.sort_values("umaban").reset_index(drop=True)
            n = len(horses)

            for i in range(n):
                for j in range(i + 1, n):
                    a = horses.iloc[i]
                    b = horses.iloc[j]

                    pair = self._build_pair(a, b, group)
                    all_pairs.append(pair)

        if not all_pairs:
            return pd.DataFrame()

        pair_df = pd.DataFrame(all_pairs)
        logger.info(f"Built {len(pair_df)} pairs from {entry_df['race_id'].nunique()} races")
        return pair_df

    def _build_pair(
        self,
        a: pd.Series,
        b: pd.Series,
        race_group: pd.DataFrame,
    ) -> dict[str, Any]:
        """単一ペアを構築"""
        pair: dict[str, Any] = {
            "race_id": a["race_id"],
            "umaban_a": int(a["umaban"]),
            "umaban_b": int(b["umaban"]),
            # 共通特徴量 (馬Aの値を使用 — レース内で同一)
            "surface": a["surface"],
            "distance_bin": a["distance_bin"],
            "track_condition_code": a["track_condition_code"],
            "grade_code": a["grade_code"],
            "field_size": a["field_size"],
            # ラベル
            "joint_hit": int(a["finish_pos"] <= 3 and b["finish_pos"] <= 3),
            "popularity_sum": int(a["popularity_rank"]) + int(b["popularity_rank"]),
            "running_style_combo": int(a["running_style"]) + int(b["running_style"]),
        }

        # wide_odds 検索
        pair["wide_odds"] = self._lookup_wide_odds(race_group, int(a["umaban"]), int(b["umaban"]))

        return pair

    @staticmethod
    def _lookup_wide_odds(
        race_group: pd.DataFrame,
        umaban_a: int,
        umaban_b: int,
    ) -> float:
        """wide_odds_{min}_{max} 列からオッズを検索"""
        lo, hi = min(umaban_a, umaban_b), max(umaban_a, umaban_b)
        col = f"wide_odds_{lo}_{hi}"
        if col in race_group.columns:
            val = race_group[col].iloc[0]
            return float(val) if not pd.isna(val) else 0.0
        return 0.0
