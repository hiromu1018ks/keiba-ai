"""ワイド馬券ペアビルダー — C(n,2) ペアを構築する"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class WideJointPairBuilder:
    """レース内の全馬ペア (C(n,2)) を構築する

    WideTwoStageModel の学習・推論の前提。
    各ペアに joint_hit ラベル, wide_odds, popularity_sum, running_style_combo を付与。

    必要列 (入力DF):
      race_id, umaban, surface, distance_bin, track_condition_code,
      grade_code, field_size, kakuteijyuni, popularity_rank, running_style,
      wide_odds_{a}_{b} (全ペア分)
    """

    def build(self, entry_df: pd.DataFrame) -> pd.DataFrame:
        """全レースの馬ペアを構築"""
        if entry_df.empty:
            return pd.DataFrame()

        all_pairs: list[dict[str, Any]] = []

        for _, group in entry_df.groupby("race_id"):
            horses = group.sort_values("umaban").reset_index(drop=True)
            n = len(horses)
            if n < 2:
                continue

            # Pre-extract as numpy arrays for fast access
            umabans = horses["umaban"].values.astype(int)
            finish_positions = horses["kakuteijyuni"].values.astype(int)
            popularity_ranks = horses["popularity_rank"].values.astype(int)
            running_styles = horses["running_style"].values.astype(int)

            # Get wide odds columns from first row
            first_row = horses.iloc[0]
            race_shared: dict[str, Any] = {
                "race_id": first_row["race_id"],
                "surface": first_row["surface"],
                "distance_bin": first_row["distance_bin"],
                "track_condition_code": first_row["track_condition_code"],
                "grade_code": first_row["grade_code"],
                "field_size": first_row["field_size"],
            }

            # Build wide_odds lookup from first row columns
            wide_odds_cache: dict[str, float] = {}
            for col in horses.columns:
                if col.startswith("wide_odds_"):
                    val = horses[col].iloc[0]
                    wide_odds_cache[col] = float(val) if not pd.isna(val) else 0.0

            for i, j in combinations(range(n), 2):
                lo, hi = min(umabans[i], umabans[j]), max(umabans[i], umabans[j])
                odds_col = f"wide_odds_{lo}_{hi}"

                all_pairs.append(
                    {
                        **race_shared,
                        "umaban_a": int(umabans[i]),
                        "umaban_b": int(umabans[j]),
                        "joint_hit": int(finish_positions[i] <= 3 and finish_positions[j] <= 3),
                        "popularity_sum": int(popularity_ranks[i] + popularity_ranks[j]),
                        "running_style_combo": int(running_styles[i] + running_styles[j]),
                        "wide_odds": wide_odds_cache.get(odds_col, 0.0),
                    }
                )

        if not all_pairs:
            return pd.DataFrame()

        pair_df = pd.DataFrame(all_pairs)
        logger.info(f"Built {len(pair_df)} pairs from {entry_df['race_id'].nunique()} races")
        return pair_df
