"""ワイド馬券ペアビルダー — C(n,2) ペアを構築する"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WideJointPairBuilder:
    """レース内の全馬ペア (C(n,2)) を構築する

    WideTwoStageModel の学習・推論の前提。
    各ペアに joint_hit ラベル, wide_odds, popularity_sum, kyakusitukubun_cd_combo を付与。

    必要列 (入力DF):
      race_id, umaban, surface, distance_bin, track_condition_code,
      grade_code, field_size, kakuteijyuni, popularity_rank, kyakusitukubun_cd (省略可),
      wide_odds_{a}_{b} (全ペア分)
    """

    def build(self, entry_df: pd.DataFrame) -> pd.DataFrame:
        """全レースの馬ペアを構築"""
        if entry_df.empty:
            return pd.DataFrame()

        all_pairs: list[dict[str, Any]] = []

        for _, group in entry_df.groupby("race_id", observed=True):
            horses = group.sort_values("umaban").reset_index(drop=True)
            n = len(horses)
            if n < 2:
                continue

            # Pre-extract as numpy arrays for fast access
            umabans = horses["umaban"].values.astype(int)
            finish_positions = (
                pd.to_numeric(horses["kakuteijyuni"], errors="coerce")
                .fillna(99)
                .values
                .astype(int)
                if "kakuteijyuni" in horses.columns
                else np.full(n, 99, dtype=int)
            )
            popularity = (
                pd.to_numeric(horses["popularity_rank"], errors="coerce")
                if "popularity_rank" in horses.columns
                else pd.Series(99, index=horses.index, dtype=float)
            )
            popularity_ranks = popularity.fillna(99).values.astype(int)
            p_ability = (
                pd.to_numeric(horses["p_ability_win"], errors="coerce").fillna(0.0).values
                if "p_ability_win" in horses.columns
                else np.zeros(n, dtype=float)
            )
            tanodds = (
                pd.to_numeric(horses["tanodds"], errors="coerce").fillna(0.0).values
                if "tanodds" in horses.columns
                else np.zeros(n, dtype=float)
            )
            running_styles = (
                horses["kyakusitukubun_cd"].fillna(0).values.astype(int)
                if "kyakusitukubun_cd" in horses.columns
                else np.zeros(n, dtype=int)
            )

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
            positive_wide_odds = [v for v in wide_odds_cache.values() if v > 0]

            for i, j in combinations(range(n), 2):
                lo, hi = min(umabans[i], umabans[j]), max(umabans[i], umabans[j])
                odds_col = f"wide_odds_{lo}_{hi}"
                wide_odds = wide_odds_cache.get(odds_col, 0.0)
                wide_rank_pct = float("nan")
                if wide_odds > 0 and positive_wide_odds:
                    wide_rank = 1 + sum(v < wide_odds for v in positive_wide_odds)
                    wide_rank_pct = wide_rank / max(len(positive_wide_odds), 1)
                odds_lo = max(float(tanodds[i]), 1e-6)
                odds_hi = max(float(tanodds[j]), 1e-6)

                all_pairs.append(
                    {
                        **race_shared,
                        "umaban_a": int(umabans[i]),
                        "umaban_b": int(umabans[j]),
                        "joint_hit": int(finish_positions[i] <= 3 and finish_positions[j] <= 3),
                        "popularity_sum": int(popularity_ranks[i] + popularity_ranks[j]),
                        "popularity_gap": int(abs(popularity_ranks[i] - popularity_ranks[j])),
                        "kyakusitukubun_cd_combo": int(running_styles[i] + running_styles[j]),
                        "wide_odds": wide_odds,
                        "wide_rank_pct": wide_rank_pct,
                        "p_ability_pair_product": float(p_ability[i] * p_ability[j]),
                        "p_ability_pair_min": float(min(p_ability[i], p_ability[j])),
                        "p_ability_pair_gap": float(abs(p_ability[i] - p_ability[j])),
                        "tanodds_ratio": float(max(odds_lo, odds_hi) / min(odds_lo, odds_hi)),
                        "draw_gap": int(abs(umabans[i] - umabans[j])),
                    }
                )

        if not all_pairs:
            return pd.DataFrame()

        pair_df = pd.DataFrame(all_pairs)
        logger.debug(f"Built {len(pair_df)} pairs from {entry_df['race_id'].nunique()} races")
        return pair_df
