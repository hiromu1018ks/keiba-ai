"""ワイド 分散ベース・リスク調整スコア (§7)"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd


class WideTwoStageModel:
    """
    ワイド2段階モデル with リスク調整スコア。

    v5.4改: score = EV / (E × √P) でスケール一致 (シャープレシオ近似)

    理論的根拠:
      Var ≈ P × E² → √Var ≈ E × √P
      score = EV / √Var ≈ EV / (E × √P)
    """

    SHARED_FEATURE_COLS: list[str] = [
        "surface",
        "distance_bin",
        "track_condition_code",
        "grade_code",
        "field_size",
    ]

    hit_model: Any
    return_model: Any

    def predict_score(self, pair_df: pd.DataFrame) -> pd.DataFrame:
        """
        ワイド馬券ペアのスコアを計算。
        score = EV / (E × √P) (Rule 3, Rule 15)
        """
        pair_df = pair_df.copy()
        features = pair_df[self.SHARED_FEATURE_COLS].copy()
        for col in ["surface", "distance_bin", "grade_code"]:
            if col in features.columns:
                features[col] = features[col].astype("category")

        pair_df["p_hit"] = self.hit_model.predict(features)
        pair_df["e_return_given_hit"] = self.return_model.predict(features)

        pair_df["ev_wide"] = pair_df["p_hit"] * pair_df["e_return_given_hit"]
        risk_denom = pair_df["e_return_given_hit"] * np.sqrt(np.clip(pair_df["p_hit"], 0.001, None))
        pair_df["wide_score_adj"] = pair_df["ev_wide"] / risk_denom

        return pair_df

    def select_bets(
        self,
        pair_df: pd.DataFrame,
        ev_threshold: float = 1.20,
        score_threshold: float = 0.015,
        max_bets: int = 3,
    ) -> list[dict[str, Any]]:
        """
        2段階フィルタ:
          1. ev_wide >= ev_threshold
          2. wide_score_adj >= score_threshold
        追加フィルタ: popularity_sum >= 6, running_style_combo != 0,
                     p_hit >= 0.05, e_return_given_hit >= 2.0
        """
        scored = self.predict_score(pair_df)
        filtered = scored[
            (scored["ev_wide"] >= ev_threshold)
            & (scored["wide_score_adj"] >= score_threshold)
            & (scored["popularity_sum"] >= 6)
            & (scored["running_style_combo"] != 0)
            & (scored["p_hit"] >= 0.05)
            & (scored["e_return_given_hit"] >= 2.0)
        ]
        top = filtered.nlargest(max_bets, "wide_score_adj")
        return cast(list[dict[str, Any]], top.to_dict("records"))
