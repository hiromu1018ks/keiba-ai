"""ワイド戦略"""

from __future__ import annotations


class WideStrategy:
    """
    ワイドベット候補を WideTwoStageModel のスコアから選択する。

    スコア = EV / (E × sqrt(P)) (シャープレシオ近似, Rule 3/15)。
    EV閾値 + スコア閾値の複合フィルタで候補を抽出。
    """

    def select_bets(
        self,
        scored_pairs: list[dict],
        ev_threshold: float,
        score_threshold: float,
        max_bets: int = 3,
    ) -> list[dict]:
        """
        スコア付きペアからワイドベット候補を選択する。

        Args:
            scored_pairs: WideTwoStageModel の出力ペアリスト
            ev_threshold: EV下限閾値
            score_threshold: ワイドスコア閾値
            max_bets: 最大ベット数

        Returns:
            フィルタ済みペアdictのリスト（スコア降順）
        """
        candidates = [
            p
            for p in scored_pairs
            if p["ev_wide"] >= ev_threshold and p["wide_score_adj"] >= score_threshold
        ]

        candidates.sort(key=lambda x: x["wide_score_adj"], reverse=True)
        return candidates[:max_bets]
