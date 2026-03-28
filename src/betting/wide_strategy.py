"""ワイド戦略"""

from __future__ import annotations


class WideStrategy:
    """
    ワイドベット候補を WideTwoStageModel のスコアから選択する。

    スコア = EV / (E × sqrt(P)) (シャープレシオ近似, Rule 3/15)。
    EV閾値 + スコア閾値の複合フィルタで候補を抽出。
    同一馬制約 + 人気帯多様性制約で重複を回避。
    """

    def select_bets(
        self,
        scored_pairs: list[dict],
        ev_threshold: float = 1.15,
        score_threshold: float = 0.0,
        max_bets: int = 3,
        used_horses: set[int] | None = None,
        used_bands: set[str] | None = None,
    ) -> tuple[list[dict], set[int], set[str]]:
        """
        スコア付きペアからワイドベット候補を選択する。

        同一馬制約: used_horses に含まれる馬番を持つペアを除外。
        人気帯多様性制約: 同じバンドのペアは最大1つまで。

        Args:
            scored_pairs: WideTwoStageModel の出力ペアリスト
            ev_threshold: EV下限閾値
            score_threshold: ワイドスコア閾値
            max_bets: 最大ベット数
            used_horses: 既に使用済みの馬番セット（in-place更新）
            used_bands: 既に使用済みの人気バンドセット（in-place更新）

        Returns:
            (フィルタ済みペアdictのリスト, used_horses, used_bands) のタプル
        """
        if used_horses is None:
            used_horses = set()
        if used_bands is None:
            used_bands = set()

        # used_horses 制約でフィルタ
        candidates = [
            p
            for p in scored_pairs
            if p["umaban_a"] not in used_horses
            and p["umaban_b"] not in used_horses
            and p["ev_wide"] >= ev_threshold
            and p["wide_score_adj"] >= score_threshold
        ]

        candidates.sort(key=lambda x: x["wide_score_adj"], reverse=True)

        selected: list[dict] = []
        for pair in candidates:
            if len(selected) >= max_bets:
                break

            # 人気帯多様性制約: 同じバンドは1つまで
            popularity_a = pair.get("popularity_a", pair.get("umaban_a", 0))
            band = self._categorize_band(popularity_a)
            if band in used_bands:
                continue

            selected.append(pair)
            used_horses.add(pair["umaban_a"])
            used_horses.add(pair["umaban_b"])
            used_bands.add(band)

        return selected, used_horses, used_bands

    @staticmethod
    def _categorize_band(popularity: float) -> str:
        """Classify popularity into bands."""
        if popularity <= 3:
            return "favorite"
        elif popularity <= 6:
            return "mid"
        else:
            return "longshot"
