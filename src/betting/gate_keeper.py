"""EV下限値ベース最終足切り"""

from __future__ import annotations

from domain.models import Bet


class GateKeeper:
    """
    EV_lower_corrected を基準にベットの最終可否を判定する。

    RegimeDetector の戦略パラメータ (ev_threshold) を適用し、
    信頼区間下限値が閾値を下回るベットを除外する。
    """

    def should_bet(self, bet: Bet, ev_threshold: float) -> bool:
        """
        単一ベットの可否判定。

        Args:
            bet: ベット候補
            ev_threshold: EV下限閾値（RegimeDetector出力）

        Returns:
            True: ベット可, False: 却下
        """
        return bet.ev_lower_corrected >= ev_threshold

    def filter_bets(self, bets: list[Bet], ev_threshold: float) -> list[Bet]:
        """
        ベットリストからEV閾値未満を除外する。

        Args:
            bets: ベット候補リスト
            ev_threshold: EV下限閾値

        Returns:
            フィルタ済みベットリスト
        """
        return [b for b in bets if b.ev_lower_corrected >= ev_threshold]
