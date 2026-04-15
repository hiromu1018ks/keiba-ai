"""edgeベース最終足切り"""

from __future__ import annotations

from domain.models import Bet


class GateKeeper:
    """
    Value Betting edge を基準にベットの最終可否を判定する。

    RegimeDetector の戦略パラメータ (edge_threshold) を適用し、
    edge が閾値を下回るベットを除外する。
    """

    def should_bet(self, bet: Bet, ev_threshold: float = 1.10) -> bool:
        """
        単一ベットの可否判定（互換性のため ev_threshold 残す、未使用）。

        Args:
            bet: ベット候補
            ev_threshold: EV下限閾値（API互換のため残す、未使用）

        Returns:
            True: ベット可, False: 却下
        """
        # edge >= 0.03 デフォルト閾値で判定（ev_threshold は無視）
        return bet.edge >= 0.03

    def filter_bets(self, bets: list[Bet], edge_threshold: float = 0.03) -> list[Bet]:
        """
        ベットリストからedge閾値未満を除外する。

        Args:
            bets: ベット候補リスト
            edge_threshold: edge閾値（Value Betting）

        Returns:
            フィルタ済みベットリスト
        """
        return [b for b in bets if b.edge >= edge_threshold]
