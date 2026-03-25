"""単勝戦略"""

from __future__ import annotations

import math

from domain.models import Bet
from domain.types import BetType


class WinStrategy:
    """
    単勝ベット候補を生成する。

    単勝は複勝より分散が大きいため、EV閾値は高めに設定する。
    Orchestrator から RegimeDetector の ev_threshold を受け取る。
    """

    KELLY_FRACTION_CAP: float = 0.25
    MIN_STAKE: int = 100

    def generate(
        self,
        feats: dict,
        bankroll: float,
        ev_threshold: float,
        max_bets: int = 2,
    ) -> list[Bet]:
        """
        単勝ベット候補を生成する。

        Args:
            feats: 特徴量dict
            bankroll: 現在の資金
            ev_threshold: EV下限閾値
            max_bets: 1レースあたりの最大ベット数

        Returns:
            Bet リスト
        """
        ev_lower_list = feats["ev_lower_win_corrected"]
        odds_list = feats["win_odds"]
        race_ids = feats["race_id"]
        umabans = feats["umaban"]

        candidates: list[tuple[float, float, str, int]] = []
        for i, ev_lower in enumerate(ev_lower_list):
            if ev_lower >= ev_threshold:
                candidates.append((ev_lower, odds_list[i], race_ids[i], umabans[i]))

        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:max_bets]

        bets: list[Bet] = []
        for ev_lower, odds, race_id, umaban in candidates:
            stake = self._calc_stake(ev_lower, odds, bankroll)
            bets.append(
                Bet(
                    race_id=race_id,
                    umaban=umaban,
                    bet_type=BetType.WIN,
                    odds=odds,
                    ev_lower_corrected=ev_lower,
                    stake=stake,
                )
            )

        return bets

    def _calc_stake(self, ev_lower: float, odds: float, bankroll: float) -> float:
        """簡易Kellyでstake計算（100円単位切り捨て）"""
        if bankroll <= 0 or odds <= 1.0:
            return 0.0
        edge = ev_lower - 1.0
        kelly = min(edge / (odds - 1.0), self.KELLY_FRACTION_CAP)
        raw = bankroll * kelly
        return float(max(0, int(math.floor(raw / self.MIN_STAKE)) * self.MIN_STAKE))
