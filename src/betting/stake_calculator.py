"""Edge連動ケリー + 1レース2%キャップ (Rule 6)"""

from __future__ import annotations

import math
from dataclasses import replace

from domain.models import Bet
from domain.types import BetType


class StakeCalculator:
    """
    賭け金計算: EV下限値に連動したケリー基準。

    Kelly fraction = (ev_lower - 1) / (odds - 1)
    実際のstake = bankroll × kelly_fraction × kelly_fraction_cap
    100円単位に切り捨て。

    Rule 6: 1レースの最大リスクは資金の2%。
    """

    MIN_EV_THRESHOLD: float = 1.05  # EV下限がこれ未満ならベットしない
    KELLY_FRACTION_CAP: float = 0.25  # Kelly fraction の最大値（full Kellyの1/4）
    RACE_EXPOSURE_CAP: float = 0.02  # 1レースの最大露出率（2%）
    MIN_STAKE: int = 100  # 最低投票額

    def calc_stake(
        self,
        ev_lower: float,
        odds: float,
        bankroll: float,
        bet_type: BetType,
    ) -> float:
        """
        EV下限値からケリー基準で賭け金を計算する。

        Args:
            ev_lower: EV下限値（RobustConfidenceEstimator出力）
            odds: オッズ
            bankroll: 現在の資金
            bet_type: 券種

        Returns:
            100円単位の賭け金(float)。EVが閾値未満なら0.0。
        """
        if bankroll <= 0 or ev_lower < self.MIN_EV_THRESHOLD or odds <= 1.0:
            return 0.0

        edge = ev_lower - 1.0
        kelly_fraction = edge / (odds - 1.0)
        kelly_fraction = min(kelly_fraction, self.KELLY_FRACTION_CAP)

        raw_stake = bankroll * kelly_fraction
        stake = max(0, int(math.floor(raw_stake / self.MIN_STAKE)) * self.MIN_STAKE)
        return float(stake)

    def check_race_exposure(
        self,
        bets: list[Bet],
        bankroll: float,
    ) -> list[Bet]:
        """
        1レースあたりの総露出が資金の2%を超えないよう調整する (Rule 6)。

        レースごとにグループ化し、キャップ超過時は各ベットのstakeを
        比例配分で削減する。入力Betは不変（dataclasses.replace使用）。

        Args:
            bets: ベット候補リスト
            bankroll: 現在の資金

        Returns:
            stake調整済みの新Betリスト（入力は変更しない）
        """
        if bankroll <= 0:
            return []

        max_total = int(bankroll * self.RACE_EXPOSURE_CAP)

        # レースごとにグループ化
        race_groups: dict[str, list[Bet]] = {}
        for bet in bets:
            race_groups.setdefault(bet.race_id, []).append(bet)

        result: list[Bet] = []
        for race_id, group in race_groups.items():
            total_stake = sum(b.stake for b in group)
            if total_stake <= max_total:
                result.extend(group)
            else:
                # 比例配分で削減（入力Betを不変に保つ）
                scale = max_total / total_stake
                for bet in group:
                    adjusted = max(
                        0,
                        int(math.floor(bet.stake * scale / self.MIN_STAKE)) * self.MIN_STAKE,
                    )
                    result.append(replace(bet, stake=float(adjusted)))

        return result
