"""Value Betting Kelly (edge-based) + 1レース2%キャップ (Rule 6)"""

from __future__ import annotations

import math
from dataclasses import replace

from domain.models import Bet
from domain.types import BetType


class StakeCalculator:
    """
    賭け金計算: Value Betting edge に連動したケリー基準。

    Kelly fraction = edge / (odds - 1)
    ただし edge = p_model * odds - 1 (期待利益/単位投資額)

    実際のstake = bankroll × kelly_fraction × FRACTIONAL_KELLY
    100円単位に切り捨て。

    Rule 6: 1レースの最大リスクは資金の2%。
    """

    MIN_EDGE_THRESHOLD: float = 0.005  # Minimum edge to consider betting (0.5%)
    FRACTIONAL_KELLY: float = 0.5  # ハーフケリー
    KELLY_FRACTION_CAP: float = 0.25  # Kelly fraction の最大値（full Kellyの1/4）
    RACE_EXPOSURE_CAP: float = 0.02  # 1レースの最大露出率（2%）
    MIN_STAKE: int = 100  # 最低投票額
    MAX_STAKE: int = 10000  # 最大投票額 (10K yen cap)

    def calc_stake(
        self,
        edge: float,
        odds: float,
        bankroll: float,
        bet_type: BetType,
    ) -> float:
        """Calculate Kelly-optimal stake for Value Betting.

        Args:
            edge: Value Betting edge = p_model * odds - 1 (期待利益/単位投資額)
            odds: Decimal odds (e.g., 1.5 means 1.5x return)
            bankroll: Current bankroll in yen
            bet_type: Type of bet (PLACE, WIN, WIDE)

        Returns:
            Stake in yen (multiple of 100), or 0.0 if no bet recommended.
        """
        if (
            bankroll <= 0
            or odds <= 1.0
            or math.isnan(edge)
            or math.isnan(odds)
            or edge < self.MIN_EDGE_THRESHOLD
        ):
            return 0.0

        # Value Betting Kelly: f* = edge / (odds - 1)
        # edge = p*odds - 1 のとき、f = (p*odds - 1)/(odds-1) は標準Kelly公式
        kelly_fraction = edge / (odds - 1.0)

        # Fractional Kelly (half-Kelly for safety)
        kelly_fraction *= self.FRACTIONAL_KELLY

        # Effective cap: max fraction of bankroll
        effective_cap = self.KELLY_FRACTION_CAP * self.FRACTIONAL_KELLY  # 0.125
        kelly_fraction = min(kelly_fraction, effective_cap)

        # Compute stake
        raw_stake = bankroll * kelly_fraction
        stake = max(0, math.floor(raw_stake / self.MIN_STAKE) * self.MIN_STAKE)

        # Absolute cap
        stake = min(stake, self.MAX_STAKE)

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
