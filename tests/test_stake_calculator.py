"""src/betting/stake_calculator.py のテスト"""

from __future__ import annotations

import pytest

from betting.stake_calculator import StakeCalculator
from domain.models import Bet
from domain.types import BetType


@pytest.fixture
def calc() -> StakeCalculator:
    return StakeCalculator()


class TestStakeCalculator:
    def test_calc_stake_positive_ev(self, calc: StakeCalculator) -> None:
        """EV下限 > 1 の場合、正のstakeが計算される"""
        stake = calc.calc_stake(
            ev_lower=1.20,
            odds=5.0,
            bankroll=100000,
            bet_type=BetType.PLACE,
        )
        assert stake > 0
        assert stake >= 100  # 最低投票額

    def test_calc_stake_below_ev_threshold(self, calc: StakeCalculator) -> None:
        """EV下限 < 1.05 の場合、ベットしない (stake=0)"""
        stake = calc.calc_stake(
            ev_lower=0.95,
            odds=5.0,
            bankroll=100000,
            bet_type=BetType.PLACE,
        )
        assert stake == 0

    def test_calc_stake_rounds_to_100(self, calc: StakeCalculator) -> None:
        """stakeは100円単位に切り捨て"""
        stake = calc.calc_stake(
            ev_lower=1.30,
            odds=3.0,
            bankroll=100000,
            bet_type=BetType.PLACE,
        )
        assert stake % 100 == 0

    def test_calc_stake_higher_ev_larger_stake(self, calc: StakeCalculator) -> None:
        """EVが高いほどstakeが大きい（edge連動）"""
        stake_low = calc.calc_stake(1.15, 5.0, 100000, BetType.PLACE)
        stake_high = calc.calc_stake(1.50, 5.0, 100000, BetType.PLACE)
        assert stake_high > stake_low

    def test_check_race_exposure_caps_at_2pct(self, calc: StakeCalculator) -> None:
        """1レースの総stakeが資金の2%を超えたら削減"""
        bets = [
            Bet(
                race_id="R1",
                umaban=1,
                bet_type=BetType.PLACE,
                odds=3.0,
                ev_lower_corrected=1.50,
                stake=3000.0,
            ),
            Bet(
                race_id="R1",
                umaban=3,
                bet_type=BetType.WIDE,
                odds=8.0,
                ev_lower_corrected=1.40,
                stake=2000.0,
            ),
        ]
        # 2% of 50000 = 1000 → 合計5000を1000に削減
        result = calc.check_race_exposure(bets, bankroll=50000)
        total_stake = sum(b.stake for b in result)
        assert total_stake <= 1000

    def test_check_race_exposure_preserves_ratio(self, calc: StakeCalculator) -> None:
        """露出キャップ時、各ベットの相対比率を維持"""
        bets = [
            Bet(
                race_id="R1",
                umaban=1,
                bet_type=BetType.PLACE,
                odds=3.0,
                ev_lower_corrected=1.50,
                stake=6000.0,
            ),
            Bet(
                race_id="R1",
                umaban=3,
                bet_type=BetType.WIDE,
                odds=8.0,
                ev_lower_corrected=1.40,
                stake=4000.0,
            ),
        ]
        result = calc.check_race_exposure(bets, bankroll=50000)
        # ratio = 3:2
        if len(result) >= 2 and result[0].stake > 0:
            assert abs(result[0].stake / result[1].stake - 1.5) < 0.01

    def test_check_race_exposure_single_race_2pct(self, calc: StakeCalculator) -> None:
        """単一ベットでも2%キャップ適用"""
        bets = [
            Bet(
                race_id="R1",
                umaban=1,
                bet_type=BetType.PLACE,
                odds=3.0,
                ev_lower_corrected=1.50,
                stake=5000.0,
            ),
        ]
        result = calc.check_race_exposure(bets, bankroll=100000)
        assert sum(b.stake for b in result) <= 2000

    def test_check_race_exposure_all_stakes_zero(self, calc: StakeCalculator) -> None:
        """露出キャップで全stakeが0になるエッジケース"""
        bets = [
            Bet(
                race_id="R1",
                umaban=1,
                bet_type=BetType.PLACE,
                odds=3.0,
                ev_lower_corrected=1.50,
                stake=100.0,
            ),
        ]
        result = calc.check_race_exposure(bets, bankroll=10)
        # 2% of 10 = 0.2 → floor(0.2/100)*100 = 0
        assert all(b.stake == 0 for b in result)

    def test_calc_stake_zero_bankroll(self, calc: StakeCalculator) -> None:
        """bankroll=0 の場合、stake=0"""
        stake = calc.calc_stake(1.50, 3.0, 0, BetType.PLACE)
        assert stake == 0

    def test_check_race_exposure_different_races_unchanged(self, calc: StakeCalculator) -> None:
        """異なるレースのベットは別々にカウント"""
        bets = [
            Bet(
                race_id="R1",
                umaban=1,
                bet_type=BetType.PLACE,
                odds=3.0,
                ev_lower_corrected=1.50,
                stake=3000.0,
            ),
            Bet(
                race_id="R2",
                umaban=3,
                bet_type=BetType.PLACE,
                odds=3.0,
                ev_lower_corrected=1.50,
                stake=3000.0,
            ),
        ]
        result = calc.check_race_exposure(bets, bankroll=100000)
        # 各レースが独立に2%キャップ
        for bet in result:
            assert bet.stake <= 2000
