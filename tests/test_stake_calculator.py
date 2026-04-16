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
    def test_calc_stake_positive_edge(self, calc: StakeCalculator) -> None:
        """正のedgeの場合、正のstakeが計算される"""
        # edge = (1.20 - 1) / 5.0 = 0.04
        stake = calc.calc_stake(
            edge=0.04,
            odds=5.0,
            bankroll=100000,
            bet_type=BetType.PLACE,
        )
        assert stake > 0
        assert stake >= 100  # 最低投票額

    def test_calc_stake_below_edge_threshold(self, calc: StakeCalculator) -> None:
        """edge < 0.005 の場合、ベットしない (stake=0)"""
        # ev_lower=0.95, odds=5.0 -> edge = (0.95-1)/5.0 = -0.01 (< 0.005)
        stake = calc.calc_stake(
            edge=-0.01,
            odds=5.0,
            bankroll=100000,
            bet_type=BetType.PLACE,
        )
        assert stake == 0

    def test_calc_stake_rounds_to_100(self, calc: StakeCalculator) -> None:
        """stakeは100円単位に切り捨て"""
        # edge = (1.30 - 1) / 3.0 = 0.10
        stake = calc.calc_stake(
            edge=0.10,
            odds=3.0,
            bankroll=100000,
            bet_type=BetType.PLACE,
        )
        assert stake % 100 == 0

    def test_calc_stake_higher_edge_larger_stake(self, calc: StakeCalculator) -> None:
        """edgeが高いほどstakeが大きい（edge連動）"""
        # ev_lower=1.15, odds=5.0 -> edge = 0.15/5.0 = 0.03
        # ev_lower=1.50, odds=5.0 -> edge = 0.50/5.0 = 0.10
        stake_low = calc.calc_stake(0.03, 5.0, 100000, BetType.PLACE)
        stake_high = calc.calc_stake(0.10, 5.0, 100000, BetType.PLACE)
        assert stake_high > stake_low

    def test_calc_stake_fractional_kelly_halves_stake(self, calc: StakeCalculator) -> None:
        """Fractional Kelly (0.5x) が適用される"""
        # edge = 0.06 (p*odds - 1), odds = 5.0
        # Kelly fraction = 0.06 / (5.0 - 1) = 0.015
        # After FRACTIONAL_KELLY: 0.015 * 0.5 = 0.0075
        # cap check: min(0.0075, 0.125) = 0.0075
        # raw_stake = 100000 * 0.0075 = 750
        # floor(750/100)*100 = 700
        stake = calc.calc_stake(edge=0.06, odds=5.0, bankroll=100000, bet_type=BetType.PLACE)
        assert stake == 700.0

    def test_calc_stake_effective_cap(self, calc: StakeCalculator) -> None:
        """有効キャップ = KELLY_FRACTION_CAP * FRACTIONAL_KELLY = 0.125"""
        # Very high edge: edge/(odds-1) would exceed cap
        # edge = 4.5 (p*odds - 1), odds = 2.0
        # kelly = 4.5 / (2.0 - 1) = 4.5
        # After fractional: 4.5 * 0.5 = 2.25
        # Capped at 0.25 * 0.5 = 0.125
        # raw_stake = 100000 * 0.125 = 12500
        # floor(12500/100)*100 = 12500 -> capped at MAX_STAKE=10000
        stake = calc.calc_stake(edge=4.5, odds=2.0, bankroll=100000, bet_type=BetType.PLACE)
        assert stake == 10000.0  # MAX_STAKE cap

    def test_calc_stake_max_stake_cap(self, calc: StakeCalculator) -> None:
        """MAX_STAKE=10000 でキャップされる"""
        # Large bankroll + high edge -> would exceed 10K without cap
        # edge = 1.0 (p*odds - 1), odds = 2.0
        stake = calc.calc_stake(edge=1.0, odds=2.0, bankroll=1000000, bet_type=BetType.PLACE)
        # kelly = 1.0 / 1 = 1.0, *0.5=0.5, capped at 0.125
        # raw = 1000000 * 0.125 = 125000 -> capped at 10000
        assert stake <= 10000.0

    def test_calc_stake_fractional_kelly_constants(self, calc: StakeCalculator) -> None:
        """FRACTIONAL_KELLY が 0.5 であることを確認"""
        assert calc.FRACTIONAL_KELLY == 0.5
        assert calc.MAX_STAKE == 10000

    def test_calc_stake_min_edge_threshold_constant(self, calc: StakeCalculator) -> None:
        """MIN_EDGE_THRESHOLD は 0.005 (0.5%)"""
        assert calc.MIN_EDGE_THRESHOLD == 0.005

    def test_calc_stake_no_kelly_fraction_cap_constant(self, calc: StakeCalculator) -> None:
        """KELLY_FRACTION_CAP は 0.25 のまま（FRACTIONAL_KELLY で乗算）"""
        assert calc.KELLY_FRACTION_CAP == 0.25

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
        # edge = (1.50 - 1) / 3.0 = 0.1667
        stake = calc.calc_stake(edge=0.1667, odds=3.0, bankroll=0, bet_type=BetType.PLACE)
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


class TestStakeCalculatorValueBetting:
    """Value Betting Kelly (edge-based) のテスト."""

    def test_calc_stake_value_betting_positive_edge(self, calc: StakeCalculator) -> None:
        """Value Betting: positive edge should produce a stake."""
        # edge=0.033, odds=1.5, bankroll=100000
        # kelly = 0.033 / (1.5 - 1) = 0.066
        # half-kelly: 0.066 * 0.5 = 0.033
        # cap check: min(0.033, 0.125) = 0.033
        # raw_stake = 100000 * 0.033 = 3300
        # rounded: 3300
        stake = calc.calc_stake(edge=0.033, odds=1.5, bankroll=100_000, bet_type=BetType.PLACE)
        assert stake == 3300.0

    def test_calc_stake_value_betting_zero_edge(self, calc: StakeCalculator) -> None:
        """Value Betting: zero edge should return 0."""
        stake = calc.calc_stake(edge=0.0, odds=1.5, bankroll=100_000, bet_type=BetType.PLACE)
        assert stake == 0.0

    def test_calc_stake_value_betting_negative_edge(self, calc: StakeCalculator) -> None:
        """Value Betting: negative edge should return 0."""
        stake = calc.calc_stake(edge=-0.05, odds=1.5, bankroll=100_000, bet_type=BetType.PLACE)
        assert stake == 0.0

    def test_calc_stake_value_betting_high_edge_respects_cap(self, calc: StakeCalculator) -> None:
        """Value Betting: high edge should still respect the Kelly cap."""
        # edge=0.20, odds=1.5, bankroll=100000
        # kelly = 0.20 / 0.5 = 0.4
        # half-kelly: 0.2, capped at 0.125
        # raw_stake = 100000 * 0.125 = 12500 -> capped at 10000
        stake = calc.calc_stake(edge=0.20, odds=1.5, bankroll=100_000, bet_type=BetType.PLACE)
        assert stake == 10000.0

    def test_calc_stake_value_betting_formula_equivalence(self, calc: StakeCalculator) -> None:
        """Verify VB Kelly = edge / (odds - 1) matches standard Kelly when edge = p*odds - 1."""
        p = 0.70
        odds = 1.5
        edge = p * odds - 1  # 0.05
        standard_kelly = (p * (odds - 1) - (1 - p)) / (odds - 1)
        vb_kelly = edge / (odds - 1)
        assert abs(standard_kelly - vb_kelly) < 1e-10
