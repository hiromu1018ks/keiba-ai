"""src/betting/gate_keeper.py のテスト"""

from __future__ import annotations

import pytest

from betting.gate_keeper import GateKeeper
from domain.models import Bet
from domain.types import BetType


@pytest.fixture
def gk() -> GateKeeper:
    return GateKeeper()


class TestGateKeeper:
    def test_passes_bet_above_threshold(self, gk: GateKeeper) -> None:
        """EV下限 > 閾値 → 通過"""
        bet = Bet(
            race_id="R1",
            umaban=1,
            bet_type=BetType.PLACE,
            odds=3.0,
            ev_lower_corrected=1.20,
            stake=500.0,
        )
        assert gk.should_bet(bet, ev_threshold=1.10) is True

    def test_rejects_bet_below_threshold(self, gk: GateKeeper) -> None:
        """EV下限 < 閾値 → 却下"""
        bet = Bet(
            race_id="R1",
            umaban=1,
            bet_type=BetType.PLACE,
            odds=3.0,
            ev_lower_corrected=1.05,
            stake=500.0,
        )
        assert gk.should_bet(bet, ev_threshold=1.10) is False

    def test_filter_bets_removes_low_ev(self, gk: GateKeeper) -> None:
        """filter_bets: EV下限未満のベットを除外"""
        bets = [
            Bet(
                race_id="R1",
                umaban=1,
                bet_type=BetType.PLACE,
                odds=3.0,
                ev_lower_corrected=1.30,
                stake=500.0,
            ),
            Bet(
                race_id="R1",
                umaban=3,
                bet_type=BetType.PLACE,
                odds=3.0,
                ev_lower_corrected=1.08,
                stake=300.0,
            ),
            Bet(
                race_id="R1",
                umaban=5,
                bet_type=BetType.PLACE,
                odds=3.0,
                ev_lower_corrected=1.15,
                stake=400.0,
            ),
        ]
        result = gk.filter_bets(bets, ev_threshold=1.10)
        assert len(result) == 2
        assert all(b.ev_lower_corrected >= 1.10 for b in result)

    def test_filter_bets_empty_input(self, gk: GateKeeper) -> None:
        """空リスト入力で空リスト返却"""
        assert gk.filter_bets([], ev_threshold=1.10) == []

    def test_should_bet_edge_case_exact_threshold(self, gk: GateKeeper) -> None:
        """EV下限 == 閾値 → 通過（境界値テスト）"""
        bet = Bet(
            race_id="R1",
            umaban=1,
            bet_type=BetType.PLACE,
            odds=3.0,
            ev_lower_corrected=1.10,
            stake=500.0,
        )
        assert gk.should_bet(bet, ev_threshold=1.10) is True
