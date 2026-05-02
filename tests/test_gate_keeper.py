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
        """edge >= 閾値 → 通過"""
        bet = Bet(
            race_id="R1",
            umaban=1,
            bet_type=BetType.PLACE,
            odds=3.0,
            ev_lower_corrected=1.20,
            stake=500.0,
            edge=0.05,
        )
        assert gk.should_bet(bet) is True

    def test_rejects_bet_below_threshold(self, gk: GateKeeper) -> None:
        """edge < 閾値 → 却下"""
        bet = Bet(
            race_id="R1",
            umaban=1,
            bet_type=BetType.PLACE,
            odds=3.0,
            ev_lower_corrected=1.05,
            stake=500.0,
            edge=0.01,
        )
        assert gk.should_bet(bet) is False

    def test_filter_bets_removes_low_edge(self, gk: GateKeeper) -> None:
        """filter_bets: edge閾値未満のベットを除外"""
        bets = [
            Bet(
                race_id="R1", umaban=1, bet_type=BetType.PLACE, odds=3.0,
                ev_lower_corrected=1.30, stake=500.0, edge=0.10,
            ),
            Bet(
                race_id="R1", umaban=3, bet_type=BetType.PLACE, odds=3.0,
                ev_lower_corrected=1.08, stake=300.0, edge=0.01,
            ),
            Bet(
                race_id="R1", umaban=5, bet_type=BetType.PLACE, odds=3.0,
                ev_lower_corrected=1.15, stake=400.0, edge=0.05,
            ),
        ]
        result = gk.filter_bets(bets, edge_threshold=0.03)
        assert len(result) == 2
        assert all(b.edge >= 0.03 for b in result)

    def test_filter_bets_empty_input(self, gk: GateKeeper) -> None:
        """空リスト入力で空リスト返却"""
        assert gk.filter_bets([], edge_threshold=0.03) == []

    def test_should_bet_edge_case_exact_threshold(self, gk: GateKeeper) -> None:
        """edge == 閾値 → 通過（境界値テスト）"""
        bet = Bet(
            race_id="R1",
            umaban=1,
            bet_type=BetType.PLACE,
            odds=3.0,
            ev_lower_corrected=1.10,
            stake=500.0,
            edge=0.04,  # Phase 3: デフォルト閾値0.04に更新
        )
        assert gk.should_bet(bet) is True

    def test_filter_bets_uses_edge(self, gk: GateKeeper) -> None:
        """GateKeeper should filter on edge, not ev_lower_corrected."""
        bets = [
            Bet(race_id="R1", umaban=1, bet_type=BetType.PLACE, odds=1.5,
                ev_lower_corrected=0.0, stake=100.0, edge=0.05),
            Bet(race_id="R1", umaban=2, bet_type=BetType.PLACE, odds=3.0,
                ev_lower_corrected=0.0, stake=100.0, edge=0.01),
        ]
        filtered = gk.filter_bets(bets, edge_threshold=0.03)
        assert len(filtered) == 1
        assert filtered[0].umaban == 1
