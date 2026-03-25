"""src/betting/win_strategy.py のテスト"""

from __future__ import annotations

import pytest

from betting.win_strategy import WinStrategy
from domain.models import Bet
from domain.types import BetType


@pytest.fixture
def strategy() -> WinStrategy:
    return WinStrategy()


class TestWinStrategy:
    def test_generate_returns_bets(self, strategy: WinStrategy) -> None:
        """generate が Bet リストを返す"""
        feats = _make_features(
            ev_win_lower=[1.30, 1.05, 1.50, 0.95],
            win_odds=[5.0, 3.0, 12.0, 25.0],
        )
        bets = strategy.generate(feats, bankroll=100000, ev_threshold=1.20)
        assert all(isinstance(b, Bet) for b in bets)
        assert all(b.bet_type == BetType.WIN for b in bets)

    def test_generate_filters_by_ev(self, strategy: WinStrategy) -> None:
        """EV閾値未満の馬は除外"""
        feats = _make_features(
            ev_win_lower=[1.30, 1.05, 1.50, 0.95],
            win_odds=[5.0, 3.0, 12.0, 25.0],
        )
        bets = strategy.generate(feats, bankroll=100000, ev_threshold=1.20)
        assert all(b.ev_lower_corrected >= 1.20 for b in bets)

    def test_generate_empty_when_no_candidates(self, strategy: WinStrategy) -> None:
        """全馬EV閾値未満 → 空リスト"""
        feats = _make_features(
            ev_win_lower=[1.10, 1.05, 1.00, 0.95],
            win_odds=[5.0, 3.0, 12.0, 25.0],
        )
        bets = strategy.generate(feats, bankroll=100000, ev_threshold=1.20)
        assert bets == []

    def test_generate_max_bets_limit(self, strategy: WinStrategy) -> None:
        """max_bets_per_race で件数制限"""
        feats = _make_features(
            ev_win_lower=[1.50, 1.40, 1.30, 1.25],
            win_odds=[5.0, 8.0, 12.0, 20.0],
        )
        bets = strategy.generate(feats, bankroll=100000, ev_threshold=1.20, max_bets=2)
        assert len(bets) <= 2


def _make_features(
    ev_win_lower: list[float],
    win_odds: list[float],
) -> dict:
    n = len(ev_win_lower)
    return {
        "race_id": ["R1"] * n,
        "umaban": list(range(1, n + 1)),
        "ev_lower_win_corrected": ev_win_lower,
        "win_odds": win_odds,
    }
