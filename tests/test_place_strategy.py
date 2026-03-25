"""src/betting/place_strategy.py のテスト"""

from __future__ import annotations

import pytest

from betting.place_strategy import PlaceStrategy
from domain.models import Bet
from domain.types import BetType


@pytest.fixture
def strategy() -> PlaceStrategy:
    return PlaceStrategy()


class TestPlaceStrategy:
    def test_generate_returns_bets(self, strategy: PlaceStrategy) -> None:
        """generate が Bet リストを返す"""
        feats = _make_features(
            ev_place_lower=[1.20, 1.05, 1.35, 0.98],
            place_odds=[2.0, 1.5, 3.0, 8.0],
        )
        bets = strategy.generate(feats, bankroll=100000, ev_threshold=1.10)
        assert all(isinstance(b, Bet) for b in bets)
        assert all(b.bet_type == BetType.PLACE for b in bets)

    def test_generate_filters_by_ev(self, strategy: PlaceStrategy) -> None:
        """EV閾値未満の馬は除外"""
        feats = _make_features(
            ev_place_lower=[1.20, 1.05, 1.35, 0.98],
            place_odds=[2.0, 1.5, 3.0, 8.0],
        )
        bets = strategy.generate(feats, bankroll=100000, ev_threshold=1.10)
        assert all(b.ev_lower_corrected >= 1.10 for b in bets)

    def test_generate_empty_when_no_candidates(self, strategy: PlaceStrategy) -> None:
        """全馬EV閾値未満 → 空リスト"""
        feats = _make_features(
            ev_place_lower=[1.00, 0.95, 1.02, 0.90],
            place_odds=[2.0, 1.5, 3.0, 8.0],
        )
        bets = strategy.generate(feats, bankroll=100000, ev_threshold=1.10)
        assert bets == []

    def test_generate_stake_is_positive(self, strategy: PlaceStrategy) -> None:
        """stakeが正の値"""
        feats = _make_features(
            ev_place_lower=[1.30, 1.15],
            place_odds=[2.0, 3.0],
        )
        bets = strategy.generate(feats, bankroll=100000, ev_threshold=1.10)
        assert all(b.stake > 0 for b in bets)

    def test_generate_max_bets_limit(self, strategy: PlaceStrategy) -> None:
        """max_bets_per_race で件数制限"""
        feats = _make_features(
            ev_place_lower=[1.30, 1.25, 1.20, 1.15, 1.10],
            place_odds=[2.0, 2.5, 3.0, 4.0, 5.0],
        )
        bets = strategy.generate(feats, bankroll=100000, ev_threshold=1.05, max_bets=2)
        assert len(bets) <= 2


def _make_features(
    ev_place_lower: list[float],
    place_odds: list[float],
) -> dict:
    """テスト用 features dict を生成"""
    n = len(ev_place_lower)
    return {
        "race_id": ["R1"] * n,
        "umaban": list(range(1, n + 1)),
        "ev_lower_place": ev_place_lower,
        "place_odds": place_odds,
    }
