"""src/betting/wide_strategy.py のテスト"""

from __future__ import annotations

import pytest

from betting.wide_strategy import WideStrategy


@pytest.fixture
def strategy() -> WideStrategy:
    return WideStrategy()


class TestWideStrategy:
    def test_select_bets_returns_list(self, strategy: WideStrategy) -> None:
        """select_bets がリストを返す"""
        scored = _make_scored_pairs(
            wide_score_adj=[0.05, 0.01, 0.03, 0.005],
            ev_wide=[1.30, 1.10, 1.20, 1.05],
            wide_odds=[5.0, 3.0, 4.0, 2.5],
        )
        result = strategy.select_bets(
            scored,
            ev_threshold=1.05,
            score_threshold=0.01,
            max_bets=3,
        )
        assert isinstance(result, list)

    def test_select_bets_filters_by_ev_and_score(self, strategy: WideStrategy) -> None:
        """EV閾値 + スコア閾値でフィルタ"""
        scored = _make_scored_pairs(
            wide_score_adj=[0.05, 0.01, 0.03, 0.005],
            ev_wide=[1.30, 1.10, 1.20, 1.05],
            wide_odds=[5.0, 3.0, 4.0, 2.5],
        )
        result = strategy.select_bets(
            scored,
            ev_threshold=1.15,
            score_threshold=0.02,
            max_bets=3,
        )
        # ev >= 1.15 AND score >= 0.02: (0.05, 1.30) and (0.03, 1.20)
        assert len(result) == 2

    def test_select_bets_max_bets_limit(self, strategy: WideStrategy) -> None:
        """max_bets で件数制限"""
        scored = _make_scored_pairs(
            wide_score_adj=[0.05, 0.04, 0.03, 0.02],
            ev_wide=[1.30, 1.25, 1.20, 1.15],
            wide_odds=[5.0, 6.0, 4.0, 3.0],
        )
        result = strategy.select_bets(
            scored,
            ev_threshold=1.10,
            score_threshold=0.01,
            max_bets=2,
        )
        assert len(result) <= 2

    def test_select_bets_empty(self, strategy: WideStrategy) -> None:
        """全ペアが閾値未満 → 空リスト"""
        scored = _make_scored_pairs(
            wide_score_adj=[0.005, 0.003],
            ev_wide=[1.05, 1.02],
            wide_odds=[3.0, 2.5],
        )
        result = strategy.select_bets(
            scored,
            ev_threshold=1.15,
            score_threshold=0.02,
            max_bets=3,
        )
        assert result == []

    def test_select_bets_sorted_by_score(self, strategy: WideStrategy) -> None:
        """スコア降順で返す"""
        scored = _make_scored_pairs(
            wide_score_adj=[0.03, 0.05, 0.01],
            ev_wide=[1.20, 1.30, 1.15],
            wide_odds=[4.0, 5.0, 3.0],
        )
        result = strategy.select_bets(
            scored,
            ev_threshold=1.10,
            score_threshold=0.01,
            max_bets=3,
        )
        scores = [r["wide_score_adj"] for r in result]
        assert scores == sorted(scores, reverse=True)


def _make_scored_pairs(
    wide_score_adj: list[float],
    ev_wide: list[float],
    wide_odds: list[float],
) -> list[dict]:
    """テスト用ワイドペアデータを生成"""
    pairs = []
    for i, (score, ev, odds) in enumerate(zip(wide_score_adj, ev_wide, wide_odds)):
        pairs.append(
            {
                "race_id": "R1",
                "umaban_a": 2 * i + 1,
                "umaban_b": 2 * i + 2,
                "wide_score_adj": score,
                "ev_wide": ev,
                "wide_odds": odds,
            }
        )
    return pairs
