"""src/betting/wide_strategy.py のテスト"""

from __future__ import annotations

import pytest

from betting.wide_strategy import WideStrategy


@pytest.fixture
def strategy() -> WideStrategy:
    return WideStrategy()


class TestWideStrategy:
    def test_select_bets_returns_tuple(self, strategy: WideStrategy) -> None:
        """select_bets が (list, set, set) タプルを返す"""
        scored = _make_scored_pairs(
            wide_score_adj=[0.05, 0.01, 0.03, 0.005],
            ev_wide=[1.30, 1.10, 1.20, 1.05],
            wide_odds=[5.0, 3.0, 4.0, 2.5],
        )
        result, horses, bands = strategy.select_bets(
            scored,
            ev_threshold=1.05,
            score_threshold=0.01,
            max_bets=3,
        )
        assert isinstance(result, list)
        assert isinstance(horses, set)
        assert isinstance(bands, set)

    def test_select_bets_filters_by_ev_and_score(self, strategy: WideStrategy) -> None:
        """EV閾値 + スコア閾値でフィルタ"""
        scored = _make_scored_pairs(
            wide_score_adj=[0.05, 0.01, 0.03, 0.005],
            ev_wide=[1.30, 1.10, 1.20, 1.05],
            wide_odds=[5.0, 3.0, 4.0, 2.5],
        )
        result, _, _ = strategy.select_bets(
            scored,
            ev_threshold=1.15,
            score_threshold=0.02,
            max_bets=3,
        )
        # ev >= 1.15 AND score >= 0.02: (0.05, 1.30) and (0.03, 1.20)
        # Both are in same band (popularity_a=1,3 -> favorite), so only 1 selected
        assert len(result) >= 1

    def test_select_bets_max_bets_limit(self, strategy: WideStrategy) -> None:
        """max_bets で件数制限"""
        # Use different popularity bands so multiple bets are selected
        scored = _make_scored_pairs(
            wide_score_adj=[0.05, 0.04, 0.03, 0.02],
            ev_wide=[1.30, 1.25, 1.20, 1.15],
            wide_odds=[5.0, 6.0, 4.0, 3.0],
            popularity_a=[1, 5, 8, 10],
        )
        result, _, _ = strategy.select_bets(
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
        result, horses, bands = strategy.select_bets(
            scored,
            ev_threshold=1.15,
            score_threshold=0.02,
            max_bets=3,
        )
        assert result == []
        assert horses == set()
        assert bands == set()

    def test_select_bets_sorted_by_score(self, strategy: WideStrategy) -> None:
        """スコア降順で返す"""
        scored = _make_scored_pairs(
            wide_score_adj=[0.03, 0.05, 0.01],
            ev_wide=[1.20, 1.30, 1.15],
            wide_odds=[4.0, 5.0, 3.0],
            popularity_a=[1, 5, 10],
        )
        result, _, _ = strategy.select_bets(
            scored,
            ev_threshold=1.10,
            score_threshold=0.01,
            max_bets=3,
        )
        scores = [r["wide_score_adj"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_used_horses_constraint(self, strategy: WideStrategy) -> None:
        """used_horses に含まれる馬番のペアは除外される"""
        scored = _make_scored_pairs(
            wide_score_adj=[0.05, 0.04, 0.03],
            ev_wide=[1.30, 1.25, 1.20],
            wide_odds=[5.0, 6.0, 4.0],
            popularity_a=[1, 5, 10],
        )
        # 最初のペアの馬番 1 が used に含まれる
        result, horses, bands = strategy.select_bets(
            scored,
            ev_threshold=1.10,
            score_threshold=0.01,
            max_bets=3,
            used_horses={1},
        )
        # ペア (1,2) は除外され、(3,4) と (5,6) が選ばれる
        for pair in result:
            assert pair["umaban_a"] != 1
            assert pair["umaban_b"] != 1

    def test_used_horses_updated(self, strategy: WideStrategy) -> None:
        """選択後、used_horses に馬番が追加される"""
        scored = _make_scored_pairs(
            wide_score_adj=[0.05, 0.04, 0.03],
            ev_wide=[1.30, 1.25, 1.20],
            wide_odds=[5.0, 6.0, 4.0],
            popularity_a=[1, 5, 10],
        )
        result, horses, bands = strategy.select_bets(
            scored,
            ev_threshold=1.10,
            score_threshold=0.01,
            max_bets=3,
        )
        for pair in result:
            assert pair["umaban_a"] in horses
            assert pair["umaban_b"] in horses

    def test_used_bands_diversity(self, strategy: WideStrategy) -> None:
        """同じ人気バンドは1つまでしか選ばれない"""
        scored = _make_scored_pairs(
            wide_score_adj=[0.05, 0.04, 0.03],
            ev_wide=[1.30, 1.25, 1.20],
            wide_odds=[5.0, 6.0, 4.0],
            popularity_a=[1, 2, 3],  # すべて favorite バンド
        )
        result, _, bands = strategy.select_bets(
            scored,
            ev_threshold=1.10,
            score_threshold=0.01,
            max_bets=3,
        )
        # 全部 favorite バンドなので最大1つしか選ばれない
        assert len(result) == 1
        assert "favorite" in bands

    def test_categorize_band(self) -> None:
        """_categorize_band の分類確認"""
        assert WideStrategy._categorize_band(1) == "favorite"
        assert WideStrategy._categorize_band(3) == "favorite"
        assert WideStrategy._categorize_band(4) == "mid"
        assert WideStrategy._categorize_band(6) == "mid"
        assert WideStrategy._categorize_band(7) == "longshot"
        assert WideStrategy._categorize_band(15) == "longshot"

    def test_none_used_sets_initialized(self, strategy: WideStrategy) -> None:
        """used_horses=None, used_bands=None の場合、空セットが初期化される"""
        scored = _make_scored_pairs(
            wide_score_adj=[0.05],
            ev_wide=[1.30],
            wide_odds=[5.0],
        )
        result, horses, bands = strategy.select_bets(
            scored,
            ev_threshold=1.10,
            score_threshold=0.01,
            max_bets=3,
            used_horses=None,
            used_bands=None,
        )
        assert isinstance(horses, set)
        assert isinstance(bands, set)

    def test_existing_used_bands_skipped(self, strategy: WideStrategy) -> None:
        """used_bands に含まれるバンドのペアは選ばれない"""
        scored = _make_scored_pairs(
            wide_score_adj=[0.05, 0.04],
            ev_wide=[1.30, 1.25],
            wide_odds=[5.0, 6.0],
            popularity_a=[1, 5],  # favorite, mid
        )
        # favorite バンドが既に使用済み
        result, _, bands = strategy.select_bets(
            scored,
            ev_threshold=1.10,
            score_threshold=0.01,
            max_bets=3,
            used_bands={"favorite"},
        )
        for pair in result:
            band = WideStrategy._categorize_band(pair["popularity_a"])
            assert band != "favorite"


def _make_scored_pairs(
    wide_score_adj: list[float],
    ev_wide: list[float],
    wide_odds: list[float],
    popularity_a: list[int] | None = None,
) -> list[dict]:
    """テスト用ワイドペアデータを生成"""
    pairs = []
    for i, (score, ev, odds) in enumerate(zip(wide_score_adj, ev_wide, wide_odds)):
        pop = popularity_a[i] if popularity_a else (i + 1)
        pairs.append(
            {
                "race_id": "R1",
                "umaban_a": 2 * i + 1,
                "umaban_b": 2 * i + 2,
                "wide_score_adj": score,
                "ev_wide": ev,
                "wide_odds": odds,
                "popularity_a": pop,
            }
        )
    return pairs
