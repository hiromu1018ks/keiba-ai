"""src/betting/orchestrator.py のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from betting.orchestrator import BettingOrchestrator
from domain.models import Bet, Race
from domain.types import BetType


def _make_race() -> Race:
    return Race(
        year=2024, month_day="0325", jyo_cd="01", kaiji="01",
        nichiji="01", race_num="01", track_cd=10, distance=1600,
        tenko_cd=1, baba_cd=1, syubetu_cd="0", jyoken_cd="0",
        grade_cd="0", field_size=8,
    )


@pytest.fixture
def mock_deps() -> tuple:
    """モック依存関係を構築"""
    stake_calc = MagicMock()
    stake_calc.calc_stake.return_value = 500
    stake_calc.check_race_exposure.side_effect = lambda bets, bk: bets

    dd_ctrl = MagicMock()
    dd_ctrl.adjust_stake.side_effect = lambda s, b: s
    dd_ctrl.update = MagicMock()

    gate_keeper = MagicMock()
    gate_keeper.filter_bets.side_effect = lambda bets, ev_t: [
        b for b in bets if b.ev_lower_corrected >= ev_t
    ]

    meta_switcher = MagicMock()
    meta_switcher.get_strategy_params.return_value = {
        "ev_threshold": 1.10,
        "score_threshold": 0.01,
        "max_bets_per_race": 3,
        "description": "テスト",
    }
    meta_switcher.should_retrain.return_value = False

    place_strategy = MagicMock()
    place_strategy.generate.return_value = [
        Bet("R1", 1, BetType.PLACE, 3.0, 1.20, 500.0),
    ]

    win_strategy = MagicMock()
    win_strategy.generate.return_value = []

    wide_strategy = MagicMock()
    wide_strategy.select_bets.return_value = []

    late_money = MagicMock()
    late_money.process_last_minute.return_value = ([], [])

    quality_screener = MagicMock()
    quality_screener.should_bet.return_value = True

    return (
        stake_calc, dd_ctrl, gate_keeper, meta_switcher,
        place_strategy, win_strategy, wide_strategy, late_money,
        quality_screener,
    )


class TestBettingOrchestrator:
    def test_process_race_returns_bets(self, mock_deps: tuple) -> None:
        """process_race が Bet リストを返す"""
        (
            stake_calc, dd_ctrl, gk, ms,
            ps, ws, wids, lm, qs,
        ) = mock_deps

        orch = BettingOrchestrator(
            stake_calculator=stake_calc,
            gate_keeper=gk,
            meta_switcher=ms,
            place_strategy=ps,
            win_strategy=ws,
            wide_strategy=wids,
            late_money_filter=lm,
            quality_screener=qs,
        )

        race = _make_race()
        feats = {
            "race_id": ["2024032501010101"] * 8,
            "umaban": list(range(1, 9)),
            "ev_lower_place": [1.20, 1.05, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40],
            "place_odds": [2.0] * 8,
        }

        bets = orch.process_race(race, feats, bankroll=100000, dd_ctrl=dd_ctrl)
        assert isinstance(bets, list)

    def test_process_race_skips_by_quality_screener(
        self, mock_deps: tuple
    ) -> None:
        """QualityScreener が却下した場合、空リスト"""
        (
            stake_calc, dd_ctrl, gk, ms,
            ps, ws, wids, lm, qs,
        ) = mock_deps
        qs.should_bet.return_value = False

        orch = BettingOrchestrator(
            stake_calculator=stake_calc,
            gate_keeper=gk,
            meta_switcher=ms,
            place_strategy=ps,
            win_strategy=ws,
            wide_strategy=wids,
            late_money_filter=lm,
            quality_screener=qs,
        )

        race = _make_race()
        feats = {"race_id": ["R1"], "umaban": [1]}
        bets = orch.process_race(race, feats, bankroll=100000, dd_ctrl=dd_ctrl)
        assert bets == []

    def test_pairs_to_bets_converts_wide_pairs(
        self, mock_deps: tuple
    ) -> None:
        """_pairs_to_bets がワイドペアdictをBetリストに変換"""
        (
            stake_calc, dd_ctrl, gk, ms,
            ps, ws, wids, lm, qs,
        ) = mock_deps

        orch = BettingOrchestrator(
            stake_calculator=stake_calc,
            gate_keeper=gk,
            meta_switcher=ms,
            place_strategy=ps,
            win_strategy=ws,
            wide_strategy=wids,
            late_money_filter=lm,
            quality_screener=qs,
        )

        pairs = [
            {"race_id": "R1", "umaban_a": 1, "umaban_b": 3,
             "wide_score_adj": 0.05, "ev_wide": 1.30, "wide_odds": 5.0},
            {"race_id": "R1", "umaban_a": 2, "umaban_b": 4,
             "wide_score_adj": 0.03, "ev_wide": 1.20, "wide_odds": 4.0},
        ]
        bets = orch._pairs_to_bets(pairs)
        assert len(bets) == 2
        assert all(b.bet_type == BetType.WIDE for b in bets)
        assert bets[0].umaban == 1  # 代表馬番
        assert bets[0].ev_lower_corrected == 1.30

    def test_finalize_bets_applies_late_money(
        self, mock_deps: tuple
    ) -> None:
        """finalize_bets が late_money_filter を適用"""
        (
            stake_calc, dd_ctrl, gk, ms,
            ps, ws, wids, lm, qs,
        ) = mock_deps

        pending = [
            Bet("R1", 1, BetType.PLACE, 3.0, 1.20, 500.0),
            Bet("R1", 3, BetType.PLACE, 3.0, 1.15, 300.0),
        ]
        lm.process_last_minute.return_value = (pending[:1], pending[1:])

        orch = BettingOrchestrator(
            stake_calculator=stake_calc,
            gate_keeper=gk,
            meta_switcher=ms,
            place_strategy=ps,
            win_strategy=ws,
            wide_strategy=wids,
            late_money_filter=lm,
            quality_screener=qs,
        )

        approved = orch.finalize_bets(
            race=_make_race(),
            pending_bets=pending,
            odds_t3_snapshot={1: 3.0, 3: 3.0},
            odds_t10_snapshot={1: 5.0, 3: 3.0},
        )
        assert len(approved) == 1
        assert approved[0].umaban == 1

    def test_finalize_bets_empty_input(
        self, mock_deps: tuple
    ) -> None:
        """空リスト入力で空リスト返却"""
        (
            stake_calc, dd_ctrl, gk, ms,
            ps, ws, wids, lm, qs,
        ) = mock_deps

        orch = BettingOrchestrator(
            stake_calculator=stake_calc,
            gate_keeper=gk,
            meta_switcher=ms,
            place_strategy=ps,
            win_strategy=ws,
            wide_strategy=wids,
            late_money_filter=lm,
            quality_screener=qs,
        )

        approved = orch.finalize_bets(
            race=_make_race(),
            pending_bets=[],
            odds_t3_snapshot={},
            odds_t10_snapshot={},
        )
        assert approved == []

    def test_process_race_applies_stake_calc(
        self, mock_deps: tuple
    ) -> None:
        """process_race が stake_calculator を呼び出す"""
        (
            stake_calc, dd_ctrl, gk, ms,
            ps, ws, wids, lm, qs,
        ) = mock_deps

        orch = BettingOrchestrator(
            stake_calculator=stake_calc,
            gate_keeper=gk,
            meta_switcher=ms,
            place_strategy=ps,
            win_strategy=ws,
            wide_strategy=wids,
            late_money_filter=lm,
            quality_screener=qs,
        )

        race = _make_race()
        feats = {
            "race_id": ["2024032501010101"] * 8,
            "umaban": list(range(1, 9)),
            "ev_lower_place": [1.20, 1.05, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40],
            "place_odds": [2.0] * 8,
        }
        orch.process_race(race, feats, bankroll=100000, dd_ctrl=dd_ctrl)
        stake_calc.calc_stake.assert_called()

    def test_process_race_applies_exposure_cap(
        self, mock_deps: tuple
    ) -> None:
        """process_race がレース露出キャップを適用"""
        (
            stake_calc, dd_ctrl, gk, ms,
            ps, ws, wids, lm, qs,
        ) = mock_deps

        orch = BettingOrchestrator(
            stake_calculator=stake_calc,
            gate_keeper=gk,
            meta_switcher=ms,
            place_strategy=ps,
            win_strategy=ws,
            wide_strategy=wids,
            late_money_filter=lm,
            quality_screener=qs,
        )

        race = _make_race()
        feats = {
            "race_id": ["2024032501010101"] * 8,
            "umaban": list(range(1, 9)),
            "ev_lower_place": [1.20, 1.05, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40],
            "place_odds": [2.0] * 8,
        }
        orch.process_race(race, feats, bankroll=100000, dd_ctrl=dd_ctrl)
        stake_calc.check_race_exposure.assert_called_once()
