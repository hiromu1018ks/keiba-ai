"""src/betting/late_money_filter.py のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from betting.late_money_filter import LastMinuteSignal, LateMoneyFilter
from domain.models import Bet
from domain.types import BetType


@pytest.fixture
def lmf() -> LateMoneyFilter:
    return LateMoneyFilter()


class TestLateMoneyFilter:
    def test_no_action_when_stable(self, lmf: LateMoneyFilter) -> None:
        """変動なし → NO_ACTION"""
        signal = lmf.check_last_3min(horse_no=1, odds_t10=5.0, odds_t3=5.0)
        assert signal == LastMinuteSignal.NO_ACTION

    def test_cancel_on_25pct_drop(self, lmf: LateMoneyFilter) -> None:
        """25%以上の急落 → CANCEL"""
        signal = lmf.check_last_3min(horse_no=1, odds_t10=10.0, odds_t3=7.0)
        assert signal == LastMinuteSignal.CANCEL

    def test_add_candidate_on_30pct_rise(self, lmf: LateMoneyFilter) -> None:
        """30%以上の急騰 → ADD_CANDIDATE"""
        signal = lmf.check_last_3min(horse_no=1, odds_t10=10.0, odds_t3=14.0)
        assert signal == LastMinuteSignal.ADD_CANDIDATE

    def test_unknown_on_zero_odds(self, lmf: LateMoneyFilter) -> None:
        """オッズ0 → UNKNOWN"""
        signal = lmf.check_last_3min(horse_no=1, odds_t10=0.0, odds_t3=5.0)
        assert signal == LastMinuteSignal.UNKNOWN

    def test_unknown_on_negative_odds(self, lmf: LateMoneyFilter) -> None:
        """負のオッズ → UNKNOWN"""
        signal = lmf.check_last_3min(horse_no=1, odds_t10=5.0, odds_t3=-1.0)
        assert signal == LastMinuteSignal.UNKNOWN

    def test_process_last_minute_cancels_drop(self, lmf: LateMoneyFilter) -> None:
        """process_last_minute: 急落ベットをキャンセル"""
        bets = [
            Bet(
                race_id="R1",
                umaban=1,
                bet_type=BetType.PLACE,
                odds=5.0,
                ev_lower_corrected=1.30,
                stake=500.0,
            ),
            Bet(
                race_id="R1",
                umaban=3,
                bet_type=BetType.PLACE,
                odds=3.0,
                ev_lower_corrected=1.20,
                stake=300.0,
            ),
        ]
        odds_t3 = {1: 3.0, 3: 3.0}  # horse 1: 5→3 (40% drop)
        odds_t10 = {1: 5.0, 3: 3.0}

        approved, cancelled = lmf.process_last_minute(
            pending_bets=bets,
            odds_t3_snapshot=odds_t3,
            odds_t10_snapshot=odds_t10,
            stage2_predictions=None,
        )
        assert len(cancelled) == 1
        assert cancelled[0].umaban == 1
        assert len(approved) == 1

    def test_process_last_minute_all_pass(self, lmf: LateMoneyFilter) -> None:
        """process_last_minute: 全ベット通過"""
        bets = [
            Bet(
                race_id="R1",
                umaban=1,
                bet_type=BetType.PLACE,
                odds=5.0,
                ev_lower_corrected=1.30,
                stake=500.0,
            ),
        ]
        odds_t3 = {1: 4.5}  # 10% drop → NO_ACTION
        odds_t10 = {1: 5.0}

        approved, cancelled = lmf.process_last_minute(
            pending_bets=bets,
            odds_t3_snapshot=odds_t3,
            odds_t10_snapshot=odds_t10,
            stage2_predictions=None,
        )
        assert len(approved) == 1
        assert len(cancelled) == 0

    @patch("betting.late_money_filter.logger")
    def test_log_last_2min_above_threshold(
        self, mock_logger: MagicMock, lmf: LateMoneyFilter
    ) -> None:
        """log_last_2min: 20%以上変動でログ出力"""
        lmf.log_last_2min(horse_no=1, odds_t3=10.0, odds_t2=7.0)
        mock_logger.info.assert_called_once()

    @patch("betting.late_money_filter.logger")
    def test_log_last_2min_below_threshold(
        self, mock_logger: MagicMock, lmf: LateMoneyFilter
    ) -> None:
        """log_last_2min: 20%未満変動でログなし"""
        lmf.log_last_2min(horse_no=1, odds_t3=10.0, odds_t2=9.5)
        mock_logger.info.assert_not_called()

    def test_log_last_2min_zero_odds(self, lmf: LateMoneyFilter) -> None:
        """log_last_2min: オッズ0で例外なし"""
        lmf.log_last_2min(horse_no=1, odds_t3=0.0, odds_t2=5.0)  # no raise
