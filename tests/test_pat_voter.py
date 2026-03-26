"""src/automation/pat_voter.py のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from domain.models import Bet, SafetyCheckResult, SafetyConfig
from domain.types import BetType


def _make_bet(umaban: int = 1, stake: float = 500.0) -> Bet:
    return Bet(
        race_id="2024032501010101", umaban=umaban,
        bet_type=BetType.PLACE, odds=3.0, ev_lower_corrected=1.20,
        stake=stake,
    )


class TestPatVoter:
    def test_submit_bets_returns_results(self) -> None:
        """ベット投票を実行して結果リストを返す"""
        mock_api = MagicMock()
        mock_api.submit.return_value = [
            {"success": True, "bet_id": "B001"},
            {"success": True, "bet_id": "B002"},
        ]

        from automation.pat_voter import PatVoter
        voter = PatVoter(api=mock_api)
        bets = [_make_bet(1), _make_bet(3)]
        results = voter.submit_bets(bets)

        assert len(results) == 2
        assert results[0].success is True
        assert results[0].bet_id == "B001"

    def test_submit_bets_returns_partial_failure(self) -> None:
        """一部失敗の場合は success=False の結果を含む"""
        mock_api = MagicMock()
        mock_api.submit.return_value = [
            {"success": True, "bet_id": "B001"},
            {"success": False, "error": "オッズ変動"},
        ]

        from automation.pat_voter import PatVoter
        voter = PatVoter(api=mock_api)
        bets = [_make_bet(1), _make_bet(3)]
        results = voter.submit_bets(bets)

        assert results[0].success is True
        assert results[1].success is False
        assert results[1].error == "オッズ変動"

    def test_submit_bets_empty_list(self) -> None:
        """空リストでは何もしない"""
        mock_api = MagicMock()

        from automation.pat_voter import PatVoter
        voter = PatVoter(api=mock_api)
        results = voter.submit_bets([])

        assert results == []
        mock_api.submit.assert_not_called()

    def test_cancel_bets_sends_cancel_request(self) -> None:
        """投票取消を実行"""
        mock_api = MagicMock()
        mock_api.cancel.return_value = True

        from automation.pat_voter import PatVoter
        voter = PatVoter(api=mock_api)
        result = voter.cancel_bets(["B001"])

        assert result is True
        mock_api.cancel.assert_called_once_with(["B001"])

    def test_submit_with_safety_guard_check(self) -> None:
        """SafetyGuard のチェックを通った場合のみ投票"""
        mock_api = MagicMock()
        mock_api.submit.return_value = [{"success": True, "bet_id": "B001"}]

        mock_guard = MagicMock()
        mock_guard.check.return_value = SafetyCheckResult(can_bet=True)

        from automation.pat_voter import PatVoter
        voter = PatVoter(api=mock_api, safety_guard=mock_guard)
        bets = [_make_bet()]
        results = voter.submit_bets(bets, bankroll=100000)

        assert len(results) == 1
        mock_guard.check.assert_called_once_with(bankroll=100000)

    def test_submit_blocked_by_safety_guard(self) -> None:
        """SafetyGuard が拒否した場合は投票しない"""
        mock_api = MagicMock()

        mock_guard = MagicMock()
        mock_guard.check.return_value = SafetyCheckResult(
            can_bet=False, reason="Daily loss exceeded",
        )

        from automation.pat_voter import PatVoter
        voter = PatVoter(api=mock_api, safety_guard=mock_guard)
        bets = [_make_bet()]
        results = voter.submit_bets(bets, bankroll=100000)

        assert results == []
        mock_api.submit.assert_not_called()

    def test_submit_without_safety_guard(self) -> None:
        """SafetyGuard なしでは bankroll チェックなしで投票"""
        mock_api = MagicMock()
        mock_api.submit.return_value = [{"success": True, "bet_id": "B001"}]

        from automation.pat_voter import PatVoter
        voter = PatVoter(api=mock_api)
        bets = [_make_bet()]
        results = voter.submit_bets(bets, bankroll=0)

        assert len(results) == 1
        mock_api.submit.assert_called_once()
