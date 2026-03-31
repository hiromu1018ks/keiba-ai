"""src/automation/scheduler.py のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from domain.models import Bet, Race, SafetyCheckResult
from domain.types import BetType


def _make_race(race_num: str = "01") -> Race:
    return Race(
        year=2024,
        month_day="0325",
        jyo_cd="01",
        kaiji="01",
        nichiji="01",
        race_num=race_num,
        track_cd=10,
        distance=1600,
        tenko_cd=1,
        baba_cd=1,
        syubetu_cd="0",
        jyoken_cd="0",
        grade_cd="0",
        field_size=8,
    )


def _make_bet(umaban: int = 1) -> Bet:
    return Bet(
        race_id="2024032501010101",
        umaban=umaban,
        bet_type=BetType.PLACE,
        odds=3.0,
        ev_lower_corrected=1.20,
        stake=500.0,
    )


class TestRaceScheduler:
    def _make_scheduler(self) -> "RaceScheduler":
        mock_orch = MagicMock()
        mock_orch.process_race.return_value = [_make_bet()]
        mock_orch.finalize_bets.return_value = [_make_bet()]

        mock_collector = MagicMock()
        mock_collector.collect_t3_snapshot.return_value = {1: 3.5}
        mock_collector.collect_t2_snapshot.return_value = {1: 3.2}

        mock_voter = MagicMock()
        mock_voter.submit_bets.return_value = [
            MagicMock(success=True, bet_id="B001"),
        ]

        mock_guard = MagicMock()
        mock_guard.check.return_value = SafetyCheckResult(can_bet=True)

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_race_cards.return_value = [_make_race()]
        mock_fetcher.fetch_odds_snapshot.return_value = {1: 5.0}

        mock_lmf = MagicMock()
        mock_lmf.log_last_2min.return_value = None

        from automation.scheduler import RaceScheduler

        return RaceScheduler(
            orchestrator=mock_orch,
            odds_collector=mock_collector,
            pat_voter=mock_voter,
            safety_guard=mock_guard,
            late_money_filter=mock_lmf,
            fetcher=mock_fetcher,
        )

    def test_fetch_race_schedule_returns_races(self) -> None:
        """指定日のレーススケジュールを取得"""
        scheduler = self._make_scheduler()
        races = scheduler.fetch_race_schedule("2024-03-25")

        assert len(races) == 1
        assert isinstance(races[0], Race)

    def test_process_race_calls_orchestrator(self) -> None:
        """process_race が orchestrator を呼び出す"""
        scheduler = self._make_scheduler()
        race = _make_race()

        pending = scheduler.process_race(
            race=race,
            feats={"race_id": ["2024032501010101"], "umaban": [1]},
            bankroll=100000,
            dd_ctrl=MagicMock(),
        )

        assert isinstance(pending, list)
        scheduler._orchestrator.process_race.assert_called_once()

    def test_finalize_bets_collects_t3_and_calls_orchestrator(self) -> None:
        """finalize_bets が t-3min スナップショットを取得し orchestrator を呼ぶ"""
        scheduler = self._make_scheduler()
        race = _make_race()
        pending = [_make_bet()]

        approved = scheduler.finalize_bets(race, pending, odds_t10_snapshot={1: 5.0})

        scheduler._odds_collector.collect_t3_snapshot.assert_called_once_with(race.race_id)
        scheduler._orchestrator.finalize_bets.assert_called_once()

    def test_log_t2_calls_late_money_filter(self) -> None:
        """log_t2_snapshot が t-2min スナップショットを収集しログに記録"""
        scheduler = self._make_scheduler()
        race = _make_race()

        scheduler.log_t2_snapshot(race, odds_t3_snapshot={1: 3.5})

        scheduler._odds_collector.collect_t2_snapshot.assert_called_once_with(race.race_id)
        scheduler._late_money_filter.log_last_2min.assert_called()

    def test_submit_bets_calls_pat_voter(self) -> None:
        """submit_bets が PatVoter に投票を依頼"""
        scheduler = self._make_scheduler()
        bets = [_make_bet(1), _make_bet(3)]

        results = scheduler.submit_bets(bets)

        assert len(results) == 1
        scheduler._pat_voter.submit_bets.assert_called_once_with(bets)

    def test_process_race_blocked_by_safety_guard(self) -> None:
        """SafetyGuard が投票不可の場合は process_race が空リストを返す"""
        scheduler = self._make_scheduler()
        scheduler._safety_guard.check.return_value = SafetyCheckResult(
            can_bet=False,
            reason="Daily loss exceeded",
        )

        race = _make_race()
        pending = scheduler.process_race(
            race=race,
            feats={"race_id": ["2024032501010101"], "umaban": [1]},
            bankroll=100000,
            dd_ctrl=MagicMock(),
        )

        assert pending == []
        scheduler._orchestrator.process_race.assert_not_called()

    def test_fetch_race_schedule_empty_day(self) -> None:
        """レースがない日は空リスト"""
        scheduler = self._make_scheduler()
        scheduler._fetcher.fetch_race_cards.return_value = []

        races = scheduler.fetch_race_schedule("2024-01-01")
        assert races == []

    def test_evaluate_model_monitor(self) -> None:
        """レース終了後に ModelMonitor を評価"""
        scheduler = self._make_scheduler()
        mock_monitor = MagicMock()
        mock_monitor.check_performance.return_value = MagicMock(
            needs_attention=False,
            should_retrain=False,
        )

        results = pd.DataFrame(
            {
                "race_id": ["R1"],
                "ev_predicted": [1.0],
                "ev_actual": [1.2],
                "hit": [1],
            }
        )
        report = scheduler.evaluate_model_performance(results, monitor=mock_monitor)

        mock_monitor.check_performance.assert_called_once_with(results)
        assert report.needs_attention is False

    def test_evaluate_model_monitor_with_retrain(self) -> None:
        """再学習が必要な場合に通知"""
        scheduler = self._make_scheduler()
        mock_monitor = MagicMock()
        mock_monitor.check_performance.return_value = MagicMock(
            needs_attention=True,
            should_retrain=True,
            hit_rate=0.05,
            regime="collapsed",
        )
        mock_notifier = MagicMock()

        results = pd.DataFrame(
            {
                "race_id": ["R1"],
                "ev_predicted": [1.0],
                "ev_actual": [0.0],
                "hit": [0],
            }
        )
        scheduler.evaluate_model_performance(
            results,
            monitor=mock_monitor,
            notifier=mock_notifier,
        )

        mock_notifier.send.assert_called()
