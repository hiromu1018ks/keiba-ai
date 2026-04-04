"""RaceScheduler — レース日のタスクスケジューリング (F-2)

設計書 §8 / §12:
  - t-10min: process_race() → ベット候補生成
  - t-3min:  finalize_bets() → 最終キャンセルチェック + 投票
  - t-2min:  log_t2_snapshot() → ログのみ
  - SafetyGuard チェック → 全体の投票可否判定
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

from domain.models import Bet, Race, SafetyCheckResult

if TYPE_CHECKING:
    from betting.orchestrator import DrawdownControllerProtocol

logger = logging.getLogger(__name__)


@runtime_checkable
class OrchestratorProtocol(Protocol):
    def process_race(
        self,
        race: Race,
        feats: dict,
        bankroll: float,
        dd_ctrl: object,
    ) -> list[Bet]: ...
    def finalize_bets(
        self,
        race: Race,
        pending_bets: list[Bet],
        odds_t3_snapshot: dict[int, float],
        odds_t10_snapshot: dict[int, float],
    ) -> list[Bet]: ...


@runtime_checkable
class OddsCollectorProtocol(Protocol):
    def collect_t3_snapshot(self, race_id: str) -> dict[int, float]: ...
    def collect_t2_snapshot(self, race_id: str) -> dict[int, float]: ...


@runtime_checkable
class PatVoterProtocol(Protocol):
    def submit_bets(self, bets: list[Bet], bankroll: float = 0) -> list[object]: ...


@runtime_checkable
class SafetyGuardProtocol(Protocol):
    def check(self, bankroll: float) -> SafetyCheckResult: ...


@runtime_checkable
class LateMoneyFilterProtocol(Protocol):
    def log_last_2min(
        self,
        horse_no: int,
        odds_t3: float,
        odds_t2: float,
    ) -> None: ...


@runtime_checkable
class FetcherProtocol(Protocol):
    def fetch_race_cards(self, date: str) -> list[Race]: ...
    def fetch_odds_snapshot(self, race_id: str) -> dict[int, float]: ...


class RaceScheduler:
    """レース日の全タスクを統括するスケジューラ

    Args:
        orchestrator: BettingOrchestrator インスタンス
        odds_collector: OddsCollector インスタンス
        pat_voter: PatVoter インスタンス
        safety_guard: SafetyGuard インスタンス
        late_money_filter: LateMoneyFilter インスタンス
        fetcher: JVLinkFetcher インスタンス
    """

    def __init__(
        self,
        orchestrator: OrchestratorProtocol,
        odds_collector: OddsCollectorProtocol,
        pat_voter: PatVoterProtocol,
        safety_guard: SafetyGuardProtocol,
        late_money_filter: LateMoneyFilterProtocol,
        fetcher: FetcherProtocol,
    ) -> None:
        self._orchestrator = orchestrator
        self._odds_collector = odds_collector
        self._pat_voter = pat_voter
        self._safety_guard = safety_guard
        self._late_money_filter = late_money_filter
        self._fetcher = fetcher

    def fetch_race_schedule(self, date: str) -> list[Race]:
        """指定日のレーススケジュールを取得

        Args:
            date: 日付 (YYYY-MM-DD)

        Returns:
            Race リスト
        """
        races = self._fetcher.fetch_race_cards(date)
        logger.info(f"[{date}] {len(races)} races scheduled")
        return races

    def process_race(
        self,
        race: Race,
        feats: dict,
        bankroll: float,
        dd_ctrl: DrawdownControllerProtocol,
    ) -> list[Bet]:
        """t-10min: ベット候補生成

        SafetyGuard チェック → orchestrator.process_race() を実行。

        Args:
            race: レース情報
            feats: モデル推論済み特徴量
            bankroll: 現在の資金
            dd_ctrl: DrawdownController インスタンス

        Returns:
            pending_bets: 保留中ベットリスト
        """
        # SafetyGuard チェック
        check = self._safety_guard.check(bankroll)
        if not check.can_bet:
            logger.warning(f"[{race.race_id}] Bets blocked: {check.reason}")
            return []

        return self._orchestrator.process_race(race, feats, bankroll, dd_ctrl)

    def finalize_bets(
        self,
        race: Race,
        pending_bets: list[Bet],
        odds_t10_snapshot: dict[int, float],
    ) -> list[Bet]:
        """t-3min: 最終キャンセルチェック

        設計書 §12 ステップ ⑫ / §8:
        1. t-3min オッズスナップショットを取得
        2. orchestrator.finalize_bets() でキャンセル判定
        """
        if not pending_bets:
            return []

        # t-3min スナップショット取得
        odds_t3 = self._odds_collector.collect_t3_snapshot(race.race_id)

        # orchestrator の finalize_bets に委譲
        approved = self._orchestrator.finalize_bets(
            race=race,
            pending_bets=pending_bets,
            odds_t3_snapshot=odds_t3,
            odds_t10_snapshot=odds_t10_snapshot,
        )

        logger.info(
            f"[{race.race_id}] Finalized: {len(approved)}/{len(pending_bets)} bets approved"
        )
        return approved

    def log_t2_snapshot(
        self,
        race: Race,
        odds_t3_snapshot: dict[int, float],
    ) -> None:
        """t-2min: オッズスナップショットをログに記録

        設計書 §8: 判定には使わない。将来のチューニングデータ。
        """
        odds_t2 = self._odds_collector.collect_t2_snapshot(race.race_id)

        for horse_no, odds_t3 in odds_t3_snapshot.items():
            odds_t2_val = odds_t2.get(horse_no, 0)
            if odds_t2_val > 0:
                self._late_money_filter.log_last_2min(horse_no, odds_t3, odds_t2_val)

    def submit_bets(self, bets: list[Bet]) -> list[object]:
        """PatVoter に投票を依頼

        Args:
            bets: 投票対象の Bet リスト

        Returns:
            BetSubmissionResult リスト
        """
        if not bets:
            return []

        return self._pat_voter.submit_bets(bets)

    def evaluate_model_performance(
        self,
        results: object,
        monitor: object,
        notifier: Optional[object] = None,
    ) -> object:
        """レース終了後にモデルパフォーマンスを評価

        Args:
            results: 直近のパフォーマンスデータ (DataFrame)
            monitor: ModelMonitor インスタンス
            notifier: Notifier インスタンス（省略可能）

        Returns:
            PerformanceReport
        """
        report = monitor.check_performance(results)

        if report.needs_attention and notifier is not None:
            notifier.send(
                f"Model needs attention: hit_rate={report.hit_rate:.2%}, regime={report.regime}",
                level="warning",
            )

        return report
