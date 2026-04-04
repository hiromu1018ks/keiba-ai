"""PatVoter — JRA-IPAT 投票インタフェース (F-1b)

設計書 §12: PAT自動投票。
JRA-IPAT API との通信は Protocol で抽象化し、
テストでは mock を注入可能にする。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from domain.models import Bet, SafetyCheckResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BetSubmissionResult:
    """投票結果"""

    success: bool
    bet_id: str = ""
    error: str = ""


@runtime_checkable
class PatApiProtocol(Protocol):
    """JRA-IPAT API の抽象プロトコル"""

    def submit(self, bets: list[Bet]) -> list[dict]: ...
    def cancel(self, bet_ids: list[str]) -> bool: ...


@runtime_checkable
class SafetyGuardProtocol(Protocol):
    def check(self, bankroll: float) -> SafetyCheckResult: ...


class PatVoter:
    """JRA-IPAT への投票を管理する

    Args:
        api: JRA-IPAT API プロトコル実装
        safety_guard: SafetyGuard インスタンス（省略時はチェックなし）
    """

    def __init__(
        self,
        api: PatApiProtocol,
        safety_guard: Optional[SafetyGuardProtocol] = None,
    ) -> None:
        self.api = api
        self.safety_guard = safety_guard

    def submit_bets(self, bets: list[Bet], bankroll: float = 0) -> list[BetSubmissionResult]:
        """ベットを JRA-IPAT に投票

        SafetyGuard が設定されている場合、チェック後に投票する。

        Args:
            bets: 投票候補リスト
            bankroll: 現在の資金（SafetyGuard チェック用）

        Returns:
            BetSubmissionResult リスト
        """
        if not bets:
            return []

        # SafetyGuard チェック
        if self.safety_guard is not None:
            check = self.safety_guard.check(bankroll=bankroll)
            if not check.can_bet:
                logger.warning(f"Bets blocked by SafetyGuard: {check.reason}")
                return []

        # API に投票
        raw_results = self.api.submit(bets)

        results: list[BetSubmissionResult] = []
        for raw in raw_results:
            results.append(
                BetSubmissionResult(
                    success=raw.get("success", False),
                    bet_id=raw.get("bet_id", ""),
                    error=raw.get("error", ""),
                )
            )

        n_success = sum(1 for r in results if r.success)
        logger.info(f"Submitted {n_success}/{len(results)} bets")

        return results

    def cancel_bets(self, bet_ids: list[str]) -> bool:
        """投票を取消

        Args:
            bet_ids: 取消する投票IDリスト

        Returns:
            成功した場合 True
        """
        if not bet_ids:
            return True

        result = self.api.cancel(bet_ids)
        if result:
            logger.info(f"Cancelled {len(bet_ids)} bets")
        else:
            logger.warning(f"Failed to cancel bets: {bet_ids}")
        return result
