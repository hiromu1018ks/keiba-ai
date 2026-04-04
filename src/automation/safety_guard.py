"""SafetyGuard — 資金/損失/緊急停止の安全チェック (F-1a)

設計書 §12 ステップ ⑪:
  bankroll が最低ラインを下回る、日次損失が上限を超える、
  緊急停止フラグが立っている場合、投票を禁止する。
"""

from __future__ import annotations

import logging

from domain.models import SafetyCheckResult, SafetyConfig

logger = logging.getLogger(__name__)


class SafetyGuard:
    """投票の安全性をチェックするガード

    Attributes:
        config: 安全設定
        _daily_pnl: 当日の累積損益
        _weekly_pnl: 当週の累積損益
        _consecutive_losses: 連続敗北数
        _emergency_stop: 緊急停止フラグ
        _emergency_reason: 緊急停止の理由
    """

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._consecutive_losses: int = 0
        self._emergency_stop: bool = False
        self._emergency_reason: str = ""

    def check(self, bankroll: float) -> SafetyCheckResult:
        """現在の状態で投票可能かチェック

        Args:
            bankroll: 現在の資金

        Returns:
            SafetyCheckResult (can_bet + reason)
        """
        # 1. 緊急停止
        if self._emergency_stop:
            return SafetyCheckResult(
                can_bet=False,
                reason=f"Emergency stop: {self._emergency_reason}",
            )

        # 2. 最低資金チェック
        if bankroll < self.config.min_bankroll:
            return SafetyCheckResult(
                can_bet=False,
                reason=f"Bankroll {bankroll:.0f} < minimum {self.config.min_bankroll:.0f}",
            )

        # 3. 日次損失チェック
        if self._daily_pnl <= -self.config.max_daily_loss:
            return SafetyCheckResult(
                can_bet=False,
                reason=(
                    f"Daily loss {abs(self._daily_pnl):.0f} >= max {self.config.max_daily_loss:.0f}"
                ),
            )

        # 4. 週次損失チェック
        if self._weekly_pnl <= -self.config.max_weekly_loss:
            return SafetyCheckResult(
                can_bet=False,
                reason=(
                    f"Weekly loss {abs(self._weekly_pnl):.0f} "
                    f">= max {self.config.max_weekly_loss:.0f}"
                ),
            )

        # 5. 連続敗北チェック
        if self._consecutive_losses >= self.config.max_consecutive_losses:
            return SafetyCheckResult(
                can_bet=False,
                reason=(
                    f"Consecutive losses {self._consecutive_losses} "
                    f">= max {self.config.max_consecutive_losses}"
                ),
            )

        return SafetyCheckResult(can_bet=True)

    def record_daily_pnl(self, pnl: float) -> None:
        """日次損益を記録

        Args:
            pnl: 損益 (負 = 損失)
        """
        self._daily_pnl += pnl
        self._weekly_pnl += pnl
        logger.info(f"Daily PnL updated: {pnl:+.0f} (total: {self._daily_pnl:+.0f})")

    def record_result(self, lost: bool) -> None:
        """勝敗を記録し、連続敗北カウンタを更新

        Args:
            lost: 敗北した場合 True
        """
        if lost:
            self._consecutive_losses += 1
        else:
            if self._consecutive_losses > 0:
                logger.info(f"Consecutive loss streak broken at {self._consecutive_losses}")
            self._consecutive_losses = 0

    def activate_emergency_stop(self, reason: str) -> None:
        """緊急停止を発動

        Args:
            reason: 停止理由
        """
        self._emergency_stop = True
        self._emergency_reason = reason
        logger.warning(f"EMERGENCY STOP activated: {reason}")

    def reset(self) -> None:
        """日次/週次の状態をリセット（毎日実行）"""
        self._daily_pnl = 0.0
        self._weekly_pnl = 0.0
        self._consecutive_losses = 0
        self._emergency_stop = False
        self._emergency_reason = ""
        logger.info("SafetyGuard reset")
