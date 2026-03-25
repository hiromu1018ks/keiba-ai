"""t-3min判定 + t-2minログ (Rule 8, Rule 14)"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import pandas as pd

from domain.models import Bet

logger = logging.getLogger(__name__)


class LastMinuteSignal(Enum):
    """直前オッズ変動シグナル"""

    NO_ACTION = "no_action"
    CANCEL = "cancel"
    ADD_CANDIDATE = "add_candidate"
    UNKNOWN = "unknown"


class LateMoneyFilter:
    """
    v5.0: t-2min の急変を独立トリガーとして追加
    v5.1: t-3min で判定、t-2min はログのみ（実務安全マージン）

    2つの機能：
    A) キャンセルトリガー: t-3min に急落（オッズ低下）を検知したら投票取消
    B) 追加投票トリガー: t-3min に急騰（オッズ上昇）を検知したら未投票馬を追加検討

    設計書 §8.2 参照。
    """

    CANCEL_DROP_THRESHOLD: float = 0.25  # 25%以上の急落 → CANCEL
    ADD_RISE_THRESHOLD: float = 0.30  # 30%以上の急騰 → ADD_CANDIDATE
    LOG_THRESHOLD: float = 0.20  # t-3min→t-2min で20%以上変動 → ログ

    def check_last_3min(
        self,
        horse_no: int,
        odds_t10: float,
        odds_t3: float,
    ) -> LastMinuteSignal:
        """
        発走3分前のオッズを確認してシグナルを返す。

        Args:
            horse_no: 馬番
            odds_t10: t-10minのオッズ
            odds_t3: t-3minのオッズ

        Returns:
            LastMinuteSignal シグナル
        """
        if odds_t10 <= 0 or odds_t3 <= 0:
            return LastMinuteSignal.UNKNOWN

        change_rate = (odds_t10 - odds_t3) / odds_t10

        if change_rate >= self.CANCEL_DROP_THRESHOLD:
            return LastMinuteSignal.CANCEL

        if change_rate <= -self.ADD_RISE_THRESHOLD:
            return LastMinuteSignal.ADD_CANDIDATE

        return LastMinuteSignal.NO_ACTION

    def log_last_2min(
        self,
        horse_no: int,
        odds_t3: float,
        odds_t2: float,
    ) -> None:
        """
        発走2分前のオッズをログに記録する（判定には使わない）。

        将来のしきい値チューニングに使用するデータ収集。
        """
        if odds_t3 <= 0 or odds_t2 <= 0:
            return

        change_rate = abs(odds_t3 - odds_t2) / odds_t3
        if change_rate >= self.LOG_THRESHOLD:
            logger.info(
                f"[LOG ONLY] horse={horse_no} "
                f"odds: {odds_t3:.1f} → {odds_t2:.1f} "
                f"(change: {change_rate:.0%})"
            )

    def process_last_minute(
        self,
        pending_bets: list[Bet],
        odds_t3_snapshot: dict[int, float],
        odds_t10_snapshot: dict[int, float],
        stage2_predictions: Optional[pd.DataFrame],
    ) -> tuple[list[Bet], list[Bet]]:
        """
        発走3分前に全ての保留中ベットを再チェックする。

        Args:
            pending_bets: 保留中ベット
            odds_t3_snapshot: horse_no → t-3minオッズ
            odds_t10_snapshot: horse_no → t-10minオッズ
            stage2_predictions: 未使用（将来拡張用）

        Returns:
            (approved_bets, cancelled_bets) タプル
        """
        approved: list[Bet] = []
        cancelled: list[Bet] = []

        for bet in pending_bets:
            odds_t10 = odds_t10_snapshot.get(bet.umaban, 0)
            odds_t3 = odds_t3_snapshot.get(bet.umaban, 0)
            signal = self.check_last_3min(bet.umaban, odds_t10, odds_t3)

            if signal == LastMinuteSignal.CANCEL:
                cancelled.append(bet)
                logger.warning(
                    f"CANCEL: race={bet.race_id} horse={bet.umaban} "
                    f"odds: {odds_t10:.1f} → {odds_t3:.1f} "
                    f"(drop: {(odds_t10 - odds_t3) / odds_t10:.0%})"
                )
            else:
                approved.append(bet)

        return approved, cancelled
