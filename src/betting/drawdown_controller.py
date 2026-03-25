"""DD×ROI + EWMA + ヒステリシス + Nベット制限 (Rule 9, Rule 17)"""

from __future__ import annotations

import logging

import numpy as np

from domain.models import DDState
from domain.types import RecoveryState

logger = logging.getLogger(__name__)


class DrawdownController:
    """
    v5.4: DD × Rolling ROI の複合判定によるベットサイズ制御。

    v5.1: ウィンドウを50→150に拡大し、EWMA ハイブリッドを導入。
    v5.3: ヒステリシス付き回復ロジックを追加。
    v5.4: MAX_ADJUSTMENT_PER_DAY → MAX_ADJUSTMENT_PER_N_BETS (20回) に変更。

    回復の3段階:
      NORMAL → REDUCED:     DD悪化時にテーブルに従って削減
      REDUCED → RECOVERING: ROI >= 0.98 かつ DD改善傾向
      RECOVERING → NORMAL:   DD < 5% または 連続回復
    """

    ROLLING_WINDOW: int = 150
    EWMA_ALPHA: float = 0.1

    # v5.3: 回復加速パラメータ
    RECOVERY_INCREMENT: float = 0.05
    RECOVERY_ROI_THRESHOLD: float = 0.98
    RECOVERY_DD_THRESHOLD: float = 0.05
    RECOVERY_MAX_MULTIPLIER: float = 1.00

    # v5.4: 試行回数ベースの過剰適応防止
    MAX_ADJUSTMENT_PER_N_BETS: int = 20
    MAX_ADJUSTMENT_AMOUNT: float = 0.15

    MULTIPLIER_TABLE: list[tuple[float, float, float, float, float]] = [
        # (DD下限, DD上限, ROI下限, ROI上限, 乗数)
        (0.00, 0.10, 0.90, 9.99, 1.00),
        (0.00, 0.10, 0.00, 0.90, 0.75),
        (0.10, 0.15, 0.95, 9.99, 0.80),
        (0.10, 0.15, 0.00, 0.95, 0.50),
        (0.15, 0.20, 0.95, 9.99, 0.60),
        (0.15, 0.20, 0.00, 0.95, 0.30),
        (0.20, 0.25, 0.00, 9.99, 0.15),
        (0.25, 9.99, 0.00, 9.99, 0.00),
    ]

    def __init__(self, peak_bankroll: float) -> None:
        self.peak_bankroll = peak_bankroll
        self.bet_history: list[float] = []
        self._recovery_state = RecoveryState.NORMAL
        self._current_multiplier = 1.0
        self._multiplier_at_window_start = 1.0
        self._bets_in_window = 0

    def update(self, bankroll: float, bet_return: float) -> None:
        """ベット結果を記録してピークを更新"""
        if bankroll > self.peak_bankroll:
            self.peak_bankroll = bankroll
        self.bet_history.append(bet_return)
        if len(self.bet_history) > self.ROLLING_WINDOW * 2:
            self.bet_history.pop(0)
        self._update_recovery_state(bankroll)

    def _update_recovery_state(self, bankroll: float) -> None:
        """v5.3: 回復状態の遷移ロジック"""
        dd = (self.peak_bankroll - bankroll) / self.peak_bankroll
        roi = self._calc_rolling_roi()

        if self._recovery_state == RecoveryState.NORMAL:
            table_mult = self._get_table_multiplier(dd, roi)
            if table_mult < 0.80:
                self._recovery_state = RecoveryState.REDUCED
                self._current_multiplier = table_mult
                self._multiplier_at_window_start = table_mult
                self._bets_in_window = 0
            else:
                self._current_multiplier = table_mult

        elif self._recovery_state == RecoveryState.REDUCED:
            if roi >= self.RECOVERY_ROI_THRESHOLD and dd < 0.15:
                self._recovery_state = RecoveryState.RECOVERING
                self._multiplier_at_window_start = self._current_multiplier
                self._bets_in_window = 0
                logger.info("DD Controller: REDUCED → RECOVERING")
            else:
                self._current_multiplier = self._get_table_multiplier(dd, roi)

        elif self._recovery_state == RecoveryState.RECOVERING:
            self._current_multiplier = min(
                self._current_multiplier + self.RECOVERY_INCREMENT,
                self.RECOVERY_MAX_MULTIPLIER,
            )
            if dd < self.RECOVERY_DD_THRESHOLD:
                self._recovery_state = RecoveryState.NORMAL
                self._current_multiplier = 1.0
                self._multiplier_at_window_start = 1.0
                self._bets_in_window = 0
                logger.info("DD Controller: RECOVERING → NORMAL")
            elif roi < 0.90:
                self._recovery_state = RecoveryState.REDUCED
                self._current_multiplier = self._get_table_multiplier(dd, roi)
                self._multiplier_at_window_start = self._current_multiplier
                self._bets_in_window = 0
                logger.info("DD Controller: RECOVERING → REDUCED")

    def get_state(self, bankroll: float) -> DDState:
        """現在のDD状態を返す"""
        dd = (self.peak_bankroll - bankroll) / self.peak_bankroll
        roi = self._calc_rolling_roi()
        return DDState(
            current_dd=dd,
            rolling_roi=roi,
            n_bets_eval=min(len(self.bet_history), self.ROLLING_WINDOW),
            recovery_state=self._recovery_state,
        )

    def _get_table_multiplier(self, dd: float, roi: float) -> float:
        """乗数テーブルからベース乗数を取得"""
        for dd_lo, dd_hi, roi_lo, roi_hi, mult in self.MULTIPLIER_TABLE:
            if dd_lo <= dd < dd_hi and roi_lo <= roi < roi_hi:
                return mult
        return 0.0

    def _calc_rolling_roi(self) -> float:
        """SMA + EWMA ハイブリッド"""
        recent = self.bet_history[-self.ROLLING_WINDOW :]
        if not recent:
            return 1.0
        if len(recent) < 20:
            return float(np.mean(recent))

        sma = float(np.mean(recent))
        ewma = recent[0]
        for r in recent[1:]:
            ewma = self.EWMA_ALPHA * r + (1 - self.EWMA_ALPHA) * ewma
        return (sma + ewma) / 2.0

    def get_multiplier(self, bankroll: float) -> float:
        """DD状態に基づくベットサイズ乗数を取得"""
        raw_mult = self._current_multiplier

        # v5.4: Nベットごとの変更幅を制限
        self._bets_in_window += 1
        if self._bets_in_window >= self.MAX_ADJUSTMENT_PER_N_BETS:
            self._multiplier_at_window_start = self._current_multiplier
            self._bets_in_window = 0

        max_change = self.MAX_ADJUSTMENT_AMOUNT
        mult = max(
            self._multiplier_at_window_start - max_change,
            min(raw_mult, self._multiplier_at_window_start + max_change),
        )
        self._current_multiplier = mult
        return mult

    def adjust_stake(self, base_stake: float, bankroll: float) -> float:
        """ベットサイズにDD乗数を適用して100円単位で返す"""
        mult = self.get_multiplier(bankroll)
        return max(0.0, float(int((base_stake * mult) // 100) * 100))

    def log_state(self, bankroll: float) -> None:
        """現在のDD状態をログ出力"""
        state = self.get_state(bankroll)
        mult = self._current_multiplier
        logger.info(
            f"DD: {state.current_dd:.1%} | "
            f"Rolling ROI({state.n_bets_eval}bets): {state.rolling_roi:.3f} | "
            f"Multiplier: {mult:.2f} | "
            f"State: {state.recovery_state.value} | "
            f"Peak: ¥{self.peak_bankroll:,.0f} | "
            f"Current: ¥{bankroll:,.0f}"
        )
