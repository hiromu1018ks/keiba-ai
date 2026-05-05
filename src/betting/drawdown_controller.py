"""DD%ベース3段階制御 + ヒステリシス + コンストラクタ注入 (v5.5 redesign)"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from domain.models import DDState
from domain.types import RecoveryState

logger = logging.getLogger(__name__)


@dataclass
class DDConfig:
    """DrawdownController の全パラメータを保持する設定 dataclass。

    __post_init__ で閾値整合性を検証し、不正な設定値を拒否する。
    """

    rolling_window: int = 400
    dd_threshold_1: float = 0.10  # NORMAL -> REDUCED 境界
    dd_threshold_2: float = 0.20  # REDUCED -> STOP 境界
    multiplier_normal: float = 1.0
    multiplier_reduced: float = 0.50
    multiplier_stop: float = 0.0
    min_stay_races: int = 10  # ヒステリシス最低滞在レース数
    max_adjustment_per_n_bets: int = 20
    max_adjustment_amount: float = 0.15

    def __post_init__(self) -> None:
        """T-13-01: 閾値整合性検証"""
        if self.dd_threshold_2 <= self.dd_threshold_1:
            raise ValueError(
                f"dd_threshold_2 ({self.dd_threshold_2}) must be > "
                f"dd_threshold_1 ({self.dd_threshold_1})"
            )
        if self.multiplier_normal <= 0:
            raise ValueError(f"multiplier_normal must be > 0, got {self.multiplier_normal}")
        if not (0 <= self.multiplier_reduced < self.multiplier_normal):
            raise ValueError(
                f"multiplier_reduced must be in [0, multiplier_normal), "
                f"got {self.multiplier_reduced}"
            )
        if self.multiplier_stop != 0.0:
            # STOP はベット停止を意味する。0.0以外の値は意図確認
            logger.warning(
                f"multiplier_stop={self.multiplier_stop} (expected 0.0 for bet stop)"
            )
        if self.rolling_window < 1:
            raise ValueError(f"rolling_window must be >= 1, got {self.rolling_window}")
        if self.min_stay_races < 1:
            raise ValueError(f"min_stay_races must be >= 1, got {self.min_stay_races}")


class DrawdownController:
    """DD%のみの3段階制御: NORMAL / REDUCED / STOP。

    v5.5: ROI依存を完全に除去。DD%閾値のみで状態判定。
    ヒステリシス(min_stay_races)で発振を防止。
    段階的リカバリ: STOP -> REDUCED -> NORMAL (即時復帰なし)。
    """

    def __init__(self, peak_bankroll: float, cfg: DDConfig | None = None) -> None:
        self.cfg = cfg or DDConfig()
        self.peak_bankroll = peak_bankroll
        self._state = RecoveryState.NORMAL
        self._races_in_state = 0
        self._current_multiplier = 1.0
        self._multiplier_at_window_start = 1.0
        self._bets_in_window = 0
        self.n_bets = 0

    def update(self, bankroll: float) -> None:
        """バンクロールを更新し、DD%に基づいて状態遷移を判定する。"""
        if bankroll > self.peak_bankroll:
            self.peak_bankroll = bankroll
        dd = (self.peak_bankroll - bankroll) / self.peak_bankroll
        self.n_bets += 1
        self._transition(dd)

    def _transition(self, dd: float) -> None:
        """ヒステリシス付き状態遷移ロジック。"""
        target = self._determine_target_state(dd)
        if target != self._state:
            # ヒステリシス: 最低滞在レース数未満なら遷移しない
            if self._races_in_state < self.cfg.min_stay_races:
                return
            # 段階的リカバリ: STOP -> NORMAL への即時遷移を禁止
            if self._state == RecoveryState.STOP and target == RecoveryState.NORMAL:
                target = RecoveryState.REDUCED  # 強制的にREDUCED経由
            old = self._state
            self._state = target
            self._races_in_state = 0
            self._update_multiplier()
            logger.info(f"DD: {old.value} -> {target.value}")
        else:
            self._races_in_state += 1

    def _determine_target_state(self, dd: float) -> RecoveryState:
        """DD%閾値に基づいて目標状態を決定する。"""
        if dd >= self.cfg.dd_threshold_2:
            return RecoveryState.STOP
        elif dd >= self.cfg.dd_threshold_1:
            return RecoveryState.REDUCED
        return RecoveryState.NORMAL

    def _update_multiplier(self) -> None:
        """状態に対応する乗数を設定する。"""
        if self._state == RecoveryState.NORMAL:
            self._current_multiplier = self.cfg.multiplier_normal
        elif self._state == RecoveryState.REDUCED:
            self._current_multiplier = self.cfg.multiplier_reduced
        else:
            self._current_multiplier = self.cfg.multiplier_stop
        self._multiplier_at_window_start = self._current_multiplier
        self._bets_in_window = 0

    def get_state(self, bankroll: float) -> DDState:
        """現在のDD状態を返す。"""
        dd = (self.peak_bankroll - bankroll) / self.peak_bankroll
        return DDState(
            current_dd=dd,
            n_bets_eval=self.n_bets,
            recovery_state=self._state,
        )

    def get_multiplier(self, bankroll: float) -> float:
        """DD状態に基づくベットサイズ乗数を取得（変更幅制限付き）。"""
        raw_mult = self._current_multiplier

        # Nベットごとの変更幅を制限
        self._bets_in_window += 1
        if self._bets_in_window >= self.cfg.max_adjustment_per_n_bets:
            self._multiplier_at_window_start = self._current_multiplier
            self._bets_in_window = 0

        max_change = self.cfg.max_adjustment_amount
        mult = max(
            self._multiplier_at_window_start - max_change,
            min(raw_mult, self._multiplier_at_window_start + max_change),
        )
        self._current_multiplier = mult
        return mult

    def adjust_stake(self, base_stake: float, bankroll: float) -> float:
        """ベットサイズにDD乗数を適用して100円単位で返す。"""
        mult = self.get_multiplier(bankroll)
        return max(0.0, float(int((base_stake * mult) // 100) * 100))

    def log_state(self, bankroll: float) -> None:
        """現在のDD状態をログ出力。"""
        state = self.get_state(bankroll)
        mult = self._current_multiplier
        logger.info(
            f"DD: {state.current_dd:.1%} | "
            f"Multiplier: {mult:.2f} | "
            f"State: {state.recovery_state.value} | "
            f"Peak: {self.peak_bankroll:,.0f} | "
            f"Current: {bankroll:,.0f}"
        )
