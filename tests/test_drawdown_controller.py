"""src/betting/drawdown_controller.py のテスト"""

from __future__ import annotations

import pytest

from betting.drawdown_controller import DrawdownController
from domain.types import RecoveryState


@pytest.fixture
def ctrl() -> DrawdownController:
    return DrawdownController(peak_bankroll=100000)


class TestDrawdownController:
    def test_initial_multiplier_is_1(self, ctrl: DrawdownController) -> None:
        """初期状態の乗数は1.0"""
        assert ctrl.get_multiplier(100000) == 1.0

    def test_multiplier_reduces_when_dd_increases(self, ctrl: DrawdownController) -> None:
        """DD悪化で乗数が下がる"""
        ctrl.update(90000, 0.5)  # DD=10%, ROI低
        mult = ctrl.get_multiplier(90000)
        assert mult < 1.0

    def test_multiplier_zero_at_extreme_dd(self, ctrl: DrawdownController) -> None:
        """DD 25%超で乗数=0（ベット停止）"""
        ctrl.update(70000, 0.5)  # DD=30%
        mult = ctrl.get_multiplier(70000)
        assert mult == 0.0

    def test_adjust_stake_applies_multiplier(self, ctrl: DrawdownController) -> None:
        """adjust_stake が乗数を適用する"""
        ctrl._current_multiplier = 0.50
        ctrl._multiplier_at_window_start = 0.50
        ctrl._bets_in_window = 0
        adjusted = ctrl.adjust_stake(1000, 90000)
        assert adjusted == 500

    def test_adjust_stake_rounds_to_100(self, ctrl: DrawdownController) -> None:
        """adjust_stake は100円単位"""
        ctrl._current_multiplier = 0.75
        ctrl._multiplier_at_window_start = 0.75
        ctrl._bets_in_window = 0
        adjusted = ctrl.adjust_stake(330, 90000)
        assert adjusted % 100 == 0

    def test_hysteresis_recovery_normal_to_reduced(self, ctrl: DrawdownController) -> None:
        """NORMAL → REDUCED: DD悪化時に遷移"""
        ctrl._recovery_state = RecoveryState.NORMAL
        ctrl.update(85000, 0.80)  # DD=15%, ROI悪化
        state = ctrl.get_state(85000)
        assert state.recovery_state == RecoveryState.REDUCED

    def test_hysteresis_recovery_reduced_to_recovering(self, ctrl: DrawdownController) -> None:
        """REDUCED → RECOVERING: ROI回復 + DD改善で遷移"""
        ctrl._recovery_state = RecoveryState.REDUCED
        ctrl._current_multiplier = 0.50
        ctrl._multiplier_at_window_start = 0.50
        ctrl._bets_in_window = 0
        # ROI >= 0.98 の状態をシミュレート
        for i in range(25):
            ctrl.update(88000 + i * 100, 1.05)  # DD改善 + ROI回復
        state = ctrl.get_state(90500)
        assert state.recovery_state == RecoveryState.RECOVERING

    def test_hysteresis_recovery_recovering_to_normal(self, ctrl: DrawdownController) -> None:
        """RECOVERING → NORMAL: DD < 5% で復帰"""
        ctrl._recovery_state = RecoveryState.RECOVERING
        ctrl._current_multiplier = 0.80
        ctrl._multiplier_at_window_start = 0.50
        ctrl._bets_in_window = 0
        # DD < 5% の状態に
        for i in range(20):
            ctrl.update(97000 + i * 50, 1.05)
        state = ctrl.get_state(98000)
        assert state.recovery_state == RecoveryState.NORMAL

    def test_n_bet_limit_prevents_rapid_change(self, ctrl: DrawdownController) -> None:
        """Nベット変更幅制限: 20ベットウィンドウ内でmax_change=0.15"""
        ctrl._recovery_state = RecoveryState.RECOVERING
        ctrl._current_multiplier = 0.50
        ctrl._multiplier_at_window_start = 0.50
        ctrl._bets_in_window = 0
        # recovering: +0.05/bet → 20ベットで+1.0だが、capで0.15に制限
        for _ in range(19):
            ctrl.get_multiplier(90000)
        mult = ctrl.get_multiplier(90000)
        assert mult <= 0.65  # 0.50 + 0.15 = 0.65

    def test_n_bet_window_resets(self, ctrl: DrawdownController) -> None:
        """Nベットウィンドウがリセットされる"""
        ctrl._multiplier_at_window_start = 0.50
        ctrl._bets_in_window = 19
        ctrl.get_multiplier(90000)  # 20回目 → リセット
        assert ctrl._bets_in_window == 0

    def test_rolling_roi_sma_ewma_hybrid(self, ctrl: DrawdownController) -> None:
        """Rolling ROI は SMA + EWMA のハイブリッド"""
        # 20回以上のベット履歴が必要
        for i in range(150):
            ctrl.update(95000, 1.02)
        state = ctrl.get_state(95000)
        assert state.rolling_roi > 0
        assert state.n_bets_eval >= 20

    def test_get_state_returns_ddstate(self, ctrl: DrawdownController) -> None:
        """get_state が正しいDDStateを返す"""
        ctrl.update(90000, 0.95)
        state = ctrl.get_state(90000)
        assert state.current_dd == pytest.approx(0.10, abs=0.01)
        assert state.recovery_state in list(RecoveryState)

    def test_peak_bankroll_updates_on_new_high(self, ctrl: DrawdownController) -> None:
        """資金が新高値を更新するとpeakも更新"""
        ctrl.update(110000, 1.0)
        assert ctrl.peak_bankroll == 110000

    def test_recovering_to_reduced_on_roi_deterioration(self, ctrl: DrawdownController) -> None:
        """RECOVERING → REDUCED: ROI < 0.90 で逆戻り"""
        ctrl._recovery_state = RecoveryState.RECOVERING
        ctrl._current_multiplier = 0.80
        ctrl._multiplier_at_window_start = 0.50
        ctrl._bets_in_window = 0
        # ROI < 0.90 をシミュレート
        for i in range(25):
            ctrl.update(90000, 0.85)
        state = ctrl.get_state(90000)
        assert state.recovery_state == RecoveryState.REDUCED

    def test_adjust_stake_zero_when_multiplier_zero(self, ctrl: DrawdownController) -> None:
        """DD極大時 adjust_stake が 0 を返す"""
        ctrl.update(70000, 0.5)  # DD=30% → multiplier=0
        result = ctrl.adjust_stake(1000, 70000)
        assert result == 0

    def test_log_state_does_not_raise(self, ctrl: DrawdownController) -> None:
        """log_state が例外を送出しない"""
        ctrl.update(95000, 1.02)
        ctrl.log_state(95000)  # Should not raise
