"""src/automation/safety_guard.py のテスト"""

from __future__ import annotations

from domain.models import SafetyConfig


class TestSafetyGuard:
    def test_check_passes_with_sufficient_bankroll(self) -> None:
        """十分な資金があれば can_bet=True"""
        from automation.safety_guard import SafetyGuard

        guard = SafetyGuard(SafetyConfig(min_bankroll=10000))
        result = guard.check(bankroll=100000)
        assert result.can_bet is True
        assert result.reason == ""

    def test_check_fails_below_min_bankroll(self) -> None:
        """最低資金を下回れば can_bet=False"""
        from automation.safety_guard import SafetyGuard

        guard = SafetyGuard(SafetyConfig(min_bankroll=10000))
        result = guard.check(bankroll=5000)
        assert result.can_bet is False
        assert "bankroll" in result.reason.lower()

    def test_check_fails_daily_loss_exceeded(self) -> None:
        """日次損失上限を超過すれば can_bet=False"""
        from automation.safety_guard import SafetyGuard

        guard = SafetyGuard(SafetyConfig(max_daily_loss=5000))
        guard.record_daily_pnl(-5000)
        result = guard.check(bankroll=100000)
        assert result.can_bet is False
        assert "daily" in result.reason.lower()

    def test_check_fails_emergency_stop(self) -> None:
        """緊急停止フラグが立っていれば can_bet=False"""
        from automation.safety_guard import SafetyGuard

        guard = SafetyGuard(SafetyConfig())
        guard.activate_emergency_stop("manual intervention")
        result = guard.check(bankroll=100000)
        assert result.can_bet is False
        assert "emergency" in result.reason.lower()

    def test_reset_clears_daily_pnl(self) -> None:
        """reset で日次損失をリセット"""
        from automation.safety_guard import SafetyGuard

        guard = SafetyGuard(SafetyConfig(max_daily_loss=5000))
        guard.record_daily_pnl(-3000)
        guard.reset()
        result = guard.check(bankroll=100000)
        assert result.can_bet is True

    def test_reset_clears_emergency_stop(self) -> None:
        """reset で緊急停止を解除"""
        from automation.safety_guard import SafetyGuard

        guard = SafetyGuard(SafetyConfig())
        guard.activate_emergency_stop("test")
        guard.reset()
        result = guard.check(bankroll=100000)
        assert result.can_bet is True

    def test_record_daily_pnl_accumulates(self) -> None:
        """日次損失は累積する"""
        from automation.safety_guard import SafetyGuard

        guard = SafetyGuard(SafetyConfig(max_daily_loss=5000))
        guard.record_daily_pnl(-1000)
        guard.record_daily_pnl(-2000)
        guard.record_daily_pnl(-3000)
        # 累計 -6000 > max_daily_loss -5000
        result = guard.check(bankroll=100000)
        assert result.can_bet is False

    def test_check_considers_consecutive_losses(self) -> None:
        """連続敗北数が閾値を超えると停止"""
        from automation.safety_guard import SafetyGuard

        guard = SafetyGuard(SafetyConfig(max_consecutive_losses=5))
        for _ in range(5):
            guard.record_result(lost=True)
        result = guard.check(bankroll=100000)
        assert result.can_bet is False
        assert "consecutive" in result.reason.lower()

    def test_win_resets_consecutive_losses(self) -> None:
        """勝利で連続敗北カウンタをリセット"""
        from automation.safety_guard import SafetyGuard

        guard = SafetyGuard(SafetyConfig(max_consecutive_losses=3))
        for _ in range(3):
            guard.record_result(lost=True)
        assert guard.check(bankroll=100000).can_bet is False

        guard.record_result(lost=False)
        assert guard.check(bankroll=100000).can_bet is True
