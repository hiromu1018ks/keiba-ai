"""src/betting/drawdown_controller.py のテスト — DD%ベース3段階制御"""

from __future__ import annotations

import pytest

from betting.drawdown_controller import DDConfig, DrawdownController
from domain.types import RecoveryState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ctrl() -> DrawdownController:
    return DrawdownController(peak_bankroll=100_000)


@pytest.fixture
def ctrl_short_stay() -> DrawdownController:
    """ヒステリシス min_stay_races=2 の短縮コントローラー"""
    cfg = DDConfig(min_stay_races=2)
    return DrawdownController(peak_bankroll=100_000, cfg=cfg)


# ---------------------------------------------------------------------------
# TestDDConfig — __post_init__ 検証
# ---------------------------------------------------------------------------


class TestDDConfig:
    def test_default_config_values(self) -> None:
        cfg = DDConfig()
        assert cfg.rolling_window == 400
        assert cfg.dd_threshold_1 == 0.10
        assert cfg.dd_threshold_2 == 0.20
        assert cfg.multiplier_normal == 1.0
        assert cfg.multiplier_reduced == 0.50
        assert cfg.multiplier_stop == 0.0
        assert cfg.min_stay_races == 10
        assert cfg.max_adjustment_per_n_bets == 20
        assert cfg.max_adjustment_amount == 0.15

    def test_custom_config(self) -> None:
        cfg = DDConfig(
            rolling_window=500,
            dd_threshold_1=0.15,
            dd_threshold_2=0.30,
            multiplier_normal=1.0,
            multiplier_reduced=0.30,
            multiplier_stop=0.0,
            min_stay_races=5,
        )
        assert cfg.rolling_window == 500
        assert cfg.dd_threshold_1 == 0.15
        assert cfg.multiplier_reduced == 0.30

    def test_post_init_rejects_threshold_2_leq_threshold_1(self) -> None:
        with pytest.raises(ValueError, match="dd_threshold_2"):
            DDConfig(dd_threshold_1=0.20, dd_threshold_2=0.10)

    def test_post_init_rejects_threshold_2_equal_threshold_1(self) -> None:
        with pytest.raises(ValueError, match="dd_threshold_2"):
            DDConfig(dd_threshold_1=0.15, dd_threshold_2=0.15)

    def test_post_init_rejects_negative_multiplier(self) -> None:
        with pytest.raises(ValueError, match="multiplier_normal"):
            DDConfig(multiplier_normal=-1)

    def test_post_init_rejects_zero_multiplier(self) -> None:
        with pytest.raises(ValueError, match="multiplier_normal"):
            DDConfig(multiplier_normal=0)

    def test_post_init_rejects_reduced_geq_normal(self) -> None:
        with pytest.raises(ValueError, match="multiplier_reduced"):
            DDConfig(multiplier_reduced=1.5)

    def test_post_init_rejects_negative_reduced(self) -> None:
        with pytest.raises(ValueError, match="multiplier_reduced"):
            DDConfig(multiplier_reduced=-0.1)

    def test_post_init_rejects_zero_rolling_window(self) -> None:
        with pytest.raises(ValueError, match="rolling_window"):
            DDConfig(rolling_window=0)

    def test_post_init_rejects_zero_min_stay(self) -> None:
        with pytest.raises(ValueError, match="min_stay_races"):
            DDConfig(min_stay_races=0)


# ---------------------------------------------------------------------------
# TestDrawdownControllerCore — 状態遷移
# ---------------------------------------------------------------------------


class TestDrawdownControllerInit:
    def test_init_rejects_zero_peak_bankroll(self) -> None:
        with pytest.raises(ValueError, match="peak_bankroll"):
            DrawdownController(peak_bankroll=0)

    def test_init_rejects_negative_peak_bankroll(self) -> None:
        with pytest.raises(ValueError, match="peak_bankroll"):
            DrawdownController(peak_bankroll=-100)


class TestDrawdownControllerCore:
    def test_initial_state_is_normal(self, ctrl: DrawdownController) -> None:
        assert ctrl._state == RecoveryState.NORMAL

    def test_initial_multiplier_is_1(self, ctrl: DrawdownController) -> None:
        assert ctrl.get_multiplier(100_000) == 1.0

    def test_update_tracks_peak_bankroll(self, ctrl: DrawdownController) -> None:
        ctrl.update(110_000)
        assert ctrl.peak_bankroll == 110_000

    def test_dd_calculation(self, ctrl: DrawdownController) -> None:
        ctrl.update(90_000)
        state = ctrl.get_state(90_000)
        assert state.current_dd == pytest.approx(0.10, abs=0.001)

    def test_normal_to_reduced_transition(self, ctrl_short_stay: DrawdownController) -> None:
        ctrl = ctrl_short_stay
        # DD=10.5% で dd_threshold_1=0.10 に到達 -> target=REDUCED
        # min_stay_races=2: _races_in_state が 0→1→2 と増え、3回目で遷移可能
        ctrl.update(89_500)  # stay: 0→1 (blocked)
        assert ctrl._state == RecoveryState.NORMAL
        ctrl.update(89_500)  # stay: 1→2 (blocked)
        assert ctrl._state == RecoveryState.NORMAL
        ctrl.update(89_500)  # stay: 2 >= 2 → 遷移
        assert ctrl._state == RecoveryState.REDUCED

    def test_reduced_to_stop_transition(self, ctrl_short_stay: DrawdownController) -> None:
        ctrl = ctrl_short_stay
        # まず REDUCED にする
        for _ in range(3):
            ctrl.update(85_000)  # DD=15% -> REDUCED (min_stay到達後)
        assert ctrl._state == RecoveryState.REDUCED
        # DD=25% にして STOP へ
        for _ in range(3):
            ctrl.update(75_000)  # DD=25% -> STOP
        assert ctrl._state == RecoveryState.STOP

    def test_stop_to_reduced_gradual_recovery(
        self, ctrl_short_stay: DrawdownController
    ) -> None:
        ctrl = ctrl_short_stay
        # STOP にする
        for _ in range(3):
            ctrl.update(75_000)  # DD=25%
        assert ctrl._state == RecoveryState.STOP

        # バンクロール回復 -> DD < dd_threshold_2
        for _ in range(3):
            ctrl.update(92_000)  # DD=8% -> target=NORMAL だが STOP→REDUCED に強制
        assert ctrl._state == RecoveryState.REDUCED

    def test_reduced_to_normal_recovery(
        self, ctrl_short_stay: DrawdownController
    ) -> None:
        ctrl = ctrl_short_stay
        # REDUCED にする
        for _ in range(3):
            ctrl.update(85_000)
        assert ctrl._state == RecoveryState.REDUCED

        # バンクロール回復 -> DD < dd_threshold_1
        for _ in range(3):
            ctrl.update(95_000)  # DD=5% -> NORMAL
        assert ctrl._state == RecoveryState.NORMAL

    def test_hysteresis_prevents_oscillation(self, ctrl: DrawdownController) -> None:
        """min_stay_races(=10)未満では遷移しない"""
        # DD=15% -> target=REDUCED だが min_stay=10 未満なので遷移しない
        for _ in range(10):
            ctrl.update(85_000)
        assert ctrl._state == RecoveryState.NORMAL  # 10回でもまだ遷移しない
        ctrl.update(85_000)  # 11回目 -> 遷移
        assert ctrl._state == RecoveryState.REDUCED

    def test_n_bets_counter(self, ctrl: DrawdownController) -> None:
        assert ctrl.n_bets == 0
        ctrl.update(95_000)
        assert ctrl.n_bets == 1
        ctrl.update(90_000)
        assert ctrl.n_bets == 2


# ---------------------------------------------------------------------------
# TestDrawdownControllerMultiplier — 乗数制御
# ---------------------------------------------------------------------------


class TestDrawdownControllerMultiplier:
    def test_multiplier_normal(self, ctrl: DrawdownController) -> None:
        mult = ctrl.get_multiplier(100_000)
        assert mult == pytest.approx(1.0)

    def test_multiplier_reduced(self, ctrl_short_stay: DrawdownController) -> None:
        ctrl = ctrl_short_stay
        for _ in range(3):
            ctrl.update(85_000)  # REDUCED へ
        assert ctrl._state == RecoveryState.REDUCED
        mult = ctrl.get_multiplier(85_000)
        assert mult == pytest.approx(ctrl.cfg.multiplier_reduced, abs=0.01)

    def test_multiplier_stop(self, ctrl_short_stay: DrawdownController) -> None:
        ctrl = ctrl_short_stay
        for _ in range(3):
            ctrl.update(75_000)  # STOP へ
        assert ctrl._state == RecoveryState.STOP
        mult = ctrl.get_multiplier(75_000)
        assert mult == pytest.approx(0.0, abs=0.01)

    def test_adjust_stake_applies_multiplier(self, ctrl: DrawdownController) -> None:
        ctrl._current_multiplier = 0.50
        ctrl._multiplier_at_window_start = 0.50
        ctrl._bets_in_window = 0
        adjusted = ctrl.adjust_stake(1000, 90_000)
        assert adjusted == 500

    def test_adjust_stake_rounds_to_100(self, ctrl: DrawdownController) -> None:
        ctrl._current_multiplier = 0.75
        ctrl._multiplier_at_window_start = 0.75
        ctrl._bets_in_window = 0
        adjusted = ctrl.adjust_stake(330, 90_000)
        assert adjusted % 100 == 0

    def test_adjust_stake_zero_when_stop(self, ctrl_short_stay: DrawdownController) -> None:
        ctrl = ctrl_short_stay
        for _ in range(3):
            ctrl.update(75_000)  # STOP へ
        assert ctrl._state == RecoveryState.STOP
        result = ctrl.adjust_stake(1000, 75_000)
        assert result == 0

    def test_max_adjustment_rate_limits_change(self, ctrl: DrawdownController) -> None:
        """max_adjustment_per_n_bets / max_adjustment_amount による変更幅制限"""
        ctrl._current_multiplier = 0.50
        ctrl._multiplier_at_window_start = 0.50
        ctrl._bets_in_window = 0
        # 状態をNORMALに戻して内部乗数が1.0を目指すが、変更幅で制限される
        ctrl._state = RecoveryState.NORMAL
        ctrl._current_multiplier = 0.50
        ctrl._multiplier_at_window_start = 0.50
        ctrl._bets_in_window = 0
        for _ in range(19):
            ctrl.get_multiplier(100_000)
        mult = ctrl.get_multiplier(100_000)
        # 0.50 + 0.15 = 0.65 に制限される
        assert mult <= 0.66

    def test_n_bet_window_resets(self, ctrl: DrawdownController) -> None:
        ctrl._multiplier_at_window_start = 0.50
        ctrl._bets_in_window = 19
        ctrl.get_multiplier(90_000)  # 20回目 → リセット
        assert ctrl._bets_in_window == 0


# ---------------------------------------------------------------------------
# TestDrawdownControllerGetState
# ---------------------------------------------------------------------------


class TestDrawdownControllerGetState:
    def test_get_state_returns_ddstate(self, ctrl: DrawdownController) -> None:
        ctrl.update(90_000)
        state = ctrl.get_state(90_000)
        assert state.current_dd == pytest.approx(0.10, abs=0.01)
        assert state.n_bets_eval == 1
        assert state.recovery_state == RecoveryState.NORMAL

    def test_get_state_dd_accuracy(self, ctrl: DrawdownController) -> None:
        ctrl.update(80_000)
        state = ctrl.get_state(80_000)
        assert state.current_dd == pytest.approx(0.20, abs=0.001)

    def test_log_state_no_error(self, ctrl: DrawdownController) -> None:
        ctrl.update(95_000)
        ctrl.log_state(95_000)  # Should not raise
