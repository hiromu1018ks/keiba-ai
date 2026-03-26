"""src/monitoring/auto_retrain_trigger.py のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest


class TestAutoRetrainTrigger:
    def _make_trigger(
        self, cooldown_hours: int = 24
    ) -> "AutoRetrainTrigger":
        mock_monitor = MagicMock()
        mock_monitor.check_performance.return_value = MagicMock(
            should_retrain=False, needs_attention=False,
            hit_rate=0.3, n_races=100, regime="conservative",
        )
        mock_monitor.detect_drift.return_value = MagicMock(
            needs_retrain=False, psi_max=0.1,
        )
        mock_notifier = MagicMock()
        mock_notifier.send.return_value = True

        from monitoring.auto_retrain_trigger import AutoRetrainTrigger
        return AutoRetrainTrigger(
            model_monitor=mock_monitor,
            notifier=mock_notifier,
            cooldown_hours=cooldown_hours,
        )

    def test_evaluate_no_retrain_needed(self) -> None:
        """再学習不要な場合は triggered=False"""
        trigger = self._make_trigger()
        results = pd.DataFrame({
            "race_id": ["R1"], "ev_predicted": [1.0],
            "ev_actual": [1.0], "hit": [1],
        })

        decision = trigger.evaluate(results)

        assert decision.triggered is False
        assert decision.reason == ""

    def test_evaluate_triggers_on_monitor_flag(self) -> None:
        """ModelMonitor が should_retrain=True の場合はトリガー"""
        trigger = self._make_trigger()
        trigger._monitor.check_performance.return_value = MagicMock(
            should_retrain=True, needs_attention=True,
            regime="collapsed", hit_rate=0.05, n_races=100,
        )

        results = pd.DataFrame({
            "race_id": ["R1"], "ev_predicted": [1.0],
            "ev_actual": [0.0], "hit": [0],
        })
        decision = trigger.evaluate(results)

        assert decision.triggered is True
        assert "retrain" in decision.reason.lower() or "performance" in decision.reason.lower()

    def test_evaluate_triggers_on_drift(self) -> None:
        """特徴量ドリフトが閾値を超える場合はトリガー"""
        trigger = self._make_trigger()
        trigger._monitor.detect_drift.return_value = MagicMock(
            needs_retrain=True, psi_max=0.5,
            drifted_features=("f1", "f2"),
        )

        results = pd.DataFrame({
            "race_id": ["R1"], "ev_predicted": [1.0],
            "ev_actual": [1.0], "hit": [1],
        })
        import numpy as np
        current = pd.DataFrame({"f1": np.random.normal(1, 1, 100)})
        reference = pd.DataFrame({"f1": np.random.normal(0, 1, 100)})

        decision = trigger.evaluate(
            results,
            current_features=current,
            reference_features=reference,
        )

        assert decision.triggered is True
        assert "drift" in decision.reason.lower()

    def test_evaluate_notifies_on_trigger(self) -> None:
        """トリガー時に通知を送信"""
        trigger = self._make_trigger()
        trigger._monitor.check_performance.return_value = MagicMock(
            should_retrain=True, needs_attention=True,
            regime="collapsed", hit_rate=0.05, n_races=100,
        )

        results = pd.DataFrame({
            "race_id": ["R1"], "ev_predicted": [1.0],
            "ev_actual": [0.0], "hit": [0],
        })
        trigger.evaluate(results)

        trigger._notifier.send.assert_called()
        call_args = trigger._notifier.send.call_args
        assert "retrain" in call_args[0][0].lower()

    def test_cooldown_prevents_rapid_retrains(self) -> None:
        """クールダウン期間中は再トリガーしない"""
        trigger = self._make_trigger(cooldown_hours=24)
        trigger._monitor.check_performance.return_value = MagicMock(
            should_retrain=True, needs_attention=True,
            regime="collapsed", hit_rate=0.05, n_races=100,
        )

        results = pd.DataFrame({
            "race_id": ["R1"], "ev_predicted": [1.0],
            "ev_actual": [0.0], "hit": [0],
        })

        # 1回目: トリガー
        decision1 = trigger.evaluate(results)
        assert decision1.triggered is True

        # 2回目: クールダウン中
        decision2 = trigger.evaluate(results)
        assert decision2.triggered is False
        assert "cooldown" in decision2.reason.lower()

    def test_callback_called_on_retrain(self) -> None:
        """再学習トリガー時にコールバックが呼ばれる"""
        mock_callback = MagicMock()
        trigger = self._make_trigger()
        trigger._monitor.check_performance.return_value = MagicMock(
            should_retrain=True, needs_attention=True,
            regime="collapsed", hit_rate=0.05, n_races=100,
        )
        trigger._retrain_callback = mock_callback

        results = pd.DataFrame({
            "race_id": ["R1"], "ev_predicted": [1.0],
            "ev_actual": [0.0], "hit": [0],
        })
        trigger.evaluate(results)

        mock_callback.assert_called_once()
