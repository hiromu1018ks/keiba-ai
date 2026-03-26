"""src/monitoring/model_monitor.py のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest


class TestModelMonitor:
    def _make_monitor(self) -> "ModelMonitor":
        from monitoring.model_monitor import ModelMonitor

        mock_regime = MagicMock()
        mock_regime.current_regime = MagicMock(value="conservative")
        mock_regime.should_retrain.return_value = False
        return ModelMonitor(regime_detector=mock_regime)

    def test_check_performance_returns_report(self) -> None:
        """直近のパフォーマンスレポートを返す"""
        import numpy as np

        monitor = self._make_monitor()
        # モックの recent_results: 100 レースのダミーデータ
        rng = np.random.RandomState(42)
        results = pd.DataFrame({
            "race_id": [f"R{i}" for i in range(100)],
            "ev_predicted": rng.uniform(0.8, 1.5, 100),
            "ev_actual": rng.uniform(0.0, 3.0, 100),
            "hit": rng.choice([0, 1], 100, p=[0.7, 0.3]),
        })

        report = monitor.check_performance(results)

        assert report.n_races == 100
        assert 0 <= report.hit_rate <= 1
        assert isinstance(report.rolling_roi, float)
        assert report.regime == "conservative"

    def test_check_performance_empty_data(self) -> None:
        """データがない場合はデフォトレポート"""
        monitor = self._make_monitor()

        report = monitor.check_performance(pd.DataFrame())

        assert report.n_races == 0
        assert report.hit_rate == 0.0
        assert report.needs_attention is True

    def test_detect_drift_returns_report(self) -> None:
        """特徴量ドリフト検知レポートを返す"""
        import numpy as np

        monitor = self._make_monitor()
        reference = pd.DataFrame({
            "feature_a": np.random.normal(0, 1, 1000),
            "feature_b": np.random.normal(5, 2, 1000),
        })
        current = pd.DataFrame({
            "feature_a": np.random.normal(0.5, 1, 500),  # シフトあり
            "feature_b": np.random.normal(5, 2, 500),
        })

        drift = monitor.detect_drift(current, reference)

        assert isinstance(drift.psi_max, float)
        assert drift.psi_max >= 0

    def test_check_performance_low_hit_rate(self) -> None:
        """的中率が著しく低い場合は needs_attention=True"""
        monitor = self._make_monitor()
        results = pd.DataFrame({
            "race_id": [f"R{i}" for i in range(100)],
            "ev_predicted": [1.0] * 100,
            "ev_actual": [0.0] * 100,
            "hit": [0] * 100,  # 全部外れ
        })

        report = monitor.check_performance(results)

        assert report.needs_attention is True

    def test_should_retrain_returns_true_when_collapsed(self) -> None:
        """COLLAPSED 状態で should_retrain=True の場合、再学習が必要"""
        mock_regime = MagicMock()
        mock_regime.current_regime = MagicMock(value="collapsed")
        mock_regime.should_retrain.return_value = True

        from monitoring.model_monitor import ModelMonitor
        monitor = ModelMonitor(regime_detector=mock_regime)

        results = pd.DataFrame({
            "race_id": ["R1"],
            "ev_predicted": [1.0],
            "ev_actual": [0.0],
            "hit": [0],
        })
        report = monitor.check_performance(results)

        assert report.should_retrain is True
