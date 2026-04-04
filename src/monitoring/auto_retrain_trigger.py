"""AutoRetrainTrigger — 再学習トリガー判定 (F-3c)

設計書 §9.5 / §15 リスク①:
  - COLLAPSED 連続100レース → 再学習
  - 的中率の著しい低下 → 再学習
  - 特徴量ドリフト (PSI) → 再学習
  - クールダウン期間で頻繁な再学習を防止
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional, Protocol, runtime_checkable

import pandas as pd

from monitoring.model_monitor import DriftReport, PerformanceReport

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrainDecision:
    """再学習判定結果"""

    triggered: bool
    reason: str = ""


@runtime_checkable
class ModelMonitorProtocol(Protocol):
    def check_performance(self, results: pd.DataFrame) -> PerformanceReport: ...
    def detect_drift(
        self,
        current: pd.DataFrame,
        reference: pd.DataFrame,
    ) -> DriftReport: ...


@runtime_checkable
class NotifierProtocol(Protocol):
    def send(self, message: str, level: str = "info") -> bool: ...


class AutoRetrainTrigger:
    """再学習の要否を判定し、必要時に通知を送る

    Args:
        model_monitor: ModelMonitor インスタンス
        notifier: Notifier インスタンス
        cooldown_hours: 前回再学習からの最小間隔 (時間)
        retrain_callback: 再学習時に呼ばれるコールバック（省略時は通知のみ）
    """

    def __init__(
        self,
        model_monitor: ModelMonitorProtocol,
        notifier: NotifierProtocol,
        cooldown_hours: int = 24,
        retrain_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._monitor = model_monitor
        self._notifier = notifier
        self.cooldown_hours = cooldown_hours
        self._last_retrain_time: Optional[datetime] = None
        self._retrain_callback = retrain_callback

    def evaluate(
        self,
        recent_results: pd.DataFrame,
        current_features: Optional[pd.DataFrame] = None,
        reference_features: Optional[pd.DataFrame] = None,
    ) -> RetrainDecision:
        """再学習の要否を判定

        Args:
            recent_results: 直近のパフォーマンスデータ
            current_features: 最新特徴量 (ドリフト検知用、任意)
            reference_features: 参照特徴量 (ドリフト検知用、任意)

        Returns:
            RetrainDecision
        """
        # クールダウンチェック
        if self._last_retrain_time is not None:
            elapsed = datetime.now() - self._last_retrain_time
            if elapsed < timedelta(hours=self.cooldown_hours):
                remaining = timedelta(hours=self.cooldown_hours) - elapsed
                return RetrainDecision(
                    triggered=False,
                    reason=f"Cooldown: {remaining.total_seconds() / 3600:.1f}h remaining",
                )

        # パフォーマンスチェック
        report = self._monitor.check_performance(recent_results)
        if report.should_retrain:
            self._fire_retrain(
                f"Performance: regime={report.regime}, hit_rate={report.hit_rate:.2%}"
            )
            return RetrainDecision(triggered=True, reason="Performance degradation")

        # ドリフトチェック
        if current_features is not None and reference_features is not None:
            drift = self._monitor.detect_drift(current_features, reference_features)
            if drift.needs_retrain:
                self._fire_retrain(
                    f"Feature drift: PSI max={drift.psi_max:.3f}, features={drift.drifted_features}"
                )
                return RetrainDecision(triggered=True, reason="Feature drift detected")

        return RetrainDecision(triggered=False)

    def _fire_retrain(self, reason: str) -> None:
        """再学習を発火し、通知を送る"""
        self._last_retrain_time = datetime.now()
        logger.warning(f"AutoRetrain triggered: {reason}")
        self._notifier.send(
            f"Retrain triggered: {reason}",
            level="critical",
        )
        if self._retrain_callback is not None:
            try:
                self._retrain_callback(reason)
            except Exception:
                logger.exception("Retrain callback failed")
