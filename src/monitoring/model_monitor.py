"""ModelMonitor — 予測精度の劣化検知 (F-3b)

設計書 §9.5 / §15:
  - Rolling ROI, 的中率, EV乖離を監視
  - RegimeDetector の COLLAPSED 状態を連携
  - PSI (Population Stability Index) で特徴量ドリフト検知
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PerformanceReport:
    """パフォーマンス監視レポート"""

    n_races: int
    hit_rate: float
    rolling_roi: float
    ev_mean_error: float
    regime: str
    needs_attention: bool
    should_retrain: bool = False


@dataclass(frozen=True)
class DriftReport:
    """特徴量ドリフト検知レポート"""

    psi_max: float
    drifted_features: tuple[str, ...] = ()
    needs_retrain: bool = False


@runtime_checkable
class RegimeDetectorProtocol(Protocol):
    @property
    def current_regime(self) -> object: ...
    def should_retrain(self) -> bool: ...


class ModelMonitor:
    """モデルパフォーマンスの監視

    Args:
        regime_detector: RegimeDetector インスタンス
        min_races: レポート生成に必要な最小レース数
        hit_rate_warning: 的中率警告閾値
        psi_warning: PSI 警告閾値 (0.25 = moderate drift)
    """

    def __init__(
        self,
        regime_detector: RegimeDetectorProtocol,
        min_races: int = 50,
        hit_rate_warning: float = 0.10,
        psi_warning: float = 0.25,
    ) -> None:
        self.regime_detector = regime_detector
        self.min_races = min_races
        self.hit_rate_warning = hit_rate_warning
        self.psi_warning = psi_warning

    def check_performance(self, recent_results: pd.DataFrame) -> PerformanceReport:
        """直近レースのパフォーマンスをチェック

        Args:
            recent_results: columns = [race_id, ev_predicted, ev_actual, hit]

        Returns:
            PerformanceReport
        """
        n = len(recent_results)
        if n == 0:
            return PerformanceReport(
                n_races=0, hit_rate=0.0, rolling_roi=0.0,
                ev_mean_error=0.0, regime="unknown",
                needs_attention=True,
            )

        hit_rate = float(recent_results["hit"].mean())
        ev_error = float(
            (recent_results["ev_actual"] - recent_results["ev_predicted"]).mean()
        )
        # Rolling ROI = sum(actual - 1) / n （単勝EVベースの近似）
        rolling_roi = float(
            (recent_results["ev_actual"].sum() - n) / max(n, 1)
        )

        # レジーム状態
        regime_val = getattr(
            self.regime_detector.current_regime, "value", "unknown"
        )

        # 注意喚起フラグ
        needs_attention = (
            n < self.min_races
            or hit_rate < self.hit_rate_warning
            or regime_val == "collapsed"
        )

        # 再学習トリガー
        should_retrain = (
            self.regime_detector.should_retrain()
            or (n >= self.min_races and hit_rate < self.hit_rate_warning * 0.5)
        )

        return PerformanceReport(
            n_races=n,
            hit_rate=hit_rate,
            rolling_roi=rolling_roi,
            ev_mean_error=ev_error,
            regime=regime_val,
            needs_attention=needs_attention,
            should_retrain=should_retrain,
        )

    def detect_drift(
        self,
        current: pd.DataFrame,
        reference: pd.DataFrame,
    ) -> DriftReport:
        """特徴量ドリフトを PSI で検知

        Args:
            current: 最新データの特徴量 DataFrame
            reference: 学習時の参照データの特徴量 DataFrame

        Returns:
            DriftReport
        """
        common_cols = list(set(current.columns) & set(reference.columns))
        if not common_cols:
            return DriftReport(psi_max=0.0)

        drifted: list[str] = []
        psi_values: list[float] = []

        for col in common_cols:
            psi = self._compute_psi(reference[col], current[col])
            psi_values.append(psi)
            if psi >= self.psi_warning:
                drifted.append(col)

        psi_max = max(psi_values) if psi_values else 0.0

        return DriftReport(
            psi_max=psi_max,
            drifted_features=tuple(drifted),
            needs_retrain=len(drifted) > len(common_cols) // 3,
        )

    @staticmethod
    def _compute_psi(expected: pd.Series, actual: pd.Series, n_bins: int = 10) -> float:
        """PSI (Population Stability Index) を計算

        Args:
            expected: 参照分布
            actual: 実際の分布
            n_bins: ビン数

        Returns:
            PSI 値 (0 = 変化なし, >0.25 = moderate, >0.5 = significant)
        """
        # 共通のビンエッジを計算（両分布の全範囲をカバー）
        combined = pd.concat([expected, actual], ignore_index=True)
        lo = combined.min()
        hi = combined.max()
        if lo == hi:
            return 0.0

        # 等間隔ビンで全データポイントを確実にカバー
        bins = np.linspace(lo, hi, n_bins + 1)

        # 各ビンの割合を計算
        e_counts = pd.cut(expected, bins=bins, include_lowest=True).value_counts(normalize=True)
        a_counts = pd.cut(actual, bins=bins, include_lowest=True).value_counts(normalize=True)

        # 全ビンをカバー
        all_bins = pd.cut(combined, bins=bins, include_lowest=True).cat.categories
        e_pct = e_counts.reindex(all_bins, fill_value=1e-6)
        a_pct = a_counts.reindex(all_bins, fill_value=1e-6)

        psi = float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))
        return max(psi, 0.0)
