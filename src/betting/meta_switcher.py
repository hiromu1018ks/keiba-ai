"""レジーム連動戦略パラメータ"""

from __future__ import annotations

from typing import Protocol

from domain.types import RegimeState


class RegimeDetectorProtocol(Protocol):
    """RegimeDetector のプロトコル（依存逆転）"""

    current_regime: RegimeState
    should_retrain: callable


class MetaSwitcher:
    """
    現在のレジーム状態に応じて戦略パラメータを動的に切り替える。

    RegimeDetector の出力をベッティング層で使いやすい形に変換する。
    設計書 §12 のオーケストレーター ② で使用。
    """

    def __init__(self, regime_detector: RegimeDetectorProtocol) -> None:
        self._regime_detector = regime_detector

    def get_strategy_params(self) -> dict[str, object]:
        """
        現在のレジームに応じた戦略パラメータを返す。

        Returns:
            ev_threshold: EV下限閾値
            score_threshold: ワイドスコア閾値
            max_bets_per_race: レースあたり最大ベット数
            description: レジーム説明
        """
        regime = self._regime_detector.current_regime
        return self._default_params(regime)

    @staticmethod
    def _default_params(regime: RegimeState) -> dict[str, object]:
        """レジームに応じたデフォルトパラメータ"""
        if regime == RegimeState.AGGRESSIVE:
            return {
                "ev_threshold": 1.15,  # raised from 1.10
                "edge_threshold": 0.05,  # 5% edge — JRA控除率25%考慮 (Phase 3)
                "score_threshold": 0.010,
                "max_bets_per_race": 3,
                "description": "歪み強い → 攻める",
            }
        elif regime == RegimeState.CONSERVATIVE:
            return {
                "ev_threshold": 1.35,  # raised from 1.30
                "edge_threshold": 0.07,  # 7% edge — JRA控除率25%考慮 (Phase 3)
                "score_threshold": 0.020,
                "max_bets_per_race": 2,
                "description": "効率的 → 絞る",
            }
        else:  # COLLAPSED
            return {
                "ev_threshold": 1.55,  # raised from 1.50
                "edge_threshold": 0.10,  # 10% edge — JRA控除率25%考慮 (Phase 3)
                "score_threshold": 0.050,
                "max_bets_per_race": 1,
                "description": "崩壊 → ほぼ停止",
            }

    def should_retrain(self) -> bool:
        """RegimeDetector の再学習トリガーを委譲"""
        return self._regime_detector.should_retrain()
