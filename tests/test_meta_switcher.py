"""src/betting/meta_switcher.py のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from betting.meta_switcher import MetaSwitcher
from domain.types import RegimeState


@pytest.fixture
def switcher() -> MetaSwitcher:
    regime_mock = MagicMock()
    regime_mock.current_regime = RegimeState.CONSERVATIVE
    return MetaSwitcher(regime_detector=regime_mock)


class TestMetaSwitcher:
    def test_get_params_returns_dict(self, switcher: MetaSwitcher) -> None:
        """戦略パラメータがdictで返る"""
        params = switcher.get_strategy_params()
        assert isinstance(params, dict)
        assert "ev_threshold" in params
        assert "score_threshold" in params
        assert "max_bets_per_race" in params

    def test_aggressive_params(self, switcher: MetaSwitcher) -> None:
        """AGGRESSIVE: EV閾値が低い・max_betsが多い"""
        switcher._regime_detector.current_regime = RegimeState.AGGRESSIVE
        params = switcher.get_strategy_params()
        assert params["ev_threshold"] < 1.20
        assert params["max_bets_per_race"] >= 3

    def test_conservative_params(self, switcher: MetaSwitcher) -> None:
        """CONSERVATIVE: EV閾値が高い・max_betsが少ない"""
        switcher._regime_detector.current_regime = RegimeState.CONSERVATIVE
        params = switcher.get_strategy_params()
        assert params["ev_threshold"] >= 1.20
        assert params["max_bets_per_race"] <= 2

    def test_collapsed_params(self, switcher: MetaSwitcher) -> None:
        """COLLAPSED: EV閾値が最高・max_bets=1"""
        switcher._regime_detector.current_regime = RegimeState.COLLAPSED
        params = switcher.get_strategy_params()
        assert params["ev_threshold"] > 1.30
        assert params["max_bets_per_race"] == 1

    def test_should_retrain_delegates(self, switcher: MetaSwitcher) -> None:
        """should_retrain は RegimeDetector に委譲"""
        switcher._regime_detector.should_retrain.return_value = True
        assert switcher.should_retrain() is True
        switcher._regime_detector.should_retrain.assert_called_once()

    def test_description_present(self, switcher: MetaSwitcher) -> None:
        """description フィールドが存在する"""
        params = switcher.get_strategy_params()
        assert "description" in params
