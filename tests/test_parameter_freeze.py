"""ParameterFreezeProtocol のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from domain.models import SubmodelSet, TrainedModelsV5


@pytest.fixture
def mock_models() -> TrainedModelsV5:
    """テスト用 TrainedModelsV5 (モックモデル付き)"""
    mock_market = MagicMock()
    mock_market.model = MagicMock()
    mock_market.model.num_trees.return_value = 100

    mock_stage1 = MagicMock()
    mock_stage1.models = {"turf": MagicMock()}

    mock_win = MagicMock()
    mock_win.hit_model = MagicMock()
    mock_win.return_model = MagicMock()

    return TrainedModelsV5(
        submodels={
            "turf": SubmodelSet(
                market=mock_market,
                stage1=mock_stage1,
                place_ability=MagicMock(),
                win=mock_win,
                ev_corrector=MagicMock(),
                place=MagicMock(),
                place_ev_corrector=MagicMock(),
                wide=MagicMock(),
                confidence=MagicMock(),
            ),
        },
        quality_screener=MagicMock(),
        regime_detector=MagicMock(),
        train_period=("2020-01-01", "2023-12-31"),
    )


class TestParameterFreezeProtocol:
    """ParameterFreezeProtocol のテスト"""

    def test_freeze_takes_snapshot(self, mock_models: TrainedModelsV5) -> None:
        """freeze() がモデルのスナップショットを取得する"""
        from backtest.parameter_freeze_protocol import ParameterFreezeProtocol

        protocol = ParameterFreezeProtocol(mock_models)
        protocol.freeze()

        assert protocol._snapshot is not None

    def test_detect_no_change_after_freeze(self, mock_models: TrainedModelsV5) -> None:
        """freeze 後に変更がない場合、verify() が PASS"""
        from backtest.parameter_freeze_protocol import ParameterFreezeProtocol

        protocol = ParameterFreezeProtocol(mock_models)
        protocol.freeze()

        result = protocol.verify()
        assert result["passed"] is True

    def test_detect_change_after_freeze(self, mock_models: TrainedModelsV5) -> None:
        """freeze 後にモデルが変更された場合、verify() が FAIL"""
        from backtest.parameter_freeze_protocol import ParameterFreezeProtocol

        protocol = ParameterFreezeProtocol(mock_models)
        protocol.freeze()

        # サブモデルを差し替えてシリアライズ差分を作成
        mock_models.submodels["turf"] = SubmodelSet(
            market=MagicMock(),
            stage1=MagicMock(),
            place_ability=MagicMock(),
            win=MagicMock(),
            ev_corrector=MagicMock(),
            place=MagicMock(),
            place_ev_corrector=MagicMock(),
            wide=MagicMock(),
            confidence=MagicMock(),
        )

        result = protocol.verify()
        assert result["passed"] is False

    def test_context_manager(self, mock_models: TrainedModelsV5) -> None:
        """コンテキストマネージャとして使用できる"""
        from backtest.parameter_freeze_protocol import ParameterFreezeProtocol

        protocol = ParameterFreezeProtocol(mock_models)

        with protocol.frozen_period():
            pass  # 変更なし

    def test_context_manager_detects_violation(self, mock_models: TrainedModelsV5) -> None:
        """コンテキスト内で変更を検出"""
        from backtest.parameter_freeze_protocol import ParameterFreezeProtocol

        protocol = ParameterFreezeProtocol(mock_models)

        violation_detected = False
        try:
            with protocol.frozen_period():
                # サブモデルを差し替えてパラメータ変更を模擬
                mock_models.submodels["turf"] = SubmodelSet(
                    market=MagicMock(),
                    stage1=MagicMock(),
                    place_ability=MagicMock(),
                    win=MagicMock(),
                    ev_corrector=MagicMock(),
                    place=MagicMock(),
                    place_ev_corrector=MagicMock(),
                    wide=MagicMock(),
                    confidence=MagicMock(),
                )
        except RuntimeError:
            violation_detected = True

        # frozen_period は変更を検出すると RuntimeError を送出
        assert violation_detected

    def test_verify_without_freeze(self, mock_models: TrainedModelsV5) -> None:
        """freeze() 前に verify() を呼ぶと FAIL"""
        from backtest.parameter_freeze_protocol import ParameterFreezeProtocol

        protocol = ParameterFreezeProtocol(mock_models)

        result = protocol.verify()
        assert result["passed"] is False
        assert "freeze" in result["message"]
