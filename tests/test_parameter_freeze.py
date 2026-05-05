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


class TestStrategyManifest:
    """戦略パラメータ JSON manifest + SHA256 テスト"""

    def test_save_creates_json_file(self, tmp_path: object) -> None:
        """save_strategy_manifest がJSON manifestを作成"""
        from pathlib import Path

        from backtest.parameter_freeze_protocol import save_strategy_manifest

        tp = tmp_path  # type: ignore[assignment]
        params: dict[str, object] = {"fk_aggressive": 0.5, "dd_threshold_1": 0.10}
        path = Path(str(tp)) / "strategy_manifest.json"
        sha = save_strategy_manifest(params, path)
        assert path.exists()
        assert len(sha) == 64  # SHA256 hex

    def test_manifest_contains_params_and_sha256(self, tmp_path: object) -> None:
        """manifestがparams + sha256を含む"""
        import json
        from pathlib import Path

        from backtest.parameter_freeze_protocol import save_strategy_manifest

        tp = tmp_path  # type: ignore[assignment]
        params: dict[str, object] = {"fk_aggressive": 0.5}
        path = Path(str(tp)) / "strategy_manifest.json"
        save_strategy_manifest(params, path)
        manifest = json.loads(path.read_text())
        assert "params" in manifest
        assert "sha256" in manifest
        assert manifest["params"] == params

    def test_verify_returns_params_on_match(self, tmp_path: object) -> None:
        """SHA256一致時にparamsを返す"""
        from pathlib import Path

        from backtest.parameter_freeze_protocol import (
            save_strategy_manifest,
            verify_strategy_manifest,
        )

        tp = tmp_path  # type: ignore[assignment]
        params: dict[str, object] = {"fk_aggressive": 0.5, "ev_threshold": 1.10}
        path = Path(str(tp)) / "strategy_manifest.json"
        save_strategy_manifest(params, path)
        result = verify_strategy_manifest(path)
        assert result == params

    def test_verify_raises_on_tampered_manifest(self, tmp_path: object) -> None:
        """SHA256不一致時にValueError"""
        import json
        from pathlib import Path

        from backtest.parameter_freeze_protocol import (
            save_strategy_manifest,
            verify_strategy_manifest,
        )

        tp = tmp_path  # type: ignore[assignment]
        params: dict[str, object] = {"fk_aggressive": 0.5}
        path = Path(str(tp)) / "strategy_manifest.json"
        save_strategy_manifest(params, path)
        # manifest改ざん
        manifest = json.loads(path.read_text())
        manifest["params"]["fk_aggressive"] = 0.9  # 値変更
        path.write_text(json.dumps(manifest, indent=2))
        with pytest.raises(ValueError, match="hash mismatch"):
            verify_strategy_manifest(path)

    def test_verify_raises_on_missing_file(self, tmp_path: object) -> None:
        """manifest不在時にFileNotFoundError"""
        from pathlib import Path

        from backtest.parameter_freeze_protocol import verify_strategy_manifest

        tp = tmp_path  # type: ignore[assignment]
        path = Path(str(tp)) / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            verify_strategy_manifest(path)

    def test_sha256_deterministic(self, tmp_path: object) -> None:
        """同じparamsで同じSHA256"""
        from pathlib import Path

        from backtest.parameter_freeze_protocol import save_strategy_manifest

        tp = tmp_path  # type: ignore[assignment]
        params: dict[str, object] = {"a": 1, "b": 2.0}
        path1 = Path(str(tp)) / "m1.json"
        path2 = Path(str(tp)) / "m2.json"
        sha1 = save_strategy_manifest(params, path1)
        sha2 = save_strategy_manifest(params, path2)
        assert sha1 == sha2

    def test_load_and_freeze_saves_new(self, tmp_path: object) -> None:
        """manifest不在時に新規保存"""
        from pathlib import Path

        from backtest.parameter_freeze_protocol import (
            load_and_freeze_strategy,
            verify_strategy_manifest,
        )

        tp = tmp_path  # type: ignore[assignment]
        params: dict[str, object] = {"fk_aggressive": 0.5}
        path = Path(str(tp)) / "strategy_manifest.json"
        load_and_freeze_strategy(params, path)
        assert path.exists()
        result = verify_strategy_manifest(path)
        assert result == params

    def test_load_and_freeze_verifies_existing(self, tmp_path: object) -> None:
        """manifest存在時に検証"""
        from pathlib import Path

        from backtest.parameter_freeze_protocol import (
            load_and_freeze_strategy,
            save_strategy_manifest,
        )

        tp = tmp_path  # type: ignore[assignment]
        params: dict[str, object] = {"fk_aggressive": 0.5}
        path = Path(str(tp)) / "strategy_manifest.json"
        save_strategy_manifest(params, path)
        load_and_freeze_strategy(params, path)  # 同じparams → エラーなし
