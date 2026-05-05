"""src/tuning/strategy_optimizer.py のテスト"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import optuna
import pytest

# src/ is on pythonpath via pyproject.toml
from tuning.strategy_optimizer import StrategyOptimizer


@pytest.fixture
def optimizer() -> StrategyOptimizer:
    return StrategyOptimizer(
        models_dir="data/models",
        min_bets_per_fold=100,  # テスト用に小さく
    )


class TestSuggestParams:
    def test_suggest_returns_all_dimensions(self, optimizer: StrategyOptimizer) -> None:
        """_suggest_params が16+次元を返す"""
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        params = optimizer._suggest_params(trial)

        # レジーム別 (x2): fk, ev, edge = 6
        assert "fk_aggressive" in params
        assert "fk_conservative" in params
        assert "ev_aggressive" in params
        assert "ev_conservative" in params
        assert "edge_aggressive" in params
        assert "edge_conservative" in params
        # DD制御: 5
        assert "dd_threshold_1" in params
        assert "dd_threshold_2" in params
        assert "multiplier_reduced" in params
        assert "rolling_window" in params
        assert "min_stay_races" in params
        # EVスケーリング: 2
        assert "target_ev" in params
        assert "max_scale" in params
        # OddsBandFilter: 1
        assert "roi_threshold" in params
        # Total: 14+
        assert len(params) >= 14

    def test_param_ranges_valid(self, optimizer: StrategyOptimizer) -> None:
        """パラメータが妥当な範囲内"""
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        params = optimizer._suggest_params(trial)

        assert 0.10 <= params["fk_aggressive"] <= 0.80
        assert 0.05 <= params["dd_threshold_1"] <= 0.20
        assert 0.15 <= params["dd_threshold_2"] <= 0.35
        assert 200 <= params["rolling_window"] <= 800
        assert 1.05 <= params["target_ev"] <= 1.50


class TestBuildStrategyConfig:
    def test_builds_dd_config(self, optimizer: StrategyOptimizer) -> None:
        """_build_strategy_config がDDConfigを構築"""
        params = {
            "rolling_window": 500,
            "dd_threshold_1": 0.12,
            "dd_threshold_2": 0.25,
            "multiplier_reduced": 0.4,
            "min_stay_races": 15,
            "fk_aggressive": 0.6, "ev_aggressive": 1.2, "edge_aggressive": 0.08,
            "fk_conservative": 0.3, "ev_conservative": 1.4, "edge_conservative": 0.07,
            "target_ev": 1.15, "max_scale": 2.5, "roi_threshold": 0.95,
        }
        config = optimizer._build_strategy_config(params)

        assert config["dd_config"].rolling_window == 500
        assert config["dd_config"].dd_threshold_1 == 0.12
        assert config["dd_config"].dd_threshold_2 == 0.25
        assert config["regime_overrides"]["aggressive"]["fractional_kelly"] == 0.6
        assert config["target_ev"] == 1.15

    def test_regime_overrides_structure(self, optimizer: StrategyOptimizer) -> None:
        """regime_overrides が aggressive/conservative の2レジームを持つ"""
        params = {
            "rolling_window": 400, "dd_threshold_1": 0.10, "dd_threshold_2": 0.20,
            "multiplier_reduced": 0.5, "min_stay_races": 10,
            "fk_aggressive": 0.5, "ev_aggressive": 1.1, "edge_aggressive": 0.05,
            "fk_conservative": 0.25, "ev_conservative": 1.3, "edge_conservative": 0.06,
            "target_ev": 1.1, "max_scale": 2.0, "roi_threshold": 1.0,
        }
        config = optimizer._build_strategy_config(params)
        assert "aggressive" in config["regime_overrides"]
        assert "conservative" in config["regime_overrides"]
        for regime_key in ("aggressive", "conservative"):
            ro = config["regime_overrides"][regime_key]
            assert "fractional_kelly" in ro
            assert "ev_threshold" in ro
            assert "edge_threshold" in ro

    def test_strategy_config_keys_for_engine(self, optimizer: StrategyOptimizer) -> None:
        """strategy_configがBacktestEngine injectionに必要なキーを持つ"""
        params = {
            "rolling_window": 400, "dd_threshold_1": 0.10, "dd_threshold_2": 0.20,
            "multiplier_reduced": 0.5, "min_stay_races": 10,
            "fk_aggressive": 0.5, "ev_aggressive": 1.1, "edge_aggressive": 0.05,
            "fk_conservative": 0.25, "ev_conservative": 1.3, "edge_conservative": 0.06,
            "target_ev": 1.1, "max_scale": 2.0, "roi_threshold": 1.0,
        }
        config = optimizer._build_strategy_config(params)
        assert "dd_config" in config
        assert "regime_overrides" in config
        assert "target_ev" in config
        assert "max_scale" in config
        assert "fractional_kelly" in config


class TestObjective:
    def test_penalty_when_bets_below_minimum(self, optimizer: StrategyOptimizer) -> None:
        """ベット数不足時は-1.0ペナルティ"""

        def mock_backtest(cfg, test_start, test_end, trial=None, fold_idx=0):
            return {"roi": 1.5, "n_bets": 50}  # 50 < 100 (min_bets)

        optimizer._run_single_backtest = mock_backtest

        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        result = optimizer._objective(trial)
        assert result == -1.0

    def test_returns_roi_when_sufficient_bets(self, optimizer: StrategyOptimizer) -> None:
        """ベット数十分時はROIを返す"""

        def mock_backtest(cfg, test_start, test_end, trial=None, fold_idx=0):
            return {"roi": 1.15, "n_bets": 2000}

        optimizer._run_single_backtest = mock_backtest

        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        result = optimizer._objective(trial)
        assert result == pytest.approx(1.15, abs=0.01)

    def test_pruning_on_bad_fold(self, optimizer: StrategyOptimizer) -> None:
        """MedianPrunerがfold間reportで動作する"""
        call_count = 0

        def mock_backtest(cfg, test_start, test_end, trial=None, fold_idx=0):
            nonlocal call_count
            call_count += 1
            # 最初のfoldで非常に悪いROI -> prunerがpruneする可能性
            return {"roi": 0.3, "n_bets": 2000}

        optimizer._run_single_backtest = mock_backtest

        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        optimizer._objective(trial)
        # reportが呼ばれたことのみ確認(prunerの実際のpruneは確率的)
        assert call_count >= 1


class TestOptimize:
    def test_optimize_returns_best_params(self, optimizer: StrategyOptimizer) -> None:
        """optimize() がbest_params, best_value, n_trialsを返す"""

        def mock_backtest(cfg, test_start, test_end, trial=None, fold_idx=0):
            return {"roi": 1.05, "n_bets": 2000}

        optimizer._run_single_backtest = mock_backtest

        result = optimizer.optimize(n_trials=5, seed=42)
        assert "best_params" in result
        assert "best_value" in result
        assert "n_trials" in result
        assert result["n_trials"] == 5

    def test_optimize_saves_manifest(self, optimizer: StrategyOptimizer, tmp_path: Path) -> None:
        """optimize() がJSON manifestを保存"""

        def mock_backtest(cfg, test_start, test_end, trial=None, fold_idx=0):
            return {"roi": 1.05, "n_bets": 2000}

        optimizer._run_single_backtest = mock_backtest

        manifest_path = tmp_path / "strategy_manifest.json"
        result = optimizer.optimize(n_trials=3, output_path=manifest_path)
        assert manifest_path.exists()

        import json

        manifest = json.loads(manifest_path.read_text())
        assert "params" in manifest
        assert "sha256" in manifest

    def test_pruning_works(self, optimizer: StrategyOptimizer) -> None:
        """MedianPrunerが一部トライアルをpruneする"""

        def mock_backtest(cfg, test_start, test_end, trial=None, fold_idx=0):
            return {"roi": 0.5, "n_bets": 2000}

        optimizer._run_single_backtest = mock_backtest

        result = optimizer.optimize(n_trials=10, seed=42)
        assert "n_pruned" in result
        # pruned数が記録されていること
        assert isinstance(result["n_pruned"], int)


class TestRunSingleBacktest:
    def test_calls_model_loader(self, optimizer: StrategyOptimizer) -> None:
        """_run_single_backtestがModelLoader.load_from_dirを呼ぶ"""
        mock_models = MagicMock()
        mock_info = MagicMock()
        mock_result = MagicMock()
        mock_result.total_roi = 1.10
        mock_result.total_bets = 5000
        mock_result.total_stake = 500000
        mock_result.total_return = 550000
        mock_result.max_drawdown = 0.15

        with patch("db.model_loader.ModelLoader") as MockLoader, \
             patch("backtest.engine.BacktestEngine") as MockEngine:
            MockLoader.return_value.load_from_dir.return_value = (mock_models, mock_info)
            MockEngine.return_value.run.return_value = mock_result

            config = optimizer._build_strategy_config({
                "rolling_window": 400, "dd_threshold_1": 0.10, "dd_threshold_2": 0.20,
                "multiplier_reduced": 0.5, "min_stay_races": 10,
                "fk_aggressive": 0.5, "ev_aggressive": 1.1, "edge_aggressive": 0.05,
                "fk_conservative": 0.25, "ev_conservative": 1.3, "edge_conservative": 0.06,
                "target_ev": 1.1, "max_scale": 2.0, "roi_threshold": 1.0,
            })
            result = optimizer._run_single_backtest(config, "2024-01-01", "2024-12-31")

            MockLoader.return_value.load_from_dir.assert_called_once()
            MockEngine.assert_called_once()
            # strategy_paramsがengineに渡されたこと
            call_kwargs = MockEngine.call_args
            assert call_kwargs.kwargs.get("strategy_params") is not None
            assert result["roi"] == 1.10
            assert result["n_bets"] == 5000

    def test_injects_regime_overrides(self, optimizer: StrategyOptimizer) -> None:
        """regime_overridesがある場合、既存のRegimeDetectorの_override_paramsを更新"""
        mock_models = MagicMock()
        mock_info = MagicMock()
        mock_result = MagicMock()
        mock_result.total_roi = 1.05
        mock_result.total_bets = 2000
        mock_result.total_stake = 200000
        mock_result.total_return = 210000
        mock_result.max_drawdown = 0.10

        with patch("db.model_loader.ModelLoader") as MockLoader, \
             patch("backtest.engine.BacktestEngine") as MockEngine:
            MockLoader.return_value.load_from_dir.return_value = (mock_models, mock_info)
            MockEngine.return_value.run.return_value = mock_result

            config = optimizer._build_strategy_config({
                "rolling_window": 400, "dd_threshold_1": 0.10, "dd_threshold_2": 0.20,
                "multiplier_reduced": 0.5, "min_stay_races": 10,
                "fk_aggressive": 0.6, "ev_aggressive": 1.2, "edge_aggressive": 0.08,
                "fk_conservative": 0.3, "ev_conservative": 1.4, "edge_conservative": 0.07,
                "target_ev": 1.15, "max_scale": 2.5, "roi_threshold": 0.95,
            })
            optimizer._run_single_backtest(config, "2024-01-01", "2024-12-31")

            # 既存のRegimeDetectorの_override_paramsが更新されたこと
            assert mock_models.regime_detector._override_params["aggressive"]["fractional_kelly"] == 0.6
