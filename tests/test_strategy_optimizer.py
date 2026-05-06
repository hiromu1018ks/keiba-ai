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
        """_suggest_params が16次元を返す (14既存 + ev_lower_threshold_turf + ev_lower_threshold_dirt)"""
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
        # EV_lower閾値: 2 (15-16次元目)
        assert "ev_lower_threshold_turf" in params
        assert "ev_lower_threshold_dirt" in params
        # Total: 16
        assert len(params) == 16

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

    def test_ev_lower_thresholds_in_range(self, optimizer: StrategyOptimizer) -> None:
        """EV_lower閾値が[0.5, 1.5]の範囲内"""
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        params = optimizer._suggest_params(trial)
        assert 0.5 <= params["ev_lower_threshold_turf"] <= 1.5
        assert 0.5 <= params["ev_lower_threshold_dirt"] <= 1.5


class TestGenerateFolds:
    def test_4fold_default(self) -> None:
        """n_folds=4, fold_start_year=2022で4fold動的生成"""
        opt = StrategyOptimizer(n_folds=4, fold_start_year=2022)
        folds = opt._generate_folds()
        assert len(folds) == 4
        assert folds[0] == ("2022-01-01", "2022-12-31")
        assert folds[1] == ("2023-01-01", "2023-12-31")
        assert folds[2] == ("2024-01-01", "2024-12-31")
        assert folds[3] == ("2025-01-01", "2025-12-31")

    def test_2fold_backward_compat(self) -> None:
        """n_folds=2で2foldを返す(後方互換)"""
        opt = StrategyOptimizer(n_folds=2, fold_start_year=2024)
        folds = opt._generate_folds()
        assert len(folds) == 2
        assert folds[0] == ("2024-01-01", "2024-12-31")
        assert folds[1] == ("2025-01-01", "2025-12-31")

    def test_custom_start_year(self) -> None:
        """カスタムfold_start_yearで正しいfold生成"""
        opt = StrategyOptimizer(n_folds=3, fold_start_year=2021)
        folds = opt._generate_folds()
        assert len(folds) == 3
        assert folds[0] == ("2021-01-01", "2021-12-31")
        assert folds[1] == ("2022-01-01", "2022-12-31")
        assert folds[2] == ("2023-01-01", "2023-12-31")


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

    def test_auto_corrects_dd_threshold_2_when_leq_threshold_1(self, optimizer: StrategyOptimizer) -> None:
        """dd_threshold_2 <= dd_threshold_1 の場合に自動補正される"""
        params = {
            "rolling_window": 400, "dd_threshold_1": 0.20, "dd_threshold_2": 0.15,
            "multiplier_reduced": 0.5, "min_stay_races": 10,
            "fk_aggressive": 0.5, "ev_aggressive": 1.1, "edge_aggressive": 0.05,
            "fk_conservative": 0.25, "ev_conservative": 1.3, "edge_conservative": 0.06,
            "target_ev": 1.1, "max_scale": 2.0, "roi_threshold": 1.0,
        }
        config = optimizer._build_strategy_config(params)
        dd_cfg = config["dd_config"]
        assert dd_cfg.dd_threshold_2 > dd_cfg.dd_threshold_1
        assert dd_cfg.dd_threshold_2 == pytest.approx(0.21, abs=0.001)


class TestObjective:
    def test_penalty_when_bets_below_minimum(self, optimizer: StrategyOptimizer) -> None:
        """ベット数不足時は-1.0ペナルティ"""

        def mock_backtest_with_models(
            models, strategy_config, test_start, test_end,
            training_bet_history, fold_idx=0,
        ):
            return {"roi": 1.5, "n_bets": 50}  # 50 < 100 (min_bets)

        optimizer._run_single_backtest_with_models = mock_backtest_with_models

        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        with patch("db.model_loader.ModelLoader") as MockLoader, \
             patch.object(optimizer, "_generate_training_bet_history", return_value=[]):
            MockLoader.return_value.load_from_dir.return_value = (MagicMock(), MagicMock())
            result = optimizer._objective(trial)
        assert result == -1.0

    def test_returns_roi_when_sufficient_bets(self, optimizer: StrategyOptimizer) -> None:
        """ベット数十分時はROIを返す"""

        def mock_backtest_with_models(
            models, strategy_config, test_start, test_end,
            training_bet_history, fold_idx=0,
        ):
            return {"roi": 1.15, "n_bets": 2000}

        optimizer._run_single_backtest_with_models = mock_backtest_with_models

        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        with patch("db.model_loader.ModelLoader") as MockLoader, \
             patch.object(optimizer, "_generate_training_bet_history", return_value=[]):
            MockLoader.return_value.load_from_dir.return_value = (MagicMock(), MagicMock())
            result = optimizer._objective(trial)
        assert result == pytest.approx(1.15, abs=0.01)

    def test_pruning_on_bad_fold(self, optimizer: StrategyOptimizer) -> None:
        """MedianPrunerがfold間reportで動作する"""
        call_count = 0

        def mock_backtest_with_models(
            models, strategy_config, test_start, test_end,
            training_bet_history, fold_idx=0,
        ):
            nonlocal call_count
            call_count += 1
            return {"roi": 0.3, "n_bets": 2000}

        optimizer._run_single_backtest_with_models = mock_backtest_with_models

        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        with patch("db.model_loader.ModelLoader") as MockLoader, \
             patch.object(optimizer, "_generate_training_bet_history", return_value=[]):
            MockLoader.return_value.load_from_dir.return_value = (MagicMock(), MagicMock())
            optimizer._objective(trial)
        assert call_count >= 1

    def test_model_load_optimization(self, optimizer: StrategyOptimizer) -> None:
        """_objective()内でModelLoader.load_from_dir()が1回のみ呼ばれる"""

        def mock_backtest_with_models(
            models, strategy_config, test_start, test_end,
            training_bet_history, fold_idx=0,
        ):
            return {"roi": 1.05, "n_bets": 2000}

        optimizer._run_single_backtest_with_models = mock_backtest_with_models

        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        with patch("db.model_loader.ModelLoader") as MockLoader, \
             patch.object(optimizer, "_generate_training_bet_history", return_value=[]):
            MockLoader.return_value.load_from_dir.return_value = (MagicMock(), MagicMock())
            optimizer._objective(trial)
            # D-05: ModelLoader.load_from_dir()はtrial内1回のみ
            MockLoader.return_value.load_from_dir.assert_called_once()

    def test_training_bet_history_cached_once(self, optimizer: StrategyOptimizer) -> None:
        """_objective()内でtraining_bet_history生成が1回のみ実行される"""
        call_count = 0

        def mock_generate_training_bet_history(models, info, default_config):
            nonlocal call_count
            call_count += 1
            return []

        optimizer._generate_training_bet_history = mock_generate_training_bet_history

        def mock_backtest_with_models(
            models, strategy_config, test_start, test_end,
            training_bet_history, fold_idx=0,
        ):
            return {"roi": 1.05, "n_bets": 2000}

        optimizer._run_single_backtest_with_models = mock_backtest_with_models

        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        with patch("db.model_loader.ModelLoader") as MockLoader:
            MockLoader.return_value.load_from_dir.return_value = (MagicMock(), MagicMock())
            optimizer._objective(trial)
        assert call_count == 1

    def test_regime_reset_per_fold(self, optimizer: StrategyOptimizer) -> None:
        """複数fold実行時に各fold開始前にRegimeDetector状態がリセットされる"""
        mock_models = MagicMock()
        mock_info = MagicMock()

        regime_states: list[dict] = []

        def mock_backtest_with_models(
            models, strategy_config, test_start, test_end,
            training_bet_history, fold_idx=0,
        ):
            # リセット後の状態をキャプチャ
            regime_states.append({
                "_current_regime": models.regime_detector._current_regime,
                "_regime_counter": models.regime_detector._regime_counter,
                "_pending_regime": models.regime_detector._pending_regime,
                "_collapsed_consecutive": models.regime_detector._collapsed_consecutive,
            })
            return {"roi": 1.05, "n_bets": 2000}

        optimizer._run_single_backtest_with_models = mock_backtest_with_models

        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        with patch("db.model_loader.ModelLoader") as MockLoader, \
             patch.object(optimizer, "_generate_training_bet_history", return_value=[]):
            MockLoader.return_value.load_from_dir.return_value = (mock_models, mock_info)
            optimizer._objective(trial)

        # 各foldでリセットされていることを確認
        from domain.types import RegimeState
        for state in regime_states:
            assert state["_current_regime"] == RegimeState.CONSERVATIVE
            assert state["_regime_counter"] == 0
            assert state["_pending_regime"] is None
            assert state["_collapsed_consecutive"] == 0

    def test_ev_lower_set_on_submodels(self, optimizer: StrategyOptimizer) -> None:
        """EV_lower閾値がSubmodelSet属性(turf/dirt)に設定される"""
        mock_turf = MagicMock()
        mock_dirt = MagicMock()
        mock_models = MagicMock()
        mock_models.submodels = {"turf": mock_turf, "dirt": mock_dirt}
        mock_info = MagicMock()

        def mock_backtest_with_models(
            models, strategy_config, test_start, test_end,
            training_bet_history, fold_idx=0,
        ):
            return {"roi": 1.05, "n_bets": 2000}

        optimizer._run_single_backtest_with_models = mock_backtest_with_models

        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        with patch("db.model_loader.ModelLoader") as MockLoader, \
             patch.object(optimizer, "_generate_training_bet_history", return_value=[]):
            MockLoader.return_value.load_from_dir.return_value = (mock_models, mock_info)
            optimizer._objective(trial)

        # EV_lower閾値が設定されていることを確認
        assert mock_turf.ev_lower_threshold_turf is not None
        assert mock_dirt.ev_lower_threshold_dirt is not None


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


class TestBuildDefaultConfig:
    def test_delegates_to_default_strategy(self, optimizer: StrategyOptimizer) -> None:
        """_build_default_config()がbuild_default_strategy_config()にデリゲートする"""
        from betting.default_strategy import build_default_strategy_config
        expected = build_default_strategy_config()
        actual = optimizer._build_default_config()
        assert actual["fractional_kelly"] == expected["fractional_kelly"]
        assert actual["target_ev"] == expected["target_ev"]
        assert actual["max_scale"] == expected["max_scale"]
        assert actual["roi_threshold"] == expected["roi_threshold"]


class TestRunSingleBacktest:
    def test_calls_model_loader(self, optimizer: StrategyOptimizer) -> None:
        """_run_single_backtestがModelLoader.load_from_dirを呼び、
        トレーニング期間バックテスト + テスト期間バックテストの2回Engineを構築する"""
        mock_models = MagicMock()
        mock_info = MagicMock()
        mock_result = MagicMock()
        mock_result.total_roi = 1.10
        mock_result.total_bets = 5000
        mock_result.total_stake = 500000
        mock_result.total_return = 550000
        mock_result.max_drawdown = 0.15
        mock_result.bet_history = [{"race_id": "1", "odds": 5.0, "result": 500, "stake": 100}]

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
            # トレーニング期間用 + テスト期間用の2回BacktestEngineが構築される
            assert MockEngine.call_count == 2

            # テスト期間用のengine (2回目のcall) にstrategy_paramsが渡されたこと
            test_engine_call = MockEngine.call_args_list[-1]
            assert test_engine_call.kwargs.get("strategy_params") is not None

            # 最後のrun()呼び出しにtraining_bet_historyが渡されたこと
            last_run_call = MockEngine.return_value.run.call_args
            assert "training_bet_history" in last_run_call.kwargs
            assert last_run_call.kwargs["training_bet_history"] is not None

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
        mock_result.bet_history = []

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

    def test_training_uses_default_config_not_optuna(self, optimizer: StrategyOptimizer) -> None:
        """D-04: training backtest が default_config を使用し、test backtest が
        Optuna提案のstrategy_configを使用することを検証 (ルックアヘッド防止)"""
        mock_models = MagicMock()
        mock_info = MagicMock()
        mock_result = MagicMock()
        mock_result.total_roi = 1.10
        mock_result.total_bets = 5000
        mock_result.total_stake = 500000
        mock_result.total_return = 550000
        mock_result.max_drawdown = 0.15
        mock_result.bet_history = [{"race_id": "1", "odds": 5.0, "result": 500, "stake": 100}]

        with patch("db.model_loader.ModelLoader") as MockLoader, \
             patch("backtest.engine.BacktestEngine") as MockEngine:
            MockLoader.return_value.load_from_dir.return_value = (mock_models, mock_info)
            MockEngine.return_value.run.return_value = mock_result

            optuna_config = optimizer._build_strategy_config({
                "rolling_window": 400, "dd_threshold_1": 0.10, "dd_threshold_2": 0.20,
                "multiplier_reduced": 0.5, "min_stay_races": 10,
                "fk_aggressive": 0.6, "ev_aggressive": 1.2, "edge_aggressive": 0.08,
                "fk_conservative": 0.3, "ev_conservative": 1.4, "edge_conservative": 0.07,
                "target_ev": 1.15, "max_scale": 2.5, "roi_threshold": 0.95,
            })
            optimizer._run_single_backtest(optuna_config, "2024-01-01", "2024-12-31")

            assert MockEngine.call_count == 2
            # ステップ3 (train_engine): default_config を使用
            train_call = MockEngine.call_args_list[0]
            train_params = train_call.kwargs.get("strategy_params", {})
            default_config = optimizer._build_default_config()
            assert train_params["fractional_kelly"] == default_config["fractional_kelly"]
            assert train_params["target_ev"] == default_config["target_ev"]
            # ステップ4 (test_engine): Optuna提案のstrategy_config を使用
            test_call = MockEngine.call_args_list[1]
            test_params = test_call.kwargs.get("strategy_params", {})
            assert test_params["fractional_kelly"] == optuna_config["fractional_kelly"]
            assert test_params["target_ev"] == optuna_config["target_ev"]

    def test_regime_overrides_switched_between_train_and_test(self, optimizer: StrategyOptimizer) -> None:
        """D-09: デフォルトregime_overridesがtrain用に注入され、
        その後Optuna regime_overridesで上書きされることを検証"""
        mock_models = MagicMock()
        mock_info = MagicMock()
        mock_result = MagicMock()
        mock_result.total_roi = 1.05
        mock_result.total_bets = 2000
        mock_result.total_stake = 200000
        mock_result.total_return = 210000
        mock_result.max_drawdown = 0.10
        mock_result.bet_history = []

        with patch("db.model_loader.ModelLoader") as MockLoader, \
             patch("backtest.engine.BacktestEngine") as MockEngine:
            MockLoader.return_value.load_from_dir.return_value = (mock_models, mock_info)
            MockEngine.return_value.run.return_value = mock_result

            optuna_config = optimizer._build_strategy_config({
                "rolling_window": 400, "dd_threshold_1": 0.10, "dd_threshold_2": 0.20,
                "multiplier_reduced": 0.5, "min_stay_races": 10,
                "fk_aggressive": 0.6, "ev_aggressive": 1.2, "edge_aggressive": 0.08,
                "fk_conservative": 0.3, "ev_conservative": 1.4, "edge_conservative": 0.07,
                "target_ev": 1.15, "max_scale": 2.5, "roi_threshold": 0.95,
            })
            optimizer._run_single_backtest(optuna_config, "2024-01-01", "2024-12-31")

            # 最終的にOptuna値で上書きされていること
            final_overrides = mock_models.regime_detector._override_params
            assert final_overrides["aggressive"]["fractional_kelly"] == 0.6
            assert final_overrides["conservative"]["fractional_kelly"] == 0.3
