"""Optuna戦略パラメータ最適化 (D-07, D-12)

Walk-forward枠組みで~16次元戦略パラメータをOptuna TPE最適化。
学習済みモデルをModelLoader.load_from_dir()でロードしてバックテストのみ実行。
WalkForwardCVは使用せず、独自軽量WFループでfold評価(pipeline.run回避)。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState

logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """全戦略パラメータのOptuna TPE最適化

    D-07: 全パラメータ一括最適化
    D-08: ~16次元探索空間
    D-09: ROI主 + ベット数制約
    D-10: Walk-forward枠組み (独自軽量ループ)
    D-11: 100トライアル + MedianPruner
    """

    def __init__(
        self,
        models_dir: Path | str = "data/models",
        data_dir: Path | str = "data",
        initial_bankroll: float = 100_000,
        n_folds: int = 2,
        train_years: int = 4,
        test_years: int = 1,
        min_bets_per_fold: int = 1000,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.initial_bankroll = initial_bankroll
        self.n_folds = n_folds
        self.train_years = train_years
        self.test_years = test_years
        self.min_bets_per_fold = min_bets_per_fold

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """D-08: ~16次元探索空間"""
        params: dict[str, Any] = {}

        # レジーム別パラメータ (AGGRESSIVE, CONSERVATIVEのみ。COLLAPSEDはfk=0固定)
        for regime in ("aggressive", "conservative"):
            params[f"fk_{regime}"] = trial.suggest_float(
                f"fk_{regime}", 0.10, 0.80,
            )
            params[f"ev_{regime}"] = trial.suggest_float(
                f"ev_{regime}", 1.05, 2.00,
            )
            params[f"edge_{regime}"] = trial.suggest_float(
                f"edge_{regime}", 0.03, 0.15,
            )

        # DD制御パラメータ
        params["dd_threshold_1"] = trial.suggest_float("dd_threshold_1", 0.05, 0.20)
        params["dd_threshold_2"] = trial.suggest_float("dd_threshold_2", 0.15, 0.35)
        params["multiplier_reduced"] = trial.suggest_float("multiplier_reduced", 0.1, 0.8)
        params["rolling_window"] = trial.suggest_int("rolling_window", 200, 800)
        params["min_stay_races"] = trial.suggest_int("min_stay_races", 5, 30)

        # EVスケーリング
        params["target_ev"] = trial.suggest_float("target_ev", 1.05, 1.50)
        params["max_scale"] = trial.suggest_float("max_scale", 1.0, 3.0)

        # OddsBandFilter ROI閾値
        params["roi_threshold"] = trial.suggest_float("roi_threshold", 0.8, 1.2)

        return params

    def _build_strategy_config(self, params: dict[str, Any]) -> dict[str, Any]:
        """Optuna params -> BacktestEngine injection用dictに変換"""
        from betting.drawdown_controller import DDConfig

        # T-13-06: dd_threshold_2 > dd_threshold_1 を保証 (DDConfig.__post_init__ 制約)
        dd_t1 = params["dd_threshold_1"]
        dd_t2 = params["dd_threshold_2"]
        if dd_t2 <= dd_t1:
            dd_t2 = dd_t1 + 0.01

        dd_config = DDConfig(
            rolling_window=params["rolling_window"],
            dd_threshold_1=dd_t1,
            dd_threshold_2=dd_t2,
            multiplier_reduced=params["multiplier_reduced"],
            min_stay_races=params["min_stay_races"],
        )

        regime_overrides = {}
        for regime in ("aggressive", "conservative"):
            regime_overrides[regime] = {
                "fractional_kelly": params[f"fk_{regime}"],
                "ev_threshold": params[f"ev_{regime}"],
                "edge_threshold": params[f"edge_{regime}"],
            }

        return {
            "dd_config": dd_config,
            "regime_overrides": regime_overrides,
            "fractional_kelly": params.get("fk_aggressive", 0.5),
            "target_ev": params["target_ev"],
            "max_scale": params["max_scale"],
            "roi_threshold": params["roi_threshold"],
        }

    def _run_single_backtest(
        self,
        strategy_config: dict[str, Any],
        test_start: str,
        test_end: str,
        trial: optuna.Trial | None = None,
        fold_idx: int = 0,
    ) -> dict[str, Any]:
        """単一foldバックテスト実行 (具象実装)

        ModelLoader.load_from_dir()で学習済みモデルをロードし、
        BacktestEngineにstrategy_paramsを注入してバックテストのみ実行。
        pipeline.run()は呼ばない(RESEARCH Pitfall 3回避)。
        """
        from backtest.engine import BacktestEngine
        from db.model_loader import ModelLoader

        # 1. 学習済みモデルをロード (学習はしない)
        loader = ModelLoader()
        models, info = loader.load_from_dir(self.models_dir, use_ensemble_override=True)
        logger.info(
            "Fold %d: loaded models from %s (trained %s ~ %s)",
            fold_idx, self.models_dir, info.train_start, info.train_end,
        )

        # 2. RegimeDetector override (Plan 02成果)
        regime_overrides = strategy_config.get("regime_overrides")
        if regime_overrides:
            from models.regime_detector import RegimeDetector

            models.regime_detector = RegimeDetector(
                override_params=regime_overrides,
            )

        # 3. BacktestEngine構築 (strategy_params注入)
        engine = BacktestEngine(
            models=models,
            initial_bankroll=self.initial_bankroll,
            betting_mode="kelly",
            diag_prefix=f"opt_fold{fold_idx}",
            betting_target="win",
            strategy_params=strategy_config,
        )

        # 4. バックテスト実行
        result = engine.run(test_start, test_end)

        return {
            "roi": result.total_roi,
            "n_bets": result.total_bets,
            "total_stake": result.total_stake,
            "total_return": result.total_return,
            "max_drawdown": result.max_drawdown,
        }

    def _objective(self, trial: optuna.Trial) -> float:
        """D-09: ROI主 + ベット数制約"""
        params = self._suggest_params(trial)
        strategy_config = self._build_strategy_config(params)

        # Walk-forward 2fold評価
        folds = self._generate_folds()
        rois: list[float] = []
        total_bets = 0

        for fold_idx, (test_start, test_end) in enumerate(folds):
            result = self._run_single_backtest(
                strategy_config, test_start, test_end, trial, fold_idx,
            )
            rois.append(result.get("roi", 0.0))
            total_bets += result.get("n_bets", 0)

            # D-11: MedianPruner用中間報告
            if trial is not None:
                trial.report(result.get("roi", 0.0), step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        mean_roi = float(np.mean(rois)) if rois else 0.0

        # D-09: ベット数制約 (foldあたりmin_bets_per_fold)
        avg_bets_per_fold = total_bets / max(len(folds), 1)
        if avg_bets_per_fold < self.min_bets_per_fold:
            return -1.0  # ペナルティ

        return mean_roi

    def _generate_folds(self) -> list[tuple[str, str]]:
        """WF fold定義。run_wf_validation.pyのパターンを踏襲。

        デフォルト: 2024年テスト、2025年テストの2fold。
        CLI引数で上書き可能。
        """
        return [
            ("2024-01-01", "2024-12-31"),
            ("2025-01-01", "2025-12-31"),
        ]

    def optimize(
        self,
        n_trials: int = 100,
        seed: int = 42,
        output_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """D-11: Optuna最適化実行 + D-14: manifest自動生成"""
        sampler = TPESampler(seed=seed)
        pruner = MedianPruner()
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )
        study.optimize(self._objective, n_trials=n_trials)

        result: dict[str, Any] = {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "n_pruned": sum(
                1 for t in study.trials if t.state == TrialState.PRUNED
            ),
        }

        # D-14: JSON manifest自動生成
        if output_path is not None:
            from backtest.parameter_freeze_protocol import save_strategy_manifest

            save_strategy_manifest(result["best_params"], Path(output_path))

        logger.info(
            "Optimization complete: best_roi=%.4f, trials=%d, pruned=%d",
            result["best_value"], result["n_trials"], result["n_pruned"],
        )
        return result
