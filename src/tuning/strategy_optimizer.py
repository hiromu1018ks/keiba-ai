"""Optuna戦略パラメータ最適化 (D-07, D-12)

Walk-forward枠組みで16次元戦略パラメータをOptuna TPE最適化。
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
    D-08: 16次元探索空間
    D-09: ROI主 + ベット数制約
    D-10: Walk-forward枠組み (独自軽量ループ)
    D-11: 100トライアル + MedianPruner
    """

    def __init__(
        self,
        models_dir: Path | str = "data/models",
        data_dir: Path | str = "data",
        initial_bankroll: float = 100_000,
        n_folds: int = 4,
        train_years: int = 4,
        test_years: int = 1,
        min_bets_per_fold: int = 1000,
        fold_start_year: int = 2022,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.initial_bankroll = initial_bankroll
        self.n_folds = n_folds
        self.train_years = train_years
        self.test_years = test_years
        self.min_bets_per_fold = min_bets_per_fold
        self.fold_start_year = fold_start_year

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """D-08: 16次元探索空間"""
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

        # D-07: EV_lower閾値 15-16次元目 (サーフェス別)
        params["ev_lower_threshold_turf"] = trial.suggest_float(
            "ev_lower_threshold_turf", 0.5, 1.5,
        )
        params["ev_lower_threshold_dirt"] = trial.suggest_float(
            "ev_lower_threshold_dirt", 0.5, 1.5,
        )

        return params

    def _build_strategy_config(self, params: dict[str, Any]) -> dict[str, Any]:
        """Optuna params -> BacktestEngine injection用dictに変換"""
        from betting.default_strategy import build_strategy_config_from_params
        return build_strategy_config_from_params(params)

    def _build_default_config(self) -> dict[str, Any]:
        """RegimeDetector既定値からデフォルトstrategy_configを構築 (ルックアヘッド防止)

        D-01: デフォルトパラメータソースはRegimeDetector._get_base_params()のハードコード既定値
        D-02: 16次元全てを適用
        D-03: _build_strategy_config()と並存
        Warning 1修正: default_strategy.pyにデリゲート (重複実装排除)
        """
        from betting.default_strategy import build_default_strategy_config
        return build_default_strategy_config()

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
        from domain.types import RegimeState

        # 1. 学習済みモデルをロード (学習はしない)
        loader = ModelLoader()
        models, info = loader.load_from_dir(self.models_dir, use_ensemble_override=True)
        logger.info(
            "Fold %d: loaded models from %s (trained %s ~ %s)",
            fold_idx, self.models_dir, info.train_start, info.train_end,
        )

        # 2. デフォルトconfigでtraining_bet_history生成 (D-04: ルックアヘッド防止)
        default_config = self._build_default_config()

        # 2a. デフォルトregime_overridesを一時的に注入
        default_regime_overrides = default_config.get("regime_overrides")
        if default_regime_overrides:
            models.regime_detector._override_params = default_regime_overrides

        # 3. トレーニング期間バックテスト (デフォルトパラメータで実行)
        train_start = info.train_start
        train_end = info.train_end
        training_bet_history: list[dict[str, Any]] | None = None
        try:
            train_engine = BacktestEngine(
                models=models,
                initial_bankroll=self.initial_bankroll,
                betting_mode="kelly",
                diag_prefix=f"opt_train_fold{fold_idx}",
                betting_target="win",
                strategy_params=default_config,
            )
            train_result = train_engine.run(train_start, train_end)
            training_bet_history = train_result.bet_history
            logger.info(
                "Fold %d: training-phase backtest completed (%d bets, ROI=%.1f%%)",
                fold_idx, train_result.total_bets, train_result.total_roi * 100,
            )
        except Exception as e:
            logger.warning("Fold %d: training-phase backtest failed: %s — skipping calibration",
                           fold_idx, e)

        # 2b. Optuna値のregime_overridesで上書き (テスト期間用)
        regime_overrides = strategy_config.get("regime_overrides")
        if regime_overrides:
            models.regime_detector._override_params = regime_overrides

        # CR-01: Reset mutable state to prevent training-to-test leakage
        models.regime_detector._current_regime = RegimeState.CONSERVATIVE
        models.regime_detector._regime_counter = 0
        models.regime_detector._pending_regime = None
        models.regime_detector._collapsed_consecutive = 0

        # 4. BacktestEngine構築 (strategy_params注入)
        engine = BacktestEngine(
            models=models,
            initial_bankroll=self.initial_bankroll,
            betting_mode="kelly",
            diag_prefix=f"opt_fold{fold_idx}",
            betting_target="win",
            strategy_params=strategy_config,
        )

        # 5. バックテスト実行 (training_bet_history を OddsBandFilter キャリブレーションに使用)
        result = engine.run(test_start, test_end, training_bet_history=training_bet_history)

        return {
            "roi": result.total_roi,
            "n_bets": result.total_bets,
            "total_stake": result.total_stake,
            "total_return": result.total_return,
            "max_drawdown": result.max_drawdown,
        }

    def _generate_training_bet_history(
        self,
        models: Any,
        info: Any,
        default_config: dict[str, Any],
    ) -> list[dict[str, Any]] | None:
        """D-05: training_bet_historyをデフォルトパラメータで1回生成"""
        from backtest.engine import BacktestEngine
        try:
            train_engine = BacktestEngine(
                models=models,
                initial_bankroll=self.initial_bankroll,
                betting_mode="kelly",
                diag_prefix="opt_train",
                betting_target="win",
                strategy_params=default_config,
            )
            train_result = train_engine.run(info.train_start, info.train_end)
            return train_result.bet_history
        except Exception as e:
            logger.warning("Training-phase backtest failed: %s — skipping calibration", e)
            return None

    def _run_single_backtest_with_models(
        self,
        models: Any,
        strategy_config: dict[str, Any],
        test_start: str,
        test_end: str,
        training_bet_history: list[dict[str, Any]] | None,
        fold_idx: int = 0,
    ) -> dict[str, Any]:
        """D-05: モデル共有版バックテスト実行"""
        from backtest.engine import BacktestEngine
        engine = BacktestEngine(
            models=models,
            initial_bankroll=self.initial_bankroll,
            betting_mode="kelly",
            diag_prefix=f"opt_fold{fold_idx}",
            betting_target="win",
            strategy_params=strategy_config,
        )
        result = engine.run(test_start, test_end, training_bet_history=training_bet_history)
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

        # D-05: モデルロード最適化 — trial内1回
        from db.model_loader import ModelLoader
        loader = ModelLoader()
        models, info = loader.load_from_dir(self.models_dir, use_ensemble_override=True)

        # D-05: training_bet_historyキャッシュ — trial内1回
        default_config = self._build_default_config()
        default_regime_overrides = default_config.get("regime_overrides")
        if default_regime_overrides:
            models.regime_detector._override_params = default_regime_overrides
        training_bet_history = self._generate_training_bet_history(
            models, info, default_config,
        )

        # D-07: EV_lower閾値をSubmodelSet属性に設定
        ev_lower_turf = params.get("ev_lower_threshold_turf", 1.0)
        ev_lower_dirt = params.get("ev_lower_threshold_dirt", 1.0)
        for surf_key, sm in models.submodels.items():
            if surf_key == "turf":
                sm.ev_lower_threshold_turf = ev_lower_turf
            elif surf_key == "dirt":
                sm.ev_lower_threshold_dirt = ev_lower_dirt

        # Walk-forward fold評価
        folds = self._generate_folds()
        rois: list[float] = []
        total_bets = 0

        for fold_idx, (test_start, test_end) in enumerate(folds):
            # CR-01: RegimeDetector状態リセット
            from domain.types import RegimeState
            models.regime_detector._current_regime = RegimeState.CONSERVATIVE
            models.regime_detector._regime_counter = 0
            models.regime_detector._pending_regime = None
            models.regime_detector._collapsed_consecutive = 0

            # regime_overridesをOptuna値で上書き
            regime_overrides = strategy_config.get("regime_overrides")
            if regime_overrides:
                models.regime_detector._override_params = regime_overrides

            result = self._run_single_backtest_with_models(
                models, strategy_config, test_start, test_end,
                training_bet_history, fold_idx,
            )
            rois.append(result.get("roi", 0.0))
            total_bets += result.get("n_bets", 0)

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

    def optimize_multi_seed(
        self,
        n_trials: int = 100,
        seeds: list[int] | None = None,
        output_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        """D-08/D-09/D-10: multi-seed安定性検証

        Args:
            n_trials: 主seedの試行回数。追加seedはn_trials//2
            seeds: seedリスト。デフォルト[42, 43, 44]
            output_dir: 安定性レポート出力ディレクトリ

        Returns:
            安定性レポートdict
        """
        import json as _json
        from datetime import datetime, timezone

        if seeds is None:
            seeds = [42, 43, 44]

        # D-08: 各seedで最適化実行
        seed_results: dict[int, dict[str, Any]] = {}
        for i, seed in enumerate(seeds):
            n = n_trials if i == 0 else n_trials // 2  # D-08: 主100 + 追加50
            logger.info(
                "Multi-seed optimization: seed=%d, n_trials=%d (%d/%d)",
                seed, n, i + 1, len(seeds),
            )
            result = self.optimize(n_trials=n, seed=seed)
            seed_results[seed] = result

        # D-09: 安定性分析
        stability_report = self._compute_stability_report(seed_results)
        stability_report["version"] = "1.0"
        stability_report["timestamp"] = datetime.now(timezone.utc).isoformat()
        stability_report["seeds"] = seeds

        # D-10: 不安定次元の固定化と再最適化
        unstable_dims = [
            d for d, info in stability_report["dimensions"].items()
            if info["is_unstable"]
        ]
        if unstable_dims:
            logger.info(
                "Unstable dimensions detected: %s — running reoptimization",
                unstable_dims,
            )
            fixed_report = self._optimize_with_fixed_dims(
                unstable_dims, seed_results[seeds[0]], n_trials, seeds,
            )
            stability_report["reoptimization"] = fixed_report
        else:
            stability_report["reoptimization"] = None
            logger.info("All dimensions stable across seeds")

        # レポート出力
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            report_path = output_path / "stability_report.json"
            report_path.write_text(
                _json.dumps(stability_report, indent=2, default=str),
                encoding="utf-8",
            )
            logger.info("Stability report saved: %s", report_path)

            # 最良パラメータのmanifest保存
            best_seed = max(
                stability_report["best_roi_by_seed"].items(), key=lambda x: x[1],
            )
            manifest_path = output_path / "strategy_manifest.json"
            from backtest.parameter_freeze_protocol import save_strategy_manifest

            save_strategy_manifest(
                seed_results[int(best_seed[0])]["best_params"], manifest_path,
            )

        return stability_report

    def _compute_stability_report(
        self,
        seed_results: dict[int, dict[str, Any]],
    ) -> dict[str, Any]:
        """D-09: 3 seed間のパラメータ安定性分析

        CV(変動係数) = std/mean で各次元の安定性を評価。
        CV > 0.20 を「不安定」と判定。
        """
        seeds = list(seed_results.keys())
        all_param_names = list(seed_results[seeds[0]]["best_params"].keys())

        dimensions: dict[str, dict[str, Any]] = {}
        for param_name in all_param_names:
            values = [
                seed_results[s]["best_params"].get(param_name, 0.0)
                for s in seeds
            ]
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            cv = std_val / mean_val if abs(mean_val) > 1e-10 else 0.0

            dimensions[param_name] = {
                "values": values,
                "mean": mean_val,
                "std": std_val,
                "cv": cv,
                "is_unstable": cv > 0.20,
            }

        best_roi_by_seed: dict[str, float] = {
            str(s): seed_results[s]["best_value"]
            for s in seeds
        }

        return {
            "dimensions": dimensions,
            "best_roi_by_seed": best_roi_by_seed,
            "mean_best_roi": float(np.mean(list(best_roi_by_seed.values()))),
        }

    def _optimize_with_fixed_dims(
        self,
        unstable_dims: list[str],
        primary_result: dict[str, Any],
        n_trials: int,
        seeds: list[int],
    ) -> dict[str, Any]:
        """D-10: 不安定次元をデフォルト値に固定し再最適化

        1回のみ再実行(Pitfall 5回避)。残りの不安定次元は
        レポートで報告しPhase 18の最終検証に委ねる。
        """
        # 主seedのbest_paramsをベースに固定次元を設定
        fixed_params = dict(primary_result["best_params"])

        # 不安定次元をデフォルト値に固定
        # T-17-07: デフォルト値をハードコード(RegimeDetector._get_base_params()と一致)
        default_param_values: dict[str, float] = {
            "fk_aggressive": 0.45, "fk_conservative": 0.30,
            "ev_aggressive": 1.10, "ev_conservative": 1.30,
            "edge_aggressive": 0.05, "edge_conservative": 0.06,
            "dd_threshold_1": 0.10, "dd_threshold_2": 0.25,
            "multiplier_reduced": 0.5, "rolling_window": 400,
            "min_stay_races": 10, "target_ev": 1.10,
            "max_scale": 2.0, "roi_threshold": 1.0,
            "ev_lower_threshold_turf": 1.0,
            "ev_lower_threshold_dirt": 1.0,
        }
        for dim in unstable_dims:
            fixed_params[dim] = default_param_values.get(dim, 1.0)

        # 固定次元を考慮した_objective_fixed()で再最適化
        original_suggest = self._suggest_params

        def _suggest_fixed(trial: optuna.Trial) -> dict[str, Any]:
            params = original_suggest(trial)
            for dim in unstable_dims:
                if dim in params:
                    del params[dim]
            for dim in unstable_dims:
                params[dim] = fixed_params[dim]
            return params

        # 一時的に_suggest_paramsを差し替え
        self._suggest_params = _suggest_fixed  # type: ignore[assignment]

        try:
            reopt_result = self.optimize(n_trials=n_trials, seed=seeds[0])
        finally:
            self._suggest_params = original_suggest  # type: ignore[assignment]

        return {
            "fixed_dimensions": unstable_dims,
            "fixed_values": {d: fixed_params[d] for d in unstable_dims},
            "reoptimized_best_params": reopt_result["best_params"],
            "reoptimized_best_roi": reopt_result["best_value"],
        }

    def _generate_folds(self) -> list[tuple[str, str]]:
        """D-02: 年次fold動的生成 (コンストラクタ引数ベース)"""
        folds: list[tuple[str, str]] = []
        for i in range(self.n_folds):
            year = self.fold_start_year + i
            folds.append((f"{year}-01-01", f"{year}-12-31"))
        return folds

    def optimize(
        self,
        n_trials: int = 100,
        seed: int = 42,
        output_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """D-11: Optuna最適化実行 + D-14: manifest自動生成"""
        sampler = TPESampler(seed=seed)
        pruner = MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=0,
            interval_steps=1,
            n_min_trials=1,
        )
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
