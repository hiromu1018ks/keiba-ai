"""デフォルト戦略パラメータ構築ユーティリティ (ルックアヘッド防止)

RegimeDetector._get_base_params()のハードコード既定値から
BacktestEngine strategy_config dictを構築する。
strategy_optimizer.py._build_strategy_config()、
strategy_optimizer.py._build_default_config()、
BacktestEngine._generate_training_bet_history()、
run_backtest.py._build_strategy_config_from_manifest()、
run_backtest.py._collect_training_bet_history()の全てで使用。
"""
from __future__ import annotations

from typing import Any

from betting.drawdown_controller import DDConfig
from domain.types import RegimeState
from models.regime_detector import RegimeDetector


def build_strategy_config_from_params(params: dict[str, Any]) -> dict[str, Any]:
    """Optunaフラットparams (best_params 形式) を BacktestEngine strategy_config に変換。

    StrategyOptimizer._build_strategy_config() と
    run_backtest._build_strategy_config_from_manifest() の共通実装。
    params.get() でアクセスし、キー欠損時はDDConfig既定値を使用する。

    Args:
        params: Optuna best_params 形式のフラットdict。
            キー: dd_threshold_1, dd_threshold_2, rolling_window,
                  multiplier_reduced, min_stay_races, fk_aggressive,
                  fk_conservative, ev_aggressive, ev_conservative,
                  edge_aggressive, edge_conservative, target_ev, max_scale,
                  roi_threshold

    Returns:
        dict: BacktestEngineのstrategy_paramsとして使用可能な設定辞書。
    """
    # T-13-06: dd_threshold_2 > dd_threshold_1 を保証 (DDConfig.__post_init__ 制約)
    dd_t1 = params.get("dd_threshold_1", 0.10)
    dd_t2 = params.get("dd_threshold_2", 0.25)
    if dd_t2 <= dd_t1:
        dd_t2 = dd_t1 + 0.01

    dd_config = DDConfig(
        rolling_window=params.get("rolling_window", 400),
        dd_threshold_1=dd_t1,
        dd_threshold_2=dd_t2,
        multiplier_reduced=params.get("multiplier_reduced", 0.5),
        min_stay_races=params.get("min_stay_races", 10),
    )

    regime_overrides: dict[str, dict[str, float]] = {}
    for regime in ("aggressive", "conservative"):
        regime_overrides[regime] = {
            "fractional_kelly": params.get(f"fk_{regime}", 0.5),
            "ev_threshold": params.get(f"ev_{regime}", 1.10),
            "edge_threshold": params.get(f"edge_{regime}", 0.05),
        }

    return {
        "dd_config": dd_config,
        "regime_overrides": regime_overrides,
        "fractional_kelly": params.get("fk_aggressive", 0.5),
        "target_ev": params.get("target_ev", 1.10),
        "max_scale": params.get("max_scale", 2.0),
        "roi_threshold": params.get("roi_threshold", 1.0),
    }


def build_default_strategy_config() -> dict[str, Any]:
    """RegimeDetector既定値からデフォルトstrategy_configを構築 (ルックアヘッド防止)。

    Returns:
        dict: BacktestEngineのstrategy_paramsとして使用可能な設定辞書。
            キー: dd_config, regime_overrides, fractional_kelly, target_ev, max_scale, roi_threshold
    """
    # _get_base_params()はインスタンス状態に依存しないハードコード値を返す
    detector = RegimeDetector()
    conservative_params = detector._get_base_params(RegimeState.CONSERVATIVE)

    # DDConfigデフォルト値 (Optuna探索範囲内にあることを確認済み: Pitfall 2)
    dd_config = DDConfig()

    # 各レジームのハードコード既定値を取得
    regime_overrides: dict[str, dict[str, float]] = {}
    for regime_key in ("aggressive", "conservative"):
        state = RegimeState(regime_key)
        base = detector._get_base_params(state)
        regime_overrides[regime_key] = {
            "fractional_kelly": base["fractional_kelly"],
            "ev_threshold": base["ev_threshold"],
            "edge_threshold": base["edge_threshold"],
        }

    return {
        "dd_config": dd_config,
        "regime_overrides": regime_overrides,
        # トップレベルfractional_kellyはCONSERVATIVE値 (最も中立な状態 per Pitfall 1)
        "fractional_kelly": conservative_params["fractional_kelly"],
        # StakeCalculatorデフォルト
        "target_ev": 1.10,
        "max_scale": 2.0,
        # OddsBandFilterデフォルト
        "roi_threshold": 1.0,
    }
