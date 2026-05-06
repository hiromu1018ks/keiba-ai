"""デフォルト戦略パラメータ構築ユーティリティ (ルックアヘッド防止)

RegimeDetector._get_base_params()のハードコード既定値から
BacktestEngine strategy_config dictを構築する。
strategy_optimizer.py._build_default_config()、
BacktestEngine._generate_training_bet_history()、
run_backtest.py._collect_training_bet_history()の全てで使用。
"""
from __future__ import annotations

from typing import Any

from betting.drawdown_controller import DDConfig
from domain.types import RegimeState
from models.regime_detector import RegimeDetector


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
