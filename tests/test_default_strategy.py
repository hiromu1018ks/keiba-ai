"""src/betting/default_strategy.py 共通ユーティリティテスト (D-01/D-02)"""
from __future__ import annotations

import pytest


class TestBuildDefaultStrategyConfig:
    """build_default_strategy_config()のテスト"""

    def test_returns_all_required_keys(self):
        from betting.default_strategy import build_default_strategy_config
        config = build_default_strategy_config()
        for key in ("dd_config", "regime_overrides", "fractional_kelly",
                    "target_ev", "max_scale", "roi_threshold"):
            assert key in config, f"Missing key: {key}"

    def test_dd_config_defaults(self):
        from betting.default_strategy import build_default_strategy_config
        config = build_default_strategy_config()
        dd = config["dd_config"]
        assert dd.rolling_window == 400
        assert dd.dd_threshold_1 == 0.10
        assert dd.dd_threshold_2 == 0.20
        assert dd.multiplier_reduced == 0.50
        assert dd.min_stay_races == 10

    def test_aggressive_fractional_kelly(self):
        from betting.default_strategy import build_default_strategy_config
        config = build_default_strategy_config()
        agg = config["regime_overrides"]["aggressive"]
        assert agg["fractional_kelly"] == 0.50
        assert agg["ev_threshold"] == 1.10
        assert agg["edge_threshold"] == 0.05

    def test_conservative_fractional_kelly(self):
        from betting.default_strategy import build_default_strategy_config
        config = build_default_strategy_config()
        con = config["regime_overrides"]["conservative"]
        assert con["fractional_kelly"] == 0.25
        assert con["ev_threshold"] == 1.30
        assert con["edge_threshold"] == 0.06

    def test_top_level_fractional_kelly_uses_conservative(self):
        from betting.default_strategy import build_default_strategy_config
        config = build_default_strategy_config()
        assert config["fractional_kelly"] == 0.25

    def test_stake_calculator_defaults(self):
        from betting.default_strategy import build_default_strategy_config
        config = build_default_strategy_config()
        assert config["target_ev"] == 1.10
        assert config["max_scale"] == 2.0
        assert config["roi_threshold"] == 1.0
