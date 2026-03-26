"""BacktestEngine のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from domain.models import SubmodelSet, TrainedModelsV5
from domain.types import RegimeState


@pytest.fixture
def mock_models() -> MagicMock:
    """モック TrainedModelsV5"""
    models = MagicMock(spec=TrainedModelsV5)
    models.submodels = {"turf": MagicMock(spec=SubmodelSet)}
    models.quality_screener = MagicMock()
    models.quality_screener.should_bet.return_value = True
    models.regime_detector = MagicMock()
    models.regime_detector.current_regime = RegimeState.CONSERVATIVE
    models.regime_detector.get_strategy_params.return_value = {
        "ev_threshold": 1.20,
        "score_threshold": 0.015,
        "max_bets_per_race": 3,
    }
    return models


class TestBacktestResult:
    """BacktestResult データクラスのテスト"""

    def test_result_structure(self) -> None:
        """BacktestResult が正しい構造を持つ"""
        from backtest.engine import BacktestResult

        result = BacktestResult(
            total_bets=100,
            total_stake=100000,
            total_return=105000,
            winning_bets=30,
            total_roi=1.05,
            max_drawdown=0.08,
            monthly_returns={},
            bet_history=[],
        )
        assert result.total_roi == 1.05
        assert result.total_return - result.total_stake == 5000

    def test_profit_property(self) -> None:
        """profit プロパティが正しく計算される"""
        from backtest.engine import BacktestResult

        result = BacktestResult(total_stake=1000, total_return=1200)
        assert result.profit == 200.0

    def test_summary_format(self) -> None:
        """summary() が文字列を返す"""
        from backtest.engine import BacktestResult

        result = BacktestResult(
            total_bets=50,
            total_stake=50000,
            total_return=55000,
            total_roi=1.10,
            max_drawdown=0.05,
            final_bankroll=105000,
        )
        s = result.summary()
        assert "50" in s
        assert "110.000%" in s


class TestBacktestEngine:
    """BacktestEngine のテスト"""

    def test_init_with_models(self, mock_models: MagicMock) -> None:
        """モデル付きで初期化できる"""
        from backtest.engine import BacktestEngine

        engine = BacktestEngine(models=mock_models)
        assert engine.models is mock_models

    def test_init_with_bankroll(self, mock_models: MagicMock) -> None:
        """初期資金を設定できる"""
        from backtest.engine import BacktestEngine

        engine = BacktestEngine(models=mock_models, initial_bankroll=200000)
        assert engine.initial_bankroll == 200000

    @patch("backtest.engine.DatabaseConnection")
    def test_run_returns_backtest_result(
        self,
        mock_db_cls: MagicMock,
        mock_models: MagicMock,
    ) -> None:
        """run() が BacktestResult を返す"""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db
        mock_db.load_races.return_value = pd.DataFrame()
        mock_db.load_entries_with_results.return_value = pd.DataFrame()
        mock_db.load_odds_snapshots.return_value = pd.DataFrame()

        from backtest.engine import BacktestEngine

        engine = BacktestEngine(models=mock_models)
        result = engine.run("2024-01-01", "2024-12-31")

        assert hasattr(result, "total_roi")
        assert hasattr(result, "max_drawdown")
        assert hasattr(result, "total_bets")

    @patch("backtest.engine.DatabaseConnection")
    def test_empty_period_returns_zero_bets(
        self,
        mock_db_cls: MagicMock,
        mock_models: MagicMock,
    ) -> None:
        """レースがない期間は0ベット"""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db
        mock_db.load_races.return_value = pd.DataFrame()
        mock_db.load_entries_with_results.return_value = pd.DataFrame()
        mock_db.load_odds_snapshots.return_value = pd.DataFrame()

        from backtest.engine import BacktestEngine

        engine = BacktestEngine(models=mock_models)
        result = engine.run("2024-01-01", "2024-12-31")

        assert result.total_bets == 0

    @patch("backtest.engine.DatabaseConnection")
    def test_bankroll_tracking(
        self,
        mock_db_cls: MagicMock,
        mock_models: MagicMock,
    ) -> None:
        """資金の推移が追跡される"""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db
        mock_db.load_races.return_value = pd.DataFrame()
        mock_db.load_entries_with_results.return_value = pd.DataFrame()
        mock_db.load_odds_snapshots.return_value = pd.DataFrame()

        from backtest.engine import BacktestEngine

        engine = BacktestEngine(models=mock_models, initial_bankroll=100000)
        result = engine.run("2024-01-01", "2024-12-31")

        # 空期間なので資金は変化しない
        assert result.final_bankroll == 100000

    @patch("backtest.engine.DatabaseConnection")
    def test_default_result_values(
        self,
        mock_db_cls: MagicMock,
        mock_models: MagicMock,
    ) -> None:
        """空データのデフォルト値が正しい"""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db
        mock_db.load_races.return_value = pd.DataFrame()
        mock_db.load_entries_with_results.return_value = pd.DataFrame()
        mock_db.load_odds_snapshots.return_value = pd.DataFrame()

        from backtest.engine import BacktestEngine

        engine = BacktestEngine(models=mock_models)
        result = engine.run("2024-01-01", "2024-12-31")

        assert result.total_stake == 0.0
        assert result.total_return == 0.0
        assert result.total_roi == 0.0
        assert result.max_drawdown == 0.0
        assert result.winning_bets == 0
