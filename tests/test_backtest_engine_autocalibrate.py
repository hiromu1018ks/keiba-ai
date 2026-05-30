"""BacktestEngine.run()内自動training_bet_history生成テスト (D-05/D-06/D-07)"""
from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestGenerateTrainingBetHistory:
    """_generate_training_bet_history()単体テスト"""

    def _make_mock_engine(self, train_period=("2020-01-01", "2023-12-31")):
        """テスト用BacktestEngineインスタンスを作成（mock models）"""
        from backtest.engine import BacktestEngine
        mock_models = MagicMock()
        mock_models.regime_detector = MagicMock()
        mock_models.train_period = train_period
        return BacktestEngine(
            models=mock_models,
            initial_bankroll=100_000,
            store=MagicMock(),
            betting_mode="kelly",
            betting_target="win",
            strategy_params={"roi_threshold": 1.0},
        )

    def test_uses_models_train_period_not_test_args(self):
        """D-07: _generate_training_bet_history()がself.models.train_periodから
        train_start/train_endを取得し、test_start/test_endを使用しない"""
        engine = self._make_mock_engine(train_period=("2019-06-01", "2023-05-31"))

        mock_result = MagicMock()
        mock_result.total_bets = 500
        mock_result.total_roi = 0.95
        mock_result.bet_history = [{"odds": 5.0, "result": 500, "stake": 100}]

        with patch("backtest.engine.BacktestEngine") as MockEngine:  # noqa: N806
            MockEngine.return_value.run.return_value = mock_result
            engine._generate_training_bet_history()

            # 内部エンジンのrun()がtrain_periodの値で呼ばれることを確認
            inner_run_call = MockEngine.return_value.run
            inner_run_call.assert_called_once_with("2019-06-01", "2023-05-31")

    def test_uses_same_betting_target_to_generate_bets(self):
        """Phase 43.5 FIX: 内部エンジンは同じbetting_targetで構築し、
        _skip_odds_band_calibration=Trueで再帰防止"""
        engine = self._make_mock_engine()

        mock_result = MagicMock()
        mock_result.total_bets = 100
        mock_result.total_roi = 0.90
        mock_result.bet_history = []

        with patch("backtest.engine.BacktestEngine") as MockEngine:  # noqa: N806
            MockEngine.return_value.run.return_value = mock_result
            engine._generate_training_bet_history()

            inner_call = MockEngine.call_args
            # Phase 43.5: 同じbetting_targetを使用 (0 bets問題を回避)
            assert inner_call.kwargs["betting_target"] == "win"
            # strategy_params=None でデフォルトOddsBandFilter
            assert inner_call.kwargs["strategy_params"] is None
            # _skip_odds_band_calibration が設定されることを確認
            inner_instance = MockEngine.return_value
            assert inner_instance._skip_odds_band_calibration is True

    def test_uses_no_strategy_params_for_inner_engine(self):
        """Phase 43.5: 内部エンジンはstrategy_params=Noneで構築"""
        engine = self._make_mock_engine()

        mock_result = MagicMock()
        mock_result.total_bets = 200
        mock_result.total_roi = 0.92
        mock_result.bet_history = []

        with patch("backtest.engine.BacktestEngine") as MockEngine:  # noqa: N806
            MockEngine.return_value.run.return_value = mock_result
            engine._generate_training_bet_history()

            inner_call = MockEngine.call_args
            # strategy_params=None → デフォルト roi_threshold=1.0
            assert inner_call.kwargs["strategy_params"] is None

    def test_returns_none_on_failure(self):
        """_generate_training_bet_history()が失敗時にNoneを返す"""
        engine = self._make_mock_engine()

        with patch("backtest.engine.BacktestEngine") as MockEngine:  # noqa: N806
            MockEngine.side_effect = RuntimeError("test failure")
            result = engine._generate_training_bet_history()
            assert result is None

    def test_default_train_period_works(self):
        """self.models.train_periodのデフォルト値でも動作する"""
        engine = self._make_mock_engine(
            train_period=("2020-01-01", "2023-12-31")
        )

        mock_result = MagicMock()
        mock_result.total_bets = 300
        mock_result.total_roi = 0.88
        mock_result.bet_history = []

        with patch("backtest.engine.BacktestEngine") as MockEngine:  # noqa: N806
            MockEngine.return_value.run.return_value = mock_result
            engine._generate_training_bet_history()

            inner_run_call = MockEngine.return_value.run
            inner_run_call.assert_called_once_with("2020-01-01", "2023-12-31")


class TestAutoCalibrateE2E:
    """E2E: _calibrate_odds_band_filter() -> _generate_training_bet_history()
    -> calibrate() フロー検証

    実際のengineメソッド (_calibrate_odds_band_filter) を呼び出し、
    side effect を patch.object で検証する。
    """

    def _make_mock_engine(self, train_period=("2020-01-01", "2023-12-31")):
        from backtest.engine import BacktestEngine
        mock_models = MagicMock()
        mock_models.regime_detector = MagicMock()
        mock_models.train_period = train_period
        return BacktestEngine(
            models=mock_models,
            initial_bankroll=100_000,
            store=MagicMock(),
            betting_mode="kelly",
            betting_target="win",
            strategy_params={"roi_threshold": 1.0},
        )

    def test_auto_generates_and_calibrates(self):
        """E2E: training_bet_history=Noneの場合に
        _generate_training_bet_history()を呼び出し、
        その結果がOddsBandFilter.calibrate()に渡される"""
        engine = self._make_mock_engine()

        generated_history = [
            {"odds": 3.0, "result": 300, "stake": 100},
            {"odds": 8.0, "result": 0, "stake": 100},
        ]

        with patch.object(engine, "_generate_training_bet_history") as mock_gen, \
             patch.object(engine, "_odds_band_filter") as mock_filter:
            mock_gen.return_value = generated_history

            # 実際のengineメソッドを呼び出す
            engine._calibrate_odds_band_filter(training_bet_history=None)

            mock_gen.assert_called_once()
            mock_filter.calibrate.assert_called_once_with(generated_history)

    def test_uses_provided_history_without_generate(self):
        """E2E: training_bet_historyを受け取った場合、
        _generate_training_bet_history()を呼ばず、
        渡されたhistoryをcalibrate()に渡す"""
        engine = self._make_mock_engine()

        provided_history = [{"odds": 5.0, "result": 500, "stake": 100}]

        with patch.object(engine, "_generate_training_bet_history") as mock_gen, \
             patch.object(engine, "_odds_band_filter") as mock_filter:
            # 実際のengineメソッドを呼び出す
            engine._calibrate_odds_band_filter(training_bet_history=provided_history)

            mock_gen.assert_not_called()
            mock_filter.calibrate.assert_called_once_with(provided_history)

    def test_skips_calibrate_when_generate_returns_none(self):
        """E2E: _generate_training_bet_history()がNoneを返した場合、
        calibrate()が呼ばれない"""
        engine = self._make_mock_engine()

        with patch.object(engine, "_generate_training_bet_history") as mock_gen, \
             patch.object(engine, "_odds_band_filter") as mock_filter:
            mock_gen.return_value = None

            # 実際のengineメソッドを呼び出す
            engine._calibrate_odds_band_filter(training_bet_history=None)

            mock_gen.assert_called_once()
            mock_filter.calibrate.assert_not_called()

    def test_skip_flag_prevents_calibration(self):
        """Phase 43.5: _skip_odds_band_calibration=Trueの場合、
        _calibrate_odds_band_filter()は即座にNoneを返す"""
        engine = self._make_mock_engine()
        engine._skip_odds_band_calibration = True

        with patch.object(engine, "_generate_training_bet_history") as mock_gen, \
             patch.object(engine, "_odds_band_filter") as mock_filter:
            result = engine._calibrate_odds_band_filter(training_bet_history=None)

            mock_gen.assert_not_called()
            mock_filter.calibrate.assert_not_called()
            assert result is None
