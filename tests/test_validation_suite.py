"""BacktestValidationSuite のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


class TestBacktestValidationSuite:
    """BacktestValidationSuite のテスト"""

    def test_suite_initialization(self) -> None:
        """スイートが初期化できる"""
        from backtest.validation_suite import BacktestValidationSuite

        suite = BacktestValidationSuite()
        assert suite is not None

    def test_run_all_returns_results(self) -> None:
        """run_all() が検証結果を返す"""
        from backtest.validation_suite import BacktestValidationSuite

        suite = BacktestValidationSuite()
        result = suite.run_all(
            models=None,
            feature_cols=[],
            stage2_feature_cols=[],
            test_df=pd.DataFrame(),
        )
        assert "passed" in result
        assert "tests" in result
        assert isinstance(result["tests"], list)

    def test_stage_b_no_zeros_with_good_data(self) -> None:
        """Stage B ゼロ検出: ゼロがないデータで PASS"""
        from backtest.validation_suite import BacktestValidationSuite

        suite = BacktestValidationSuite()

        df = pd.DataFrame(
            {
                "kakuteijyuni": [1, 2, 3, 4, 5],
                "odds": [3.0, 5.0, 8.0, 15.0, 30.0],
            }
        )
        result = suite.test_stage_b_no_zeros(df)
        assert result["passed"] is True

    def test_stage_b_no_zeros_with_zeros(self) -> None:
        """Stage B ゼロ検出: ゼロがあるデータで FAIL"""
        from backtest.validation_suite import BacktestValidationSuite

        suite = BacktestValidationSuite()

        df = pd.DataFrame(
            {
                "kakuteijyuni": [1, 2, 3, 4, 5],
                "odds": [0.0, 5.0, 8.0, 15.0, 30.0],
            }
        )
        result = suite.test_stage_b_no_zeros(df)
        assert result["passed"] is False

    def test_market_model_no_pred_in_stage2(self) -> None:
        """p_market_pred が Stage2 に入っていないことを確認"""
        from backtest.validation_suite import BacktestValidationSuite

        suite = BacktestValidationSuite()

        cols = ["signed_log_error_win", "abs_log_error_win", "ev_win"]
        result = suite.test_market_model_no_pred_in_stage2(cols)
        assert result["passed"] is True

        cols_bad = ["p_market_pred_win", "signed_log_error_win"]
        result = suite.test_market_model_no_pred_in_stage2(cols_bad)
        assert result["passed"] is False

    def test_market_model_uses_log_error(self) -> None:
        """log_error が Stage2 に含まれることを確認"""
        from backtest.validation_suite import BacktestValidationSuite

        suite = BacktestValidationSuite()

        cols = ["market_log_error_win", "signed_log_error_win", "abs_log_error_win"]
        result = suite.test_market_model_uses_log_error(cols)
        assert result["passed"] is True

        cols_bad = ["p_market_pred_win"]
        result = suite.test_market_model_uses_log_error(cols_bad)
        assert result["passed"] is False

    def test_ev_correction_pe_independent(self) -> None:
        """P補正とE補正が独立に動作していることを確認"""
        from backtest.validation_suite import BacktestValidationSuite

        suite = BacktestValidationSuite()

        df = pd.DataFrame(
            {
                "p_win_corrected": [0.2, 0.1, 0.3],
                "e_return_win_corrected": [5.0, 10.0, 3.0],
                "ev_win_corrected": [1.0, 1.0, 0.9],
            }
        )
        result = suite.test_ev_correction_pe_independent(df)
        assert result["passed"] is True

    def test_wide_score_variance_based(self) -> None:
        """ワイドスコアが EV / (E × √P) で計算されていることを確認"""
        from backtest.validation_suite import BacktestValidationSuite

        suite = BacktestValidationSuite()

        df = pd.DataFrame(
            {
                "ev_wide": [2.0, 3.0, 1.5],
                "e_return_given_hit": [5.0, 6.0, 4.0],
                "p_hit": [0.1, 0.2, 0.05],
                "wide_score_adj": [
                    2.0 / (5.0 * np.sqrt(0.1)),
                    3.0 / (6.0 * np.sqrt(0.2)),
                    1.5 / (4.0 * np.sqrt(0.05)),
                ],
            }
        )
        result = suite.test_wide_score_variance_based(df)
        assert result["passed"] is True

    def test_log_error_clipping(self) -> None:
        """log_error クリップの発散防止"""
        from backtest.validation_suite import BacktestValidationSuite

        suite = BacktestValidationSuite()

        result = suite.test_log_error_clipping()
        assert result["passed"] is True

    def test_submodel_sample_sufficiency(self) -> None:
        """サブモデルのサンプル数チェック"""
        from backtest.validation_suite import BacktestValidationSuite

        suite = BacktestValidationSuite()

        result = suite.test_submodel_sample_sufficiency("turf", 50000)
        assert result["passed"] is True

        result = suite.test_submodel_sample_sufficiency("dirt", 10000)
        assert result["passed"] is False

    def test_ev_correction_log_denominator_stable(self) -> None:
        """P/E分解補正が低確率帯で発散しないことを確認"""
        from backtest.validation_suite import BacktestValidationSuite

        suite = BacktestValidationSuite()

        df = pd.DataFrame(
            {
                "ev_win": [0.03, 0.10, 0.50],
                "ev_win_corrected": [0.04, 0.09, 0.45],
                "p_win_corrected": [0.05, 0.15, 0.40],
            }
        )
        result = suite.test_ev_correction_log_denominator_stable(df)
        assert result["passed"] is True

    def test_ev_correction_log_denominator_unstable(self) -> None:
        """P_corrected が範囲外の場合は FAIL"""
        from backtest.validation_suite import BacktestValidationSuite

        suite = BacktestValidationSuite()

        df = pd.DataFrame(
            {
                "ev_win": [0.03, 0.10],
                "ev_win_corrected": [0.04, 0.09],
                "p_win_corrected": [1.5, 0.15],  # 1.5 は範囲外
            }
        )
        result = suite.test_ev_correction_log_denominator_stable(df)
        assert result["passed"] is False

    def test_race_quality_uses_distribution_features(self) -> None:
        """分布特徴量が含まれていることを確認"""
        from backtest.validation_suite import BacktestValidationSuite

        suite = BacktestValidationSuite()

        cols = [
            "market_log_error_mean",
            "n_positive_errors",
            "top_k_error_sum",
            "positive_error_ratio",
            "hist_hit_rate_topk",
            "hist_roi_topk",
        ]
        result = suite.test_race_quality_uses_distribution_features(cols)
        assert result["passed"] is True

        cols_bad = ["market_log_error_mean"]
        result = suite.test_race_quality_uses_distribution_features(cols_bad)
        assert result["passed"] is False

    def test_race_quality_uses_profitability_proxy(self) -> None:
        """結果ベースproxyが含まれ、EV依存proxyが含まれないことを確認"""
        from backtest.validation_suite import BacktestValidationSuite

        suite = BacktestValidationSuite()

        cols = [
            "hist_hit_rate_topk",
            "hist_roi_topk",
            "hist_positive_return_ratio",
            "market_log_error_mean",
        ]
        result = suite.test_race_quality_uses_profitability_proxy(cols)
        assert result["passed"] is True

        cols_bad = ["hist_top3_ev_mean", "hist_hit_rate_topk"]
        result = suite.test_race_quality_uses_profitability_proxy(cols_bad)
        assert result["passed"] is False

    def test_ev_correction_reduces_error(self) -> None:
        """EV補正でMAE改善"""
        from backtest.validation_suite import BacktestValidationSuite

        suite = BacktestValidationSuite()
        df = pd.DataFrame(
            {
                "ev_win": [0.50, 0.80, 1.20],
                "ev_win_corrected": [0.55, 0.75, 1.15],
                "odds": [4.0, 6.0, 10.0],
                "kakuteijyuni": [1, 2, 1],
            }
        )
        result = suite.test_ev_correction_reduces_error(df)
        assert result["passed"] is True
        assert result["name"] == "ev_correction_reduces_error"

    def test_ev_correction_reduces_error_fail(self) -> None:
        """EV補正でMAE悪化の場合は FAIL"""
        from backtest.validation_suite import BacktestValidationSuite

        suite = BacktestValidationSuite()
        # All non-winners: actual_ev = 0 for all rows
        # ev_win is close to 0 (good), ev_win_corrected is far from 0 (bad)
        df = pd.DataFrame(
            {
                "ev_win": [0.10, 0.20, 0.05, 0.15],
                "ev_win_corrected": [5.00, 6.00, 4.00, 7.00],
                "odds": [10.0, 8.0, 20.0, 15.0],
                "kakuteijyuni": [2, 3, 4, 5],
            }
        )
        result = suite.test_ev_correction_reduces_error(df)
        assert result["passed"] is False

    def test_ev_correction_mid_range_improvement(self) -> None:
        """中穴ゾーン改善率テスト"""
        from backtest.validation_suite import BacktestValidationSuite

        suite = BacktestValidationSuite()
        rows = []
        for _ in range(150):
            rows.append(
                {
                    "p_win_pred": 0.10,
                    "ev_win": 1.0,
                    "ev_win_corrected": 0.70,
                    "odds": 5.0,
                    "kakuteijyuni": 2,
                }
            )
        df = pd.DataFrame(rows)
        result = suite.test_ev_correction_mid_range_improvement(df)
        assert result["passed"] is True

    def test_ev_correction_winner_weight(self) -> None:
        """1着馬P_corrected中央値テスト"""
        from backtest.validation_suite import BacktestValidationSuite

        suite = BacktestValidationSuite()
        rows = []
        for _ in range(60):
            rows.append({"kakuteijyuni": 1, "p_win_pred": 0.15, "p_win_corrected": 0.20})
        for _ in range(40):
            rows.append({"kakuteijyuni": 2, "p_win_pred": 0.10, "p_win_corrected": 0.08})
        df = pd.DataFrame(rows)
        result = suite.test_ev_correction_winner_weight(df)
        assert result["passed"] is True

    def test_race_quality_screener_independence(self) -> None:
        """品質スコア独立性テスト"""
        from backtest.validation_suite import BacktestValidationSuite

        suite = BacktestValidationSuite()
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "quality_score": np.random.normal(0.5, 0.15, 200),
                "edge_max_per_race": np.random.normal(1.1, 0.2, 200),
            }
        )
        result = suite.test_race_quality_screener_independence(df)
        assert result["passed"] is True

    def test_race_quality_no_temporal_leak(self) -> None:
        """時間リークテスト"""
        from backtest.validation_suite import BacktestValidationSuite

        suite = BacktestValidationSuite()
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        values = np.arange(1.0, 11.0)
        hist_means = [float("nan")] + [float(np.mean(values[:i])) for i in range(1, 10)]
        df = pd.DataFrame({"race_date": dates, "value": values, "hist_mean": hist_means})
        result = suite.test_race_quality_no_temporal_leak(df)
        assert result["passed"] is True

    def test_check_holdout_criteria(self) -> None:
        """§13.2 Hold-out 合格基準のチェック"""
        from backtest.validation_suite import BacktestValidationSuite

        suite = BacktestValidationSuite()

        mock_result = MagicMock()
        mock_result.total_roi = 1.08
        mock_result.max_drawdown = 0.10

        result = suite.check_holdout_criteria(mock_result)
        assert result["passed"] is True

        mock_result.total_roi = 0.95
        result = suite.check_holdout_criteria(mock_result)
        assert result["passed"] is False


class TestWalkForwardCV:
    """run_walk_forward_cv() のテスト (mock-based)"""

    @patch("backtest.parameter_freeze_protocol.ParameterFreezeProtocol")
    @patch("backtest.engine.BacktestEngine")
    @patch("pipelines.training_pipeline.TrainingPipelineV5")
    def test_run_walk_forward_cv_returns_all_windows(
        self,
        mock_pipeline_cls: MagicMock,
        mock_engine_cls: MagicMock,
        mock_freeze_cls: MagicMock,
    ) -> None:
        """run_walk_forward_cv() が3ウィンドウ分の結果と _overall を返す"""
        from backtest.validation_suite import BacktestValidationSuite

        # --- arrange ---
        suite = BacktestValidationSuite(store=None)

        # TrainingPipelineV5().run() → TrainedModelsV5 mock
        mock_models = MagicMock(name="TrainedModelsV5")
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = mock_models
        mock_pipeline_cls.return_value = mock_pipeline

        # BacktestEngine(models=..., db=...).run() → BacktestResult mock
        mock_bt_result = MagicMock(name="BacktestResult")
        mock_bt_result.total_roi = 1.05
        mock_bt_result.max_drawdown = 0.08
        mock_bt_result.total_bets = 500
        mock_engine = MagicMock()
        mock_engine.run.return_value = mock_bt_result
        mock_engine_cls.return_value = mock_engine

        # ParameterFreezeProtocol(models).freeze() / verify()
        mock_protocol = MagicMock()
        mock_protocol.freeze.return_value = None
        mock_protocol.verify.return_value = {"passed": True, "message": "OK"}
        mock_freeze_cls.return_value = mock_protocol

        # --- act ---
        results = suite.run_walk_forward_cv()

        # --- assert ---
        assert "Window 1" in results
        assert "Window 2" in results
        assert "Window 3" in results
        assert "_overall" in results

        # 各ウィンドウの構造を確認
        for name in ["Window 1", "Window 2", "Window 3"]:
            w = results[name]
            assert "roi" in w
            assert "max_dd" in w
            assert "total_bets" in w
            assert "git_hash" in w
            assert "rule7_passed" in w
            assert "train_period" in w
            assert "test_period" in w
            assert w["roi"] == 1.05
            assert w["rule7_passed"] is True

        # _overall の構造を確認
        overall = results["_overall"]
        assert "mean_roi" in overall
        assert "std_roi" in overall
        assert "git_hash" in overall
        assert "n_windows" in overall
        assert overall["n_windows"] == 3

        # TrainingPipeline.run が3回呼ばれたこと
        assert mock_pipeline.run.call_count == 3
        # BacktestEngine.run が3回呼ばれたこと
        assert mock_engine.run.call_count == 3
        # ParameterFreezeProtocol が3回インスタンス化されたこと
        assert mock_freeze_cls.call_count == 3

    @patch("backtest.parameter_freeze_protocol.ParameterFreezeProtocol")
    @patch("backtest.engine.BacktestEngine")
    @patch("pipelines.training_pipeline.TrainingPipelineV5")
    def test_run_walk_forward_cv_per_window_roi(
        self,
        mock_pipeline_cls: MagicMock,
        mock_engine_cls: MagicMock,
        mock_freeze_cls: MagicMock,
    ) -> None:
        """各ウィンドウで異なる ROI を返す場合、_overall の集計が正しい"""
        from backtest.validation_suite import BacktestValidationSuite

        suite = BacktestValidationSuite(store=None)

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = MagicMock(name="TrainedModelsV5")
        mock_pipeline_cls.return_value = mock_pipeline

        # 各ウィンドウで異なる ROI を返す
        rois = [1.02, 1.08, 0.95]
        mock_engine_instances = []
        for roi in rois:
            bt = MagicMock()
            bt.total_roi = roi
            bt.max_drawdown = 0.10
            bt.total_bets = 300
            eng = MagicMock()
            eng.run.return_value = bt
            mock_engine_instances.append(eng)
        mock_engine_cls.side_effect = mock_engine_instances

        mock_protocol = MagicMock()
        mock_protocol.freeze.return_value = None
        mock_protocol.verify.return_value = {"passed": True, "message": "OK"}
        mock_freeze_cls.return_value = mock_protocol

        results = suite.run_walk_forward_cv()

        # 各ウィンドウの ROI を確認
        assert results["Window 1"]["roi"] == 1.02
        assert results["Window 2"]["roi"] == 1.08
        assert results["Window 3"]["roi"] == 0.95

        # _overall の mean_roi を確認
        expected_mean = np.mean(rois)
        assert abs(results["_overall"]["mean_roi"] - expected_mean) < 1e-10

    @patch("backtest.parameter_freeze_protocol.ParameterFreezeProtocol")
    @patch("backtest.engine.BacktestEngine")
    @patch("pipelines.training_pipeline.TrainingPipelineV5")
    def test_run_walk_forward_cv_rule7_violation(
        self,
        mock_pipeline_cls: MagicMock,
        mock_engine_cls: MagicMock,
        mock_freeze_cls: MagicMock,
    ) -> None:
        """Rule 7 違反があった場合、rule7_passed が False になる"""
        from backtest.validation_suite import BacktestValidationSuite

        suite = BacktestValidationSuite(store=None)

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = MagicMock(name="TrainedModelsV5")
        mock_pipeline_cls.return_value = mock_pipeline

        mock_bt = MagicMock()
        mock_bt.total_roi = 1.05
        mock_bt.max_drawdown = 0.08
        mock_bt.total_bets = 500
        mock_engine = MagicMock()
        mock_engine.run.return_value = mock_bt
        mock_engine_cls.return_value = mock_engine

        # 2番目のウィンドウで Rule 7 違反
        mock_protocol_ok = MagicMock()
        mock_protocol_ok.freeze.return_value = None
        mock_protocol_ok.verify.return_value = {"passed": True, "message": "OK"}

        mock_protocol_fail = MagicMock()
        mock_protocol_fail.freeze.return_value = None
        mock_protocol_fail.verify.return_value = {
            "passed": False,
            "message": "Parameters changed",
        }

        mock_freeze_cls.side_effect = [mock_protocol_ok, mock_protocol_fail, mock_protocol_ok]

        results = suite.run_walk_forward_cv()

        assert results["Window 1"]["rule7_passed"] is True
        assert results["Window 2"]["rule7_passed"] is False
        assert results["Window 3"]["rule7_passed"] is True
