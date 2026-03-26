"""BacktestValidationSuite のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock

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
                "finish_pos": [1, 2, 3, 4, 5],
                "win_odds_actual": [3.0, 5.0, 8.0, 15.0, 30.0],
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
                "finish_pos": [1, 2, 3, 4, 5],
                "win_odds_actual": [0.0, 5.0, 8.0, 15.0, 30.0],
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
