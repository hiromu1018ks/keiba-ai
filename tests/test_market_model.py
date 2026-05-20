"""src/models/market_model.py のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from models.market_model import MarketModel


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """4頭立てのテストデータ"""
    return pd.DataFrame(
        {
            "race_id": ["R1"] * 4,
            "surface": ["turf"] * 4,
            "distance_bin": ["mile"] * 4,
            "track_condition_code": [1] * 4,
            "grade_code": ["_"] * 4,
            "field_size": [4] * 4,
            "weight_diff_from_mean": [0.0, -2.0, 1.0, 5.0],
            "difficulty_score": [0.5, 0.5, 0.5, 0.5],
            "p_market_win_adj": [0.40, 0.30, 0.20, 0.10],
        }
    )


@pytest.fixture
def trained_market_model(sample_df: pd.DataFrame) -> MarketModel:
    """学習済みMarketModel (mock)"""
    model = MarketModel()
    mock_lgb = MagicMock()
    # 予測値: 市場確率に近いが少しズレた値
    mock_lgb.predict.return_value = np.array([0.35, 0.28, 0.22, 0.15])
    mock_lgb.best_iteration = 150
    model.model = mock_lgb
    return model


class TestMarketModelPredict:
    def test_log_error_computation(
        self, trained_market_model: MarketModel, sample_df: pd.DataFrame
    ) -> None:
        """log(p_market / p_pred) が正しく計算される"""
        result = trained_market_model.predict_and_calc_error(sample_df)
        # log(0.40 / 0.35) ≈ 0.1335
        expected = np.log(0.40 / 0.35)
        assert abs(result["market_log_error_win"].iloc[0] - expected) < 1e-10

    def test_signed_and_abs_log_error(
        self, trained_market_model: MarketModel, sample_df: pd.DataFrame
    ) -> None:
        """signed_log_error_win と abs_log_error_win が正しく分離される"""
        result = trained_market_model.predict_and_calc_error(sample_df)
        for i in range(len(result)):
            signed = result["signed_log_error_win"].iloc[i]
            log_err = result["market_log_error_win"].iloc[i]
            assert signed == log_err
            assert result["abs_log_error_win"].iloc[i] == abs(log_err)

    def test_p_market_pred_dropped(
        self, trained_market_model: MarketModel, sample_df: pd.DataFrame
    ) -> None:
        """p_market_pred_win は出力から削除される (Rule 11)"""
        result = trained_market_model.predict_and_calc_error(sample_df)
        assert "_p_market_pred_win" not in result.columns
        assert "p_market_pred_win" not in result.columns

    def test_market_error_rank_in_race(
        self, trained_market_model: MarketModel, sample_df: pd.DataFrame
    ) -> None:
        """レース内相対ランクが正しく計算される"""
        result = trained_market_model.predict_and_calc_error(sample_df)
        ranks = result["market_error_rank_in_race"].values
        assert sorted(ranks) == [1, 2, 3, 4]

    def test_raw_error_preserved(
        self, trained_market_model: MarketModel, sample_df: pd.DataFrame
    ) -> None:
        """生の差分 market_pred_error_win も保持される"""
        result = trained_market_model.predict_and_calc_error(sample_df)
        expected = 0.40 - 0.35
        assert abs(result["market_pred_error_win"].iloc[0] - expected) < 1e-10

    def test_predict_uses_best_iteration(
        self, trained_market_model: MarketModel, sample_df: pd.DataFrame
    ) -> None:
        """predict が best_iteration を使用する"""
        trained_market_model.predict_and_calc_error(sample_df)
        trained_market_model.model.predict.assert_called_once()
        call_kwargs = trained_market_model.model.predict.call_args
        assert call_kwargs[1]["num_iteration"] == 150

    def test_clipping_prevents_divergence(self) -> None:
        """極端な p_pred がクリップされる (Rule 13)"""
        model = MarketModel()
        mock_lgb = MagicMock()
        mock_lgb.predict.return_value = np.array([0.001, 0.999, 0.5, 0.5])
        mock_lgb.best_iteration = 100
        model.model = mock_lgb

        df = pd.DataFrame(
            {
                "race_id": ["R1"] * 4,
                "surface": ["turf"] * 4,
                "distance_bin": ["mile"] * 4,
                "track_condition_code": [1] * 4,
                "grade_code": ["_"] * 4,
                "field_size": [4] * 4,
                "weight_diff_from_mean": [0.0, 0.0, 0.0, 0.0],
                "difficulty_score": [0.5, 0.5, 0.5, 0.5],
                "p_market_win_adj": [0.40, 0.10, 0.25, 0.25],
            }
        )
        result = model.predict_and_calc_error(df)
        # クリップ後のlog_errorは発散しない
        assert result["market_log_error_win"].notna().all()
        assert np.isfinite(result["market_log_error_win"].values).all()


class TestMarketModelTrain:
    @patch("models.market_model.lgb.train")
    @patch("models.market_model.lgb.Dataset")
    def test_train_uses_early_stopping(
        self, mock_dataset_cls: MagicMock, mock_lgb_train: MagicMock, sample_df: pd.DataFrame
    ) -> None:
        """train() が valid_sets と callbacks (early_stopping) を使用する"""
        mock_booster = MagicMock()
        mock_booster.best_iteration = 120
        mock_lgb_train.return_value = mock_booster

        model = MarketModel()
        model.train(sample_df)

        # lgb.train が呼ばれた
        mock_lgb_train.assert_called_once()
        call_kwargs = mock_lgb_train.call_args[1]

        # valid_sets が渡されている (= 2要素: train, valid)
        assert "valid_sets" in call_kwargs
        assert call_kwargs["valid_sets"] is not None

        # callbacks に early_stopping が含まれる
        assert "callbacks" in call_kwargs
        assert len(call_kwargs["callbacks"]) == 1

    @patch("models.market_model.lgb.train")
    @patch("models.market_model.lgb.Dataset")
    def test_train_uses_80_20_split(
        self, mock_dataset_cls: MagicMock, mock_lgb_train: MagicMock
    ) -> None:
        """train() が 80/20 split で lgb.Dataset を2回作成する"""
        mock_booster = MagicMock()
        mock_booster.best_iteration = 100
        mock_lgb_train.return_value = mock_booster

        # テストデータを少し大きくしてsplitを確認
        df = pd.DataFrame(
            {
                "race_id": ["R1"] * 10,
                "surface": ["turf"] * 10,
                "distance_bin": ["mile"] * 10,
                "track_condition_code": [1] * 10,
                "grade_code": ["_"] * 10,
                "field_size": [10] * 10,
                "weight_diff_from_mean": np.random.randn(10),
                "difficulty_score": [0.5] * 10,
                "p_market_win_adj": np.random.dirichlet(np.ones(10)),
            }
        )

        model = MarketModel()
        model.train(df)

        # lgb.Dataset が 2回呼ばれる (train + valid)
        assert mock_dataset_cls.call_count == 2

    @patch("models.market_model.lgb.train")
    @patch("models.market_model.lgb.Dataset")
    def test_train_uses_time_based_split_not_random(
        self, mock_dataset_cls: MagicMock, mock_lgb_train: MagicMock
    ) -> None:
        """train() が時間ベースの分割を使用し、ランダム置換を使わないことを確認"""
        mock_booster = MagicMock()
        mock_booster.best_iteration = 50
        mock_lgb_train.return_value = mock_booster

        n = 100
        # p_market_win_adj に一意の値を使い、train/valid に渡る feature の順序を検証
        df = pd.DataFrame(
            {
                "race_id": ["R1"] * 50 + ["R2"] * 50,
                "surface": ["turf"] * n,
                "distance_bin": ["sprint"] * n,
                "track_condition_code": [1] * n,
                "grade_code": [0] * n,
                "field_size": [10] * n,
                "weight_diff_from_mean": np.arange(n, dtype=float),  # 一意な値
                "difficulty_score": [0.5] * n,
                "p_market_win_adj": np.linspace(0.1, 0.5, n),
            }
        )

        model = MarketModel()
        model.train(df)

        # lgb.train が呼ばれたことを確認
        assert mock_lgb_train.called

        # train_idx が [0, 79]、valid_idx が [80, 99] であることを確認
        # (時間ベースの最初80% = 学習、最後20% = 検証)
        call_args = mock_dataset_cls.call_args_list
        train_features = call_args[0][0][0]  # first lgb.Dataset call = train data
        assert len(train_features) == 80  # 最初の80%
        valid_features = call_args[1][0][0]  # second lgb.Dataset call = valid data
        assert len(valid_features) == 20  # 最後の20%

        # weight_diff_from_mean が sequential に渡されることを確認
        # ランダム分割なら train に行0-79以外のインデックスが含まれる
        train_weight = call_args[0][0][0]["weight_diff_from_mean"].values
        np.testing.assert_array_equal(train_weight, np.arange(80, dtype=float))
        valid_weight = call_args[1][0][0]["weight_diff_from_mean"].values
        np.testing.assert_array_equal(valid_weight, np.arange(80, 100, dtype=float))


class TestMarketModelStage2Features:
    def test_no_p_market_pred_in_features(self) -> None:
        """get_stage2_features() に p_market_pred は含まれない (Rule 11)"""
        model = MarketModel()
        features = model.get_stage2_features()
        assert "p_market_pred_win" not in features

    def test_returns_expected_features(self) -> None:
        model = MarketModel()
        features = model.get_stage2_features()
        assert "signed_log_error_win" in features
        assert "abs_log_error_win" in features
        assert "market_error_rank_in_race" in features


# Phase36 fundamental features that must NOT be in MarketModel (RTG-01)
_PHASE36_FEATURES = [
    "form_trend_race_rank",
    "blood_total_wr_race_rank",
    "blood_surface_wr_race_rank",
    "weighted_recent_form_finish",
    "weighted_recent_form_time",
    "grade_x_form_trend",
    "distance_x_closing_index",
    "grade_x_blood_prize_log",
    "closing_speed_ratio_avg",
    "closing_speed_ratio_zscore",
    "closing_speed_ratio_trend",
    "harontime_last3f_avg",
    "harontime_last3f_zscore",
    "harontime_last3f_trend",
    "haron_race_gap_avg",
    "haron_race_gap_zscore",
    "haron_race_gap_trend",
    "pace_adj_finish_avg",
    "pace_ratio_avg",
    "pace_ratio_zscore",
    "pace_ratio_trend",
    "pace_early_avg",
    "pace_mid_avg",
    "pace_late_avg",
    "closing_speed_ratio_avg_race_rank",
    "harontime_last3f_avg_race_rank",
]


class TestMarketModelFeatureRouting:
    """RTG-01: MarketModel must NOT contain Phase36 fundamental features."""

    def test_no_phase36_features_in_feature_cols(self) -> None:
        """MarketModel.FEATURE_COLS に Phase36 特徴量が含まれない (RTG-01)"""
        for feat in _PHASE36_FEATURES:
            assert feat not in MarketModel.FEATURE_COLS, (
                f"Phase36 feature '{feat}' found in MarketModel.FEATURE_COLS"
            )

    def test_market_features_still_present(self) -> None:
        """MarketModel.FEATURE_COLS に market-only 特徴量が残っている"""
        must_have = [
            "implied_prob_hhi",
            "odds_skewness",
            "rl_log_odds_entropy",
            "rl_odds_dispersion",
        ]
        for feat in must_have:
            assert feat in MarketModel.FEATURE_COLS, (
                f"Required market feature '{feat}' missing from MarketModel.FEATURE_COLS"
            )
