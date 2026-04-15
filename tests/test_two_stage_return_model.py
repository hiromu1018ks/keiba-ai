"""src/models/two_stage_return_model.py のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from models.two_stage_return_model import (
    PlaceTwoStageModel,
    WinTwoStageModel,
    _train_valid_split,
)


@pytest.fixture
def feature_df() -> pd.DataFrame:
    """WinTwoStageModel FEATURE_COLS を満たす8頭テストデータ"""
    return pd.DataFrame(
        {
            "race_id": ["R1"] * 8,
            "p_ability_win": [0.30, 0.25, 0.20, 0.10, 0.08, 0.04, 0.02, 0.01],
            "signed_log_error_win": [0.1, -0.1, 0.2, -0.3, 0.0, 0.5, -0.2, 0.3],
            "abs_log_error_win": [0.1, 0.1, 0.2, 0.3, 0.0, 0.5, 0.2, 0.3],
            "odds_drop_rate_60_10": [0.0, -0.1, 0.1, -0.2, 0.0, 0.1, -0.1, 0.0],
            "odds_drop_rate_30_10": [0.0, -0.05, 0.05, -0.1, 0.0, 0.05, -0.05, 0.0],
            "odds_velocity": [0.0, -0.02, 0.02, -0.03, 0.0, 0.01, -0.01, 0.0],
            "odds_volatility": [0.01, 0.02, 0.01, 0.03, 0.01, 0.02, 0.01, 0.01],
            "popularity_change_30_10": [0, -1, 1, -2, 0, 1, -1, 0],
            "market_entropy": [2.5] * 8,
            "popularity_rank": [1, 2, 3, 4, 5, 6, 7, 8],
            "overround": [0.22] * 8,
            "surface": ["turf"] * 8,
            "distance_bin": ["mile"] * 8,
            "track_condition_code": [1] * 8,
            "grade_code": ["_"] * 8,
            "field_size": [8] * 8,
            # FLB slope (市場歪みの非対称性)
            "odds_skewness": [1.5] * 8,
            # Place固有特徴量
            "fukuoddslow": [1.3, 1.5, 1.8, 2.1, 2.5, 3.0, 3.5, 4.0],
            "tanodds": [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0],
            "p_ability_place": [0.55, 0.48, 0.42, 0.38, 0.32, 0.25, 0.20, 0.15],
            "finish_pos": [1, 2, 3, 4, 5, 6, 7, 8],
            "win_odds_actual": [3.5, 5.0, 8.0, 15.0, 25.0, 40.0, 80.0, 150.0],
            "place_odds_actual": [1.3, 1.6, 2.1, 3.5, 5.0, 8.0, 15.0, 30.0],
        }
    )


def _make_mock_booster(pred_values: list[float]) -> MagicMock:
    """best_iteration 属性付きの mock Booster を生成"""
    mock = MagicMock()
    mock.predict.return_value = np.array(pred_values)
    mock.best_iteration = 100
    return mock


@pytest.fixture
def trained_win_model(feature_df: pd.DataFrame) -> WinTwoStageModel:
    """学習済みWinTwoStageModel (mock)"""
    model = WinTwoStageModel()
    model.hit_model = _make_mock_booster([0.28, 0.22, 0.18, 0.12, 0.08, 0.06, 0.04, 0.02])
    model.return_model = _make_mock_booster([4.0, 6.0, 9.0, 16.0, 28.0, 45.0, 90.0, 160.0])
    return model


class TestWinTwoStageModel:
    def test_ev_equals_p_times_e(
        self,
        trained_win_model: WinTwoStageModel,
        feature_df: pd.DataFrame,
    ) -> None:
        """EV = P(win) × E(win_odds | win)"""
        result = trained_win_model.predict_ev(feature_df)
        expected_ev = result["p_win_pred"] * result["e_return_win_pred"]
        assert np.allclose(result["ev_win"].values, expected_ev.values)

    def test_output_columns(
        self,
        trained_win_model: WinTwoStageModel,
        feature_df: pd.DataFrame,
    ) -> None:
        result = trained_win_model.predict_ev(feature_df)
        assert "p_win_pred" in result.columns
        assert "e_return_win_pred" in result.columns
        assert "ev_win" in result.columns

    def test_predict_uses_best_iteration(
        self,
        trained_win_model: WinTwoStageModel,
        feature_df: pd.DataFrame,
    ) -> None:
        """predict が num_iteration=best_iteration を使用すること"""
        trained_win_model.predict_ev(feature_df)

        trained_win_model.hit_model.predict.assert_called_once()
        call_kwargs = trained_win_model.hit_model.predict.call_args
        assert call_kwargs[1]["num_iteration"] == 100

        trained_win_model.return_model.predict.assert_called_once()
        call_kwargs = trained_win_model.return_model.predict.call_args
        assert call_kwargs[1]["num_iteration"] == 100

    @patch("models.two_stage_return_model.lgb")
    def test_train_hit_model_uses_early_stopping(
        self, mock_lgb: MagicMock, feature_df: pd.DataFrame
    ) -> None:
        """train_hit_model が valid_sets と callbacks を渡すこと"""
        mock_booster = MagicMock()
        mock_booster.best_iteration = 50
        mock_lgb.train.return_value = mock_booster
        mock_lgb.Dataset.return_value = MagicMock()
        mock_lgb.early_stopping.return_value = lambda: None

        df = feature_df.copy()
        df["kakuteijyuni"] = [1, 2, 3, 4, 5, 6, 7, 8]

        model = WinTwoStageModel()
        model.train_hit_model(df)

        call_args = mock_lgb.train.call_args
        assert "valid_sets" in call_args[1]
        assert "callbacks" in call_args[1]
        mock_lgb.early_stopping.assert_called_once_with(stopping_rounds=100, verbose=False)

    @patch("models.two_stage_return_model.lgb")
    def test_train_return_model_early_stopping_guard(self, mock_lgb: MagicMock) -> None:
        """return_model: サンプル < 10 は early stopping なし"""
        mock_booster = MagicMock()
        mock_booster.best_iteration = 50
        mock_lgb.train.return_value = mock_booster
        mock_lgb.Dataset.return_value = MagicMock()

        # 5件のみの的中データ (min_hit_samples を満たさない可能性があるので注意)
        df = pd.DataFrame(
            {
                "kakuteijyuni": [1] * 5,
                "odds": [3.0, 5.0, 8.0, 15.0, 25.0],
                "confirmed_odds": [3.0, 5.0, 8.0, 15.0, 25.0],
                "p_ability_win": [0.3] * 5,
                "signed_log_error_win": [0.1] * 5,
                "abs_log_error_win": [0.1] * 5,
                "odds_drop_rate_60_10": [0.0] * 5,
                "odds_drop_rate_30_10": [0.0] * 5,
                "odds_velocity": [0.0] * 5,
                "odds_volatility": [0.01] * 5,
                "popularity_change_30_10": [0] * 5,
                "market_entropy": [2.5] * 5,
                "popularity_rank": [1] * 5,
                "overround": [0.22] * 5,
                "surface": ["turf"] * 5,
                "distance_bin": ["mile"] * 5,
                "track_condition_code": [1] * 5,
                "grade_code": ["_"] * 5,
                "field_size": [5] * 5,
                "odds_skewness": [1.5] * 5,
            }
        )

        # min_hit_samples を下回る場合は ValueError が発生するので、
        # cfg を調整して回避
        from domain.models import TwoStageConfig

        cfg = TwoStageConfig(min_hit_samples=3)
        model = WinTwoStageModel(cfg=cfg)
        model.train_return_model(df)

        call_args = mock_lgb.train.call_args
        # サンプル数 < 10 のため valid_sets が無いことを確認
        assert "valid_sets" not in call_args[1]

    def test_stage_b_no_zeros_in_label(self, feature_df: pd.DataFrame) -> None:
        """Stage B の学習データにゼロが含まれないことを確認するテスト"""
        hit_df = feature_df[feature_df["finish_pos"] == 1]
        assert (hit_df["win_odds_actual"] > 0).all()

    def test_feature_cols_no_p_market_pred(self) -> None:
        """p_market_pred が特徴量に含まれない (Rule 11)"""
        assert "p_market_pred_win" not in WinTwoStageModel.FEATURE_COLS


class TestPlaceTwoStageModel:
    @pytest.fixture
    def trained_place_model(self, feature_df: pd.DataFrame) -> PlaceTwoStageModel:
        model = PlaceTwoStageModel()
        model.hit_model = _make_mock_booster([0.40, 0.35, 0.30, 0.15, 0.10, 0.05, 0.03, 0.01])
        model.return_model = _make_mock_booster([1.4, 1.7, 2.2, 3.8, 5.5, 9.0, 16.0, 32.0])
        return model

    def test_ev_equals_p_times_e(
        self,
        trained_place_model: PlaceTwoStageModel,
        feature_df: pd.DataFrame,
    ) -> None:
        result = trained_place_model.predict_ev(feature_df)
        expected_ev = result["p_place_pred"] * result["e_return_place_pred"]
        assert np.allclose(result["ev_place"].values, expected_ev.values)

    def test_output_columns(
        self,
        trained_place_model: PlaceTwoStageModel,
        feature_df: pd.DataFrame,
    ) -> None:
        result = trained_place_model.predict_ev(feature_df)
        assert "p_place_pred" in result.columns
        assert "e_return_place_pred" in result.columns
        assert "ev_place" in result.columns

    def test_predict_uses_best_iteration(
        self,
        trained_place_model: PlaceTwoStageModel,
        feature_df: pd.DataFrame,
    ) -> None:
        """predict が num_iteration=best_iteration を使用すること"""
        trained_place_model.predict_ev(feature_df)

        trained_place_model.hit_model.predict.assert_called_once()
        call_kwargs = trained_place_model.hit_model.predict.call_args
        assert call_kwargs[1]["num_iteration"] == 100

        trained_place_model.return_model.predict.assert_called_once()
        call_kwargs = trained_place_model.return_model.predict.call_args
        assert call_kwargs[1]["num_iteration"] == 100

    @patch("models.two_stage_return_model.lgb")
    def test_train_hit_model_uses_early_stopping(
        self, mock_lgb: MagicMock, feature_df: pd.DataFrame
    ) -> None:
        """train_hit_model が valid_sets と callbacks を渡すこと"""
        mock_booster = MagicMock()
        mock_booster.best_iteration = 50
        mock_lgb.train.return_value = mock_booster
        mock_lgb.Dataset.return_value = MagicMock()
        mock_lgb.early_stopping.return_value = lambda: None

        df = feature_df.copy()
        df["kakuteijyuni"] = [1, 2, 3, 4, 5, 6, 7, 8]

        model = PlaceTwoStageModel()
        model.train_hit_model(df)

        call_args = mock_lgb.train.call_args
        assert "valid_sets" in call_args[1]
        assert "callbacks" in call_args[1]
        mock_lgb.early_stopping.assert_called_once_with(stopping_rounds=100, verbose=False)

    def test_hit_and_return_features_separated(self) -> None:
        """Hit model と Return model で特徴量が分離されていること"""
        # Return model のみがオッズ特徴量を持つ
        assert "fukuoddslow" in PlaceTwoStageModel.RETURN_FEATURE_COLS
        assert "tanodds" in PlaceTwoStageModel.RETURN_FEATURE_COLS
        # Hit model はオッズ特徴量を持たない
        assert "fukuoddslow" not in PlaceTwoStageModel.HIT_FEATURE_COLS
        assert "tanodds" not in PlaceTwoStageModel.HIT_FEATURE_COLS
        # p_ability_place は両方に含まれる
        assert "p_ability_place" in PlaceTwoStageModel.HIT_FEATURE_COLS
        assert "p_ability_place" in PlaceTwoStageModel.RETURN_FEATURE_COLS

    def test_place_return_feature_cols_include_place_specific(self) -> None:
        """Return model should have place-specific features beyond win features"""
        assert "fukuoddslow" in PlaceTwoStageModel.RETURN_FEATURE_COLS
        assert "p_ability_place" in PlaceTwoStageModel.RETURN_FEATURE_COLS
        assert "tanodds" in PlaceTwoStageModel.RETURN_FEATURE_COLS
        # Win特徴量も全て含む
        for col in WinTwoStageModel.FEATURE_COLS:
            assert col in PlaceTwoStageModel.RETURN_FEATURE_COLS
        # Place固有特徴量が追加されている
        assert len(PlaceTwoStageModel.RETURN_FEATURE_COLS) > len(WinTwoStageModel.FEATURE_COLS)

    @patch("models.two_stage_return_model.lgb")
    def test_train_hit_model_stores_val_predictions(
        self, mock_lgb: MagicMock, feature_df: pd.DataFrame
    ) -> None:
        """train_hit_model がバリデーション予測を保存すること"""
        mock_booster = MagicMock()
        mock_booster.best_iteration = 50
        mock_booster.predict.return_value = np.array([0.3, 0.5])
        mock_lgb.train.return_value = mock_booster
        mock_lgb.Dataset.return_value = MagicMock()
        mock_lgb.early_stopping.return_value = lambda: None

        df = feature_df.copy()
        df["kakuteijyuni"] = [1, 2, 3, 4, 5, 6, 7, 8]

        model = PlaceTwoStageModel()
        model.train_hit_model(df)

        assert model._val_predictions is not None
        assert model._val_labels is not None
        # 8 rows * 0.8 = 6.4 → split=6, val=2 rows
        assert len(model._val_predictions) == 2
        assert len(model._val_labels) == 2
        # Val labels: kakuteijyuni=7,8 → both > 3 → [0, 0]
        np.testing.assert_array_equal(model._val_labels, [0, 0])

    def test_fit_calibrator_creates_isotonic(self) -> None:
        """fit_calibrator がサンプル >= 1000 の場合 IsotonicRegression を作成すること"""
        from sklearn.isotonic import IsotonicRegression

        model = PlaceTwoStageModel()
        model._val_predictions = np.random.rand(1500)
        model._val_labels = (model._val_predictions > 0.5).astype(int)

        model.fit_calibrator()

        assert model._place_calibrator is not None
        assert isinstance(model._place_calibrator, IsotonicRegression)

    def test_fit_calibrator_skips_below_min_samples(self) -> None:
        """fit_calibrator がサンプル < 1000 の場合校正をスキップすること"""
        model = PlaceTwoStageModel()
        model._val_predictions = np.random.rand(500)
        model._val_labels = (model._val_predictions > 0.5).astype(int)

        model.fit_calibrator()

        assert model._place_calibrator is None

    def test_predict_ev_applies_isotonic_calibration(self, feature_df: pd.DataFrame) -> None:
        """predict_ev が _place_calibrator を適用して p_place_pred を補正すること"""
        model = PlaceTwoStageModel()
        model.hit_model = _make_mock_booster([0.40, 0.35, 0.30, 0.15, 0.10, 0.05, 0.03, 0.01])
        model.return_model = _make_mock_booster([1.4, 1.7, 2.2, 3.8, 5.5, 9.0, 16.0, 32.0])

        # Fit a fake calibrator that maps p → p * 0.5
        from sklearn.isotonic import IsotonicRegression

        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(np.array([0.01, 0.5, 0.99]), np.array([0.005, 0.25, 0.495]))
        model._place_calibrator = cal

        result = model.predict_ev(feature_df)

        # Raw predictions from mock: [0.40, 0.35, 0.30, 0.15, 0.10, 0.05, 0.03, 0.01]
        # After calibration (roughly p * 0.5) → race-sum normalization → clip
        # The exact values change due to normalization, but sum should be ~3.0
        raw_preds = np.array([0.40, 0.35, 0.30, 0.15, 0.10, 0.05, 0.03, 0.01])
        calibrated = cal.transform(raw_preds)
        normalized = calibrated * (3.0 / calibrated.sum())
        normalized = np.clip(normalized, 0.01, 0.99)
        np.testing.assert_allclose(result["p_place_pred"].values, normalized, rtol=1e-6)

    def test_predict_ev_race_sum_normalization(self, feature_df: pd.DataFrame) -> None:
        """predict_ev がレース内で sum(p_place_pred) ≈ 3.0 に正規化すること"""
        model = PlaceTwoStageModel()
        # Raw probabilities sum > 3.0 (typical overestimation pattern)
        model.hit_model = _make_mock_booster([0.70, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30])
        model.return_model = _make_mock_booster([1.4, 1.7, 2.2, 3.8, 5.5, 9.0, 16.0, 32.0])
        model._place_calibrator = None  # Skip calibration for this test

        result = model.predict_ev(feature_df)

        # sum(p_place_pred) should be ~ 3.0 per race
        race_sum = result.groupby("race_id")["p_place_pred"].sum()
        np.testing.assert_allclose(race_sum.values, 3.0, rtol=1e-6)

    def test_predict_ev_consistency_constraint(self, feature_df: pd.DataFrame) -> None:
        """p_place_pred >= p_ability_win の整合性制約が機能すること"""
        model = PlaceTwoStageModel()
        model.hit_model = _make_mock_booster([0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01])
        model.return_model = _make_mock_booster([1.4, 1.7, 2.2, 3.8, 5.5, 9.0, 16.0, 32.0])
        model._place_calibrator = None

        df = feature_df.copy()
        # Set p_ability_win high for horse 0 — should enforce floor
        df["p_ability_win"] = [0.50, 0.25, 0.20, 0.10, 0.08, 0.04, 0.02, 0.01]

        result = model.predict_ev(df)

        # After normalization, p_place_pred should be >= p_ability_win for all horses
        assert (result["p_place_pred"] >= result["p_ability_win"] - 1e-10).all()
        # Race sum should still be ~ 3.0
        race_sum = result.groupby("race_id")["p_place_pred"].sum()
        np.testing.assert_allclose(race_sum.values, 3.0, rtol=1e-6)


class TestTrainValidSplit:
    def test_split_ratio(self) -> None:
        """80/20 分割が正しいサイズを返す"""
        features = pd.DataFrame({"a": range(100), "b": range(100)})
        label = pd.Series(range(100))

        with patch("models.two_stage_return_model.lgb.Dataset") as mock_ds:
            mock_ds.return_value = MagicMock()
            _train_valid_split(features, label)

        # 呼び出し時に渡された DataFrame のサイズを確認
        first_call = mock_ds.call_args_list[0]
        second_call = mock_ds.call_args_list[1]

        train_features = first_call[0][0]
        valid_features = second_call[0][0]

        assert len(train_features) == 80
        assert len(valid_features) == 20

    def test_no_overlap(self) -> None:
        """train と valid に重複がないこと"""
        features = pd.DataFrame({"a": range(50), "b": range(50)})
        label = pd.Series(range(50))

        with patch("models.two_stage_return_model.lgb.Dataset") as mock_ds:
            mock_ds.return_value = MagicMock()
            _train_valid_split(features, label)

        train_idx = mock_ds.call_args_list[0][0][0].index
        valid_idx = mock_ds.call_args_list[1][0][0].index

        assert len(set(train_idx) & set(valid_idx)) == 0

    def test_train_valid_split_is_chronological(self) -> None:
        """_train_valid_split が時系列順で前80%/後20%に分割することを確認"""
        # 10行のデータ、明確なラベルで時系列順を確認
        features = pd.DataFrame({"f1": np.arange(10, dtype=float)})
        label = pd.Series(np.arange(10, dtype=float))  # 0,1,...,9

        train_data, valid_data = _train_valid_split(features, label, valid_ratio=0.2)

        # train は前8行 (label 0-7)、valid は後2行 (label 8-9)
        assert len(train_data.get_label()) == 8
        assert len(valid_data.get_label()) == 2
        train_labels = sorted(train_data.get_label())
        assert train_labels == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        valid_labels = sorted(valid_data.get_label())
        assert valid_labels == [8.0, 9.0]

    def test_train_valid_split_no_random_permutation(self) -> None:
        """_train_valid_split が np.random.permutation を使わないことを確認"""
        import inspect

        source = inspect.getsource(_train_valid_split)
        assert "permutation" not in source, "Still using random permutation!"
        assert "RandomState" not in source, "Still using RandomState!"
