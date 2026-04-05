"""src/models/wide_two_stage_model.py のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from models.wide_two_stage_model import WideTwoStageModel, _train_valid_split


@pytest.fixture
def pair_df() -> pd.DataFrame:
    """ワイド馬券ペアのテストデータ"""
    return pd.DataFrame(
        {
            "race_id": ["R1"] * 3,
            "umaban_a": [1, 1, 2],
            "umaban_b": [2, 3, 3],
            "popularity_sum": [3, 5, 9],
            "running_style_combo": [3, 0, 5],
            "p_hit": [0.30, 0.10, 0.05],
            "e_return_given_hit": [4.0, 12.0, 24.0],
            "surface": ["turf"] * 3,
            "distance_bin": ["mile"] * 3,
            "track_condition_code": [1] * 3,
            "grade_code": ["_"] * 3,
            "field_size": [3] * 3,
        }
    )


def _make_mock_booster(pred_values: list[float]) -> MagicMock:
    """best_iteration 属性付きの mock Booster を生成"""
    mock = MagicMock()
    mock.predict.return_value = np.array(pred_values)
    mock.best_iteration = 100
    return mock


@pytest.fixture
def trained_wide_model(pair_df: pd.DataFrame) -> WideTwoStageModel:
    """学習済みWideTwoStageModel (mock)"""
    model = WideTwoStageModel()
    model.hit_model = _make_mock_booster([0.30, 0.10, 0.05])
    model.return_model = _make_mock_booster([4.0, 12.0, 24.0])
    return model


class TestWideTwoStageModel:
    def test_score_is_ev_over_e_sqrt_p(
        self,
        trained_wide_model: WideTwoStageModel,
        pair_df: pd.DataFrame,
    ) -> None:
        """score = EV / (E * sqrt(P)) (Rule 3, Rule 15)"""
        result = trained_wide_model.predict_score(pair_df)
        for i in range(len(result)):
            ev = result["ev_wide"].iloc[i]
            e = result["e_return_given_hit"].iloc[i]
            p = result["p_hit"].iloc[i]
            expected_score = ev / (e * np.sqrt(max(p, 0.001)))
            assert abs(result["wide_score_adj"].iloc[i] - expected_score) < 1e-10

    def test_ev_equals_p_times_e(
        self,
        trained_wide_model: WideTwoStageModel,
        pair_df: pd.DataFrame,
    ) -> None:
        result = trained_wide_model.predict_score(pair_df)
        expected_ev = result["p_hit"] * result["e_return_given_hit"]
        assert np.allclose(result["ev_wide"].values, expected_ev.values)

    def test_select_bets_dual_filter(
        self,
        trained_wide_model: WideTwoStageModel,
        pair_df: pd.DataFrame,
    ) -> None:
        """2段階フィルタ: ev_threshold + score_threshold"""
        bets = trained_wide_model.select_bets(
            pair_df,
            ev_threshold=1.20,
            score_threshold=0.015,
            max_bets=3,
        )
        for bet in bets:
            assert bet["ev_wide"] >= 1.20
            assert bet["wide_score_adj"] >= 0.015

    def test_select_bets_max_bets(
        self,
        trained_wide_model: WideTwoStageModel,
        pair_df: pd.DataFrame,
    ) -> None:
        """max_bets で返す数が制限される"""
        bets = trained_wide_model.select_bets(pair_df, max_bets=2)
        assert len(bets) <= 2

    def test_select_bets_excludes_zero_style(
        self,
        trained_wide_model: WideTwoStageModel,
        pair_df: pd.DataFrame,
    ) -> None:
        """running_style_combo == 0 のペアは除外される"""
        bets = trained_wide_model.select_bets(pair_df, ev_threshold=0.0, score_threshold=0.0)
        for bet in bets:
            assert bet["running_style_combo"] != 0

    def test_select_bets_popularity_filter(
        self,
        trained_wide_model: WideTwoStageModel,
        pair_df: pd.DataFrame,
    ) -> None:
        """popularity_sum >= 6 のペアのみ"""
        bets = trained_wide_model.select_bets(pair_df, ev_threshold=0.0, score_threshold=0.0)
        for bet in bets:
            assert bet["popularity_sum"] >= 6

    def test_sharpe_ratio_approximation(self) -> None:
        """異なるP/Eペアでスコアがシャープレシオ的になることを確認"""
        # ペアA: P=0.30, E=4.0 -> score = 1.20 / (4.0 * sqrt(0.30)) = 0.548
        # ペアB: P=0.05, E=24.0 -> score = 1.20 / (24.0 * sqrt(0.05)) = 0.224
        p_a, e_a = 0.30, 4.0
        p_b, e_b = 0.05, 24.0
        score_a = (p_a * e_a) / (e_a * np.sqrt(p_a))
        score_b = (p_b * e_b) / (e_b * np.sqrt(p_b))
        # 高P低E (A) の方が低P高E (B) よりスコアが高い
        assert score_a > score_b


class TestEarlyStopping:
    """早期停止のテスト群"""

    @patch("models.wide_two_stage_model.lgb")
    def test_train_hit_model_uses_early_stopping(
        self, mock_lgb: MagicMock, pair_df: pd.DataFrame
    ) -> None:
        """train_hit_model が valid_sets と callbacks を渡すこと"""
        mock_booster = MagicMock()
        mock_booster.best_iteration = 50
        mock_lgb.train.return_value = mock_booster
        mock_lgb.Dataset.return_value = MagicMock()
        mock_lgb.early_stopping.return_value = lambda: None

        df = pair_df.copy()
        df["joint_hit"] = [1, 0, 1]

        model = WideTwoStageModel()
        model.train_hit_model(df)

        call_args = mock_lgb.train.call_args
        assert "valid_sets" in call_args[1]
        assert "callbacks" in call_args[1]
        mock_lgb.early_stopping.assert_called_once_with(stopping_rounds=100, verbose=False)

    @patch("models.wide_two_stage_model.lgb")
    def test_train_return_model_uses_early_stopping(self, mock_lgb: MagicMock) -> None:
        """train_return_model がサンプル十分時に valid_sets と callbacks を渡すこと"""
        mock_booster = MagicMock()
        mock_booster.best_iteration = 50
        mock_lgb.train.return_value = mock_booster
        mock_lgb.Dataset.return_value = MagicMock()
        mock_lgb.early_stopping.return_value = lambda: None

        # 15件の的中ペア (> 10) -- early stopping 有効
        n = 15
        df = pd.DataFrame(
            {
                "joint_hit": [1] * n,
                "wide_odds": list(range(2, 2 + n)),
                "surface": ["turf"] * n,
                "distance_bin": ["mile"] * n,
                "track_condition_code": [1] * n,
                "grade_code": ["_"] * n,
                "field_size": [n] * n,
            }
        )

        from domain.models import TwoStageConfig

        cfg = TwoStageConfig(min_hit_samples=5)
        model = WideTwoStageModel()
        model.train_return_model(df, cfg=cfg)

        call_args = mock_lgb.train.call_args
        assert "valid_sets" in call_args[1]
        assert "callbacks" in call_args[1]
        mock_lgb.early_stopping.assert_called_once_with(stopping_rounds=100, verbose=False)

    @patch("models.wide_two_stage_model.lgb")
    def test_train_return_model_early_stopping_guard(self, mock_lgb: MagicMock) -> None:
        """return_model: サンプル < 10 は early stopping なし"""
        mock_booster = MagicMock()
        mock_booster.best_iteration = 50
        mock_lgb.train.return_value = mock_booster
        mock_lgb.Dataset.return_value = MagicMock()

        # 7件のみの的中データ (< 10) -- early stopping 無効
        n = 7
        df = pd.DataFrame(
            {
                "joint_hit": [1] * n,
                "wide_odds": list(range(2, 2 + n)),
                "surface": ["turf"] * n,
                "distance_bin": ["mile"] * n,
                "track_condition_code": [1] * n,
                "grade_code": ["_"] * n,
                "field_size": [n] * n,
            }
        )

        from domain.models import TwoStageConfig

        cfg = TwoStageConfig(min_hit_samples=5)
        model = WideTwoStageModel()
        model.train_return_model(df, cfg=cfg)

        call_args = mock_lgb.train.call_args
        # サンプル数 < 10 のため valid_sets が無いことを確認
        assert "valid_sets" not in call_args[1]
        assert "callbacks" not in call_args[1]

    def test_predict_uses_best_iteration(
        self,
        trained_wide_model: WideTwoStageModel,
        pair_df: pd.DataFrame,
    ) -> None:
        """predict が num_iteration=best_iteration を使用すること"""
        trained_wide_model.predict_score(pair_df)

        trained_wide_model.hit_model.predict.assert_called_once()
        call_kwargs = trained_wide_model.hit_model.predict.call_args
        assert call_kwargs[1]["num_iteration"] == 100

        trained_wide_model.return_model.predict.assert_called_once()
        call_kwargs = trained_wide_model.return_model.predict.call_args
        assert call_kwargs[1]["num_iteration"] == 100

    def test_predict_best_iteration_zero_falls_back(
        self,
        pair_df: pd.DataFrame,
    ) -> None:
        """best_iteration == 0 の場合は num_iteration=None にフォールバック"""
        model = WideTwoStageModel()
        mock_hit = MagicMock()
        mock_hit.predict.return_value = np.array([0.30, 0.10, 0.05])
        mock_hit.best_iteration = 0
        mock_return = MagicMock()
        mock_return.predict.return_value = np.array([4.0, 12.0, 24.0])
        mock_return.best_iteration = 0
        model.hit_model = mock_hit
        model.return_model = mock_return

        model.predict_score(pair_df)

        call_kwargs = mock_hit.predict.call_args
        assert call_kwargs[1]["num_iteration"] is None
        call_kwargs = mock_return.predict.call_args
        assert call_kwargs[1]["num_iteration"] is None


class TestTrainValidSplit:
    def test_split_ratio(self) -> None:
        """80/20 分割が正しいサイズを返す"""
        features = pd.DataFrame({"a": range(100), "b": range(100)})
        label = pd.Series(range(100))

        with patch("models.wide_two_stage_model.lgb.Dataset") as mock_ds:
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

        with patch("models.wide_two_stage_model.lgb.Dataset") as mock_ds:
            mock_ds.return_value = MagicMock()
            _train_valid_split(features, label)

        train_idx = mock_ds.call_args_list[0][0][0].index
        valid_idx = mock_ds.call_args_list[1][0][0].index

        assert len(set(train_idx) & set(valid_idx)) == 0

    def test_ndarray_label(self) -> None:
        """np.ndarray の label も処理できること"""
        features = pd.DataFrame({"a": range(20), "b": range(20)})
        label = np.arange(20)

        with patch("models.wide_two_stage_model.lgb.Dataset") as mock_ds:
            mock_ds.return_value = MagicMock()
            _train_valid_split(features, label)

        # 2回呼ばれる (train_data, valid_data)
        assert mock_ds.call_count == 2
        assert len(mock_ds.call_args_list[0][0][0]) == 16
        assert len(mock_ds.call_args_list[1][0][0]) == 4
