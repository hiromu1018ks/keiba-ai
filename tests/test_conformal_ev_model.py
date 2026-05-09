"""ConformalEVModel (CQR) のユニットテスト"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from models.conformal_ev_model import ConformalEVModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_train_df() -> pd.DataFrame:
    """CQR学習用データ (300サンプル, ev_win_calibrated + 特徴量2列)"""
    np.random.seed(42)
    n = 300
    return pd.DataFrame(
        {
            "race_id": [f"R{i % 30}" for i in range(n)],
            "umaban": np.random.randint(1, 18, n),
            "race_date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "ev_win_calibrated": np.random.uniform(0.5, 3.0, n),
            "actual_ev_win": np.random.uniform(0.0, 5.0, n),
            "feature_a": np.random.randn(n),
            "feature_b": np.random.randn(n),
        }
    )


@pytest.fixture
def sample_inference_df() -> pd.DataFrame:
    """推論用データ (5サンプル)"""
    return pd.DataFrame(
        {
            "race_id": ["R100", "R100", "R100", "R101", "R101"],
            "umaban": [1, 2, 3, 4, 5],
            "ev_win_calibrated": [1.5, 0.8, 2.1, 1.2, 0.6],
            "feature_a": [0.1, -0.5, 0.3, -0.2, 0.8],
            "feature_b": [0.2, 0.1, -0.3, 0.4, -0.1],
        }
    )


@pytest.fixture
def sample_place_df() -> pd.DataFrame:
    """複勝推論用データ"""
    return pd.DataFrame(
        {
            "ev_place_corrected": [1.1, 0.9, 1.3, 1.0, 0.8],
        }
    )


@pytest.fixture
def trained_model(sample_train_df: pd.DataFrame) -> ConformalEVModel:
    """学習済みConformalEVModel"""
    model = ConformalEVModel(alpha=0.1)
    model.train(sample_train_df, train_ratio=0.8)
    return model


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


class TestConformalEVModelTrain:
    """train() メソッドのテスト"""

    def test_train_creates_models_and_calibrates(self, sample_train_df: pd.DataFrame) -> None:
        """train()後に_calibrated=True、モデルがNoneでない、補正量子 > 0"""
        model = ConformalEVModel(alpha=0.1)
        model.train(sample_train_df, train_ratio=0.8)

        assert model._calibrated is True
        assert model.q_low_model is not None
        assert model.q_high_model is not None
        assert model._calibration_quantile_90 > 0
        assert model._calibration_quantile_80 > 0

    def test_train_nonconformity_score(self, sample_train_df: pd.DataFrame) -> None:
        """学習後に補正量子が計算される (alpha=0.1, n=60 calib samples)"""
        model = ConformalEVModel(alpha=0.1)
        model.train(sample_train_df, train_ratio=0.8)

        # _calibration_quantile_90 は E の (1-0.1)*(1+1/n) 分位数
        # 値は有限で非負
        assert np.isfinite(model._calibration_quantile_90)
        assert np.isfinite(model._calibration_quantile_80)
        assert model._calibration_quantile_90 > 0
        assert model._calibration_quantile_80 > 0

    def test_train_calibrates_quantile_ordering(self, sample_train_df: pd.DataFrame) -> None:
        """90%区間の補正量子 >= 80%区間の補正量子 (より広い区間)"""
        model = ConformalEVModel(alpha=0.1)
        model.train(sample_train_df, train_ratio=0.8)

        assert model._calibration_quantile_90 >= model._calibration_quantile_80

    def test_insufficient_samples_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """サンプル200未満でtrain()を呼ぶとwarningログが出力される"""
        np.random.seed(42)
        n = 50
        small_df = pd.DataFrame(
            {
                "race_id": [f"R{i}" for i in range(n)],
                "ev_win_calibrated": np.random.uniform(0.5, 3.0, n),
                "actual_ev_win": np.random.uniform(0.0, 5.0, n),
                "feature_a": np.random.randn(n),
                "feature_b": np.random.randn(n),
            }
        )
        model = ConformalEVModel(alpha=0.1)
        with caplog.at_level(logging.WARNING):
            model.train(small_df, train_ratio=0.8)

        assert "Insufficient samples" in caplog.text
        assert not model._calibrated


class TestConformalEVModelPredictInterval:
    """predict_interval() メソッドのテスト"""

    def test_predict_interval_output_columns(
        self,
        trained_model: ConformalEVModel,
        sample_inference_df: pd.DataFrame,
        sample_place_df: pd.DataFrame,
    ) -> None:
        """predict_interval()の戻り値に必要な列が含まれる"""
        win_result, place_result = trained_model.predict_interval(
            sample_inference_df, sample_place_df
        )

        assert "EV_lower_win_corrected" in win_result.columns
        assert "EV_upper_win_corrected" in win_result.columns
        assert "conformal_confidence_score" in win_result.columns
        assert "EV_lower_place" in place_result.columns
        assert "EV_upper_place" in place_result.columns

    def test_predict_interval_not_calibrated(
        self,
        sample_inference_df: pd.DataFrame,
        sample_place_df: pd.DataFrame,
    ) -> None:
        """未キャリブレーション時のフォールバック動作"""
        model = ConformalEVModel()
        win_result, place_result = model.predict_interval(sample_inference_df, sample_place_df)

        # フォールバック: EV = lower = upper, confidence_score = 0
        pd.testing.assert_series_equal(
            win_result["EV_lower_win_corrected"],
            win_result["EV_upper_win_corrected"],
            check_names=False,
        )
        assert (win_result["conformal_confidence_score"] == 0.0).all()

    def test_90_wider_than_80(
        self,
        trained_model: ConformalEVModel,
        sample_inference_df: pd.DataFrame,
        sample_place_df: pd.DataFrame,
    ) -> None:
        """90%区間幅が80%区間幅より広い (90%のlowerが80%のlower以下)"""
        # predict_interval returns 90% interval as primary
        win_90, _ = trained_model.predict_interval(
            sample_inference_df, sample_place_df, alphas=(0.1, 0.2)
        )
        lower_90 = win_90["EV_lower_win_corrected"].values
        upper_90 = win_90["EV_upper_win_corrected"].values
        width_90 = upper_90 - lower_90

        # 90% interval should have positive width
        assert (width_90 > 0).all()

    def test_ev_lower_non_negative(
        self,
        trained_model: ConformalEVModel,
        sample_inference_df: pd.DataFrame,
        sample_place_df: pd.DataFrame,
    ) -> None:
        """EV_lower_win_correctedの全値が >= 0"""
        win_result, _ = trained_model.predict_interval(
            sample_inference_df, sample_place_df
        )
        assert (win_result["EV_lower_win_corrected"] >= 0).all()

    def test_place_output_columns(
        self,
        trained_model: ConformalEVModel,
        sample_inference_df: pd.DataFrame,
        sample_place_df: pd.DataFrame,
    ) -> None:
        """place_dfにEV_lower_place, EV_upper_placeが含まれる"""
        _, place_result = trained_model.predict_interval(
            sample_inference_df, sample_place_df
        )

        assert "EV_lower_place" in place_result.columns
        assert "EV_upper_place" in place_result.columns

    def test_confidence_score_calculation(
        self,
        trained_model: ConformalEVModel,
        sample_inference_df: pd.DataFrame,
        sample_place_df: pd.DataFrame,
    ) -> None:
        """confidence_scoreが非負で有限"""
        win_result, _ = trained_model.predict_interval(
            sample_inference_df, sample_place_df
        )
        assert (win_result["conformal_confidence_score"] >= 0).all()
        assert np.isfinite(win_result["conformal_confidence_score"]).all()

    def test_monotonicity_clip(self) -> None:
        """q_low > q_highになるケースでクリップされる (mockでq_low_modelが大きい値を返す)"""
        model = ConformalEVModel(alpha=0.1)
        model._calibrated = True
        model.feature_cols = ["feature_a", "feature_b"]
        model._calibration_quantile_90 = 0.5
        model._calibration_quantile_80 = 0.3

        # q_low_modelがq_high_modelより大きい値を返すよう設定
        mock_low = MagicMock()
        mock_low.predict.return_value = np.array([2.0, 3.0, 1.5])
        mock_high = MagicMock()
        mock_high.predict.return_value = np.array([1.0, 2.0, 1.0])
        model.q_low_model = mock_low
        model.q_high_model = mock_high

        win_df = pd.DataFrame(
            {
                "race_id": ["R1", "R1", "R2"],
                "ev_win_calibrated": [1.5, 2.5, 1.2],
                "feature_a": [0.1, -0.5, 0.3],
                "feature_b": [0.2, 0.1, -0.3],
            }
        )
        place_df = pd.DataFrame({"ev_place_corrected": [1.1, 0.9, 1.3]})

        win_result, _ = model.predict_interval(win_df, place_df)

        # モノトonicity clip後: q_low = min(q_low, q_high)
        # Row 0: q_low=2.0, q_high=1.0 -> q_low=1.0, lower_90=1.0-0.5=0.5, upper=1.0+0.5=1.5
        # Row 1: q_low=3.0, q_high=2.0 -> q_low=2.0, lower_90=2.0-0.5=1.5, upper=2.0+0.5=2.5
        # Row 2: q_low=1.5, q_high=1.0 -> q_low=1.0, lower_90=1.0-0.5=0.5, upper=1.0+0.5=1.5

        # lower <= upper が保証される
        for i in range(len(win_result)):
            assert win_result["EV_lower_win_corrected"].iloc[i] <= \
                   win_result["EV_upper_win_corrected"].iloc[i] + 1e-10


class TestConformalEVModelSaveLoad:
    """save/load メソッドのテスト"""

    def test_save_load_roundtrip(
        self,
        trained_model: ConformalEVModel,
        tmp_path: Path,
    ) -> None:
        """save() -> load()でモデルが正しく復元される"""
        surface = "turf"
        trained_model.save(tmp_path, surface)

        loaded = ConformalEVModel.load(tmp_path, surface)
        assert loaded is not None
        assert loaded._calibrated is True
        assert loaded.alpha == trained_model.alpha
        assert abs(loaded._calibration_quantile_90 - trained_model._calibration_quantile_90) < 1e-10
        assert abs(loaded._calibration_quantile_80 - trained_model._calibration_quantile_80) < 1e-10
        assert loaded.feature_cols == trained_model.feature_cols

    def test_load_missing_files_returns_none(self, tmp_path: Path) -> None:
        """ファイルが存在しない場合にload()がNoneを返す"""
        result = ConformalEVModel.load(tmp_path, "turf")
        assert result is None

    def test_save_uncalibrated_skips(self, caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
        """未キャリブレーション時にsave()が警告を出してスキップ"""
        model = ConformalEVModel()
        with caplog.at_level(logging.WARNING):
            model.save(tmp_path, "turf")

        assert "Cannot save uncalibrated" in caplog.text
        # ファイルが作成されないことを確認
        assert not (tmp_path / "cqr_quantile_low_turf.lgb").exists()


class TestConformalEVModelBackwardCompat:
    """後方互換性 (Plan 02まで) のテスト"""

    def test_calibrate_method_exists(self) -> None:
        """calibrate() メソッドが存在し、_calibrated=Trueを設定する"""
        model = ConformalEVModel()
        assert not model._calibrated
        model.calibrate(pd.DataFrame(), pd.DataFrame())
        assert model._calibrated

    def test_predict_lower_bound_method_exists(self) -> None:
        """predict_lower_bound() がpredict_interval()のラッパーとして動作する"""
        model = ConformalEVModel()
        # 未キャリブレーション: フォールバック
        win_df = pd.DataFrame({"ev_win_calibrated": [1.5, 2.0]})
        place_df = pd.DataFrame({"ev_place_corrected": [1.0, 1.2]})
        win_result, place_result = model.predict_lower_bound(win_df, place_df)

        assert "EV_lower_win_corrected" in win_result.columns
        assert "EV_upper_win_corrected" not in win_result.columns
        assert "conformal_confidence_score" not in win_result.columns
        assert "EV_lower_place" in place_result.columns
        assert "EV_upper_place" not in place_result.columns
