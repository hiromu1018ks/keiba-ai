"""MLflow ロギング拡張のテスト"""

from unittest.mock import MagicMock, patch

from pipelines.training_pipeline import TrainingPipelineV5


class TestExtendedMLflowLogging:
    """_log_to_mlflow が全モデルを保存することを確認"""

    def test_market_model_logged_per_surface(self) -> None:
        """MarketModel が各surfaceごとにログされる"""
        with (
            patch("pipelines.training_pipeline.mlflow") as mock_mlflow,
            patch("pipelines.training_pipeline.joblib"),
            patch("pipelines.training_pipeline.os"),
            patch("pipelines.training_pipeline.tempfile") as mock_tempfile,
            patch("pipelines.training_pipeline.TrainingPipelineV5._save_models_local"),
        ):
            mock_mlflow.start_run.return_value.__enter__ = MagicMock()
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            mock_tmp = MagicMock()
            mock_tmp.name = "/tmp/place_ability_turf.joblib"
            mock_tempfile.NamedTemporaryFile.return_value = mock_tmp

            mock_sub = MagicMock()
            mock_sub.market.model = MagicMock()
            mock_sub.stage1.models = {"turf": MagicMock()}
            mock_sub.win.hit_model = MagicMock()
            mock_sub.win.return_model = MagicMock()
            mock_sub.ev_corrector.p_correction_model = MagicMock()
            mock_sub.ev_corrector.e_correction_model = MagicMock()
            mock_sub.place.hit_model = MagicMock()
            mock_sub.place.return_model = MagicMock()
            mock_sub.place_ability._model = MagicMock()
            mock_sub.place_ability._calibrated = MagicMock()
            mock_sub.wide.hit_model = MagicMock()
            mock_sub.wide.return_model = MagicMock()

            mock_quality = MagicMock()
            mock_quality.model = MagicMock()
            mock_quality.threshold = 0.42
            mock_regime = MagicMock()
            mock_regime.model = MagicMock()

            mock_confidence = MagicMock()
            mock_confidence.alpha = 0.1
            mock_confidence.rolling_window = 200
            mock_confidence._calibrated = True
            mock_confidence._win_cp_quantile = 0.05
            mock_confidence._place_cp_quantile = 0.08
            mock_confidence._win_rolling_quantile = 0.06
            mock_confidence._place_rolling_quantile = 0.09
            mock_sub.confidence = mock_confidence

            pipeline = TrainingPipelineV5.__new__(TrainingPipelineV5)
            pipeline._log_to_mlflow(
                models={"turf": mock_sub, "dirt": mock_sub},
                quality_screen=mock_quality,
                regime_det=mock_regime,
                train_start="2020-01-01",
                train_end="2024-12-31",
            )

            # market_turf が呼ばれる
            log_model_calls = [
                c.kwargs.get("artifact_path", c[0][1] if len(c[0]) > 1 else "")
                for c in mock_mlflow.lightgbm.log_model.call_args_list
            ]
            assert "market_turf" in log_model_calls
            assert "market_dirt" in log_model_calls
            # wide もログされる
            assert "wide_hit_turf" in log_model_calls
            assert "wide_ret_turf" in log_model_calls
            # confidence params がログされる
            mock_mlflow.log_dict.assert_called()
            # quality threshold がログされる
            log_param_calls = [c[0][0] for c in mock_mlflow.log_param.call_args_list]
            assert "quality_threshold" in log_param_calls

    def test_place_ability_saved_as_artifact(self) -> None:
        """PlaceAbilityModel が joblib artifact として保存される"""
        with (
            patch("pipelines.training_pipeline.mlflow") as mock_mlflow,
            patch("pipelines.training_pipeline.joblib"),
            patch("pipelines.training_pipeline.os"),
            patch("pipelines.training_pipeline.tempfile") as mock_tempfile,
            patch("pipelines.training_pipeline.TrainingPipelineV5._save_models_local"),
        ):
            mock_mlflow.start_run.return_value.__enter__ = MagicMock()
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            mock_tmp = MagicMock()
            mock_tmp.name = "/tmp/place_ability_turf.joblib"
            mock_tempfile.NamedTemporaryFile.return_value = mock_tmp

            mock_sub = MagicMock()
            mock_sub.market.model = MagicMock()
            mock_sub.stage1.models = {"turf": MagicMock()}
            mock_sub.win.hit_model = MagicMock()
            mock_sub.win.return_model = MagicMock()
            mock_sub.ev_corrector.p_correction_model = MagicMock()
            mock_sub.ev_corrector.e_correction_model = MagicMock()
            mock_sub.place.hit_model = MagicMock()
            mock_sub.place.return_model = MagicMock()
            mock_sub.place_ability._model = MagicMock()
            mock_sub.place_ability._calibrated = MagicMock()

            mock_quality = MagicMock()
            mock_quality.model = MagicMock()
            mock_quality.threshold = 0.42
            mock_regime = MagicMock()
            mock_regime.model = MagicMock()

            mock_confidence = MagicMock()
            mock_confidence._calibrated = False
            mock_sub.confidence = mock_confidence

            pipeline = TrainingPipelineV5.__new__(TrainingPipelineV5)
            pipeline._log_to_mlflow(
                models={"turf": mock_sub},
                quality_screen=mock_quality,
                regime_det=mock_regime,
                train_start="2020-01-01",
                train_end="2024-12-31",
            )

            mock_mlflow.log_artifact.assert_called()
