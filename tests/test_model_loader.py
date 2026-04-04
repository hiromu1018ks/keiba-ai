"""ModelLoader のテスト"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from domain.models import SubmodelSet, TrainedModelsV5


def _patch_no_local_models():
    """data/models/ が存在しないことを模倣するパッチ"""
    return patch.object(Path, "is_dir", return_value=False)


class TestModelInfo:
    def test_model_info_fields(self) -> None:
        from db.model_loader import ModelInfo

        info = ModelInfo(
            mlflow_run_id="abc123",
            train_start="2020-01-01",
            train_end="2023-12-31",
            loaded_at="2026-04-01 00:00:00",
        )
        assert info.mlflow_run_id == "abc123"
        assert info.train_start == "2020-01-01"
        assert info.train_end == "2023-12-31"
        assert info.loaded_at == "2026-04-01 00:00:00"


class TestModelLoader:
    @patch("db.model_loader.joblib")
    @patch("db.model_loader.mlflow")
    def test_load_returns_trained_models(
        self, mock_mlflow: MagicMock, mock_joblib: MagicMock
    ) -> None:
        from db.model_loader import ModelLoader

        with _patch_no_local_models():
            mock_booster = MagicMock()
            mock_mlflow.lightgbm.load_model.return_value = mock_booster
            mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/artifacts"
            mock_run = MagicMock()
            mock_run.data.params = {
                "train_end": "2023-12-31",
                "train_start": "2020-01-01",
                "quality_threshold": "0.42",
            }
            mock_mlflow.get_run.return_value = mock_run
            mock_mlflow.get_artifact_uri.return_value = "mlruns/1/abc/artifacts"
            mock_mlflow.search_runs.return_value = MagicMock()

            mock_joblib.load.return_value = MagicMock()

            loader = ModelLoader(tracking_uri="file:///mlruns")
            models, info = loader.load(run_id="test_run")

            assert isinstance(models, TrainedModelsV5)
            assert "turf" in models.submodels
            assert "dirt" in models.submodels
            assert info.mlflow_run_id == "test_run"
            assert info.train_start == "2020-01-01"
            assert info.train_end == "2023-12-31"

    @patch("db.model_loader.joblib")
    @patch("db.model_loader.mlflow")
    def test_load_sets_model_attributes(
        self, mock_mlflow: MagicMock, mock_joblib: MagicMock
    ) -> None:
        from db.model_loader import ModelLoader

        with _patch_no_local_models():
            mock_booster = MagicMock()
            mock_mlflow.lightgbm.load_model.return_value = mock_booster
            mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/artifacts"
            mock_run = MagicMock()
            mock_run.data.params = {
                "train_end": "2023-12-31",
                "train_start": "2020-01-01",
                "quality_threshold": "0.42",
            }
            mock_mlflow.get_run.return_value = mock_run
            mock_mlflow.get_artifact_uri.return_value = "mlruns/1/abc/artifacts"
            mock_mlflow.search_runs.return_value = MagicMock()

            mock_joblib.load.return_value = MagicMock()

            loader = ModelLoader(tracking_uri="file:///mlruns")
            models, _ = loader.load(run_id="test_run")

            # Verify submodel set structure
            turf_sub = models.submodels["turf"]
            assert isinstance(turf_sub, SubmodelSet)
            assert turf_sub.market.model is not None
            assert turf_sub.stage1.models is not None
            assert "turf" in turf_sub.stage1.models
            assert turf_sub.win.hit_model is not None
            assert turf_sub.win.return_model is not None
            assert turf_sub.ev_corrector.p_correction_model is not None
            assert turf_sub.ev_corrector.e_correction_model is not None
            assert turf_sub.place.hit_model is not None
            assert turf_sub.place.return_model is not None
            assert turf_sub.wide.hit_model is not None
            assert turf_sub.wide.return_model is not None

            # Verify global models
            assert models.quality_screener.model is not None
            assert models.quality_screener.threshold == 0.42
            assert models.regime_detector.model is not None
            assert models.train_period == ("2020-01-01", "2023-12-31")

    @patch("db.model_loader.joblib")
    @patch("db.model_loader.mlflow")
    def test_load_uses_latest_run_when_no_run_id(
        self, mock_mlflow: MagicMock, mock_joblib: MagicMock
    ) -> None:
        from db.model_loader import ModelLoader

        with _patch_no_local_models():
            mock_booster = MagicMock()
            mock_mlflow.lightgbm.load_model.return_value = mock_booster
            mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/artifacts"
            mock_run = MagicMock()
            mock_run.data.params = {
                "train_end": "2023-12-31",
                "train_start": "2020-01-01",
                "quality_threshold": "0.42",
            }
            mock_mlflow.get_run.return_value = mock_run
            mock_mlflow.get_artifact_uri.return_value = "mlruns/1/abc/artifacts"

            # Mock search_runs to return a DataFrame-like object
            mock_df = MagicMock()
            mock_df.empty = False
            mock_df.iloc = MagicMock()
            mock_df.iloc.__getitem__ = MagicMock(return_value=MagicMock())
            mock_df.iloc.__getitem__().__getitem__ = MagicMock(return_value="latest_run_id")
            mock_mlflow.search_runs.return_value = mock_df

            mock_joblib.load.return_value = MagicMock()

            loader = ModelLoader(tracking_uri="file:///mlruns")
            models, info = loader.load()

            mock_mlflow.search_runs.assert_called_once()
            assert info.mlflow_run_id == "latest_run_id"

    @patch("db.model_loader.mlflow")
    @patch("db.model_loader.Path")
    def test_find_latest_run_raises_on_empty(
        self, mock_path: MagicMock, mock_mlflow: MagicMock
    ) -> None:
        from db.model_loader import ModelLoader

        mock_df = MagicMock()
        mock_df.empty = True
        mock_mlflow.search_runs.return_value = mock_df

        # _find_latest_run_from_fs が mlruns/ ディレクトリを見つけられないようにする
        mock_path.return_value.exists.return_value = False

        loader = ModelLoader(tracking_uri="file:///mlruns")
        with pytest.raises(ValueError, match="mlruns"):
            loader._find_latest_run()

    @patch("db.model_loader.joblib")
    @patch("db.model_loader.mlflow")
    def test_load_handles_missing_place_ability(
        self, mock_mlflow: MagicMock, mock_joblib: MagicMock
    ) -> None:
        from db.model_loader import ModelLoader

        with _patch_no_local_models():
            mock_booster = MagicMock()
            mock_mlflow.lightgbm.load_model.return_value = mock_booster
            mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/empty_artifacts"
            mock_run = MagicMock()
            mock_run.data.params = {
                "train_end": "2023-12-31",
                "train_start": "2020-01-01",
                "quality_threshold": "0.0",
            }
            mock_mlflow.get_run.return_value = mock_run
            mock_mlflow.get_artifact_uri.return_value = "mlruns/1/abc/artifacts"
            mock_mlflow.search_runs.return_value = MagicMock()

            loader = ModelLoader(tracking_uri="file:///mlruns")
            models, _ = loader.load(run_id="test_run")

            # Should still succeed even without PlaceAbilityModel artifacts
            assert "turf" in models.submodels
            assert "dirt" in models.submodels

    @patch("db.model_loader.mlflow")
    def test_init_sets_tracking_uri(self, mock_mlflow: MagicMock) -> None:
        from db.model_loader import ModelLoader

        ModelLoader(tracking_uri="file:///custom_mlruns")
        mock_mlflow.set_tracking_uri.assert_called_once_with("file:///custom_mlruns")
