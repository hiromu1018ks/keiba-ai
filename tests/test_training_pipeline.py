"""TrainingPipeline + 関連コンポーネントのテスト"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from domain.models import SubmodelSet, TrainedModelsV5
from features.feature_engine import FeatureEngine
from models.submodel_manager import SubModelManager
from pipelines.training_pipeline import TrainingPipelineV5


def _make_mock_repo() -> MagicMock:
    """DataRepository のモックを作成"""
    mock_repo = MagicMock()
    mock_repo.load_races.return_value = pd.DataFrame()
    mock_repo.load_entries.return_value = pd.DataFrame()
    mock_repo.load_odds_snapshots.return_value = pd.DataFrame()
    return mock_repo


class _FakeHistFeatures:
    """HorseHistoryFeatures のスタブ (DB不要)"""

    def __init__(self, *args, **kwargs):  # noqa: ARG002
        pass

    def compute(self, race_df, entry_df, target_race_ids=None):  # noqa: ARG002
        return pd.DataFrame(columns=["race_id", "umaban"])

    @staticmethod
    def add_race_transforms(df):
        return df


class _FakePlaceAbilityModel:
    """PlaceAbilityModel のスタブ (学習不要)"""

    def train(self, df):  # noqa: ARG002
        pass

    def predict(self, df):
        df = df.copy()
        df["p_ability_place"] = 0.3
        df["p_ability_place_raw"] = 0.3
        return df


class TestTrainedModelsV5:
    """TrainedModelsV5 コンテナのテスト"""

    def test_submodel_set_holds_models(self) -> None:
        """SubmodelSet が全モデルを保持できる"""
        sub = SubmodelSet(
            market=None,
            stage1=None,
            place_ability=None,
            win=None,
            ev_corrector=None,
            place=None,
            wide=None,
            confidence=None,
        )
        assert sub.market is None
        assert sub.confidence is None

    def test_trained_models_v5_structure(self) -> None:
        """TrainedModelsV5 が submodels + screener + detector を保持"""
        models = TrainedModelsV5(
            submodels={
                "turf": SubmodelSet(
                    market=None,
                    stage1=None,
                    place_ability=None,
                    win=None,
                    ev_corrector=None,
                    place=None,
                    wide=None,
                    confidence=None,
                )
            },
            quality_screener=None,
            regime_detector=None,
            train_period=("2020-01-01", "2023-12-31"),
        )
        assert "turf" in models.submodels
        assert models.train_period == ("2020-01-01", "2023-12-31")

    def test_trained_models_v5_supports_both_surfaces(self) -> None:
        """芝・ダート両方のサブモデルを保持できる"""
        models = TrainedModelsV5(
            submodels={
                "turf": SubmodelSet(
                    market="m_turf",
                    stage1="s_turf",
                    place_ability=None,
                    win="w_turf",
                    ev_corrector="e_turf",
                    place="p_turf",
                    wide="wd_turf",
                    confidence="c_turf",
                ),
                "dirt": SubmodelSet(
                    market="m_dirt",
                    stage1="s_dirt",
                    place_ability=None,
                    win="w_dirt",
                    ev_corrector="e_dirt",
                    place="p_dirt",
                    wide="wd_dirt",
                    confidence="c_dirt",
                ),
            },
            quality_screener="qs",
            regime_detector="rd",
            train_period=("2020-01-01", "2023-12-31"),
        )
        assert len(models.submodels) == 2
        assert models.submodels["dirt"].win == "w_dirt"


def _make_feature_df(n: int = 5000, n_races: int = 500) -> pd.DataFrame:
    """テスト用特徴量DataFrameを生成 (モデル学習に必要な全列を含む)"""
    np.random.seed(42)
    horses_per_race = n // n_races
    rows = []
    for r in range(n_races):
        race_id = f"2020{r:04d}0101{r:02d}"
        surface = "turf" if r < n_races // 2 else "dirt"
        distance = np.random.choice([1200, 1400, 1600, 1800, 2000, 2400])
        dist_bin = np.random.choice(["sprint", "mile", "intermediate", "long"])
        for h in range(horses_per_race):
            kyakusitu = np.random.randint(1, 5)
            rows.append(
                {
                    "race_id": race_id,
                    "umaban": h + 1,
                    "surface": surface,
                    "surface_key": surface,
                    "distance": distance,
                    "distance_bin": dist_bin,
                    "track_condition_code": np.random.randint(1, 4),
                    "grade_code": np.random.choice(["A", "B", "C", "D", "E"]),
                    "field_size": horses_per_race,
                    "finish_pos": h + 1 if h == 0 else np.random.randint(2, horses_per_race + 1),
                    "win_odds_actual": np.random.uniform(1.5, 100.0),
                    "place_odds_actual": np.random.uniform(1.1, 20.0),
                    "tan_odds": np.random.uniform(1.5, 100.0),
                    "popularity_rank": h + 1,
                    "running_style": np.random.randint(0, 5),
                    "p_market_win_adj": np.random.uniform(0.01, 0.5),
                    "market_entropy": np.random.uniform(1.0, 3.0),
                    "overround": np.random.uniform(0.15, 0.30),
                    "weight_diff_from_mean": np.random.uniform(-10, 10),
                    "difficulty_score": np.random.uniform(0, 1),
                    # 過去成績 (8)
                    "norm_finish_logit_avg": np.random.uniform(-2, 2),
                    "haron_time_l3_avg": np.random.uniform(-3, 3),
                    "haron_time_l3_zscore": np.random.uniform(-2, 2),
                    "time_diff_avg": np.random.uniform(-1, 1),
                    "corner_1c_avg": np.random.uniform(1, 10),
                    "corner_4c_avg": np.random.uniform(1, 10),
                    "closing_index_avg": np.random.uniform(-0.5, 0.5),
                    "kyakusitu_cd": kyakusitu,
                    # 血統 (6)
                    "blood_surface_wr": np.random.uniform(0.05, 0.2),
                    "blood_distance_wr": np.random.uniform(0.05, 0.2),
                    "blood_condition_wr": float("nan"),
                    "blood_total_wr": np.random.uniform(0.05, 0.2),
                    "blood_prize_log": np.random.uniform(10, 15),
                    "blood_keito_cd": float("nan"),
                    # 交互作用 (3)
                    "kyakusitu_x_distance": f"{kyakusitu}_{dist_bin}",
                    "kyakusitu_x_surface": f"{kyakusitu}_{surface}",
                    "weight_x_distance": np.random.uniform(640000, 880000),
                    # レース内正規化 (5) — race_rank
                    "norm_finish_logit_avg_race_rank": np.random.uniform(0, 1),
                    "haron_time_l3_avg_race_rank": np.random.uniform(0, 1),
                    "time_diff_avg_race_rank": np.random.uniform(0, 1),
                    "corner_1c_avg_race_rank": np.random.uniform(0, 1),
                    "closing_index_avg_race_rank": np.random.uniform(0, 1),
                    # 馬体 (1)
                    "weight_absolute": np.random.uniform(400, 550),
                    "odds_change_rate_30min": np.random.normal(0, 0.1),
                    "odds_volatility_60min": np.random.uniform(0, 0.5),
                    "signed_log_error_win": np.random.normal(0, 0.3),
                    "abs_log_error_win": np.random.uniform(0, 1.0),
                    "market_error_rank_in_race": np.random.uniform(0, 1),
                    "p_ability_win": np.random.uniform(0.01, 0.5),
                    # WinTwoStageModel 必須
                    "odds_drop_rate_60_10": np.random.normal(0, 0.1),
                    "odds_drop_rate_30_10": np.random.normal(0, 0.05),
                    "odds_velocity": np.random.normal(0, 0.02),
                    "odds_volatility": np.random.uniform(0, 0.3),
                    "popularity_change_30_10": np.random.normal(0, 1),
                    "race_date": f"2020-01-{(r % 28 + 1):02d}",
                }
            )
    return pd.DataFrame(rows)


class TestTrainingPipelineV5:
    """TrainingPipelineV5 のテスト"""

    @patch("pipelines.training_pipeline.mlflow")
    def test_run_returns_trained_models_v5(
        self,
        mock_mlflow: MagicMock,
    ) -> None:
        """run() が TrainedModelsV5 を返す"""
        mock_repo = _make_mock_repo()

        feat_df = _make_feature_df(5000, 500)
        with patch.object(FeatureEngine, "build_all", return_value=feat_df):
            with patch.object(
                SubModelManager,
                "add_distance_band_features",
                side_effect=lambda df: df.copy(),
            ):
                with patch(
                    "features.horse_history_features.HorseHistoryFeatures",
                    _FakeHistFeatures,
                ):
                    with patch(
                        "models.place_ability_model.PlaceAbilityModel",
                        _FakePlaceAbilityModel,
                    ):
                        pipeline = TrainingPipelineV5.__new__(TrainingPipelineV5)
                        pipeline.repo = mock_repo
                        pipeline.db = None
                        pipeline.feature_engine = FeatureEngine()
                        pipeline.submodel_mgr = SubModelManager()

                        result = pipeline.run("2020-01-01", "2023-12-31")

        assert isinstance(result, TrainedModelsV5)
        assert "turf" in result.submodels or "dirt" in result.submodels
        assert result.quality_screener is not None
        assert result.regime_detector is not None

    @patch("pipelines.training_pipeline.mlflow")
    def test_pipeline_trains_per_surface(
        self,
        mock_mlflow: MagicMock,
    ) -> None:
        """芝・ダートそれぞれでサブモデルが学習される"""
        feat_df = _make_feature_df(5000, 500)
        feat_df.loc[:2500, "surface_key"] = "turf"
        feat_df.loc[2500:, "surface_key"] = "dirt"
        feat_df.loc[:2500, "surface"] = "turf"
        feat_df.loc[2500:, "surface"] = "dirt"

        mock_repo = _make_mock_repo()

        with patch.object(FeatureEngine, "build_all", return_value=feat_df):
            with patch.object(
                SubModelManager,
                "add_distance_band_features",
                side_effect=lambda df: df.copy(),
            ):
                with patch(
                    "features.horse_history_features.HorseHistoryFeatures",
                    _FakeHistFeatures,
                ):
                    with patch(
                        "models.place_ability_model.PlaceAbilityModel",
                        _FakePlaceAbilityModel,
                    ):
                        pipeline = TrainingPipelineV5.__new__(TrainingPipelineV5)
                        pipeline.repo = mock_repo
                        pipeline.db = None
                        pipeline.feature_engine = FeatureEngine()
                        pipeline.submodel_mgr = SubModelManager()

                        result = pipeline.run("2020-01-01", "2023-12-31")

        assert len(result.submodels) >= 1

    @patch("pipelines.training_pipeline.mlflow")
    def test_pipeline_logs_to_mlflow(
        self,
        mock_mlflow: MagicMock,
    ) -> None:
        """MLflow にモデルが記録される"""
        mock_repo = _make_mock_repo()

        feat_df = _make_feature_df(5000, 500)
        with patch.object(FeatureEngine, "build_all", return_value=feat_df):
            with patch.object(
                SubModelManager,
                "add_distance_band_features",
                side_effect=lambda df: df.copy(),
            ):
                with patch(
                    "features.horse_history_features.HorseHistoryFeatures",
                    _FakeHistFeatures,
                ):
                    with patch(
                        "models.place_ability_model.PlaceAbilityModel",
                        _FakePlaceAbilityModel,
                    ):
                        pipeline = TrainingPipelineV5.__new__(TrainingPipelineV5)
                        pipeline.repo = mock_repo
                        pipeline.db = None
                        pipeline.feature_engine = FeatureEngine()
                        pipeline.submodel_mgr = SubModelManager()

                        pipeline.run("2020-01-01", "2023-12-31")

        mock_mlflow.start_run.assert_called_once()
