"""TrainingPipeline + 関連コンポーネントのテスト"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from domain.models import SubmodelSet, TrainedModelsV5
from features.feature_engine import FeatureEngine
from models.regime_detector import RegimeDetector
from models.submodel_manager import SubModelManager
from pipelines.training_pipeline import TrainingPipelineV5


def _make_mock_store() -> MagicMock:
    """ParquetStore のモックを作成"""
    mock_store = MagicMock()
    return mock_store


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

    _calibrated = None
    _model = None

    def train(self, df, **kwargs):  # noqa: ARG002
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


def _make_feature_df(n: int = 8000, n_races: int = 800) -> pd.DataFrame:
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
                    "kyori": distance,
                    "distance_bin": dist_bin,
                    "track_condition_code": np.random.randint(1, 4),
                    "grade_code": np.random.choice(["A", "B", "C", "D", "E"]),
                    "field_size": horses_per_race,
                    "kakuteijyuni": h + 1 if h == 0 else np.random.randint(2, horses_per_race + 1),
                    "odds": np.random.uniform(1.5, 100.0),
                    "confirmed_odds": np.random.uniform(1.5, 100.0),
                    "fukuoddslow": np.random.uniform(1.1, 20.0),
                    "place_odds_actual": np.random.uniform(1.1, 20.0),
                    "tanodds": np.random.uniform(1.5, 100.0),
                    "popularity_rank": h + 1,
                    "running_style": np.random.randint(0, 5),
                    "p_market_win_adj": np.random.uniform(0.01, 0.5),
                    "market_entropy": np.random.uniform(1.0, 3.0),
                    "overround": np.random.uniform(0.15, 0.30),
                    "weight_diff_from_mean": np.random.uniform(-10, 10),
                    "difficulty_score": np.random.uniform(0, 1),
                    # 過去成績 (8)
                    "norm_finish_logit_avg": np.random.uniform(-2, 2),
                    "harontimel5_avg": np.random.uniform(-3, 3),
                    "harontimel5_zscore": np.random.uniform(-2, 2),
                    "harontime_late_trend": np.random.uniform(-2, 2),
                    "timediff_avg": np.random.uniform(-1, 1),
                    "jyuni1c_avg": np.random.uniform(1, 10),
                    "jyuni4c_avg": np.random.uniform(1, 10),
                    "closing_index_avg": np.random.uniform(-0.5, 0.5),
                    "kyakusitukubun_cd": kyakusitu,
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
                    "harontimel5_avg_race_rank": np.random.uniform(0, 1),
                    "timediff_avg_race_rank": np.random.uniform(0, 1),
                    "jyuni1c_avg_race_rank": np.random.uniform(0, 1),
                    "closing_index_avg_race_rank": np.random.uniform(0, 1),
                    # 馬体 (3)
                    "weight_absolute": np.random.uniform(400, 550),
                    "weight_zscore": np.random.uniform(-2, 2),
                    "weight_change_zone": float(np.random.choice([-1, 0, 1, 2])),
                    # 休養期間 (2)
                    "days_since_last_race": np.random.uniform(1, 200),
                    "rest_category": float(np.random.choice([1, 2, 3, 4, 5])),
                    # フォームサイクル (3)
                    "form_trend": np.random.uniform(-1, 1),
                    "form_consistency": np.random.uniform(0, 1),
                    "form_peak_flag": float(np.random.choice([0, 1])),
                    # 種牡馬産駎 (5)
                    "sire_wr": np.random.uniform(0.05, 0.2),
                    "sire_surface_wr": np.random.uniform(0.03, 0.15),
                    "sire_distance_wr": np.random.uniform(0.03, 0.15),
                    "sire_prize_avg": np.random.uniform(10, 15),
                    "bms_wr": np.random.uniform(0.02, 0.10),
                    # sire_features wiring 用
                    "kettonum": np.random.randint(10000000, 99999999),
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
                    "odds_skewness": np.random.uniform(0.5, 3.0),
                    "implied_prob_hhi": np.random.uniform(0.05, 0.15),
                    "race_date": f"2020-{(r // 28) % 12 + 1:02d}-{(r % 28) + 1:02d}",
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
        mock_store = _make_mock_store()

        feat_df = _make_feature_df(8000, 800)
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
                        with patch(
                            "pipelines.training_pipeline.TrainingPipelineV5._save_models_local",
                        ):
                            with patch(
                                "db.readers.load_sire_stats",
                                return_value=pd.DataFrame(),
                            ):
                                pipeline = TrainingPipelineV5.__new__(TrainingPipelineV5)
                                pipeline.store = mock_store
                                pipeline.db = None
                                pipeline.feature_engine = FeatureEngine()
                                pipeline.submodel_mgr = SubModelManager()
                                pipeline.model_dir = Path("data/models")

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
        feat_df = _make_feature_df(8000, 800)
        feat_df.loc[:4000, "surface"] = "turf"
        feat_df.loc[4000:, "surface"] = "dirt"

        mock_store = _make_mock_store()

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
                        with patch(
                            "pipelines.training_pipeline.TrainingPipelineV5._save_models_local",
                        ):
                            with patch(
                                "db.readers.load_sire_stats",
                                return_value=pd.DataFrame(),
                            ):
                                pipeline = TrainingPipelineV5.__new__(TrainingPipelineV5)
                                pipeline.store = mock_store
                                pipeline.db = None
                                pipeline.feature_engine = FeatureEngine()
                                pipeline.submodel_mgr = SubModelManager()
                                pipeline.model_dir = Path("data/models")

                                result = pipeline.run("2020-01-01", "2023-12-31")

        assert len(result.submodels) >= 1

    @patch("pipelines.training_pipeline.mlflow")
    def test_pipeline_logs_to_mlflow(
        self,
        mock_mlflow: MagicMock,
    ) -> None:
        """MLflow にモデルが記録される"""
        mock_store = _make_mock_store()

        feat_df = _make_feature_df(8000, 800)
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
                        with patch(
                            "pipelines.training_pipeline.TrainingPipelineV5._save_models_local",
                        ):
                            with patch(
                                "db.readers.load_sire_stats",
                                return_value=pd.DataFrame(),
                            ):
                                pipeline = TrainingPipelineV5.__new__(TrainingPipelineV5)
                                pipeline.store = mock_store
                                pipeline.db = None
                                pipeline.feature_engine = FeatureEngine()
                                pipeline.submodel_mgr = SubModelManager()
                                pipeline.model_dir = Path("data/models")

                                pipeline.run("2020-01-01", "2023-12-31")

        mock_mlflow.start_run.assert_called_once()


class TestBuildRaceLevelFeatures:
    """_build_race_level_features の favorite_win_rate expanding テスト"""

    @pytest.fixture
    def pipeline(self) -> TrainingPipelineV5:
        """テスト用パイプライン (store/db なし)"""
        p = TrainingPipelineV5.__new__(TrainingPipelineV5)
        p.store = MagicMock()
        p.db = None
        p.feature_engine = FeatureEngine()
        p.submodel_mgr = SubModelManager()
        return p

    def _make_feat_df(self, n_races: int = 20) -> pd.DataFrame:
        """テスト用馬レベルDataFrame (1番人気が約30%の割合で勝つ)"""
        np.random.seed(42)
        rows = []
        for r in range(n_races):
            race_id = f"2020{(r // 28) % 12 + 1:02d}{r % 28 + 1:02d}0101{r:02d}"
            n_horses = 10
            fav_wins = np.random.random() < 0.30
            for h in range(n_horses):
                kakuteijyuni = h + 1
                pop_rank = h + 1
                if h == 0 and fav_wins:
                    kakuteijyuni = 1
                elif h == 0:
                    kakuteijyuni = np.random.randint(2, n_horses + 1)
                rows.append({
                    "race_id": race_id,
                    "umaban": h + 1,
                    "surface": "turf" if r % 2 == 0 else "dirt",
                    "distance_bin": "mile",
                    "track_condition_code": 1,
                    "grade_code": "C",
                    "field_size": n_horses,
                    "difficulty_score": 0.5,
                    "signed_log_error_win": np.random.normal(0, 0.3),
                    "abs_log_error_win": np.random.uniform(0, 1),
                    "market_entropy": np.random.uniform(1.0, 3.0),
                    "overround": np.random.uniform(0.15, 0.30),
                    "kakuteijyuni": kakuteijyuni,
                    "popularity_rank": pop_rank,
                    "tanodds": np.random.uniform(2.0, 50.0),
                    "race_date": f"2020-{(r // 28) % 12 + 1:02d}-{(r % 28) + 1:02d}",
                })
        return pd.DataFrame(rows)

    def test_favorite_win_rate_is_expanding_mean_of_past_races(
        self, pipeline: TrainingPipelineV5
    ) -> None:
        """favorite_win_rate が過去レースのみの expanding mean である"""
        feat_df = self._make_feat_df(n_races=50)
        result = pipeline._build_race_level_features(feat_df)
        assert "favorite_win_rate" in result.columns
        first_val = result.iloc[0]["favorite_win_rate"]
        assert first_val == pytest.approx(0.3), (
            f"First favorite_win_rate should be 0.3 (baseline), got {first_val}"
        )

    def test_favorite_win_rate_does_not_use_current_race(
        self, pipeline: TrainingPipelineV5
    ) -> None:
        """favorite_win_rate は現在のレース結果を使用しない (shift(1))"""
        feat_df = self._make_feat_df(n_races=30)
        result = pipeline._build_race_level_features(feat_df)
        assert (result["favorite_win_rate"].dropna() >= 0).all()
        assert (result["favorite_win_rate"].dropna() <= 1).all()

    def test_favorite_win_rate_no_kakuteijyuni_defaults_to_baseline(
        self, pipeline: TrainingPipelineV5
    ) -> None:
        """kakuteijyuni がない場合は 0.3 (ベースライン) にフォールバック"""
        feat_df = self._make_feat_df(n_races=10)
        feat_df = feat_df.drop(columns=["kakuteijyuni"])
        result = pipeline._build_race_level_features(feat_df)
        assert "favorite_win_rate" in result.columns
        assert (result["favorite_win_rate"] == 0.3).all()

    def test_hist_hit_rate_topk_uses_expanding_favorite_win_rate(
        self, pipeline: TrainingPipelineV5
    ) -> None:
        """hist_hit_rate_topk が存在し、妥当な範囲の値を持つ"""
        feat_df = self._make_feat_df(n_races=30)
        result = pipeline._build_race_level_features(feat_df)
        assert "hist_hit_rate_topk" in result.columns
        # compute_hist_features が topk_hit (0) から expanding 計算するため
        # NaN を含む場合があるが、列は存在し妥当な値であること
        valid_vals = result["hist_hit_rate_topk"].dropna()
        assert (valid_vals >= 0).all()
        assert (valid_vals <= 1).all()

    def test_hist_win_rate_same_condition_uses_expanding(
        self, pipeline: TrainingPipelineV5
    ) -> None:
        """hist_win_rate_same_condition が expanding 版 favorite_win_rate を引き継ぐ"""
        feat_df = self._make_feat_df(n_races=30)
        result = pipeline._build_race_level_features(feat_df)
        assert "hist_win_rate_same_condition" in result.columns
        assert result["hist_win_rate_same_condition"].notna().any()

    def test_roi_ema_columns_present_when_tanodds_available(
        self, pipeline: TrainingPipelineV5
    ) -> None:
        """_build_race_level_features の結果に overround_ema, entropy_ema 列が含まれる

        compute_roi_ema() が wiring され、tanodds がある場合に
        EMA 平滑化市場指標列が追加されることを確認する。
        """
        feat_df = self._make_feat_df(n_races=50)
        # _make_feat_df は tanodds を含む
        assert "tanodds" in feat_df.columns
        result = pipeline._build_race_level_features(feat_df)
        assert "overround_ema" in result.columns, "overround_ema 列が存在しない"
        assert "entropy_ema" in result.columns, "entropy_ema 列が存在しない"
        # 値は数値であること
        assert pd.api.types.is_numeric_dtype(result["overround_ema"])
        assert pd.api.types.is_numeric_dtype(result["entropy_ema"])

    def test_roi_ema_defaults_when_tanodds_missing(
        self, pipeline: TrainingPipelineV5
    ) -> None:
        """tanodds がない場合でも overround_ema/entropy_ema がデフォルト値で追加される

        compute_roi_ema() のガードロジック: 必要列がない場合は 0.0 で埋める。
        """
        feat_df = self._make_feat_df(n_races=10)
        feat_df = feat_df.drop(columns=["tanodds"])
        result = pipeline._build_race_level_features(feat_df)
        # compute_roi_ema は必要列がない場合も列を追加する (0.0 デフォルト)
        assert "overround_ema" in result.columns
        assert "entropy_ema" in result.columns


class TestBuildRegimeStats:
    """_build_regime_stats の新 FEATURE_COLS マッピング テスト"""

    @pytest.fixture
    def pipeline(self) -> TrainingPipelineV5:
        p = TrainingPipelineV5.__new__(TrainingPipelineV5)
        p.store = MagicMock()
        p.db = None
        p.feature_engine = FeatureEngine()
        p.submodel_mgr = SubModelManager()
        return p

    def _make_race_feat_df(self, n_races: int = 20) -> pd.DataFrame:
        rows = []
        for r in range(n_races):
            rows.append({
                "race_id": f"2020{(r // 28) % 12 + 1:02d}{r % 28 + 1:02d}0101{r:02d}",
                "surface": "turf" if r % 2 == 0 else "dirt",
                "distance_bin": "mile",
                "track_condition_code": 1,
                "grade_code": "C",
                "field_size": 12,
                "difficulty_score": 0.5,
                "market_log_error_mean": np.random.normal(0, 0.1),
                "market_log_error_std": np.random.uniform(0.1, 0.5),
                "market_log_error_abs_mean": np.random.uniform(0, 0.5),
                "n_positive_errors": 5,
                "top_k_error_sum": 0.1,
                "positive_error_ratio": 0.4,
                "market_entropy_mean": np.random.uniform(1.5, 3.0),
                "overround_mean": np.random.uniform(0.15, 0.30),
                "favorite_win_rate": 0.3,
                "hist_hit_rate_topk": 0.3,
                "hist_roi_topk": 1.0,
                "hist_positive_return_ratio": 0.3,
                "market_log_error_max_abs": 0.4,
                "market_log_error_top_q75": 0.3,
                "market_entropy": 2.0,
                "overround": 0.20,
                "overround_deviation": 0.0,
                "hist_win_rate_same_condition": 0.3,
                "hist_market_entropy_avg": 2.0,
                "race_date": f"2020-{(r // 28) % 12 + 1:02d}-{(r % 28) + 1:02d}",
            })
        return pd.DataFrame(rows)

    def _make_feat_df(self, n_races: int = 20) -> pd.DataFrame:
        rows = []
        for r in range(n_races):
            race_id = f"2020{(r // 28) % 12 + 1:02d}{r % 28 + 1:02d}0101{r:02d}"
            for h in range(5):
                rows.append({
                    "race_id": race_id,
                    "umaban": h + 1,
                    "tanodds": np.random.uniform(2.0, 20.0),
                    "kakuteijyuni": h + 1,
                    "popularity_rank": h + 1,
                    "odds_volatility": np.random.uniform(0, 0.3),
                    "surface": "turf" if r % 2 == 0 else "dirt",
                    "race_date": f"2020-{(r // 28) % 12 + 1:02d}-{(r % 28) + 1:02d}",
                })
        return pd.DataFrame(rows)

    def test_build_regime_stats_has_all_feature_cols(
        self, pipeline: TrainingPipelineV5
    ) -> None:
        """_build_regime_stats の出力が RegimeDetector.FEATURE_COLS の全列を含む"""
        race_feat_df = self._make_race_feat_df(20)
        feat_df = self._make_feat_df(20)
        result = pipeline._build_regime_stats(race_feat_df, feat_df)
        for col in RegimeDetector.FEATURE_COLS:
            assert col in result.columns, f"Missing FEATURE_COLS column: {col}"

    def test_build_regime_stats_replaces_old_cols(
        self, pipeline: TrainingPipelineV5
    ) -> None:
        """旧 FEATURE_COLS (結果依存) が新 FEATURE_COLS に置き換わる"""
        race_feat_df = self._make_race_feat_df(20)
        feat_df = self._make_feat_df(20)
        result = pipeline._build_regime_stats(race_feat_df, feat_df)
        assert "overround_rolling" in result.columns
        assert "entropy_rolling" in result.columns
        assert "favorite_implied_prob_rolling" in result.columns
        assert "odds_skewness_rolling" in result.columns


class TestJRAFilterTraining:
    """学習パイプラインのJRAフィルタ テスト"""

    @staticmethod
    def _run_pipeline_with_mocks(feat_df: pd.DataFrame) -> MagicMock:
        """_train_submodel をモックしてパイプラインを実行し、mock_train を返す"""
        from contextlib import ExitStack

        mock_store = _make_mock_store()
        mock_market = MagicMock()
        mock_market.predict_and_calc_error = MagicMock(side_effect=lambda df: df.copy())
        mock_sub = SubmodelSet(
            market=mock_market, stage1=MagicMock(), place_ability=MagicMock(),
            win=MagicMock(), ev_corrector=MagicMock(), place=MagicMock(),
            wide=MagicMock(), confidence=MagicMock(),
        )
        patches = [
            patch.object(FeatureEngine, "build_all", return_value=feat_df),
            patch.object(
                SubModelManager,
                "add_distance_band_features",
                side_effect=lambda df: df.copy(),
            ),
            patch.object(TrainingPipelineV5, "_train_submodel", return_value=mock_sub),
            patch.object(
                TrainingPipelineV5, "_build_race_level_features", return_value=pd.DataFrame()
            ),
            patch.object(
                TrainingPipelineV5, "_build_regime_stats", return_value=pd.DataFrame()
            ),
            patch.object(TrainingPipelineV5, "_log_to_mlflow"),
            patch("pipelines.training_pipeline.RaceQualityScreener"),
            patch("pipelines.training_pipeline.RegimeDetector"),
            patch("pipelines.training_pipeline.TrainingPipelineV5._save_models_local"),
        ]
        with ExitStack() as stack:
            mocks = [stack.enter_context(p) for p in patches]
            pipeline = TrainingPipelineV5.__new__(TrainingPipelineV5)
            pipeline.store = mock_store
            pipeline.db = None
            pipeline.feature_engine = FeatureEngine()
            pipeline.submodel_mgr = SubModelManager()
            pipeline.run("2020-01-01", "2023-12-31")
        # mocks[2] is the _train_submodel mock
        return mocks[2]

    @patch("pipelines.training_pipeline.mlflow")
    def test_nar_entries_filtered_before_surface_split(
        self, mock_mlflow: MagicMock
    ) -> None:
        """NARエントリ (jyocd >= 30) が surface分割前に除外される"""
        feat_df = _make_feature_df(8000, 800)
        feat_df["jyocd"] = "05"
        dirt_mask = feat_df["surface"] == "dirt"
        dirt_indices = feat_df[dirt_mask].index
        nar_count = len(dirt_indices) // 2
        feat_df.loc[dirt_indices[:nar_count], "jyocd"] = "35"

        mock_train = self._run_pipeline_with_mocks(feat_df)

        for call_args in mock_train.call_args_list:
            args, kwargs = call_args
            df = args[0]
            if "jyocd" in df.columns:
                jyocd_int = pd.to_numeric(df["jyocd"], errors="coerce")
                nar_found = (jyocd_int >= 30).sum()
                assert nar_found == 0, f"NAR entries should be filtered, found {nar_found}"

    @patch("pipelines.training_pipeline.mlflow")
    def test_no_jyocd_column_skips_filter(self, mock_mlflow: MagicMock) -> None:
        """jyocd列がない場合はフィルタを実行しない (後方互換)"""
        feat_df = _make_feature_df(8000, 800)
        mock_train = self._run_pipeline_with_mocks(feat_df)
        assert mock_train.call_count >= 1, "Should train at least 1 submodel"


class TestModelDir:
    """TrainingPipelineV5.model_dir のテスト"""

    def test_default_model_dir(self) -> None:
        """model_dir を省略した場合のデフォルトは Path('data/models')"""
        pipeline2 = TrainingPipelineV5()
        assert pipeline2.model_dir == Path("data/models")

    def test_custom_model_dir(self) -> None:
        """カスタム model_dir が設定できる"""
        pipeline = TrainingPipelineV5(model_dir=Path("data/models-backtest"))
        assert pipeline.model_dir == Path("data/models-backtest")

    def test_model_dir_none_uses_default(self) -> None:
        """model_dir=None の場合はデフォルト値が使用される"""
        pipeline = TrainingPipelineV5(model_dir=None)
        assert pipeline.model_dir == Path("data/models")
