"""TrainingPipeline + 関連コンポーネントのテスト"""

from __future__ import annotations

from domain.models import SubmodelSet, TrainedModelsV5


class TestTrainedModelsV5:
    """TrainedModelsV5 コンテナのテスト"""

    def test_submodel_set_holds_models(self) -> None:
        """SubmodelSet が全モデルを保持できる"""
        sub = SubmodelSet(
            market=None,
            stage1=None,
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
                    win="w_turf",
                    ev_corrector="e_turf",
                    place="p_turf",
                    wide="wd_turf",
                    confidence="c_turf",
                ),
                "dirt": SubmodelSet(
                    market="m_dirt",
                    stage1="s_dirt",
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
