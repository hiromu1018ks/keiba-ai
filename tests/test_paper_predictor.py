"""PaperPredictor のテスト"""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from domain.models import SubmodelSet, TrainedModelsV5
from domain.types import RegimeState


@pytest.fixture
def mock_models() -> MagicMock:
    models = MagicMock(spec=TrainedModelsV5)
    models.submodels = {"turf": MagicMock(spec=SubmodelSet)}
    models.quality_screener = MagicMock()
    models.quality_screener.should_bet.return_value = True
    models.regime_detector = MagicMock()
    models.regime_detector.current_regime = RegimeState.CONSERVATIVE
    models.regime_detector.get_strategy_params.return_value = {
        "ev_threshold": 1.20,
        "max_bets_per_race": 3,
    }
    return models


class TestPaperPredictor:
    @patch("features.trainer_context_features.TrainerContextFeatures")
    @patch("features.jockey_context_features.JockeyContextFeatures")
    @patch("features.horse_history_features.HorseHistoryFeatures")
    @patch("models.submodel_manager.SubModelManager")
    @patch("features.feature_engine.FeatureEngine")
    @patch("paper_trading.predictor.load_odds_snapshots")
    @patch("paper_trading.predictor.load_entries")
    @patch("paper_trading.predictor.load_races")
    def test_setup_returns_race_schedule(
        self,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds: MagicMock,
        mock_feat_cls: MagicMock,
        mock_submgr_cls: MagicMock,
        mock_hist_cls: MagicMock,
        mock_jockey_cls: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_models: MagicMock,
        tmp_path: Path,
    ) -> None:
        from paper_trading.predictor import PaperPredictor

        mock_load_races.return_value = pd.DataFrame(
            {
                "race_id": ["2026040510010101"],
                "race_date": pd.to_datetime("2026-04-05"),
            }
        )
        mock_load_entries.return_value = pd.DataFrame(
            {
                "race_id": ["2026040510010101"],
                "umaban": [1],
                "kettonum": [1234],
            }
        )
        mock_load_odds.return_value = pd.DataFrame()

        mock_feat = MagicMock()
        mock_feat_cls.return_value = mock_feat
        mock_feat.build_all.return_value = pd.DataFrame(
            {
                "race_id": ["2026040510010101"],
                "umaban": [1],
                "surface": ["turf"],
                "kyori": [1200],
                "kakuteijyuni": [0],
                "odds": [0.0],
                "ninki": [0],
                "bataijyu": [0],
                "kettonum": [1234],
            }
        )
        mock_submgr = MagicMock()
        mock_submgr_cls.return_value = mock_submgr
        mock_submgr.add_distance_band_features.return_value = mock_feat.build_all.return_value

        mock_hist = MagicMock()
        mock_hist_cls.return_value = mock_hist
        mock_hist.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])

        mock_jockey = MagicMock()
        mock_jockey_cls.return_value = mock_jockey
        mock_jockey.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])

        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer
        mock_trainer.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])

        mock_everydb2 = MagicMock()
        mock_everydb2.get_race_schedule.return_value = [
            {
                "race_id": "2026040510010101",
                "venue": "中山",
                "race_num": 1,
                "post_time": "10:05",
                "surface": "turf",
                "distance": 1200,
            },
        ]

        mock_store = MagicMock()
        predictor = PaperPredictor(
            store=mock_store,
            race_predictor=MagicMock(),
            models=mock_models,
            output_dir=tmp_path / "pt",
        )
        schedule = predictor.setup(date(2026, 4, 5), everydb2=mock_everydb2)

        assert schedule is not None
        assert len(schedule) == 1

    def test_predict_race_returns_bets(self, mock_models: MagicMock) -> None:
        from paper_trading.predictor import PaperPredictor

        mock_store = MagicMock()
        mock_race_predictor = MagicMock()

        pre_computed = pd.DataFrame(
            {
                "race_id": ["2026040510010101"] * 2,
                "umaban": [1, 2],
                "surface": ["turf"] * 2,
                "ev_place": [1.5, 0.8],
                "fukuoddslow": [2.4, 1.5],
            }
        )
        horse_weights = pd.DataFrame({"umaban": [1, 2], "weight": [480, 470]})
        odds = pd.DataFrame({"umaban": [1, 2], "tanodds": [5.0, 2.0], "fukuoddslow": [2.4, 1.5]})

        mock_bet = MagicMock()
        mock_bet.race_id = "2026040510010101"
        mock_bet.umaban = 1
        mock_bet.bet_type = MagicMock()
        mock_bet.bet_type.value = "place"
        mock_bet.odds = 2.4
        mock_bet.ev_lower_corrected = 1.5
        mock_bet.stake = 100.0

        mock_race_predictor.predict.return_value = pre_computed
        mock_race_predictor.should_bet.return_value = True
        mock_race_predictor.select_bets.return_value = [mock_bet]

        predictor = PaperPredictor(
            store=mock_store,
            race_predictor=mock_race_predictor,
            models=mock_models,
        )
        bets = predictor.predict_race(
            race_id="2026040510010101",
            pre_computed_features=pre_computed,
            horse_weights=horse_weights,
            odds=odds,
            bankroll=100000.0,
        )

        assert len(bets) == 1
        mock_race_predictor.predict.assert_called_once()
