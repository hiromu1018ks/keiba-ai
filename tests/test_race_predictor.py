"""RacePredictor のテスト"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from domain.models import TrainedModelsV5
from domain.types import RegimeState


def _make_submodel_mock() -> MagicMock:
    """SubmodelSet の各フィールドを MagicMock で構成するヘルパー"""
    sm = MagicMock()
    sm.market = MagicMock()
    sm.stage1 = MagicMock()
    sm.place_ability = MagicMock()
    sm.win = MagicMock()
    sm.ev_corrector = MagicMock()
    sm.place = MagicMock()
    sm.wide = MagicMock()
    sm.confidence = MagicMock()
    return sm


@pytest.fixture
def mock_models() -> MagicMock:
    models = MagicMock(spec=TrainedModelsV5)
    models.submodels = {"turf": _make_submodel_mock()}
    models.quality_screener = MagicMock()
    models.quality_screener.should_bet.return_value = True
    models.regime_detector = MagicMock()
    models.regime_detector.current_regime = RegimeState.CONSERVATIVE
    models.regime_detector.get_strategy_params.return_value = {
        "ev_threshold": 1.20,
        "max_bets_per_race": 3,
    }
    return models


class TestRacePredictor:
    def test_predict_returns_dataframe_with_ev_columns(self, mock_models: MagicMock) -> None:
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)

        race_df = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "umaban": [1],
                "surface": ["turf"],
                "kyori": [1200],
                "distance_bin": ["sprint"],
                "popularity_rank": [3],
                "ninki": [3],
                "fukuoddslow": [2.4],
                "kakuteijyuni": [2],
                "kettonum": [1234],
                "odds": [5.0],
                "bataijyu": [480],
                "field_size": [10],
                "track_condition_code": [2],
                "grade_code": ["C"],
            }
        )

        submodel = mock_models.submodels["turf"]
        submodel.market.predict_and_calc_error.return_value = race_df.copy()
        submodel.stage1.add_ability_probs.return_value = race_df.copy()
        submodel.place_ability.predict.return_value = race_df.copy()
        submodel.win.predict_ev.return_value = race_df.copy()
        submodel.ev_corrector.correct_ev.return_value = race_df.copy()
        submodel.place.predict_ev.return_value = race_df.copy()
        submodel.confidence.predict_lower_bound.return_value = (
            race_df.copy(),
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )

        result = predictor.predict(race_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_predict_skips_unknown_surface(self, mock_models: MagicMock) -> None:
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)

        race_df = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "umaban": [1],
                "surface": ["unknown"],
                "kyori": [1200],
                "distance_bin": ["sprint"],
                "popularity_rank": [3],
                "ninki": [3],
                "fukuoddslow": [2.4],
                "kakuteijyuni": [2],
                "kettonum": [1234],
                "odds": [5.0],
                "bataijyu": [480],
            }
        )

        result = predictor.predict(race_df)
        assert result.empty

    def test_select_bets_returns_list(self, mock_models: MagicMock) -> None:
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)

        race_df = pd.DataFrame(
            {
                "race_id": ["20240101010101"] * 3,
                "umaban": [1, 2, 3],
                "surface": ["turf"] * 3,
                "kyori": [1200] * 3,
                "distance_bin": ["sprint"] * 3,
                "popularity_rank": [3, 5, 7],
                "ninki": [3, 5, 7],
                "fukuoddslow": [2.4, 1.5, 5.0],
                "ev_place": [1.5, 0.8, 1.8],
                "kakuteijyuni": [2, 1, 3],
                "kettonum": [1234, 5678, 9012],
                "odds": [5.0, 2.0, 10.0],
                "bataijyu": [480, 470, 490],
            }
        )

        bets = predictor.select_bets(race_df, bankroll=100000.0)
        assert isinstance(bets, list)
        assert len(bets) >= 1
        assert all(b.stake == 100.0 for b in bets)

    def test_build_race_features(self, mock_models: MagicMock) -> None:
        from backtest.race_predictor import RacePredictor

        race_df = pd.DataFrame(
            {
                "race_id": ["20240101010101"] * 2,
                "umaban": [1, 2],
                "surface": ["turf"] * 2,
                "distance_bin": ["sprint"] * 2,
                "track_condition_code": [2] * 2,
                "grade_code": ["C"] * 2,
                "field_size": [10] * 2,
                "difficulty_score": [0.5] * 2,
                "signed_log_error_win": [0.1, -0.2],
                "abs_log_error_win": [0.1, 0.2],
                "market_entropy": [2.0] * 2,
                "overround": [0.2] * 2,
            }
        )

        features = RacePredictor.build_race_features(race_df)
        assert isinstance(features, dict)
        assert features["surface"] == "turf"
        assert features["field_size"] == 10
