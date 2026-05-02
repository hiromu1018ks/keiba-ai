"""RacePredictor のテスト"""

from unittest.mock import MagicMock

import numpy as np
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
    sm.place_selection_gate = None
    sm.benter_combo = None
    sm.isotonic_calibrator = None
    sm.win_benter = None
    sm.win_isotonic_calibrator = None
    sm.win_temperature_scaler = None
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
        "edge_threshold": 0.03,
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

        # place.predict_ev は p_place_pred を付与するため、mock の返り値に含める
        result_df = race_df.copy()
        result_df["p_place_pred"] = [0.50]

        submodel = mock_models.submodels["turf"]
        submodel.market.predict_and_calc_error.return_value = race_df.copy()
        submodel.stage1.add_ability_probs.return_value = race_df.copy()
        submodel.place_ability.predict.return_value = race_df.copy()
        submodel.win.predict_ev.return_value = race_df.copy()
        submodel.ev_corrector.correct_ev.return_value = race_df.copy()
        submodel.place.predict_ev.return_value = result_df
        submodel.confidence.predict_lower_bound.return_value = (
            result_df.copy(),
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
                "EV_lower_place": [1.5, 0.8, 1.8],
                "ev_place_corrected": [1.5, 0.8, 1.8],
                "edge_place": [0.08, -0.07, 0.05],  # horses 1 & 3 pass edge_threshold=0.03
                "kakuteijyuni": [2, 1, 3],
                "kettonnum": [1234, 5678, 9012],
                "odds": [5.0, 2.0, 10.0],
                "bataijyu": [480, 470, 490],
            }
        )

        bets = predictor.select_bets(race_df, bankroll=100000.0)
        assert isinstance(bets, list)
        assert len(bets) >= 1
        assert all(b.stake == 100.0 for b in bets)

    def test_select_bets_flat_mode_uses_100_yen(self, mock_models: MagicMock) -> None:
        """flat モード (デフォルト) は100円固定ベット"""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(mock_models)
        race_df = pd.DataFrame(
            {
                "race_id": ["R1", "R1"],
                "umaban": [1, 2],
                "EV_lower_place": [1.5, 1.3],
                "ev_place_corrected": [1.5, 1.3],
                "edge_place": [0.05, 0.04],  # both pass edge_threshold=0.03
                "fukuoddslow": [3.0, 2.5],
                "surface": ["turf", "turf"],
            }
        )
        bets = predictor.select_bets(race_df, bankroll=100000)
        assert len(bets) > 0
        assert all(b.stake == 100.0 for b in bets)

    def test_select_bets_kelly_mode_uses_stake_calculator(self, mock_models: MagicMock) -> None:
        """kelly モードは StakeCalculator を使用する"""
        from backtest.race_predictor import RacePredictor
        from betting.drawdown_controller import DrawdownController
        from betting.stake_calculator import StakeCalculator

        stake_calc = MagicMock(spec=StakeCalculator)
        stake_calc.calc_stake.return_value = 200.0
        dd_ctrl = MagicMock(spec=DrawdownController)
        dd_ctrl.adjust_stake.return_value = 200.0

        predictor = RacePredictor(
            mock_models,
            stake_calculator=stake_calc,
            dd_controller=dd_ctrl,
        )
        race_df = pd.DataFrame(
            {
                "race_id": ["R1", "R1"],
                "umaban": [1, 2],
                "EV_lower_place": [1.5, 1.3],
                "ev_place_corrected": [1.5, 1.3],
                "edge_place": [0.05, 0.04],  # both pass edge_threshold=0.03
                "fukuoddslow": [3.0, 2.5],
                "surface": ["turf", "turf"],
            }
        )
        predictor.select_bets(race_df, bankroll=100000)
        assert stake_calc.calc_stake.called or dd_ctrl.adjust_stake.called

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

    def test_predict_computes_edge_place(self, mock_models: MagicMock) -> None:
        """predict() should compute edge using Benter combined probability (alpha=0.4)."""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)

        # p_place_pred=0.70, fukuoddslow=1.5 -> p_market=0.6667
        # Benter (alpha=0.4): p_combined ≈ 0.6802, edge ≈ 0.0135
        race_df = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "umaban": [1],
                "surface": ["turf"],
                "kyori": [1200],
                "distance_bin": ["sprint"],
                "popularity_rank": [3],
                "ninki": [3],
                "fukuoddslow": [1.5],
                "kakuteijyuni": [2],
                "kettonum": [1234],
                "odds": [5.0],
                "bataijyu": [480],
                "field_size": [10],
                "track_condition_code": [2],
                "grade_code": ["C"],
            }
        )

        result_df = race_df.copy()
        result_df["p_place_pred"] = [0.70]

        submodel = mock_models.submodels["turf"]
        submodel.market.predict_and_calc_error.return_value = race_df.copy()
        submodel.stage1.add_ability_probs.return_value = race_df.copy()
        submodel.place_ability.predict.return_value = race_df.copy()
        submodel.win.predict_ev.return_value = race_df.copy()
        submodel.ev_corrector.correct_ev.return_value = race_df.copy()
        submodel.place.predict_ev.return_value = result_df
        submodel.place_ev_corrector.correct_ev.return_value = result_df.copy()
        submodel.confidence.predict_lower_bound.return_value = (
            result_df.copy(),
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )

        result = predictor.predict(race_df)

        assert "edge_place" in result.columns
        assert "p_place_combined" in result.columns
        # EV-based edge: edge = p_place_pred * fukuoddslow - 1.0
        # = 0.70 * 1.5 - 1.0 = 0.05
        assert abs(result["edge_place"].iloc[0] - 0.05) < 1e-10
        assert abs(result["p_place_combined"].iloc[0] - 0.70) < 1e-10

    def test_select_bets_uses_safety_floor_without_reviving_low_ev_horses(
        self,
        mock_models: MagicMock,
    ) -> None:
        """Safety floor should not turn sub-threshold corrected EV into bets."""
        from backtest.race_predictor import RacePredictor
        from domain.types import RegimeState

        predictor = RacePredictor(models=mock_models)

        # Override regime to AGGRESSIVE with edge_threshold=0.03
        mock_models.regime_detector.current_regime = RegimeState.AGGRESSIVE
        mock_models.regime_detector.get_strategy_params.return_value = {
            "ev_threshold": 1.05,
            "edge_threshold": 0.03,
            "min_place_prob": 0.0,
            "max_place_odds": 99.0,
            "wide_enabled": False,
            "max_bets_per_race": 3,
        }

        race_df = pd.DataFrame(
            {
                "race_id": ["R1", "R1", "R1"],
                "umaban": [1, 2, 3],
                "p_place_pred": [0.60, 0.40, 0.10],
                "fukuoddslow": [1.5, 3.0, 10.0],
                "edge_place": [-0.06666666666666665, 0.06666666666666663, 0.0],
                "ev_place_corrected": [1.0, 0.9, 1.0],
                "EV_lower_place": [0.95, 1.08, 1.01],
                "surface": ["turf"] * 3,
            }
        )

        bets = predictor.select_bets(race_df, bankroll=100000.0)

        # Only horse 2 should be selected.
        assert len(bets) == 1
        assert bets[0].umaban == 2
        assert bets[0].edge == pytest.approx(0.08, abs=1e-3)

    def test_select_bets_applies_probability_and_odds_safety(self, mock_models: MagicMock) -> None:
        from backtest.race_predictor import RacePredictor
        from domain.types import RegimeState

        predictor = RacePredictor(models=mock_models)
        mock_models.regime_detector.current_regime = RegimeState.AGGRESSIVE
        mock_models.regime_detector.get_strategy_params.return_value = {
            "ev_threshold": 1.05,
            "edge_threshold": 0.03,
            "min_place_prob": 0.12,
            "max_place_odds": 18.0,
            "wide_enabled": False,
            "max_bets_per_race": 3,
        }

        race_df = pd.DataFrame(
            {
                "race_id": ["R1", "R1", "R1"],
                "umaban": [1, 2, 3],
                "p_place_corrected": [0.09, 0.13, 0.16],
                "EV_lower_place": [1.40, 1.25, 1.30],
                "fukuoddslow": [14.0, 22.0, 12.0],
                "surface": ["turf"] * 3,
            }
        )

        bets = predictor.select_bets(race_df, bankroll=100000.0)

        assert [bet.umaban for bet in bets] == [3]

    def test_select_bets_prefers_learned_gate_when_available(self, mock_models: MagicMock) -> None:
        from backtest.race_predictor import RacePredictor
        from domain.types import RegimeState

        class StubGate:
            is_trained = True

            def score(self, df: pd.DataFrame) -> pd.DataFrame:
                scored = df.copy()
                scored["place_gate_score"] = [0.2, 1.3, 0.1]
                scored["place_gate_pass"] = [False, True, False]
                return scored

        predictor = RacePredictor(models=mock_models)
        mock_models.submodels["turf"].place_selection_gate = StubGate()
        mock_models.regime_detector.current_regime = RegimeState.CONSERVATIVE
        mock_models.regime_detector.get_strategy_params.return_value = {
            "edge_threshold": 0.50,
            "min_place_prob": 0.50,
            "max_place_odds": 3.0,
            "max_bets_per_race": 1,
        }

        race_df = pd.DataFrame(
            {
                "race_id": ["R1", "R1", "R1"],
                "umaban": [1, 2, 3],
                "surface": ["turf", "turf", "turf"],
                "place_selection_prob": [0.60, 0.55, 0.15],
                "place_selection_edge": [0.55, 0.04, 0.03],
                "place_selection_ev": [1.55, 1.04, 1.03],
                "fukuoddslow": [2.0, 2.8, 12.0],
            }
        )

        bets = predictor.select_bets(race_df, bankroll=100000.0)

        assert [bet.umaban for bet in bets] == [2]

    def test_learned_gate_still_blocks_negative_edge_horses(self, mock_models: MagicMock) -> None:
        from backtest.race_predictor import RacePredictor
        from domain.types import RegimeState

        class StubGate:
            is_trained = True

            def score(self, df: pd.DataFrame) -> pd.DataFrame:
                scored = df.copy()
                scored["place_gate_score"] = [1.4]
                scored["place_gate_pass"] = [True]
                return scored

        predictor = RacePredictor(models=mock_models)
        mock_models.submodels["turf"].place_selection_gate = StubGate()
        mock_models.regime_detector.current_regime = RegimeState.CONSERVATIVE
        mock_models.regime_detector.get_strategy_params.return_value = {
            "edge_threshold": 0.05,
            "min_place_prob": 0.10,
            "max_place_odds": 18.0,
            "max_bets_per_race": 1,
        }

        race_df = pd.DataFrame(
            {
                "race_id": ["R1"],
                "umaban": [1],
                "surface": ["turf"],
                "place_selection_prob": [0.45],
                "place_selection_edge": [-0.01],
                "place_selection_ev": [0.99],
                "fukuoddslow": [3.2],
            }
        )

        bets = predictor.select_bets(race_df, bankroll=100000.0)

        assert bets == []

    def test_select_bets_softens_learned_gate_on_no_pass_races(
        self,
        mock_models: MagicMock,
    ) -> None:
        from backtest.race_predictor import RacePredictor
        from domain.types import RegimeState

        class StubGate:
            is_trained = True

            def score(self, df: pd.DataFrame) -> pd.DataFrame:
                scored = df.copy()
                scored["place_gate_score"] = [1.2, 0.9]
                scored["place_gate_pass"] = [False, False]
                return scored

            def soft_pass_mask(
                self,
                df: pd.DataFrame,
                *,
                edge_floor: float = 0.0,
                min_prob: float = 0.0,
                max_odds: float = float("inf"),
                max_per_race: int = 1,
            ) -> pd.Series:
                return pd.Series([True, False], index=df.index, dtype=bool)

        predictor = RacePredictor(models=mock_models)
        mock_models.submodels["turf"].place_selection_gate = StubGate()
        mock_models.regime_detector.current_regime = RegimeState.AGGRESSIVE
        mock_models.regime_detector.get_strategy_params.return_value = {
            "edge_threshold": 0.04,
            "min_place_prob": 0.08,
            "max_place_odds": 18.0,
            "max_bets_per_race": 1,
        }

        race_df = pd.DataFrame(
            {
                "race_id": ["R1", "R1"],
                "umaban": [1, 2],
                "surface": ["turf", "turf"],
                "place_selection_prob": [0.37, 0.30],
                "place_selection_edge": [0.03, 0.06],
                "place_selection_ev": [1.03, 1.06],
                "fukuoddslow": [8.5, 7.0],
            }
        )

        bets = predictor.select_bets(race_df, bankroll=100000.0)

        assert [bet.umaban for bet in bets] == [1]

    def test_select_bets_allows_second_runner_when_gate_margin_is_small(
        self,
        mock_models: MagicMock,
    ) -> None:
        from backtest.race_predictor import RacePredictor
        from domain.types import RegimeState

        class StubGate:
            is_trained = True

            def score(self, df: pd.DataFrame) -> pd.DataFrame:
                scored = df.copy()
                scored["place_gate_score"] = [1.20, 0.82, 0.10]
                scored["place_gate_pass"] = [True, True, False]
                return scored

        predictor = RacePredictor(models=mock_models)
        mock_models.submodels["turf"].place_selection_gate = StubGate()
        mock_models.regime_detector.current_regime = RegimeState.AGGRESSIVE
        mock_models.regime_detector.get_strategy_params.return_value = {
            "edge_threshold": 0.04,
            "min_place_prob": 0.08,
            "max_place_odds": 18.0,
            "max_bets_per_race": 1,
            "soft_gate_second_margin": 0.50,
            "soft_gate_second_min_edge": 0.03,
        }

        race_df = pd.DataFrame(
            {
                "race_id": ["R1", "R1", "R1"],
                "umaban": [1, 2, 3],
                "surface": ["turf", "turf", "turf"],
                "place_selection_prob": [0.40, 0.36, 0.12],
                "place_selection_edge": [0.05, 0.04, 0.02],
                "place_selection_ev": [1.05, 1.04, 1.02],
                "fukuoddslow": [4.0, 5.0, 10.0],
            }
        )

        bets = predictor.select_bets(race_df, bankroll=100000.0)

        assert [bet.umaban for bet in bets] == [1, 2]

    def test_select_bets_allows_quality_second_runner_in_aggressive_regime(
        self,
        mock_models: MagicMock,
    ) -> None:
        from backtest.race_predictor import RacePredictor
        from domain.types import RegimeState

        class StubGate:
            is_trained = True

            def score(self, df: pd.DataFrame) -> pd.DataFrame:
                scored = df.copy()
                scored["place_gate_score"] = [2.0, 1.15]
                scored["place_gate_pass"] = [True, False]
                return scored

        predictor = RacePredictor(models=mock_models)
        mock_models.submodels["turf"].place_selection_gate = StubGate()
        mock_models.regime_detector.current_regime = RegimeState.AGGRESSIVE
        mock_models.regime_detector.get_strategy_params.return_value = {
            "edge_threshold": 0.04,
            "min_place_prob": 0.08,
            "max_place_odds": 18.0,
            "max_bets_per_race": 1,
            "soft_gate_second_margin": 0.50,
            "soft_gate_second_min_edge": 0.03,
            "quality_second_margin": 1.00,
            "quality_second_min_edge": 0.06,
            "quality_second_min_prob": 0.25,
        }

        race_df = pd.DataFrame(
            {
                "race_id": ["R1", "R1"],
                "umaban": [1, 2],
                "surface": ["turf", "turf"],
                "place_selection_prob": [0.42, 0.30],
                "place_selection_edge": [0.09, 0.07],
                "place_selection_ev": [1.09, 1.07],
                "fukuoddslow": [3.5, 4.8],
            }
        )

        bets = predictor.select_bets(race_df, bankroll=100000.0)

        assert [bet.umaban for bet in bets] == [1, 2]

    def test_select_bets_rescues_runner_up_on_aggressive_no_bet_race(
        self,
        mock_models: MagicMock,
    ) -> None:
        from backtest.race_predictor import RacePredictor
        from domain.types import RegimeState

        class StubGate:
            is_trained = True

            def score(self, df: pd.DataFrame) -> pd.DataFrame:
                scored = df.copy()
                scored["place_gate_score"] = [1.20, 1.00, 0.40]
                scored["place_gate_pass"] = [False, False, False]
                return scored

        predictor = RacePredictor(models=mock_models)
        mock_models.submodels["turf"].place_selection_gate = StubGate()
        mock_models.regime_detector.current_regime = RegimeState.AGGRESSIVE
        mock_models.regime_detector.get_strategy_params.return_value = {
            "edge_threshold": 0.04,
            "min_place_prob": 0.08,
            "max_place_odds": 18.0,
            "max_bets_per_race": 1,
            "runner_up_rescue_margin": 0.25,
            "runner_up_rescue_min_edge": 0.04,
            "runner_up_rescue_min_prob": 0.25,
        }

        race_df = pd.DataFrame(
            {
                "race_id": ["R1", "R1", "R1"],
                "umaban": [1, 2, 3],
                "surface": ["turf", "turf", "turf"],
                "place_selection_prob": [0.18, 0.28, 0.12],
                "place_selection_edge": [0.12, 0.05, 0.01],
                "place_selection_ev": [1.12, 1.05, 1.01],
                "fukuoddslow": [7.0, 6.0, 12.0],
            }
        )

        bets = predictor.select_bets(race_df, bankroll=100000.0)

        assert [bet.umaban for bet in bets] == [2]

    def test_select_bets_rescues_rank2_with_aggressive_rerank_band(
        self,
        mock_models: MagicMock,
    ) -> None:
        from backtest.race_predictor import RacePredictor
        from domain.types import RegimeState

        class StubGate:
            is_trained = True

            def score(self, df: pd.DataFrame) -> pd.DataFrame:
                scored = df.copy()
                scored["place_gate_score"] = [2.8, 1.1, 0.2]
                scored["place_gate_pass"] = [False, False, False]
                return scored

        predictor = RacePredictor(models=mock_models)
        mock_models.submodels["turf"].place_selection_gate = StubGate()
        mock_models.regime_detector.current_regime = RegimeState.AGGRESSIVE
        mock_models.regime_detector.get_strategy_params.return_value = {
            "edge_threshold": 0.04,
            "min_place_prob": 0.08,
            "max_place_odds": 18.0,
            "max_bets_per_race": 1,
            "runner_up_rescue_margin": 0.25,
            "runner_up_rescue_min_edge": 0.04,
            "runner_up_rescue_min_prob": 0.25,
            "runner_up_rerank_market_condition_max": 0.20,
            "runner_up_rerank_entropy_min": 1.80,
            "runner_up_rerank_entropy_max": 2.30,
            "runner_up_rerank_min_edge": 0.01,
            "runner_up_rerank_min_prob": 0.10,
            "runner_up_rerank_max_odds": 12.0,
        }

        race_df = pd.DataFrame(
            {
                "race_id": ["R1", "R1", "R1"],
                "umaban": [1, 2, 3],
                "surface": ["turf", "turf", "turf"],
                "popularity_rank": [1, 4, 7],
                "odds": [6.0, 10.0, 18.0],
                "overround": [0.22, 0.22, 0.22],
                "market_entropy": [2.05, 2.05, 2.05],
                "place_selection_prob": [0.07, 0.12, 0.08],
                "place_selection_edge": [-0.02, 0.02, 0.01],
                "place_selection_ev": [0.98, 1.02, 1.01],
                "fukuoddslow": [3.2, 6.0, 11.0],
            }
        )

        bets = predictor.select_bets(race_df, bankroll=100000.0)

        assert [bet.umaban for bet in bets] == [2]

    def test_select_bets_does_not_rescue_rank2_outside_aggressive_rerank_band(
        self,
        mock_models: MagicMock,
    ) -> None:
        from backtest.race_predictor import RacePredictor
        from domain.types import RegimeState

        class StubGate:
            is_trained = True

            def score(self, df: pd.DataFrame) -> pd.DataFrame:
                scored = df.copy()
                scored["place_gate_score"] = [2.8, 1.1, 0.2]
                scored["place_gate_pass"] = [False, False, False]
                return scored

        predictor = RacePredictor(models=mock_models)
        mock_models.submodels["turf"].place_selection_gate = StubGate()
        mock_models.regime_detector.current_regime = RegimeState.AGGRESSIVE
        mock_models.regime_detector.get_strategy_params.return_value = {
            "edge_threshold": 0.04,
            "min_place_prob": 0.08,
            "max_place_odds": 18.0,
            "max_bets_per_race": 1,
            "runner_up_rescue_margin": 0.25,
            "runner_up_rescue_min_edge": 0.04,
            "runner_up_rescue_min_prob": 0.25,
            "runner_up_rerank_market_condition_max": 0.20,
            "runner_up_rerank_entropy_min": 1.80,
            "runner_up_rerank_entropy_max": 2.30,
            "runner_up_rerank_min_edge": 0.01,
            "runner_up_rerank_min_prob": 0.10,
            "runner_up_rerank_max_odds": 12.0,
        }

        race_df = pd.DataFrame(
            {
                "race_id": ["R1", "R1", "R1"],
                "umaban": [1, 2, 3],
                "surface": ["turf", "turf", "turf"],
                "popularity_rank": [1, 4, 7],
                "odds": [2.4, 10.0, 18.0],
                "overround": [0.22, 0.22, 0.22],
                "market_entropy": [2.05, 2.05, 2.05],
                "place_selection_prob": [0.07, 0.12, 0.08],
                "place_selection_edge": [-0.02, 0.02, 0.01],
                "place_selection_ev": [0.98, 1.02, 1.01],
                "fukuoddslow": [3.2, 6.0, 11.0],
            }
        )

        bets = predictor.select_bets(race_df, bankroll=100000.0)

        assert bets == []

    def test_select_bets_does_not_add_second_outside_aggressive_regime(
        self,
        mock_models: MagicMock,
    ) -> None:
        from backtest.race_predictor import RacePredictor
        from domain.types import RegimeState

        class StubGate:
            is_trained = True

            def score(self, df: pd.DataFrame) -> pd.DataFrame:
                scored = df.copy()
                scored["place_gate_score"] = [1.4, 1.2]
                scored["place_gate_pass"] = [True, False]
                return scored

            def annotate_race_context(self, df: pd.DataFrame) -> pd.DataFrame:
                annotated = df.copy()
                annotated["aggressive_tier"] = ["strong", "strong"]
                return annotated

            def runner_up_candidate_reason(
                self,
                df: pd.DataFrame,
                *,
                selected_races: pd.Series,
                max_odds: float,
            ) -> pd.Series:
                return pd.Series(["", "add_second"], index=df.index, dtype=object)

        predictor = RacePredictor(models=mock_models)
        mock_models.submodels["turf"].place_selection_gate = StubGate()
        mock_models.regime_detector.current_regime = RegimeState.CONSERVATIVE
        mock_models.regime_detector.get_strategy_params.return_value = {
            "edge_threshold": 0.04,
            "min_place_prob": 0.08,
            "max_place_odds": 18.0,
            "max_bets_per_race": 1,
        }

        race_df = pd.DataFrame(
            {
                "race_id": ["R1", "R1"],
                "umaban": [1, 2],
                "surface": ["turf", "turf"],
                "place_selection_prob": [0.40, 0.30],
                "place_selection_edge": [0.12, 0.15],
                "place_selection_ev": [1.12, 1.15],
                "fukuoddslow": [4.0, 5.0],
            }
        )

        bets = predictor.select_bets(race_df, bankroll=100000.0)

        assert [bet.umaban for bet in bets] == [1]

    def test_get_place_candidates_prunes_weak_high_prob_candidates(
        self,
        mock_models: MagicMock,
    ) -> None:
        from backtest.race_predictor import RacePredictor
        from domain.types import RegimeState

        class StubGate:
            is_trained = True

            def score(self, df: pd.DataFrame) -> pd.DataFrame:
                scored = df.copy()
                scored["place_gate_score"] = [1.5, 1.3]
                scored["place_gate_pass"] = [True, True]
                return scored

            def annotate_race_context(self, df: pd.DataFrame) -> pd.DataFrame:
                annotated = df.copy()
                annotated["aggressive_tier"] = ["weak", "strong"]
                return annotated

        predictor = RacePredictor(models=mock_models)
        mock_models.submodels["turf"].place_selection_gate = StubGate()
        mock_models.regime_detector.current_regime = RegimeState.AGGRESSIVE
        mock_models.regime_detector.get_strategy_params.return_value = {
            "edge_threshold": 0.04,
            "min_place_prob": 0.08,
            "max_place_odds": 18.0,
            "max_bets_per_race": 2,
            "weak_prob_prune_threshold": 0.35,
        }

        race_df = pd.DataFrame(
            {
                "race_id": ["R1", "R1"],
                "umaban": [1, 2],
                "surface": ["turf", "turf"],
                "place_selection_prob": [0.42, 0.30],
                "place_selection_edge": [0.09, 0.08],
                "place_selection_ev": [1.09, 1.08],
                "fukuoddslow": [4.0, 5.0],
            }
        )

        candidates = predictor.get_place_candidates(race_df)

        assert candidates["umaban"].tolist() == [2]
        assert candidates["place_prune_reason"].isna().all()

    def test_select_bets_prunes_conservative_turf_candidates(
        self,
        mock_models: MagicMock,
    ) -> None:
        from backtest.race_predictor import RacePredictor
        from domain.types import RegimeState

        predictor = RacePredictor(models=mock_models)
        mock_models.regime_detector.current_regime = RegimeState.CONSERVATIVE
        mock_models.regime_detector.get_strategy_params.return_value = {
            "edge_threshold": 0.04,
            "min_place_prob": 0.08,
            "max_place_odds": 18.0,
            "max_bets_per_race": 1,
            "prune_turf_candidates": True,
        }

        race_df = pd.DataFrame(
            {
                "race_id": ["R1"],
                "umaban": [1],
                "surface": ["turf"],
                "place_selection_prob": [0.30],
                "place_selection_edge": [0.10],
                "place_selection_ev": [1.10],
                "fukuoddslow": [4.0],
            }
        )

        bets = predictor.select_bets(race_df, bankroll=100000.0)

        assert bets == []

    def test_select_bets_prunes_add_second_outside_kept_edge_band(
        self,
        mock_models: MagicMock,
    ) -> None:
        from backtest.race_predictor import RacePredictor
        from domain.types import RegimeState

        class StubGate:
            is_trained = True

            def score(self, df: pd.DataFrame) -> pd.DataFrame:
                scored = df.copy()
                scored["place_gate_score"] = [1.6, 1.4]
                scored["place_gate_pass"] = [True, False]
                return scored

            def annotate_race_context(self, df: pd.DataFrame) -> pd.DataFrame:
                annotated = df.copy()
                annotated["aggressive_tier"] = ["strong", "strong"]
                return annotated

            def runner_up_candidate_reason(
                self,
                df: pd.DataFrame,
                *,
                selected_races: pd.Series,
                max_odds: float,
            ) -> pd.Series:
                return pd.Series(["", "add_second"], index=df.index, dtype=object)

        predictor = RacePredictor(models=mock_models)
        mock_models.submodels["turf"].place_selection_gate = StubGate()
        mock_models.regime_detector.current_regime = RegimeState.AGGRESSIVE
        mock_models.regime_detector.get_strategy_params.return_value = {
            "edge_threshold": 0.04,
            "min_place_prob": 0.08,
            "max_place_odds": 18.0,
            "max_bets_per_race": 1,
            "add_second_keep_min_edge": 0.10,
            "add_second_keep_max_edge": 0.20,
        }

        race_df = pd.DataFrame(
            {
                "race_id": ["R1", "R1"],
                "umaban": [1, 2],
                "surface": ["turf", "turf"],
                "place_selection_prob": [0.35, 0.28],
                "place_selection_edge": [0.11, 0.04],
                "place_selection_ev": [1.11, 1.04],
                "fukuoddslow": [4.0, 5.0],
            }
        )

        bets = predictor.select_bets(race_df, bankroll=100000.0)

        assert [bet.umaban for bet in bets] == [1]

    def test_place_selection_ev_keeps_corrected_ev_floor(self, mock_models: MagicMock) -> None:
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)
        race_df = pd.DataFrame(
            {
                "EV_lower_place": [0.20],
                "ev_place_corrected": [1.50],
            }
        )

        selection_ev = predictor._build_place_selection_ev(race_df)

        assert selection_ev.iloc[0] == pytest.approx(1.275)

    def test_select_bets_edge_threshold_respects_regime(self, mock_models: MagicMock) -> None:
        """Horse should NOT be selected when edge < regime edge_threshold.

        CONSERVATIVE regime with edge_threshold=0.05.
        Horse has edge=0.023 which is below threshold -> no bet.
        """
        from backtest.race_predictor import RacePredictor
        from domain.types import RegimeState

        predictor = RacePredictor(models=mock_models)

        # CONSERVATIVE regime with higher edge_threshold
        mock_models.regime_detector.current_regime = RegimeState.CONSERVATIVE
        mock_models.regime_detector.get_strategy_params.return_value = {
            "edge_threshold": 0.05,
            "max_bets_per_race": 3,
        }

        # Horse with edge=0.023 (< 0.05 threshold)
        race_df = pd.DataFrame(
            {
                "race_id": ["R1"],
                "umaban": [1],
                "p_place_pred": [0.35],
                "EV_lower_place": [1.023],
                "fukuoddslow": [3.0],  # p_market=0.333, edge=0.017 -- below 0.05
                "edge_place": [0.023],
                "ev_place_corrected": [1.02],
                "surface": ["turf"],
            }
        )

        bets = predictor.select_bets(race_df, bankroll=100000.0)

        assert len(bets) == 0

    def test_predict_ev_edge_formula(self, mock_models: MagicMock) -> None:
        """edge = p_place_pred * fukuoddslow - 1.0 (EV-based edge)."""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models, alpha=0.5)

        race_df = pd.DataFrame(
            {
                "race_id": ["R1"],
                "umaban": [1],
                "surface": ["turf"],
                "kyori": [1200],
                "distance_bin": ["sprint"],
                "popularity_rank": [3],
                "ninki": [3],
                "fukuoddslow": [1.5],
                "kakuteijyuni": [2],
                "kettonum": [1234],
                "odds": [5.0],
                "bataijyu": [480],
                "field_size": [10],
                "track_condition_code": [2],
                "grade_code": ["C"],
            }
        )

        result_df = race_df.copy()
        result_df["p_place_pred"] = [0.70]

        submodel = mock_models.submodels["turf"]
        submodel.market.predict_and_calc_error.return_value = race_df.copy()
        submodel.stage1.add_ability_probs.return_value = race_df.copy()
        submodel.place_ability.predict.return_value = race_df.copy()
        submodel.win.predict_ev.return_value = race_df.copy()
        submodel.ev_corrector.correct_ev.return_value = race_df.copy()
        submodel.place.predict_ev.return_value = result_df
        submodel.place_ev_corrector.correct_ev.return_value = result_df.copy()
        submodel.confidence.predict_lower_bound.return_value = (
            result_df.copy(),
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )

        result = predictor.predict(race_df)

        # EV-based: edge = 0.70 * 1.5 - 1.0 = 0.05
        p_combined = result["p_place_combined"].iloc[0]
        assert abs(p_combined - 0.70) < 1e-10
        edge = result["edge_place"].iloc[0]
        assert abs(edge - 0.05) < 1e-10

    def test_predict_edge_positive_when_ev_above_one(self, mock_models: MagicMock) -> None:
        """When p_place_pred * fukuoddslow > 1.0, edge should be positive."""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models, alpha=0.0)

        race_df = pd.DataFrame(
            {
                "race_id": ["R1"],
                "umaban": [1],
                "surface": ["turf"],
                "kyori": [1200],
                "distance_bin": ["sprint"],
                "popularity_rank": [3],
                "ninki": [3],
                "fukuoddslow": [2.0],
                "kakuteijyuni": [2],
                "kettonum": [1234],
                "odds": [5.0],
                "bataijyu": [480],
                "field_size": [10],
                "track_condition_code": [2],
                "grade_code": ["C"],
            }
        )

        result_df = race_df.copy()
        result_df["p_place_pred"] = [0.90]  # high model prob

        submodel = mock_models.submodels["turf"]
        submodel.market.predict_and_calc_error.return_value = race_df.copy()
        submodel.stage1.add_ability_probs.return_value = race_df.copy()
        submodel.place_ability.predict.return_value = race_df.copy()
        submodel.win.predict_ev.return_value = race_df.copy()
        submodel.ev_corrector.correct_ev.return_value = race_df.copy()
        submodel.place.predict_ev.return_value = result_df
        submodel.place_ev_corrector.correct_ev.return_value = result_df.copy()
        submodel.confidence.predict_lower_bound.return_value = (
            result_df.copy(),
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )

        result = predictor.predict(race_df)

        # EV-based: edge = 0.90 * 2.0 - 1.0 = 0.80
        p_combined = result["p_place_combined"].iloc[0]
        assert abs(p_combined - 0.90) < 1e-10
        edge = result["edge_place"].iloc[0]
        assert abs(edge - 0.80) < 1e-10

    def test_predict_edge_negative_when_ev_below_one(self, mock_models: MagicMock) -> None:
        """When p_place_pred * fukuoddslow < 1.0, edge should be negative."""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models, alpha=1.0)

        race_df = pd.DataFrame(
            {
                "race_id": ["R1"],
                "umaban": [1],
                "surface": ["turf"],
                "kyori": [1200],
                "distance_bin": ["sprint"],
                "popularity_rank": [3],
                "ninki": [3],
                "fukuoddslow": [1.5],
                "kakuteijyuni": [2],
                "kettonum": [1234],
                "odds": [5.0],
                "bataijyu": [480],
                "field_size": [10],
                "track_condition_code": [2],
                "grade_code": ["C"],
            }
        )

        result_df = race_df.copy()
        result_df["p_place_pred"] = [0.50]  # model prob too low for odds

        submodel = mock_models.submodels["turf"]
        submodel.market.predict_and_calc_error.return_value = race_df.copy()
        submodel.stage1.add_ability_probs.return_value = race_df.copy()
        submodel.place_ability.predict.return_value = race_df.copy()
        submodel.win.predict_ev.return_value = race_df.copy()
        submodel.ev_corrector.correct_ev.return_value = race_df.copy()
        submodel.place.predict_ev.return_value = result_df
        submodel.place_ev_corrector.correct_ev.return_value = result_df.copy()
        submodel.confidence.predict_lower_bound.return_value = (
            result_df.copy(),
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )

        result = predictor.predict(race_df)

        # EV-based: edge = 0.50 * 1.5 - 1.0 = -0.25 (negative)
        p_combined = result["p_place_combined"].iloc[0]
        assert abs(p_combined - 0.50) < 1e-10
        edge = result["edge_place"].iloc[0]
        expected_edge = 0.50 * 1.5 - 1.0
        assert abs(edge - expected_edge) < 1e-10

    def test_alpha_validation_rejects_out_of_range(self, mock_models: MagicMock) -> None:
        """alpha outside [0, 1] should raise ValueError."""
        from backtest.race_predictor import RacePredictor

        with pytest.raises(ValueError, match="alpha must be in"):
            RacePredictor(models=mock_models, alpha=1.5)

        with pytest.raises(ValueError, match="alpha must be in"):
            RacePredictor(models=mock_models, alpha=-0.1)


class TestRacePredictorEVEdge:
    """Tests for EV-based edge computation in RacePredictor."""

    def _make_race_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "race_id": ["R1"] * 4,
                "umaban": [1, 2, 3, 4],
                "surface": ["turf"] * 4,
                "kyori": [1200] * 4,
                "distance_bin": ["sprint"] * 4,
                "popularity_rank": [1, 2, 3, 4],
                "ninki": [1, 2, 3, 4],
                "fukuoddslow": [1.5, 2.0, 3.0, 5.0],
                "kakuteijyuni": [1, 2, 3, 4],
                "kettonum": [100, 200, 300, 400],
                "odds": [2.0, 3.0, 5.0, 10.0],
                "bataijyu": [480, 470, 490, 460],
                "field_size": [10] * 4,
                "track_condition_code": [2] * 4,
                "grade_code": ["C"] * 4,
            }
        )

    def _setup_mock_chain(self, mock_models: MagicMock, race_df: pd.DataFrame) -> None:
        """Wire up mock submodel chain to pass p_place_pred through."""
        p_place_values = [0.6, 0.5, 0.35, 0.2]
        result_df = race_df.copy()
        result_df["p_place_pred"] = p_place_values

        submodel = mock_models.submodels["turf"]
        submodel.market.predict_and_calc_error.return_value = race_df.copy()
        submodel.stage1.add_ability_probs.return_value = race_df.copy()
        submodel.place_ability.predict.return_value = race_df.copy()
        submodel.win.predict_ev.return_value = race_df.copy()
        submodel.ev_corrector.correct_ev.return_value = race_df.copy()
        submodel.place.predict_ev.return_value = result_df
        submodel.place_ev_corrector.correct_ev.return_value = result_df.copy()
        submodel.confidence.predict_lower_bound.return_value = (
            result_df.copy(),
            pd.DataFrame({"EV_lower_place": [1.5, 1.2, 1.0, 0.8]}),
        )

    def test_predict_computes_ev_edge(self, mock_models: MagicMock) -> None:
        """predict が EV ベースの edge を計算すること (benter=None フォールバック)"""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models, alpha=0.4)

        race_df = self._make_race_df()
        self._setup_mock_chain(mock_models, race_df)

        result = predictor.predict(race_df)

        assert "edge_place" in result.columns
        assert "p_market" in result.columns
        assert "p_place_combined" in result.columns
        # benter=None フォールバック: edge = p_place_pred * fukuoddslow - 1.0
        p_pred = np.array([0.6, 0.5, 0.35, 0.2])
        odds = np.array([1.5, 2.0, 3.0, 5.0])
        expected_edge = p_pred * odds - 1.0
        np.testing.assert_allclose(result["edge_place"].values, expected_edge, rtol=1e-6)
        np.testing.assert_allclose(result["p_place_combined"].values, p_pred, rtol=1e-6)
