"""RacePredictor のテスト"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from domain.models import TrainedModelsV5
from domain.types import RegimeState
from models.win_selection_policy import DEFAULT_DIRT_LOG_ODDS_PENALTY, DEFAULT_LOG_ODDS_PENALTY


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
    sm.conformal_ev_model = MagicMock()
    sm.place_selection_gate = None
    sm.benter_combo = None
    sm.isotonic_calibrator = None
    sm.market_aware_win_calibrator = None
    sm.win_selection_gate = None
    sm.win_selection_policy = None
    sm.win_profit_selector = None
    sm.ev_lower_threshold_turf = 1.0
    sm.ev_lower_threshold_dirt = 1.0
    sm.target_encoder = None
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
        submodel.conformal_ev_model.predict_lower_bound.return_value = (
            result_df.copy(),
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )
        submodel.conformal_ev_model.predict_interval.return_value = (
            result_df.copy(),
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )
        submodel.conformal_ev_model.predict_interval.return_value = (
            result_df.copy(),
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )

        result = predictor.predict(race_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_predict_win_target_skips_place_only_models(self, mock_models: MagicMock) -> None:
        """単勝推論では複勝専用モデルを実行しない"""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models, betting_target="win")

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
                "tanodds": [5.0],
                "kakuteijyuni": [2],
                "kettonum": [1234],
                "odds": [5.0],
                "bataijyu": [480],
                "field_size": [10],
                "track_condition_code": [2],
                "grade_code": ["C"],
            }
        )

        win_df = race_df.copy()
        win_df["p_win_pred"] = [0.2]
        win_df["e_return_win_pred"] = [5.0]
        win_df["ev_win"] = [1.0]
        win_df["ev_win_calibrated"] = [1.0]
        win_df["win_selection_ev"] = [1.0]
        win_df["win_selection_edge"] = [0.0]
        win_df["EV_lower_win_corrected"] = [1.0]

        submodel = mock_models.submodels["turf"]
        submodel.market.predict_and_calc_error.return_value = race_df.copy()
        submodel.stage1.add_ability_probs.return_value = race_df.assign(p_ability_win=0.2)
        submodel.win.predict_ev.return_value = win_df.copy()
        submodel.ev_corrector.correct_ev.return_value = win_df.copy()
        submodel.conformal_ev_model.predict_interval.return_value = (
            win_df.copy(),
            pd.DataFrame({"EV_lower_place": [0.0]}),
        )

        result = predictor.predict(race_df)

        assert len(result) == 1
        submodel.place_ability.predict.assert_not_called()
        submodel.place.predict_ev.assert_not_called()

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
        submodel.conformal_ev_model.predict_lower_bound.return_value = (
            result_df.copy(),
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )
        submodel.conformal_ev_model.predict_interval.return_value = (
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

    def test_select_bets_adds_second_in_aggressive_regime(
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
                "place_selection_prob": [0.40, 0.30],
                "place_selection_edge": [0.12, 0.15],
                "place_selection_ev": [1.12, 1.15],
                "fukuoddslow": [4.0, 5.0],
            }
        )

        bets = predictor.select_bets(race_df, bankroll=100000.0)

        # AGGRESSIVE regime with runner_up adds second bet (max_bets bumped to 2)
        assert [bet.umaban for bet in bets] == [1, 2]

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

    def test_select_bets_aggressive_does_not_prune_turf_candidates(
        self,
        mock_models: MagicMock,
    ) -> None:
        from backtest.race_predictor import RacePredictor
        from domain.types import RegimeState

        predictor = RacePredictor(models=mock_models)
        mock_models.regime_detector.current_regime = RegimeState.AGGRESSIVE
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

        # AGGRESSIVE regime does not prune turf candidates
        assert len(bets) == 1

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
        submodel.conformal_ev_model.predict_lower_bound.return_value = (
            result_df.copy(),
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )
        submodel.conformal_ev_model.predict_interval.return_value = (
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
        submodel.conformal_ev_model.predict_lower_bound.return_value = (
            result_df.copy(),
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )
        submodel.conformal_ev_model.predict_interval.return_value = (
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
        submodel.conformal_ev_model.predict_lower_bound.return_value = (
            result_df.copy(),
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )
        submodel.conformal_ev_model.predict_interval.return_value = (
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
        submodel.conformal_ev_model.predict_lower_bound.return_value = (
            result_df.copy(),
            pd.DataFrame({"EV_lower_place": [1.5, 1.2, 1.0, 0.8]}),
        )
        submodel.conformal_ev_model.predict_interval.return_value = (
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


class TestRacePredictorConfidenceIntegration:
    """ODDS-03: predict_interval pipeline integration tests"""

    def test_predict_produces_conformal_confidence_score(self, mock_models: MagicMock) -> None:
        """RacePredictor.predict() produces conformal_confidence_score when
        confidence estimator supports predict_interval."""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)

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

        # Mock predict_interval to return EV interval columns
        win_interval = result_df.copy()
        win_interval["EV_lower_win_corrected"] = [1.2]
        win_interval["EV_upper_win_corrected"] = [1.8]
        win_interval["conformal_confidence_score"] = [0.5]
        place_interval = pd.DataFrame({"EV_lower_place": [1.1], "EV_upper_place": [1.5]})
        submodel.conformal_ev_model.predict_interval.return_value = (win_interval, place_interval)

        result = predictor.predict(race_df)

        assert "conformal_confidence_score" in result.columns
        assert result["conformal_confidence_score"].iloc[0] == pytest.approx(0.5)

    def test_predict_uses_predict_interval_not_lower_bound(self, mock_models: MagicMock) -> None:
        """RacePredictor.predict() calls predict_interval (not predict_lower_bound)."""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)

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

        win_interval = result_df.copy()
        win_interval["EV_lower_win_corrected"] = [1.2]
        win_interval["EV_upper_win_corrected"] = [1.8]
        win_interval["conformal_confidence_score"] = [0.5]
        place_interval = pd.DataFrame({"EV_lower_place": [1.1], "EV_upper_place": [1.5]})
        submodel.conformal_ev_model.predict_interval.return_value = (win_interval, place_interval)

        predictor.predict(race_df)

        submodel.conformal_ev_model.predict_interval.assert_called_once()
        submodel.conformal_ev_model.predict_lower_bound.assert_not_called()


class TestGetWinCandidates:
    """get_win_candidates() のテスト (WIN-03)"""

    def _make_win_race_df(self, n: int = 5, **overrides: object) -> pd.DataFrame:
        """get_win_candidates テスト用の DataFrame を構築"""
        data: dict[str, list[object]] = {
            "race_id": ["R1"] * n,
            "umaban": list(range(1, n + 1)),
            "win_selection_edge": [0.1] * n,
            "win_selection_ev": [1.1] * n,
            "win_selection_prob": [0.3] * n,
            "tanodds": [2.4] * n,
            "win_gate_score": [0.8] * n,
            "win_gate_pass": [True] * n,
            "conformal_confidence_score": [0.5] * n,
        }
        for key, val in overrides.items():
            if isinstance(val, list):
                data[key] = val
        return pd.DataFrame(data)

    def test_basic_filter_returns_one_candidate(self, mock_models: MagicMock) -> None:
        """Test 1: tanodds>=1.0 のレースから単勝候補を1頭返す"""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)
        race_df = self._make_win_race_df(n=3)
        result = predictor.get_win_candidates(race_df)
        assert len(result) == 1
        assert result.iloc[0]["tanodds"] >= 1.0

    def test_negative_edge_is_diagnostic_not_exclusion(self, mock_models: MagicMock) -> None:
        """Test 2: win_selection_edge<0 は除外せず、リスクとして診断する"""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)
        race_df = self._make_win_race_df(n=2, win_selection_edge=[-0.1, -0.2])
        result = predictor.get_win_candidates(race_df)
        diag_df = result.attrs["win_diagnostic_df"]

        assert len(result) == 1
        assert set(diag_df["risk_flags"].dropna().unique()) == {"negative_or_zero_edge"}

    def test_low_odds_returns_empty(self, mock_models: MagicMock) -> None:
        """Test 3: tanodds < 1.0 の馬は除外される"""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)
        race_df = self._make_win_race_df(n=2, tanodds=[0.5, 0.8])
        result = predictor.get_win_candidates(race_df)
        assert len(result) == 0

    def test_high_win_odds_tail_flagged_not_excluded(self, mock_models: MagicMock) -> None:
        """tanodds >= 30 は除外せず、リスクとして診断する"""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)
        race_df = self._make_win_race_df(n=2, tanodds=[29.9, 30.0])
        result = predictor.get_win_candidates(race_df)
        diag_df = result.attrs["win_diagnostic_df"]

        assert len(result) == 1
        row = diag_df.loc[diag_df["umaban"].eq(2)].iloc[0]
        assert pd.isna(row["excluded_reason"])
        assert "high_odds_tail" in row["risk_flags"]

    def test_longshot_low_probability_flagged_not_excluded(self, mock_models: MagicMock) -> None:
        """10倍以上かつ p_win_final < 0.05 は除外せず、リスクとして診断する"""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)
        race_df = self._make_win_race_df(
            n=2,
            tanodds=[9.9, 10.0],
            p_win_final=[0.04, 0.04],
        )
        result = predictor.get_win_candidates(race_df)
        diag_df = result.attrs["win_diagnostic_df"]

        assert len(result) == 1
        row = diag_df.loc[diag_df["umaban"].eq(2)].iloc[0]
        assert pd.isna(row["excluded_reason"])
        assert "longshot_low_probability" in row["risk_flags"]

    def test_high_win_ev_tail_flagged_not_excluded(self, mock_models: MagicMock) -> None:
        """win_selection_ev >= 5.0 は除外せず、リスクとして診断する"""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)
        race_df = self._make_win_race_df(
            n=2,
            win_selection_ev=[1.4, 5.0],
            win_selection_edge=[0.4, 4.0],
        )
        result = predictor.get_win_candidates(race_df)
        diag_df = result.attrs["win_diagnostic_df"]

        assert len(result) == 1
        row = diag_df.loc[diag_df["umaban"].eq(2)].iloc[0]
        assert pd.isna(row["excluded_reason"])
        assert "high_ev_tail" in row["risk_flags"]

    def test_overconfident_calibrated_ev_tail_flagged_not_excluded(
        self, mock_models: MagicMock
    ) -> None:
        """校正後 win_selection_ev >= 1.5 は除外せず、リスクとして診断する"""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)
        race_df = self._make_win_race_df(
            n=2,
            win_selection_ev=[1.4, 3.0],
            win_selection_edge=[0.4, 2.0],
            p_win_final=[0.50, 0.50],
        )
        result = predictor.get_win_candidates(race_df)
        diag_df = result.attrs["win_diagnostic_df"]

        assert len(result) == 1
        row = diag_df.loc[diag_df["umaban"].eq(2)].iloc[0]
        assert pd.isna(row["excluded_reason"])
        assert "overconfident_ev_tail" in row["risk_flags"]

    def test_low_win_probability_rank_flagged_not_excluded(self, mock_models: MagicMock) -> None:
        """p_win_final のレース内順位が9位以下でも除外せず、診断に残す"""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)
        race_df = self._make_win_race_df(
            n=10,
            p_win_final=[0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03],
        )
        result = predictor.get_win_candidates(race_df)
        diag_df = result.attrs["win_diagnostic_df"]

        assert len(result) == 1
        assert diag_df.loc[diag_df["umaban"].eq(9), "selected_rank_by_p_win_final"].iloc[0] == 9
        flags = diag_df.loc[diag_df["umaban"].eq(9), "risk_flags"].iloc[0]
        assert "low_probability_rank" in flags

    def test_low_win_probability_floor_flagged_not_excluded(self, mock_models: MagicMock) -> None:
        """p_win_final < 0.03 でも除外せず、診断に残す"""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)
        race_df = self._make_win_race_df(n=1, p_win_final=[0.029])
        result = predictor.get_win_candidates(race_df)
        diag_df = result.attrs["win_diagnostic_df"]

        assert len(result) == 1
        flags = diag_df.loc[diag_df["umaban"].eq(1), "risk_flags"].iloc[0]
        assert "low_probability" in flags

    def test_tail_calibrated_ev_is_used_for_valid_candidates(self, mock_models: MagicMock) -> None:
        """EV tail calibration 後の値から win_selection_edge が再計算される"""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)
        race_df = self._make_win_race_df(
            n=1,
            win_selection_ev=[2.0],
            win_selection_edge=[1.0],
            p_win_final=[0.50],
        )
        result = predictor.get_win_candidates(race_df)

        assert len(result) == 1
        assert result.iloc[0]["win_selection_ev_tail_calibrated"] == pytest.approx(1.4)
        assert result.iloc[0]["win_selection_edge"] == pytest.approx(0.4)

    def test_max_one_candidate(self, mock_models: MagicMock) -> None:
        """Test 4: 5候補いても単勝は上位1頭のみ返す"""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)
        race_df = self._make_win_race_df(
            n=5,
            win_gate_score=[0.9, 0.7, 0.5, 0.3, 0.1],
        )
        result = predictor.get_win_candidates(race_df)
        assert len(result) == 1

    def test_missing_edge_column_returns_empty(self, mock_models: MagicMock) -> None:
        """Test 5: win_selection_edge 列なし → 空 DataFrame"""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)
        race_df = pd.DataFrame({"race_id": ["R1"], "umaban": [1], "tanodds": [2.4]})
        result = predictor.get_win_candidates(race_df)
        assert len(result) == 0

    def test_nan_gate_score_fallback_to_edge(self, mock_models: MagicMock) -> None:
        """Test 6: dirt は市場残差ではなく win_selection_edge を主軸にソートされる"""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)
        race_df = self._make_win_race_df(
            n=3,
            surface=["dirt", "dirt", "dirt"],
            win_gate_score=[float("nan"), float("nan"), float("nan")],
            win_selection_edge=[0.05, 0.20, 0.10],
        )
        result = predictor.get_win_candidates(race_df)
        assert len(result) == 1
        assert result.iloc[0]["win_selection_edge"] == pytest.approx(0.20)

    def test_dirt_win_ranking_uses_edge_not_market_residual(self, mock_models: MagicMock) -> None:
        """dirt は人気寄りの市場残差でなく、修正前に強かったedge主軸で順位付けする。"""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)
        race_df = self._make_win_race_df(
            n=2,
            surface=["dirt", "dirt"],
            tanodds=[2.0, 12.0],
            p_win_final=[0.60, 0.08],
            win_selection_prob=[0.60, 0.08],
            win_selection_edge=[0.05, 0.30],
            win_selection_ev=[1.05, 1.30],
            win_gate_score=[2.0, 0.2],
            win_market_logit_edge=[2.0, 0.1],
            win_market_value_ratio=[2.0, 0.1],
        )

        result = predictor.get_win_candidates(race_df)

        assert len(result) == 1
        assert result.iloc[0]["umaban"] == 2
        assert result.iloc[0]["win_log_odds_penalty"] == pytest.approx(
            DEFAULT_DIRT_LOG_ODDS_PENALTY
        )

    def test_final_win_ranking_uses_probability_market_residual_first(
        self, mock_models: MagicMock
    ) -> None:
        """単勝最終選択はEV過信を避け、市場対比の勝率残差を主軸にする。"""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)
        race_df = self._make_win_race_df(
            n=3,
            win_selection_edge=[0.05, 0.30, 0.10],
            win_selection_ev=[1.05, 1.30, 1.10],
            p_win_final=[0.60, 0.08, 0.20],
            win_gate_score=[2.0, 0.2, 1.0],
            win_market_logit_edge=[2.0, 0.1, 1.0],
            win_market_value_ratio=[2.0, 0.1, 1.0],
        )

        result = predictor.get_win_candidates(race_df)

        assert len(result) == 1
        assert result.iloc[0]["umaban"] == 1
        assert result.iloc[0]["win_market_selection_score"] < result.iloc[0]["p_win_final"]

    def test_late_odds_drop_penalty_can_break_close_edge_ties(self, mock_models: MagicMock) -> None:
        """直前でオッズ低下が強い馬は、近いedge差なら過熱として減点する。"""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)
        race_df = self._make_win_race_df(
            n=2,
            win_selection_edge=[0.20, 0.13],
            win_selection_ev=[1.20, 1.13],
            p_win_final=[0.20, 0.20],
            odds_drop_rate_30_10=[1.0, -1.0],
        )

        result = predictor.get_win_candidates(race_df)

        assert len(result) == 1
        assert result.iloc[0]["umaban"] == 2
        assert "win_late_odds_drop_z" in result.columns

    def test_log_odds_penalty_can_break_close_edge_ties(self, mock_models: MagicMock) -> None:
        """極端な高オッズ候補は、近いedge差なら滑らかに減点する。"""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)
        race_df = self._make_win_race_df(
            n=2,
            tanodds=[50.0, 5.0],
            win_selection_edge=[0.20, 0.15],
            win_selection_ev=[1.20, 1.15],
            p_win_final=[0.10, 0.91],
        )

        result = predictor.get_win_candidates(race_df)

        assert len(result) == 1
        assert result.iloc[0]["umaban"] == 2
        assert result.iloc[0]["win_log_odds_penalty"] == pytest.approx(DEFAULT_LOG_ODDS_PENALTY)

    def test_learned_odds_prior_does_not_dominate_win_candidates(
        self, mock_models: MagicMock
    ) -> None:
        """OOF学習済みオッズpriorは診断用で、最終選択を単独支配しない."""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)
        race_df = self._make_win_race_df(
            n=3,
            tanodds=[2.0, 12.0, 8.0],
            win_selection_prob=[0.60, 0.04, 0.08],
            p_win_final=[0.60, 0.04, 0.08],
            win_selection_edge=[0.20, 0.20, 0.05],
            win_selection_ev=[1.20, 1.20, 1.05],
            win_gate_score=[2.0, 1.0, 0.5],
            win_gate_odds_score=[0.5, 2.0, 1.0],
            win_gate_edge_odds_score=[0.5, 2.0, 1.0],
        )

        result = predictor.get_win_candidates(race_df)

        assert result.iloc[0]["umaban"] == 1

    def test_win_segment_factors_are_neutral_by_default(
        self, mock_models: MagicMock
    ) -> None:
        """WinSegmentCalibrator removed -- segment factors are always neutral (1.0)."""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)
        race_df = self._make_win_race_df(
            n=2,
            surface=["turf", "turf"],
            tanodds=[3.0, 3.0],
            p_win_final=[0.55, 0.50],
            win_selection_prob=[0.55, 0.50],
            win_selection_ev=[1.20, 1.20],
            win_selection_edge=[0.20, 0.20],
        )

        result = predictor.get_win_candidates(race_df)
        diag_df = result.attrs["win_diagnostic_df"]

        # segment factors should be neutral (1.0) since WinSegmentCalibrator is removed
        assert len(result) == 1
        for idx in diag_df.index:
            assert diag_df.loc[idx, "win_segment_prob_factor"] == pytest.approx(1.0)
            assert diag_df.loc[idx, "win_segment_ev_factor"] == pytest.approx(1.0)

    def test_missing_tanodds_returns_empty(self, mock_models: MagicMock) -> None:
        """tanodds 列なし → 空 DataFrame"""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)
        race_df = pd.DataFrame({"race_id": ["R1"], "umaban": [1], "win_selection_edge": [0.1]})
        result = predictor.get_win_candidates(race_df)
        assert len(result) == 0

    def test_ev_lower_not_used_for_filtering(self, mock_models: MagicMock) -> None:
        """EV_lower_win_corrected はベット除外に使われない (CQR過学習防止)"""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)
        race_df = self._make_win_race_df(
            n=3,
            EV_lower_win_corrected=[0.8, 1.2, 0.5],  # 値に関わらず有効オッズなら候補
        )
        result = predictor.get_win_candidates(race_df)
        assert len(result) == 1

    def test_ev_lower_nan_no_effect(self, mock_models: MagicMock) -> None:
        """EV_lower_win_corrected が NaN でも有効オッズで判定"""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)
        race_df = self._make_win_race_df(
            n=2,
            EV_lower_win_corrected=[float("nan"), float("nan")],
        )
        result = predictor.get_win_candidates(race_df)
        assert len(result) == 1

    def test_ev_lower_column_missing_keeps_existing_behavior(self, mock_models: MagicMock) -> None:
        """EV_lower_win_corrected 列なし → 有効オッズだけで判定"""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)
        race_df = self._make_win_race_df(n=3)
        race_df = race_df.drop(columns=["EV_lower_win_corrected"], errors="ignore")
        result = predictor.get_win_candidates(race_df)
        assert len(result) == 1

    def test_ev_lower_high_values_no_effect(self, mock_models: MagicMock) -> None:
        """EV_lower_win_corrected が高くても有効オッズだけで判定"""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)
        race_df = self._make_win_race_df(
            n=3,
            EV_lower_win_corrected=[1.5, 1.2, 2.0],
        )
        result = predictor.get_win_candidates(race_df)
        assert len(result) == 1

    def test_ev_lower_dynamic_threshold_turf_no_filter(self, mock_models: MagicMock) -> None:
        """EV_lower閾値が設定されていても、ベット判定には使われない"""
        from backtest.race_predictor import RacePredictor

        mock_models.submodels["turf"].ev_lower_threshold_turf = 0.85
        predictor = RacePredictor(models=mock_models)
        race_df = self._make_win_race_df(
            n=3,
            surface=["turf"] * 3,
            EV_lower_win_corrected=[0.90, 0.80, 1.5],
        )
        result = predictor.get_win_candidates(race_df)
        assert len(result) == 1

    def test_ev_lower_dynamic_threshold_dirt_no_filter(self, mock_models: MagicMock) -> None:
        """dirt EV_lower閾値が設定されていても、ベット判定には使われない"""
        from backtest.race_predictor import RacePredictor

        sm = _make_submodel_mock()
        sm.ev_lower_threshold_dirt = 0.70
        mock_models.submodels["dirt"] = sm
        predictor = RacePredictor(models=mock_models)
        race_df = self._make_win_race_df(
            n=3,
            surface=["dirt"] * 3,
            EV_lower_win_corrected=[0.75, 0.65, 1.2],
        )
        result = predictor.get_win_candidates(race_df)
        assert len(result) == 1

    def test_ev_lower_nan_ignored(self, mock_models: MagicMock) -> None:
        """EV_lower NaN でも有効オッズで判定"""
        from backtest.race_predictor import RacePredictor

        mock_models.submodels["turf"].ev_lower_threshold_turf = 0.80
        predictor = RacePredictor(models=mock_models)
        race_df = self._make_win_race_df(
            n=2,
            surface=["turf"] * 2,
            EV_lower_win_corrected=[float("nan"), float("nan")],
        )
        result = predictor.get_win_candidates(race_df)
        assert len(result) == 1


class TestMarketAwareWinCalibratorIntegration:
    """MarketAwareWinCalibrator integration tests in RacePredictor (CAL-04)."""

    def test_predict_calls_calibrator_apply_when_available(
        self, mock_models: MagicMock
    ) -> None:
        """predict() calls market_aware_win_calibrator.apply() when calibrator is available."""
        from backtest.race_predictor import RacePredictor

        calibrator_mock = MagicMock()
        calibrator_mock.is_trained = True
        calibrator_mock.apply.side_effect = lambda df: df.assign(
            p_win_combined=df.get("p_win_corrected", pd.Series([0.2])),
            p_win_final=pd.Series([0.2]),
            edge_win=pd.Series([0.0]),
        )
        mock_models.submodels["turf"].market_aware_win_calibrator = calibrator_mock

        predictor = RacePredictor(models=mock_models, betting_target="win")

        race_df = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "umaban": [1],
                "surface": ["turf"],
                "kyori": [1200],
                "distance_bin": ["sprint"],
                "popularity_rank": [3],
                "ninki": [3],
                "tanodds": [5.0],
                "kakuteijyuni": [2],
                "kettonum": [1234],
                "odds": [5.0],
                "bataijyu": [480],
                "field_size": [10],
                "track_condition_code": [2],
                "grade_code": ["C"],
            }
        )

        win_df = race_df.copy()
        win_df["p_win_pred"] = [0.2]
        win_df["e_return_win_pred"] = [5.0]
        win_df["ev_win"] = [1.0]
        win_df["p_win_corrected"] = [0.2]
        win_df["ev_win_calibrated"] = [1.0]
        win_df["win_selection_ev"] = [1.0]
        win_df["win_selection_edge"] = [0.0]
        win_df["EV_lower_win_corrected"] = [1.0]

        submodel = mock_models.submodels["turf"]
        submodel.market.predict_and_calc_error.return_value = race_df.copy()
        submodel.stage1.add_ability_probs.return_value = race_df.assign(p_ability_win=0.2)
        submodel.win.predict_ev.return_value = win_df.copy()
        submodel.ev_corrector.correct_ev.return_value = win_df.copy()
        submodel.conformal_ev_model.predict_interval.return_value = (
            win_df.copy(),
            pd.DataFrame({"EV_lower_place": [0.0]}),
        )

        result = predictor.predict(race_df)

        calibrator_mock.apply.assert_called_once()
        assert len(result) == 1

    def test_predict_fallback_when_calibrator_is_none(
        self, mock_models: MagicMock
    ) -> None:
        """predict() works when calibrator is None (fallback behavior)."""
        from backtest.race_predictor import RacePredictor

        mock_models.submodels["turf"].market_aware_win_calibrator = None
        predictor = RacePredictor(models=mock_models, betting_target="win")

        race_df = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "umaban": [1],
                "surface": ["turf"],
                "kyori": [1200],
                "distance_bin": ["sprint"],
                "popularity_rank": [3],
                "ninki": [3],
                "tanodds": [5.0],
                "kakuteijyuni": [2],
                "kettonum": [1234],
                "odds": [5.0],
                "bataijyu": [480],
                "field_size": [10],
                "track_condition_code": [2],
                "grade_code": ["C"],
            }
        )

        win_df = race_df.copy()
        win_df["p_win_pred"] = [0.2]
        win_df["e_return_win_pred"] = [5.0]
        win_df["ev_win"] = [1.0]
        win_df["p_win_corrected"] = [0.2]
        win_df["ev_win_calibrated"] = [1.0]
        win_df["win_selection_ev"] = [1.0]
        win_df["win_selection_edge"] = [0.0]
        win_df["EV_lower_win_corrected"] = [1.0]

        submodel = mock_models.submodels["turf"]
        submodel.market.predict_and_calc_error.return_value = race_df.copy()
        submodel.stage1.add_ability_probs.return_value = race_df.assign(p_ability_win=0.2)
        submodel.win.predict_ev.return_value = win_df.copy()
        # ev_corrector.correct_ev must return df with p_win_corrected for fallback
        ev_df = win_df.copy()
        ev_df["p_win_corrected"] = [0.2]
        submodel.ev_corrector.correct_ev.return_value = ev_df
        # predict_interval receives df after fallback adds p_win_final/edge_win;
        # mock must preserve those columns (production ConformalEVModel does too)
        interval_df = ev_df.copy()
        interval_df["p_win_final"] = [0.2]
        interval_df["edge_win"] = [0.0]
        submodel.conformal_ev_model.predict_interval.return_value = (
            interval_df,
            pd.DataFrame({"EV_lower_place": [0.0]}),
        )

        result = predictor.predict(race_df)

        # predict_interval replaces df with win_df, so columns from fallback
        # must survive through the conformal interval step. In production the
        # conformal model preserves all columns; the mock must do the same.
        assert "p_win_final" in result.columns
        assert "edge_win" in result.columns
        assert len(result) == 1


class TestSelectBetsWinPath:
    """select_bets() の win path テスト (WIN-03)"""

    def test_win_path_produces_win_bets(self, mock_models: MagicMock) -> None:
        """Test 7: betting_target='win' で BetType.WIN の Bet を生成"""
        from backtest.race_predictor import RacePredictor
        from domain.models import BetType

        predictor = RacePredictor(models=mock_models)
        race_df = pd.DataFrame(
            {
                "race_id": ["R1", "R1"],
                "umaban": [1, 2],
                "win_selection_edge": [0.10, 0.05],
                "win_selection_ev": [1.10, 1.05],
                "win_selection_prob": [0.30, 0.25],
                "tanodds": [3.0, 5.0],
                "win_gate_score": [0.8, 0.5],
                "surface": ["turf", "turf"],
            }
        )
        bets = predictor.select_bets(race_df, bankroll=100000.0, betting_target="win")
        assert len(bets) >= 1
        assert all(b.bet_type == BetType.WIN for b in bets)
        assert all(b.odds > 0 for b in bets)

    def test_win_path_kelly_uses_stake_calc(self, mock_models: MagicMock) -> None:
        """Test 8: win + kelly モードで calc_stake が呼ばれる"""
        from backtest.race_predictor import RacePredictor
        from betting.drawdown_controller import DrawdownController
        from betting.stake_calculator import StakeCalculator
        from domain.models import BetType

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
                "win_selection_edge": [0.10, 0.05],
                "win_selection_ev": [1.10, 1.05],
                "win_selection_prob": [0.30, 0.25],
                "tanodds": [3.0, 5.0],
                "win_gate_score": [0.8, 0.5],
                "surface": ["turf", "turf"],
            }
        )
        bets = predictor.select_bets(race_df, bankroll=100000.0, betting_target="win")
        assert len(bets) >= 1
        assert all(b.bet_type == BetType.WIN for b in bets)
        assert stake_calc.calc_stake.called

    def test_win_path_respects_max_bets(self, mock_models: MagicMock) -> None:
        """win path は max_bets_per_race を超えない"""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)
        mock_models.regime_detector.get_strategy_params.return_value = {
            "ev_threshold": 1.20,
            "edge_threshold": 0.03,
            "max_bets_per_race": 1,
        }
        race_df = pd.DataFrame(
            {
                "race_id": ["R1", "R1", "R1"],
                "umaban": [1, 2, 3],
                "win_selection_edge": [0.10, 0.05, 0.03],
                "win_selection_ev": [1.10, 1.05, 1.03],
                "win_selection_prob": [0.30, 0.25, 0.20],
                "tanodds": [3.0, 5.0, 8.0],
                "win_gate_score": [0.9, 0.7, 0.5],
                "surface": ["turf", "turf", "turf"],
            }
        )
        bets = predictor.select_bets(race_df, bankroll=100000.0, betting_target="win")
        assert len(bets) <= 1

    def test_win_path_no_candidates_returns_empty(self, mock_models: MagicMock) -> None:
        """有効な単勝オッズがない → 空 list"""
        from backtest.race_predictor import RacePredictor

        predictor = RacePredictor(models=mock_models)
        race_df = pd.DataFrame(
            {
                "race_id": ["R1"],
                "umaban": [1],
                "win_selection_edge": [-0.1],
                "tanodds": [0.0],
                "surface": ["turf"],
            }
        )
        bets = predictor.select_bets(race_df, bankroll=100000.0, betting_target="win")
        assert bets == []
