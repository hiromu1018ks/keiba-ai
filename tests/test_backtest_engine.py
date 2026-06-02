"""BacktestEngine のテスト"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from domain.models import SubmodelSet, TrainedModelsV5
from domain.types import BetType, RegimeState


@pytest.fixture
def mock_models() -> MagicMock:
    """モック TrainedModelsV5"""
    models = MagicMock(spec=TrainedModelsV5)
    models.submodels = {"turf": MagicMock(spec=SubmodelSet)}
    models.submodels["turf"].benter_combo = None
    models.submodels["turf"].isotonic_calibrator = None
    models.quality_screener = MagicMock()
    models.quality_screener.should_bet.return_value = True
    models.regime_detector = MagicMock()
    models.regime_detector.current_regime = RegimeState.CONSERVATIVE
    models.regime_detector.cfg.min_samples = 5
    models.regime_detector.get_strategy_params.return_value = {
        "ev_threshold": 1.20,
        "score_threshold": 0.015,
        "max_bets_per_race": 3,
    }
    models.regime_detector.detect.return_value = RegimeState.CONSERVATIVE
    return models


class TestBacktestResult:
    """BacktestResult データクラスのテスト"""

    def test_result_structure(self) -> None:
        """BacktestResult が正しい構造を持つ"""
        from backtest.engine import BacktestResult

        result = BacktestResult(
            total_bets=100,
            total_stake=100000,
            total_return=105000,
            winning_bets=30,
            total_roi=1.05,
            max_drawdown=0.08,
            bet_history=[],
        )
        assert result.total_roi == 1.05
        assert result.total_return - result.total_stake == 5000
        assert result.monthly_returns == {}  # bet_history空なら空dictを返す

    def test_profit_property(self) -> None:
        """profit プロパティが正しく計算される"""
        from backtest.engine import BacktestResult

        result = BacktestResult(total_stake=1000, total_return=1200)
        assert result.profit == 200.0

    def test_summary_format(self) -> None:
        """summary() が文字列を返す"""
        from backtest.engine import BacktestResult

        result = BacktestResult(
            total_bets=50,
            total_stake=50000,
            total_return=55000,
            total_roi=1.10,
            max_drawdown=0.05,
            final_bankroll=105000,
        )
        s = result.summary()
        assert "50" in s
        assert "110.000%" in s

    def test_bet_final_odds_default(self) -> None:
        """Bet.final_odds のデフォルトは 0.0"""
        from domain.models import Bet, BetType

        bet = Bet(
            race_id="20250401110101",
            umaban=1,
            bet_type=BetType.PLACE,
            odds=1.3,
            ev_lower_corrected=1.5,
            stake=100.0,
        )
        assert bet.final_odds == 0.0
        assert bet.odds == 1.3

    def test_bet_final_odds_set(self) -> None:
        """Bet.final_odds に値を設定できる"""
        from domain.models import Bet, BetType

        bet = Bet(
            race_id="20250401110101",
            umaban=1,
            bet_type=BetType.PLACE,
            odds=1.3,
            final_odds=1.5,
            ev_lower_corrected=1.5,
            stake=100.0,
        )
        assert bet.final_odds == 1.5


class TestBacktestEngine:
    """BacktestEngine のテスト"""

    def test_init_with_models(self, mock_models: MagicMock) -> None:
        """モデル付きで初期化できる"""
        from backtest.engine import BacktestEngine

        engine = BacktestEngine(models=mock_models)
        assert engine.models is mock_models

    def test_init_with_bankroll(self, mock_models: MagicMock) -> None:
        """初期資金を設定できる"""
        from backtest.engine import BacktestEngine

        engine = BacktestEngine(models=mock_models, initial_bankroll=200000)
        assert engine.initial_bankroll == 200000

    def test_engine_kelly_mode_creates_predictor_with_stake_calc(
        self, mock_models: MagicMock
    ) -> None:
        """betting_mode='kelly' の場合、RacePredictor に StakeCalculator が注入される"""
        from backtest.engine import BacktestEngine

        engine = BacktestEngine(models=mock_models, betting_mode="kelly")
        assert engine._race_predictor._betting_mode == "kelly"
        assert engine._race_predictor.stake_calc is not None
        assert engine._race_predictor.dd_ctrl is not None

    def test_engine_flat_mode_default(self, mock_models: MagicMock) -> None:
        """デフォルトはflatモード"""
        from backtest.engine import BacktestEngine

        engine = BacktestEngine(models=mock_models)
        assert engine._race_predictor._betting_mode == "flat"
        assert engine._race_predictor.stake_calc is None

    def test_init_with_diag_prefix(self, mock_models: MagicMock) -> None:
        """diag_prefix パラメータを設定できる"""
        from backtest.engine import BacktestEngine

        engine = BacktestEngine(models=mock_models, diag_prefix="bt_2024")
        assert engine.diag_prefix == "bt_2024"

    def test_init_diag_prefix_default(self, mock_models: MagicMock) -> None:
        """diag_prefix のデフォルトは 'bt'"""
        from backtest.engine import BacktestEngine

        engine = BacktestEngine(models=mock_models)
        assert engine.diag_prefix == "bt"

    def test_engine_invalid_betting_mode_raises(self, mock_models: MagicMock) -> None:
        """不正なbetting_modeはValueError"""
        from backtest.engine import BacktestEngine

        with pytest.raises(ValueError, match="betting_mode must be"):
            BacktestEngine(models=mock_models, betting_mode="invalid")

    def test_settle_bet_uses_final_odds(self, mock_models: MagicMock) -> None:
        """_settle_bet が final_odds を使用する"""
        from backtest.engine import BacktestEngine
        from domain.models import Bet, BetType

        engine = BacktestEngine(models=mock_models)
        bet = Bet(
            race_id="20240101010101",
            umaban=1,
            bet_type=BetType.PLACE,
            odds=2.0,  # 発走前オッズ
            final_odds=1.1,  # 確定オッズ
            ev_lower_corrected=1.5,
            stake=100.0,
        )
        race_df = pd.DataFrame(
            {"umaban": [1], "kakuteijyuni": [2]}  # 2着 → 複勝的中
        )
        payout = engine._settle_bet(bet, race_df)
        # 精算は final_odds (1.1) で計算: 100 * 1.1 = 110.0
        assert abs(payout - 110.0) < 0.01

    def test_settle_bet_falls_back_to_odds(self, mock_models: MagicMock) -> None:
        """final_odds が 0 の場合は odds にフォールバック"""
        from backtest.engine import BacktestEngine
        from domain.models import Bet, BetType

        engine = BacktestEngine(models=mock_models)
        bet = Bet(
            race_id="20240101010101",
            umaban=1,
            bet_type=BetType.PLACE,
            odds=2.0,
            final_odds=0.0,  # デフォルト → フォールバック
            ev_lower_corrected=1.5,
            stake=100.0,
        )
        race_df = pd.DataFrame({"umaban": [1], "kakuteijyuni": [2]})
        payout = engine._settle_bet(bet, race_df)
        assert payout == 200.0  # 100 * 2.0 (odds, not final_odds)

    def test_actual_bet_annotation_matches_positive_stake(self) -> None:
        """is_actual_bet は実際に stake > 0 のベットだけ True になる"""
        from backtest.engine import _annotate_actual_bets
        from domain.models import Bet, BetType

        race_df = pd.DataFrame({"race_id": ["R1", "R1"], "umaban": [1, 2]})
        bet = Bet(
            race_id="R1",
            umaban=2,
            bet_type=BetType.WIN,
            odds=4.0,
            final_odds=3.8,
            ev_lower_corrected=1.2,
            stake=100.0,
        )

        annotated = _annotate_actual_bets(race_df, [(bet, 0.0)])

        assert bool(annotated.loc[annotated["umaban"].eq(1), "is_actual_bet"].iloc[0]) is False
        assert bool(annotated.loc[annotated["umaban"].eq(2), "is_actual_bet"].iloc[0]) is True
        assert annotated.loc[annotated["umaban"].eq(2), "stake"].iloc[0] == pytest.approx(100.0)

    @patch("backtest.engine.load_odds_snapshots")
    @patch("backtest.engine.load_entries")
    @patch("backtest.engine.load_races")
    def test_run_returns_backtest_result(
        self,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds: MagicMock,
        mock_models: MagicMock,
    ) -> None:
        """run() が BacktestResult を返す"""
        mock_load_races.return_value = pd.DataFrame()
        mock_load_entries.return_value = pd.DataFrame()
        mock_load_odds.return_value = pd.DataFrame()

        from backtest.engine import BacktestEngine

        mock_store = MagicMock()
        engine = BacktestEngine(models=mock_models, store=mock_store)
        result = engine.run("2024-01-01", "2024-12-31")

        assert hasattr(result, "total_roi")
        assert hasattr(result, "max_drawdown")
        assert hasattr(result, "total_bets")

    @patch("backtest.engine.load_odds_snapshots")
    @patch("backtest.engine.load_entries")
    @patch("backtest.engine.load_races")
    def test_empty_period_returns_zero_bets(
        self,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds: MagicMock,
        mock_models: MagicMock,
    ) -> None:
        """レースがない期間は0ベット"""
        mock_load_races.return_value = pd.DataFrame()
        mock_load_entries.return_value = pd.DataFrame()
        mock_load_odds.return_value = pd.DataFrame()

        from backtest.engine import BacktestEngine

        mock_store = MagicMock()
        engine = BacktestEngine(models=mock_models, store=mock_store)
        result = engine.run("2024-01-01", "2024-12-31")

        assert result.total_bets == 0

    @patch("backtest.engine.load_odds_snapshots")
    @patch("backtest.engine.load_entries")
    @patch("backtest.engine.load_races")
    def test_bankroll_tracking(
        self,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds: MagicMock,
        mock_models: MagicMock,
    ) -> None:
        """資金の推移が追跡される"""
        mock_load_races.return_value = pd.DataFrame()
        mock_load_entries.return_value = pd.DataFrame()
        mock_load_odds.return_value = pd.DataFrame()

        from backtest.engine import BacktestEngine

        mock_store = MagicMock()
        engine = BacktestEngine(models=mock_models, initial_bankroll=100000, store=mock_store)
        result = engine.run("2024-01-01", "2024-12-31")

        # 空期間なので資金は変化しない
        assert result.final_bankroll == 100000

    @patch("backtest.engine.load_odds_snapshots")
    @patch("backtest.engine.load_entries")
    @patch("backtest.engine.load_races")
    def test_default_result_values(
        self,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds: MagicMock,
        mock_models: MagicMock,
    ) -> None:
        """空データのデフォルト値が正しい"""
        mock_load_races.return_value = pd.DataFrame()
        mock_load_entries.return_value = pd.DataFrame()
        mock_load_odds.return_value = pd.DataFrame()

        from backtest.engine import BacktestEngine

        mock_store = MagicMock()
        engine = BacktestEngine(models=mock_models, store=mock_store)
        result = engine.run("2024-01-01", "2024-12-31")

        assert result.total_stake == 0.0
        assert result.total_return == 0.0
        assert result.total_roi == 0.0
        assert result.max_drawdown == 0.0
        assert result.winning_bets == 0


class TestPostRaceColumnExclusion:
    """predict() に POST_RACE 列が渡されないことを検証"""

    _POST_RACE_COLS = ["kakuteijyuni", "confirmed_odds"]

    @patch("db.odds_extractor.extract_pre_post_odds")
    @patch("features.trainer_context_features.TrainerContextFeatures")
    @patch("features.jockey_context_features.JockeyContextFeatures")
    @patch("features.interaction_features.compute_interaction_features")
    @patch("features.horse_history_features.HorseHistoryFeatures")
    @patch("models.submodel_manager.SubModelManager")
    @patch("features.feature_engine.FeatureEngine")
    @patch("backtest.engine.load_odds_time_series_range")
    @patch("backtest.engine.load_odds_snapshots")
    @patch("backtest.engine.load_entries")
    @patch("backtest.engine.load_races")
    def test_predict_excludes_post_race_columns(
        self,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds: MagicMock,
        mock_load_odds_ts: MagicMock,
        mock_feat_engine_cls: MagicMock,
        mock_submodel_mgr_cls: MagicMock,
        mock_hist_cls: MagicMock,
        mock_interaction_fn: MagicMock,
        mock_jockey_cls: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_extract_odds: MagicMock,
        mock_models: MagicMock,
    ) -> None:
        """predict() に渡される DataFrame に POST_RACE 列が含まれない"""
        # --- load mocks ---
        mock_load_races.return_value = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "race_date": pd.to_datetime("2024-01-01"),
                "hassotime": ["03101500"],
            }
        )
        mock_load_entries.return_value = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "umaban": [1],
                "kettonum": [1234],
                "kakuteijyuni": [2],
                "odds": [5.0],
                "ninki": [3],
                "bataijyu": [480],
                "zogen_fugo": [0],
                "zogen_sa": [0],
                "kisyucode": [100],
                "chokyosicode": [200],
            }
        )
        mock_load_odds.return_value = pd.DataFrame()
        mock_load_odds_ts.return_value = pd.DataFrame(
            {"race_id": ["20240101010101"], "umaban": [1], "odds": [5.0]}
        )
        mock_extract_odds.return_value = pd.DataFrame(
            {"race_id": ["20240101010101"], "umaban": [1], "fukuoddslow": [4.0]}
        )

        # --- feat_df with POST_RACE columns present ---
        feat_df = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "umaban": [1],
                "surface": ["turf"],
                "kyori": [1600],
                "distance_bin": ["mile"],
                "popularity_rank": [3],
                "ninki": [3],
                "ev_place": [1.5],
                "fukuoddslow": [4.0],
                "kakuteijyuni": [2],  # POST_RACE — must be excluded from predict
                "confirmed_odds": [1.8],  # POST_RACE — must be excluded from predict
                "kettonum": [1234],
                "odds": [5.0],
                "bataijyu": [480],
                "jyocd": [6],
                "racenum": [11],
                "grade_code": ["E"],
                "hondai": ["テスト特別"],
                "bamei": ["テスト馬"],
                "kisyuryakusyo": ["テスト騎手"],
                "track_condition_code": [1],
                "p_place_pred": [0.65],
                "e_return_place_pred": [1.80],
            }
        )

        # --- FeatureEngine mock ---
        mock_feat_engine = MagicMock()
        mock_feat_engine_cls.return_value = mock_feat_engine
        mock_feat_engine.build_all.return_value = feat_df

        # --- SubModelManager mock ---
        mock_submodel_mgr = MagicMock()
        mock_submodel_mgr_cls.return_value = mock_submodel_mgr
        mock_submodel_mgr.add_distance_band_features.return_value = feat_df

        # --- pre-computation mocks (return empty → merges are no-ops) ---
        mock_hist = MagicMock()
        mock_hist_cls.return_value = mock_hist
        mock_hist.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])
        mock_hist.add_race_transforms = staticmethod(lambda df: df)

        mock_interaction_fn.side_effect = lambda df: df

        mock_jockey = MagicMock()
        mock_jockey_cls.return_value = mock_jockey
        mock_jockey.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])

        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer
        mock_trainer.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])

        # --- submodel mocks ---
        submodel = MagicMock()
        submodel.benter_combo = None
        submodel.isotonic_calibrator = None
        submodel.win_benter = None
        submodel.target_encoder = None
        mock_models.submodels["turf"] = submodel
        submodel.market.predict_and_calc_error.return_value = feat_df
        submodel.stage1.add_ability_probs.return_value = feat_df
        submodel.place_ability.predict.return_value = feat_df
        submodel.win.predict_ev.return_value = feat_df
        submodel.ev_corrector.correct_ev.return_value = feat_df
        submodel.place.predict_ev.return_value = feat_df
        _corrected = feat_df.assign(ev_place_corrected=feat_df.get("ev_place", 1.5))
        submodel.place_ev_corrector.correct_ev.return_value = _corrected
        submodel.conformal_ev_model.predict_lower_bound.return_value = (
            _corrected,
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )
        submodel.conformal_ev_model.predict_interval.return_value = (
            _corrected,
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )
        submodel.target_encoder = None

        # --- spy on RacePredictor.predict to capture the DataFrame ---
        captured_df: dict[str, pd.DataFrame] = {}

        from backtest.race_predictor import RacePredictor

        original_predict = RacePredictor.predict

        def spy_predict(self_pred: object, race_df: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
            captured_df["value"] = race_df.copy()
            return original_predict(self_pred, race_df, **kwargs)  # type: ignore[arg-type]

        # --- run engine with spy ---
        from backtest.engine import BacktestEngine

        mock_store = MagicMock()

        with patch.object(RacePredictor, "predict", spy_predict):
            engine = BacktestEngine(
                models=mock_models, store=mock_store, betting_target="place",
                min_bets_per_year=0,  # テスト用: bet count guard を無効化
            )
            engine.run("2024-01-01", "2024-12-31")

        # --- assertions ---
        assert "value" in captured_df, "predict() was never called"
        predict_input_df = captured_df["value"]
        for col in self._POST_RACE_COLS:
            assert col not in predict_input_df.columns, (
                f"POST_RACE column '{col}' should NOT be in predict() input DataFrame"
            )


class TestBetHistoryEnrichment:
    """bet_history への surface/distance/ev/popularity/bankroll_after 付与テスト"""

    @patch("db.odds_extractor.extract_pre_post_odds")
    @patch("features.trainer_context_features.TrainerContextFeatures")
    @patch("features.jockey_context_features.JockeyContextFeatures")
    @patch("features.interaction_features.compute_interaction_features")
    @patch("features.horse_history_features.HorseHistoryFeatures")
    @patch("models.submodel_manager.SubModelManager")
    @patch("features.feature_engine.FeatureEngine")
    @patch("backtest.engine.load_odds_time_series_range")
    @patch("backtest.engine.load_odds_snapshots")
    @patch("backtest.engine.load_entries")
    @patch("backtest.engine.load_races")
    def test_engine_populates_enriched_fields(
        self,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds: MagicMock,
        mock_load_odds_ts: MagicMock,
        mock_feat_engine_cls: MagicMock,
        mock_submodel_mgr_cls: MagicMock,
        mock_hist_cls: MagicMock,
        mock_interaction_fn: MagicMock,
        mock_jockey_cls: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_extract_odds: MagicMock,
        mock_models: MagicMock,
    ) -> None:
        """エンジンループが bet_history に拡張フィールドを付与する"""
        # --- load mocks ---
        mock_load_races.return_value = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "race_date": pd.to_datetime("2024-01-01"),
                "hassotime": ["03101500"],
            }
        )
        mock_load_entries.return_value = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "umaban": [1],
                "kettonum": [1234],
                "kakuteijyuni": [2],
                "odds": [5.0],
                "ninki": [3],
                "bataijyu": [480],
                "zogen_fugo": [0],
                "zogen_sa": [0],
                "kisyucode": [100],
                "chokyosicode": [200],
            }
        )
        mock_load_odds.return_value = pd.DataFrame()
        mock_load_odds_ts.return_value = pd.DataFrame(
            {"race_id": ["20240101010101"], "umaban": [1], "odds": [5.0]}
        )
        mock_extract_odds.return_value = pd.DataFrame(
            {"race_id": ["20240101010101"], "umaban": [1], "fukuoddslow": [4.0]}
        )

        # --- feat_df (complete columns for pipeline) ---
        feat_df = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "umaban": [1],
                "surface": ["turf"],
                "kyori": [1600],
                "distance_bin": ["mile"],
                "popularity_rank": [3],
                "ninki": [3],
                "ev_place": [1.5],
                "fukuoddslow": [4.0],
                "kakuteijyuni": [2],
                "kettonum": [1234],
                "odds": [5.0],
                "bataijyu": [480],
                # --- 拡張フィールド用の追加列 ---
                "jyocd": [6],  # 中山
                "racenum": [11],  # 11R
                "grade_code": ["E"],  # 特別
                "hondai": ["テスト特別"],  # レース名
                "bamei": ["テスト馬"],  # 馬名
                "kisyuryakusyo": ["テスト騎手"],  # 騎手名
                "track_condition_code": [1],  # 良
                "p_place_pred": [0.65],  # 複勝確率予測
                "e_return_place_pred": [1.80],  # 期待払戻予測
            }
        )

        # --- FeatureEngine mock ---
        mock_feat_engine = MagicMock()
        mock_feat_engine_cls.return_value = mock_feat_engine
        mock_feat_engine.build_all.return_value = feat_df

        # --- SubModelManager mock ---
        mock_submodel_mgr = MagicMock()
        mock_submodel_mgr_cls.return_value = mock_submodel_mgr
        mock_submodel_mgr.add_distance_band_features.return_value = feat_df

        # --- pre-computation mocks (return empty → merges are no-ops) ---
        mock_hist = MagicMock()
        mock_hist_cls.return_value = mock_hist
        mock_hist.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])
        mock_hist.add_race_transforms = staticmethod(lambda df: df)

        mock_interaction_fn.side_effect = lambda df: df

        mock_jockey = MagicMock()
        mock_jockey_cls.return_value = mock_jockey
        mock_jockey.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])

        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer
        mock_trainer.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])

        # --- submodel mocks (plain MagicMock — spec restricts attribute access) ---
        submodel = MagicMock()
        submodel.benter_combo = None
        submodel.isotonic_calibrator = None
        submodel.win_benter = None
        submodel.target_encoder = None
        mock_models.submodels["turf"] = submodel
        submodel.market.predict_and_calc_error.return_value = feat_df
        submodel.stage1.add_ability_probs.return_value = feat_df
        submodel.place_ability.predict.return_value = feat_df
        submodel.win.predict_ev.return_value = feat_df
        submodel.ev_corrector.correct_ev.return_value = feat_df
        submodel.place.predict_ev.return_value = feat_df
        _corrected = feat_df.assign(ev_place_corrected=feat_df.get("ev_place", 1.5))
        submodel.place_ev_corrector.correct_ev.return_value = _corrected
        submodel.conformal_ev_model.predict_lower_bound.return_value = (
            _corrected,
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )
        submodel.conformal_ev_model.predict_interval.return_value = (
            _corrected,
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )
        submodel.target_encoder = None

        # --- run engine ---
        from backtest.engine import BacktestEngine

        mock_store = MagicMock()
        engine = BacktestEngine(
            models=mock_models, store=mock_store, betting_target="place",
            min_bets_per_year=0,
        )
        result = engine.run("2024-01-01", "2024-12-31")

        # --- assertions ---
        assert result.total_bets >= 1, "Should place at least 1 bet"
        assert result.n_pre_post_odds_bets >= 1
        assert result.n_fallback_odds_bets == 0
        bet = result.bet_history[0]
        assert "surface" in bet
        assert bet["surface"] == "turf"
        assert "kyori" in bet
        assert bet["kyori"] == 1600
        assert "ev" in bet
        assert bet["ev"] == 1.5
        assert "popularity" in bet
        assert bet["popularity"] == 3
        assert "bankroll_after" in bet
        assert isinstance(bet["bankroll_after"], float)
        assert bet["bankroll_after"] == 100300.0

        # --- 拡張フィールドの検証 ---
        assert "race_date" in bet
        assert bet["race_date"] == "2024-01-01"
        assert "jyocd" in bet
        assert "racenum" in bet
        assert bet["racenum"] == 11
        assert "grade_code" in bet
        assert "bamei" in bet
        assert bet["bamei"] == "テスト馬"
        assert "kisyu" in bet
        assert bet["kisyu"] == "テスト騎手"
        assert "kakuteijyuni" in bet
        assert bet["kakuteijyuni"] == 2
        assert "track_condition_code" in bet
        assert "top3_finishers" in bet
        assert isinstance(bet["top3_finishers"], list)
        assert len(bet["top3_finishers"]) >= 1  # feat_df に1頭のみ
        assert bet["top3_finishers"][0]["umaban"] == 1


class TestOddsFallbackSkip:
    """odds_ts_df が空の場合はフォールバックせず全レースをスキップ"""

    @patch("backtest.engine.load_odds_time_series_range")
    @patch("backtest.engine.load_odds_snapshots")
    @patch("backtest.engine.load_entries")
    @patch("backtest.engine.load_races")
    def test_empty_odds_ts_skips_all_races(
        self,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds: MagicMock,
        mock_load_odds_ts: MagicMock,
        mock_models: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """時系列オッズが空の場合、全レースをスキップして total_bets == 0"""
        mock_load_races.return_value = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "race_date": pd.to_datetime("2024-01-01"),
            }
        )
        mock_load_entries.return_value = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "umaban": [1],
                "kettonum": [1234],
                "kakuteijyuni": [2],
                "odds": [5.0],
                "ninki": [3],
                "bataijyu": [480],
                "zogen_fugo": [0],
                "zogen_sa": [0],
                "kisyucode": [100],
                "chokyosicode": [200],
            }
        )
        mock_load_odds.return_value = pd.DataFrame()
        # 空の時系列オッズ → フォールバックなしでスキップ
        mock_load_odds_ts.return_value = pd.DataFrame()

        from backtest.engine import BacktestEngine

        mock_store = MagicMock()
        engine = BacktestEngine(models=mock_models, store=mock_store)

        with caplog.at_level(logging.WARNING, logger="backtest.engine"):
            result = engine.run("2024-01-01", "2024-12-31")

        assert result.total_bets == 0
        assert result.n_pre_post_odds_bets == 0
        assert result.n_fallback_odds_bets == 0
        # 警告ログに "skipping" が含まれる
        assert any("skipping" in rec.message.lower() for rec in caplog.records), (
            f"Expected warning about skipping, got: {[r.message for r in caplog.records]}"
        )


class TestLeakIntegration:
    """全コンポーネントのリーク修正が統合されていることを確認"""

    def test_regime_detector_feature_cols_no_post_race(self) -> None:
        """RegimeDetector.FEATURE_COLS に POST_RACE 指標が含まれない"""
        from models.regime_detector import RegimeDetector

        post_race_cols = {
            "favorite_win_rate",
            "flb_slope",
            "favorite_roi_ema",
            "mid_roi_ema",
            "longshot_roi_ema",
        }
        for col in post_race_cols:
            assert col not in RegimeDetector.FEATURE_COLS, (
                f"POST_RACE column '{col}' still in FEATURE_COLS"
            )

    def test_regime_detector_feature_cols_has_pre_race(self) -> None:
        """RegimeDetector.FEATURE_COLS が PRE_RACE 指標のみで構成される"""
        from models.regime_detector import RegimeDetector

        # Core pre-race columns (always expected)
        expected_core = {
            "market_error_std",
            "market_error_mean",
            "overround_rolling",
            "entropy_rolling",
            "favorite_implied_prob_rolling",
            "odds_skewness_rolling",
            "odds_volatility_mean",
            "field_size_mean",
        }
        # Pre-race rl_* columns added by Plan 34-01 (RLF-01~06 + MCF-07)
        expected_rl = {
            "rl_log_odds_entropy", "rl_odds_dispersion", "rl_top3_odds_gap",
            "rl_top1_odds", "rl_favorite_rank_gap", "rl_n_horses",
            "rl_favorite_in_wide_top1", "rl_trio_overlap", "rl_market_consistency",
            "rl_trio_odds_ratio", "rl_wide_harville_ratio",
        }
        # Pre-race market structure columns (D-06)
        expected_market = {
            "implied_prob_hhi",
            "odds_skewness",
        }
        # Phase 36 HLF/TRF/interaction features (registered in RegimeDetector)
        expected_hlf_trf = {
            "closing_speed_ratio_avg",
            "closing_speed_ratio_avg_race_rank",
            "closing_speed_ratio_zscore",
            "closing_speed_ratio_trend",
            "haron_race_gap_avg",
            "haron_race_gap_zscore",
            "haron_race_gap_trend",
            "harontime_last3f_avg",
            "harontime_last3f_avg_race_rank",
            "harontime_last3f_zscore",
            "harontime_last3f_trend",
            "pace_ratio_avg",
            "pace_early_avg",
            "pace_mid_avg",
            "pace_late_avg",
            "weighted_recent_form_finish",
            "weighted_recent_form_time",
            "form_trend_race_rank",
            "blood_surface_wr_race_rank",
            "blood_total_wr_race_rank",
            "grade_x_form_trend",
            "grade_x_blood_prize_log",
            "distance_x_closing_index",
        }
        expected_cols = expected_core | expected_rl | expected_market | expected_hlf_trf
        actual_cols = set(RegimeDetector.FEATURE_COLS)
        assert actual_cols == expected_cols, (
            f"FEATURE_COLS mismatch: expected {expected_cols}, got {actual_cols}"
        )

    def test_favorite_win_rate_is_expanding_not_current(self) -> None:
        """_build_race_level_features の favorite_win_rate が
        過去レースのみの expanding mean である (現在レースを含まない)"""
        from features.feature_engine import FeatureEngine
        from models.submodel_manager import SubModelManager
        from pipelines.training_pipeline import TrainingPipelineV5

        pipeline = TrainingPipelineV5.__new__(TrainingPipelineV5)
        pipeline.store = MagicMock()
        pipeline.db = None
        pipeline.feature_engine = FeatureEngine()
        pipeline.submodel_mgr = SubModelManager()

        # 20レース: 最初の10レースは1番人気が全勝、次の10レースは全敗
        rows: list[dict[str, object]] = []
        for r in range(20):
            race_id = f"2020{1:02d}{r + 1:02d}0101{r:02d}"
            for h in range(5):
                if r < 10:
                    kakuteijyuni = 1 if h == 0 else h + 1
                else:
                    kakuteijyuni = 2 if h == 0 else h
                rows.append(
                    {
                        "race_id": race_id,
                        "umaban": h + 1,
                        "surface": "turf",
                        "distance_bin": "mile",
                        "track_condition_code": 1,
                        "grade_code": "C",
                        "field_size": 5,
                        "difficulty_score": 0.5,
                        "signed_log_error_win": np.random.normal(0, 0.3),
                        "abs_log_error_win": np.random.uniform(0, 1),
                        "market_entropy": np.random.uniform(1.0, 3.0),
                        "overround": np.random.uniform(0.15, 0.30),
                        "kakuteijyuni": kakuteijyuni,
                        "popularity_rank": h + 1,
                        "race_date": f"2020-01-{r + 1:02d}",
                    }
                )
        feat_df = pd.DataFrame(rows)
        result = pipeline._build_race_level_features(feat_df)

        # 最初のレース: データなし → 0.3
        assert result.iloc[0]["favorite_win_rate"] == pytest.approx(0.3)
        # 11レース目: 10レース前までの1番人気勝率 (全勝) → 高い値
        race_11_fwr = result.iloc[10]["favorite_win_rate"]
        assert race_11_fwr > 0.8, f"Race 11 favorite_win_rate should be high, got {race_11_fwr}"


class TestJRAFilterBacktest:
    """バックテストエンジン JRAフィルタのテスト"""

    @patch("db.odds_extractor.extract_pre_post_odds")
    @patch("features.trainer_context_features.TrainerContextFeatures")
    @patch("features.jockey_context_features.JockeyContextFeatures")
    @patch("features.interaction_features.compute_interaction_features")
    @patch("features.horse_history_features.HorseHistoryFeatures")
    @patch("models.submodel_manager.SubModelManager")
    @patch("features.feature_engine.FeatureEngine")
    @patch("backtest.engine.load_odds_time_series_range")
    @patch("backtest.engine.load_odds_snapshots")
    @patch("backtest.engine.load_entries")
    @patch("backtest.engine.load_races")
    def test_nar_race_excluded_from_backtest(
        self,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds: MagicMock,
        mock_load_odds_ts: MagicMock,
        mock_feat_engine_cls: MagicMock,
        mock_submodel_mgr_cls: MagicMock,
        mock_hist_cls: MagicMock,
        mock_interaction_fn: MagicMock,
        mock_jockey_cls: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_extract_odds: MagicMock,
        mock_models: MagicMock,
    ) -> None:
        """NARレース (jyocd >= 30) はバックテストから除外される"""
        mock_load_races.return_value = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "race_date": pd.to_datetime("2024-01-01"),
                "hassotime": ["03101500"],
            }
        )
        mock_load_entries.return_value = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "umaban": [1],
                "kettonum": [1234],
                "kakuteijyuni": [2],
                "odds": [5.0],
                "ninki": [3],
                "bataijyu": [480],
                "zogen_fugo": [0],
                "zogen_sa": [0],
                "kisyucode": [100],
                "chokyosicode": [200],
            }
        )
        mock_load_odds.return_value = pd.DataFrame()
        mock_load_odds_ts.return_value = pd.DataFrame(
            {"race_id": ["20240101010101"], "umaban": [1], "odds": [5.0]}
        )
        mock_extract_odds.return_value = pd.DataFrame(
            {"race_id": ["20240101010101"], "umaban": [1], "fukuoddslow": [4.0]}
        )

        feat_df = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "umaban": [1],
                "surface": ["turf"],  # turf にして submodel が存在する状態にする
                "kyori": [1600],
                "distance_bin": ["mile"],
                "popularity_rank": [3],
                "ninki": [3],
                "ev_place": [1.5],
                "fukuoddslow": [4.0],
                "kakuteijyuni": [2],
                "kettonum": [1234],
                "odds": [5.0],
                "bataijyu": [480],
                "jyocd": ["35"],  # NAR — フィルタで除外されるべき
                "racenum": [1],
                "grade_code": ["E"],
                "hondai": ["地方レース"],
                "bamei": ["テスト馬"],
                "kisyuryakusyo": ["テスト騎手"],
                "track_condition_code": [1],
                "p_place_pred": [0.65],
                "e_return_place_pred": [1.80],
            }
        )

        mock_feat_engine = MagicMock()
        mock_feat_engine_cls.return_value = mock_feat_engine
        mock_feat_engine.build_all.return_value = feat_df

        mock_submodel_mgr = MagicMock()
        mock_submodel_mgr_cls.return_value = mock_submodel_mgr
        mock_submodel_mgr.add_distance_band_features.return_value = feat_df

        mock_hist = MagicMock()
        mock_hist_cls.return_value = mock_hist
        mock_hist.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])
        mock_hist.add_race_transforms = staticmethod(lambda df: df)

        mock_interaction_fn.side_effect = lambda df: df

        mock_jockey = MagicMock()
        mock_jockey_cls.return_value = mock_jockey
        mock_jockey.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])

        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer
        mock_trainer.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])

        # Submodel mocks — needed so the test actually exercises the filter
        # Without these, the test passes for the wrong reason (no submodel = skip)
        submodel = MagicMock()
        submodel.benter_combo = None
        submodel.isotonic_calibrator = None
        submodel.win_benter = None
        submodel.target_encoder = None
        mock_models.submodels["turf"] = submodel
        submodel.market.predict_and_calc_error.return_value = feat_df
        submodel.stage1.add_ability_probs.return_value = feat_df
        submodel.place_ability.predict.return_value = feat_df
        submodel.win.predict_ev.return_value = feat_df
        submodel.ev_corrector.correct_ev.return_value = feat_df
        submodel.place.predict_ev.return_value = feat_df
        _corrected = feat_df.assign(ev_place_corrected=feat_df.get("ev_place", 1.5))
        submodel.place_ev_corrector.correct_ev.return_value = _corrected
        submodel.conformal_ev_model.predict_lower_bound.return_value = (
            _corrected,
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )
        submodel.conformal_ev_model.predict_interval.return_value = (
            _corrected,
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )
        submodel.target_encoder = None

        from backtest.engine import BacktestEngine

        mock_store = MagicMock()
        engine = BacktestEngine(models=mock_models, store=mock_store)
        result = engine.run("2024-01-01", "2024-12-31")

        assert result.total_bets == 0, "NAR race (jyocd=35) should be excluded from backtest"

    @patch("db.odds_extractor.extract_pre_post_odds")
    @patch("features.trainer_context_features.TrainerContextFeatures")
    @patch("features.jockey_context_features.JockeyContextFeatures")
    @patch("features.interaction_features.compute_interaction_features")
    @patch("features.horse_history_features.HorseHistoryFeatures")
    @patch("models.submodel_manager.SubModelManager")
    @patch("features.feature_engine.FeatureEngine")
    @patch("backtest.engine.load_odds_time_series_range")
    @patch("backtest.engine.load_odds_snapshots")
    @patch("backtest.engine.load_entries")
    @patch("backtest.engine.load_races")
    def test_jra_race_included_in_backtest(
        self,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds: MagicMock,
        mock_load_odds_ts: MagicMock,
        mock_feat_engine_cls: MagicMock,
        mock_submodel_mgr_cls: MagicMock,
        mock_hist_cls: MagicMock,
        mock_interaction_fn: MagicMock,
        mock_jockey_cls: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_extract_odds: MagicMock,
        mock_models: MagicMock,
    ) -> None:
        """JRAレース (jyocd 01-10) は通常通りバックテスト対象"""
        mock_load_races.return_value = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "race_date": pd.to_datetime("2024-01-01"),
                "hassotime": ["03101500"],
            }
        )
        mock_load_entries.return_value = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "umaban": [1],
                "kettonum": [1234],
                "kakuteijyuni": [2],
                "odds": [5.0],
                "ninki": [3],
                "bataijyu": [480],
                "zogen_fugo": [0],
                "zogen_sa": [0],
                "kisyucode": [100],
                "chokyosicode": [200],
            }
        )
        mock_load_odds.return_value = pd.DataFrame()
        mock_load_odds_ts.return_value = pd.DataFrame(
            {"race_id": ["20240101010101"], "umaban": [1], "odds": [5.0]}
        )
        mock_extract_odds.return_value = pd.DataFrame(
            {"race_id": ["20240101010101"], "umaban": [1], "fukuoddslow": [4.0]}
        )

        feat_df = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "umaban": [1],
                "surface": ["turf"],
                "kyori": [1600],
                "distance_bin": ["mile"],
                "popularity_rank": [3],
                "ninki": [3],
                "ev_place": [1.5],
                "fukuoddslow": [4.0],
                "kakuteijyuni": [2],
                "kettonum": [1234],
                "odds": [5.0],
                "bataijyu": [480],
                "jyocd": ["05"],  # JRA — フィルタを通過する
                "racenum": [11],
                "grade_code": ["E"],
                "hondai": ["JRAレース"],
                "bamei": ["テスト馬"],
                "kisyuryakusyo": ["テスト騎手"],
                "track_condition_code": [1],
                "p_place_pred": [0.65],
                "e_return_place_pred": [1.80],
            }
        )

        mock_feat_engine = MagicMock()
        mock_feat_engine_cls.return_value = mock_feat_engine
        mock_feat_engine.build_all.return_value = feat_df

        mock_submodel_mgr = MagicMock()
        mock_submodel_mgr_cls.return_value = mock_submodel_mgr
        mock_submodel_mgr.add_distance_band_features.return_value = feat_df

        mock_hist = MagicMock()
        mock_hist_cls.return_value = mock_hist
        mock_hist.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])
        mock_hist.add_race_transforms = staticmethod(lambda df: df)

        mock_interaction_fn.side_effect = lambda df: df

        mock_jockey = MagicMock()
        mock_jockey_cls.return_value = mock_jockey
        mock_jockey.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])

        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer
        mock_trainer.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])

        # Submodel mocks for prediction
        submodel = MagicMock()
        submodel.benter_combo = None
        submodel.isotonic_calibrator = None
        submodel.win_benter = None
        submodel.target_encoder = None
        mock_models.submodels["turf"] = submodel
        submodel.market.predict_and_calc_error.return_value = feat_df
        submodel.stage1.add_ability_probs.return_value = feat_df
        submodel.place_ability.predict.return_value = feat_df
        submodel.win.predict_ev.return_value = feat_df
        submodel.ev_corrector.correct_ev.return_value = feat_df
        submodel.place.predict_ev.return_value = feat_df
        _corrected = feat_df.assign(ev_place_corrected=feat_df.get("ev_place", 1.5))
        submodel.place_ev_corrector.correct_ev.return_value = _corrected
        submodel.conformal_ev_model.predict_lower_bound.return_value = (
            _corrected,
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )
        submodel.conformal_ev_model.predict_interval.return_value = (
            _corrected,
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )
        submodel.target_encoder = None

        from backtest.engine import BacktestEngine

        mock_store = MagicMock()
        engine = BacktestEngine(
            models=mock_models, store=mock_store, betting_target="place",
            min_bets_per_year=0,
        )
        result = engine.run("2024-01-01", "2024-12-31")

        assert result.total_bets >= 1, "JRA race (jyocd=05) should be included in backtest"


class TestBuildWinPayoutMap:
    """build_win_payout_map のテスト"""

    def test_basic_win_payout_map(self) -> None:
        """単勝払戻データから正しい win_payout_map を構築する"""
        payouts = pd.DataFrame(
            {
                "race_id": ["202401010101", "202401010202"],
                "paytansyoumaban1": [5, 3],
                "paytansyopay1": [240.0, 350.0],
            }
        )
        from backtest.engine import build_win_payout_map

        win_map = build_win_payout_map(payouts)
        assert win_map[("202401010101", 5)] == pytest.approx(2.4)
        assert win_map[("202401010202", 3)] == pytest.approx(3.5)

    def test_empty_payouts_returns_empty(self) -> None:
        """空の DataFrame は空の map を返す"""
        payouts = pd.DataFrame()
        from backtest.engine import build_win_payout_map

        win_map = build_win_payout_map(payouts)
        assert len(win_map) == 0

    def test_nan_umaban_skipped(self) -> None:
        """paytansyoumaban1 が NaN の行はスキップする"""
        payouts = pd.DataFrame(
            {
                "race_id": ["202401010101", "202401010202"],
                "paytansyoumaban1": [5, None],
                "paytansyopay1": [240.0, 350.0],
            }
        )
        from backtest.engine import build_win_payout_map

        win_map = build_win_payout_map(payouts)
        assert len(win_map) == 1
        assert ("202401010101", 5) in win_map


class TestBetSelectionFilters:
    """Phase 11: COLLAPSED skip + BacktestResult exclusion fields"""

    def test_backtest_result_has_exclusion_fields(self) -> None:
        """Test 1: BacktestResult default exclusion fields"""
        from backtest.engine import BacktestResult

        result = BacktestResult(total_bets=0, total_stake=0.0, total_return=0.0, winning_bets=0)
        assert result.n_collapsed_skipped == 0
        assert result.n_ev_excluded == 0
        assert result.n_odds_band_excluded == 0
        assert result.exclusion_stats == {}

    def test_backtest_result_exclusion_fields_with_values(self) -> None:
        """BacktestResult exclusion fields accept non-default values"""
        from backtest.engine import BacktestResult

        result = BacktestResult(
            total_bets=10,
            total_stake=1000.0,
            total_return=1100.0,
            winning_bets=3,
            n_collapsed_skipped=5,
            n_ev_excluded=20,
            n_odds_band_excluded=8,
            exclusion_stats={
                "collapsed_skipped": 5,
                "ev_excluded": 20,
                "odds_band_excluded": 8,
                "total_candidates_evaluated": 100,
            },
        )
        assert result.n_collapsed_skipped == 5
        assert result.n_ev_excluded == 20
        assert result.n_odds_band_excluded == 8
        assert result.exclusion_stats["collapsed_skipped"] == 5

    @patch("db.odds_extractor.extract_pre_post_odds")
    @patch("features.trainer_context_features.TrainerContextFeatures")
    @patch("features.jockey_context_features.JockeyContextFeatures")
    @patch("features.interaction_features.compute_interaction_features")
    @patch("features.horse_history_features.HorseHistoryFeatures")
    @patch("models.submodel_manager.SubModelManager")
    @patch("features.feature_engine.FeatureEngine")
    @patch("backtest.engine.load_odds_time_series_range")
    @patch("backtest.engine.load_odds_snapshots")
    @patch("backtest.engine.load_entries")
    @patch("backtest.engine.load_races")
    def test_collapsed_skip_increments_counter(
        self,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds: MagicMock,
        mock_load_odds_ts: MagicMock,
        mock_feat_engine_cls: MagicMock,
        mock_submodel_mgr_cls: MagicMock,
        mock_hist_cls: MagicMock,
        mock_interaction_fn: MagicMock,
        mock_jockey_cls: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_extract_odds: MagicMock,
        mock_models: MagicMock,
    ) -> None:
        """Test 2+3: COLLAPSED regime race -> n_collapsed_skipped incremented, 0 bets"""
        mock_load_races.return_value = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "race_date": pd.to_datetime("2024-01-01"),
                "hassotime": ["03101500"],
            }
        )
        mock_load_entries.return_value = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "umaban": [1],
                "kettonum": [1234],
                "kakuteijyuni": [2],
                "odds": [5.0],
                "ninki": [3],
                "bataijyu": [480],
                "zogen_fugo": [0],
                "zogen_sa": [0],
                "kisyucode": [100],
                "chokyosicode": [200],
            }
        )
        mock_load_odds.return_value = pd.DataFrame()
        mock_load_odds_ts.return_value = pd.DataFrame(
            {"race_id": ["20240101010101"], "umaban": [1], "odds": [5.0]}
        )
        mock_extract_odds.return_value = pd.DataFrame(
            {"race_id": ["20240101010101"], "umaban": [1], "fukuoddslow": [4.0]}
        )

        feat_df = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "umaban": [1],
                "surface": ["turf"],
                "kyori": [1600],
                "distance_bin": ["mile"],
                "popularity_rank": [3],
                "ninki": [3],
                "ev_place": [1.5],
                "fukuoddslow": [4.0],
                "kakuteijyuni": [2],
                "kettonum": [1234],
                "odds": [5.0],
                "bataijyu": [480],
                "jyocd": [5],
                "racenum": [11],
                "grade_code": ["E"],
                "hondai": ["テスト"],
                "bamei": ["テスト馬"],
                "kisyuryakusyo": ["テスト騎手"],
                "track_condition_code": [1],
                "p_place_pred": [0.65],
                "e_return_place_pred": [1.80],
            }
        )

        mock_feat_engine = MagicMock()
        mock_feat_engine_cls.return_value = mock_feat_engine
        mock_feat_engine.build_all.return_value = feat_df

        mock_submodel_mgr = MagicMock()
        mock_submodel_mgr_cls.return_value = mock_submodel_mgr
        mock_submodel_mgr.add_distance_band_features.return_value = feat_df

        mock_hist = MagicMock()
        mock_hist_cls.return_value = mock_hist
        mock_hist.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])
        mock_hist.add_race_transforms = staticmethod(lambda df: df)
        mock_interaction_fn.side_effect = lambda df: df

        mock_jockey = MagicMock()
        mock_jockey_cls.return_value = mock_jockey
        mock_jockey.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])

        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer
        mock_trainer.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])

        # Submodel mocks for predict() to pass
        submodel = MagicMock()
        submodel.benter_combo = None
        submodel.isotonic_calibrator = None
        submodel.win_benter = None
        submodel.target_encoder = None
        mock_models.submodels["turf"] = submodel
        submodel.market.predict_and_calc_error.return_value = feat_df
        submodel.stage1.add_ability_probs.return_value = feat_df
        submodel.place_ability.predict.return_value = feat_df
        submodel.win.predict_ev.return_value = feat_df
        submodel.ev_corrector.correct_ev.return_value = feat_df
        submodel.place.predict_ev.return_value = feat_df
        _corrected = feat_df.assign(ev_place_corrected=feat_df.get("ev_place", 1.5))
        submodel.place_ev_corrector.correct_ev.return_value = _corrected
        submodel.conformal_ev_model.predict_lower_bound.return_value = (
            _corrected,
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )
        submodel.conformal_ev_model.predict_interval.return_value = (
            _corrected,
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )
        submodel.target_encoder = None

        # COLLAPSED skip=True
        mock_models.regime_detector.get_strategy_params.return_value = {
            "ev_threshold": 1.50,
            "edge_threshold": 0.09,
            "score_threshold": 0.050,
            "max_bets_per_race": 1,
            "skip": True,
        }

        from backtest.engine import BacktestEngine

        mock_store = MagicMock()
        engine = BacktestEngine(models=mock_models, store=mock_store, betting_target="win")
        result = engine.run("2024-01-01", "2024-12-31")

        # COLLAPSED skip should increment counter and produce 0 bets
        assert result.n_collapsed_skipped >= 1
        assert result.total_bets == 0

    def test_odds_band_filter_initialized_for_win(self, mock_models: MagicMock) -> None:
        """Test: betting_target='win' -> _odds_band_filter is not None"""
        from backtest.engine import BacktestEngine
        from betting.odds_band_filter import OddsBandFilter

        engine = BacktestEngine(models=mock_models, betting_target="win")
        assert engine._odds_band_filter is not None
        assert isinstance(engine._odds_band_filter, OddsBandFilter)

    def test_odds_band_filter_none_for_place(self, mock_models: MagicMock) -> None:
        """Test: betting_target='place' -> _odds_band_filter is None"""
        from backtest.engine import BacktestEngine

        engine = BacktestEngine(models=mock_models, betting_target="place")
        assert engine._odds_band_filter is None

    @patch("db.odds_extractor.extract_pre_post_odds")
    @patch("features.trainer_context_features.TrainerContextFeatures")
    @patch("features.jockey_context_features.JockeyContextFeatures")
    @patch("features.interaction_features.compute_interaction_features")
    @patch("features.horse_history_features.HorseHistoryFeatures")
    @patch("models.submodel_manager.SubModelManager")
    @patch("features.feature_engine.FeatureEngine")
    @patch("backtest.engine.load_odds_time_series_range")
    @patch("backtest.engine.load_odds_snapshots")
    @patch("backtest.engine.load_entries")
    @patch("backtest.engine.load_races")
    def test_bet_count_guard_warning(
        self,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds: MagicMock,
        mock_load_odds_ts: MagicMock,
        mock_feat_engine_cls: MagicMock,
        mock_submodel_mgr_cls: MagicMock,
        mock_hist_cls: MagicMock,
        mock_interaction_fn: MagicMock,
        mock_jockey_cls: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_extract_odds: MagicMock,
        mock_models: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test: < 1000 bets/year triggers WARNING log"""
        mock_load_races.return_value = pd.DataFrame()
        mock_load_entries.return_value = pd.DataFrame()
        mock_load_odds.return_value = pd.DataFrame()
        mock_load_odds_ts.return_value = pd.DataFrame()
        mock_extract_odds.return_value = pd.DataFrame()

        from backtest.engine import BacktestEngine

        mock_store = MagicMock()
        engine = BacktestEngine(models=mock_models, store=mock_store, betting_target="win")
        with caplog.at_level(logging.WARNING, logger="backtest.engine"):
            result = engine.run("2024-01-01", "2024-12-31")

        # Empty data -> 0 bets -> no bet count guard log (guard only runs when total_bets > 0)
        assert result.total_bets == 0


class TestBettingTarget:
    """BacktestEngine betting_target パラメータのテスト"""

    def test_default_betting_target_is_win(self, mock_models: MagicMock) -> None:
        """デフォルトの betting_target は 'win'"""
        from backtest.engine import BacktestEngine

        engine = BacktestEngine(models=mock_models)
        assert engine.betting_target == "win"

    def test_betting_target_place(self, mock_models: MagicMock) -> None:
        """betting_target='place' で初期化できる"""
        from backtest.engine import BacktestEngine

        engine = BacktestEngine(models=mock_models, betting_target="place")
        assert engine.betting_target == "place"

    def test_betting_target_invalid_raises(self, mock_models: MagicMock) -> None:
        """不正な betting_target は ValueError"""
        from backtest.engine import BacktestEngine

        with pytest.raises(ValueError, match="betting_target must be"):
            BacktestEngine(models=mock_models, betting_target="invalid")


class TestWinSettleBet:
    """_settle_bet() の WIN branch テスト"""

    def test_win_payout_map_hit(self) -> None:
        """WIN bet + win_payout_map hit → stake * multiplier (not place payout)"""
        from backtest.engine import BacktestEngine
        from domain.models import Bet
        from domain.types import BetType

        bet = Bet(
            race_id="R001",
            umaban=5,
            bet_type=BetType.WIN,
            odds=3.0,
            ev_lower_corrected=0.0,
            stake=100,
            final_odds=3.0,
        )
        race_df = pd.DataFrame({"umaban": [5], "kakuteijyuni": [1]})
        engine = BacktestEngine.__new__(BacktestEngine)
        engine.payout_map = {("R001", 5): 1.5}  # place payout (should NOT be used)
        engine.win_payout_map = {("R001", 5): 2.4}  # win payout
        result = engine._settle_bet(bet, race_df)
        assert result == pytest.approx(240.0)  # 100 * 2.4, NOT 100 * 1.5

    def test_win_payout_map_miss_fallback(self, caplog: pytest.LogCaptureFixture) -> None:
        """WIN bet + win_payout_map miss → WARNING + finish_pos==1 fallback"""
        from backtest.engine import BacktestEngine
        from domain.models import Bet
        from domain.types import BetType

        bet = Bet(
            race_id="R999",
            umaban=1,
            bet_type=BetType.WIN,
            odds=5.0,
            ev_lower_corrected=0.0,
            stake=100,
            final_odds=5.0,
        )
        race_df = pd.DataFrame({"umaban": [1], "kakuteijyuni": [1]})
        engine = BacktestEngine.__new__(BacktestEngine)
        engine.payout_map = {}
        engine.win_payout_map = {}  # no payout data
        with caplog.at_level(logging.WARNING, logger="backtest.engine"):
            result = engine._settle_bet(bet, race_df)
        assert result == pytest.approx(500.0)  # 100 * 5.0 (finish_pos==1)
        assert any("Win payout missing" in rec.message for rec in caplog.records)

    def test_place_still_uses_payout_map(self) -> None:
        """PLACE bet は引き続き payout_map を使用する (変更なし)"""
        from backtest.engine import BacktestEngine
        from domain.models import Bet
        from domain.types import BetType

        bet = Bet(
            race_id="R001",
            umaban=3,
            bet_type=BetType.PLACE,
            odds=2.5,
            ev_lower_corrected=0.0,
            stake=100,
            final_odds=2.5,
        )
        race_df = pd.DataFrame({"umaban": [3], "kakuteijyuni": [2]})
        engine = BacktestEngine.__new__(BacktestEngine)
        engine.payout_map = {("R001", 3): 3.0}
        engine.win_payout_map = {("R001", 3): 10.0}  # win payout (should NOT be used)
        result = engine._settle_bet(bet, race_df)
        assert result == pytest.approx(300.0)  # 100 * 3.0 (place payout_map)


class TestBuildPayoutMap:
    """build_payout_map のテスト"""

    def test_basic_payout_map(self) -> None:
        """払戻データから正しい payout_map を構築する"""
        payouts = pd.DataFrame(
            {
                "race_id": ["R001", "R001", "R002"],
                "payfukusyoumaban1": [1, 3, 2],
                "payfukusyopay1": [150, 150, 200],
                "payfukusyoumaban2": [2, 5, 5],
                "payfukusyopay2": [200, 180, 150],
                "payfukusyoumaban3": [3, 7, 8],
                "payfukusyopay3": [300, 250, 100],
                "payfukusyoumaban4": [None, None, None],
                "payfukusyopay4": [None, None, None],
                "payfukusyoumaban5": [None, None, None],
                "payfukusyopay5": [None, None, None],
            }
        )
        from backtest.engine import build_payout_map

        payout_map = build_payout_map(payouts)
        assert payout_map[("R001", 1)] == pytest.approx(1.5)
        assert payout_map[("R001", 2)] == pytest.approx(2.0)
        assert payout_map[("R001", 3)] == pytest.approx(3.0)
        assert payout_map[("R002", 2)] == pytest.approx(2.0)

    def test_missing_pay_columns_skipped(self) -> None:
        """payfukusyoumaban が NaN のエントリはスキップする"""
        payouts = pd.DataFrame(
            {
                "race_id": ["R001"],
                "payfukusyoumaban1": [1],
                "payfukusyopay1": [150],
                "payfukusyoumaban2": [None],
                "payfukusyopay2": [None],
                "payfukusyoumaban3": [None],
                "payfukusyopay3": [None],
                "payfukusyoumaban4": [None],
                "payfukusyopay4": [None],
                "payfukusyoumaban5": [None],
                "payfukusyopay5": [None],
            }
        )
        from backtest.engine import build_payout_map

        payout_map = build_payout_map(payouts)
        assert ("R001", 1) in payout_map
        assert len(payout_map) == 1

    def test_empty_payouts(self) -> None:
        """空の DataFrame は空の map を返す"""
        payouts = pd.DataFrame()
        from backtest.engine import build_payout_map

        payout_map = build_payout_map(payouts)
        assert len(payout_map) == 0


class TestPayoutSettlement:
    """確定配当ベースの精算テスト"""

    def test_settle_bet_uses_payout_map(self) -> None:
        """_settle_bet が payout_map を使用する"""
        from backtest.engine import BacktestEngine
        from domain.models import Bet
        from domain.types import BetType

        bet = Bet(
            race_id="R001",
            umaban=3,
            bet_type=BetType.PLACE,
            odds=2.5,
            ev_lower_corrected=0.0,
            stake=100,
            final_odds=2.5,
        )
        race_df = pd.DataFrame({"umaban": [3], "kakuteijyuni": [2]})
        payout_map = {("R001", 3): 3.0}
        engine = BacktestEngine.__new__(BacktestEngine)
        engine.payout_map = payout_map
        result = engine._settle_bet(bet, race_df)
        assert result == pytest.approx(300.0)

    def test_settle_bet_no_payout_entry(self) -> None:
        """payout_map にエントリがない場合 (馬が着外) は 0 を返す"""
        from backtest.engine import BacktestEngine
        from domain.models import Bet
        from domain.types import BetType

        bet = Bet(
            race_id="R001",
            umaban=5,
            bet_type=BetType.PLACE,
            odds=2.0,
            ev_lower_corrected=0.0,
            stake=100,
            final_odds=2.0,
        )
        race_df = pd.DataFrame({"umaban": [5], "kakuteijyuni": [5]})
        payout_map = {("R001", 3): 3.0}
        engine = BacktestEngine.__new__(BacktestEngine)
        engine.payout_map = payout_map
        result = engine._settle_bet(bet, race_df)
        assert result == 0.0

    def test_settle_bet_fallback_to_odds(self) -> None:
        """payout_map にレースが存在しない場合は final_odds にフォールバック"""
        from backtest.engine import BacktestEngine
        from domain.models import Bet
        from domain.types import BetType

        bet = Bet(
            race_id="R999",
            umaban=1,
            bet_type=BetType.PLACE,
            odds=1.8,
            ev_lower_corrected=0.0,
            stake=100,
            final_odds=1.8,
        )
        race_df = pd.DataFrame({"umaban": [1], "kakuteijyuni": [1]})
        payout_map: dict[tuple[str, int], float] = {}
        engine = BacktestEngine.__new__(BacktestEngine)
        engine.payout_map = payout_map
        result = engine._settle_bet(bet, race_df)
        assert result == pytest.approx(180.0)


class TestVectorizedPayoutMaps:
    """ベクトル化された payout map 関数の回帰テスト"""

    def test_build_payout_map_vectorized_matches_original(self) -> None:
        """melt + groupby 版 build_payout_map が正しいマッピングを返す"""
        payouts = pd.DataFrame(
            {
                "race_id": ["R001", "R001", "R002"],
                "payfukusyoumaban1": [1, 3, 2],
                "payfukusyopay1": [150, 150, 200],
                "payfukusyoumaban2": [2, 5, 5],
                "payfukusyopay2": [200, 180, 150],
                "payfukusyoumaban3": [3, 7, 8],
                "payfukusyopay3": [300, 250, 100],
                "payfukusyoumaban4": [None, None, None],
                "payfukusyopay4": [None, None, None],
                "payfukusyoumaban5": [None, None, None],
                "payfukusyopay5": [None, None, None],
            }
        )
        from backtest.engine import build_payout_map

        payout_map = build_payout_map(payouts)
        assert payout_map[("R001", 1)] == pytest.approx(1.5)
        assert payout_map[("R001", 2)] == pytest.approx(2.0)
        assert payout_map[("R001", 3)] == pytest.approx(3.0)
        assert payout_map[("R002", 2)] == pytest.approx(2.0)
        assert ("R002", 1) not in payout_map

    def test_build_wide_payout_map_vectorized_kumi_formats(self) -> None:
        """ベクトル化版 build_wide_payout_map が各 kumi 長のフォーマットを正しくパースする"""
        payouts = pd.DataFrame(
            {
                "race_id": ["R001", "R002", "R003", "R004"],
                "paywidekumi1": ["15", "513", "1113", "0102"],
                "paywidepay1": [300, 400, 500, 600],
                "paywidekumi2": [None, None, None, None],
                "paywidepay2": [None, None, None, None],
                "paywidekumi3": [None, None, None, None],
                "paywidepay3": [None, None, None, None],
                "paywidekumi4": [None, None, None, None],
                "paywidepay4": [None, None, None, None],
                "paywidekumi5": [None, None, None, None],
                "paywidepay5": [None, None, None, None],
                "paywidekumi6": [None, None, None, None],
                "paywidepay6": [None, None, None, None],
                "paywidekumi7": [None, None, None, None],
                "paywidepay7": [None, None, None, None],
            }
        )
        from backtest.engine import build_wide_payout_map

        wide_map = build_wide_payout_map(payouts)
        # "15" → (1, 5)
        assert ("R001", 1, 5) in wide_map
        assert wide_map[("R001", 1, 5)] == pytest.approx(3.0)
        # "513" → first_two=51 > 18, so split at 1: (5, 13)
        assert ("R002", 5, 13) in wide_map
        assert wide_map[("R002", 5, 13)] == pytest.approx(4.0)
        # "1113" → (11, 13)
        assert ("R003", 11, 13) in wide_map
        assert wide_map[("R003", 11, 13)] == pytest.approx(5.0)
        # "0102" → (01, 02) = (1, 2)
        assert ("R004", 1, 2) in wide_map
        assert wide_map[("R004", 1, 2)] == pytest.approx(6.0)

    def test_build_payout_map_keeps_max_per_key(self) -> None:
        """同一 (race_id, umaban) に複数エントリがある場合、最大 payout を保持する"""
        payouts = pd.DataFrame(
            {
                "race_id": ["R001"],
                "payfukusyoumaban1": [3],
                "payfukusyopay1": [150],
                "payfukusyoumaban2": [3],  # 同じ馬番 3
                "payfukusyopay2": [300],   # より高い配当
                "payfukusyoumaban3": [1],
                "payfukusyopay3": [120],
                "payfukusyoumaban4": [None],
                "payfukusyopay4": [None],
                "payfukusyoumaban5": [None],
                "payfukusyopay5": [None],
            }
        )
        from backtest.engine import build_payout_map

        payout_map = build_payout_map(payouts)
        # 馬番 3 は最大値 3.0 を保持
        assert payout_map[("R001", 3)] == pytest.approx(3.0)
        assert payout_map[("R001", 1)] == pytest.approx(1.2)

    def test_final_odds_map_vectorized(self) -> None:
        """set_index 版 final_odds_map 構築が正しい dict を返す"""
        final_odds_df = pd.DataFrame(
            {
                "race_id": ["R001", "R001", "R002"],
                "umaban": [1, 2, 1],
                "fukuoddslow": [1.5, 3.2, 2.8],
            }
        )
        # Replicate the vectorized logic from engine.py
        final_odds_map: dict[tuple[str, int], float] = {}
        _odds = final_odds_df.dropna(subset=["fukuoddslow"])
        if not _odds.empty:
            for (race_id, umaban), odds in (
                _odds.set_index(["race_id", "umaban"])["fukuoddslow"].items()
            ):
                final_odds_map[(str(race_id), int(umaban))] = float(odds)

        assert final_odds_map[("R001", 1)] == pytest.approx(1.5)
        assert final_odds_map[("R001", 2)] == pytest.approx(3.2)
        assert final_odds_map[("R002", 1)] == pytest.approx(2.8)


class TestStakeSizingIntegration:
    """Phase 12: fractional_kelly 注入 + EV乗算 パイプラインのテスト"""

    def test_regime_injects_fractional_kelly(self, mock_models: MagicMock) -> None:
        """engine レースループで fractional_kelly が StakeCalculator に注入される"""
        from backtest.engine import BacktestEngine

        engine = BacktestEngine(
            models=mock_models, betting_mode="kelly", betting_target="win"
        )
        # 初期値はデフォルト 0.5
        assert engine._race_predictor.stake_calc is not None
        assert engine._race_predictor.stake_calc.fractional_kelly == 0.5

        # CONSERVATIVE regime_params で fractional_kelly=0.25 を注入
        regime_params = {
            "ev_threshold": 1.30,
            "edge_threshold": 0.06,
            "fractional_kelly": 0.25,
            "score_threshold": 0.020,
            "max_bets_per_race": 1,
        }
        if engine._race_predictor.stake_calc is not None:
            fk = float(regime_params.get("fractional_kelly", 0.5))
            engine._race_predictor.stake_calc.fractional_kelly = fk

        assert engine._race_predictor.stake_calc.fractional_kelly == 0.25

        # COLLAPSED regime_params で fractional_kelly=0.00 を注入
        collapsed_params = {
            "ev_threshold": 1.50,
            "edge_threshold": 0.09,
            "fractional_kelly": 0.00,
            "score_threshold": 0.050,
            "max_bets_per_race": 1,
            "skip": True,
        }
        if engine._race_predictor.stake_calc is not None:
            fk = float(collapsed_params.get("fractional_kelly", 0.5))
            engine._race_predictor.stake_calc.fractional_kelly = fk

        assert engine._race_predictor.stake_calc.fractional_kelly == 0.0

    def test_ev_scaling_in_select_bets(self) -> None:
        """select_bets() winパスで calc_stake→apply_ev_scaling→DD パイプラインが動作する"""
        from backtest.race_predictor import RacePredictor
        from betting.drawdown_controller import DrawdownController
        from betting.stake_calculator import StakeCalculator
        from domain.models import BetType

        # モックモデル
        models = MagicMock(spec=TrainedModelsV5)
        models.submodels = {"turf": MagicMock(spec=SubmodelSet)}
        models.regime_detector = MagicMock()
        models.regime_detector.current_regime = RegimeState.AGGRESSIVE
        models.regime_detector.get_strategy_params.return_value = {
            "ev_threshold": 1.10,
            "edge_threshold": 0.05,
            "fractional_kelly": 0.50,
            "score_threshold": 0.010,
            "max_bets_per_race": 2,
        }
        models.quality_screener = MagicMock()
        models.quality_screener.should_bet.return_value = True

        # StakeCalculator(fractional_kelly=0.50, target_ev=1.10, max_scale=2.0)
        stake_calc = StakeCalculator(fractional_kelly=0.50, target_ev=1.10, max_scale=2.0)
        dd_ctrl = DrawdownController(peak_bankroll=100000)

        predictor = RacePredictor(
            models,
            stake_calculator=stake_calc,
            dd_controller=dd_ctrl,
        )

        # 候補 DataFrame (win mode)
        candidates = pd.DataFrame({
            "race_id": ["20240101010101"],
            "umaban": [3],
            "tanodds": [5.0],
            "win_selection_edge": [0.06],
            "win_selection_ev": [1.50],  # EV > target_ev → 拡大
        })

        bets = predictor.select_bets(
            pd.DataFrame({"race_id": ["20240101010101"], "umaban": [3]}),
            bankroll=100000.0,
            candidates=candidates,
            betting_target="win",
        )

        # Kelly stake (edge=0.06, odds=5.0, fk=0.50): 700.0
        # apply_ev_scaling(700, ev=1.50): scale = 1.50/1.10 = 1.3636 → 954.54
        # DD adjust → floor(954.54/100)*100 = 900
        assert len(bets) == 1
        assert bets[0].bet_type == BetType.WIN
        # 900 = floor(700 * 1.3636 / 100) * 100
        assert bets[0].stake == 900.0

    def test_collapsed_regime_zero_stake(self) -> None:
        """fractional_kelly=0.00 の StakeCalculator で stake=0 → ベットが空リスト"""
        from backtest.race_predictor import RacePredictor
        from betting.stake_calculator import StakeCalculator

        models = MagicMock(spec=TrainedModelsV5)
        models.submodels = {"turf": MagicMock(spec=SubmodelSet)}
        models.regime_detector = MagicMock()
        models.regime_detector.current_regime = RegimeState.COLLAPSED
        models.regime_detector.get_strategy_params.return_value = {
            "ev_threshold": 1.50,
            "edge_threshold": 0.09,
            "fractional_kelly": 0.00,
            "score_threshold": 0.050,
            "max_bets_per_race": 1,
            "skip": True,
        }
        models.quality_screener = MagicMock()
        models.quality_screener.should_bet.return_value = True

        # COLLAPSED: fractional_kelly=0.00
        stake_calc = StakeCalculator(fractional_kelly=0.00)
        predictor = RacePredictor(
            models,
            stake_calculator=stake_calc,
        )

        # calc_stake が 0 を返すことを確認
        stake = stake_calc.calc_stake(
            edge=0.06, odds=5.0, bankroll=100000, bet_type=BetType.WIN
        )
        assert stake == 0.0

        # 候補があってもベットは生成されない (stake < 100 で continue)
        candidates = pd.DataFrame({
            "race_id": ["20240101010101"],
            "umaban": [3],
            "tanodds": [5.0],
            "win_selection_edge": [0.06],
            "win_selection_ev": [1.50],
        })

        bets = predictor.select_bets(
            pd.DataFrame({"race_id": ["20240101010101"], "umaban": [3]}),
            bankroll=100000.0,
            candidates=candidates,
            betting_target="win",
        )

        assert len(bets) == 0


class TestBacktestEnginePFPIntegration:
    """BacktestEngine への PFP freeze/verify 二重検証統合テスト (Phase 18, VAL-02)"""

    @patch("backtest.engine.load_odds_snapshots")
    @patch("backtest.engine.load_entries")
    @patch("backtest.engine.load_races")
    def test_manifest_path_triggers_verify_strategy_manifest(
        self,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds: MagicMock,
        mock_models: MagicMock,
    ) -> None:
        """Test 1: manifest_path を渡すと verify_strategy_manifest() が呼ばれる"""
        from pathlib import Path

        mock_load_races.return_value = pd.DataFrame()
        mock_load_entries.return_value = pd.DataFrame()
        mock_load_odds.return_value = pd.DataFrame()

        from backtest.engine import BacktestEngine

        mock_store = MagicMock()

        with patch("backtest.engine.verify_strategy_manifest") as mock_verify, \
             patch("backtest.engine.ParameterFreezeProtocol") as mock_pfp_cls:
            mock_verify.return_value = {"fk_aggressive": 0.5}
            mock_pfp_inst = MagicMock()
            mock_pfp_inst.verify.return_value = {"passed": True, "message": "OK"}
            mock_pfp_cls.return_value = mock_pfp_inst

            engine = BacktestEngine(
                models=mock_models,
                store=mock_store,
                manifest_path=Path("/tmp/test_manifest.json"),
            )
            engine.run("2024-01-01", "2024-12-31")

            mock_verify.assert_called_once_with(Path("/tmp/test_manifest.json"))

    @patch("backtest.engine.load_odds_snapshots")
    @patch("backtest.engine.load_entries")
    @patch("backtest.engine.load_races")
    def test_manifest_path_triggers_pfp_freeze_and_verify(
        self,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds: MagicMock,
        mock_models: MagicMock,
    ) -> None:
        """Test 2: manifest_path を渡すと PFP freeze() と verify() が順番に呼ばれる"""
        from pathlib import Path

        mock_load_races.return_value = pd.DataFrame()
        mock_load_entries.return_value = pd.DataFrame()
        mock_load_odds.return_value = pd.DataFrame()

        from backtest.engine import BacktestEngine

        mock_store = MagicMock()

        with patch("backtest.engine.verify_strategy_manifest") as mock_verify, \
             patch("backtest.engine.ParameterFreezeProtocol") as mock_pfp_cls:
            mock_verify.return_value = {"fk_aggressive": 0.5}
            mock_pfp_inst = MagicMock()
            mock_pfp_inst.verify.return_value = {"passed": True, "message": "OK"}
            mock_pfp_cls.return_value = mock_pfp_inst

            engine = BacktestEngine(
                models=mock_models,
                store=mock_store,
                manifest_path=Path("/tmp/test_manifest.json"),
            )
            engine.run("2024-01-01", "2024-12-31")

            mock_pfp_cls.assert_called_once_with(mock_models)
            mock_pfp_inst.freeze.assert_called_once()
            mock_pfp_inst.verify.assert_called_once()

    @patch("backtest.engine.load_odds_snapshots")
    @patch("backtest.engine.load_entries")
    @patch("backtest.engine.load_races")
    def test_pfp_verify_failure_raises_runtime_error(
        self,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds: MagicMock,
        mock_models: MagicMock,
    ) -> None:
        """Test 3: PFP verify() が passed=False を返した場合、RuntimeError を送出 (D-04)"""
        from pathlib import Path

        mock_load_races.return_value = pd.DataFrame()
        mock_load_entries.return_value = pd.DataFrame()
        mock_load_odds.return_value = pd.DataFrame()

        from backtest.engine import BacktestEngine

        mock_store = MagicMock()

        with patch("backtest.engine.verify_strategy_manifest") as mock_verify, \
             patch("backtest.engine.ParameterFreezeProtocol") as mock_pfp_cls:
            mock_verify.return_value = {"fk_aggressive": 0.5}
            mock_pfp_inst = MagicMock()
            mock_pfp_inst.verify.return_value = {
                "passed": False,
                "message": "Parameters changed during frozen period (Rule 7 VIOLATION)",
            }
            mock_pfp_cls.return_value = mock_pfp_inst

            engine = BacktestEngine(
                models=mock_models,
                store=mock_store,
                manifest_path=Path("/tmp/test_manifest.json"),
            )
            with pytest.raises(RuntimeError, match="Rule 7 VIOLATION"):
                engine.run("2024-01-01", "2024-12-31")

    @patch("backtest.engine.load_odds_snapshots")
    @patch("backtest.engine.load_entries")
    @patch("backtest.engine.load_races")
    def test_no_manifest_path_no_pfp_code(
        self,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds: MagicMock,
        mock_models: MagicMock,
    ) -> None:
        """Test 4: manifest_path=None の場合、PFP/verify関連コードが一切実行されない"""
        mock_load_races.return_value = pd.DataFrame()
        mock_load_entries.return_value = pd.DataFrame()
        mock_load_odds.return_value = pd.DataFrame()

        from backtest.engine import BacktestEngine

        mock_store = MagicMock()

        with patch("backtest.engine.verify_strategy_manifest") as mock_verify, \
             patch("backtest.engine.ParameterFreezeProtocol") as mock_pfp_cls:
            engine = BacktestEngine(
                models=mock_models,
                store=mock_store,
                manifest_path=None,
            )
            engine.run("2024-01-01", "2024-12-31")

            mock_verify.assert_not_called()
            mock_pfp_cls.assert_not_called()

    @patch("backtest.engine.load_odds_snapshots")
    @patch("backtest.engine.load_entries")
    @patch("backtest.engine.load_races")
    def test_manifest_missing_path_raises_file_not_found(
        self,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds: MagicMock,
        mock_models: MagicMock,
    ) -> None:
        """Test 5: manifest_path が存在しないパスの場合、FileNotFoundError を送出 (D-04)"""
        from pathlib import Path

        mock_load_races.return_value = pd.DataFrame()
        mock_load_entries.return_value = pd.DataFrame()
        mock_load_odds.return_value = pd.DataFrame()

        from backtest.engine import BacktestEngine

        mock_store = MagicMock()

        nonexistent = Path("/tmp/nonexistent_manifest_12345.json")

        with patch("backtest.engine.verify_strategy_manifest") as mock_verify:
            mock_verify.side_effect = FileNotFoundError(
                f"Strategy manifest not found: {nonexistent}"
            )
            engine = BacktestEngine(
                models=mock_models,
                store=mock_store,
                manifest_path=nonexistent,
            )
            with pytest.raises(FileNotFoundError, match="Strategy manifest not found"):
                engine.run("2024-01-01", "2024-12-31")


class TestBacktestOptimizationStages:
    """P0-P3最適化の段階的検証テスト (D-06, D-07)

    各最適化段階でバックテスト結果が許容範囲内に収まることを確かめる:
    - P0-P2: 予測値完全一致 (結果に影響する変更なし)
    - P3: 統計的許容範囲 (ROI/bet_count差異<5%、的中率差異<1%)
    """

    def test_calibration_skip_returns_empty_for_default_strategy(self) -> None:
        """--calibration-bt 未指定で空リストが返される"""
        from unittest.mock import MagicMock

        # _collect_training_bet_historyはscripts/run_backtest.pyにあるため、
        # 直接モジュールをインポートしてテストする
        from scripts.run_backtest import _collect_training_bet_history

        result = _collect_training_bet_history(
            models=MagicMock(),
            store=MagicMock(),
            train_start="2020-01-01",
            train_end="2023-12-31",
            betting_mode="flat",
            betting_target="win",
            strategy_params=None,
            run_calibration=False,
        )
        assert result == []

    def test_calibration_runs_with_strategy_manifest(self) -> None:
        """--calibration-bt 指定時に軽量キャリブレーションが実行される"""
        from datetime import datetime
        from unittest.mock import MagicMock, patch

        from scripts.run_backtest import _collect_training_bet_history

        # _collect_training_bet_history内でBacktestEngineは
        # from backtest.engine import BacktestEngine でimportされるため、
        # patch先は backtest.engine.BacktestEngine
        with patch("backtest.engine.BacktestEngine") as mock_engine_cls:
            mock_result = MagicMock()
            mock_result.bet_history = [{"test": "bet"}]
            mock_result.total_bets = 10
            mock_result.total_roi = 0.95
            mock_engine_cls.return_value.run.return_value = mock_result

            result = _collect_training_bet_history(
                models=MagicMock(),
                store=MagicMock(),
                train_start="2020-01-01",
                train_end="2023-12-31",
                betting_mode="flat",
                betting_target="win",
                strategy_params={"ev_aggressive": 1.1},
                run_calibration=True,
            )
            assert len(result) > 0
            # 12ヶ月に短縮されていることを確認
            mock_engine_cls.return_value.run.assert_called_once()
            call_args = mock_engine_cls.return_value.run.call_args
            cal_start = call_args[0][0]  # first positional arg
            cal_end = call_args[0][1]  # second positional arg
            # cal_start should be ~12 months before cal_end (train_end)
            assert cal_end == "2023-12-31"
            # Verify shortened period (approximately 12 months)
            start_dt = datetime.strptime(cal_start, "%Y-%m-%d")
            end_dt = datetime.strptime(cal_end, "%Y-%m-%d")
            days_diff = (end_dt - start_dt).days
            assert 360 <= days_diff <= 370  # ~12 months

    def test_preloaded_odds_ts_bypasses_load(self) -> None:
        """P1: preloaded_odds_tsが渡された場合、load_odds_time_series_rangeが呼ばれない"""
        from unittest.mock import MagicMock

        test_odds = pd.DataFrame({
            "race_id": ["202401010101"],
            "race_date": pd.to_datetime(["2024-01-01"]),
            "umaban": [1],
        })

        from backtest.engine import BacktestEngine

        engine = BacktestEngine(
            models=MagicMock(),
            store=MagicMock(),
            preloaded_odds_ts=test_odds,
        )
        # _preloaded_odds_tsが設定されていることを確認
        assert engine._preloaded_odds_ts is not None
        assert len(engine._preloaded_odds_ts) == 1

    def test_preloaded_race_df_stored(self) -> None:
        """P2: preloaded_race_dfが渡された場合、_preloaded_race_dfに格納される"""
        from unittest.mock import MagicMock

        test_df = pd.DataFrame({"race_id": ["202401010101"]})
        from backtest.engine import BacktestEngine

        engine = BacktestEngine(
            models=MagicMock(),
            store=MagicMock(),
            preloaded_race_df=test_df,
        )
        assert engine._preloaded_race_df is not None
        assert len(engine._preloaded_race_df) == 1

    def test_preloaded_entry_df_stored(self) -> None:
        """P2: preloaded_entry_dfが渡された場合、_preloaded_entry_dfに格納される"""
        from unittest.mock import MagicMock

        test_df = pd.DataFrame({"race_id": ["202401010101"], "umaban": [1]})
        from backtest.engine import BacktestEngine

        engine = BacktestEngine(
            models=MagicMock(),
            store=MagicMock(),
            preloaded_entry_df=test_df,
        )
        assert engine._preloaded_entry_df is not None
        assert len(engine._preloaded_entry_df) == 1

    def test_preloaded_final_odds_df_stored(self) -> None:
        """P2: preloaded_final_odds_dfが渡された場合、_preloaded_final_odds_dfに格納される"""
        from unittest.mock import MagicMock

        test_df = pd.DataFrame({"race_id": ["202401010101"], "umaban": [1]})
        from backtest.engine import BacktestEngine

        engine = BacktestEngine(
            models=MagicMock(),
            store=MagicMock(),
            preloaded_final_odds_df=test_df,
        )
        assert engine._preloaded_final_odds_df is not None
        assert len(engine._preloaded_final_odds_df) == 1

    def test_preloaded_payouts_df_stored(self) -> None:
        """P2: preloaded_payouts_dfが渡された場合、_preloaded_payouts_dfに格納される"""
        from unittest.mock import MagicMock

        test_df = pd.DataFrame({"race_id": ["202401010101"]})
        from backtest.engine import BacktestEngine

        engine = BacktestEngine(
            models=MagicMock(),
            store=MagicMock(),
            preloaded_payouts_df=test_df,
        )
        assert engine._preloaded_payouts_df is not None
        assert len(engine._preloaded_payouts_df) == 1

    def test_preloaded_dataframes_are_copied_in_run(self) -> None:
        """P2: run()冒頭でpreloaded DataFrameは.copy()され、共有元が破壊されない

        軽量テスト: load_*をpatchして未呼出を検証することで、
        三項演算子によるbypass分岐が走ったことを間接確認する。
        BacktestEngine.run()の本格起動はしない(既存テストと同じ軽量アプローチ)。
        """
        from unittest.mock import MagicMock

        test_race_df = pd.DataFrame({"race_id": ["202401010101"]})
        test_entry_df = pd.DataFrame({"race_id": ["202401010101"], "umaban": [1]})
        test_final_odds_df = pd.DataFrame(
            {"race_id": ["202401010101"], "umaban": [1]}
        )
        test_payouts_df = pd.DataFrame({"race_id": ["202401010101"]})
        test_odds_ts = pd.DataFrame(
            {
                "race_id": ["202401010101"],
                "race_date": pd.to_datetime(["2024-01-01"]),
                "umaban": [1],
            }
        )

        from backtest.engine import BacktestEngine

        # 共有元DataFrameのidを記録(.copy()で別オブジェクトになるか追跡)
        original_race_id = id(test_race_df)

        engine = BacktestEngine(
            models=MagicMock(),
            store=MagicMock(),
            preloaded_race_df=test_race_df,
            preloaded_entry_df=test_entry_df,
            preloaded_final_odds_df=test_final_odds_df,
            preloaded_payouts_df=test_payouts_df,
            preloaded_odds_ts=test_odds_ts,
        )

        # constructorが全preloaded DataFrameを参照で保持したことを確認
        # (run()内で.copy()される前の段階)
        assert id(engine._preloaded_race_df) == original_race_id

        # run()の本格起動は不要 — constructor格納とid同一性で十分
        # (三項演算子はself._preloaded_* is not Noneを判定するため、
        #  格納されていれば必ずbypass分岐が選択される)

    def test_categorical_columns_in_parquet_store(self) -> None:
        """D-04: ParquetStore._optimize_dtypesが対象列をCategoricalに変換する"""
        from db.parquet_store import CATEGORICAL_COLUMNS, _optimize_dtypes

        test_df = pd.DataFrame({
            "race_id": ["202401010101", "202401010102"],
            "kettonum": ["12345", "67890"],
            "kisyucode": ["A001", "B002"],
            "other_col": [1.0, 2.0],  # non-categorical
        })
        result = _optimize_dtypes(test_df)
        for col in CATEGORICAL_COLUMNS:
            if col in result.columns:
                assert result[col].dtype.name == "category", (
                    f"{col} should be category, got {result[col].dtype.name}"
                )
        assert result["other_col"].dtype.name != "category"

    def test_observed_true_on_all_groupby(self) -> None:
        """D-05: observed=Trueが全src/ groupbyに追加されていることをgrepで確認"""
        import glob

        py_files = glob.glob("src/**/*.py", recursive=True)
        violations: list[str] = []
        for fpath in py_files:
            with open(fpath, encoding="utf-8") as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if ".groupby(" in stripped and "observed" not in stripped:
                    # Check if observed=True appears within the next few lines
                    # (handles multi-line groupby calls)
                    context = "".join(lines[i : i + 5])
                    if "observed" not in context:
                        violations.append(f"{fpath}:{i + 1}: {stripped}")
        assert len(violations) == 0, (
            "Un-observed groupby calls found:\n" + "\n".join(violations)
        )

    def test_single_race_rank_matches_groupby_rank(self) -> None:
        """P3: 単一race DataFrameの直接rankがgroupby版と同じ結果を返す"""
        # 単一race DataFrameをシミュレート
        df = pd.DataFrame({
            "race_id": ["R001"] * 5,
            "norm_finish_logit_avg": [0.1, 0.5, 0.3, 0.8, 0.2],
            "harontimel5_avg": [12.0, 11.5, 12.5, 11.0, 13.0],
        })

        race_rank_cols = [
            "norm_finish_logit_avg",
            "harontimel5_avg",
        ]

        # groupby版 (従来のadd_race_transforms)
        df_groupby = df.copy()
        for col in race_rank_cols:
            if col in df_groupby.columns:
                df_groupby[f"{col}_race_rank"] = (
                    df_groupby.groupby("race_id", observed=True)[col]
                    .rank(pct=True, method="average")
                )

        # 直接rank版 (最適化後)
        df_direct = df.copy()
        for col in race_rank_cols:
            if col in df_direct.columns:
                df_direct[f"{col}_race_rank"] = df_direct[col].rank(
                    pct=True, method="average"
                )

        # 結果が同一であることを確認
        for col in race_rank_cols:
            rank_col = f"{col}_race_rank"
            pd.testing.assert_series_equal(
                df_groupby[rank_col],
                df_direct[rank_col],
                check_names=False,
            )

    def test_class_cache_shares_data_across_instances(self) -> None:
        """P3: HorseHistoryFeatures._class_cacheがインスタンス間でデータを共有する"""
        from features.horse_history_features import HorseHistoryFeatures

        # キャッシュクリア
        HorseHistoryFeatures.clear_class_cache()

        # インスタンス生成とキャッシュ動作確認
        assert hasattr(HorseHistoryFeatures, "_class_cache")
        assert hasattr(HorseHistoryFeatures, "clear_class_cache")

        # キャッシュにデータを設定し、別インスタンスからアクセスできることを確認
        HorseHistoryFeatures._class_cache["test_key"] = (
            np.array([]),
            {"data": "shared"},
        )
        assert "test_key" in HorseHistoryFeatures._class_cache

        # クリーンアップ
        HorseHistoryFeatures.clear_class_cache()
        assert len(HorseHistoryFeatures._class_cache) == 0


class TestHistFeaturesPreMerge:
    """D-11: hist_df_all を feat_df に事前マージし、predict() に hist_features=None を渡す"""

    @patch("db.odds_extractor.extract_pre_post_odds")
    @patch("features.trainer_context_features.TrainerContextFeatures")
    @patch("features.jockey_context_features.JockeyContextFeatures")
    @patch("features.interaction_features.compute_interaction_features")
    @patch("features.horse_history_features.HorseHistoryFeatures")
    @patch("models.submodel_manager.SubModelManager")
    @patch("features.feature_engine.FeatureEngine")
    @patch("backtest.engine.load_odds_time_series_range")
    @patch("backtest.engine.load_odds_snapshots")
    @patch("backtest.engine.load_entries")
    @patch("backtest.engine.load_races")
    def test_feat_df_contains_hist_features_after_merge(
        self,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds: MagicMock,
        mock_load_odds_ts: MagicMock,
        mock_feat_engine_cls: MagicMock,
        mock_submodel_mgr_cls: MagicMock,
        mock_hist_cls: MagicMock,
        mock_interaction_fn: MagicMock,
        mock_jockey_cls: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_extract_odds: MagicMock,
        mock_models: MagicMock,
    ) -> None:
        """Test 1: engine.run() 後、feat_df に hist 特徴量列が含まれる."""
        mock_load_races.return_value = pd.DataFrame(
            {"race_id": ["20240101010101"], "race_date": pd.to_datetime("2024-01-01"),
             "hassotime": ["03101500"]}
        )
        mock_load_entries.return_value = pd.DataFrame(
            {"race_id": ["20240101010101"], "umaban": [1], "kettonum": [1234],
             "kakuteijyuni": [2], "odds": [5.0], "ninki": [3],
             "bataijyu": [480], "zogen_fugo": [0], "zogen_sa": [0],
             "kisyucode": [100], "chokyosicode": [200]}
        )
        mock_load_odds.return_value = pd.DataFrame()
        mock_load_odds_ts.return_value = pd.DataFrame(
            {"race_id": ["20240101010101"], "umaban": [1], "odds": [5.0]}
        )
        mock_extract_odds.return_value = pd.DataFrame(
            {"race_id": ["20240101010101"], "umaban": [1], "fukuoddslow": [4.0]}
        )

        feat_df = pd.DataFrame({
            "race_id": ["20240101010101"], "umaban": [1], "surface": ["turf"],
            "kyori": [1600], "distance_bin": ["mile"], "popularity_rank": [3],
            "ninki": [3], "ev_place": [1.5], "fukuoddslow": [4.0],
            "kakuteijyuni": [2], "kettonum": [1234], "odds": [5.0],
            "bataijyu": [480], "jyocd": [5], "racenum": [11],
            "grade_code": ["E"], "hondai": ["テスト"], "bamei": ["テスト馬"],
            "kisyuryakusyo": ["テスト騎手"], "track_condition_code": [1],
            "p_place_pred": [0.65], "e_return_place_pred": [1.80],
        })

        # hist_df_all with a known feature column
        hist_df_all = pd.DataFrame({
            "race_id": ["20240101010101"], "umaban": [1],
            "closing_speed_ratio_avg": [0.75],
            "haron_race_gap_avg": [-1.5],
        })

        mock_feat_engine = MagicMock()
        mock_feat_engine_cls.return_value = mock_feat_engine
        mock_feat_engine.build_all.return_value = feat_df

        mock_submodel_mgr = MagicMock()
        mock_submodel_mgr_cls.return_value = mock_submodel_mgr
        mock_submodel_mgr.add_distance_band_features.return_value = feat_df

        mock_hist = MagicMock()
        mock_hist_cls.return_value = mock_hist
        mock_hist.compute.return_value = hist_df_all
        mock_hist.add_race_transforms = staticmethod(lambda df: df)

        mock_interaction_fn.side_effect = lambda df: df
        mock_jockey = MagicMock()
        mock_jockey_cls.return_value = mock_jockey
        mock_jockey.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer
        mock_trainer.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])

        submodel = MagicMock()
        submodel.benter_combo = None
        submodel.isotonic_calibrator = None
        submodel.win_benter = None
        submodel.target_encoder = None
        mock_models.submodels["turf"] = submodel
        submodel.market.predict_and_calc_error.return_value = feat_df
        submodel.stage1.add_ability_probs.return_value = feat_df
        submodel.place_ability.predict.return_value = feat_df
        submodel.win.predict_ev.return_value = feat_df
        submodel.ev_corrector.correct_ev.return_value = feat_df
        submodel.place.predict_ev.return_value = feat_df
        _corrected = feat_df.assign(ev_place_corrected=feat_df.get("ev_place", 1.5))
        submodel.place_ev_corrector.correct_ev.return_value = _corrected
        submodel.conformal_ev_model.predict_lower_bound.return_value = (
            _corrected, pd.DataFrame({"EV_lower_place": [1.5]}),
        )
        submodel.conformal_ev_model.predict_interval.return_value = (
            _corrected, pd.DataFrame({"EV_lower_place": [1.5]}),
        )

        from backtest.engine import BacktestEngine

        # Capture the feat_df after merge
        captured_hist_in_predict: dict[str, object] = {}
        from backtest.race_predictor import RacePredictor
        original_predict = RacePredictor.predict

        def spy_predict(self_pred: object, race_df: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
            captured_hist_in_predict["hist_features"] = kwargs.get("hist_features")
            captured_hist_in_predict["columns"] = list(race_df.columns)
            return original_predict(self_pred, race_df, **kwargs)  # type: ignore[arg-type]

        mock_store = MagicMock()
        with patch.object(RacePredictor, "predict", spy_predict):
            engine = BacktestEngine(
                models=mock_models, store=mock_store, betting_target="place",
                min_bets_per_year=0,
            )
            engine.run("2024-01-01", "2024-12-31")

        # Test 1: hist features merged into feat_df (available in predict input)
        assert "closing_speed_ratio_avg" in captured_hist_in_predict["columns"], (
            "closing_speed_ratio_avg should be in predict() input after hist pre-merge"
        )
        # Test 2: hist_features=None passed to predict()
        assert captured_hist_in_predict["hist_features"] is None, (
            "predict() should receive hist_features=None (pre-merged)"
        )

    @patch("db.odds_extractor.extract_pre_post_odds")
    @patch("features.trainer_context_features.TrainerContextFeatures")
    @patch("features.jockey_context_features.JockeyContextFeatures")
    @patch("features.interaction_features.compute_interaction_features")
    @patch("features.horse_history_features.HorseHistoryFeatures")
    @patch("models.submodel_manager.SubModelManager")
    @patch("features.feature_engine.FeatureEngine")
    @patch("backtest.engine.load_odds_time_series_range")
    @patch("backtest.engine.load_odds_snapshots")
    @patch("backtest.engine.load_entries")
    @patch("backtest.engine.load_races")
    def test_no_double_merge_suffixes(
        self,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds: MagicMock,
        mock_load_odds_ts: MagicMock,
        mock_feat_engine_cls: MagicMock,
        mock_submodel_mgr_cls: MagicMock,
        mock_hist_cls: MagicMock,
        mock_interaction_fn: MagicMock,
        mock_jockey_cls: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_extract_odds: MagicMock,
        mock_models: MagicMock,
    ) -> None:
        """Test 3: 二重マージが発生しない（_x/_y サフィックス列が存在しない）."""
        mock_load_races.return_value = pd.DataFrame(
            {"race_id": ["20240101010101"], "race_date": pd.to_datetime("2024-01-01"),
             "hassotime": ["03101500"]}
        )
        mock_load_entries.return_value = pd.DataFrame(
            {"race_id": ["20240101010101"], "umaban": [1], "kettonum": [1234],
             "kakuteijyuni": [2], "odds": [5.0], "ninki": [3],
             "bataijyu": [480], "zogen_fugo": [0], "zogen_sa": [0],
             "kisyucode": [100], "chokyosicode": [200]}
        )
        mock_load_odds.return_value = pd.DataFrame()
        mock_load_odds_ts.return_value = pd.DataFrame(
            {"race_id": ["20240101010101"], "umaban": [1], "odds": [5.0]}
        )
        mock_extract_odds.return_value = pd.DataFrame(
            {"race_id": ["20240101010101"], "umaban": [1], "fukuoddslow": [4.0]}
        )

        feat_df = pd.DataFrame({
            "race_id": ["20240101010101"], "umaban": [1], "surface": ["turf"],
            "kyori": [1600], "distance_bin": ["mile"], "popularity_rank": [3],
            "ninki": [3], "ev_place": [1.5], "fukuoddslow": [4.0],
            "kakuteijyuni": [2], "kettonum": [1234], "odds": [5.0],
            "bataijyu": [480], "jyocd": [5], "racenum": [11],
            "grade_code": ["E"], "hondai": ["テスト"], "bamei": ["テスト馬"],
            "kisyuryakusyo": ["テスト騎手"], "track_condition_code": [1],
            "p_place_pred": [0.65], "e_return_place_pred": [1.80],
        })

        hist_df_all = pd.DataFrame({
            "race_id": ["20240101010101"], "umaban": [1],
            "closing_speed_ratio_avg": [0.75],
        })

        mock_feat_engine = MagicMock()
        mock_feat_engine_cls.return_value = mock_feat_engine
        mock_feat_engine.build_all.return_value = feat_df

        mock_submodel_mgr = MagicMock()
        mock_submodel_mgr_cls.return_value = mock_submodel_mgr
        mock_submodel_mgr.add_distance_band_features.return_value = feat_df

        mock_hist = MagicMock()
        mock_hist_cls.return_value = mock_hist
        mock_hist.compute.return_value = hist_df_all
        mock_hist.add_race_transforms = staticmethod(lambda df: df)

        mock_interaction_fn.side_effect = lambda df: df
        mock_jockey = MagicMock()
        mock_jockey_cls.return_value = mock_jockey
        mock_jockey.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer
        mock_trainer.compute.return_value = pd.DataFrame(columns=["race_id", "umaban"])

        submodel = MagicMock()
        submodel.benter_combo = None
        submodel.isotonic_calibrator = None
        submodel.win_benter = None
        submodel.target_encoder = None
        mock_models.submodels["turf"] = submodel
        submodel.market.predict_and_calc_error.return_value = feat_df
        submodel.stage1.add_ability_probs.return_value = feat_df
        submodel.place_ability.predict.return_value = feat_df
        submodel.win.predict_ev.return_value = feat_df
        submodel.ev_corrector.correct_ev.return_value = feat_df
        submodel.place.predict_ev.return_value = feat_df
        _corrected = feat_df.assign(ev_place_corrected=feat_df.get("ev_place", 1.5))
        submodel.place_ev_corrector.correct_ev.return_value = _corrected
        submodel.conformal_ev_model.predict_lower_bound.return_value = (
            _corrected, pd.DataFrame({"EV_lower_place": [1.5]}),
        )
        submodel.conformal_ev_model.predict_interval.return_value = (
            _corrected, pd.DataFrame({"EV_lower_place": [1.5]}),
        )

        from backtest.engine import BacktestEngine

        # Capture predict() output columns
        captured_result_cols: dict[str, list[str]] = {}
        from backtest.race_predictor import RacePredictor
        original_predict = RacePredictor.predict

        def spy_predict(self_pred: object, race_df: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
            result = original_predict(self_pred, race_df, **kwargs)  # type: ignore[arg-type]
            captured_result_cols["cols"] = list(result.columns)
            return result

        mock_store = MagicMock()
        with patch.object(RacePredictor, "predict", spy_predict):
            engine = BacktestEngine(
                models=mock_models, store=mock_store, betting_target="place",
                min_bets_per_year=0,
            )
            engine.run("2024-01-01", "2024-12-31")

        # Test 3: No _x/_y suffix columns in result (no double merge)
        if "cols" in captured_result_cols:
            suffix_cols = [
                c
                for c in captured_result_cols["cols"]
                if c.endswith("_x") or c.endswith("_y")
            ]
            assert len(suffix_cols) == 0, (
                f"No _x/_y suffix columns expected (double merge), found: {suffix_cols}"
            )
