"""BacktestEngine のテスト"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from domain.models import SubmodelSet, TrainedModelsV5
from domain.types import RegimeState


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
            monthly_returns={},
            bet_history=[],
        )
        assert result.total_roi == 1.05
        assert result.total_return - result.total_stake == 5000

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
        mock_models.submodels["turf"] = submodel
        submodel.market.predict_and_calc_error.return_value = feat_df
        submodel.stage1.add_ability_probs.return_value = feat_df
        submodel.place_ability.predict.return_value = feat_df
        submodel.win.predict_ev.return_value = feat_df
        submodel.ev_corrector.correct_ev.return_value = feat_df
        submodel.place.predict_ev.return_value = feat_df
        _corrected = feat_df.assign(ev_place_corrected=feat_df.get("ev_place", 1.5))
        submodel.place_ev_corrector.correct_ev.return_value = _corrected
        submodel.confidence.predict_lower_bound.return_value = (
            _corrected,
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )
        submodel.confidence.predict_interval.return_value = (
            _corrected,
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )

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
            engine = BacktestEngine(models=mock_models, store=mock_store, betting_target="place")
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
        mock_models.submodels["turf"] = submodel
        submodel.market.predict_and_calc_error.return_value = feat_df
        submodel.stage1.add_ability_probs.return_value = feat_df
        submodel.place_ability.predict.return_value = feat_df
        submodel.win.predict_ev.return_value = feat_df
        submodel.ev_corrector.correct_ev.return_value = feat_df
        submodel.place.predict_ev.return_value = feat_df
        _corrected = feat_df.assign(ev_place_corrected=feat_df.get("ev_place", 1.5))
        submodel.place_ev_corrector.correct_ev.return_value = _corrected
        submodel.confidence.predict_lower_bound.return_value = (
            _corrected,
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )
        submodel.confidence.predict_interval.return_value = (
            _corrected,
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )

        # --- run engine ---
        from backtest.engine import BacktestEngine

        mock_store = MagicMock()
        engine = BacktestEngine(models=mock_models, store=mock_store, betting_target="place")
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

        expected_cols = {
            "market_error_std",
            "market_error_mean",
            "overround_rolling",
            "entropy_rolling",
            "favorite_implied_prob_rolling",
            "odds_skewness_rolling",
            "odds_volatility_mean",
            "field_size_mean",
        }
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
        mock_models.submodels["turf"] = submodel
        submodel.market.predict_and_calc_error.return_value = feat_df
        submodel.stage1.add_ability_probs.return_value = feat_df
        submodel.place_ability.predict.return_value = feat_df
        submodel.win.predict_ev.return_value = feat_df
        submodel.ev_corrector.correct_ev.return_value = feat_df
        submodel.place.predict_ev.return_value = feat_df
        _corrected = feat_df.assign(ev_place_corrected=feat_df.get("ev_place", 1.5))
        submodel.place_ev_corrector.correct_ev.return_value = _corrected
        submodel.confidence.predict_lower_bound.return_value = (
            _corrected,
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )
        submodel.confidence.predict_interval.return_value = (
            _corrected,
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )

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
        mock_models.submodels["turf"] = submodel
        submodel.market.predict_and_calc_error.return_value = feat_df
        submodel.stage1.add_ability_probs.return_value = feat_df
        submodel.place_ability.predict.return_value = feat_df
        submodel.win.predict_ev.return_value = feat_df
        submodel.ev_corrector.correct_ev.return_value = feat_df
        submodel.place.predict_ev.return_value = feat_df
        _corrected = feat_df.assign(ev_place_corrected=feat_df.get("ev_place", 1.5))
        submodel.place_ev_corrector.correct_ev.return_value = _corrected
        submodel.confidence.predict_lower_bound.return_value = (
            _corrected,
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )
        submodel.confidence.predict_interval.return_value = (
            _corrected,
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )

        from backtest.engine import BacktestEngine

        mock_store = MagicMock()
        engine = BacktestEngine(models=mock_models, store=mock_store, betting_target="place")
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
