"""BacktestEngine のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from domain.models import SubmodelSet, TrainedModelsV5
from domain.types import RegimeState


@pytest.fixture
def mock_models() -> MagicMock:
    """モック TrainedModelsV5"""
    models = MagicMock(spec=TrainedModelsV5)
    models.submodels = {"turf": MagicMock(spec=SubmodelSet)}
    models.quality_screener = MagicMock()
    models.quality_screener.should_bet.return_value = True
    models.regime_detector = MagicMock()
    models.regime_detector.current_regime = RegimeState.CONSERVATIVE
    models.regime_detector.get_strategy_params.return_value = {
        "ev_threshold": 1.20,
        "score_threshold": 0.015,
        "max_bets_per_race": 3,
    }
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


class TestBetHistoryEnrichment:
    """bet_history への surface/distance/ev/popularity/bankroll_after 付与テスト"""

    @patch("features.trainer_context_features.TrainerContextFeatures")
    @patch("features.jockey_context_features.JockeyContextFeatures")
    @patch("features.interaction_features.compute_interaction_features")
    @patch("features.horse_history_features.HorseHistoryFeatures")
    @patch("models.submodel_manager.SubModelManager")
    @patch("features.feature_engine.FeatureEngine")
    @patch("backtest.engine.load_odds_snapshots")
    @patch("backtest.engine.load_entries")
    @patch("backtest.engine.load_races")
    def test_engine_populates_enriched_fields(
        self,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds: MagicMock,
        mock_feat_engine_cls: MagicMock,
        mock_submodel_mgr_cls: MagicMock,
        mock_hist_cls: MagicMock,
        mock_interaction_fn: MagicMock,
        mock_jockey_cls: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_models: MagicMock,
    ) -> None:
        """エンジンループが bet_history に拡張フィールドを付与する"""
        # --- load mocks ---
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

        # --- feat_df (complete columns for pipeline) ---
        feat_df = pd.DataFrame(
            {
                "race_id": ["20240101010101"],
                "umaban": [1],
                "surface": ["turf"],
                "kyori": [1200],
                "distance_bin": ["sprint"],
                "popularity_rank": [3],
                "ninki": [3],
                "ev_place": [1.5],
                "fukuoddslow": [2.4],
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
        mock_models.submodels["turf"] = submodel
        submodel.market.predict_and_calc_error.return_value = feat_df
        submodel.stage1.add_ability_probs.return_value = feat_df
        submodel.place_ability.predict.return_value = feat_df
        submodel.win.predict_ev.return_value = feat_df
        submodel.ev_corrector.correct_ev.return_value = feat_df
        submodel.place.predict_ev.return_value = feat_df
        submodel.confidence.predict_lower_bound.return_value = (
            feat_df,
            pd.DataFrame({"EV_lower_place": [1.5]}),
        )

        # --- run engine ---
        from backtest.engine import BacktestEngine

        mock_store = MagicMock()
        engine = BacktestEngine(models=mock_models, store=mock_store)
        result = engine.run("2024-01-01", "2024-12-31")

        # --- assertions ---
        assert result.total_bets >= 1, "Should place at least 1 bet"
        assert result.n_pre_post_odds_bets + result.n_fallback_odds_bets >= 1
        # mock で odds_ts が空 → フォールバック扱い
        assert result.n_fallback_odds_bets >= 1
        assert result.n_pre_post_odds_bets == 0
        bet = result.bet_history[0]
        assert "surface" in bet
        assert bet["surface"] == "turf"
        assert "kyori" in bet
        assert bet["kyori"] == 1200
        assert "ev" in bet
        assert bet["ev"] == 1.5
        assert "popularity" in bet
        assert bet["popularity"] == 3
        assert "bankroll_after" in bet
        assert isinstance(bet["bankroll_after"], float)
        assert bet["bankroll_after"] == 100140.0

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
