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
    submodel = MagicMock(spec=SubmodelSet)
    submodel.track_stats = {"turf": {"avg": 1.0}}
    submodel.track_month_stats = {"turf": {"avg": 1.0}}
    models.submodels = {"turf": submodel}
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
    @patch("features.feature_builder.FeatureBuilder")
    @patch("paper_trading.predictor.load_odds_snapshots")
    @patch("paper_trading.predictor.load_entries")
    @patch("paper_trading.predictor.load_races")
    def test_setup_returns_race_schedule(
        self,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds: MagicMock,
        mock_feature_builder_cls: MagicMock,
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

        # FeatureBuilder mock (Phase 52)
        from features.feature_manifest import FeatureBuildResult, FeatureManifest

        feat_df = pd.DataFrame(
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
        _manifest = FeatureManifest(column_names=(), column_dtypes=(), feature_version="1.0")
        mock_builder = MagicMock()
        mock_feature_builder_cls.return_value = mock_builder
        mock_builder.build_for_inference.return_value = FeatureBuildResult(
            frame=feat_df, manifest=_manifest,
        )

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


class TestPaperPredictorPreRaceOdds:
    """発走前オッズ優先使用のテスト (フォールバック: 確定オッズ)"""

    @patch("paper_trading.predictor.extract_pre_post_odds")
    @patch("paper_trading.predictor.load_odds_time_series_range")
    @patch("paper_trading.predictor.load_odds_snapshots")
    @patch("paper_trading.predictor.load_entries")
    @patch("paper_trading.predictor.load_races")
    @patch("features.feature_builder.FeatureBuilder")
    def test_setup_uses_pre_race_odds_when_available(
        self,
        mock_feature_builder_cls: MagicMock,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds: MagicMock,
        mock_load_ts: MagicMock,
        mock_extract: MagicMock,
        tmp_path: Path,
    ) -> None:
        """時系列オッズがある場合、発走前オッズが使用される"""
        from paper_trading.predictor import PaperPredictor

        race_df = pd.DataFrame(
            {
                "race_id": ["2026040510010101"],
                "race_date": pd.to_datetime("2026-04-05"),
                "hassotime": [1005],  # 10:05
            }
        )
        confirmed_odds = pd.DataFrame(
            {"race_id": ["2026040510010101"], "umaban": [1], "tanodds": [3.0], "fukuoddslow": [1.5]}
        )
        pre_race_odds = pd.DataFrame(
            {"race_id": ["2026040510010101"], "umaban": [1], "tanodds": [5.0], "fukuoddslow": [2.5]}
        )

        mock_load_races.return_value = race_df
        mock_load_entries.return_value = pd.DataFrame(
            {"race_id": ["2026040510010101"], "umaban": [1], "kettonum": [1234]}
        )
        mock_load_odds.return_value = confirmed_odds
        mock_load_ts.return_value = pd.DataFrame({"race_id": ["x"]})  # 非空
        mock_extract.return_value = pre_race_odds

        # FeatureBuilder mock (Phase 52)
        from features.feature_manifest import FeatureBuildResult, FeatureManifest

        feat_df = pd.DataFrame(
            {
                "race_id": ["2026040510010101"],
                "umaban": [1],
                "surface": ["turf"],
                "kyori": [1200],
            }
        )
        _manifest = FeatureManifest(column_names=(), column_dtypes=(), feature_version="1.0")
        mock_builder = MagicMock()
        mock_feature_builder_cls.return_value = mock_builder
        mock_builder.build_for_inference.return_value = FeatureBuildResult(
            frame=feat_df, manifest=_manifest,
        )

        mock_everydb2 = MagicMock()
        mock_everydb2.get_race_schedule.return_value = [
            {"race_id": "2026040510010101", "venue": "中山", "race_num": 1}
        ]

        submodel = MagicMock()
        submodel.track_stats = {"turf": {"avg": 1.0}}
        submodel.track_month_stats = {"turf": {"avg": 1.0}}
        mock_models_pt = MagicMock()
        mock_models_pt.submodels = {"turf": submodel}

        predictor = PaperPredictor(
            store=MagicMock(), race_predictor=MagicMock(), models=mock_models_pt, output_dir=tmp_path / "pt"
        )
        predictor.setup(date(2026, 4, 5), everydb2=mock_everydb2)

        # extract_pre_post_odds が呼ばれる
        mock_extract.assert_called_once()
        # FeatureBuilder.build_for_inference に pre_race_odds が渡される
        call_args = mock_builder.build_for_inference.call_args
        assert call_args is not None
        # 第3引数 (odds_df) が pre_race_odds
        odds_arg = call_args[0][2]
        assert len(odds_arg) == len(pre_race_odds)
        assert odds_arg["tanodds"].values[0] == 5.0

    @patch("paper_trading.predictor.load_odds_time_series_range")
    @patch("paper_trading.predictor.load_odds_snapshots")
    @patch("paper_trading.predictor.load_entries")
    @patch("paper_trading.predictor.load_races")
    @patch("features.feature_builder.FeatureBuilder")
    def test_setup_falls_back_to_confirmed_when_no_ts(
        self,
        mock_feature_builder_cls: MagicMock,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds: MagicMock,
        mock_load_ts: MagicMock,
        tmp_path: Path,
    ) -> None:
        """時系列オッズがない場合、確定オッズが使用される"""
        from paper_trading.predictor import PaperPredictor

        race_df = pd.DataFrame(
            {
                "race_id": ["2026040510010101"],
                "race_date": pd.to_datetime("2026-04-05"),
                "hassotime": [1005],
            }
        )
        confirmed_odds = pd.DataFrame(
            {"race_id": ["2026040510010101"], "umaban": [1], "tanodds": [3.0], "fukuoddslow": [1.5]}
        )

        mock_load_races.return_value = race_df
        mock_load_entries.return_value = pd.DataFrame(
            {"race_id": ["2026040510010101"], "umaban": [1], "kettonum": [1234]}
        )
        mock_load_odds.return_value = confirmed_odds
        mock_load_ts.return_value = pd.DataFrame()  # 空のDF → フォールバック

        # FeatureBuilder mock (Phase 52)
        from features.feature_manifest import FeatureBuildResult, FeatureManifest

        feat_df = pd.DataFrame(
            {
                "race_id": ["2026040510010101"],
                "umaban": [1],
                "surface": ["turf"],
                "kyori": [1200],
            }
        )
        _manifest = FeatureManifest(column_names=(), column_dtypes=(), feature_version="1.0")
        mock_builder = MagicMock()
        mock_feature_builder_cls.return_value = mock_builder
        mock_builder.build_for_training.return_value = FeatureBuildResult(
            frame=feat_df, manifest=_manifest,
        )

        mock_everydb2 = MagicMock()
        mock_everydb2.get_race_schedule.return_value = [
            {"race_id": "2026040510010101", "venue": "中山", "race_num": 1}
        ]

        # track_statsなしのsubmodel → build_for_training フォールバック
        submodel_no_stats = MagicMock()
        submodel_no_stats.track_stats = None
        mock_models_pt = MagicMock()
        mock_models_pt.submodels = {"turf": submodel_no_stats}

        predictor = PaperPredictor(
            store=MagicMock(), race_predictor=MagicMock(), models=mock_models_pt, output_dir=tmp_path / "pt"
        )
        predictor.setup(date(2026, 4, 5), everydb2=mock_everydb2)

        # FeatureBuilder.build_for_training に confirmed_odds が渡される
        call_args = mock_builder.build_for_training.call_args
        assert call_args is not None
        odds_arg = call_args[0][2]
        assert len(odds_arg) == len(confirmed_odds)
        assert odds_arg["tanodds"].values[0] == 3.0  # 確定オッズの値

    @patch("paper_trading.predictor.extract_pre_post_odds")
    @patch("paper_trading.predictor.load_odds_time_series_range")
    @patch("paper_trading.predictor.load_odds_snapshots")
    @patch("paper_trading.predictor.load_entries")
    @patch("paper_trading.predictor.load_races")
    @patch("features.feature_builder.FeatureBuilder")
    def test_setup_falls_back_when_pre_race_empty(
        self,
        mock_feature_builder_cls: MagicMock,
        mock_load_races: MagicMock,
        mock_load_entries: MagicMock,
        mock_load_odds: MagicMock,
        mock_load_ts: MagicMock,
        mock_extract: MagicMock,
        tmp_path: Path,
    ) -> None:
        """extract_pre_post_odds が空DFを返した場合、確定オッズにフォールバック"""
        from paper_trading.predictor import PaperPredictor

        race_df = pd.DataFrame(
            {
                "race_id": ["2026040510010101"],
                "race_date": pd.to_datetime("2026-04-05"),
                "hassotime": [1005],
            }
        )
        confirmed_odds = pd.DataFrame(
            {"race_id": ["2026040510010101"], "umaban": [1], "tanodds": [3.0], "fukuoddslow": [1.5]}
        )

        mock_load_races.return_value = race_df
        mock_load_entries.return_value = pd.DataFrame(
            {"race_id": ["2026040510010101"], "umaban": [1], "kettonum": [1234]}
        )
        mock_load_odds.return_value = confirmed_odds
        mock_load_ts.return_value = pd.DataFrame({"race_id": ["x"]})  # 非空
        mock_extract.return_value = pd.DataFrame()  # 空の結果

        # FeatureBuilder mock (Phase 52)
        from features.feature_manifest import FeatureBuildResult, FeatureManifest

        feat_df = pd.DataFrame(
            {
                "race_id": ["2026040510010101"],
                "umaban": [1],
                "surface": ["turf"],
                "kyori": [1200],
            }
        )
        _manifest = FeatureManifest(column_names=(), column_dtypes=(), feature_version="1.0")
        mock_builder = MagicMock()
        mock_feature_builder_cls.return_value = mock_builder
        mock_builder.build_for_training.return_value = FeatureBuildResult(
            frame=feat_df, manifest=_manifest,
        )

        mock_everydb2 = MagicMock()
        mock_everydb2.get_race_schedule.return_value = [
            {"race_id": "2026040510010101", "venue": "中山", "race_num": 1}
        ]

        # track_statsなしのsubmodel → build_for_training フォールバック
        submodel_no_stats = MagicMock()
        submodel_no_stats.track_stats = None
        mock_models_pt = MagicMock()
        mock_models_pt.submodels = {"turf": submodel_no_stats}

        predictor = PaperPredictor(
            store=MagicMock(), race_predictor=MagicMock(), models=mock_models_pt, output_dir=tmp_path / "pt"
        )
        predictor.setup(date(2026, 4, 5), everydb2=mock_everydb2)

        # FeatureBuilder.build_for_training に confirmed_odds が渡される (フォールバック)
        call_args = mock_builder.build_for_training.call_args
        assert call_args is not None
        odds_arg = call_args[0][2]
        assert len(odds_arg) == len(confirmed_odds)
        assert odds_arg["tanodds"].values[0] == 3.0  # 確定オッズの値
