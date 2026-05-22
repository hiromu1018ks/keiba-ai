"""src/models/two_stage_return_model.py のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from models.two_stage_return_model import (
    PlaceTwoStageModel,
    WinTwoStageModel,
    _train_valid_split,
)


@pytest.fixture
def feature_df() -> pd.DataFrame:
    """WinTwoStageModel + PlaceTwoStageModel (HIT + RETURN) FEATURE_COLS を満たす8頭テストデータ"""
    return pd.DataFrame(
        {
            "race_id": ["R1"] * 8,
            "p_ability_win": [0.30, 0.25, 0.20, 0.10, 0.08, 0.04, 0.02, 0.01],
            "signed_log_error_win": [0.1, -0.1, 0.2, -0.3, 0.0, 0.5, -0.2, 0.3],
            "abs_log_error_win": [0.1, 0.1, 0.2, 0.3, 0.0, 0.5, 0.2, 0.3],
            "odds_drop_rate_60_10": [0.0, -0.1, 0.1, -0.2, 0.0, 0.1, -0.1, 0.0],
            "odds_drop_rate_30_10": [0.0, -0.05, 0.05, -0.1, 0.0, 0.05, -0.05, 0.0],
            "odds_velocity": [0.0, -0.02, 0.02, -0.03, 0.0, 0.01, -0.01, 0.0],
            "odds_volatility": [0.01, 0.02, 0.01, 0.03, 0.01, 0.02, 0.01, 0.01],
            "popularity_change_30_10": [0, -1, 1, -2, 0, 1, -1, 0],
            "market_entropy": [2.5] * 8,
            "popularity_rank": [1, 2, 3, 4, 5, 6, 7, 8],
            "overround": [0.22] * 8,
            "surface": ["turf"] * 8,
            "distance_bin": ["mile"] * 8,
            "track_condition_code": [1] * 8,
            "grade_code": ["_"] * 8,
            "field_size": [8] * 8,
            # FLB slope (市場歪みの非対称性)
            "odds_skewness": [1.5] * 8,
            # Place固有特徴量
            "fukuoddslow": [1.3, 1.5, 1.8, 2.1, 2.5, 3.0, 3.5, 4.0],
            "tanodds": [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0],
            "p_ability_place": [0.55, 0.48, 0.42, 0.38, 0.32, 0.25, 0.20, 0.15],
            # --- 馬レベル特徴量 (Place HIT model 専用) ---
            "norm_finish_logit_avg": [0.1, 0.05, -0.1, -0.3, -0.5, -0.7, -0.9, -1.2],
            "harontimel5_zscore": [-0.5, -0.2, 0.0, 0.3, 0.6, 1.0, 1.3, 1.8],
            "closing_index_avg": [0.55, 0.50, 0.45, 0.35, 0.28, 0.20, 0.12, 0.05],
            "weight_zscore": [-0.3, -0.1, 0.0, 0.2, 0.4, 0.6, 0.9, 1.2],
            "days_since_last_race": [14, 21, 28, 35, 45, 60, 90, 120],
            "rest_category": [0, 0, 1, 1, 2, 2, 3, 3],
            "form_trend": [0.3, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5],
            "form_consistency": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            "blood_surface_wr": [0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.06, 0.04],
            "blood_distance_wr": [0.15, 0.13, 0.12, 0.10, 0.09, 0.07, 0.05, 0.03],
            "jockey_wr_overall": [0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.07, 0.05],
            "jockey_wr_distance": [0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.05, 0.03],
            "jockey_wr_venue": [0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.06, 0.04],
            "jockey_prize_log": [14.0, 13.5, 13.0, 12.5, 12.0, 11.5, 10.5, 9.5],
            "trainer_wr_overall": [0.12, 0.11, 0.10, 0.09, 0.08, 0.06, 0.05, 0.03],
            "trainer_wr_distance": [0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.04, 0.02],
            "trainer_wr_venue": [0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.02],
            "trainer_prize_log": [13.5, 13.0, 12.5, 12.0, 11.5, 11.0, 10.0, 9.0],
            "jt_combo_wr": [0.15, 0.13, 0.11, 0.09, 0.07, 0.05, 0.03, 0.01],
            "jt_combo_place_rate": [0.32, 0.29, 0.26, 0.23, 0.19, 0.16, 0.12, 0.08],
            "jt_combo_starts": [50, 45, 40, 35, 30, 25, 20, 15],
            "jt_combo_prize_log": [14.0, 13.5, 13.0, 12.5, 12.0, 11.5, 11.0, 10.0],
            "course_wr": [0.17, 0.15, 0.13, 0.11, 0.09, 0.07, 0.05, 0.03],
            # n_mining予想特徴量 (DATA-04)
            "dm_time_rank": [1, 2, 3, 4, 5, 6, 7, 8],
            "dm_time_zscore": [0.5, -0.3, 0.1, -0.8, 1.2, -0.5, 0.3, -0.1],
            "dm_confidence_range": [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            "dm_time_margin_to_fav": [0.0, 0.3, 0.5, 0.8, 1.0, 1.3, 1.8, 2.5],
            # 繁殖牝馬産駒特徴量 (DATA-01)
            "dam_wr": [0.14, 0.12, 0.10, 0.09, 0.08, 0.07, 0.05, 0.03],
            "dam_surface_wr": [0.13, 0.11, 0.10, 0.08, 0.07, 0.06, 0.04, 0.02],
            "dam_prize_log": [14.0, 13.5, 13.0, 12.5, 12.0, 11.5, 10.5, 9.5],
            "breeder_strength": [1.39, 1.39, 1.39, 1.10, 1.39, 1.10, 0.69, 0.69],
            # BMS拡張特徴量 (DATA-01)
            "bms_distance_wr": [0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.05, 0.03],
            "bms_surface_wr": [0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.04, 0.02],
            # コースレコード特徴量 (DATA-02)
            "course_record_time": [95.3, 95.3, 95.3, 95.3, 95.3, 95.3, 95.3, 95.3],
            # レース内相対比較特徴量 (DATA-03)
            "rel_norm_finish_zscore": [1.2, 0.3, -0.5, -1.0, 0.8, 0.1, -0.4, -0.5],
            "rel_haron_vs_mean": [0.5, 0.2, -0.1, -0.3, 0.4, 0.1, -0.2, -0.3],
            "rel_timediff_rank": [1, 2, 3, 4, 1, 2, 3, 4],
            "rel_blood_quality_rank": [1, 2, 3, 4, 1, 2, 3, 4],
            "rel_sire_quality_rank": [1, 2, 3, 4, 1, 2, 3, 4],
            "rel_weight_zscore": [0.8, -0.3, 0.1, -0.6, 0.7, -0.2, 0.2, -0.5],
            "rel_closing_index_rank": [4, 3, 2, 1, 4, 3, 2, 1],
            # INTER-01: オッズ相対特徴量
            "rel_popularity_rank_zscore": [1.5, 0.5, -0.3, -1.2, 0.0, 0.0, 0.0, 0.0],
            "rel_fuku_odds_zscore": [-1.1, -0.5, 0.3, 1.3, 0.0, 0.0, 0.0, 0.0],
            # INTER-01: Stage2能力値相対特徴量
            "rel_p_ability_win_zscore": [1.6, 0.4, -0.2, -1.0, 0.0, 0.0, 0.0, 0.0],
            "rel_p_ability_win_rank": [1, 2, 3, 4, 1, 1, 1, 1],
            "rel_odds_ability_deviation": [-1.2, 0.1, 0.5, 1.3, 0.0, 0.0, 0.0, 0.0],
            # INTER-02: 交互作用特徴量 (12)
            "kyakusitu_x_distance": ["2.0_mile"] * 8,
            "kyakusitu_x_surface": ["2.0_turf"] * 8,
            "weight_x_distance": [450.0 * 1200] * 8,
            "surface_x_distance_bin": ["turf_mile"] * 8,
            "blood_keito_x_surface": ["1.0_turf"] * 8,
            "grade_code_x_distance_bin": ["_mile"] * 8,
            "sire_wr_x_distance": [0.15 * 1200] * 8,
            "blood_surface_wr_x_condition": [0.18 * 1] * 8,
            "pace_pressure_x_closing_index": [0.5 * 0.4] * 8,
            "haron_x_distance": [35.5 * 1200] * 8,
            "surface_x_past_perf": [0.1 * 1] * 8,
            "weight_x_class": [450.0 * 1] * 8,
            # INTER-03: Target Encoding (OOF-safe)
            "te_blood_keito_cd": [0.08, 0.07, 0.09, 0.06, 0.10, 0.05, 0.07, 0.04],
            "te_kisyucode": [0.12, 0.10, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05],
            "te_chokyosicode": [0.11, 0.09, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04],
            # その他
            "finish_pos": [1, 2, 3, 4, 5, 6, 7, 8],
            "win_odds_actual": [3.5, 5.0, 8.0, 15.0, 25.0, 40.0, 80.0, 150.0],
            "place_odds_actual": [1.3, 1.6, 2.1, 3.5, 5.0, 8.0, 15.0, 30.0],
        }
    )


def _make_mock_booster(pred_values: list[float]) -> MagicMock:
    """best_iteration 属性付きの mock Booster を生成"""
    mock = MagicMock()
    mock.predict.return_value = np.array(pred_values)
    mock.best_iteration = 100
    return mock


@pytest.fixture
def trained_win_model(feature_df: pd.DataFrame) -> WinTwoStageModel:
    """学習済みWinTwoStageModel (mock)"""
    model = WinTwoStageModel()
    model.hit_model = _make_mock_booster([0.28, 0.22, 0.18, 0.12, 0.08, 0.06, 0.04, 0.02])
    model.return_model = _make_mock_booster([4.0, 6.0, 9.0, 16.0, 28.0, 45.0, 90.0, 160.0])
    return model


class TestWinTwoStageModel:
    def test_ev_equals_p_times_e(
        self,
        trained_win_model: WinTwoStageModel,
        feature_df: pd.DataFrame,
    ) -> None:
        """EV = P(win) × E(win_odds | win)"""
        result = trained_win_model.predict_ev(feature_df)
        expected_ev = result["p_win_pred"] * result["e_return_win_pred"]
        assert np.allclose(result["ev_win"].values, expected_ev.values)

    def test_output_columns(
        self,
        trained_win_model: WinTwoStageModel,
        feature_df: pd.DataFrame,
    ) -> None:
        result = trained_win_model.predict_ev(feature_df)
        assert "p_win_pred" in result.columns
        assert "e_return_win_pred" in result.columns
        assert "ev_win" in result.columns

    def test_predict_uses_best_iteration(
        self,
        trained_win_model: WinTwoStageModel,
        feature_df: pd.DataFrame,
    ) -> None:
        """predict が num_iteration=best_iteration を使用すること"""
        trained_win_model.predict_ev(feature_df)

        trained_win_model.hit_model.predict.assert_called_once()
        call_kwargs = trained_win_model.hit_model.predict.call_args
        assert call_kwargs[1]["num_iteration"] == 100

        trained_win_model.return_model.predict.assert_called_once()
        call_kwargs = trained_win_model.return_model.predict.call_args
        assert call_kwargs[1]["num_iteration"] == 100

    @patch("models.two_stage_return_model.lgb")
    def test_train_hit_model_uses_early_stopping(
        self, mock_lgb: MagicMock, feature_df: pd.DataFrame
    ) -> None:
        """train_hit_model が valid_sets と callbacks を渡すこと"""
        mock_booster = MagicMock()
        mock_booster.best_iteration = 50
        mock_lgb.train.return_value = mock_booster
        mock_lgb.Dataset.return_value = MagicMock()
        mock_lgb.early_stopping.return_value = lambda: None

        df = feature_df.copy()
        df["kakuteijyuni"] = [1, 2, 3, 4, 5, 6, 7, 8]

        model = WinTwoStageModel()
        model.train_hit_model(df)

        call_args = mock_lgb.train.call_args
        assert "valid_sets" in call_args[1]
        assert "callbacks" in call_args[1]
        mock_lgb.early_stopping.assert_called_once_with(stopping_rounds=100, verbose=False)

    @patch("models.two_stage_return_model.lgb")
    def test_train_return_model_early_stopping_guard(self, mock_lgb: MagicMock) -> None:
        """return_model: サンプル < 10 は early stopping なし"""
        mock_booster = MagicMock()
        mock_booster.best_iteration = 50
        mock_lgb.train.return_value = mock_booster
        mock_lgb.Dataset.return_value = MagicMock()

        # 5件のみの的中データ (min_hit_samples を満たさない可能性があるので注意)
        df = pd.DataFrame(
            {
                "kakuteijyuni": [1] * 5,
                "odds": [3.0, 5.0, 8.0, 15.0, 25.0],
                "confirmed_odds": [3.0, 5.0, 8.0, 15.0, 25.0],
                "p_ability_win": [0.3] * 5,
                "signed_log_error_win": [0.1] * 5,
                "abs_log_error_win": [0.1] * 5,
                "odds_drop_rate_60_10": [0.0] * 5,
                "odds_drop_rate_30_10": [0.0] * 5,
                "odds_velocity": [0.0] * 5,
                "odds_volatility": [0.01] * 5,
                "popularity_change_30_10": [0] * 5,
                "market_entropy": [2.5] * 5,
                "popularity_rank": [1] * 5,
                "overround": [0.22] * 5,
                "surface": ["turf"] * 5,
                "distance_bin": ["mile"] * 5,
                "track_condition_code": [1] * 5,
                "grade_code": ["_"] * 5,
                "field_size": [5] * 5,
                "odds_skewness": [1.5] * 5,
            }
        )

        # min_hit_samples を下回る場合は ValueError が発生するので、
        # cfg を調整して回避
        from domain.models import TwoStageConfig

        cfg = TwoStageConfig(min_hit_samples=3)
        model = WinTwoStageModel(cfg=cfg)
        model.train_return_model(df)

        call_args = mock_lgb.train.call_args
        # サンプル数 < 10 のため valid_sets が無いことを確認
        assert "valid_sets" not in call_args[1]

    def test_stage_b_no_zeros_in_label(self, feature_df: pd.DataFrame) -> None:
        """Stage B の学習データにゼロが含まれないことを確認するテスト"""
        hit_df = feature_df[feature_df["finish_pos"] == 1]
        assert (hit_df["win_odds_actual"] > 0).all()

    def test_feature_cols_no_p_market_pred(self) -> None:
        """p_market_pred が特徴量に含まれない (Rule 11)"""
        assert "p_market_pred_win" not in WinTwoStageModel.FEATURE_COLS

    def test_get_filtered_feature_cols_returns_new_list(self) -> None:
        """get_filtered_feature_cols がクラス変数を変更せずに新リストを返す"""
        original = list(WinTwoStageModel.FEATURE_COLS)
        noise = ["odds_skewness", "popularity_rank"]
        filtered = WinTwoStageModel.get_filtered_feature_cols(noise)
        # 戻り値はノイズ特徴量を含まない
        assert "odds_skewness" not in filtered
        assert "popularity_rank" not in filtered
        # クラス変数は変更されていない
        assert WinTwoStageModel.FEATURE_COLS == original
        assert len(filtered) == len(original) - len(noise)

    def test_get_filtered_feature_cols_with_nonexistent(self) -> None:
        """存在しない特徴量名を指定してもエラーにならない"""
        original = list(WinTwoStageModel.FEATURE_COLS)
        filtered = WinTwoStageModel.get_filtered_feature_cols(["nonexistent_feature"])
        assert filtered == original


class TestPlaceTwoStageModel:
    @pytest.fixture
    def trained_place_model(self, feature_df: pd.DataFrame) -> PlaceTwoStageModel:
        model = PlaceTwoStageModel()
        model.hit_model = _make_mock_booster([0.40, 0.35, 0.30, 0.15, 0.10, 0.05, 0.03, 0.01])
        model.return_model = _make_mock_booster([1.4, 1.7, 2.2, 3.8, 5.5, 9.0, 16.0, 32.0])
        return model

    def test_ev_equals_p_times_e(
        self,
        trained_place_model: PlaceTwoStageModel,
        feature_df: pd.DataFrame,
    ) -> None:
        result = trained_place_model.predict_ev(feature_df)
        expected_ev = result["p_place_pred"] * result["e_return_place_pred"]
        assert np.allclose(result["ev_place"].values, expected_ev.values)

    def test_output_columns(
        self,
        trained_place_model: PlaceTwoStageModel,
        feature_df: pd.DataFrame,
    ) -> None:
        result = trained_place_model.predict_ev(feature_df)
        assert "p_place_pred" in result.columns
        assert "e_return_place_pred" in result.columns
        assert "ev_place" in result.columns

    def test_predict_uses_best_iteration(
        self,
        trained_place_model: PlaceTwoStageModel,
        feature_df: pd.DataFrame,
    ) -> None:
        """predict が num_iteration=best_iteration を使用すること"""
        trained_place_model.predict_ev(feature_df)

        trained_place_model.hit_model.predict.assert_called_once()
        call_kwargs = trained_place_model.hit_model.predict.call_args
        assert call_kwargs[1]["num_iteration"] == 100

        trained_place_model.return_model.predict.assert_called_once()
        call_kwargs = trained_place_model.return_model.predict.call_args
        assert call_kwargs[1]["num_iteration"] == 100

    @patch("models.two_stage_return_model.lgb")
    def test_train_hit_model_uses_early_stopping(
        self, mock_lgb: MagicMock, feature_df: pd.DataFrame
    ) -> None:
        """train_hit_model が valid_sets と callbacks を渡すこと"""
        mock_booster = MagicMock()
        mock_booster.best_iteration = 50
        mock_lgb.train.return_value = mock_booster
        mock_lgb.Dataset.return_value = MagicMock()
        mock_lgb.early_stopping.return_value = lambda: None

        df = feature_df.copy()
        df["kakuteijyuni"] = [1, 2, 3, 4, 5, 6, 7, 8]

        model = PlaceTwoStageModel()
        model.train_hit_model(df)

        call_args = mock_lgb.train.call_args
        assert "valid_sets" in call_args[1]
        assert "callbacks" in call_args[1]
        mock_lgb.early_stopping.assert_called_once_with(stopping_rounds=100, verbose=False)

    def test_hit_and_return_features_separated(self) -> None:
        """Hit model と Return model で特徴量が適切に分離されていること

        Hit model は fukuoddslow/tanodds を除外し、馬レベル特徴量を含む。
        Return model は tanodds のみ保持 (fukuoddslow は target と同じのため除外)。
        """
        # Hit model はオッズ特徴量を含まない（二重計数防止）
        assert "fukuoddslow" not in PlaceTwoStageModel.HIT_FEATURE_COLS
        assert "tanodds" not in PlaceTwoStageModel.HIT_FEATURE_COLS
        # Return model: tanodds は保持 (市場規模の代理指標)、fukuoddslow は除外 (target leakage)
        assert "fukuoddslow" not in PlaceTwoStageModel.RETURN_FEATURE_COLS
        assert "tanodds" in PlaceTwoStageModel.RETURN_FEATURE_COLS
        # p_ability_place は両方に含まれる
        assert "p_ability_place" in PlaceTwoStageModel.HIT_FEATURE_COLS
        assert "p_ability_place" in PlaceTwoStageModel.RETURN_FEATURE_COLS
        # 馬レベル特徴量が HIT に追加されている
        horse_features = [
            "norm_finish_logit_avg",
            "harontimel5_zscore",
            "closing_index_avg",
            "weight_zscore",
            "days_since_last_race",
            "rest_category",
            "form_trend",
            "form_consistency",
            "blood_surface_wr",
            "blood_distance_wr",
            "jockey_wr_overall",
            "trainer_wr_overall",
            "jt_combo_place_rate",
            "course_wr",
        ]
        for col in horse_features:
            assert col in PlaceTwoStageModel.HIT_FEATURE_COLS, (
                f"{col} should be in HIT_FEATURE_COLS"
            )

    def test_place_return_feature_cols_include_place_specific(self) -> None:
        """Return model should have place-specific features beyond win features"""
        # fukuoddslow は target と同じため除外 (target leakage 防止)
        assert "fukuoddslow" not in PlaceTwoStageModel.RETURN_FEATURE_COLS
        assert "p_ability_place" in PlaceTwoStageModel.RETURN_FEATURE_COLS
        assert "tanodds" in PlaceTwoStageModel.RETURN_FEATURE_COLS
        # Win特徴量も全て含む
        for col in WinTwoStageModel.FEATURE_COLS:
            assert col in PlaceTwoStageModel.RETURN_FEATURE_COLS
        # Place固有特徴量が追加されている
        assert len(PlaceTwoStageModel.RETURN_FEATURE_COLS) > len(WinTwoStageModel.FEATURE_COLS)


class TestTrainValidSplit:
    def test_split_ratio(self) -> None:
        """80/20 分割が正しいサイズを返す"""
        features = pd.DataFrame({"a": range(100), "b": range(100)})
        label = pd.Series(range(100))

        with patch("models.two_stage_return_model.lgb.Dataset") as mock_ds:
            mock_ds.return_value = MagicMock()
            _train_valid_split(features, label)

        # 呼び出し時に渡された DataFrame のサイズを確認
        first_call = mock_ds.call_args_list[0]
        second_call = mock_ds.call_args_list[1]

        train_features = first_call[0][0]
        valid_features = second_call[0][0]

        assert len(train_features) == 80
        assert len(valid_features) == 20

    def test_no_overlap(self) -> None:
        """train と valid に重複がないこと"""
        features = pd.DataFrame({"a": range(50), "b": range(50)})
        label = pd.Series(range(50))

        with patch("models.two_stage_return_model.lgb.Dataset") as mock_ds:
            mock_ds.return_value = MagicMock()
            _train_valid_split(features, label)

        train_idx = mock_ds.call_args_list[0][0][0].index
        valid_idx = mock_ds.call_args_list[1][0][0].index

        assert len(set(train_idx) & set(valid_idx)) == 0

    def test_train_valid_split_is_chronological(self) -> None:
        """_train_valid_split が時系列順で前80%/後20%に分割することを確認"""
        # 10行のデータ、明確なラベルで時系列順を確認
        features = pd.DataFrame({"f1": np.arange(10, dtype=float)})
        label = pd.Series(np.arange(10, dtype=float))  # 0,1,...,9

        train_data, valid_data = _train_valid_split(features, label, valid_ratio=0.2)

        # train は前8行 (label 0-7)、valid は後2行 (label 8-9)
        assert len(train_data.get_label()) == 8
        assert len(valid_data.get_label()) == 2
        train_labels = sorted(train_data.get_label())
        assert train_labels == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        valid_labels = sorted(valid_data.get_label())
        assert valid_labels == [8.0, 9.0]

    def test_train_valid_split_no_random_permutation(self) -> None:
        """_train_valid_split が np.random.permutation を使わないことを確認"""
        import inspect

        source = inspect.getsource(_train_valid_split)
        assert "permutation" not in source, "Still using random permutation!"
        assert "RandomState" not in source, "Still using RandomState!"


# ---------------------------------------------------------------------------
# FEAT-02: 6新特徴量のWinTwoStageModel統合テスト
# ---------------------------------------------------------------------------


class TestOddsToAbilityRatio:
    """odds_to_ability_ratio のFEATURE_COLS統合テスト"""

    def test_odds_to_ability_ratio_in_feature_cols(self) -> None:
        """odds_to_ability_ratio が FEATURE_COLS に含まれる"""
        assert "odds_to_ability_ratio" in WinTwoStageModel.FEATURE_COLS

    def test_ratio_computation_range(self) -> None:
        """odds_to_ability_ratio が [0.1, 10.0] の範囲にクリップされる"""
        df = pd.DataFrame({
            "p_market_win_adj": [0.5, 0.01, 0.99],
            "p_ability_win": [0.1, 0.5, 0.001],
        })
        p_market = df["p_market_win_adj"].clip(lower=1e-6)
        p_ability = df["p_ability_win"].clip(lower=1e-6)
        ratio = (p_market / p_ability).clip(0.1, 10.0)
        assert (ratio >= 0.1).all()
        assert (ratio <= 10.0).all()
        # 0.5 / 0.1 = 5.0
        assert abs(ratio.iloc[0] - 5.0) < 0.01

    def test_ratio_nan_when_ability_nan(self) -> None:
        """p_ability_win が NaN → ratio も NaN"""
        df = pd.DataFrame({
            "p_market_win_adj": [0.5],
            "p_ability_win": [float("nan")],
        })
        p_market = df["p_market_win_adj"].clip(lower=1e-6)
        p_ability = df["p_ability_win"].clip(lower=1e-6)
        ratio = (p_market / p_ability).clip(0.1, 10.0)
        assert np.isnan(ratio.iloc[0])

    def test_ratio_extreme_values_clipped(self) -> None:
        """極端な値が [0.1, 10.0] にクリップされる"""
        df = pd.DataFrame({
            "p_market_win_adj": [0.99, 0.0001],
            "p_ability_win": [0.0001, 0.99],
        })
        p_market = df["p_market_win_adj"].clip(lower=1e-6)
        p_ability = df["p_ability_win"].clip(lower=1e-6)
        ratio = (p_market / p_ability).clip(0.1, 10.0)
        assert ratio.iloc[0] == 10.0  # 990000 → 10.0
        assert ratio.iloc[1] == 0.1   # ~0 → 0.1


class TestInferencePathComputation:
    """_prepare_features() 推論パスのodds_to_ability_ratio計算テスト"""

    def test_prepare_features_computes_ratio_when_missing(self) -> None:
        """odds_to_ability_ratio が df にない場合、_prepare_features が計算する"""
        model = WinTwoStageModel()
        df = pd.DataFrame({
            "p_ability_win": [0.3, 0.2],
            "signed_log_error_win": [0.1, -0.1],
            "abs_log_error_win": [0.1, 0.1],
            "odds_drop_rate_60_10": [0.0, -0.1],
            "odds_drop_rate_30_10": [0.0, -0.05],
            "odds_velocity": [0.0, -0.02],
            "odds_volatility": [0.01, 0.02],
            "popularity_change_30_10": [0, -1],
            "market_entropy": [2.5, 2.5],
            "popularity_rank": [1, 2],
            "overround": [0.22, 0.22],
            "surface": ["turf", "turf"],
            "distance_bin": ["mile", "mile"],
            "track_condition_code": [1, 1],
            "grade_code": ["_", "_"],
            "field_size": [8, 8],
            "odds_skewness": [1.5, 1.5],
            "pace_pressure": [0.5, 0.5],
            "pace_scenario_fit": [0.5, 0.5],
            "class_move": [0.0, 0.0],
            "blinker_change": [0.0, 0.0],
            "is_nar_transfer": [0.0, 0.0],
            "nar_recent_ratio": [0.0, 0.0],
            "track_condition_delta": [0.0, 0.0],
            "p_market_win_adj": [0.25, 0.15],
        })
        features = model._prepare_features(df)
        assert "odds_to_ability_ratio" in features.columns
        # 0.25 / 0.3 ≈ 0.833
        assert abs(features["odds_to_ability_ratio"].iloc[0] - 0.833) < 0.05

    def test_prepare_features_does_not_overwrite_existing(self) -> None:
        """odds_to_ability_ratio が既に df にある場合、上書きしない"""
        model = WinTwoStageModel()
        df = pd.DataFrame({
            "p_ability_win": [0.3],
            "signed_log_error_win": [0.1],
            "abs_log_error_win": [0.1],
            "odds_drop_rate_60_10": [0.0],
            "odds_drop_rate_30_10": [0.0],
            "odds_velocity": [0.0],
            "odds_volatility": [0.01],
            "popularity_change_30_10": [0],
            "market_entropy": [2.5],
            "popularity_rank": [1],
            "overround": [0.22],
            "surface": ["turf"],
            "distance_bin": ["mile"],
            "track_condition_code": [1],
            "grade_code": ["_"],
            "field_size": [8],
            "odds_skewness": [1.5],
            "pace_pressure": [0.5],
            "pace_scenario_fit": [0.5],
            "class_move": [0.0],
            "blinker_change": [0.0],
            "is_nar_transfer": [0.0],
            "nar_recent_ratio": [0.0],
            "track_condition_delta": [0.0],
            "p_market_win_adj": [0.25],
            "odds_to_ability_ratio": [42.0],  # 既存値
        })
        features = model._prepare_features(df)
        # 既存値42.0が維持される (上書きされない)
        assert features["odds_to_ability_ratio"].iloc[0] == 42.0

    def test_prepare_features_skips_when_inputs_missing(self) -> None:
        """p_market_win_adj または p_ability_win がない場合、エラーなくスキップ"""
        model = WinTwoStageModel()
        df = pd.DataFrame({
            "p_ability_win": [0.3],
            "signed_log_error_win": [0.1],
            "abs_log_error_win": [0.1],
            "odds_drop_rate_60_10": [0.0],
            "odds_drop_rate_30_10": [0.0],
            "odds_velocity": [0.0],
            "odds_volatility": [0.01],
            "popularity_change_30_10": [0],
            "market_entropy": [2.5],
            "popularity_rank": [1],
            "overround": [0.22],
            "surface": ["turf"],
            "distance_bin": ["mile"],
            "track_condition_code": [1],
            "grade_code": ["_"],
            "field_size": [8],
            "odds_skewness": [1.5],
            "pace_pressure": [0.5],
            "pace_scenario_fit": [0.5],
            "class_move": [0.0],
            "blinker_change": [0.0],
            "is_nar_transfer": [0.0],
            "nar_recent_ratio": [0.0],
            "track_condition_delta": [0.0],
            # p_market_win_adj がない
        })
        # エラーなく実行されること
        features = model._prepare_features(df)
        assert isinstance(features, pd.DataFrame)


class TestHistoryFeaturesInFeatureCols:
    """5履歴特徴量がFEATURE_COLSに含まれているかのテスト"""

    def test_all_history_features_in_feature_cols(self) -> None:
        """5つの履歴特徴量が全てFEATURE_COLSに含まれる"""
        expected = [
            "distance_change", "surface_change", "class_drop_bounce",
            "freshness_score",
        ]
        for name in expected:
            assert name in WinTwoStageModel.FEATURE_COLS, (
                f"{name} should be in WinTwoStageModel.FEATURE_COLS"
            )

    def test_feature_cols_no_duplicates(self) -> None:
        """FEATURE_COLSに重複がない"""
        cols = WinTwoStageModel.FEATURE_COLS
        assert len(cols) == len(set(cols)), (
            f"FEATURE_COLS has duplicates: {[c for c in cols if cols.count(c) > 1]}"
        )

    def test_feature_cols_minimum_length(self) -> None:
        """FEATURE_COLSが最低31件 (25既存 + 6新) である"""
        cols = WinTwoStageModel.FEATURE_COLS
        assert len(cols) >= 31, (
            f"Expected >= 31 FEATURE_COLS, got {len(cols)}: {cols}"
        )


class TestJockeyTrainerComboInFeatureCols:
    """騎手・調教師・コンビ12特徴量のFEATURE_COLS統合テスト"""

    JT_FEATURES: list[str] = [
        "jockey_wr_overall",
        "jockey_wr_distance",
        "jockey_wr_venue",
        "jockey_prize_log",
        "trainer_wr_overall",
        "trainer_wr_distance",
        "trainer_wr_venue",
        "trainer_prize_log",
        "jt_combo_wr",
        "jt_combo_place_rate",
        "jt_combo_starts",
        "jt_combo_prize_log",
    ]

    def test_jockey_trainer_combo_in_win_feature_cols(self) -> None:
        """12特徴量がWinTwoStageModel.FEATURE_COLSに含まれる"""
        for name in self.JT_FEATURES:
            assert name in WinTwoStageModel.FEATURE_COLS, (
                f"{name} should be in WinTwoStageModel.FEATURE_COLS"
            )

    def test_jockey_trainer_combo_in_place_hit_feature_cols(self) -> None:
        """12特徴量がPlaceTwoStageModel.HIT_FEATURE_COLSに含まれる"""
        for name in self.JT_FEATURES:
            assert name in PlaceTwoStageModel.HIT_FEATURE_COLS, (
                f"{name} should be in PlaceTwoStageModel.HIT_FEATURE_COLS"
            )

    def test_jockey_trainer_combo_in_place_return_feature_cols(self) -> None:
        """12特徴量がPlaceTwoStageModel.RETURN_FEATURE_COLSに含まれる"""
        for name in self.JT_FEATURES:
            assert name in PlaceTwoStageModel.RETURN_FEATURE_COLS, (
                f"{name} should be in PlaceTwoStageModel.RETURN_FEATURE_COLS"
            )

    def test_no_duplicates(self) -> None:
        """FEATURE_COLSに重複がない"""
        for name, cols in [
            ("WinTwoStageModel", WinTwoStageModel.FEATURE_COLS),
            ("PlaceTwoStageModel.HIT", PlaceTwoStageModel.HIT_FEATURE_COLS),
            ("PlaceTwoStageModel.RETURN", PlaceTwoStageModel.RETURN_FEATURE_COLS),
        ]:
            assert len(cols) == len(set(cols)), (
                f"{name} FEATURE_COLS has duplicates"
            )

    def test_minimum_count(self) -> None:
        """FEATURE_COLSが最低50件である"""
        assert len(WinTwoStageModel.FEATURE_COLS) >= 50
        assert len(PlaceTwoStageModel.HIT_FEATURE_COLS) >= 54
        assert len(PlaceTwoStageModel.RETURN_FEATURE_COLS) >= 55


# ---------------------------------------------------------------------------
# INTER-01: 相対特徴量のFEATURE_COLS統合テスト
# ---------------------------------------------------------------------------


class TestRelativeFeaturesInFeatureCols:
    """INTER-01: オッズ相対+能力値相対特徴量のFEATURE_COLS統合テスト"""

    def test_stage1_has_rel_weight_zscore(self) -> None:
        """Stage1AbilityModel.FEATURE_COLSにrel_weight_zscoreが含まれる (per D-01)"""
        from models.stage1_ability_model import AbilityModel

        assert "rel_weight_zscore" in AbilityModel.FEATURE_COLS

    def test_win_has_odds_relative_features(self) -> None:
        """WinTwoStageModel.FEATURE_COLSにオッズ相対特徴量が含まれる"""
        expected = [
            "rel_popularity_rank_zscore",
            "rel_fuku_odds_zscore",
        ]
        for name in expected:
            assert name in WinTwoStageModel.FEATURE_COLS, (
                f"{name} should be in WinTwoStageModel.FEATURE_COLS"
            )

    def test_win_has_stage2_relative_features(self) -> None:
        """WinTwoStageModel.FEATURE_COLSに能力値相対特徴量が含まれる"""
        expected = [
            "rel_p_ability_win_zscore",
            "rel_p_ability_win_rank",
            "rel_odds_ability_deviation",
        ]
        for name in expected:
            assert name in WinTwoStageModel.FEATURE_COLS, (
                f"{name} should be in WinTwoStageModel.FEATURE_COLS"
            )

    def test_place_hit_has_relative_features(self) -> None:
        """PlaceTwoStageModel.HIT_FEATURE_COLSに相対特徴量が含まれる"""
        expected = [
            "rel_popularity_rank_zscore",
            "rel_fuku_odds_zscore",
            "rel_p_ability_win_zscore",
            "rel_p_ability_win_rank",
            "rel_odds_ability_deviation",
        ]
        for name in expected:
            assert name in PlaceTwoStageModel.HIT_FEATURE_COLS, (
                f"{name} should be in PlaceTwoStageModel.HIT_FEATURE_COLS"
            )

    def test_place_return_has_relative_features(self) -> None:
        """PlaceTwoStageModel.RETURN_FEATURE_COLSに相対特徴量が含まれる"""
        expected = [
            "rel_popularity_rank_zscore",
            "rel_fuku_odds_zscore",
            "rel_p_ability_win_zscore",
            "rel_p_ability_win_rank",
            "rel_odds_ability_deviation",
        ]
        for name in expected:
            assert name in PlaceTwoStageModel.RETURN_FEATURE_COLS, (
                f"{name} should be in PlaceTwoStageModel.RETURN_FEATURE_COLS"
            )

    def test_place_return_includes_all_win_features(self) -> None:
        """Place RETURN_FEATURE_COLSにWin FEATURE_COLSが全て含まれる"""
        for col in WinTwoStageModel.FEATURE_COLS:
            assert col in PlaceTwoStageModel.RETURN_FEATURE_COLS, (
                f"Win feature {col} missing from Place RETURN_FEATURE_COLS"
            )

    def test_compute_stage2_relative_features_in_pipeline(self) -> None:
        """_train_submodel内でcompute_stage2_relative_featuresが呼ばれている (grep検証)"""
        import inspect
        from pipelines.training_pipeline import TrainingPipelineV5

        source = inspect.getsource(TrainingPipelineV5._train_submodel)
        assert "compute_stage2_relative_features" in source, (
            "compute_stage2_relative_features not found in _train_submodel"
        )
