"""3層 POST_RACE 漏洩検証テスト (SAFE-01)

Layer 1: build_all() 出力に POST_RACE_COLS が含まれないことを検証
Layer 2: 全モデルの FEATURE_COLS と POST_RACE_COLS の積集合が空であることを検証
Layer 3: predict() 入力での POST_RACE_COLS 伝播を検証
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from domain.types import POST_RACE_COLS
from features.feature_engine import FeatureEngine
from models.conformal_ev_model import ConformalEVModel
from models.ev_correction_model import EVCorrectionModel, PlaceEVCorrectionModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_race_df() -> pd.DataFrame:
    """最小限の race_df"""
    return pd.DataFrame(
        {
            "race_id": ["R001"] * 3,
            "trackcd": [11] * 3,
            "kyori": [1600] * 3,
            "syussotosu": [3] * 3,
            "surface": ["turf"] * 3,
            "gradecd": ["_"] * 3,
            "jyocd": [5] * 3,
        }
    )


def _make_entry_df() -> pd.DataFrame:
    """entry_df with some POST_RACE columns included"""
    return pd.DataFrame(
        {
            "race_id": ["R001"] * 3,
            "umaban": [1, 2, 3],
            "odds": [3.0, 5.0, 8.0],
            "kakuteijyuni": [1, 2, 3],
            "ninki": [1, 2, 3],
            "time": [95.0, 95.5, 96.0],
            "bataijyu": [480.0, 470.0, 490.0],
        }
    )


def _make_odds_df() -> pd.DataFrame:
    """odds_df"""
    return pd.DataFrame(
        {
            "race_id": ["R001"] * 3,
            "umaban": [1, 2, 3],
            "tanodds": [3.0, 5.0, 8.0],
            "fukuoddslow": [1.5, 2.0, 2.5],
            "tanninki": [1, 2, 3],
        }
    )


# ---------------------------------------------------------------------------
# Layer 1: build_all() output verification
# ---------------------------------------------------------------------------


class TestPostRaceLeakage:
    """3層 POST_RACE 漏洩検証 CI テスト"""

    def test_build_all_output_no_post_race_cols(self) -> None:
        """Layer 1: build_all() の出力に POST_RACE_COLS が含まれない"""
        engine = FeatureEngine(use_cache=False)
        race_df = _make_race_df()
        entry_df = _make_entry_df()
        odds_df = _make_odds_df()

        result = engine.build_all(race_df, entry_df, odds_df)

        post_race_in_output = set(result.columns) & set(POST_RACE_COLS)
        assert not post_race_in_output, (
            f"POST_RACE_COLS found in build_all() output: {post_race_in_output}"
        )

    def test_model_feature_cols_no_post_race(self) -> None:
        """Layer 2: 全モデルの FEATURE_COLS に POST_RACE_COLS が含まれない"""
        from models.stage1_ability_model import AbilityModel
        from models.two_stage_return_model import PlaceTwoStageModel, WinTwoStageModel

        model_classes = [
            ("AbilityModel", AbilityModel),
            ("WinTwoStageModel", WinTwoStageModel),
            ("EVCorrectionModel", EVCorrectionModel),
            ("PlaceEVCorrectionModel", PlaceEVCorrectionModel),
            ("ConformalEVModel", ConformalEVModel),
        ]

        for model_name, model_cls in model_classes:
            feature_cols = getattr(model_cls, "FEATURE_COLS", None)
            assert feature_cols is not None, (
                f"{model_name} has no FEATURE_COLS class variable"
            )
            overlap = set(feature_cols) & set(POST_RACE_COLS)
            assert not overlap, (
                f"{model_name}.FEATURE_COLS contains POST_RACE_COLS: {overlap}"
            )

        # PlaceTwoStageModel uses HIT_FEATURE_COLS and RETURN_FEATURE_COLS
        place_feature_lists = [
            ("PlaceTwoStageModel.HIT_FEATURE_COLS", PlaceTwoStageModel.HIT_FEATURE_COLS),
            ("PlaceTwoStageModel.RETURN_FEATURE_COLS", PlaceTwoStageModel.RETURN_FEATURE_COLS),
        ]
        for list_name, cols in place_feature_lists:
            overlap = set(cols) & set(POST_RACE_COLS)
            assert not overlap, (
                f"{list_name} contains POST_RACE_COLS: {overlap}"
            )

    def test_ev_correction_odds_col_uses_pre_race_odds(self) -> None:
        """Layer 3: EVCorrectionModel.correct_ev() が confirmed_odds を使用しない"""
        # Create a trained EVCorrectionModel with mock models
        model = EVCorrectionModel()
        model._trained = True

        # Mock the P/E correction models
        mock_p = MagicMock()
        mock_p.predict.return_value = np.array([0.5, -0.3, 0.2])  # raw margins
        mock_p.best_iteration = 0
        model.p_correction_model = mock_p

        mock_e = MagicMock()
        mock_e.predict.return_value = np.array([0.1, -0.05, 0.2])  # log corrections
        mock_e.best_iteration = 0
        model.e_correction_model = mock_e

        # Test with DataFrame containing both "odds" and "confirmed_odds"
        df = pd.DataFrame(
            {
                "race_id": ["R1", "R1", "R1"],
                "p_win_pred": [0.4, 0.3, 0.3],
                "e_return_win_pred": [3.0, 5.0, 8.0],
                "odds": [3.0, 5.0, 8.0],          # pre-race odds
                "confirmed_odds": [2.8, 4.5, 9.0],  # post-race odds (different)
                "surface": ["turf"] * 3,
                "distance_bin": ["mile"] * 3,
                "track_condition_code": [1.0] * 3,
                "field_size": [12.0] * 3,
                "market_entropy": [2.5] * 3,
                "popularity_rank": [1.0, 2.0, 3.0],
                "implied_prob_hhi": [0.15] * 3,
                "jockey_wr_overall": [0.15] * 3,
                "jockey_wr_distance": [0.15] * 3,
                "jockey_wr_venue": [0.15] * 3,
                "jockey_prize_log": [5.0] * 3,
                "trainer_wr_overall": [0.12] * 3,
                "trainer_wr_distance": [0.12] * 3,
                "trainer_wr_venue": [0.12] * 3,
                "trainer_prize_log": [4.0] * 3,
                "jt_combo_wr": [0.13] * 3,
                "jt_combo_place_rate": [0.35] * 3,
                "jt_combo_starts": [10.0] * 3,
                "jt_combo_prize_log": [3.0] * 3,
                "signed_log_error_win": [0.1] * 3,
                "abs_log_error_win": [0.2] * 3,
                "odds_skewness": [0.5] * 3,
                # 市場クロス整合性 (MCF-07)
                "rl_favorite_in_wide_top1": [1.0] * 3,
                "rl_trio_overlap": [2.0] * 3,
                "rl_market_consistency": [1.0] * 3,
                "rl_trio_odds_ratio": [0.8] * 3,
                "rl_wide_harville_ratio": [1.1] * 3,
            }
        )

        # With ev_odds_band_scales set, correct_ev should use "odds" not "confirmed_odds"
        # Use actual band names from OddsBandFilter.BAND_NAMES
        model.ev_odds_band_scales = {"1.0-3.0": 1.1, "3.0-10.0": 0.95, "10.0-30.0": 1.0, "30.0+": 1.0}

        result = model.correct_ev(df)

        # Verify "ev_win_calibrated" exists and values are computed using "odds"
        assert "ev_win_calibrated" in result.columns
        # The key check: odds-band scaling should use "odds" (3.0, 5.0, 8.0)
        # not "confirmed_odds" (2.8, 4.5, 9.0)
        # All three odds values (3.0, 5.0, 8.0) fall into band "3.0-10.0" (scale=0.95),
        # so calibrated values should differ from uncalibrated by that factor.
        assert "ev_win_corrected" in result.columns
        for i in range(3):
            corrected = result["ev_win_corrected"].iloc[i]
            calibrated = result["ev_win_calibrated"].iloc[i]
            if corrected != 0.0:
                ratio = calibrated / corrected
                assert abs(ratio - 0.95) < 1e-6, (
                    f"Row {i}: calibrated/corrected = {ratio:.6f}, expected 0.95"
                )

    def test_conformal_ev_feature_cols_whitelist(self) -> None:
        """Layer 2+: ConformalEVModel.FEATURE_COLS に POST_RACE_COLS が含まれない"""
        overlap = set(ConformalEVModel.FEATURE_COLS) & set(POST_RACE_COLS)
        assert not overlap, (
            f"ConformalEVModel.FEATURE_COLS whitelist contains POST_RACE_COLS: {overlap}"
        )


# ---------------------------------------------------------------------------
# Race-Level Features: POST_RACE Safety Verification
# ---------------------------------------------------------------------------


class TestRaceLevelFeatures:
    """rl_* 特徴量の POST_RACE 安全性検証"""

    def test_race_level_features_no_post_race_input(self) -> None:
        """compute_race_level_features() のソースコードに POST_RACE_COLS が含まれない"""
        import ast
        import inspect

        from features.race_level_features import compute_race_level_features

        source = inspect.getsource(compute_race_level_features)
        tree = ast.parse(source)

        # ソースコード内の全ての文字列リテラルを収集
        string_literals: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                string_literals.add(node.value)

        # POST_RACE_COLS に含まれる列名が文字列リテラルとして参照されていないことを確認
        post_race_referenced = string_literals & set(POST_RACE_COLS)
        assert not post_race_referenced, (
            f"compute_race_level_features() references POST_RACE column names: "
            f"{post_race_referenced}"
        )

    def test_rl_feature_cols_not_in_post_race(self) -> None:
        """6つの rl_* 列名が POST_RACE_COLS に含まれないことを検証"""
        from features.race_level_features import RL_COLS

        overlap = set(RL_COLS) & set(POST_RACE_COLS)
        assert not overlap, (
            f"rl_* feature column names overlap with POST_RACE_COLS: {overlap}"
        )

    def test_build_all_produces_rl_features(self) -> None:
        """build_all() の出力に6つの rl_* 列が含まれることを検証"""
        from features.race_level_features import RL_COLS

        engine = FeatureEngine(use_cache=False)
        race_df = _make_race_df()
        entry_df = _make_entry_df()
        odds_df = _make_odds_df()

        result = engine.build_all(race_df, entry_df, odds_df)

        for col in RL_COLS:
            assert col in result.columns, (
                f"build_all() output missing rl_* column: {col}. "
                f"Available columns: {sorted(result.columns.tolist())}"
            )

        # 値がNaNでないことを確認 (テストデータには有効なtanoddsがあるため)
        for col in RL_COLS:
            assert result[col].notna().any(), (
                f"build_all() output column {col} is all-NaN "
                f"(expected at least one valid value from test data)"
            )


# ---------------------------------------------------------------------------
# Market Cross-Consistency Features: POST_RACE Safety Verification
# ---------------------------------------------------------------------------


class TestMarketCrossFeatures:
    """MCF特徴量の POST_RACE 安全性検証 + build_all統合テスト"""

    def test_market_cross_features_no_post_race_input(self) -> None:
        """compute_market_cross_features() のソースコードに POST_RACE_COLS が含まれない"""
        import ast
        import inspect

        from features.market_cross_features import compute_market_cross_features

        source = inspect.getsource(compute_market_cross_features)
        tree = ast.parse(source)

        # ソースコード内の全ての文字列リテラルを収集
        string_literals: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                string_literals.add(node.value)

        # POST_RACE_COLS に含まれる列名が文字列リテラルとして参照されていないことを確認
        post_race_referenced = string_literals & set(POST_RACE_COLS)
        assert not post_race_referenced, (
            f"compute_market_cross_features() references POST_RACE column names: "
            f"{post_race_referenced}"
        )

    def test_mcf_cols_not_in_post_race(self) -> None:
        """MCF_COLSの5列名がPOST_RACE_COLSに含まれないことを検証"""
        from features.market_cross_features import MCF_COLS

        overlap = set(MCF_COLS) & set(POST_RACE_COLS)
        assert not overlap, (
            f"MCF column names overlap with POST_RACE_COLS: {overlap}"
        )

    def test_build_all_produces_mcf_features(self) -> None:
        """build_all() の出力に5つのMCF列が含まれることを検証 (NaNでも可)"""
        from features.market_cross_features import MCF_COLS

        engine = FeatureEngine(use_cache=False)
        race_df = _make_race_df()
        entry_df = _make_entry_df()
        odds_df = _make_odds_df()

        result = engine.build_all(race_df, entry_df, odds_df)

        for col in MCF_COLS:
            assert col in result.columns, (
                f"build_all() output missing MCF column: {col}. "
                f"Available columns: {sorted(result.columns.tolist())}"
            )

    def test_all_models_have_mcf_features(self) -> None:
        """全12モデルのFEATURE_COLSに5つのMCF特徴量が含まれる"""
        from features.market_cross_features import MCF_COLS
        from models.market_model import MarketModel
        from models.place_ability_model import PlaceAbilityModel
        from models.race_quality_screener import RaceQualityScreener
        from models.regime_detector import RegimeDetector
        from models.stage1_ability_model import AbilityModel
        from models.two_stage_return_model import PlaceTwoStageModel, WinTwoStageModel
        from models.wide_two_stage_model import WideTwoStageModel

        model_feature_lists = [
            ("AbilityModel.FEATURE_COLS", AbilityModel.FEATURE_COLS),
            ("MarketModel.FEATURE_COLS", MarketModel.FEATURE_COLS),
            ("RegimeDetector.FEATURE_COLS", RegimeDetector.FEATURE_COLS),
            ("PlaceAbilityModel.FEATURE_COLS", PlaceAbilityModel.FEATURE_COLS),
            ("RaceQualityScreener.FEATURE_COLS", RaceQualityScreener.FEATURE_COLS),
            ("WideTwoStageModel.SHARED_FEATURE_COLS", WideTwoStageModel.SHARED_FEATURE_COLS),
            ("WinTwoStageModel.FEATURE_COLS", WinTwoStageModel.FEATURE_COLS),
            ("PlaceTwoStageModel.HIT_FEATURE_COLS", PlaceTwoStageModel.HIT_FEATURE_COLS),
            ("PlaceTwoStageModel.RETURN_FEATURE_COLS", PlaceTwoStageModel.RETURN_FEATURE_COLS),
            ("EVCorrectionModel.FEATURE_COLS", EVCorrectionModel.FEATURE_COLS),
            ("PlaceEVCorrectionModel.FEATURE_COLS", PlaceEVCorrectionModel.FEATURE_COLS),
            ("ConformalEVModel.FEATURE_COLS", ConformalEVModel.FEATURE_COLS),
        ]

        for list_name, cols in model_feature_lists:
            for mcf_col in MCF_COLS:
                assert mcf_col in cols, (
                    f"{list_name} missing MCF feature: {mcf_col}"
                )

    def test_all_models_have_rl_features(self) -> None:
        """全12モデルのFEATURE_COLSに6つのrl_*レースレベル特徴量が含まれる"""
        from features.race_level_features import RL_COLS
        from models.market_model import MarketModel
        from models.place_ability_model import PlaceAbilityModel
        from models.race_quality_screener import RaceQualityScreener
        from models.regime_detector import RegimeDetector
        from models.stage1_ability_model import AbilityModel
        from models.two_stage_return_model import PlaceTwoStageModel, WinTwoStageModel
        from models.wide_two_stage_model import WideTwoStageModel

        model_feature_lists = [
            ("AbilityModel.FEATURE_COLS", AbilityModel.FEATURE_COLS),
            ("MarketModel.FEATURE_COLS", MarketModel.FEATURE_COLS),
            ("RegimeDetector.FEATURE_COLS", RegimeDetector.FEATURE_COLS),
            ("PlaceAbilityModel.FEATURE_COLS", PlaceAbilityModel.FEATURE_COLS),
            ("RaceQualityScreener.FEATURE_COLS", RaceQualityScreener.FEATURE_COLS),
            ("WideTwoStageModel.SHARED_FEATURE_COLS", WideTwoStageModel.SHARED_FEATURE_COLS),
            ("WinTwoStageModel.FEATURE_COLS", WinTwoStageModel.FEATURE_COLS),
            ("PlaceTwoStageModel.HIT_FEATURE_COLS", PlaceTwoStageModel.HIT_FEATURE_COLS),
            ("PlaceTwoStageModel.RETURN_FEATURE_COLS", PlaceTwoStageModel.RETURN_FEATURE_COLS),
            ("EVCorrectionModel.FEATURE_COLS", EVCorrectionModel.FEATURE_COLS),
            ("PlaceEVCorrectionModel.FEATURE_COLS", PlaceEVCorrectionModel.FEATURE_COLS),
            ("ConformalEVModel.FEATURE_COLS", ConformalEVModel.FEATURE_COLS),
        ]

        for list_name, cols in model_feature_lists:
            for rl_col in RL_COLS:
                assert rl_col in cols, (
                    f"{list_name} missing rl_* feature: {rl_col}"
                )

    def test_gpd_category_map_has_rl_features(self) -> None:
        """GPD FEATURE_CATEGORY_MAPに6つのrl_*特徴量がmarket分類で登録されている"""
        from features.race_level_features import RL_COLS
        from models.gpd_diagnostics import FEATURE_CATEGORY_MAP

        for rl_col in RL_COLS:
            assert rl_col in FEATURE_CATEGORY_MAP, (
                f"FEATURE_CATEGORY_MAP missing rl_* feature: {rl_col}"
            )
            assert FEATURE_CATEGORY_MAP[rl_col] == "market", (
                f"FEATURE_CATEGORY_MAP[{rl_col}] = '{FEATURE_CATEGORY_MAP[rl_col]}', "
                f"expected 'market'"
            )
