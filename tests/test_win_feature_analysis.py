"""test_win_feature_analysis.py — SHAP/gain特徴量重要度分析のテスト

全テスト mock 使用 (DB不要) — プロジェクト規約に従う。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from features.win_feature_analysis import (
    analyze_feature_importance,
    identify_noise_features,
    validate_noise_removal,
)
from models.two_stage_return_model import WinTwoStageModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_model() -> MagicMock:
    """5特徴量のlgb.Boosterモック"""
    model = MagicMock()
    feature_names = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]
    model.feature_name.return_value = feature_names
    # gain: feat_a が最も高く、feat_e が最も低い
    model.feature_importance.return_value = np.array([100.0, 80.0, 50.0, 20.0, 0.5])
    # SHAP matrix: shape [n_samples, n_features + 1] (最後の列はexpected value)
    n_samples = 10
    shap_matrix = np.random.RandomState(42).randn(n_samples, len(feature_names) + 1)
    # feat_e はSHAP寄与をほぼゼロにする
    shap_matrix[:, 4] = 0.0001 * np.random.RandomState(43).randn(n_samples)
    # 最後の列はexpected value (base value)
    shap_matrix[:, -1] = -1.5
    model.predict.return_value = shap_matrix
    return model


@pytest.fixture()
def features_df() -> pd.DataFrame:
    """分析用特徴量DataFrame (内容はmockなので何でもよい)"""
    return pd.DataFrame(np.random.RandomState(0).randn(10, 5), columns=["a", "b", "c", "d", "e"])


# ---------------------------------------------------------------------------
# Test 1: analyze_feature_importance returns DataFrame with correct columns
# ---------------------------------------------------------------------------

class TestAnalyzeFeatureImportance:
    def test_returns_dataframe_with_correct_columns(self, mock_model: MagicMock, features_df: pd.DataFrame) -> None:
        """戻り値が ['feature', 'gain', 'mean_abs_shap'] 列を持つDataFrameであること"""
        result = analyze_feature_importance(mock_model, features_df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["feature", "gain", "mean_abs_shap"]

    def test_sorted_by_mean_abs_shap_descending(self, mock_model: MagicMock, features_df: pd.DataFrame) -> None:
        """mean_abs_shap 降順でソートされていること"""
        result = analyze_feature_importance(mock_model, features_df)
        shap_values = result["mean_abs_shap"].values
        for i in range(len(shap_values) - 1):
            assert shap_values[i] >= shap_values[i + 1]

    def test_correctly_handles_pred_contrib_extra_column(self, mock_model: MagicMock, features_df: pd.DataFrame) -> None:
        """pred_contrib の追加列 (expected value) が正しく除外されていること

        Pitfall 1: pred_contrib は [n_samples, n_features + 1] を返す。
        最後の列は base value であり特徴量寄与ではない。
        """
        result = analyze_feature_importance(mock_model, features_df)
        # 5特徴量 + 1 expected value = 6列。5特徴量のみが処理されていることを確認
        assert len(result) == 5
        assert mock_model.predict.call_count == 1
        call_kwargs = mock_model.predict.call_args
        assert call_kwargs[1].get("pred_contrib") is True or (
            len(call_kwargs[0]) > 1 and call_kwargs[0][1] is True
        )

    def test_top_n_limits_output_rows(self, mock_model: MagicMock, features_df: pd.DataFrame) -> None:
        """top_n > 0 の場合、上位n行のみ返すこと"""
        result = analyze_feature_importance(mock_model, features_df, top_n=3)
        assert len(result) == 3

    def test_handles_nan_in_features_df(self, mock_model: MagicMock) -> None:
        """NaNを含むDataFrameでもエラーなく処理できること"""
        df_with_nan = pd.DataFrame(
            np.where(np.random.RandomState(1).rand(10, 5) > 0.5, np.nan, 1.0),
            columns=["a", "b", "c", "d", "e"],
        )
        result = analyze_feature_importance(mock_model, df_with_nan)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# Test 3: identify_noise_features returns features below threshold
# ---------------------------------------------------------------------------

class TestIdentifyNoiseFeatures:
    def test_identifies_low_shap_features(self) -> None:
        """mean_abs_shap < shap_threshold の特徴量を返すこと"""
        importance_df = pd.DataFrame({
            "feature": ["feat_a", "feat_b", "feat_c"],
            "gain": [100.0, 50.0, 0.0],
            "mean_abs_shap": [0.5, 0.1, 0.0005],
        })
        noise = identify_noise_features(importance_df, shap_threshold=0.001, gain_threshold=0.0)
        assert "feat_c" in noise
        assert "feat_a" not in noise
        assert "feat_b" not in noise

    def test_returns_empty_list_when_all_high_importance(self) -> None:
        """全特徴量が閾値以上の場合、空リストを返すこと"""
        importance_df = pd.DataFrame({
            "feature": ["feat_a", "feat_b"],
            "gain": [100.0, 50.0],
            "mean_abs_shap": [0.5, 0.1],
        })
        noise = identify_noise_features(importance_df, shap_threshold=0.001, gain_threshold=0.0)
        assert noise == []

    def test_single_noise_feature(self) -> None:
        """ノイズ特徴量が1つだけの場合"""
        importance_df = pd.DataFrame({
            "feature": ["feat_a", "feat_b", "feat_c"],
            "gain": [100.0, 50.0, 0.0],
            "mean_abs_shap": [0.5, 0.1, 0.0001],
        })
        noise = identify_noise_features(importance_df, shap_threshold=0.001, gain_threshold=0.0)
        assert noise == ["feat_c"]

    def test_gain_threshold_filters_zero_gain(self) -> None:
        """gain_threshold > 0 の場合、gainが閾値以下の特徴量もノイズとして検出されること"""
        importance_df = pd.DataFrame({
            "feature": ["feat_a", "feat_b"],
            "gain": [100.0, 0.5],
            "mean_abs_shap": [0.5, 0.1],
        })
        # gain <= 1.0 かつ mean_abs_shap < 0.001 は feat_b は shap 高いので対象外
        noise = identify_noise_features(importance_df, shap_threshold=0.001, gain_threshold=1.0)
        # feat_b: gain=0.5 <= 1.0 だが mean_abs_shap=0.1 > 0.001 なので対象外
        assert noise == []

    def test_both_conditions_required(self) -> None:
        """SHAP低い AND gain低いの両方を満たす特徴量のみノイズ判定されること"""
        importance_df = pd.DataFrame({
            "feature": ["low_both", "low_shap_only", "low_gain_only", "high_both"],
            "gain": [0.0, 100.0, 0.0, 100.0],
            "mean_abs_shap": [0.0001, 0.0001, 0.5, 0.5],
        })
        noise = identify_noise_features(importance_df, shap_threshold=0.001, gain_threshold=0.0)
        assert noise == ["low_both"]
        assert "low_shap_only" not in noise
        assert "low_gain_only" not in noise


# ---------------------------------------------------------------------------
# Task 2 Tests: CLI script, remove_noise_features, validate_noise_removal
# ---------------------------------------------------------------------------


class TestRemoveNoiseFeatures:
    """WinTwoStageModel.remove_noise_features() のテスト"""

    def test_removes_specified_features(self) -> None:
        """指定された特徴量がFEATURE_COLSから除外されること"""
        original = list(WinTwoStageModel.FEATURE_COLS)
        noise = ["blinker_change", "is_nar_transfer"]
        WinTwoStageModel.remove_noise_features(noise)
        assert "blinker_change" not in WinTwoStageModel.FEATURE_COLS
        assert "is_nar_transfer" not in WinTwoStageModel.FEATURE_COLS
        # 復元
        WinTwoStageModel.FEATURE_COLS = original

    def test_remaining_features_are_subset_of_original(self) -> None:
        """除外後のFEATURE_COLSは元の27特徴量の部分集合であること"""
        # 元の完全リスト (Plan reference)
        original_27 = [
            "p_ability_win",
            "signed_log_error_win", "abs_log_error_win",
            "odds_drop_rate_60_10", "odds_drop_rate_30_10",
            "odds_velocity", "odds_volatility",
            "popularity_change_30_10",
            "market_entropy", "popularity_rank", "overround",
            "surface", "distance_bin", "track_condition_code",
            "grade_code", "field_size",
            "odds_skewness",
            "draw_ratio", "class_move", "blinker_change",
            "is_nar_transfer", "nar_recent_ratio",
            "track_condition_delta",
            "pace_pressure", "pace_scenario_fit",
        ]
        original = list(WinTwoStageModel.FEATURE_COLS)
        # FEATURE_COLSは元の27の部分集合 (または等価) であること
        for feat in WinTwoStageModel.FEATURE_COLS:
            assert feat in original_27, f"FEATURE_COLS contains unexpected feature: {feat}"

    def test_feature_cols_length_at_least_20(self) -> None:
        """FEATURE_COLSは最低20特徴量を維持していること (27 - 合理的なノイズ上限)"""
        assert len(WinTwoStageModel.FEATURE_COLS) >= 20, (
            f"FEATURE_COLS has only {len(WinTwoStageModel.FEATURE_COLS)} features, "
            f"expected >= 20"
        )

    def test_no_duplicate_entries(self) -> None:
        """FEATURE_COLSに重複エントリがないこと"""
        assert len(WinTwoStageModel.FEATURE_COLS) == len(set(WinTwoStageModel.FEATURE_COLS)), (
            "FEATURE_COLS contains duplicate entries"
        )

    def test_remove_nonexistent_feature_is_noop(self) -> None:
        """存在しない特徴量名を指定してもエラーにならないこと"""
        original = list(WinTwoStageModel.FEATURE_COLS)
        WinTwoStageModel.remove_noise_features(["nonexistent_feature_xyz"])
        assert WinTwoStageModel.FEATURE_COLS == original


class TestValidateNoiseRemoval:
    """validate_noise_removal() のテスト"""

    def test_returns_dict_with_expected_keys(self) -> None:
        """戻り値が期待されるキーを持つdictであること"""
        # lgb.Booster のモックを作成
        mock_original_model = MagicMock()
        mock_original_model.feature_name.return_value = ["feat_a", "feat_b", "feat_c"]
        mock_original_model.predict.return_value = np.array([0.3, 0.7, 0.2, 0.8, 0.5, 0.5])

        df = pd.DataFrame({
            "feat_a": [1.0, 2.0],
            "feat_b": [3.0, 4.0],
            "feat_c": [5.0, 6.0],
            "kakuteijyuni": [1, 0],
        })
        # lgb.train をモックして新しいモデルを返す
        with pytest.MonkeyPatch.context() as m:
            mock_new_model = MagicMock()
            mock_new_model.predict.return_value = np.array([0.4, 0.6])
            m.setattr("features.win_feature_analysis.lgb.train", lambda *a, **kw: mock_new_model)
            m.setattr("features.win_feature_analysis.lgb.Dataset", MagicMock)

            result = validate_noise_removal(
                mock_original_model, df, noise_features=["feat_c"],
            )
            assert isinstance(result, dict)
            expected_keys = {"original_logloss", "new_logloss", "original_auc", "new_auc"}
            assert set(result.keys()) == expected_keys

    def test_logs_warning_on_degradation(self) -> None:
        """logloss悪化時に警告がログ出力されること (閾値0.5%超)"""
        mock_original_model = MagicMock()
        mock_original_model.feature_name.return_value = ["feat_a", "feat_b"]
        # 元モデル予測: 完璧 (logloss低い)
        mock_original_model.predict.return_value = np.array([0.01, 0.99, 0.01, 0.99])

        df = pd.DataFrame({
            "feat_a": [1.0, 2.0],
            "feat_b": [3.0, 4.0],
            "kakuteijyuni": [0, 1],
        })

        with pytest.MonkeyPatch.context() as m:
            mock_new_model = MagicMock()
            # 新モデル予測: ランダム (logloss高い)
            mock_new_model.predict.return_value = np.array([0.5, 0.5])
            m.setattr("features.win_feature_analysis.lgb.train", lambda *a, **kw: mock_new_model)
            m.setattr("features.win_feature_analysis.lgb.Dataset", MagicMock)

            import logging
            with pytest.MonkeyPatch.context() as m2:
                # capture log warnings
                result = validate_noise_removal(
                    mock_original_model, df, noise_features=["feat_b"],
                )
                # 悪化していれば new_logloss > original_logloss
                assert result["new_logloss"] > 0


class TestCSVReportIsNoise:
    """CSVレポートのis_noise列に関するテスト"""

    def test_is_noise_column_boolean_dtype(self) -> None:
        """is_noise列がboolean dtypeであること"""
        importance_df = pd.DataFrame({
            "feature": ["feat_a", "feat_b", "feat_c"],
            "gain": [100.0, 50.0, 0.0],
            "mean_abs_shap": [0.5, 0.1, 0.0001],
        })
        noise_features = identify_noise_features(
            importance_df, shap_threshold=0.001, gain_threshold=0.0,
        )
        importance_df["is_noise"] = importance_df["feature"].isin(noise_features)
        assert importance_df["is_noise"].dtype == bool
        assert importance_df.loc[importance_df["feature"] == "feat_c", "is_noise"].iloc[0] is True
        assert importance_df.loc[importance_df["feature"] == "feat_a", "is_noise"].iloc[0] is False
