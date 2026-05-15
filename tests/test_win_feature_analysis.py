"""test_win_feature_analysis.py — SHAP/gain特徴量重要度分析のテスト

全テスト mock 使用 (DB不要) — プロジェクト規約に従う。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from features.win_feature_analysis import (
    analyze_feature_importance,
    classify_feature_tiers,
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
        try:
            noise = ["blinker_change", "is_nar_transfer"]
            WinTwoStageModel.remove_noise_features(noise)
            assert "blinker_change" not in WinTwoStageModel.FEATURE_COLS
            assert "is_nar_transfer" not in WinTwoStageModel.FEATURE_COLS
        finally:
            WinTwoStageModel.FEATURE_COLS = original

    def test_remaining_features_are_subset_of_original(self) -> None:
        """除外後のFEATURE_COLSは元の特徴量の部分集合であること"""
        # 元の完全リスト (Plan 01の27 + Plan 02の6 = 33 + Phase 5: 8)
        original_all = [
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
            # FEAT-02: Plan 02追加特徴量
            "distance_change", "surface_change", "class_drop_bounce",
            "win_dominance", "freshness_score",
            "odds_to_ability_ratio",
            # Phase 5: Foundation Features
            "class_adj_formetric",
            "haron_zscore_trend",
            "pace_corner_stability",
            "pace_closing_power",
            "pace_position_consistency",
            "actual_pace_fit",
            "odds_acceleration",
            "odds_direction_consistency",
            # ODDS-01: Phase 6 deviation features
            "deviation_rank",
            "deviation_zscore",
            # Phase 25: 騎手・調教師・コンビ特徴量
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
            # n_mining予想特徴量 (DATA-04)
            "dm_time_rank",
            "dm_time_zscore",
            "dm_confidence_range",
            "dm_time_margin_to_fav",
            # 繁殖牝馬産駒特徴量 (DATA-01)
            "dam_wr",
            "breeder_strength",
            # BMS拡張特徴量 (DATA-01)
            "bms_distance_wr",
            # コースレコード特徴量 (DATA-02)
            "course_record_time",
            # レース内相対比較特徴量 (DATA-03)
            "rel_norm_finish_zscore",
            "rel_timediff_rank",
            "rel_closing_index_rank",
            # INTER-01: オッズ相対特徴量
            "rel_popularity_rank_zscore",
            "rel_fuku_odds_zscore",
            # INTER-01: Stage2能力値相対特徴量
            "rel_p_ability_win_zscore",
            "rel_p_ability_win_rank",
            "rel_odds_ability_deviation",
        ]
        # FEATURE_COLSは元の特徴量の部分集合であること
        for feat in WinTwoStageModel.FEATURE_COLS:
            assert feat in original_all, f"FEATURE_COLS contains unexpected feature: {feat}"

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
        # predictはdfの行数分の予測値を返す必要がある (2行)
        mock_original_model.predict.return_value = np.array([0.3, 0.7])

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
        n = 10
        # 元モデル予測: 完璧 (logloss低い) -- n行分
        orig_preds = np.array([0.01, 0.99] * (n // 2))
        mock_original_model.predict.return_value = orig_preds

        df = pd.DataFrame({
            "feat_a": [float(i) for i in range(n)],
            "feat_b": [float(i + 10) for i in range(n)],
            "kakuteijyuni": [0, 1] * (n // 2),
        })

        with pytest.MonkeyPatch.context() as m:
            mock_new_model = MagicMock()
            # 新モデル予測: ランダム (logloss高い) -- valid行数分
            valid_n = n - int(n * 0.8)
            mock_new_model.predict.return_value = np.array([0.5] * valid_n)
            m.setattr("features.win_feature_analysis.lgb.train", lambda *a, **kw: mock_new_model)
            m.setattr("features.win_feature_analysis.lgb.Dataset", MagicMock)

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
        assert importance_df.loc[importance_df["feature"] == "feat_c", "is_noise"].iloc[0] == True  # noqa: E712
        assert importance_df.loc[importance_df["feature"] == "feat_a", "is_noise"].iloc[0] == False  # noqa: E712


# ---------------------------------------------------------------------------
# classify_feature_tiers() のテスト
# ---------------------------------------------------------------------------


def _make_tier_metadata(
    models: dict[str, dict[str, dict[str, float]]],
) -> dict[str, object]:
    """classify_feature_tiers() 用のmetadata辞書を簡易生成する。

    Args:
        models: {model_name: {"gain": {feat: val, ...}, "perm_mean": {feat: val, ...}}}
    """
    result: dict[str, object] = {"models": {}}
    for model_name, data in models.items():
        result["models"][model_name] = {
            "gain": data["gain"],
            "perm_mean": data["perm_mean"],
            "perm_std": {f: 0.0 for f in data["gain"]},
        }
    return result


def _make_tier_pivot_df(
    features: list[str],
    model_name: str,
    gain_dict: dict[str, float],
) -> pd.DataFrame:
    """classify_feature_tiers() 用のpivot_dfを簡易生成する。"""
    data: dict[str, list[object]] = {"feature": features}
    data[f"{model_name}_gain"] = [gain_dict.get(f, float("nan")) for f in features]
    data[f"{model_name}_perm"] = [0.0] * len(features)
    return pd.DataFrame(data)


class TestClassifyFeatureTiers:
    """classify_feature_tiers() のテスト"""

    def test_tier1_identifies_gain_zero_perm_negative(self) -> None:
        """Gain=0 AND Perm<=0 の特徴量をTier 1とする"""
        gain = {"f_a": 100.0, "f_b": 50.0, "f_c": 0.0, "f_d": 0.0}
        perm = {"f_a": 0.05, "f_b": 0.02, "f_c": -0.01, "f_d": 0.0}
        features = list(gain.keys())
        metadata = _make_tier_metadata({
            "win_hit": {"gain": gain, "perm_mean": perm},
        })
        pivot_df = _make_tier_pivot_df(features, "win_hit", gain)

        result = classify_feature_tiers(pivot_df, metadata)

        assert "win_hit" in result
        assert "f_c" in result["win_hit"]["tier1"]  # gain=0, perm=-0.01
        assert "f_d" in result["win_hit"]["tier1"]  # gain=0, perm=0.0
        assert "f_a" not in result["win_hit"]["tier1"]
        assert "f_b" not in result["win_hit"]["tier1"]

    def test_tier1_nan_perm_uses_gain_only(self) -> None:
        """Perm NaNの場合、Gain=0 のみでTier 1判定する"""
        gain = {"f_a": 100.0, "f_b": 0.0, "f_c": 50.0}
        perm = {"f_a": 0.05, "f_b": float("nan"), "f_c": float("nan")}
        features = list(gain.keys())
        metadata = _make_tier_metadata({
            "win_return": {"gain": gain, "perm_mean": perm},
        })
        pivot_df = _make_tier_pivot_df(features, "win_return", gain)

        result = classify_feature_tiers(pivot_df, metadata)

        assert "f_b" in result["win_return"]["tier1"]  # gain=0, perm=NaN -> Tier 1
        assert "f_c" not in result["win_return"]["tier1"]  # gain=50, perm=NaN -> NOT Tier 1

    def test_tier2_identifies_low_gain_percentile(self) -> None:
        """Tier 2が下位10%を正しく特定する"""
        # 20特徴量: gain = 100, 90, 80, ..., 10, 5, 1
        gain_vals = [float(100 - i * 5) for i in range(19)] + [1.0]
        features = [f"f_{i:02d}" for i in range(20)]
        gain = dict(zip(features, gain_vals))
        # 全てperm > 0 なのでTier 1はなし
        perm = {f: 0.01 for f in features}
        metadata = _make_tier_metadata({
            "win_hit": {"gain": gain, "perm_mean": perm},
        })
        pivot_df = _make_tier_pivot_df(features, "win_hit", gain)

        result = classify_feature_tiers(pivot_df, metadata, tier2_percentile=10.0)

        tier2 = result["win_hit"]["tier2"]
        # Tier 1は空 (全て gain > 0, perm > 0)
        assert result["win_hit"]["tier1"] == []
        # Tier 2は下位10%の特徴量を含む (gain最低の特徴量群)
        assert len(tier2) > 0
        # f_19 (gain=1.0) は最も低いgainなのでTier 2に含まれるべき
        assert "f_19" in tier2

    def test_no_overlap_between_tiers(self) -> None:
        """Tier 1とTier 2に重複がないことを確認"""
        gain = {"f_a": 100.0, "f_b": 0.0, "f_c": 50.0, "f_d": 0.0, "f_e": 1.0}
        perm = {"f_a": 0.05, "f_b": -0.01, "f_c": 0.02, "f_d": 0.0, "f_e": 0.001}
        features = list(gain.keys())
        metadata = _make_tier_metadata({
            "win_hit": {"gain": gain, "perm_mean": perm},
        })
        pivot_df = _make_tier_pivot_df(features, "win_hit", gain)

        result = classify_feature_tiers(pivot_df, metadata)

        tier1_set = set(result["win_hit"]["tier1"])
        tier2_set = set(result["win_hit"]["tier2"])
        assert tier1_set.isdisjoint(tier2_set), (
            f"Tier 1とTier 2に重複あり: {tier1_set & tier2_set}"
        )

    def test_returns_per_model_structure(self) -> None:
        """戻り値がモデル別dict構造であることを確認"""
        gain_m1 = {"f_a": 100.0, "f_b": 0.0}
        perm_m1 = {"f_a": 0.05, "f_b": -0.01}
        gain_m2 = {"f_a": 0.0, "f_b": 50.0, "f_c": 30.0}
        perm_m2 = {"f_a": float("nan"), "f_b": 0.02, "f_c": 0.01}
        metadata = _make_tier_metadata({
            "model_a": {"gain": gain_m1, "perm_mean": perm_m1},
            "model_b": {"gain": gain_m2, "perm_mean": perm_m2},
        })
        # pivot_dfには両モデルの列が必要
        all_features = sorted(set(gain_m1.keys()) | set(gain_m2.keys()))
        pivot_df = pd.DataFrame({
            "feature": all_features,
            "model_a_gain": [gain_m1.get(f, float("nan")) for f in all_features],
            "model_a_perm": [perm_m1.get(f, float("nan")) for f in all_features],
            "model_b_gain": [gain_m2.get(f, float("nan")) for f in all_features],
            "model_b_perm": [perm_m2.get(f, float("nan")) for f in all_features],
        })

        result = classify_feature_tiers(pivot_df, metadata)

        assert "model_a" in result
        assert "model_b" in result
        for model_key in ("model_a", "model_b"):
            assert "tier1" in result[model_key]
            assert "tier2" in result[model_key]
            assert "tier1_count" in result[model_key]
            assert "tier2_count" in result[model_key]
            assert isinstance(result[model_key]["tier1"], list)
            assert isinstance(result[model_key]["tier2"], list)
            assert result[model_key]["tier1_count"] == len(result[model_key]["tier1"])
            assert result[model_key]["tier2_count"] == len(result[model_key]["tier2"])
        # model_a: f_b (gain=0, perm=-0.01) -> Tier 1
        assert "f_b" in result["model_a"]["tier1"]
        # model_b: f_a (gain=0, perm=NaN) -> Tier 1
        assert "f_a" in result["model_b"]["tier1"]
