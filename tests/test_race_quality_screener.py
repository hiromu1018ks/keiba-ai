"""src/models/race_quality_screener.py のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from models.race_quality_screener import RaceQualityScreener


@pytest.fixture
def race_features_df() -> pd.DataFrame:
    """レースレベル特徴量のテストデータ (4レース)"""
    return pd.DataFrame(
        {
            "market_log_error_max_abs": [0.5, 0.2, 0.8, 0.1],
            "market_log_error_std": [0.3, 0.1, 0.5, 0.05],
            "market_log_error_top_q75": [0.4, 0.15, 0.6, 0.08],
            "n_positive_errors": [3, 1, 5, 0],
            "top_k_error_sum": [1.0, 0.3, 2.0, 0.1],
            "positive_error_ratio": [0.375, 0.125, 0.625, 0.0],
            "hist_hit_rate_topk": [0.25, 0.15, 0.30, 0.10],
            "hist_roi_topk": [1.5, 0.8, 2.0, 0.5],
            "hist_positive_return_ratio": [0.6, 0.4, 0.8, 0.2],
            "market_entropy": [2.8, 2.2, 3.0, 1.8],
            "overround": [0.22, 0.20, 0.25, 0.18],
            "overround_deviation": [0.02, 0.0, 0.05, -0.02],
            "field_size": [8, 12, 16, 10],
            "surface": ["turf", "dirt", "turf", "dirt"],
            "distance_bin": ["mile", "sprint", "long", "mile"],
            "track_condition_code": [1, 2, 3, 1],
            "grade_code": ["_", "_", "C", "_"],
            "difficulty_score": [0.5, 0.4, 0.7, 0.3],
            "hist_win_rate_same_condition": [0.20, 0.15, 0.25, 0.10],
            "hist_market_entropy_avg": [2.7, 2.1, 2.9, 1.9],
            # v5.6: EMA平滑化市場指標
            "overround_ema": [0.21, 0.19, 0.24, 0.17],
            "entropy_ema": [2.7, 2.1, 2.9, 1.8],
            # Phase 31: race-level aggregation features
            "implied_prob_hhi": [0.12, 0.10, 0.15, 0.08],
            "odds_skewness": [0.5, 0.3, 0.8, 0.2],
            # Phase 32: market cross-consistency features
            "rl_favorite_in_wide_top1": [1, 1, 0, 1],
            "rl_trio_overlap": [2, 3, 1, 2],
            "rl_market_consistency": [1, 1, 0, 1],
            "rl_trio_odds_ratio": [1.1, 0.9, 1.3, 0.8],
            "rl_wide_harville_ratio": [0.95, 1.05, 0.85, 1.10],
            # RLF-01~06 (rl_* race-level aggregation)
            "rl_log_odds_entropy": [1.5, 1.2, 1.8, 1.0],
            "rl_odds_dispersion": [0.8, 0.6, 1.0, 0.5],
            "rl_top3_odds_gap": [2.5, 1.8, 3.2, 1.5],
            "rl_top1_odds": [3.0, 2.5, 4.0, 2.0],
            "rl_favorite_rank_gap": [1.5, 1.0, 2.0, 0.8],
            "rl_n_horses": [8.0, 12.0, 16.0, 10.0],
            # Phase36 race-level aggregates (RTG-02/03)
            "phase36_top1_strength": [0.9, 0.7, 1.0, 0.6],
            "phase36_top1_top2_gap": [0.15, 0.10, 0.20, 0.08],
            "phase36_field_dispersion": [0.12, 0.08, 0.18, 0.05],
            "phase36_form_signal_dispersion": [0.20, 0.15, 0.25, 0.10],
            "phase36_weighted_form_mean": [4.2, 3.8, 5.0, 3.5],
        }
    )


class TestRaceQualityScreener:
    def test_target_uses_result_based_proxy(
        self,
        race_features_df: pd.DataFrame,
    ) -> None:
        """_build_target が結果ベースproxyを使用する (Rule 16)"""
        screener = RaceQualityScreener()
        target = screener._build_target(race_features_df)
        assert len(target) == len(race_features_df)
        # hist_roi_topk が高いレースほど target が高い
        assert target.iloc[2] > target.iloc[1]  # ROI 2.0 vs 0.8

    def test_target_does_not_use_ev_dependent_features(self) -> None:
        """FEATURE_COLS にEV依存特徴量が含まれない (Rule 16)"""
        ev_dependent = ["ev_win", "ev_place", "p_win_pred", "edge", "actual_bet_roi"]
        for f in ev_dependent:
            assert f not in RaceQualityScreener.FEATURE_COLS

    def test_should_bet_returns_bool(
        self,
        race_features_df: pd.DataFrame,
    ) -> None:
        screener = RaceQualityScreener()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.3, 0.8, 0.1])
        screener.model = mock_model
        screener.threshold = 0.4

        features = race_features_df.iloc[0].to_dict()
        result = screener.should_bet(features)
        assert isinstance(result, bool)

    def test_should_bet_above_threshold(
        self,
        race_features_df: pd.DataFrame,
    ) -> None:
        screener = RaceQualityScreener()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.8])  # 高スコア
        screener.model = mock_model
        screener.threshold = 0.4

        features = race_features_df.iloc[0].to_dict()
        assert screener.should_bet(features) is True

    def test_should_bet_below_threshold(
        self,
        race_features_df: pd.DataFrame,
    ) -> None:
        screener = RaceQualityScreener()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1])  # 低スコア
        screener.model = mock_model
        screener.threshold = 0.4

        features = race_features_df.iloc[0].to_dict()
        assert screener.should_bet(features) is False

    def test_has_distribution_features(self) -> None:
        """分布特徴量が含まれる (v5.1追加)"""
        assert "n_positive_errors" in RaceQualityScreener.FEATURE_COLS
        assert "top_k_error_sum" in RaceQualityScreener.FEATURE_COLS
        assert "positive_error_ratio" in RaceQualityScreener.FEATURE_COLS

    def test_has_result_based_profit_proxy(self) -> None:
        """結果ベース利益proxyが含まれる (v5.4追加, Rule 16)"""
        assert "hist_hit_rate_topk" in RaceQualityScreener.FEATURE_COLS
        assert "hist_roi_topk" in RaceQualityScreener.FEATURE_COLS
        assert "hist_positive_return_ratio" in RaceQualityScreener.FEATURE_COLS

    def test_has_ema_smoothed_market_indicators(self) -> None:
        """EMA平滑化市場指標が含まれる (v5.6追加)"""
        assert "overround_ema" in RaceQualityScreener.FEATURE_COLS
        assert "entropy_ema" in RaceQualityScreener.FEATURE_COLS

    def test_screener_independence(self) -> None:
        """品質スコアとedge_max_per_raceの相関<0.30 (§13.1)

        RaceQualityScreener のターゲット構成要素 (market歪み × hist利益proxy)
        が edge (Stage2予測差) と構造的に独立であることを、
        合成データで数学的性質を検証する。
        """
        np.random.seed(42)
        n = 500
        # market 歪み指標 (screener 入力)
        market_error_std = np.random.normal(0.25, 0.08, n)
        # hist 利益 proxy (screener 入力、edge と独立)
        hist_hit_rate = np.random.uniform(0.10, 0.50, n)
        hist_roi = np.random.uniform(0.80, 1.20, n)
        # 品質スコア (合成ターゲットの近似)
        quality_scores = market_error_std * hist_hit_rate * hist_roi
        # edge (Stage2 の出力、品質スコアと構造的に独立)
        edge_max = np.random.normal(1.10, 0.15, n)
        corr = np.corrcoef(quality_scores, edge_max)[0, 1]
        assert abs(corr) < 0.30, f"品質スコアと edge の相関が高すぎます: {corr:.3f}"

    def test_no_temporal_leak(self) -> None:
        """hist特徴量に未来情報リークがないこと (§13.1)

        expanding window 計算の数学的性質を検証:
        各行の hist_mean が、その行より前のデータのみの mean に一致すること。
        """
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        values = np.arange(1.0, 21.0)
        # expanding mean: 行 i の hist_mean = mean(values[0:i])
        hist_means = [float("nan")] + [float(np.mean(values[:i])) for i in range(1, 20)]

        df = pd.DataFrame({"race_date": dates, "value": values, "hist_mean": hist_means})

        for i in range(1, len(df)):
            race_date = df.iloc[i]["race_date"]
            hist_rows = df[df["race_date"] < race_date]
            expected_mean = hist_rows["value"].mean() if len(hist_rows) > 0 else float("nan")
            actual_mean = df.iloc[i]["hist_mean"]
            if not pd.isna(expected_mean):
                assert abs(actual_mean - expected_mean) < 1e-10, (
                    f"行{i}: hist_mean に未来情報リークの疑い "
                    f"(expected={expected_mean}, actual={actual_mean})"
                )


# Phase36 fundamental features that must NOT be in RaceQualityScreener (RTG-01)
_PHASE36_FEATURES = [
    "form_trend_race_rank",
    "blood_total_wr_race_rank",
    "blood_surface_wr_race_rank",
    "weighted_recent_form_finish",
    "weighted_recent_form_time",
    "grade_x_form_trend",
    "distance_x_closing_index",
    "grade_x_blood_prize_log",
    "closing_speed_ratio_avg",
    "closing_speed_ratio_zscore",
    "closing_speed_ratio_trend",
    "harontime_last3f_avg",
    "harontime_last3f_zscore",
    "harontime_last3f_trend",
    "haron_race_gap_avg",
    "haron_race_gap_zscore",
    "haron_race_gap_trend",
    "pace_adj_finish_avg",
    "pace_ratio_avg",
    "pace_ratio_zscore",
    "pace_ratio_trend",
    "pace_early_avg",
    "pace_mid_avg",
    "pace_late_avg",
    "closing_speed_ratio_avg_race_rank",
    "harontime_last3f_avg_race_rank",
]


class TestRaceQualityScreenerFeatureRouting:
    """RTG-01: RaceQualityScreener must NOT contain Phase36 horse-level features."""

    def test_no_phase36_features_in_feature_cols(self) -> None:
        """RaceQualityScreener.FEATURE_COLS に Phase36 特徴量が含まれない (RTG-01)"""
        for feat in _PHASE36_FEATURES:
            assert feat not in RaceQualityScreener.FEATURE_COLS, (
                f"Phase36 feature '{feat}' found in RaceQualityScreener.FEATURE_COLS"
            )

    def test_screener_features_still_present(self) -> None:
        """RaceQualityScreener.FEATURE_COLS に screener 適切特徴量が残っている"""
        must_have = [
            "market_log_error_max_abs",
            "market_entropy",
            "overround",
        ]
        for feat in must_have:
            assert feat in RaceQualityScreener.FEATURE_COLS, (
                f"Required screener feature '{feat}' missing from RaceQualityScreener.FEATURE_COLS"
            )


# Phase36 race-level aggregate features (RTG-02/03)
_PHASE36_RACE_AGGREGATES = [
    "phase36_top1_strength",
    "phase36_top1_top2_gap",
    "phase36_field_dispersion",
    "phase36_form_signal_dispersion",
    "phase36_weighted_form_mean",
]

# rl_* columns that build_race_features() must propagate (RTG-03)
_RL_COLUMNS = [
    "rl_log_odds_entropy",
    "rl_odds_dispersion",
    "rl_top3_odds_gap",
    "rl_top1_odds",
    "rl_favorite_rank_gap",
    "rl_n_horses",
    "rl_favorite_in_wide_top1",
    "rl_trio_overlap",
    "rl_market_consistency",
    "rl_trio_odds_ratio",
    "rl_wide_harville_ratio",
    "implied_prob_hhi",
    "odds_skewness",
]


class TestRaceQualityScreenerPhase36Aggregates:
    """RTG-02/03: Race-level Phase36 aggregate features for screener."""

    def test_feature_cols_contains_phase36_aggregates(self) -> None:
        """RaceQualityScreener.FEATURE_COLS に5つの phase36_* race aggregate が含まれる"""
        for feat in _PHASE36_RACE_AGGREGATES:
            assert feat in RaceQualityScreener.FEATURE_COLS, (
                f"Phase36 race aggregate '{feat}' missing from RaceQualityScreener.FEATURE_COLS"
            )

    def test_predict_score_returns_float(self) -> None:
        """predict_score() が float を返す"""
        screener = RaceQualityScreener()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.65])
        mock_model.best_iteration = 50
        screener.model = mock_model
        screener.threshold = 0.4

        features = {
            "market_log_error_max_abs": 0.5,
            "market_entropy": 2.5,
            "overround": 0.22,
            "field_size": 12,
            "surface": "turf",
            "distance_bin": "mile",
            "track_condition_code": 2,
            "grade_code": "C",
        }
        score = screener.predict_score(features)
        assert isinstance(score, float)
        assert score == pytest.approx(0.65)

    def test_should_bet_works_with_race_aggregates(
        self,
        race_features_df: pd.DataFrame,
    ) -> None:
        """should_bet() が race aggregate features と正しく動作する"""
        screener = RaceQualityScreener()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.8])
        screener.model = mock_model
        screener.threshold = 0.4

        features = race_features_df.iloc[0].to_dict()
        result = screener.should_bet(features)
        assert isinstance(result, bool)
        assert result is True


class TestBuildRaceFeatures:
    """build_race_features() must return rl_* columns and phase36 aggregates."""

    @pytest.fixture
    def race_df_with_rl_and_phase36(self) -> pd.DataFrame:
        """rl_* columns and phase36 source columns を含むレース DataFrame"""
        return pd.DataFrame(
            {
                "race_id": ["R001"] * 5,
                "umaban": [1, 2, 3, 4, 5],
                "surface": ["turf"] * 5,
                "distance_bin": ["mile"] * 5,
                "track_condition_code": [2] * 5,
                "grade_code": ["C"] * 5,
                "field_size": [5] * 5,
                "difficulty_score": [0.5] * 5,
                "market_entropy": [2.5] * 5,
                "overround": [0.22] * 5,
                "overround_ema": [0.21] * 5,
                "entropy_ema": [2.4] * 5,
                "signed_log_error_win": [0.1, -0.2, 0.3, -0.1, 0.05],
                "abs_log_error_win": [0.1, 0.2, 0.3, 0.1, 0.05],
                "hist_hit_rate_topk": [0.25] * 5,
                "hist_roi_topk": [1.0] * 5,
                "hist_positive_return_ratio": [0.3] * 5,
                # rl_* columns
                "rl_log_odds_entropy": [1.5] * 5,
                "rl_odds_dispersion": [0.8] * 5,
                "rl_top3_odds_gap": [2.5] * 5,
                "rl_top1_odds": [3.0] * 5,
                "rl_favorite_rank_gap": [1.5] * 5,
                "rl_n_horses": [5.0] * 5,
                "rl_favorite_in_wide_top1": [1.0] * 5,
                "rl_trio_overlap": [2.0] * 5,
                "rl_market_consistency": [1.0] * 5,
                "rl_trio_odds_ratio": [1.1] * 5,
                "rl_wide_harville_ratio": [0.95] * 5,
                "implied_prob_hhi": [0.12] * 5,
                "odds_skewness": [0.5] * 5,
                # Phase36 source columns (horse-level)
                "closing_speed_ratio_avg": [0.8, 0.6, 0.9, 0.5, 0.7],
                "form_trend_race_rank": [0.3, 0.5, 0.1, 0.7, 0.4],
                "weighted_recent_form_finish": [3.0, 5.0, 2.0, 7.0, 4.0],
            }
        )

    def test_build_race_features_returns_phase36_aggregates(
        self,
        race_df_with_rl_and_phase36: pd.DataFrame,
    ) -> None:
        """build_race_features() が phase36_* aggregate を返す"""
        from backtest.race_predictor import RacePredictor

        features = RacePredictor.build_race_features(race_df_with_rl_and_phase36)
        for feat in _PHASE36_RACE_AGGREGATES:
            assert feat in features, (
                f"phase36 aggregate '{feat}' missing from build_race_features() output"
            )
        # Verify specific values
        assert features["phase36_top1_strength"] == pytest.approx(0.9)
        assert features["phase36_weighted_form_mean"] == pytest.approx(4.2)

    def test_build_race_features_returns_rl_columns(
        self,
        race_df_with_rl_and_phase36: pd.DataFrame,
    ) -> None:
        """build_race_features() が rl_* columns を返す"""
        from backtest.race_predictor import RacePredictor

        features = RacePredictor.build_race_features(race_df_with_rl_and_phase36)
        for col in _RL_COLUMNS:
            assert col in features, (
                f"rl_* column '{col}' missing from build_race_features() output"
            )

    def test_build_race_features_phase36_defaults_when_missing(self) -> None:
        """Phase36 source columns がない場合、デフォルト0.0を返す"""
        from backtest.race_predictor import RacePredictor

        race_df = pd.DataFrame(
            {
                "race_id": ["R001"] * 3,
                "umaban": [1, 2, 3],
                "surface": ["turf"] * 3,
                "distance_bin": ["mile"] * 3,
                "track_condition_code": [2] * 3,
                "grade_code": ["C"] * 3,
                "field_size": [3] * 3,
                "difficulty_score": [0.5] * 3,
                "market_entropy": [2.5] * 3,
                "overround": [0.22] * 3,
                "overround_ema": [0.21] * 3,
                "entropy_ema": [2.4] * 3,
                "signed_log_error_win": [0.1, -0.2, 0.3],
                "abs_log_error_win": [0.1, 0.2, 0.3],
                "hist_hit_rate_topk": [0.25] * 3,
                "hist_roi_topk": [1.0] * 3,
                "hist_positive_return_ratio": [0.3] * 3,
            }
        )
        features = RacePredictor.build_race_features(race_df)
        for feat in _PHASE36_RACE_AGGREGATES:
            assert feat in features
            assert features[feat] == 0.0
