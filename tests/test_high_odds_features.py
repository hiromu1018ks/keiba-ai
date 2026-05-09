"""高オッズ的中パターン特徴量の単体テスト

クラストラジェクトリ (compute_class_trajectory) と
フォーム改善率 (compute_form_improvement_rate) のテスト。
"""

from __future__ import annotations

import numpy as np
import pytest
from features.high_odds_features import (
    FEATURE_COLS,
    compute_class_trajectory,
    compute_form_improvement_rate,
)


class TestComputeClassTrajectory:
    """compute_class_trajectory のテスト"""

    def test_promotion_pattern(self):
        """昇級パターン [未勝利, 1勝, 2勝, OP, 重賞]
        gradecd: [E, D, C, B, A] → class_promotions=4, class_demotions=0, class_net_change > 0
        """
        gradecd = np.array(["E", "D", "C", "B", "A"])
        jyokencd = np.array([np.nan] * 5)
        (
            promotions,
            demotions,
            net_change,
            max_level,
            level_std,
            v_flag,
            v_duration,
        ) = compute_class_trajectory(gradecd, jyokencd)
        assert promotions == 4
        assert demotions == 0
        assert net_change > 0

    def test_demotion_pattern(self):
        """降級パターン [重賞, OP, 2勝, 1勝, 未勝利]
        gradecd: [A, B, C, D, E] → class_promotions=0, class_demotions=4, class_net_change < 0
        """
        gradecd = np.array(["A", "B", "C", "D", "E"])
        jyokencd = np.array([np.nan] * 5)
        (
            promotions,
            demotions,
            net_change,
            max_level,
            level_std,
            v_flag,
            v_duration,
        ) = compute_class_trajectory(gradecd, jyokencd)
        assert promotions == 0
        assert demotions == 4
        assert net_change < 0

    def test_v_recovery_pattern(self):
        """V字回復 [OP, 2勝, 1勝, 2勝, OP]
        gradecd: [B, C, D, C, B] → v_recovery_flag=1.0, v_recovery_duration > 0
        """
        gradecd = np.array(["B", "C", "D", "C", "B"])
        jyokencd = np.array([np.nan] * 5)
        (
            promotions,
            demotions,
            net_change,
            max_level,
            level_std,
            v_flag,
            v_duration,
        ) = compute_class_trajectory(gradecd, jyokencd)
        assert v_flag == 1.0
        assert not np.isnan(v_duration)
        assert v_duration > 0

    def test_no_v_recovery(self):
        """V字回復なし [1勝, 2勝, OP, 重賞, 重賞]
        gradecd: [D, C, B, A, A] → v_recovery_flag=0.0
        """
        gradecd = np.array(["D", "C", "B", "A", "A"])
        jyokencd = np.array([np.nan] * 5)
        (
            promotions,
            demotions,
            net_change,
            max_level,
            level_std,
            v_flag,
            v_duration,
        ) = compute_class_trajectory(gradecd, jyokencd)
        assert v_flag == 0.0

    def test_insufficient_data(self):
        """データ不足 (< 2走) → 全フィールド NaN"""
        gradecd = np.array(["B"])
        jyokencd = np.array([np.nan])
        result = compute_class_trajectory(gradecd, jyokencd)
        assert len(result) == 7
        for val in result:
            assert np.isnan(val), f"Expected NaN but got {val}"

    def test_all_same_class(self):
        """全同じクラス → class_promotions=0, class_demotions=0, class_level_std ≈ 0"""
        gradecd = np.array(["C", "C", "C", "C", "C"])
        jyokencd = np.array([np.nan] * 5)
        (
            promotions,
            demotions,
            net_change,
            max_level,
            level_std,
            v_flag,
            v_duration,
        ) = compute_class_trajectory(gradecd, jyokencd)
        assert promotions == 0
        assert demotions == 0
        assert abs(level_std) < 0.01
        assert net_change == 0.0

    def test_returns_seven_elements(self):
        """戻り値が7要素のタプルであること"""
        gradecd = np.array(["A", "B", "C"])
        jyokencd = np.array([np.nan, np.nan, np.nan])
        result = compute_class_trajectory(gradecd, jyokencd)
        assert isinstance(result, tuple)
        assert len(result) == 7

    def test_jyokencd_fallback(self):
        """jyoken_code数値フォールバック: gradecdがマップ外の場合"""
        gradecd = np.array([np.nan, np.nan])
        jyokencd = np.array([5.0, 7.0])
        (
            promotions,
            demotions,
            net_change,
            max_level,
            level_std,
            v_flag,
            v_duration,
        ) = compute_class_trajectory(gradecd, jyokencd)
        # 5.0 → 7.0 は昇級
        assert promotions == 1
        assert demotions == 0


class TestComputeFormImprovementRate:
    """compute_form_improvement_rate のテスト"""

    def test_time_improvement(self):
        """タイム改善 (z-score配列が減少) → time_improvement_rate > 0
        配列は古い順 → 新しいほど低い = 改善
        """
        zscore = np.array([2.0, 1.5, 1.0, 0.5, 0.0])
        positions = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        sizes = np.array([16.0] * 5)
        time_rate, pos_rate = compute_form_improvement_rate(zscore, positions, sizes)
        assert time_rate > 0

    def test_position_improvement(self):
        """着順改善 (正規化着順が減少) → position_improvement_rate > 0"""
        zscore = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        positions = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        sizes = np.array([16.0] * 5)
        time_rate, pos_rate = compute_form_improvement_rate(zscore, positions, sizes)
        assert pos_rate > 0

    def test_insufficient_data(self):
        """データ不足 (< 2走) → 両方 NaN"""
        zscore = np.array([1.0])
        positions = np.array([3.0])
        sizes = np.array([16.0])
        time_rate, pos_rate = compute_form_improvement_rate(zscore, positions, sizes)
        assert np.isnan(time_rate)
        assert np.isnan(pos_rate)

    def test_all_same_values(self):
        """全同じ値 → 両方 ≈ 0"""
        zscore = np.array([1.0, 1.0, 1.0, 1.0])
        positions = np.array([3.0, 3.0, 3.0, 3.0])
        sizes = np.array([16.0, 16.0, 16.0, 16.0])
        time_rate, pos_rate = compute_form_improvement_rate(zscore, positions, sizes)
        assert abs(time_rate) < 0.01
        assert abs(pos_rate) < 0.01

    def test_returns_two_elements(self):
        """戻り値が2要素のタプルであること"""
        zscore = np.array([1.0, 2.0, 3.0])
        positions = np.array([3.0, 2.0, 1.0])
        sizes = np.array([16.0, 16.0, 16.0])
        result = compute_form_improvement_rate(zscore, positions, sizes)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_nan_handling(self):
        """NaN値を含むデータの処理"""
        zscore = np.array([np.nan, 2.0, 1.0, 0.5, np.nan])
        positions = np.array([np.nan, 4.0, 3.0, 2.0, np.nan])
        sizes = np.array([16.0, 16.0, 16.0, 16.0, 16.0])
        time_rate, pos_rate = compute_form_improvement_rate(zscore, positions, sizes)
        # NaN除外後に有効データ3件で計算される
        assert not np.isnan(time_rate)
        assert not np.isnan(pos_rate)

    def test_halflife_parameter(self):
        """halflife パラメータの変更が結果に反映される"""
        zscore = np.array([3.0, 2.0, 1.0, 0.5, 0.0])
        positions = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        sizes = np.array([16.0] * 5)
        rate_hl3_t, rate_hl3_p = compute_form_improvement_rate(
            zscore, positions, sizes, halflife=3
        )
        rate_hl1_t, rate_hl1_p = compute_form_improvement_rate(
            zscore, positions, sizes, halflife=1
        )
        # 異なるhalflifeで異なる結果になること
        assert rate_hl3_t != rate_hl1_t or rate_hl3_p != rate_hl1_p


class TestFeatureCols:
    """FEATURE_COLS の定義確認"""

    def test_feature_cols_count(self):
        """FEATURE_COLS が9特徴量を含む"""
        assert len(FEATURE_COLS) == 9

    def test_feature_cols_names(self):
        """FEATURE_COLS が期待される特徴量名を含む"""
        expected = [
            "class_promotions",
            "class_demotions",
            "class_net_change",
            "class_max_level",
            "class_level_std",
            "v_recovery_flag",
            "v_recovery_duration",
            "time_improvement_rate",
            "position_improvement_rate",
        ]
        assert FEATURE_COLS == expected
