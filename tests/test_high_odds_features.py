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
    compute_env_adaptability,
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
        """FEATURE_COLS が18特徴量を含む (HODDS-02/03: 9 + HODDS-04: 9)"""
        assert len(FEATURE_COLS) == 18

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
            "dist_change_avg_pos",
            "dist_change_win_rate",
            "dist_change_exp_count",
            "surf_change_avg_pos",
            "surf_change_win_rate",
            "surf_change_exp_count",
            "cond_change_avg_pos",
            "cond_change_win_rate",
            "cond_change_exp_count",
        ]
        assert FEATURE_COLS == expected


class TestComputeEnvAdaptability:
    """compute_env_adaptability のテスト"""

    def _make_env_result(
        self,
        kj: list[float],
        ss: list[float],
        dist_bins: list[str],
        surfaces: list[str],
        conds: list[float],
        cur_db: str,
        cur_surf: str,
        cur_cond: float,
    ) -> dict[str, float]:
        """テストヘルパー: 引数をnumpy配列に変換してcompute_env_adaptabilityを呼ぶ"""
        return compute_env_adaptability(
            np.array(kj, dtype=float),
            np.array(ss, dtype=float),
            np.array(dist_bins, dtype=object),
            np.array(surfaces, dtype=object),
            np.array(conds, dtype=float),
            cur_db,
            cur_surf,
            cur_cond,
        )

    def test_dist_change_with_experience(self):
        """Test 1: 距離変更あり(sprint→mile) — 過去走にmile経験あり"""
        # 最後の過去走=sprint, 現在=mile → 距離変更検出
        # 最初2走=sprint, 3番目=mile → mile経験1走
        result = self._make_env_result(
            kj=[3.0, 5.0, 2.0],
            ss=[16.0, 16.0, 16.0],
            dist_bins=["mile", "sprint", "sprint"],
            surfaces=["turf", "turf", "turf"],
            conds=[1.0, 1.0, 1.0],
            cur_db="mile",
            cur_surf="turf",
            cur_cond=1.0,
        )
        # dist_change_* should be non-NaN
        assert not np.isnan(result["dist_change_avg_pos"])
        assert not np.isnan(result["dist_change_win_rate"])
        assert not np.isnan(result["dist_change_exp_count"])

    def test_dist_change_no_experience(self):
        """Test 2: 距離変更ありだが経験なし → dist_change_* がNaN"""
        result = self._make_env_result(
            kj=[3.0, 5.0, 2.0],
            ss=[16.0, 16.0, 16.0],
            dist_bins=["sprint", "sprint", "sprint"],
            surfaces=["turf", "turf", "turf"],
            conds=[1.0, 1.0, 1.0],
            cur_db="mile",
            cur_surf="turf",
            cur_cond=1.0,
        )
        # 距離変更ありだがmile経験なしなのでNaN
        assert np.isnan(result["dist_change_avg_pos"])
        assert np.isnan(result["dist_change_win_rate"])
        assert np.isnan(result["dist_change_exp_count"])

    def test_surface_change_with_experience(self):
        """Test 3: サーフェス変更あり(芝→ダート) — 過去走にダート経験あり"""
        result = self._make_env_result(
            kj=[4.0, 3.0, 6.0],
            ss=[16.0, 16.0, 16.0],
            dist_bins=["mile", "mile", "mile"],
            surfaces=["turf", "dirt", "turf"],
            conds=[1.0, 1.0, 1.0],
            cur_db="mile",
            cur_surf="dirt",
            cur_cond=1.0,
        )
        assert not np.isnan(result["surf_change_avg_pos"])
        assert not np.isnan(result["surf_change_win_rate"])
        assert not np.isnan(result["surf_change_exp_count"])

    def test_condition_change_with_experience(self):
        """Test 4: 馬場状態変更あり(良→稍重) — 過去走に稍重経験あり"""
        result = self._make_env_result(
            kj=[5.0, 2.0, 7.0],
            ss=[16.0, 16.0, 16.0],
            dist_bins=["mile", "mile", "mile"],
            surfaces=["turf", "turf", "turf"],
            conds=[1.0, 2.0, 1.0],
            cur_db="mile",
            cur_surf="turf",
            cur_cond=2.0,
        )
        assert not np.isnan(result["cond_change_avg_pos"])
        assert not np.isnan(result["cond_change_win_rate"])
        assert not np.isnan(result["cond_change_exp_count"])

    def test_no_changes_all_nan(self):
        """Test 5: 全変更なし(同条件) — 全9特徴量がNaN"""
        result = self._make_env_result(
            kj=[3.0, 5.0, 2.0],
            ss=[16.0, 16.0, 16.0],
            dist_bins=["mile", "mile", "mile"],
            surfaces=["turf", "turf", "turf"],
            conds=[1.0, 1.0, 1.0],
            cur_db="mile",
            cur_surf="turf",
            cur_cond=1.0,
        )
        for key in [
            "dist_change_avg_pos", "dist_change_win_rate", "dist_change_exp_count",
            "surf_change_avg_pos", "surf_change_win_rate", "surf_change_exp_count",
            "cond_change_avg_pos", "cond_change_win_rate", "cond_change_exp_count",
        ]:
            assert np.isnan(result[key]), f"{key} should be NaN when no change"

    def test_no_history_all_nan(self):
        """Test 6: データ不足 (過去走0) → 全9特徴量がNaN"""
        result = self._make_env_result(
            kj=[],
            ss=[],
            dist_bins=[],
            surfaces=[],
            conds=[],
            cur_db="mile",
            cur_surf="turf",
            cur_cond=1.0,
        )
        for key in [
            "dist_change_avg_pos", "dist_change_win_rate", "dist_change_exp_count",
            "surf_change_avg_pos", "surf_change_win_rate", "surf_change_exp_count",
            "cond_change_avg_pos", "cond_change_win_rate", "cond_change_exp_count",
        ]:
            assert np.isnan(result[key]), f"{key} should be NaN with no history"

    def test_dist_change_exp_count(self):
        """Test 7: 距離変更あり、3走中3走が該当条件 → exp_count == 3.0"""
        # 過去走の最後がsprint、現在がmile → 距離変更検出
        # 過去走3走中mileなのは... 全部mileなら最後もmile=変更なし
        # 戦略: 最後の過去走だけsprintにして変更検出、残り3走mileで経験あり
        result = self._make_env_result(
            kj=[3.0, 5.0, 2.0, 4.0],
            ss=[16.0, 16.0, 16.0, 16.0],
            dist_bins=["mile", "mile", "mile", "sprint"],
            surfaces=["turf", "turf", "turf", "turf"],
            conds=[1.0, 1.0, 1.0, 1.0],
            cur_db="mile",
            cur_surf="turf",
            cur_cond=1.0,
        )
        # 最後の過去走=sprint、現在=mile → 距離変更検出
        # 過去走のうちmileは最初の3走 → exp_count == 3.0
        assert not np.isnan(result["dist_change_exp_count"])
        assert result["dist_change_exp_count"] == 3.0

    def test_win_rate_calculation(self):
        """Test 8: 勝率計算 — 3走中1着1回 → win_rate ≈ 0.333"""
        # 過去走の最後=sprint、現在=mile → 距離変更検出
        # mile経験: 1着(1位)、5着、3着 → win_rate = 1/3 ≈ 0.333
        result = self._make_env_result(
            kj=[1.0, 5.0, 3.0, 4.0],
            ss=[16.0, 16.0, 16.0, 16.0],
            dist_bins=["mile", "mile", "mile", "sprint"],
            surfaces=["turf", "turf", "turf", "turf"],
            conds=[1.0, 1.0, 1.0, 1.0],
            cur_db="mile",
            cur_surf="turf",
            cur_cond=1.0,
        )
        assert not np.isnan(result["dist_change_win_rate"])
        assert abs(result["dist_change_win_rate"] - (1.0 / 3.0)) < 0.01
