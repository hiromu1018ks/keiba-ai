import numpy as np
import pytest
from features.form_cycle_features import compute_form_features, FEATURE_COLS


class TestComputeFormFeatures:
    def test_improving_trend(self):
        """着順が上昇傾向 → form_trend > 0
        データは古い順: [3着, 2着, 1着] → 新しいほど着順が良い = 改善
        """
        kj = np.array([3.0, 2.0, 1.0])
        ss = np.array([16.0, 16.0, 16.0])
        trend, consistency, peak = compute_form_features(kj, ss)
        assert trend > 0  # 改善傾向

    def test_declining_trend(self):
        """着順が下降傾向 → form_trend < 0
        データは古い順: [1着, 2着, 3着] → 新しいほど着順が悪い = 悪化
        """
        kj = np.array([1.0, 2.0, 3.0])
        ss = np.array([16.0, 16.0, 16.0])
        trend, _, _ = compute_form_features(kj, ss)
        assert trend < 0  # 悪化傾向

    def test_insufficient_data(self):
        """データ不足 (< 2走) → 全て NaN"""
        kj = np.array([1.0])
        ss = np.array([16.0])
        trend, consistency, peak = compute_form_features(kj, ss)
        assert np.isnan(trend)
        assert np.isnan(consistency)
        assert np.isnan(peak)

    def test_peak_flag_true(self):
        """直近2走が全体より良い → peak=1.0
        データは古い順: [5着, 2着, 1着] → 最新2走(2着,1着)が全体平均より良い
        """
        kj = np.array([5.0, 2.0, 1.0])
        ss = np.array([16.0, 16.0, 16.0])
        _, _, peak = compute_form_features(kj, ss)
        assert peak == 1.0

    def test_peak_flag_false(self):
        """直近2走が全体より悪い → peak=0.0
        データは古い順: [1着, 4着, 5着] → 最新2走(4着,5着)が全体平均より悪い
        """
        kj = np.array([1.0, 4.0, 5.0])
        ss = np.array([16.0, 16.0, 16.0])
        _, _, peak = compute_form_features(kj, ss)
        assert peak == 0.0

    def test_consistency_low(self):
        """全て同じ着順 → consistency は 0 に近い"""
        kj = np.array([3.0, 3.0, 3.0])
        ss = np.array([16.0, 16.0, 16.0])
        _, consistency, _ = compute_form_features(kj, ss)
        assert consistency < 0.01

    def test_feature_cols(self):
        assert FEATURE_COLS == ["form_trend", "form_consistency", "form_peak_flag"]
