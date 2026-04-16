"""BenterCombination のテスト"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from models.benter_combination import BenterCombination


class TestBenterCombine:
    """combine() メソッドのテスト"""

    def test_combine_returns_same_shape(self) -> None:
        """出力が入力と同じ形状の配列を返す"""
        combo = BenterCombination(alpha=0.5, beta=0.5, gamma=0.0)
        p_fund = np.array([0.3, 0.5, 0.7])
        p_market = np.array([0.2, 0.4, 0.6])
        result = combo.combine(p_fund, p_market)
        assert result.shape == p_fund.shape

    def test_combine_output_in_01(self) -> None:
        """出力が [0, 1] の範囲内"""
        combo = BenterCombination(alpha=0.5, beta=0.5, gamma=0.0)
        p_fund = np.array([0.01, 0.3, 0.5, 0.9, 0.99])
        p_market = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
        result = combo.combine(p_fund, p_market)
        assert np.all(result > 0)
        assert np.all(result < 1)

    def test_combine_equal_inputs_alpha_dominant(self) -> None:
        """alpha=1, beta=0 なら p_fund をそのまま返す（恒等写像）"""
        combo = BenterCombination(alpha=1.0, beta=0.0, gamma=0.0)
        p_fund = np.array([0.3, 0.5, 0.7])
        p_market = np.array([0.1, 0.2, 0.9])  # ignored
        result = combo.combine(p_fund, p_market)
        np.testing.assert_allclose(result, p_fund, atol=1e-6)

    def test_combine_equal_inputs_beta_dominant(self) -> None:
        """alpha=0, beta=1 なら p_market をそのまま返す"""
        combo = BenterCombination(alpha=0.0, beta=1.0, gamma=0.0)
        p_fund = np.array([0.1, 0.2, 0.9])  # ignored
        p_market = np.array([0.3, 0.5, 0.7])
        result = combo.combine(p_fund, p_market)
        np.testing.assert_allclose(result, p_market, atol=1e-6)

    def test_combine_equal_probs_returns_same(self) -> None:
        """p_fund == p_market なら結果も同じ（gamma=0）"""
        combo = BenterCombination(alpha=0.3, beta=0.7, gamma=0.0)
        p = np.array([0.3, 0.5, 0.7])
        result = combo.combine(p, p)
        np.testing.assert_allclose(result, p, atol=1e-6)

    def test_combine_extreme_values_no_nan(self) -> None:
        """極端な確率値 (0.001, 0.999) でも NaN を返さない"""
        combo = BenterCombination(alpha=0.5, beta=0.5, gamma=0.0)
        p_fund = np.array([0.001, 0.999])
        p_market = np.array([0.001, 0.999])
        result = combo.combine(p_fund, p_market)
        assert not np.any(np.isnan(result))

    def test_combine_bias_positive(self) -> None:
        """gamma > 0 は確率を引き上げる"""
        combo_zero = BenterCombination(alpha=0.5, beta=0.5, gamma=0.0)
        combo_pos = BenterCombination(alpha=0.5, beta=0.5, gamma=1.0)
        p = np.array([0.3, 0.5])
        m = np.array([0.3, 0.5])
        r_zero = combo_zero.combine(p, m)
        r_pos = combo_pos.combine(p, m)
        assert np.all(r_pos > r_zero)


class TestBenterFit:
    """fit() クラスメソッドのテスト"""

    def test_fit_returns_instance(self) -> None:
        """fit() が BenterCombination インスタンスを返す"""
        rng = np.random.default_rng(42)
        n = 1000
        p_fund = rng.uniform(0.1, 0.9, n)
        p_market = rng.uniform(0.1, 0.9, n)
        y = (rng.random(n) < 0.3).astype(float)
        combo = BenterCombination.fit(p_fund, p_market, y)
        assert isinstance(combo, BenterCombination)

    def test_fit_alpha_beta_positive(self) -> None:
        """fit() の結果 alpha, beta は正の値"""
        rng = np.random.default_rng(42)
        n = 1000
        p_fund = rng.uniform(0.1, 0.9, n)
        p_market = rng.uniform(0.1, 0.9, n)
        y = (rng.random(n) < 0.3).astype(float)
        combo = BenterCombination.fit(p_fund, p_market, y)
        assert combo.alpha > 0
        assert combo.beta > 0

    def test_fit_improves_log_likelihood(self) -> None:
        """fit() の結果が naive (p_fund) より対数尤度が良い"""
        rng = np.random.default_rng(42)
        n = 2000
        p_market_true = rng.uniform(0.1, 0.9, n)
        y = (rng.random(n) < p_market_true * 0.5).astype(float)  # ~15% hit rate
        p_fund = p_market_true + rng.normal(0, 0.1, n)
        p_fund = np.clip(p_fund, 0.01, 0.99)
        p_market = p_market_true + rng.normal(0, 0.05, n)
        p_market = np.clip(p_market, 0.01, 0.99)

        combo = BenterCombination.fit(p_fund, p_market, y)
        p_combined = combo.combine(p_fund, p_market)

        # log-likelihood of combined vs fund-only
        def loglik(p: np.ndarray, y: np.ndarray) -> float:
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

        ll_combined = loglik(p_combined, y)
        ll_fund = loglik(p_fund, y)
        assert ll_combined >= ll_fund - 1.0  # allow small tolerance


class TestBenterSerialization:
    """to_dict / from_dict のテスト"""

    def test_roundtrip(self) -> None:
        """to_dict → from_dict で同じパラメータを復元"""
        original = BenterCombination(alpha=0.35, beta=0.65, gamma=-0.05)
        d = original.to_dict()
        restored = BenterCombination.from_dict(d)
        assert restored.alpha == original.alpha
        assert restored.beta == original.beta
        assert restored.gamma == original.gamma

    def test_dict_has_required_keys(self) -> None:
        """to_dict() が alpha, beta, gamma キーを持つ"""
        combo = BenterCombination(alpha=0.5, beta=0.5, gamma=0.0)
        d = combo.to_dict()
        assert "alpha" in d
        assert "beta" in d
        assert "gamma" in d

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """save → load で同じパラメータを復元"""
        original = BenterCombination(alpha=0.35, beta=0.65, gamma=-0.05)
        path = tmp_path / "benter.json"
        original.save(path)
        loaded = BenterCombination.load(path)
        assert loaded.alpha == original.alpha
        assert loaded.beta == original.beta
        assert loaded.gamma == original.gamma
