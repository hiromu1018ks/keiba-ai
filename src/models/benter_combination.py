"""Benter (1994) 第二段階ロジット合成レイヤー。

ファンダメンタルモデルの予測確率と市場の暗黙確率を最適な重みで合成する。
logit(p_combined) = alpha * logit(p_fundamental) + beta * logit(p_market) + gamma
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class TemperatureScaling:
    """Temperature Scaling — キャリブレーション後処理 (Guo et al., 2017).

    p_calibrated = sigmoid(logit(p_raw) / T)
    T > 1 で過信を抑制、T < 1 で信頼度を増加。
    Brier Score を最小化する T をバリデーションデータから最適化。
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-10, 1 - 1e-10)
        return np.log(p / (1 - p))  # type: ignore[no-any-return]

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))  # type: ignore[no-any-return]

    def transform(self, p: np.ndarray) -> np.ndarray:
        """温度スケーリングを適用"""
        logits = self._logit(p)
        return self._sigmoid(logits / self.temperature)

    @classmethod
    def fit(
        cls, p: np.ndarray, y: np.ndarray, *, bounds: tuple[float, float] = (0.3, 3.0)
    ) -> TemperatureScaling:
        """Brier Score を最小化する温度を最適化"""
        p_arr = np.clip(np.asarray(p, dtype=float), 1e-10, 1 - 1e-10)
        y_arr = np.asarray(y, dtype=float)
        logits = np.log(p_arr / (1 - p_arr))

        def neg_log_likelihood(T: np.ndarray) -> float:
            t = float(T[0])
            scaled = 1.0 / (1.0 + np.exp(-logits / t))
            scaled = np.clip(scaled, 1e-10, 1 - 1e-10)
            brier = np.mean((scaled - y_arr) ** 2)
            # NLLも加味して正則化
            nll = -np.mean(y_arr * np.log(scaled) + (1 - y_arr) * np.log(1 - scaled))
            return brier + 0.1 * nll

        result = minimize(
            neg_log_likelihood,
            x0=[1.0],
            method="L-BFGS-B",
            bounds=[bounds],
        )
        temp = float(result.x[0])
        logger.info("Temperature Scaling: T=%.4f (Brier=%.6f)", temp, result.fun)
        return cls(temperature=temp)

    def to_dict(self) -> dict[str, float]:
        return {"temperature": self.temperature}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> TemperatureScaling:
        return cls(temperature=d["temperature"])

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TemperatureScaling:
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(d)


class BenterCombination:
    """第二段階ロジット合成: ファンダメンタルモデル + 市場確率。

    Benter (1994) の多項ロジット合成を二項分類（複勝予測）に適応。
    バイアス項 gamma を含む（多項版は正規化定数で暗黙に持つ）。
    """

    def __init__(self, alpha: float, beta: float, gamma: float) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-10, 1 - 1e-10)
        return np.log(p / (1 - p))  # type: ignore[no-any-return]

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))  # type: ignore[no-any-return]

    def combine(self, p_fund: np.ndarray, p_market: np.ndarray) -> np.ndarray:
        """ロジット空間で確率を合成する。"""
        logit_combined = (
            self.alpha * self._logit(p_fund) + self.beta * self._logit(p_market) + self.gamma
        )
        return self._sigmoid(logit_combined)

    @classmethod
    def fit(cls, p_fund: np.ndarray, p_market: np.ndarray, y: np.ndarray) -> BenterCombination:
        """最尤推定で alpha, beta, gamma を推定する。"""
        logit_f = cls._logit(p_fund)
        logit_m = cls._logit(p_market)
        y = np.asarray(y, dtype=float)

        def neg_log_likelihood(params: np.ndarray) -> float:
            alpha, beta, gamma = params
            logit_c = alpha * logit_f + beta * logit_m + gamma
            p_c = cls._sigmoid(logit_c)
            p_c = np.clip(p_c, 1e-10, 1 - 1e-10)
            return float(-np.sum(y * np.log(p_c) + (1 - y) * np.log(1 - p_c)))

        result = minimize(
            neg_log_likelihood,
            x0=[0.5, 0.5, 0.0],
            method="L-BFGS-B",
            # v5: β下限 0.01→0.20 — 市場重みを確保しfundamental過信を抑制
            bounds=[(0.01, 5.0), (0.20, 5.0), (-5.0, 5.0)],
        )
        return cls(
            alpha=float(result.x[0]),
            beta=float(result.x[1]),
            gamma=float(result.x[2]),
        )

    def to_dict(self) -> dict[str, float]:
        return {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> BenterCombination:
        return cls(alpha=d["alpha"], beta=d["beta"], gamma=d["gamma"])

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BenterCombination:
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(d)
