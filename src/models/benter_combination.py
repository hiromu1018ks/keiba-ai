"""Benter (1994) 第二段階ロジット合成レイヤー。

ファンダメンタルモデルの予測確率と市場の暗黙確率を最適な重みで合成する。
logit(p_combined) = alpha * logit(p_fundamental) + beta * logit(p_market) + gamma
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]


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
            bounds=[(0.01, 5.0), (0.01, 5.0), (-5.0, 5.0)],
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
