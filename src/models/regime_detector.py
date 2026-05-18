"""レジーム検知モデル -- 3状態分類 + ヒステリシス (9.5)"""

from __future__ import annotations

import logging
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from domain.models import RegimeConfig
from domain.types import RegimeState

logger = logging.getLogger(__name__)


def calc_odds_skewness(race_df: pd.DataFrame) -> float:
    """tanodds 分布の歪度 (レース単位、発走前のみ)"""
    if "odds" not in race_df.columns:
        return 0.0
    odds = race_df["odds"].dropna()
    if len(odds) < 3:
        return 0.0
    return float(odds.skew())


def calc_favorite_implied_prob(race_df: pd.DataFrame) -> float:
    """1番人気の implied probability (1/tanodds、発走前のみ)"""
    if "popularity_rank" not in race_df.columns or "odds" not in race_df.columns:
        return 0.3
    fav = race_df[race_df["popularity_rank"] == 1]
    if fav.empty:
        return 0.3
    odds_val = fav["odds"].iloc[0]
    if pd.isna(odds_val) or odds_val <= 0:
        return 0.3
    return float(1.0 / odds_val)


class RegimeDetector:
    """
    市場状態の切り替えを検知し、戦略パラメータを動的に調整する。

    v5.4: 軽量モデル + Stage2非依存特徴量 + ヒステリシス
    v5.5: 教師ラベルを市場指標ベースに変更 (Rule 19)
    """

    FEATURE_COLS: list[str] = [
        # 市場歪み (MarketModel 出力、発走前)
        "market_error_std",
        "market_error_mean",
        # 市場構造 (オッズ分布由来、発走前)
        "overround_rolling",
        "entropy_rolling",
        "favorite_implied_prob_rolling",
        "odds_skewness_rolling",
        # オッズボラティリティ (発走前)
        "odds_volatility_mean",
        # レース構造 (発走前確定)
        "field_size_mean",
        # 市場構造指標 (D-06: 市場集中度・歪度)
        "implied_prob_hhi",
        "odds_skewness",
    ]

    def __init__(
        self,
        cfg: RegimeConfig | None = None,
        override_params: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.cfg = cfg or RegimeConfig()
        # override_params: {"aggressive": {"fractional_kelly": 0.6, ...}, ...}
        self._override_params: dict[str, dict[str, float]] = override_params or {}
        self._current_regime: RegimeState = RegimeState.CONSERVATIVE
        self._regime_counter: int = 0
        self._transition_hysteresis: int = 5
        self._pending_regime: RegimeState | None = None
        self._collapsed_consecutive: int = 0

    @property
    def current_regime(self) -> RegimeState:
        return self._current_regime

    def train(self, df_race: pd.DataFrame, *, num_threads: int = 0) -> None:
        """
        レジーム分類器の学習 (軽量・3状態分類)。
        v5.5 leak-fix: 教師ラベルを PRE_RACE 指標のみで計算。
        """
        if num_threads <= 0:
            num_threads = max(1, (os.cpu_count() or 4) // 2)
        features = df_race[self.FEATURE_COLS].copy()
        for col in features.columns:
            if pd.api.types.is_integer_dtype(features[col]):
                features[col] = features[col].astype(float)

        favorite_implied = df_race["favorite_implied_prob_rolling"]
        overround = df_race["overround_rolling"]
        entropy = df_race["entropy_rolling"]

        # 市場状態スコア: 1番人気の implied prob が高い + overround 低い = 効率的
        market_condition_score = favorite_implied * (1 - np.clip(overround - 0.20, 0, 0.15) / 0.15)

        y = np.where(
            (market_condition_score < 0.28) & (entropy > np.median(entropy)),
            0,  # AGGRESSIVE
            np.where(
                market_condition_score > 0.50,
                2,  # COLLAPSED（市場が効率的すぎる → ほぼ停止）
                1,  # CONSERVATIVE
            ),
        )

        # 時系列ベース 80/20 split (最後20%をvalidに)
        n = len(features)
        split = int(n * 0.8)
        train_features = features.iloc[:split]
        train_y = y[:split]
        valid_features = features.iloc[split:]
        valid_y = y[split:]

        train_data = lgb.Dataset(train_features, label=train_y)
        valid_data = lgb.Dataset(valid_features, label=valid_y, reference=train_data)

        self.model = lgb.train(
            {
                "objective": "multiclass",
                "num_class": 3,
                "metric": "multi_logloss",
                "learning_rate": 0.05,
                "num_leaves": 7,
                "min_data_in_leaf": 30,
                "feature_fraction": 0.8,
                "num_threads": num_threads,
                "verbose": -1,
            },
            train_data,
            num_boost_round=100,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        )

    def detect(self, recent_stats: pd.DataFrame) -> RegimeState:
        """
        直近レースの集計値から現在のレジームを判定。
        ヒステリシス付き: 連続Nレースで同じ状態が続いた場合のみ遷移。
        """
        if len(recent_stats) < self.cfg.min_samples:
            return RegimeState.CONSERVATIVE

        features = recent_stats[self.FEATURE_COLS].iloc[[-1]]
        best_iter = self.model.best_iteration
        probs = self.model.predict(features, num_iteration=best_iter)[0]

        raw_regime_idx = int(np.argmax(probs))
        regime_map: list[RegimeState] = [
            RegimeState.AGGRESSIVE,
            RegimeState.CONSERVATIVE,
            RegimeState.COLLAPSED,
        ]
        raw_regime = regime_map[raw_regime_idx]

        # Track consecutive COLLAPSED detections for retrain trigger
        if raw_regime == RegimeState.COLLAPSED:
            self._collapsed_consecutive += 1
        else:
            self._collapsed_consecutive = 0

        # ヒステリシス判定
        if raw_regime == self._current_regime:
            # 同じ状態ならカウンタをリセット
            self._regime_counter = 0
            self._pending_regime = None
        elif raw_regime == self._pending_regime:
            # 前回と同じ候補状態ならカウンタを増加
            self._regime_counter += 1
        else:
            # 新しい候補状態ならカウンタをリセット
            self._pending_regime = raw_regime
            self._regime_counter = 0

        if self._regime_counter >= self._transition_hysteresis:
            old = self._current_regime
            self._current_regime = raw_regime
            self._regime_counter = 0
            self._pending_regime = None
            logger.info(
                f"Regime transition: {old.value} -> {raw_regime.value} "
                f"(probs: A={probs[0]:.2f}, C={probs[1]:.2f}, X={probs[2]:.2f})"
            )

        return self._current_regime

    def get_strategy_params(self, regime: RegimeState) -> dict[str, object]:
        """レジームに応じた戦略パラメータを返す (override_params 上書き付き)"""
        params = self._get_base_params(regime)
        regime_key = regime.value  # "aggressive", "conservative", "collapsed"
        if regime_key in self._override_params:
            for key in ("fractional_kelly", "ev_threshold", "edge_threshold"):
                if key in self._override_params[regime_key]:
                    params[key] = self._override_params[regime_key][key]
        return params

    def _get_base_params(self, regime: RegimeState) -> dict[str, object]:
        """レジームに応じたハードコード戦略パラメータ (ベースライン)"""
        if regime == RegimeState.AGGRESSIVE:
            return {
                "ev_threshold": 1.10,
                "edge_threshold": 0.05,  # Phase 3: JRA控除率25%考慮 +0.01
                "fractional_kelly": 0.50,   # D-01: half-Kelly
                "min_place_prob": 0.08,
                "max_place_odds": 18.0,
                "wide_enabled": False,
                "score_threshold": 0.010,
                "max_bets_per_race": 1,
                "soft_gate_second_margin": 0.50,
                "soft_gate_second_min_edge": 0.03,
                "quality_second_margin": 1.00,
                "quality_second_min_edge": 0.06,
                "quality_second_min_prob": 0.25,
                "runner_up_rescue_margin": 0.25,
                "runner_up_rescue_min_edge": 0.04,
                "runner_up_rescue_min_prob": 0.25,
                "runner_up_rerank_market_condition_max": 0.20,
                "runner_up_rerank_entropy_min": 1.80,
                "runner_up_rerank_entropy_max": 2.30,
                "runner_up_rerank_min_edge": 0.01,
                "runner_up_rerank_min_prob": 0.10,
                "runner_up_rerank_max_odds": 12.0,
                "add_second_keep_min_edge": 0.10,
                "add_second_keep_max_edge": 0.20,
                "weak_prob_prune_threshold": 0.35,
                "description": "歪み強い -> 攻める",
            }
        elif regime == RegimeState.CONSERVATIVE:
            return {
                "ev_threshold": 1.30,
                "edge_threshold": 0.06,  # 6% edge — JRA控除率考慮 (Phase 3)
                "fractional_kelly": 0.25,   # D-01: quarter-Kelly
                "min_place_prob": 0.09,
                "max_place_odds": 18.0,
                "wide_enabled": False,
                "score_threshold": 0.020,
                "max_bets_per_race": 1,
                "prune_turf_candidates": True,
                "weak_prob_prune_threshold": 0.35,
                "description": "効率的 -> 絞る",
            }
        else:  # COLLAPSED
            return {
                "ev_threshold": 1.50,
                "edge_threshold": 0.09,  # 9% edge — JRA控除率考慮 (Phase 3)
                "fractional_kelly": 0.00,   # D-01: no betting
                "min_place_prob": 0.10,
                "max_place_odds": 16.0,
                "wide_enabled": False,
                "score_threshold": 0.050,
                "max_bets_per_race": 1,
                "weak_prob_prune_threshold": 0.35,
                "skip": True,  # D-11: COLLAPSED regime skip flag
                "description": "崩壊 -> ほぼ停止",
            }

    def should_retrain(self) -> bool:
        """COLLAPSED状態が連続100レース続いた場合に再学習をトリガー"""
        return self._collapsed_consecutive >= self.cfg.retrain_trigger
