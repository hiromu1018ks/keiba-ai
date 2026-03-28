"""レジーム検知モデル -- 3状態分類 + ヒステリシス (9.5)"""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from domain.models import RegimeConfig
from domain.types import RegimeState

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    市場状態の切り替えを検知し、戦略パラメータを動的に調整する。

    v5.4: 軽量モデル + Stage2非依存特徴量 + ヒステリシス
    v5.5: 教師ラベルを市場指標ベースに変更 (Rule 19)
    """

    FEATURE_COLS: list[str] = [
        # 市場歪み (直近200レース集計)
        "market_error_std",
        "market_error_mean",
        "market_entropy_mean",
        "overround_mean",
        # 市場側指標 (v5.4改: 戦略非依存の市場状態)
        "favorite_win_rate",
        "flb_slope",
        "odds_volatility_mean",
        # ROI EMA (Phase 3: 実データ化, v5.5: 教師ラベルには不使用)
        "favorite_roi_ema",
        "mid_roi_ema",
        "longshot_roi_ema",
        # レース構造
        "field_size_mean",
    ]

    def __init__(self, cfg: RegimeConfig | None = None) -> None:
        self.cfg = cfg or RegimeConfig()
        self._current_regime: RegimeState = RegimeState.CONSERVATIVE
        self._regime_counter: int = 0
        self._transition_hysteresis: int = 5
        self._pending_regime: RegimeState | None = None

    @property
    def current_regime(self) -> RegimeState:
        return self._current_regime

    def train(self, df_race: pd.DataFrame) -> None:
        """
        レジーム分類器の学習 (軽量・3状態分類)。
        v5.5: 教師ラベルを市場指標ベースに変更 (Rule 19)。
        """
        features = df_race[self.FEATURE_COLS].copy()
        for col in features.columns:
            if pd.api.types.is_integer_dtype(features[col]):
                features[col] = features[col].astype(float)

        fav = df_race["favorite_win_rate"]
        overround = df_race["overround_mean"]
        entropy = df_race["market_entropy_mean"]

        # 市場効率スコア: favorite勝率が高い + overround が低い = 効率的
        market_efficiency = fav * (1 - np.clip(overround - 0.20, 0, 0.15) / 0.15)

        y = np.where(
            (market_efficiency < 0.28) & (entropy > np.median(entropy)),
            0,  # AGGRESSIVE
            np.where(
                market_efficiency < 0.18,
                2,  # COLLAPSED
                1,
            ),  # CONSERVATIVE
        )

        self.model = lgb.train(
            {
                "objective": "multiclass",
                "num_class": 3,
                "metric": "multi_logloss",
                "learning_rate": 0.05,
                "num_leaves": 7,
                "min_data_in_leaf": 50,
                "feature_fraction": 0.8,
                "verbose": -1,
            },
            lgb.Dataset(features, label=y),
            num_boost_round=100,
        )

    def detect(self, recent_stats: pd.DataFrame) -> RegimeState:
        """
        直近レースの集計値から現在のレジームを判定。
        ヒステリシス付き: 連続Nレースで同じ状態が続いた場合のみ遷移。
        """
        if len(recent_stats) < self.cfg.min_samples:
            return RegimeState.CONSERVATIVE

        features = recent_stats[self.FEATURE_COLS].iloc[[-1]]
        probs = self.model.predict(features)[0]

        raw_regime_idx = int(np.argmax(probs))
        regime_map: list[RegimeState] = [
            RegimeState.AGGRESSIVE,
            RegimeState.CONSERVATIVE,
            RegimeState.COLLAPSED,
        ]
        raw_regime = regime_map[raw_regime_idx]

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
        """レジームに応じた戦略パラメータを返す"""
        if regime == RegimeState.AGGRESSIVE:
            return {
                "ev_threshold": 1.10,
                "score_threshold": 0.010,
                "max_bets_per_race": 3,
                "description": "歪み強い -> 攻める",
            }
        elif regime == RegimeState.CONSERVATIVE:
            return {
                "ev_threshold": 1.30,
                "score_threshold": 0.020,
                "max_bets_per_race": 2,
                "description": "効率的 -> 絞る",
            }
        else:  # COLLAPSED
            return {
                "ev_threshold": 1.50,
                "score_threshold": 0.050,
                "max_bets_per_race": 1,
                "description": "崩壊 -> ほぼ停止",
            }

    def should_retrain(self) -> bool:
        """COLLAPSED状態が連続100レース続いた場合に再学習をトリガー"""
        return (
            self._current_regime == RegimeState.COLLAPSED
            and self._regime_counter >= self.cfg.retrain_trigger
        )
