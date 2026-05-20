"""RaceQualityScreener — 分布特徴量 + 結果ベースproxy (§5)"""

from __future__ import annotations

import logging
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_feature_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Select columns, filling missing ones with NaN."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.debug("Missing cols filled with NaN: %s", missing[:5])
        df = df.copy()
        for c in missing:
            df[c] = float("nan")
    return df[cols]


class RaceQualityScreener:
    """
    「このレースは投票する価値があるか」を判定するスクリーナー。

    設計原則:
    1. y に actual_bet_roi を使わない (リーク排除)
    2. Stage2 の出力 (edge) に依存しない指標のみを y に使う
    3. 「このレースの市場は歪んでいるか」をモデル化する
    4. v5.4: 利益proxyは「結果ベース」のみ使用 (Rule 16)
    """

    FEATURE_COLS: list[str] = [
        # Market Model 由来 (正規化差分)
        "market_log_error_max_abs",
        "market_log_error_std",
        "market_log_error_top_q75",
        # 分布特徴量 (v5.1追加)
        "n_positive_errors",
        "top_k_error_sum",
        "positive_error_ratio",
        # v5.4: 結果ベース利益 proxy (Rule 16)
        "hist_hit_rate_topk",
        "hist_roi_topk",
        "hist_positive_return_ratio",
        # 市場構造
        "market_entropy",
        "overround",
        "overround_deviation",
        "field_size",
        # レース条件
        "surface",
        "distance_bin",
        "track_condition_code",
        "grade_code",
        # 難易度スコア
        "difficulty_score",
        # 過去統計
        "hist_win_rate_same_condition",
        "hist_market_entropy_avg",
        # v5.6: EMA平滑化市場指標
        "overround_ema",
        "entropy_ema",
        # 市場構造 (D-06: 市場集中度・歪度)
        "implied_prob_hhi",
        "odds_skewness",
        # 市場クロス整合性 (MCF-07)
        "rl_favorite_in_wide_top1",
        "rl_trio_overlap",
        "rl_market_consistency",
        "rl_trio_odds_ratio",
        "rl_wide_harville_ratio",
        # レースレベル集約 (RLF-01~06)
        "rl_log_odds_entropy",
        "rl_odds_dispersion",
        "rl_top3_odds_gap",
        "rl_top1_odds",
        "rl_favorite_rank_gap",
        "rl_n_horses",
        # Phase36 race-level aggregates (RTG-02/03)
        "phase36_top1_strength",
        "phase36_top1_top2_gap",
        "phase36_field_dispersion",
        "phase36_form_signal_dispersion",
        "phase36_weighted_form_mean",
    ]

    _CATEGORY_COLS: list[str] = ["surface", "distance_bin", "grade_code"]

    def _build_target(self, df_race: pd.DataFrame) -> pd.Series:
        """
        目的変数: レースの「市場歪み × 利益性スコア」
        v5.4: 利益proxyを結果ベースに変更 (Rule 16)
        """
        distortion_score = (
            df_race["market_log_error_max_abs"]
            * df_race["market_entropy"]
            * (1.0 + df_race["n_positive_errors"] / df_race["field_size"])
        )

        # v5.4: 結果ベース利益 proxy (モデル非依存)
        profitability_proxy = np.clip(df_race["hist_roi_topk"], 0.5, 2.0)
        stability_factor = 0.5 + 0.5 * np.clip(
            df_race["hist_positive_return_ratio"],
            0.0,
            1.0,
        )

        target = distortion_score * profitability_proxy * stability_factor
        return target

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """カテゴリ列を category 型に変換"""
        df = df.copy()
        for col in df.columns:
            if pd.api.types.is_integer_dtype(df[col]):
                df[col] = df[col].astype(float)
        for col in self._CATEGORY_COLS:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df

    def train(self, df_race: pd.DataFrame, *, num_threads: int = 0) -> None:
        """スクリーナーモデルを学習 (時系列80/20 split + early_stopping)"""
        if num_threads <= 0:
            num_threads = max(1, (os.cpu_count() or 4) // 2)
        features = self._prepare_features(_safe_feature_cols(df_race, self.FEATURE_COLS))
        y = self._build_target(df_race)

        # 時系列ベース 80/20 split (最後20%をvalidに)
        n = len(features)
        split = int(n * 0.8)
        train_features = features.iloc[:split]
        train_y = y.iloc[:split]
        valid_features = features.iloc[split:]
        valid_y = y.iloc[split:]

        train_data = lgb.Dataset(train_features, label=train_y)
        valid_data = lgb.Dataset(valid_features, label=valid_y, reference=train_data)

        self.model = lgb.train(
            {
                "objective": "regression_l1",
                "metric": "mae",
                "learning_rate": 0.05,
                "num_leaves": 15,
                "num_threads": num_threads,
                "verbose": -1,
            },
            train_data,
            num_boost_round=200,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        self.threshold = float(y.quantile(0.60))

    def should_bet(self, race_features: dict) -> bool:
        """レースが投票対象かを判定"""
        score = self.predict_score(race_features)
        return score >= self.threshold

    def predict_score(self, race_features: dict) -> float:
        """品質スコアを返す (should_bet と同じモデル推論、bool 変換なし)"""
        features = self._prepare_features(
            _safe_feature_cols(pd.DataFrame([race_features]), self.FEATURE_COLS),
        )
        best_iter = self.model.best_iteration
        return float(self.model.predict(features, num_iteration=best_iter)[0])

    def calibrate_threshold(
        self,
        df_race: pd.DataFrame,
        target_investment_rate: float = 0.40,
    ) -> None:
        """
        投票レース比率が target_investment_rate になるよう閾値を調整。
        訓練データでのみ使用。out-of-sample では固定。
        """
        features = self._prepare_features(_safe_feature_cols(df_race, self.FEATURE_COLS))
        best_iter = self.model.best_iteration
        scores = self.model.predict(features, num_iteration=best_iter)
        self.threshold = float(
            np.quantile(scores, 1.0 - target_investment_rate),
        )
