"""RaceQualityScreener — 分布特徴量 + 結果ベースproxy (§5)"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd


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
        for col in self._CATEGORY_COLS:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df

    def train(self, df_race: pd.DataFrame) -> None:
        """スクリーナーモデルを学習"""
        features = self._prepare_features(df_race[self.FEATURE_COLS])
        y = self._build_target(df_race)

        self.model = lgb.train(
            {
                "objective": "regression_l1",
                "metric": "mae",
                "learning_rate": 0.05,
                "num_leaves": 15,
                "verbose": -1,
            },
            lgb.Dataset(features, label=y),
            num_boost_round=200,
        )
        self.threshold = float(y.quantile(0.60))

    def should_bet(self, race_features: dict) -> bool:
        """レースが投票対象かを判定"""
        features = self._prepare_features(
            pd.DataFrame([race_features])[self.FEATURE_COLS],
        )
        score = float(self.model.predict(features)[0])
        return score >= self.threshold

    def calibrate_threshold(
        self,
        df_race: pd.DataFrame,
        target_investment_rate: float = 0.40,
    ) -> None:
        """
        投票レース比率が target_investment_rate になるよう閾値を調整。
        訓練データでのみ使用。out-of-sample では固定。
        """
        features = self._prepare_features(df_race[self.FEATURE_COLS])
        scores = self.model.predict(features)
        self.threshold = float(
            np.quantile(scores, 1.0 - target_investment_rate),
        )
