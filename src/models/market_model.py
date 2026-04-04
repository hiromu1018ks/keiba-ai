"""Market Model 差分専用化 + log_error正規化 (§4)"""

from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np
import pandas as pd


class MarketModel:
    """
    市場確率を予測し、予測値と実際の市場確率の差分 (log_error) のみを下流に渡す。

    v5.0: p_market_pred を出力から除外、market_pred_error のみを Stage2 に渡す
    v5.1: market_pred_error を正規化 (log_error) してスケール問題を解消
    v5.3: p_pred と p_market の両側クリップで log_error の発散を防止 (Rule 13)
    """

    FEATURE_COLS: list[str] = [
        # レース条件
        "surface",
        "distance_bin",
        "track_condition_code",
        "grade_code",
        "field_size",
        # 馬の基本情報
        "weight_diff_from_mean",
        # レース難易度
        "difficulty_score",
    ]

    P_PRED_CLIP_MIN: float = 0.01
    P_PRED_CLIP_MAX: float = 0.99

    def __init__(self) -> None:
        self.model: lgb.Booster | None = None

    def train(self, df: pd.DataFrame, *, num_threads: int = 0) -> None:
        """p_market_win_adj を予測するLightGBMモデルを学習 (early stopping付き)"""
        if num_threads <= 0:
            num_threads = max(1, (os.cpu_count() or 4) // 2)
        features = df[self.FEATURE_COLS].copy()
        target = df["p_market_win_adj"]

        # Int64 (nullable int) → float64
        for col in features.columns:
            if pd.api.types.is_integer_dtype(features[col]):
                features[col] = features[col].astype(float)
        # カテゴリカル特徴量の処理
        for col in ["surface", "distance_bin", "grade_code"]:
            if col in features.columns:
                features[col] = features[col].astype("category")

        # 80/20 train/valid split (再現性のため固定seed)
        n = len(features)
        perm = np.random.RandomState(42).permutation(n)
        split = int(n * 0.8)
        train_idx, valid_idx = perm[:split], perm[split:]

        train_data = lgb.Dataset(features.iloc[train_idx], label=target.iloc[train_idx])
        valid_data = lgb.Dataset(
            features.iloc[valid_idx], label=target.iloc[valid_idx], reference=train_data
        )

        self.model = lgb.train(
            {
                "objective": "regression_l1",
                "metric": "mae",
                "learning_rate": 0.03,
                "num_leaves": 31,
                "feature_fraction": 0.7,
                "num_threads": num_threads,
                "verbose": -1,
            },
            train_data,
            num_boost_round=300,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

    def predict_and_calc_error(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        予測を実行し、正規化差分 (log_error) を計算する。
        p_market_pred は Stage2 に渡さない (Rule 11)。
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        df = df.copy()
        features = df[self.FEATURE_COLS].copy()
        for col in ["surface", "distance_bin", "grade_code"]:
            if col in features.columns:
                features[col] = features[col].astype("category")

        df["_p_market_pred_win"] = self.model.predict(
            features, num_iteration=self.model.best_iteration
        )

        # v5.3: 両側クリップ (Rule 13)
        p_pred_clipped = np.clip(
            df["_p_market_pred_win"],
            self.P_PRED_CLIP_MIN,
            self.P_PRED_CLIP_MAX,
        )
        p_market_clipped = np.clip(
            df["p_market_win_adj"],
            self.P_PRED_CLIP_MIN,
            self.P_PRED_CLIP_MAX,
        )

        # 生の差分 (分析用)
        raw_error = df["p_market_win_adj"] - df["_p_market_pred_win"]

        # v5.1: 正規化 log_error
        df["market_log_error_win"] = np.log(p_market_clipped / p_pred_clipped)

        # v5.3: signed/abs を分離
        df["signed_log_error_win"] = df["market_log_error_win"]
        df["abs_log_error_win"] = np.abs(df["market_log_error_win"])

        # 生の差分も保持 (後方互換・分析用)
        df["market_pred_error_win"] = raw_error

        # レース内相対ランク (NaN対応: nullable int)
        df["market_error_rank_in_race"] = (
            df["market_log_error_win"]
            .groupby(df["race_id"])
            .rank(method="first", ascending=True)
            .astype("Int64")
        )

        # p_market_pred は Stage2 に渡さない (Rule 11)
        df = df.drop(columns=["_p_market_pred_win"])

        return df

    def get_stage2_features(self) -> list[str]:
        """
        Stage2 に渡す Market Model 由来の特徴量リスト。
        差分のみ。予測値は含まない (Rule 11)。
        """
        return [
            "signed_log_error_win",
            "abs_log_error_win",
            "market_error_rank_in_race",
        ]
