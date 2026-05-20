"""Market Model 差分専用化 + log_error正規化 (§4)"""

from __future__ import annotations

import logging
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_feature_select(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Select feature columns, filling missing ones with NaN.

    rl_* columns may be absent in test fixtures or edge cases.
    Logs a debug message when filling missing columns.
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.debug("Missing feature columns filled with NaN: %s", missing[:5])
        df = df.copy()
        for c in missing:
            df[c] = float("nan")
    return df[feature_cols].copy()


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
        # TRF-01/02/03 + INT-01/02/03/04: Phase 36
        "form_trend_race_rank",
        "blood_total_wr_race_rank",
        "blood_surface_wr_race_rank",
        "weighted_recent_form_finish",
        "weighted_recent_form_time",
        "grade_x_form_trend",
        "distance_x_closing_index",
        "grade_x_blood_prize_log",
        # HLF-01/02/03: Phase 36 HaronTime L4 + LapTime pace features
        "closing_speed_ratio_avg",
        "closing_speed_ratio_zscore",
        "closing_speed_ratio_trend",
        "harontime_last3f_avg",
        "harontime_last3f_zscore",
        "harontime_last3f_trend",
        "pace_ratio_avg",
        "pace_ratio_zscore",
        "pace_ratio_trend",
        "pace_early_avg",
        "pace_mid_avg",
        "pace_late_avg",
        # HLF-02: HaronTime race-rank
        "closing_speed_ratio_avg_race_rank",
        "harontime_last3f_avg_race_rank",
    ]

    P_PRED_CLIP_MIN: float = 0.01
    P_PRED_CLIP_MAX: float = 0.99

    def __init__(self) -> None:
        self.model: lgb.Booster | None = None

    def train(self, df: pd.DataFrame, *, num_threads: int = 0) -> None:
        """p_market_win_adj を予測するLightGBMモデルを学習 (early stopping付き)"""
        if num_threads <= 0:
            num_threads = max(1, (os.cpu_count() or 4) // 2)
        features = _safe_feature_select(df, self.FEATURE_COLS)
        target = df["p_market_win_adj"]

        # Int64 (nullable int) → float64
        for col in features.columns:
            if pd.api.types.is_integer_dtype(features[col]):
                features[col] = features[col].astype(float)
        # カテゴリカル特徴量の処理
        for col in ["surface", "distance_bin", "grade_code"]:
            if col in features.columns:
                features[col] = features[col].astype("category")

        # 80/20 time-based split (過去→未来、リーク防止)
        n = len(features)
        split = int(n * 0.8)
        train_idx, valid_idx = np.arange(split), np.arange(split, n)

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
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )

    def predict_and_calc_error(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        予測を実行し、正規化差分 (log_error) を計算する。
        p_market_pred は Stage2 に渡さない (Rule 11)。
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        df = df.copy()
        features = _safe_feature_select(df, self.FEATURE_COLS)
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
            .groupby(df["race_id"], observed=True)
            .rank(method="first", ascending=True)
            .astype("Int64")
        )

        # p_market_pred は Stage2 に渡さない (Rule 11)
        df = df.drop(columns=["_p_market_pred_win"])

        return df

    def predict_oof(self, df: pd.DataFrame, n_splits: int = 5, *, num_threads: int = 0) -> pd.DataFrame:
        """OOF (out-of-fold) 予測を生成し、DataFrame の該当列を上書きする。

        学習データ内で KFold CV を行い、各foldのvalid予測を結合。
        最後に全データで再学習して推論用モデルを更新。

        PIT安全: shuffle=False (時系列順序維持)、各foldのvalidデータは
        そのfoldの学習に使用されない。
        """
        from sklearn.model_selection import KFold

        features = _safe_feature_select(df, self.FEATURE_COLS)
        for col in features.columns:
            if pd.api.types.is_integer_dtype(features[col]):
                features[col] = features[col].astype(float)
        for col in ["surface", "distance_bin", "grade_code"]:
            if col in features.columns:
                features[col] = features[col].astype("category")

        target = df["p_market_win_adj"]
        oof_pred = pd.Series(np.nan, index=df.index, name="_p_market_pred_win_oof")

        kf = KFold(n_splits=n_splits, shuffle=False)
        for train_idx, valid_idx in kf.split(features):
            train_data = lgb.Dataset(features.iloc[train_idx], label=target.iloc[train_idx])
            valid_data = lgb.Dataset(
                features.iloc[valid_idx],
                label=target.iloc[valid_idx],
                reference=train_data,
            )
            fold_model = lgb.train(
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
                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
            )
            oof_pred.iloc[valid_idx] = fold_model.predict(
                features.iloc[valid_idx],
                num_iteration=fold_model.best_iteration,
            )

        # 全データで再学習 (推論用)
        self.train(df, num_threads=num_threads)

        # OOF予測で log_error を再計算
        df = df.copy()
        df["_p_market_pred_win"] = oof_pred
        p_pred = oof_pred.clip(self.P_PRED_CLIP_MIN, self.P_PRED_CLIP_MAX)
        p_actual = df["p_market_win_adj"].clip(self.P_PRED_CLIP_MIN, self.P_PRED_CLIP_MAX)
        df["signed_log_error_win"] = np.log(p_actual / p_pred)
        df["abs_log_error_win"] = np.abs(df["signed_log_error_win"])

        # 既存列も上書き (後方互換)
        df["market_log_error_win"] = df["signed_log_error_win"]
        raw_error = df["p_market_win_adj"] - oof_pred
        df["market_pred_error_win"] = raw_error

        # レース内相対ランク
        df["market_error_rank_in_race"] = (
            df["market_log_error_win"]
            .groupby(df["race_id"], observed=True)
            .rank(method="first", ascending=True)
            .astype("Int64")
        )

        # Rule 11: p_market_pred は Stage2 に渡さない
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
