"""ワイド 分散ベース・リスク調整スコア (§7)"""

from __future__ import annotations

import logging
import os
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import pandas as pd

from domain.models import TwoStageConfig
from models.reproducibility import lightgbm_native_params

logger = logging.getLogger(__name__)


def _safe_feature_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Select columns, filling missing ones with NaN."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.debug("Missing cols filled with NaN: %s", missing[:5])
        df = df.copy()
        for c in missing:
            df[c] = float("nan")
    return df[cols].copy()


def _train_valid_split(
    features: pd.DataFrame,
    label: pd.Series | np.ndarray,
    valid_ratio: float = 0.2,
    seed: int = 42,  # noqa: ARG001 — kept for API compat
) -> tuple[lgb.Dataset, lgb.Dataset]:
    """時系列順に train/valid に分割。未来データのリークを防ぐためランダム分割しない。"""
    n = len(features)
    split = int(n * (1 - valid_ratio))

    label_series = label if isinstance(label, pd.Series) else pd.Series(label)
    train_data = lgb.Dataset(features.iloc[:split], label=label_series.iloc[:split])
    valid_data = lgb.Dataset(
        features.iloc[split:], label=label_series.iloc[split:], reference=train_data
    )
    return train_data, valid_data


class WideTwoStageModel:
    """
    ワイド2段階モデル with リスク調整スコア。

    v5.4改: score = EV / (E × √P) でスケール一致 (シャープレシオ近似)

    理論的根拠:
      Var ≈ P × E² → √Var ≈ E × √P
      score = EV / √Var ≈ EV / (E × √P)
    """

    SHARED_FEATURE_COLS: list[str] = [
        "surface",
        "distance_bin",
        "track_condition_code",
        "grade_code",
        "field_size",
        # Pair-specific market and model features
        "wide_odds",
        "wide_rank_pct",
        "popularity_sum",
        "popularity_gap",
        "kyakusitukubun_cd_combo",
        "p_ability_pair_product",
        "p_ability_pair_min",
        "p_ability_pair_gap",
        "tanodds_ratio",
        "draw_gap",
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
        # D-02: haron_race_gap (Phase 36.1 new)
        "haron_race_gap_avg",
        "haron_race_gap_zscore",
        "haron_race_gap_trend",
        # D-03: pace_adj_finish (Phase 36.1 new)
        "pace_ratio_avg",
        "pace_early_avg",
        "pace_mid_avg",
        "pace_late_avg",
        # HLF-02: HaronTime race-rank
        "closing_speed_ratio_avg_race_rank",
        "harontime_last3f_avg_race_rank",
    ]

    hit_model: Any
    return_model: Any

    def __init__(self) -> None:
        self.hit_model: Any = None
        self.return_model: Any = None

    def train_hit_model(
        self,
        pair_df: pd.DataFrame,
        cfg: TwoStageConfig | None = None,
        *,
        num_threads: int = 0,
    ) -> None:
        """ワイド的中モデルの学習 (binary classification)

        Args:
            pair_df: WideJointPairBuilder.build() の出力
            cfg: 学習ハイパーパラメータ
            num_threads: LightGBM スレッド数。0 の場合は自動計算。
        """
        if num_threads <= 0:
            num_threads = max(1, (os.cpu_count() or 4) // 2)
        cfg = cfg or TwoStageConfig(hit_leaves=15, hit_rounds=300)

        features = _safe_feature_cols(pair_df, self.SHARED_FEATURE_COLS)
        for col in features.columns:
            if pd.api.types.is_integer_dtype(features[col]):
                features[col] = features[col].astype(float)
        for col in ["surface", "distance_bin", "grade_code"]:
            if col in features.columns:
                features[col] = features[col].astype("category")

        label = pair_df["joint_hit"]

        train_data, valid_data = _train_valid_split(features, label)
        self.hit_model = lgb.train(
            {
                **lightgbm_native_params(),
                "objective": "binary",
                "metric": "auc",
                "learning_rate": cfg.hit_lr,
                "num_leaves": cfg.hit_leaves,
                "num_threads": num_threads,
                "verbose": -1,
                "is_unbalance": True,
            },
            train_data,
            num_boost_round=cfg.hit_rounds,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )

    def train_return_model(
        self,
        pair_df: pd.DataFrame,
        cfg: TwoStageConfig | None = None,
        *,
        num_threads: int = 0,
    ) -> None:
        """ワイド払戻モデルの学習 (L1 regression — 的中ペアのみ)

        Args:
            pair_df: WideJointPairBuilder.build() の出力
            cfg: 学習ハイパーパラメータ
            num_threads: LightGBM スレッド数。0 の場合は自動計算。
        """
        if num_threads <= 0:
            num_threads = max(1, (os.cpu_count() or 4) // 2)
        cfg = cfg or TwoStageConfig(return_leaves=15, return_rounds=200)

        hit_df = pair_df[pair_df["joint_hit"] == 1].copy()
        if len(hit_df) < cfg.min_hit_samples:
            raise ValueError(f"的中ペアが不足: {len(hit_df)} < {cfg.min_hit_samples}")

        features = _safe_feature_cols(hit_df, self.SHARED_FEATURE_COLS)
        for col in features.columns:
            if pd.api.types.is_integer_dtype(features[col]):
                features[col] = features[col].astype(float)
        for col in ["surface", "distance_bin", "grade_code"]:
            if col in features.columns:
                features[col] = features[col].astype("category")

        label = hit_df["wide_odds"]

        params = {
            **lightgbm_native_params(),
            "objective": "regression_l1",
            "metric": "mae",
            "learning_rate": cfg.return_lr,
            "num_leaves": cfg.return_leaves,
            "num_threads": num_threads,
            "verbose": -1,
        }
        callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]

        if len(features) < 10:
            # サンプル数が少なすぎる場合は early stopping なし
            self.return_model = lgb.train(
                params,
                lgb.Dataset(features, label=label),
                num_boost_round=cfg.return_rounds,
            )
        else:
            train_data, valid_data = _train_valid_split(features, label)
            self.return_model = lgb.train(
                params,
                train_data,
                num_boost_round=cfg.return_rounds,
                valid_sets=[valid_data],
                callbacks=callbacks,
            )

    def predict_score(self, pair_df: pd.DataFrame) -> pd.DataFrame:
        """
        ワイド馬券ペアのスコアを計算。
        score = EV / (E × √P) (Rule 3, Rule 15)
        """
        pair_df = pair_df.copy()
        features = _safe_feature_cols(pair_df, self.SHARED_FEATURE_COLS)
        for col in features.columns:
            if pd.api.types.is_integer_dtype(features[col]):
                features[col] = features[col].astype(float)
        for col in ["surface", "distance_bin", "grade_code"]:
            if col in features.columns:
                features[col] = features[col].astype("category")

        hit_iter = self.hit_model.best_iteration if self.hit_model.best_iteration > 0 else None
        ret_iter = (
            self.return_model.best_iteration if self.return_model.best_iteration > 0 else None
        )
        pair_df["p_hit"] = self.hit_model.predict(features, num_iteration=hit_iter)
        pair_df["e_return_given_hit"] = self.return_model.predict(features, num_iteration=ret_iter)

        pair_df["ev_wide"] = pair_df["p_hit"] * pair_df["e_return_given_hit"]
        risk_denom = pair_df["e_return_given_hit"] * np.sqrt(np.clip(pair_df["p_hit"], 0.001, None))
        pair_df["wide_score_adj"] = pair_df["ev_wide"] / risk_denom

        return pair_df

    def select_bets(
        self,
        pair_df: pd.DataFrame,
        ev_threshold: float = 1.10,
        score_threshold: float = 0.015,
        max_bets: int = 3,
    ) -> list[dict[str, Any]]:
        """
        2段階フィルタ:
          1. ev_wide >= ev_threshold
          2. wide_score_adj >= score_threshold
        追加フィルタ: popularity_sum >= 6, kyakusitukubun_cd_combo != 0,
                     p_hit >= 0.05, e_return_given_hit >= 2.0
        """
        scored = self.predict_score(pair_df)
        filtered = scored[
            (scored["ev_wide"] >= ev_threshold)
            & (scored["wide_score_adj"] >= score_threshold)
            & (scored["popularity_sum"] >= 6)
            & (scored["p_hit"] >= 0.05)
            & (scored["e_return_given_hit"] >= 2.0)
        ]
        top = filtered.nlargest(max_bets, "wide_score_adj")
        return cast(list[dict[str, Any]], top.to_dict("records"))
