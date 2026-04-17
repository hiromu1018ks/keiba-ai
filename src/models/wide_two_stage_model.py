"""ワイド 分散ベース・リスク調整スコア (§7)"""

from __future__ import annotations

import os
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import pandas as pd

from domain.models import TwoStageConfig


def _train_valid_split(
    features: pd.DataFrame,
    label: pd.Series | np.ndarray,
    valid_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[lgb.Dataset, lgb.Dataset]:
    """学習データを train/valid にランダム分割して (train_data, valid_data) を返す。"""
    n = len(features)
    perm = np.random.RandomState(seed).permutation(n)
    split = int(n * (1 - valid_ratio))
    train_idx, valid_idx = perm[:split], perm[split:]

    label_series = label if isinstance(label, pd.Series) else pd.Series(label)
    train_data = lgb.Dataset(features.iloc[train_idx], label=label_series.iloc[train_idx])
    valid_data = lgb.Dataset(
        features.iloc[valid_idx], label=label_series.iloc[valid_idx], reference=train_data
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

        features = pair_df[self.SHARED_FEATURE_COLS].copy()
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

        features = hit_df[self.SHARED_FEATURE_COLS].copy()
        for col in features.columns:
            if pd.api.types.is_integer_dtype(features[col]):
                features[col] = features[col].astype(float)
        for col in ["surface", "distance_bin", "grade_code"]:
            if col in features.columns:
                features[col] = features[col].astype("category")

        label = hit_df["wide_odds"]

        params = {
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
        features = pair_df[self.SHARED_FEATURE_COLS].copy()
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
