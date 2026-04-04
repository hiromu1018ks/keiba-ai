"""単勝/複勝 2段階モデル -- ゼロ偏重問題の根本解決 (§2)"""

from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from domain.models import TwoStageConfig


def _train_valid_split(
    features: pd.DataFrame,
    label: pd.Series,
    valid_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[lgb.Dataset, lgb.Dataset]:
    """学習データを train/valid にランダム分割して (train_data, valid_data) を返す。"""
    n = len(features)
    perm = np.random.RandomState(seed).permutation(n)
    split = int(n * (1 - valid_ratio))
    train_idx, valid_idx = perm[:split], perm[split:]

    train_data = lgb.Dataset(features.iloc[train_idx], label=label.iloc[train_idx])
    valid_data = lgb.Dataset(
        features.iloc[valid_idx], label=label.iloc[valid_idx], reference=train_data
    )
    return train_data, valid_data


class WinTwoStageModel:
    """
    単勝2段階モデル
    Stage A: P(win)              ← 2値分類
    Stage B: E(win_odds | win)   ← 的中時払戻の回帰
    EV = P(win) × E(win_odds | win)

    市場コピー防止のため p_market_pred_win は除外し、
    market_log_error (正規化差分) のみを使用する (§4)。
    """

    FEATURE_COLS: list[str] = [
        # Stage1出力
        "p_ability_win",
        # Market Model正規化差分 (v5.3: signed/abs log_error)
        "signed_log_error_win",
        "abs_log_error_win",
        # オッズ変化率
        "odds_drop_rate_60_10",
        "odds_drop_rate_30_10",
        "odds_velocity",
        "odds_volatility",
        "popularity_change_30_10",
        # 市場歪み
        "market_entropy",
        "popularity_rank",
        "overround",
        # レース条件
        "surface",
        "distance_bin",
        "track_condition_code",
        "grade_code",
        "field_size",
    ]

    def __init__(self, cfg: TwoStageConfig | None = None) -> None:
        self.cfg = cfg or TwoStageConfig()

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.FEATURE_COLS].copy()
        # Int64 (nullable int) → float64 (LightGBMが対応する型)
        for col in features.columns:
            if pd.api.types.is_integer_dtype(features[col]):
                features[col] = features[col].astype(float)
        for col in ["surface", "distance_bin", "grade_code"]:
            if col in features.columns:
                features[col] = features[col].astype("category")
        return features

    def train_hit_model(self, df: pd.DataFrame) -> None:
        """P(win) の学習 (全出走馬・1着=1 / 他=0)"""
        features = self._prepare_features(df)
        y = (df["kakuteijyuni"] == 1).astype(int)

        train_data, valid_data = _train_valid_split(features, y)
        self.hit_model = lgb.train(
            {
                "objective": "binary",
                "metric": self.cfg.hit_metric,
                "learning_rate": self.cfg.hit_lr,
                "num_leaves": self.cfg.hit_leaves,
                "is_unbalance": True,
                "feature_fraction": 0.7,
                "num_threads": max(1, (os.cpu_count() or 4) // 2),
                "verbose": -1,
            },
            train_data,
            num_boost_round=self.cfg.hit_rounds,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

    def train_return_model(self, df: pd.DataFrame) -> None:
        """
        E(win_odds | win) の学習 (1着馬のみ)。
        ゼロ偏重を完全に排除 -- 学習データにゼロが含まれない。
        """
        hit_df = df[df["kakuteijyuni"] == 1].copy()
        if len(hit_df) < self.cfg.min_hit_samples:
            raise ValueError(
                f"Stage B 学習には最低 {self.cfg.min_hit_samples} 件の"
                f"的中サンプルが必要。現在: {len(hit_df)} 件"
            )

        features = self._prepare_features(hit_df)
        y = hit_df["odds"]

        params = {
            "objective": "regression_l1",
            "metric": self.cfg.return_metric,
            "learning_rate": self.cfg.return_lr,
            "num_leaves": self.cfg.return_leaves,
            "feature_fraction": 0.7,
            "num_threads": max(1, (os.cpu_count() or 4) // 2),
            "verbose": -1,
        }
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]

        if len(features) < 10:
            # サンプル数が少なすぎる場合は early stopping なし
            self.return_model = lgb.train(
                params,
                lgb.Dataset(features, label=y),
                num_boost_round=self.cfg.return_rounds,
            )
        else:
            train_data, valid_data = _train_valid_split(features, y)
            self.return_model = lgb.train(
                params,
                train_data,
                num_boost_round=self.cfg.return_rounds,
                valid_sets=[valid_data],
                callbacks=callbacks,
            )

    def predict_ev(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        EV_win = P(win) × E(win_odds | win)
        """
        df = df.copy()
        features = self._prepare_features(df)

        hit_iter = self.hit_model.best_iteration if self.hit_model.best_iteration > 0 else None
        ret_iter = self.return_model.best_iteration if self.return_model.best_iteration > 0 else None
        df["p_win_pred"] = self.hit_model.predict(
            features, num_iteration=hit_iter
        )
        df["e_return_win_pred"] = self.return_model.predict(
            features, num_iteration=ret_iter
        )
        df["ev_win"] = df["p_win_pred"] * df["e_return_win_pred"]
        return df


class PlaceTwoStageModel:
    """
    複勝2段階モデル
    Stage A: P(place)               ← 3着以内かどうかの分類
    Stage B: E(place_odds | place)  ← 的中時払戻の回帰

    複勝は的中率が高い (約18〜35%) ため Stage B の学習データが豊富。
    return_leaves を少し増やせる (25)。
    """

    FEATURE_COLS: list[str] = WinTwoStageModel.FEATURE_COLS

    def __init__(self, cfg: TwoStageConfig | None = None) -> None:
        self.cfg = cfg or TwoStageConfig()

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.FEATURE_COLS].copy()
        # Int64 (nullable int) → float64 (LightGBMが対応する型)
        for col in features.columns:
            if pd.api.types.is_integer_dtype(features[col]):
                features[col] = features[col].astype(float)
        for col in ["surface", "distance_bin", "grade_code"]:
            if col in features.columns:
                features[col] = features[col].astype("category")
        return features

    def train_hit_model(self, df: pd.DataFrame) -> None:
        """P(place) の学習 (3着以内=1 / それ以外=0)"""
        features = self._prepare_features(df)
        y = (df["kakuteijyuni"] <= 3).astype(int)

        train_data, valid_data = _train_valid_split(features, y)
        self.hit_model = lgb.train(
            {
                "objective": "binary",
                "metric": "auc",
                "learning_rate": self.cfg.hit_lr,
                "num_leaves": self.cfg.hit_leaves,
                "is_unbalance": True,
                "feature_fraction": 0.7,
                "num_threads": max(1, (os.cpu_count() or 4) // 2),
                "verbose": -1,
            },
            train_data,
            num_boost_round=self.cfg.hit_rounds,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

    def train_return_model(self, df: pd.DataFrame) -> None:
        """E(place_odds | place) の学習 (3着以内のみ)"""
        hit_df = df[df["kakuteijyuni"] <= 3].copy()

        features = self._prepare_features(hit_df)
        y = hit_df["fukuoddslow"]

        params = {
            "objective": "regression_l1",
            "metric": "mae",
            "learning_rate": self.cfg.return_lr,
            "num_leaves": 25,  # 複勝はサンプル多めなので少し深く
            "feature_fraction": 0.7,
            "num_threads": max(1, (os.cpu_count() or 4) // 2),
            "verbose": -1,
        }
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]

        if len(features) < 10:
            # サンプル数が少なすぎる場合は early stopping なし
            self.return_model = lgb.train(
                params,
                lgb.Dataset(features, label=y),
                num_boost_round=self.cfg.return_rounds,
            )
        else:
            train_data, valid_data = _train_valid_split(features, y)
            self.return_model = lgb.train(
                params,
                train_data,
                num_boost_round=self.cfg.return_rounds,
                valid_sets=[valid_data],
                callbacks=callbacks,
            )

    def predict_ev(self, df: pd.DataFrame) -> pd.DataFrame:
        """EV_place = P(place) × E(place_odds | place)"""
        df = df.copy()
        features = self._prepare_features(df)

        hit_iter = self.hit_model.best_iteration if self.hit_model.best_iteration > 0 else None
        ret_iter = self.return_model.best_iteration if self.return_model.best_iteration > 0 else None
        df["p_place_pred"] = self.hit_model.predict(
            features, num_iteration=hit_iter
        )
        df["e_return_place_pred"] = self.return_model.predict(
            features, num_iteration=ret_iter
        )
        df["ev_place"] = df["p_place_pred"] * df["e_return_place_pred"]
        return df
