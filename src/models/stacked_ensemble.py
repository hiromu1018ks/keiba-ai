"""スタックド・アンサンブル — LightGBM + XGBoost + CatBoost → Ridge メタラーナー

Nguyen et al. (2024) の設計に基づく:
- Level 1: 3つのGBMモデルを独立学習 (K-fold OOF予測生成)
- Level 2: OOF予測を特徴量に Ridge 回帰で統合

TwoStageModel の hit_model のドロップイン代替として設計。
best_iteration=0 + predict(X) → ndarray を返すことで互換。
"""

from __future__ import annotations

import os
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


class StackedEnsemble:
    """3モデル stacked ensemble for binary classification.

    lgb.Booster のインターフェース互換:
    - best_iteration: int (=0, アンサンブルでは使用しない)
    - predict(X, num_iteration=None) → np.ndarray of probabilities
    """

    best_iteration: int = 0

    def __init__(self, cat_cols: list[str] | None = None, n_folds: int = 3) -> None:
        self.cat_cols = cat_cols or []
        self.n_folds = n_folds
        self.lgbm_model: lgb.Booster | None = None
        self.xgb_model = None
        self.cat_model = None
        self.meta_model: Ridge | None = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        *,
        num_threads: int = 0,
    ) -> None:
        """K-fold OOF でメタラーナーを学習後、全データでベースモデルを再学習。"""
        if num_threads <= 0:
            num_threads = max(1, (os.cpu_count() or 4) // 2)

        # --- Level 1: K-fold OOF 予測生成 ---
        n = len(X_train)
        oof_preds = np.full((n, 3), np.nan)

        for i in range(self.n_folds):
            # 時系列考慮: 各foldのvalidは後半部分、trainは前半 (expanding window)
            val_start = int(n * (i + 1) / (self.n_folds + 1))
            val_end = int(n * (i + 2) / (self.n_folds + 1)) if i < self.n_folds - 1 else n

            # train: [0, val_start), valid: [val_start, val_end)
            X_tr, y_tr = X_train.iloc[:val_start], y_train.iloc[:val_start]
            X_va = X_train.iloc[val_start:val_end]

            oof_preds[val_start:val_end, 0] = self._train_lgbm_fold(X_tr, y_tr, X_va, num_threads)
            oof_preds[val_start:val_end, 1] = self._train_xgb_fold(X_tr, y_tr, X_va, num_threads)
            oof_preds[val_start:val_end, 2] = self._train_cat_fold(X_tr, y_tr, X_va, num_threads)

        # --- Level 2: Ridge メタラーナー ---
        # NaNが残る行 (OOF対象外) を除外して学習
        valid_mask = ~np.any(np.isnan(oof_preds), axis=1)
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(oof_preds[valid_mask], y_train.values[valid_mask])

        # --- 最終ベースモデル: train+valid 全データで再学習 ---
        X_all = pd.concat([X_train, X_valid], ignore_index=True)
        y_all = pd.concat([y_train, y_valid], ignore_index=True)

        self.lgbm_model = self._train_lgbm_full(X_all, y_all, num_threads)
        self.xgb_model = self._train_xgb_full(X_all, y_all, num_threads)
        self.cat_model = self._train_cat_full(X_all, y_all, num_threads)

    def predict(self, X: pd.DataFrame, num_iteration: int | None = None) -> np.ndarray:
        """アンサンブル予測。Ridge で3モデルの予測を統合。"""
        p_lgbm = self.lgbm_model.predict(X)

        import xgboost as xgb
        p_xgb = self.xgb_model.predict(xgb.DMatrix(X))

        # CatBoost: predict() はクラスラベル(0/1)を返すため predict_proba() を使用
        p_cat = self.cat_model.predict_proba(X)[:, 1]

        stacked = np.column_stack([p_lgbm, p_xgb, p_cat])
        return np.clip(self.meta_model.predict(stacked), 0, 1)

    # --- LightGBM helpers ---
    def _train_lgbm_fold(
        self, X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame, nt: int
    ) -> np.ndarray:
        m = lgb.train(
            {"objective": "binary", "metric": "auc", "learning_rate": 0.03,
             "num_leaves": 31, "verbose": -1, "num_threads": nt},
            lgb.Dataset(X_tr, label=y_tr), num_boost_round=300,
        )
        return m.predict(X_va)

    def _train_lgbm_full(self, X: pd.DataFrame, y: pd.Series, nt: int) -> lgb.Booster:
        return lgb.train(
            {"objective": "binary", "metric": "auc", "learning_rate": 0.03,
             "num_leaves": 31, "verbose": -1, "num_threads": nt},
            lgb.Dataset(X, label=y), num_boost_round=300,
        )

    # --- XGBoost helpers ---
    def _train_xgb_fold(
        self, X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame, nt: int
    ) -> np.ndarray:
        import xgboost as xgb
        m = xgb.train(
            {"objective": "binary:logistic", "learning_rate": 0.03,
             "max_depth": 6, "nthread": nt},
            xgb.DMatrix(X_tr, label=y_tr), num_boost_round=300,
        )
        return m.predict(xgb.DMatrix(X_va))

    def _train_xgb_full(self, X: pd.DataFrame, y: pd.Series, nt: int) -> Any:
        import xgboost as xgb
        return xgb.train(
            {"objective": "binary:logistic", "learning_rate": 0.03,
             "max_depth": 6, "nthread": nt},
            xgb.DMatrix(X, label=y), num_boost_round=300,
        )

    # --- CatBoost helpers ---
    def _train_cat_fold(
        self, X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame, nt: int
    ) -> np.ndarray:
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(
            iterations=300, learning_rate=0.03, depth=6,
            thread_count=nt, verbose=0,
        )
        m.fit(X_tr, y_tr)
        return m.predict_proba(X_va)[:, 1]

    def _train_cat_full(self, X: pd.DataFrame, y: pd.Series, nt: int) -> Any:
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(
            iterations=300, learning_rate=0.03, depth=6,
            thread_count=nt, verbose=0,
        )
        m.fit(X, y)
        return m
