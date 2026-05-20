"""スタックド・アンサンブル — LightGBM + XGBoost + CatBoost → Ridge メタラーナー

Nguyen et al. (2024) の設計に基づく:
- Level 1: 3つのGBMモデルを独立学習 (K-fold OOF予測生成)
- Level 2: OOF予測を特徴量に Ridge 回帰で統合

TwoStageModel の hit_model のドロップイン代替として設計。
best_iteration=0 + predict(X) → ndarray を返すことで互換。
"""

from __future__ import annotations

import logging
import os
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


class StackedEnsemble:
    """3モデル stacked ensemble for binary classification.

    lgb.Booster のインターフェース互換:
    - best_iteration: int (=0, アンサンブルでは使用しない)
    - predict(X, num_iteration=None) → np.ndarray of probabilities
    - feature_name() → list[str]
    - feature_importance(importance_type=) → np.ndarray
    """

    best_iteration: int = 0

    def __init__(
        self,
        cat_cols: list[str] | None = None,
        n_folds: int = 3,
        n_trials: int = 30,
        corr_penalty_weight: float = 0.5,
        corr_threshold: float = 0.85,
    ) -> None:
        self.cat_cols = cat_cols or []
        self.n_folds = n_folds
        self.n_trials = n_trials
        self.corr_penalty_weight = corr_penalty_weight
        self.corr_threshold = corr_threshold
        self._cat_codes: dict[str, dict[str, int]] = {}
        self.lgbm_model: lgb.Booster | None = None
        self.xgb_model = None
        self.cat_model = None
        self.meta_model: Ridge | None = None
        self.best_params: dict[str, dict[str, Any]] = {}

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

        # --- Optuna HP Tuning Phase ---
        self.best_params = self._tune_hyperparams(X_train, y_train, num_threads)

        # --- Level 1: K-fold OOF 予測生成 ---
        n = len(X_train)
        oof_preds = np.full((n, 3), np.nan)
        self._learn_cat_codes(X_train)

        for i in range(self.n_folds):
            # 時系列考慮: 各foldのvalidは後半部分、trainは前半 (expanding window)
            val_start = int(n * (i + 1) / (self.n_folds + 1))
            val_end = int(n * (i + 2) / (self.n_folds + 1)) if i < self.n_folds - 1 else n

            # train: [0, val_start), valid: [val_start, val_end)
            X_tr, y_tr = X_train.iloc[:val_start], y_train.iloc[:val_start]
            X_va = X_train.iloc[val_start:val_end]

            oof_preds[val_start:val_end, 0] = self._train_lgbm_fold(
                X_tr, y_tr, X_va, num_threads, self.best_params["lgbm"],
            )
            oof_preds[val_start:val_end, 1] = self._train_xgb_fold(
                X_tr, y_tr, X_va, num_threads, self.best_params["xgb"],
            )
            oof_preds[val_start:val_end, 2] = self._train_cat_fold(
                X_tr, y_tr, X_va, num_threads, self.best_params["cat"],
            )

        # --- Level 2: Ridge メタラーナー ---
        # NaNが残る行 (OOF対象外) を除外して学習
        valid_mask = ~np.any(np.isnan(oof_preds), axis=1)
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(oof_preds[valid_mask], y_train.values[valid_mask])

        # --- 最終ベースモデル: train+valid 全データで再学習 ---
        X_all = pd.concat([X_train, X_valid], ignore_index=True)
        y_all = pd.concat([y_train, y_valid], ignore_index=True)

        self.lgbm_model = self._train_lgbm_full(X_all, y_all, num_threads, self.best_params["lgbm"])
        self.xgb_model = self._train_xgb_full(X_all, y_all, num_threads, self.best_params["xgb"])
        self.cat_model = self._train_cat_full(X_all, y_all, num_threads, self.best_params["cat"])

        # --- 多様性検証 (D-09, D-10, D-11) ---
        feature_names = list(X_train.columns)
        importances = self._compute_importance(feature_names)
        self._check_diversity(
            oof_preds[valid_mask], y_train.iloc[valid_mask],
            importances, feature_names,
        )

    def predict(self, X: pd.DataFrame, num_iteration: int | None = None) -> np.ndarray:
        """アンサンブル予測。Ridge で3モデルの予測を統合。"""
        p_lgbm = self.lgbm_model.predict(X)

        import xgboost as xgb
        X_num = self._encode_cats(X)
        p_xgb = self.xgb_model.predict(xgb.DMatrix(X_num))

        # CatBoost: predict() はクラスラベル(0/1)を返すため predict_proba() を使用
        p_cat = self.cat_model.predict_proba(X_num)[:, 1]

        stacked = np.column_stack([p_lgbm, p_xgb, p_cat])
        return np.clip(self.meta_model.predict(stacked), 0, 1)

    def feature_name(self) -> list[str]:
        """特徴量名を返す (lgb.Booster 互換)。

        アンサンブル内の LightGBM モデルの特徴量名をそのまま返す。
        3モデルは同じ特徴量空間で学習されるためこれで十分。
        """
        if self.lgbm_model is None:
            return []
        return self.lgbm_model.feature_name()

    def feature_importance(self, importance_type: str = "split") -> np.ndarray:
        """特徴量重要度を返す (lgb.Booster 互換)。

        3ベースモデルの重要度を正規化して平均化した値を返す。
        各モデルの重要度を [0, 1] に正規化後、単純平均する。

        Args:
            importance_type: "split" or "gain" (LightGBM のみ使用)
        """
        if self.lgbm_model is None:
            return np.array([])

        feature_names = self.lgbm_model.feature_name()

        # LightGBM
        lgb_imp = self.lgbm_model.feature_importance(importance_type=importance_type).astype(float)

        # XGBoost
        xgb_scores = self.xgb_model.get_score(importance_type="gain")
        xgb_imp = np.array([xgb_scores.get(f, 0.0) for f in feature_names], dtype=float)

        # CatBoost
        cat_imp = self.cat_model.get_feature_importance().astype(float)

        # 各モデルの重要度を [0, 1] に正規化して平均
        def _normalize(arr: np.ndarray) -> np.ndarray:
            total = arr.sum()
            return arr / total if total > 0 else arr

        normalized = [_normalize(imp) for imp in [lgb_imp, xgb_imp, cat_imp]]
        return np.mean(normalized, axis=0)

    def _encode_cats(self, X: pd.DataFrame) -> pd.DataFrame:
        """カテゴリ列を数値コードに変換 (XGBoost/CatBoost 用)。

        _cat_codesが利用可能な場合は学習時のマッピングを使用し、
        未知カテゴリを-1として扱う。そうでなければcat.codesにフォールバック。
        """
        X_out = X.copy()
        all_cat_cols = [
            c for c in X_out.columns
            if X_out[c].dtype.name == "category"
        ]
        for col in all_cat_cols:
            if col in self._cat_codes:
                codes = self._cat_codes[col]
                X_out[col] = pd.to_numeric(X_out[col].map(codes), errors="coerce").fillna(-1)
            else:
                X_out[col] = X_out[col].cat.codes.astype(float)
        return X_out

    def _learn_cat_codes(self, X: pd.DataFrame) -> None:
        """最初の学習データからカテゴリのコードマップを構築。"""
        for col in X.columns:
            if X[col].dtype.name == "category":
                self._cat_codes[col] = {cat: code for code, cat in enumerate(X[col].cat.categories)}

    # --- Optuna suggest functions (exploration space separation) ---

    def _suggest_lgbm_params(self, trial: Any) -> dict[str, Any]:
        """LightGBM: 浅い木 + 中程度のlr"""
        return {
            "lgb_num_leaves": trial.suggest_int("lgb_num_leaves", 31, 63),
            "lgb_lr": trial.suggest_float("lgb_lr", 0.01, 0.05, log=True),
            "lgb_feat_frac": trial.suggest_float("lgb_feat_frac", 0.3, 0.9),
        }

    def _suggest_xgb_params(self, trial: Any) -> dict[str, Any]:
        """XGBoost: 中程度の深さ + 高めのlr"""
        return {
            "xgb_max_depth": trial.suggest_int("xgb_max_depth", 4, 8),
            "xgb_lr": trial.suggest_float("xgb_lr", 0.03, 0.1, log=True),
            "xgb_col_sample": trial.suggest_float("xgb_col_sample", 0.3, 0.9),
        }

    def _suggest_cat_params(self, trial: Any) -> dict[str, Any]:
        """CatBoost: 深い木 + 低めのlr"""
        return {
            "cat_depth": trial.suggest_int("cat_depth", 6, 10),
            "cat_lr": trial.suggest_float("cat_lr", 0.005, 0.03, log=True),
            "cat_rsm": trial.suggest_float("cat_rsm", 0.3, 0.9),
        }

    # --- Optuna tuning ---

    def _tune_hyperparams(
        self, X_train: pd.DataFrame, y_train: pd.Series, num_threads: int,
    ) -> dict[str, dict[str, Any]]:
        """Optunaで各モデルのHPを個別最適化（相関ペナルティ付き）"""
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        n = len(X_train)
        # K-fold OOF の最終fold validation区間と重複しないよう、
        # OOF対象外の前半データ内でHPチューニング用の80/20 splitを行う
        oob_start = int(n * self.n_folds / (self.n_folds + 1))  # 最終foldのval_start
        split = int(oob_start * 0.8)
        X_t, y_t = X_train.iloc[:split], y_train.iloc[:split]
        X_v, y_v = X_train.iloc[split:oob_start], y_train.iloc[split:oob_start]

        best_params: dict[str, dict[str, Any]] = {}
        ref_preds_list: list[np.ndarray] = []

        for model_name, suggest_fn, eval_fn, ref_fn in [
            ("lgbm", self._suggest_lgbm_params, self._eval_lgbm, self._train_ref_lgbm),
            ("xgb", self._suggest_xgb_params, self._eval_xgb, self._train_ref_xgb),
            ("cat", self._suggest_cat_params, self._eval_cat, self._train_ref_cat),
        ]:
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial, fn=suggest_fn, tf=eval_fn: tf(
                    trial, fn, X_t, y_t, X_v, y_v, num_threads,
                    ref_preds_list=ref_preds_list,
                    corr_penalty_weight=self.corr_penalty_weight,
                    corr_threshold=self.corr_threshold,
                ),
                n_trials=self.n_trials,
            )
            best_params[model_name] = study.best_params

            # Train reference model for next model's correlation penalty
            ref_preds = ref_fn(X_t, y_t, X_v, y_v, num_threads, study.best_params)
            ref_preds_list.append(ref_preds)

            # Log correlation penalty info
            if ref_preds_list and self.corr_penalty_weight > 0:
                corrs = [
                    np.corrcoef(ref_preds, rp)[0, 1]
                    for rp in ref_preds_list[:-1]
                ]
                if corrs:
                    mean_corr = float(np.mean(corrs))
                    if mean_corr > self.corr_threshold:
                        logger.info(
                            "%s correlation penalty applied: mean_corr=%.4f > threshold=%.4f",
                            model_name.upper(), mean_corr, self.corr_threshold,
                        )

        return best_params

    # --- Reference model training for correlation penalty ---

    def _train_ref_lgbm(
        self, X_t: pd.DataFrame, y_t: pd.Series,
        X_v: pd.DataFrame, y_v: pd.Series,
        num_threads: int, best_params: dict[str, Any],
    ) -> np.ndarray:
        """Train reference LGB model with best params for correlation computation."""
        train_data = lgb.Dataset(X_t, label=y_t)
        valid_data = lgb.Dataset(X_v, label=y_v, reference=train_data)
        m = lgb.train(
            {
                "objective": "binary", "metric": "auc",
                "num_leaves": best_params["lgb_num_leaves"],
                "learning_rate": best_params["lgb_lr"],
                "feature_fraction": best_params["lgb_feat_frac"],
                "verbose": -1, "num_threads": num_threads,
            },
            train_data, num_boost_round=500,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )
        return m.predict(X_v)

    def _train_ref_xgb(
        self, X_t: pd.DataFrame, y_t: pd.Series,
        X_v: pd.DataFrame, y_v: pd.Series,
        num_threads: int, best_params: dict[str, Any],
    ) -> np.ndarray:
        """Train reference XGB model with best params for correlation computation."""
        import xgboost as xgb
        X_t_num = self._encode_cats(X_t)
        X_v_num = self._encode_cats(X_v)
        dtrain = xgb.DMatrix(X_t_num, label=y_t)
        dvalid = xgb.DMatrix(X_v_num, label=y_v)
        m = xgb.train(
            {
                "objective": "binary:logistic", "eval_metric": "auc",
                "max_depth": best_params["xgb_max_depth"],
                "learning_rate": best_params["xgb_lr"],
                "colsample_bytree": best_params["xgb_col_sample"],
                "nthread": num_threads,
            },
            dtrain, num_boost_round=500,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )
        return m.predict(dvalid)

    def _train_ref_cat(
        self, X_t: pd.DataFrame, y_t: pd.Series,
        X_v: pd.DataFrame, y_v: pd.Series,
        num_threads: int, best_params: dict[str, Any],
    ) -> np.ndarray:
        """Train reference CAT model with best params for correlation computation."""
        from catboost import CatBoostClassifier
        X_t_num = self._encode_cats(X_t)
        X_v_num = self._encode_cats(X_v)
        m = CatBoostClassifier(
            iterations=500,
            learning_rate=best_params["cat_lr"],
            depth=best_params["cat_depth"],
            rsm=best_params["cat_rsm"],
            thread_count=num_threads,
            verbose=0,
            early_stopping_rounds=100,
            eval_metric="AUC",
        )
        m.fit(X_t_num, y_t, eval_set=(X_v_num, y_v))
        return m.predict_proba(X_v_num)[:, 1]

    @staticmethod
    def _compute_corr_penalty(
        preds: np.ndarray,
        ref_preds_list: list[np.ndarray] | None,
        weight: float,
        threshold: float,
    ) -> float:
        """Compute correlation penalty: weight * max(0, mean_corr - threshold)."""
        if not ref_preds_list or weight <= 0:
            return 0.0
        corrs = [np.corrcoef(preds, ref)[0, 1] for ref in ref_preds_list]
        mean_corr = float(np.mean(corrs))
        return weight * max(0.0, mean_corr - threshold)

    def _eval_lgbm(
        self,
        trial: Any,
        suggest_fn: Any,
        X_t: pd.DataFrame,
        y_t: pd.Series,
        X_v: pd.DataFrame,
        y_v: pd.Series,
        num_threads: int,
        ref_preds_list: list[np.ndarray] | None = None,
        corr_penalty_weight: float = 0.5,
        corr_threshold: float = 0.85,
    ) -> float:
        """LightGBM Optuna objective"""
        from sklearn.metrics import roc_auc_score

        params = suggest_fn(trial)
        train_data = lgb.Dataset(X_t, label=y_t)
        valid_data = lgb.Dataset(X_v, label=y_v, reference=train_data)
        m = lgb.train(
            {
                "objective": "binary", "metric": "auc",
                "num_leaves": params["lgb_num_leaves"],
                "learning_rate": params["lgb_lr"],
                "feature_fraction": params["lgb_feat_frac"],
                "verbose": -1, "num_threads": num_threads,
            },
            train_data, num_boost_round=500,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )
        preds = m.predict(X_v)
        auc = float(roc_auc_score(y_v, preds))

        # Correlation penalty (LGB is first model, ref_preds_list is empty)
        penalty = self._compute_corr_penalty(
            preds, ref_preds_list, corr_penalty_weight, corr_threshold,
        )
        return auc - penalty

    def _eval_xgb(
        self,
        trial: Any,
        suggest_fn: Any,
        X_t: pd.DataFrame,
        y_t: pd.Series,
        X_v: pd.DataFrame,
        y_v: pd.Series,
        num_threads: int,
        ref_preds_list: list[np.ndarray] | None = None,
        corr_penalty_weight: float = 0.5,
        corr_threshold: float = 0.85,
    ) -> float:
        """XGBoost Optuna objective"""
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score

        params = suggest_fn(trial)
        X_t_num = self._encode_cats(X_t)
        X_v_num = self._encode_cats(X_v)
        dtrain = xgb.DMatrix(X_t_num, label=y_t)
        dvalid = xgb.DMatrix(X_v_num, label=y_v)
        m = xgb.train(
            {
                "objective": "binary:logistic", "eval_metric": "auc",
                "max_depth": params["xgb_max_depth"],
                "learning_rate": params["xgb_lr"],
                "colsample_bytree": params["xgb_col_sample"],
                "nthread": num_threads,
            },
            dtrain, num_boost_round=500,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )
        preds = m.predict(dvalid)
        auc = float(roc_auc_score(y_v, preds))

        penalty = self._compute_corr_penalty(
            preds, ref_preds_list, corr_penalty_weight, corr_threshold,
        )
        return auc - penalty

    def _eval_cat(
        self,
        trial: Any,
        suggest_fn: Any,
        X_t: pd.DataFrame,
        y_t: pd.Series,
        X_v: pd.DataFrame,
        y_v: pd.Series,
        num_threads: int,
        ref_preds_list: list[np.ndarray] | None = None,
        corr_penalty_weight: float = 0.5,
        corr_threshold: float = 0.85,
    ) -> float:
        """CatBoost Optuna objective"""
        from catboost import CatBoostClassifier
        from sklearn.metrics import roc_auc_score

        params = suggest_fn(trial)
        X_t_num = self._encode_cats(X_t)
        X_v_num = self._encode_cats(X_v)
        m = CatBoostClassifier(
            iterations=500,
            learning_rate=params["cat_lr"],
            depth=params["cat_depth"],
            rsm=params["cat_rsm"],
            thread_count=num_threads,
            verbose=0,
            early_stopping_rounds=100,
            eval_metric="AUC",
        )
        m.fit(X_t_num, y_t, eval_set=(X_v_num, y_v))
        preds = m.predict_proba(X_v_num)[:, 1]
        auc = float(roc_auc_score(y_v, preds))

        penalty = self._compute_corr_penalty(
            preds, ref_preds_list, corr_penalty_weight, corr_threshold,
        )
        return auc - penalty

    # --- LightGBM helpers ---

    def _train_lgbm_fold(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        nt: int,
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        lr = params["lgb_lr"] if params else 0.03
        num_leaves = params["lgb_num_leaves"] if params else 31
        feat_frac = params["lgb_feat_frac"] if params else 1.0

        # K-fold train部を80/20に分割 (D-05)
        n_tr = len(X_tr)
        es_split = int(n_tr * 0.8)
        X_t, y_t = X_tr.iloc[:es_split], y_tr.iloc[:es_split]
        X_v, y_v = X_tr.iloc[es_split:], y_tr.iloc[es_split:]

        train_data = lgb.Dataset(X_t, label=y_t)
        valid_data = lgb.Dataset(X_v, label=y_v, reference=train_data)

        m = lgb.train(
            {
                "objective": "binary", "metric": "auc",
                "learning_rate": lr, "num_leaves": num_leaves,
                "feature_fraction": feat_frac,
                "verbose": -1, "num_threads": nt,
            },
            train_data, num_boost_round=500,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )
        return m.predict(X_va)

    def _train_lgbm_full(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        nt: int,
        params: dict[str, Any] | None = None,
    ) -> lgb.Booster:
        lr = params["lgb_lr"] if params else 0.03
        num_leaves = params["lgb_num_leaves"] if params else 31
        feat_frac = params["lgb_feat_frac"] if params else 1.0

        # 80/20 split for validation
        n = len(X)
        es_split = int(n * 0.8)
        X_t, y_t = X.iloc[:es_split], y.iloc[:es_split]
        X_v, y_v = X.iloc[es_split:], y.iloc[es_split:]

        train_data = lgb.Dataset(X_t, label=y_t)
        valid_data = lgb.Dataset(X_v, label=y_v, reference=train_data)

        return lgb.train(
            {
                "objective": "binary", "metric": "auc",
                "learning_rate": lr, "num_leaves": num_leaves,
                "feature_fraction": feat_frac,
                "verbose": -1, "num_threads": nt,
            },
            train_data, num_boost_round=500,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )

    # --- XGBoost helpers ---

    def _train_xgb_fold(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        nt: int,
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        import xgboost as xgb
        X_tr_num = self._encode_cats(X_tr)
        X_va_num = self._encode_cats(X_va)

        max_depth = params["xgb_max_depth"] if params else 6
        lr = params["xgb_lr"] if params else 0.03
        col_sample = params["xgb_col_sample"] if params else 1.0

        # K-fold train部を80/20に分割 (D-05)
        n_tr = len(X_tr_num)
        es_split = int(n_tr * 0.8)
        dtrain = xgb.DMatrix(X_tr_num.iloc[:es_split], label=y_tr.iloc[:es_split])
        dvalid = xgb.DMatrix(X_tr_num.iloc[es_split:], y_tr.iloc[es_split:])

        m = xgb.train(
            {
                "objective": "binary:logistic", "eval_metric": "auc",
                "max_depth": max_depth, "learning_rate": lr,
                "colsample_bytree": col_sample, "nthread": nt,
            },
            dtrain, num_boost_round=500,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )
        return m.predict(xgb.DMatrix(X_va_num))

    def _train_xgb_full(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        nt: int,
        params: dict[str, Any] | None = None,
    ) -> Any:
        import xgboost as xgb
        X_num = self._encode_cats(X)

        max_depth = params["xgb_max_depth"] if params else 6
        lr = params["xgb_lr"] if params else 0.03
        col_sample = params["xgb_col_sample"] if params else 1.0

        # 80/20 split for validation
        n = len(X_num)
        es_split = int(n * 0.8)
        dtrain = xgb.DMatrix(X_num.iloc[:es_split], label=y.iloc[:es_split])
        dvalid = xgb.DMatrix(X_num.iloc[es_split:], y.iloc[es_split:])

        return xgb.train(
            {
                "objective": "binary:logistic", "eval_metric": "auc",
                "max_depth": max_depth, "learning_rate": lr,
                "colsample_bytree": col_sample, "nthread": nt,
            },
            dtrain, num_boost_round=500,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )

    # --- CatBoost helpers ---

    def _train_cat_fold(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        nt: int,
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        from catboost import CatBoostClassifier
        X_tr_num = self._encode_cats(X_tr)
        X_va_num = self._encode_cats(X_va)

        depth = params["cat_depth"] if params else 6
        lr = params["cat_lr"] if params else 0.03
        rsm = params["cat_rsm"] if params else 1.0

        # K-fold train部を80/20に分割 (D-05)
        n_tr = len(X_tr_num)
        es_split = int(n_tr * 0.8)

        m = CatBoostClassifier(
            iterations=500,
            learning_rate=lr,
            depth=depth,
            rsm=rsm,
            thread_count=nt,
            verbose=0,
            early_stopping_rounds=100,
            eval_metric="AUC",
        )
        m.fit(
            X_tr_num.iloc[:es_split], y_tr.iloc[:es_split],
            eval_set=(X_tr_num.iloc[es_split:], y_tr.iloc[es_split:]),
        )
        return m.predict_proba(X_va_num)[:, 1]

    def _train_cat_full(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        nt: int,
        params: dict[str, Any] | None = None,
    ) -> Any:
        from catboost import CatBoostClassifier
        X_num = self._encode_cats(X)

        depth = params["cat_depth"] if params else 6
        lr = params["cat_lr"] if params else 0.03
        rsm = params["cat_rsm"] if params else 1.0

        # 80/20 split for validation
        n = len(X_num)
        es_split = int(n * 0.8)

        m = CatBoostClassifier(
            iterations=500,
            learning_rate=lr,
            depth=depth,
            rsm=rsm,
            thread_count=nt,
            verbose=0,
            early_stopping_rounds=100,
            eval_metric="AUC",
        )
        m.fit(
            X_num.iloc[:es_split], y.iloc[:es_split],
            eval_set=(X_num.iloc[es_split:], y.iloc[es_split:]),
        )
        return m

    # --- Diversity verification ---

    def _compute_importance(self, feature_names: list[str]) -> list[np.ndarray]:
        """各ベースモデルのfeature importanceを抽出"""
        # LightGBM
        lgb_imp = self.lgbm_model.feature_importance(importance_type="gain")

        # XGBoost
        xgb_scores = self.xgb_model.get_score(importance_type="gain")
        # get_scoreは存在する特徴量のみ返す — 全特徴量分の配列に変換
        xgb_imp = np.array([xgb_scores.get(f, 0.0) for f in feature_names], dtype=float)

        # CatBoost
        cat_imp = self.cat_model.get_feature_importance()

        return [lgb_imp, xgb_imp, cat_imp]

    def _check_diversity(
        self,
        oof_preds: np.ndarray,
        y_train: pd.Series,
        importances: list[np.ndarray],
        feature_names: list[str],
    ) -> None:
        """OOF予測の多様性を検証 (D-09, D-10, D-11)"""
        from scipy.stats import spearmanr

        # ペアワイズ相関 (D-09)
        corr_matrix = np.corrcoef(oof_preds.T)
        pairs = [(0, 1, "LGB-XGB"), (0, 2, "LGB-CAT"), (1, 2, "XGB-CAT")]
        for i, j, name in pairs:
            c = corr_matrix[i, j]
            logger.info("OOF prediction correlation %s: %.4f", name, c)
            if c >= 0.95:
                logger.warning(
                    "High prediction correlation %s: %.4f >= 0.95"
                    " — diversity may be insufficient",
                    name, c,
                )

        # Feature importance Spearman順位相関 (D-10)
        for i, j, name in pairs:
            rho, _ = spearmanr(importances[i], importances[j])
            logger.info("Feature importance rank correlation %s: %.4f", name, rho)
            if rho > 0.8:
                logger.warning(
                    "High importance correlation %s: %.4f > 0.8"
                    " — models rely on similar features",
                    name, rho,
                )
