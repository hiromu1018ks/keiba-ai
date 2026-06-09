"""スタックド・アンサンブル — LightGBM + XGBoost + CatBoost → Ridge メタラーナー

Nguyen et al. (2024) の設計に基づく:
- Level 1: 3つのGBMモデルを独立学習 (K-fold OOF予測生成)
- Level 2: OOF予測を特徴量に Ridge 回帰で統合

TwoStageModel の hit_model のドロップイン代替として設計。
best_iteration=0 + predict(X) → ndarray を返すことで互換。
"""

# ruff: noqa: N803,N806

from __future__ import annotations

import logging
import os
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from models.reproducibility import (
    DEFAULT_RANDOM_SEED,
    catboost_params,
    lightgbm_native_params,
    xgboost_params,
)

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

    VALID_SURFACES: set[str] = {"turf", "dirt"}

    # ── 目的関数の重み (クラス定数 dict で変更可能) ──
    OBJECTIVE_WEIGHTS: dict[str, float] = {
        "auc": 0.30,
        "brier": 0.20,
        "top1_hit": 0.25,
        "top1_roi": 0.20,
        "stability": 0.05,
    }

    # ── Surface 別探索空間 ──
    # 既存値を包含しつつ、ダートは正則化側を広く探索
    SEARCH_SPACES: dict[str, dict[str, dict[str, tuple]]] = {
        "turf": {
            "lgbm": {
                "lgb_num_leaves": (31, 63),
                "lgb_lr": (0.01, 0.05),
                "lgb_feat_frac": (0.3, 0.9),
                "lgb_min_child_samples": (10, 100),
                "lgb_lambda_l1": (0.0, 5.0),
                "lgb_lambda_l2": (0.0, 5.0),
                "lgb_bagging_fraction": (0.6, 1.0),
            },
            "xgb": {
                "xgb_max_depth": (4, 8),
                "xgb_lr": (0.03, 0.1),
                "xgb_col_sample": (0.3, 0.9),
                "xgb_min_child_weight": (1, 50),
                "xgb_reg_alpha": (0.0, 5.0),
                "xgb_reg_lambda": (0.0, 5.0),
                "xgb_subsample": (0.6, 1.0),
            },
            "cat": {
                "cat_depth": (6, 10),
                "cat_lr": (0.005, 0.03),
                "cat_rsm": (0.3, 0.9),
                "cat_l2_leaf_reg": (0.0, 10.0),
                "cat_random_strength": (0.0, 10.0),
                "cat_subsample": (0.6, 1.0),
            },
        },
        "dirt": {
            "lgbm": {
                "lgb_num_leaves": (31, 63),
                "lgb_lr": (0.01, 0.05),
                "lgb_feat_frac": (0.3, 0.9),
                "lgb_min_child_samples": (10, 100),
                "lgb_lambda_l1": (0.0, 10.0),
                "lgb_lambda_l2": (0.0, 10.0),
                "lgb_bagging_fraction": (0.5, 1.0),
            },
            "xgb": {
                "xgb_max_depth": (4, 8),
                "xgb_lr": (0.03, 0.1),
                "xgb_col_sample": (0.3, 0.9),
                "xgb_min_child_weight": (1, 50),
                "xgb_reg_alpha": (0.0, 10.0),
                "xgb_reg_lambda": (0.0, 10.0),
                "xgb_subsample": (0.5, 1.0),
            },
            "cat": {
                "cat_depth": (6, 10),
                "cat_lr": (0.005, 0.03),
                "cat_rsm": (0.3, 0.9),
                "cat_l2_leaf_reg": (0.0, 20.0),
                "cat_random_strength": (0.0, 10.0),
                "cat_subsample": (0.5, 1.0),
            },
        },
    }

    # ── 旧 best_params キーなし時の既定値 ──
    PARAM_DEFAULTS: dict[str, dict[str, Any]] = {
        "lgbm": {
            "lgb_min_child_samples": 20,
            "lgb_lambda_l1": 0.0,
            "lgb_lambda_l2": 0.0,
            "lgb_bagging_fraction": 1.0,
        },
        "xgb": {
            "xgb_min_child_weight": 1,
            "xgb_reg_alpha": 0.0,
            "xgb_reg_lambda": 0.0,
            "xgb_subsample": 1.0,
        },
        "cat": {
            "cat_l2_leaf_reg": 3.0,
            "cat_random_strength": 1.0,
            "cat_subsample": 1.0,
        },
        "meta": {
            "ridge_alpha": 1.0,
            "orthogonalize_threshold": 0.95,
            "orthogonalize_strength": 0.5,
        },
    }

    # ──────────────────────── init ────────────────────────

    def __init__(
        self,
        cat_cols: list[str] | None = None,
        n_folds: int = 3,
        n_trials: int = 30,
        corr_penalty_weight: float = 0.10,
        corr_threshold: float = 0.85,
        orthogonalize_threshold: float = 0.95,
        orthogonalize_strength: float = 0.5,
        surface: str | None = None,
    ) -> None:
        self.cat_cols = cat_cols or []
        self.n_folds = n_folds
        self.n_trials = n_trials
        self.corr_penalty_weight = corr_penalty_weight
        self.corr_threshold = corr_threshold
        self.orthogonalize_threshold = orthogonalize_threshold
        self.orthogonalize_strength = orthogonalize_strength
        self.surface = surface
        self._cat_codes: dict[str, dict[str, int]] = {}
        self.lgbm_model: lgb.Booster | None = None
        self.xgb_model = None
        self.cat_model = None
        self.meta_model: Ridge | None = None
        self.best_params: dict[str, dict[str, Any]] = {}
        self._train_feature_names: list[str] = []
        self._orthogonalization: list[dict[str, Any]] = []

    # ──────────────────────── public API ────────────────────────

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        *,
        num_threads: int = 0,
        train_race_ids: pd.Series | None = None,
        valid_race_ids: pd.Series | None = None,
        train_odds: pd.Series | None = None,
        train_dates: pd.Series | None = None,
    ) -> None:
        """K-fold OOF でメタラーナーを学習後、全データでベースモデルを再学習。"""
        # Surface 検証: 新規学習時は必須
        if self.surface is None or self.surface not in self.VALID_SURFACES:
            raise ValueError(f"surface must be one of {self.VALID_SURFACES}, got {self.surface!r}")

        if num_threads <= 0:
            num_threads = max(1, (os.cpu_count() or 4) // 2)

        self._learn_cat_codes(X_train)
        self._train_feature_names = list(X_train.columns)
        train_groups = self._normalize_groups(train_race_ids, len(X_train))

        # --- Optuna HP Tuning Phase (ベース3モデル) ---
        self.best_params = self._tune_hyperparams(
            X_train,
            y_train,
            num_threads,
            race_ids=train_groups,
            train_odds=train_odds,
            train_dates=train_dates,
        )

        # --- Level 1: K-fold OOF 予測生成 ---
        n = len(X_train)
        oof_preds = np.full((n, 3), np.nan)

        for train_idx, valid_idx in self._expanding_group_splits(train_groups, self.n_folds):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_va = X_train.iloc[valid_idx]

            oof_preds[valid_idx, 0] = self._train_lgbm_fold(
                X_tr,
                y_tr,
                X_va,
                num_threads,
                self.best_params["lgbm"],
                train_groups=train_groups.iloc[train_idx],
            )
            oof_preds[valid_idx, 1] = self._train_xgb_fold(
                X_tr,
                y_tr,
                X_va,
                num_threads,
                self.best_params["xgb"],
                train_groups=train_groups.iloc[train_idx],
            )
            oof_preds[valid_idx, 2] = self._train_cat_fold(
                X_tr,
                y_tr,
                X_va,
                num_threads,
                self.best_params["cat"],
                train_groups=train_groups.iloc[train_idx],
            )

        # --- Level 1.5: Ridge/直交化の別 Study ---
        valid_mask = ~np.any(np.isnan(oof_preds), axis=1)
        oof_valid_groups = train_groups[valid_mask].reset_index(drop=True)
        oof_valid_odds = (
            train_odds[valid_mask].reset_index(drop=True) if train_odds is not None else None
        )
        oof_valid_dates = (
            train_dates[valid_mask].reset_index(drop=True) if train_dates is not None else None
        )
        meta_params = self._tune_meta_params(
            oof_preds[valid_mask],
            y_train.values[valid_mask],
            race_groups=oof_valid_groups,
            odds=oof_valid_odds,
            dates=oof_valid_dates,
        )
        self.best_params["meta"] = meta_params

        # 最良 meta params を反映して OOF 全体で直交化 + Ridge を fit し直す
        self.orthogonalize_threshold = meta_params.get(
            "orthogonalize_threshold",
            self.PARAM_DEFAULTS["meta"]["orthogonalize_threshold"],
        )
        self.orthogonalize_strength = meta_params.get(
            "orthogonalize_strength",
            self.PARAM_DEFAULTS["meta"]["orthogonalize_strength"],
        )
        self._orthogonalization = []  # reset before full-OOF fit

        # --- Level 2: Ridge メタラーナー ---
        stack_features = self._fit_prediction_orthogonalizer(oof_preds[valid_mask])
        self.meta_model = Ridge(
            alpha=float(meta_params.get("ridge_alpha", self.PARAM_DEFAULTS["meta"]["ridge_alpha"]))
        )
        self.meta_model.fit(stack_features, y_train.values[valid_mask])

        # --- 最終ベースモデル: train+valid 全データで再学習 ---
        X_all = pd.concat([X_train, X_valid], ignore_index=True)
        y_all = pd.concat([y_train, y_valid], ignore_index=True)
        valid_groups = self._normalize_groups(
            valid_race_ids,
            len(X_valid),
            offset=int(train_groups.max()) + 1,
        )
        all_groups = pd.concat([train_groups, valid_groups], ignore_index=True)

        self.lgbm_model = self._train_lgbm_full(
            X_all,
            y_all,
            num_threads,
            self.best_params["lgbm"],
            train_groups=all_groups,
        )
        self.xgb_model = self._train_xgb_full(
            X_all,
            y_all,
            num_threads,
            self.best_params["xgb"],
            train_groups=all_groups,
        )
        self.cat_model = self._train_cat_full(
            X_all,
            y_all,
            num_threads,
            self.best_params["cat"],
            train_groups=all_groups,
        )

        # --- 多様性検証 (D-09, D-10, D-11) ---
        feature_names = list(X_train.columns)
        importances = self._compute_importance(feature_names)
        self._check_diversity(
            stack_features,
            y_train.iloc[valid_mask],
            importances,
            feature_names,
        )

    def predict(self, X: pd.DataFrame, num_iteration: int | None = None) -> np.ndarray:
        """アンサンブル予測。Ridge で3モデルの予測を統合。"""
        if self._train_feature_names:
            missing = [c for c in self._train_feature_names if c not in X.columns]
            if missing:
                raise ValueError(
                    f"StackedEnsemble.predict(): missing feature columns: {missing[:5]}"
                )
            X = X[self._train_feature_names]
        from models.categorical_alignment import align_lightgbm_categories

        X = align_lightgbm_categories(X, self.lgbm_model)
        p_lgbm = self.lgbm_model.predict(X)

        import xgboost as xgb

        X_num = self._encode_cats(X)
        p_xgb = self._predict_xgb_best(self.xgb_model, xgb.DMatrix(X_num))

        # CatBoost: predict() はクラスラベル(0/1)を返すため predict_proba() を使用
        p_cat = self.cat_model.predict_proba(X_num)[:, 1]

        stacked = self._apply_prediction_orthogonalizer(np.column_stack([p_lgbm, p_xgb, p_cat]))
        return np.clip(self.meta_model.predict(stacked), 0, 1)

    def feature_name(self) -> list[str]:
        """特徴量名を返す (lgb.Booster 互換)。"""
        if self.lgbm_model is None:
            return []
        return self.lgbm_model.feature_name()

    def feature_importance(self, importance_type: str = "split") -> np.ndarray:
        """特徴量重要度を返す (lgb.Booster 互換)。"""
        if self.lgbm_model is None:
            return np.array([])

        feature_names = self.lgbm_model.feature_name()
        lgb_imp = self.lgbm_model.feature_importance(importance_type=importance_type).astype(float)
        xgb_scores = self.xgb_model.get_score(importance_type="gain")
        xgb_imp = np.array([xgb_scores.get(f, 0.0) for f in feature_names], dtype=float)
        cat_imp = self.cat_model.get_feature_importance().astype(float)

        def _normalize(arr: np.ndarray) -> np.ndarray:
            total = arr.sum()
            return arr / total if total > 0 else np.zeros_like(arr)

        normalized = [_normalize(imp) for imp in [lgb_imp, xgb_imp, cat_imp]]
        active = [n for n in normalized if n.sum() > 0]
        if not active:
            return np.zeros(len(lgb_imp), dtype=float)
        avg = np.mean(active, axis=0)
        total = avg.sum()
        return avg / total if total > 0 else avg

    # ──────────────────────── cat encoding ────────────────────────

    def _encode_cats(self, X: pd.DataFrame) -> pd.DataFrame:
        """カテゴリ列を数値コードに変換 (XGBoost/CatBoost 用)。"""
        all_cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]
        if not all_cat_cols:
            return X

        X_out = X.copy()
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

    # ──────────────────────── group splitting ────────────────────────

    @staticmethod
    def _normalize_groups(
        race_ids: pd.Series | None,
        n_rows: int,
        *,
        offset: int = 0,
    ) -> pd.Series:
        """行順を維持したレースグループ番号へ正規化する。"""
        if race_ids is None:
            return pd.Series(np.arange(offset, offset + n_rows), dtype="int64")
        if len(race_ids) != n_rows:
            raise ValueError("race_ids length must match feature rows")
        codes, _ = pd.factorize(race_ids.reset_index(drop=True), sort=False)
        if np.any(codes < 0):
            raise ValueError("race_ids must not contain missing values")
        if len(codes) > 1 and np.any(np.diff(codes) < 0):
            raise ValueError("rows for each race_id must be contiguous and chronologically ordered")
        return pd.Series(codes + offset, dtype="int64")

    @staticmethod
    def _group_boundary(groups: pd.Series, ratio: float) -> int:
        """指定比率以下のレースを丸ごと含む行境界を返す。"""
        unique_groups = pd.unique(groups)
        if len(unique_groups) < 2:
            raise ValueError("at least two races are required for time split")
        n_left = min(max(1, int(len(unique_groups) * ratio)), len(unique_groups) - 1)
        left_groups = set(unique_groups[:n_left])
        return int(groups.isin(left_groups).sum())

    @classmethod
    def race_group_split_index(cls, race_ids: pd.Series, ratio: float = 0.8) -> int:
        """レースを分断しない行分割位置を返す。"""
        groups = cls._normalize_groups(race_ids, len(race_ids))
        return cls._group_boundary(groups, ratio)

    @classmethod
    def _expanding_group_splits(
        cls,
        groups: pd.Series,
        n_folds: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """レースを分断しないexpanding-window OOF splitを返す。"""
        unique_groups = pd.unique(groups)
        if len(unique_groups) < n_folds + 1:
            raise ValueError(f"at least {n_folds + 1} races are required")
        group_chunks = np.array_split(unique_groups, n_folds + 1)
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for fold in range(n_folds):
            train_groups = np.concatenate(group_chunks[: fold + 1])
            valid_groups = group_chunks[fold + 1]
            train_idx = np.flatnonzero(groups.isin(train_groups).to_numpy())
            valid_idx = np.flatnonzero(groups.isin(valid_groups).to_numpy())
            splits.append((train_idx, valid_idx))
        return splits

    # ──────────────────────── prediction helpers ────────────────────────

    @staticmethod
    def _predict_xgb_best(model: Any, data: Any) -> np.ndarray:
        """early stoppingの最良反復までに限定してXGBoost予測する。"""
        best_iteration = getattr(model, "best_iteration", None)
        if best_iteration is None:
            return model.predict(data)
        return model.predict(data, iteration_range=(0, int(best_iteration) + 1))

    # ──────────────────────── objective functions ────────────────────────

    @staticmethod
    def _probability_objective(y_true: pd.Series, preds: np.ndarray) -> float:
        """順位性能と確率精度を両立するOptuna目的関数 (旧版、後方互換用)。"""
        from sklearn.metrics import brier_score_loss, roc_auc_score

        clipped = np.clip(np.asarray(preds, dtype=float), 1e-6, 1 - 1e-6)
        auc = float(roc_auc_score(y_true, clipped)) if y_true.nunique() >= 2 else 0.5
        brier = float(brier_score_loss(y_true, clipped))
        return auc - 0.25 * brier

    @staticmethod
    def _race_top1_objective(
        y_true: pd.Series,
        preds: np.ndarray,
        *,
        race_ids: pd.Series,
        odds: pd.Series | None = None,
        dates: pd.Series | None = None,
        weights: dict[str, float] | None = None,
    ) -> float:
        """Race Top-1 目的関数。

        各レースで予測確率最大の1頭だけを選び、ROI を評価。
        指標: AUC, Brier, Top1HitRate, Top1ROI, 時系列安定性 の加重和。
        常に有限値。単一クラスにも対応。
        """
        from sklearn.metrics import brier_score_loss, roc_auc_score

        w = weights if weights is not None else StackedEnsemble.OBJECTIVE_WEIGHTS
        preds_arr = np.clip(np.asarray(preds, dtype=float), 1e-6, 1 - 1e-6)
        n = len(y_true)
        if n == 0:
            return 0.0

        # ── 基本指標 (全行) ──
        has_two_classes = y_true.nunique() >= 2
        auc = float(roc_auc_score(y_true, preds_arr)) if has_two_classes else 0.5
        brier = float(brier_score_loss(y_true, preds_arr))

        # ── レース Top-1 集計 ──
        df = pd.DataFrame(
            {
                "y": y_true.values,
                "pred": preds_arr,
                "race_id": race_ids.values,
                "odds": odds.values if odds is not None else np.nan,
            }
        )
        if dates is not None:
            df["date"] = dates.values

        # Top-1: 各レースで予測確率最大の1頭。同率時は行順 (idxmax)
        top1 = df.loc[df.groupby("race_id", observed=True)["pred"].idxmax()]
        n_races = len(top1)
        if n_races == 0:
            return 0.0

        top1_hit = float((top1["y"] == 1).mean())
        # 集計ROIを先に計算し、その後クリップ (個別クリップは高オッズ的中を潰す)
        valid_odds = np.where(
            np.isfinite(top1["odds"].values) & (top1["odds"].values > 0),
            top1["odds"].values,
            0.0,
        )
        raw_roi = float(np.mean(np.where(top1["y"].values == 1, valid_odds, 0.0)))
        top1_roi = float(np.clip(raw_roi, 0.0, 2.0)) / 2.0

        # ── 時系列安定性 ──
        stability = StackedEnsemble._compute_stability(
            top1,
            dates_col="date" if "date" in df.columns else None,
        )

        # ── 加权和 ──
        score = (
            w["auc"] * auc
            + w["brier"] * (1.0 - brier)
            + w["top1_hit"] * top1_hit
            + w["top1_roi"] * top1_roi
            + w["stability"] * stability
        )
        return float(score) if np.isfinite(score) else 0.0

    @staticmethod
    def _compute_stability(
        top1_df: pd.DataFrame,
        *,
        dates_col: str | None = None,
    ) -> float:
        """Top-1 ROI の時系列安定性を返す (常に [0, 1] の有限値)。

        年が2年以上あれば年別ROI。そうでなければ3ブロック分割。
        consistency (1-CV) と clipped minimum ROI を 0.5/0.5 で合成。
        """

        def _roi_of(sub: pd.DataFrame) -> float:
            """Mean ROI of a subset (winner→valid odds, else→0)."""
            if len(sub) == 0:
                return 0.0
            valid_odds = np.where(
                np.isfinite(sub["odds"].values) & (sub["odds"].values > 0),
                sub["odds"].values,
                0.0,
            )
            return float(np.mean(np.where(sub["y"].values == 1, valid_odds, 0.0)))

        rois: list[float] = []

        # 年別ROI
        if dates_col and dates_col in top1_df.columns:
            dt = pd.to_datetime(top1_df[dates_col], errors="coerce")
            years = dt.dt.year
            unique_years = sorted(years.dropna().unique())
            if len(unique_years) >= 2:
                rois = [
                    _roi_of(top1_df.loc[years == yr])
                    for yr in unique_years
                    if (years == yr).sum() > 0
                ]

        # ブロック安定性 (3分割) — 年別が不十分な場合のフォールバック
        if len(rois) < 2:
            n = len(top1_df)
            if n < 3:
                return 0.5
            block_size = n // 3
            rois = []
            for i in range(3):
                s = i * block_size
                e = (i + 1) * block_size if i < 2 else n
                rois.append(_roi_of(top1_df.iloc[s:e]))

        if len(rois) < 2:
            return 0.5

        # consistency: 1 - CV (coefficient of variation)
        mean_roi = float(np.mean(rois))
        std_roi = float(np.std(rois))
        denom = max(abs(mean_roi), 1e-6)
        consistency = float(np.clip(1.0 - std_roi / denom, 0.0, 1.0))

        # clipped minimum ROI: 悪期間でもどれだけの払戻があるか
        min_roi = float(min(rois))
        clipped_min = float(np.clip(min_roi, 0.0, 2.0)) / 2.0

        return 0.5 * consistency + 0.5 * clipped_min

    # ──────────────────────── correlation helpers ────────────────────────

    @staticmethod
    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        """有限値かつ分散がある場合のみ相関を返す."""
        valid = np.isfinite(a) & np.isfinite(b)
        if int(valid.sum()) < 2:
            return float("nan")
        a_valid = a[valid]
        b_valid = b[valid]
        if float(np.std(a_valid)) <= 1e-12 or float(np.std(b_valid)) <= 1e-12:
            return float("nan")
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = float(np.corrcoef(a_valid, b_valid)[0, 1])
        return corr if np.isfinite(corr) else float("nan")

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
        corrs = []
        for ref in ref_preds_list:
            c = StackedEnsemble._safe_corr(preds, ref)
            corrs.append(float(c) if np.isfinite(c) else 0.0)
        mean_corr = float(np.mean(corrs))
        return weight * max(0.0, mean_corr - threshold)

    # ──────────────────────── param helpers ────────────────────────

    @staticmethod
    def _get_param(params: dict[str, Any], key: str, model_name: str) -> Any:
        """best_params からキーを取得。旧モデルでキーがない場合は既定値。"""
        if key in params:
            return params[key]
        defaults = StackedEnsemble.PARAM_DEFAULTS.get(model_name, {})
        return defaults.get(key)

    def _build_lgbm_dict(self, params: dict[str, Any], num_threads: int) -> dict[str, Any]:
        """Optuna params → LightGBM 完整パラメータ dict。"""
        d: dict[str, Any] = {
            **lightgbm_native_params(),
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": int(params["lgb_num_leaves"]),
            "learning_rate": float(params["lgb_lr"]),
            "feature_fraction": float(params["lgb_feat_frac"]),
            "min_child_samples": int(self._get_param(params, "lgb_min_child_samples", "lgbm")),
            "lambda_l1": float(self._get_param(params, "lgb_lambda_l1", "lgbm")),
            "lambda_l2": float(self._get_param(params, "lgb_lambda_l2", "lgbm")),
            "verbose": -1,
            "num_threads": num_threads,
        }
        bag_frac = float(self._get_param(params, "lgb_bagging_fraction", "lgbm"))
        if bag_frac < 1.0:
            d["bagging_fraction"] = bag_frac
            d["bagging_freq"] = 1
        return d

    def _build_xgb_dict(self, params: dict[str, Any], num_threads: int) -> dict[str, Any]:
        """Optuna params → XGBoost 完整パラメータ dict。"""
        return {
            **xgboost_params(),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": int(params["xgb_max_depth"]),
            "learning_rate": float(params["xgb_lr"]),
            "colsample_bytree": float(params["xgb_col_sample"]),
            "min_child_weight": int(self._get_param(params, "xgb_min_child_weight", "xgb")),
            "reg_alpha": float(self._get_param(params, "xgb_reg_alpha", "xgb")),
            "reg_lambda": float(self._get_param(params, "xgb_reg_lambda", "xgb")),
            "subsample": float(self._get_param(params, "xgb_subsample", "xgb")),
            "nthread": num_threads,
        }

    def _build_cat_dict(self, params: dict[str, Any], num_threads: int) -> dict[str, Any]:
        """Optuna params → CatBoost 完整パラメータ dict。"""
        d: dict[str, Any] = {
            **catboost_params(),
            "iterations": 500,
            "learning_rate": float(params["cat_lr"]),
            "depth": int(params["cat_depth"]),
            "rsm": float(params["cat_rsm"]),
            "l2_leaf_reg": float(self._get_param(params, "cat_l2_leaf_reg", "cat")),
            "random_strength": float(self._get_param(params, "cat_random_strength", "cat")),
            "thread_count": num_threads,
            "verbose": 0,
            "early_stopping_rounds": 100,
            "eval_metric": "Logloss",
        }
        subsample_val = float(self._get_param(params, "cat_subsample", "cat"))
        if subsample_val < 1.0:
            d["subsample"] = subsample_val
            d["bootstrap_type"] = "Bernoulli"
        return d

    # ──────────────────────── Optuna suggest (surface 別) ────────────────

    def _suggest_lgbm_params(self, trial: Any) -> dict[str, Any]:
        """LightGBM: 浅い木 + 中程度のlr"""
        sp = self.SEARCH_SPACES[self.surface]["lgbm"]
        return {
            "lgb_num_leaves": trial.suggest_int("lgb_num_leaves", *sp["lgb_num_leaves"]),
            "lgb_lr": trial.suggest_float("lgb_lr", *sp["lgb_lr"], log=True),
            "lgb_feat_frac": trial.suggest_float("lgb_feat_frac", *sp["lgb_feat_frac"]),
            "lgb_min_child_samples": trial.suggest_int(
                "lgb_min_child_samples",
                *sp["lgb_min_child_samples"],
            ),
            "lgb_lambda_l1": trial.suggest_float("lgb_lambda_l1", *sp["lgb_lambda_l1"]),
            "lgb_lambda_l2": trial.suggest_float("lgb_lambda_l2", *sp["lgb_lambda_l2"]),
            "lgb_bagging_fraction": trial.suggest_float(
                "lgb_bagging_fraction",
                *sp["lgb_bagging_fraction"],
            ),
        }

    def _suggest_xgb_params(self, trial: Any) -> dict[str, Any]:
        """XGBoost: 中程度の深さ + 高めのlr"""
        sp = self.SEARCH_SPACES[self.surface]["xgb"]
        return {
            "xgb_max_depth": trial.suggest_int("xgb_max_depth", *sp["xgb_max_depth"]),
            "xgb_lr": trial.suggest_float("xgb_lr", *sp["xgb_lr"], log=True),
            "xgb_col_sample": trial.suggest_float("xgb_col_sample", *sp["xgb_col_sample"]),
            "xgb_min_child_weight": trial.suggest_int(
                "xgb_min_child_weight",
                *sp["xgb_min_child_weight"],
            ),
            "xgb_reg_alpha": trial.suggest_float("xgb_reg_alpha", *sp["xgb_reg_alpha"]),
            "xgb_reg_lambda": trial.suggest_float("xgb_reg_lambda", *sp["xgb_reg_lambda"]),
            "xgb_subsample": trial.suggest_float("xgb_subsample", *sp["xgb_subsample"]),
        }

    def _suggest_cat_params(self, trial: Any) -> dict[str, Any]:
        """CatBoost: 深い木 + 低めのlr"""
        sp = self.SEARCH_SPACES[self.surface]["cat"]
        return {
            "cat_depth": trial.suggest_int("cat_depth", *sp["cat_depth"]),
            "cat_lr": trial.suggest_float("cat_lr", *sp["cat_lr"], log=True),
            "cat_rsm": trial.suggest_float("cat_rsm", *sp["cat_rsm"]),
            "cat_l2_leaf_reg": trial.suggest_float("cat_l2_leaf_reg", *sp["cat_l2_leaf_reg"]),
            "cat_random_strength": trial.suggest_float(
                "cat_random_strength",
                *sp["cat_random_strength"],
            ),
            "cat_subsample": trial.suggest_float("cat_subsample", *sp["cat_subsample"]),
        }

    # ──────────────────────── Optuna tuning ────────────────────────

    def _tune_hyperparams(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        num_threads: int,
        *,
        race_ids: pd.Series | None = None,
        train_odds: pd.Series | None = None,
        train_dates: pd.Series | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Optunaで各モデルのHPを個別最適化（相関ペナルティ付き）"""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        groups = self._normalize_groups(race_ids, len(X_train))
        first_oof_train_idx, _ = self._expanding_group_splits(groups, self.n_folds)[0]
        tune_groups = groups.iloc[first_oof_train_idx].reset_index(drop=True)
        tune_split = self._group_boundary(tune_groups, 0.8)
        tune_idx = first_oof_train_idx
        train_idx, valid_idx = tune_idx[:tune_split], tune_idx[tune_split:]
        X_t, y_t = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_v, y_v = X_train.iloc[valid_idx], y_train.iloc[valid_idx]
        X_t_num = self._encode_cats(X_t)
        X_v_num = self._encode_cats(X_v)

        # 検証用 odds / dates / race_groups
        v_groups = groups.iloc[valid_idx].reset_index(drop=True)
        v_odds = (
            train_odds.iloc[valid_idx].reset_index(drop=True) if train_odds is not None else None
        )
        v_dates = (
            train_dates.iloc[valid_idx].reset_index(drop=True) if train_dates is not None else None
        )

        best_params: dict[str, dict[str, Any]] = {}
        ref_preds_list: list[np.ndarray] = []

        for model_name, suggest_fn, eval_fn, ref_fn in [
            ("lgbm", self._suggest_lgbm_params, self._eval_lgbm, self._train_ref_lgbm),
            ("xgb", self._suggest_xgb_params, self._eval_xgb, self._train_ref_xgb),
            ("cat", self._suggest_cat_params, self._eval_cat, self._train_ref_cat),
        ]:
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=DEFAULT_RANDOM_SEED),
            )
            eval_kwargs: dict[str, Any] = {
                "valid_race_groups": v_groups,
                "valid_odds": v_odds,
                "valid_dates": v_dates,
            }
            ref_kwargs: dict[str, Any] = {}
            if model_name == "xgb":
                import xgboost as xgb

                dtrain = xgb.DMatrix(X_t_num, label=y_t)
                dvalid = xgb.DMatrix(X_v_num, label=y_v)
                eval_kwargs.update(dtrain=dtrain, dvalid=dvalid)
                ref_kwargs = dict(eval_kwargs)
            elif model_name == "cat":
                ref_kwargs = {}

            study.optimize(
                lambda trial, fn=suggest_fn, tf=eval_fn, kwargs=eval_kwargs: tf(
                    trial,
                    fn,
                    X_t,
                    y_t,
                    X_v,
                    y_v,
                    num_threads,
                    ref_preds_list=ref_preds_list,
                    corr_penalty_weight=self.corr_penalty_weight,
                    corr_threshold=self.corr_threshold,
                    **kwargs,
                ),
                n_trials=self.n_trials,
            )
            best_params[model_name] = study.best_params

            # Train reference model for next model's correlation penalty
            ref_preds = ref_fn(
                X_t,
                y_t,
                X_v,
                y_v,
                num_threads,
                study.best_params,
                **ref_kwargs,
            )
            ref_preds_list.append(ref_preds)

            # Log correlation penalty info
            if ref_preds_list and self.corr_penalty_weight > 0:
                corrs = []
                for rp in ref_preds_list[:-1]:
                    c = self._safe_corr(ref_preds, rp)
                    corrs.append(float(c) if np.isfinite(c) else 0.0)
                if corrs:
                    mean_corr = float(np.mean(corrs))
                    if mean_corr > self.corr_threshold:
                        logger.info(
                            "%s correlation penalty applied: mean_corr=%.4f > threshold=%.4f",
                            model_name.upper(),
                            mean_corr,
                            self.corr_threshold,
                        )

        return best_params

    def _tune_meta_params(
        self,
        oof_preds: np.ndarray,
        y: np.ndarray,
        *,
        race_groups: pd.Series,
        odds: pd.Series | None = None,
        dates: pd.Series | None = None,
    ) -> dict[str, Any]:
        """Ridge / 直交化パラメータを独立 Optuna Study で探索 (seed=42)。

        OOF 有効領域をレース単位・時系列で meta-train / meta-valid へ分ける。
        meta-train で直交化係数と Ridge を fit し、meta-valid だけで目的関数を評価。
        最良 params 決定後、train() 側で OOF 全体に fit し直す。
        """
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        groups = race_groups.reset_index(drop=True)
        unique_groups = pd.unique(groups)

        if len(unique_groups) < 10:
            logger.info(
                "Too few unique races (%d) for meta study, using defaults",
                len(unique_groups),
            )
            return dict(self.PARAM_DEFAULTS["meta"])

        split = self._group_boundary(groups, 0.8)

        mt_preds = oof_preds[:split]
        mt_y = y[:split]
        mv_preds = oof_preds[split:]
        mv_y = y[split:]
        mv_groups = groups.iloc[split:].reset_index(drop=True)
        mv_odds = odds.iloc[split:].reset_index(drop=True) if odds is not None else None
        mv_dates = dates.iloc[split:].reset_index(drop=True) if dates is not None else None

        # 現在の状態を保存
        old_threshold = self.orthogonalize_threshold
        old_strength = self.orthogonalize_strength
        old_ortho = list(self._orthogonalization)

        def _meta_objective(trial: Any) -> float:
            threshold = trial.suggest_float("orthogonalize_threshold", 0.80, 0.99)
            strength = trial.suggest_float("orthogonalize_strength", 0.1, 1.0)
            alpha = trial.suggest_float("ridge_alpha", 0.01, 100.0, log=True)

            # meta-train で直交化 + Ridge を fit
            self.orthogonalize_threshold = threshold
            self.orthogonalize_strength = strength
            self._orthogonalization = []
            mt_transformed = self._fit_prediction_orthogonalizer(mt_preds)

            ridge = Ridge(alpha=alpha)
            ridge.fit(mt_transformed, mt_y)

            # meta-valid で評価
            mv_transformed = self._apply_prediction_orthogonalizer(mv_preds)
            mv_pred = np.clip(ridge.predict(mv_transformed), 0, 1)

            return self._race_top1_objective(
                pd.Series(mv_y),
                mv_pred,
                race_ids=mv_groups,
                odds=mv_odds,
                dates=mv_dates,
            )

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=DEFAULT_RANDOM_SEED),
        )
        study.optimize(_meta_objective, n_trials=self.n_trials)

        # 状態を復元
        self.orthogonalize_threshold = old_threshold
        self.orthogonalize_strength = old_strength
        self._orthogonalization = old_ortho

        return dict(study.best_params)

    # ──────────────────────── reference model training ────────────────────────

    def _train_ref_lgbm(
        self,
        X_t: pd.DataFrame,
        y_t: pd.Series,
        X_v: pd.DataFrame,
        y_v: pd.Series,
        num_threads: int,
        best_params: dict[str, Any],
    ) -> np.ndarray:
        """Train reference LGB model with best params for correlation computation."""
        train_data = lgb.Dataset(X_t, label=y_t)
        valid_data = lgb.Dataset(X_v, label=y_v, reference=train_data)
        m = lgb.train(
            self._build_lgbm_dict(best_params, num_threads),
            train_data,
            num_boost_round=500,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )
        return m.predict(X_v)

    def _train_ref_xgb(
        self,
        X_t: pd.DataFrame,
        y_t: pd.Series,
        X_v: pd.DataFrame,
        y_v: pd.Series,
        num_threads: int,
        best_params: dict[str, Any],
        *,
        dtrain: Any | None = None,
        dvalid: Any | None = None,
        **_kwargs: Any,
    ) -> np.ndarray:
        """Train reference XGB model with best params for correlation computation."""
        import xgboost as xgb

        X_t_num = self._encode_cats(X_t)
        X_v_num = self._encode_cats(X_v)
        if dtrain is None:
            dtrain = xgb.DMatrix(X_t_num, label=y_t)
        if dvalid is None:
            dvalid = xgb.DMatrix(X_v_num, label=y_v)
        m = xgb.train(
            self._build_xgb_dict(best_params, num_threads),
            dtrain,
            num_boost_round=500,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )
        return self._predict_xgb_best(m, dvalid)

    def _train_ref_cat(
        self,
        X_t: pd.DataFrame,
        y_t: pd.Series,
        X_v: pd.DataFrame,
        y_v: pd.Series,
        num_threads: int,
        best_params: dict[str, Any],
        **_kwargs: Any,
    ) -> np.ndarray:
        """Train reference CAT model with best params for correlation computation."""
        from catboost import CatBoostClassifier

        X_t_num = self._encode_cats(X_t)
        X_v_num = self._encode_cats(X_v)
        m = CatBoostClassifier(**self._build_cat_dict(best_params, num_threads))
        m.fit(X_t_num, y_t, eval_set=(X_v_num, y_v))
        return m.predict_proba(X_v_num)[:, 1]

    # ──────────────────────── eval functions ────────────────────────

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
        valid_race_groups: pd.Series | None = None,
        valid_odds: pd.Series | None = None,
        valid_dates: pd.Series | None = None,
        **_kwargs: Any,
    ) -> float:
        """LightGBM Optuna objective"""
        params = suggest_fn(trial)
        train_data = lgb.Dataset(X_t, label=y_t)
        valid_data = lgb.Dataset(X_v, label=y_v, reference=train_data)
        m = lgb.train(
            self._build_lgbm_dict(params, num_threads),
            train_data,
            num_boost_round=500,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )
        preds = m.predict(X_v)

        if valid_race_groups is not None:
            score = self._race_top1_objective(
                y_v,
                preds,
                race_ids=valid_race_groups,
                odds=valid_odds,
                dates=valid_dates,
            )
        else:
            score = self._probability_objective(y_v, preds)

        penalty = self._compute_corr_penalty(
            preds,
            ref_preds_list,
            corr_penalty_weight,
            corr_threshold,
        )
        return score - penalty

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
        dtrain: Any | None = None,
        dvalid: Any | None = None,
        valid_race_groups: pd.Series | None = None,
        valid_odds: pd.Series | None = None,
        valid_dates: pd.Series | None = None,
        **_kwargs: Any,
    ) -> float:
        """XGBoost Optuna objective"""
        import xgboost as xgb

        params = suggest_fn(trial)
        if dtrain is None:
            X_t_num = self._encode_cats(X_t)
            dtrain = xgb.DMatrix(X_t_num, label=y_t)
        if dvalid is None:
            X_v_num = self._encode_cats(X_v)
            dvalid = xgb.DMatrix(X_v_num, label=y_v)
        m = xgb.train(
            self._build_xgb_dict(params, num_threads),
            dtrain,
            num_boost_round=500,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )
        preds = self._predict_xgb_best(m, dvalid)

        if valid_race_groups is not None:
            score = self._race_top1_objective(
                y_v,
                preds,
                race_ids=valid_race_groups,
                odds=valid_odds,
                dates=valid_dates,
            )
        else:
            score = self._probability_objective(y_v, preds)

        penalty = self._compute_corr_penalty(
            preds,
            ref_preds_list,
            corr_penalty_weight,
            corr_threshold,
        )
        return score - penalty

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
        valid_race_groups: pd.Series | None = None,
        valid_odds: pd.Series | None = None,
        valid_dates: pd.Series | None = None,
        **_kwargs: Any,
    ) -> float:
        """CatBoost Optuna objective"""
        from catboost import CatBoostClassifier

        params = suggest_fn(trial)
        X_t_num = self._encode_cats(X_t)
        X_v_num = self._encode_cats(X_v)
        m = CatBoostClassifier(**self._build_cat_dict(params, num_threads))
        m.fit(X_t_num, y_t, eval_set=(X_v_num, y_v))
        preds = m.predict_proba(X_v_num)[:, 1]

        if valid_race_groups is not None:
            score = self._race_top1_objective(
                y_v,
                preds,
                race_ids=valid_race_groups,
                odds=valid_odds,
                dates=valid_dates,
            )
        else:
            score = self._probability_objective(y_v, preds)

        penalty = self._compute_corr_penalty(
            preds,
            ref_preds_list,
            corr_penalty_weight,
            corr_threshold,
        )
        return score - penalty

    # ──────────────────────── LightGBM helpers ────────────────────────

    def _train_lgbm_fold(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        nt: int,
        params: dict[str, Any] | None = None,
        *,
        train_groups: pd.Series | None = None,
    ) -> np.ndarray:
        if params is not None:
            lgb_params = self._build_lgbm_dict(params, nt)
        else:
            lgb_params = {
                **lightgbm_native_params(),
                "objective": "binary",
                "metric": "binary_logloss",
                "learning_rate": 0.03,
                "num_leaves": 31,
                "feature_fraction": 1.0,
                "verbose": -1,
                "num_threads": nt,
            }

        # K-fold train部を80/20に分割 (D-05)
        groups = self._normalize_groups(train_groups, len(X_tr))
        es_split = self._group_boundary(groups, 0.8)
        X_t, y_t = X_tr.iloc[:es_split], y_tr.iloc[:es_split]
        X_v, y_v = X_tr.iloc[es_split:], y_tr.iloc[es_split:]

        train_data = lgb.Dataset(X_t, label=y_t)
        valid_data = lgb.Dataset(X_v, label=y_v, reference=train_data)

        m = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=500,
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
        *,
        train_groups: pd.Series | None = None,
    ) -> lgb.Booster:
        if params is not None:
            lgb_params = self._build_lgbm_dict(params, nt)
        else:
            lgb_params = {
                **lightgbm_native_params(),
                "objective": "binary",
                "metric": "binary_logloss",
                "learning_rate": 0.03,
                "num_leaves": 31,
                "feature_fraction": 1.0,
                "verbose": -1,
                "num_threads": nt,
            }

        groups = self._normalize_groups(train_groups, len(X))
        es_split = self._group_boundary(groups, 0.8)
        X_t, y_t = X.iloc[:es_split], y.iloc[:es_split]
        X_v, y_v = X.iloc[es_split:], y.iloc[es_split:]

        train_data = lgb.Dataset(X_t, label=y_t)
        valid_data = lgb.Dataset(X_v, label=y_v, reference=train_data)

        return lgb.train(
            lgb_params,
            train_data,
            num_boost_round=500,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )

    # ──────────────────────── XGBoost helpers ────────────────────────

    def _train_xgb_fold(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        nt: int,
        params: dict[str, Any] | None = None,
        *,
        train_groups: pd.Series | None = None,
    ) -> np.ndarray:
        import xgboost as xgb

        X_tr_num = self._encode_cats(X_tr)
        X_va_num = self._encode_cats(X_va)

        if params is not None:
            xgb_params = self._build_xgb_dict(params, nt)
        else:
            xgb_params = {
                **xgboost_params(),
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": 6,
                "learning_rate": 0.03,
                "colsample_bytree": 1.0,
                "nthread": nt,
            }

        groups = self._normalize_groups(train_groups, len(X_tr_num))
        es_split = self._group_boundary(groups, 0.8)
        dtrain = xgb.DMatrix(X_tr_num.iloc[:es_split], label=y_tr.iloc[:es_split])
        dvalid = xgb.DMatrix(X_tr_num.iloc[es_split:], y_tr.iloc[es_split:])

        m = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=500,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )
        return self._predict_xgb_best(m, xgb.DMatrix(X_va_num))

    def _train_xgb_full(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        nt: int,
        params: dict[str, Any] | None = None,
        *,
        train_groups: pd.Series | None = None,
    ) -> Any:
        import xgboost as xgb

        X_num = self._encode_cats(X)

        if params is not None:
            xgb_params = self._build_xgb_dict(params, nt)
        else:
            xgb_params = {
                **xgboost_params(),
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": 6,
                "learning_rate": 0.03,
                "colsample_bytree": 1.0,
                "nthread": nt,
            }

        groups = self._normalize_groups(train_groups, len(X_num))
        es_split = self._group_boundary(groups, 0.8)
        dtrain = xgb.DMatrix(X_num.iloc[:es_split], label=y.iloc[:es_split])
        dvalid = xgb.DMatrix(X_num.iloc[es_split:], y.iloc[es_split:])

        return xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=500,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )

    # ──────────────────────── CatBoost helpers ────────────────────────

    def _train_cat_fold(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_va: pd.DataFrame,
        nt: int,
        params: dict[str, Any] | None = None,
        *,
        train_groups: pd.Series | None = None,
    ) -> np.ndarray:
        from catboost import CatBoostClassifier

        X_tr_num = self._encode_cats(X_tr)
        X_va_num = self._encode_cats(X_va)

        if params is not None:
            cat_params = self._build_cat_dict(params, nt)
        else:
            cat_params = {
                **catboost_params(),
                "iterations": 500,
                "learning_rate": 0.03,
                "depth": 6,
                "rsm": 1.0,
                "thread_count": nt,
                "verbose": 0,
                "early_stopping_rounds": 100,
                "eval_metric": "Logloss",
            }

        groups = self._normalize_groups(train_groups, len(X_tr_num))
        es_split = self._group_boundary(groups, 0.8)

        m = CatBoostClassifier(**cat_params)
        m.fit(
            X_tr_num.iloc[:es_split],
            y_tr.iloc[:es_split],
            eval_set=(X_tr_num.iloc[es_split:], y_tr.iloc[es_split:]),
        )
        return m.predict_proba(X_va_num)[:, 1]

    def _train_cat_full(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        nt: int,
        params: dict[str, Any] | None = None,
        *,
        train_groups: pd.Series | None = None,
    ) -> Any:
        from catboost import CatBoostClassifier

        X_num = self._encode_cats(X)

        if params is not None:
            cat_params = self._build_cat_dict(params, nt)
        else:
            cat_params = {
                **catboost_params(),
                "iterations": 500,
                "learning_rate": 0.03,
                "depth": 6,
                "rsm": 1.0,
                "thread_count": nt,
                "verbose": 0,
                "early_stopping_rounds": 100,
                "eval_metric": "Logloss",
            }

        groups = self._normalize_groups(train_groups, len(X_num))
        es_split = self._group_boundary(groups, 0.8)

        m = CatBoostClassifier(**cat_params)
        m.fit(
            X_num.iloc[:es_split],
            y.iloc[:es_split],
            eval_set=(X_num.iloc[es_split:], y.iloc[es_split:]),
        )
        return m

    # ──────────────────────── orthogonalization ────────────────────────

    def _fit_prediction_orthogonalizer(self, preds: np.ndarray) -> np.ndarray:
        """高相関のベース予測をメタ特徴量として直交化する."""
        transformed = preds.astype(float).copy()
        self._orthogonalization = []
        model_names = ["LGB", "XGB", "CAT"]

        for i in range(preds.shape[1]):
            if i == 0:
                self._orthogonalization.append({"enabled": False})
                continue

            raw = preds[:, i].astype(float)
            refs = transformed[:, :i]
            corrs = [abs(self._safe_corr(raw, refs[:, j])) for j in range(refs.shape[1])]
            finite_corrs = [c for c in corrs if np.isfinite(c)]
            max_corr = max(finite_corrs) if finite_corrs else 0.0
            if max_corr < self.orthogonalize_threshold:
                self._orthogonalization.append({"enabled": False})
                continue

            design = np.column_stack([np.ones(len(raw)), refs])
            coef, *_ = np.linalg.lstsq(design, raw, rcond=None)
            resid = raw - design @ coef
            raw_mean = float(np.mean(raw))
            raw_std = float(np.std(raw))
            resid_mean = float(np.mean(resid))
            resid_std = float(np.std(resid))
            if raw_std <= 1e-12 or resid_std <= 1e-12:
                self._orthogonalization.append({"enabled": False})
                continue

            transformed[:, i] = ((resid - resid_mean) / resid_std) * raw_std + raw_mean
            transformed[:, i] = (
                1 - self.orthogonalize_strength
            ) * raw + self.orthogonalize_strength * transformed[:, i]
            self._orthogonalization.append(
                {
                    "enabled": True,
                    "coef": coef.tolist(),
                    "raw_mean": raw_mean,
                    "raw_std": raw_std,
                    "resid_mean": resid_mean,
                    "resid_std": resid_std,
                }
            )
            logger.info(
                "Orthogonalized %s stack feature: max_corr=%.4f >= threshold=%.4f",
                model_names[i] if i < len(model_names) else f"model_{i}",
                max_corr,
                self.orthogonalize_threshold,
            )

        return transformed

    def _apply_prediction_orthogonalizer(self, preds: np.ndarray) -> np.ndarray:
        """学習時に保存した直交化を推論時のLevel-1特徴量へ適用する."""
        orthogonalization = getattr(self, "_orthogonalization", [])
        if not orthogonalization:
            return preds

        transformed = preds.astype(float).copy()
        for i, params in enumerate(orthogonalization):
            if i == 0 or not params.get("enabled", False):
                continue
            coef = np.asarray(params["coef"], dtype=float)
            design = np.column_stack([np.ones(len(preds)), transformed[:, :i]])
            resid = preds[:, i].astype(float) - design @ coef
            transformed[:, i] = (
                (resid - float(params["resid_mean"])) / float(params["resid_std"])
            ) * float(params["raw_std"]) + float(params["raw_mean"])
            raw_col = preds[:, i].astype(float)
            transformed[:, i] = (
                1 - self.orthogonalize_strength
            ) * raw_col + self.orthogonalize_strength * transformed[:, i]
        return transformed

    # ──────────────────────── diversity verification ────────────────────────

    def _compute_importance(self, feature_names: list[str]) -> list[np.ndarray]:
        """各ベースモデルのfeature importanceを抽出"""
        lgb_imp = self.lgbm_model.feature_importance(importance_type="gain")

        xgb_scores = self.xgb_model.get_score(importance_type="gain")
        xgb_imp = np.array([xgb_scores.get(f, 0.0) for f in feature_names], dtype=float)

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

        pairs = [(0, 1, "LGB-XGB"), (0, 2, "LGB-CAT"), (1, 2, "XGB-CAT")]
        for i, j, name in pairs:
            c = self._safe_corr(oof_preds[:, i], oof_preds[:, j])
            if not np.isfinite(c):
                logger.info("OOF prediction correlation %s: skipped (non-finite)", name)
                continue
            logger.info("OOF prediction correlation %s: %.4f", name, c)
            if c >= 0.95:
                logger.warning(
                    "High prediction correlation %s: %.4f >= 0.95 — diversity may be insufficient",
                    name,
                    c,
                )

        for i, j, name in pairs:
            imp_i = importances[i].astype(float)
            imp_j = importances[j].astype(float)
            if float(np.std(imp_i)) <= 1e-12 or float(np.std(imp_j)) <= 1e-12:
                logger.info(
                    "Feature importance rank correlation %s: skipped (constant importance)",
                    name,
                )
                continue
            rho, _ = spearmanr(imp_i, imp_j)
            if not np.isfinite(rho):
                logger.info(
                    "Feature importance rank correlation %s: skipped (non-finite)",
                    name,
                )
                continue
            rho = float(rho)
            logger.info("Feature importance rank correlation %s: %.4f", name, rho)
            if rho > 0.8:
                logger.warning(
                    "High importance correlation %s: %.4f > 0.8 — models rely on similar features",
                    name,
                    rho,
                )
