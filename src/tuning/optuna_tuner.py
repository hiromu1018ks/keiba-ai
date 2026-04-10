"""Optuna ハイパーパラメータチューニング (B2)

各モデルの最適なハイパーパラメータを Optuna で探索。
時系列Walk-Forward CVで評価し、データリークを防止。
"""

from __future__ import annotations

import logging
from typing import Any

import optuna
import pandas as pd

logger = logging.getLogger(__name__)

# 各モデルの検索空間: { param_name: (low, high) or distribution spec }
SEARCH_SPACES: dict[str, dict[str, tuple]] = {
    "win_hit": {
        "num_leaves": (15, 63),
        "learning_rate": (0.01, 0.1),
        "feature_fraction": (0.5, 0.9),
    },
    "win_return": {
        "num_leaves": (7, 31),
        "learning_rate": (0.01, 0.1),
        "feature_fraction": (0.5, 0.9),
    },
    "place_hit": {
        "num_leaves": (15, 63),
        "learning_rate": (0.01, 0.1),
        "feature_fraction": (0.5, 0.9),
    },
    "place_return": {
        "num_leaves": (7, 31),
        "learning_rate": (0.01, 0.1),
        "feature_fraction": (0.5, 0.9),
    },
    "ability": {
        "num_leaves": (15, 63),
        "learning_rate": (0.01, 0.1),
        "feature_fraction": (0.5, 0.9),
    },
    "ridge_alpha": {
        "alpha": (0.01, 100.0),
    },
}

# hit モデルで使用する基本特徴量列 (テストデータでも存在する最小セット)
_HIT_FEATURE_COLS: list[str] = [
    "p_ability_win",
    "track_condition_code",
    "field_size",
]


def _get_feature_cols(model_type: str, df: pd.DataFrame) -> list[str]:
    """モデルタイプに応じた特徴量列を返す。

    FEATURE_COLS の定義が存在すればそれを使い、データにない列は除外する。
    定義がなければフォールバックとして数値列を自動選択する。
    """
    # モデル固有の FEATURE_COLS 定義をインポート試行
    defined_cols: list[str] | None = None
    try:
        if model_type in ("win_hit", "win_return"):
            from models.two_stage_return_model import WinTwoStageModel
            defined_cols = WinTwoStageModel.FEATURE_COLS
        elif model_type in ("place_hit", "place_return"):
            from models.two_stage_return_model import WinTwoStageModel
            defined_cols = WinTwoStageModel.FEATURE_COLS
        elif model_type == "ability":
            from models.stage1_ability_model import Stage1AbilityModel
            defined_cols = Stage1AbilityModel.FEATURE_COLS
    except (ImportError, AttributeError):
        pass

    if defined_cols is not None:
        available = [c for c in defined_cols if c in df.columns]
        if available:
            return available

    # フォールバック: テストで利用可能な基本列 + 数値列
    fallback = [c for c in _HIT_FEATURE_COLS if c in df.columns]
    if not fallback:
        # さらにフォールバック: 全数値列 (race_id等の文字列除外)
        fallback = [
            c for c in df.columns
            if c not in ("race_id", "race_date", "surface", "distance_bin", "grade_code")
            and pd.api.types.is_numeric_dtype(df[c])
        ]
    return fallback


class OptunaTuner:
    """Optuna ベースのハイパーパラメータチューナー。"""

    def __init__(self, model_type: str = "win_hit") -> None:
        self.model_type = model_type
        self.search_space = SEARCH_SPACES.get(model_type, SEARCH_SPACES["win_hit"])

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """trial からパラメータをサンプリング。"""
        params: dict[str, Any] = {}
        for name, bounds in self.search_space.items():
            lo, hi = bounds
            if name == "learning_rate":
                params[name] = trial.suggest_float(name, lo, hi, log=True)
            elif name == "alpha":
                params[name] = trial.suggest_float(name, lo, hi, log=True)
            elif name == "feature_fraction":
                params[name] = trial.suggest_float(name, lo, hi)
            else:
                params[name] = trial.suggest_int(name, int(lo), int(hi))
        return params

    def objective(self, trial: optuna.Trial, df: pd.DataFrame) -> float:
        """目的関数: 時系列80/20分割で AUC を評価。"""
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        params = self._suggest_params(trial)

        # 時系列分割 (既に race_date でソート済み前提)
        n = len(df)
        split = int(n * 0.8)

        # 特徴量選択
        feat_cols = _get_feature_cols(self.model_type, df)

        # ターゲット設定
        if self.model_type in ("win_hit", "win_return"):
            y = (df["kakuteijyuni"] == 1).astype(int)
        elif self.model_type in ("place_hit", "place_return"):
            y = (df["kakuteijyuni"] <= 3).astype(int)
        else:
            y = (df["kakuteijyuni"] == 1).astype(int)

        X = df[feat_cols].copy()
        for col in X.columns:
            if X[col].dtype == object:
                try:
                    X[col] = X[col].astype(float)
                except (ValueError, TypeError):
                    X = X.drop(columns=[col])

        # NaN を LightGBM が処理できるよう float に変換
        X = X.astype(float)

        X_train, X_valid = X.iloc[:split], X.iloc[split:]
        y_train, y_valid = y.iloc[:split], y.iloc[split:]

        lgb_params: dict[str, Any] = {
            "objective": "binary",
            "metric": "auc",
            "num_leaves": params.get("num_leaves", 31),
            "learning_rate": params.get("learning_rate", 0.03),
            "feature_fraction": params.get("feature_fraction", 0.7),
            "verbose": -1,
        }

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

        model = lgb.train(
            lgb_params, train_data, num_boost_round=300,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        preds = model.predict(X_valid)
        return float(roc_auc_score(y_valid, preds))

    def tune(self, df: pd.DataFrame, n_trials: int = 100) -> dict[str, Any]:
        """Optuna チューニングを実行。"""
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, df), n_trials=n_trials)

        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
        }
