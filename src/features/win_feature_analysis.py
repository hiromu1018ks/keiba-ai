"""win_feature_analysis.py — 単勝モデル特徴量重要度分析 (SHAP + gain + permutation)

WinTwoStageModel.hit_model の特徴量重要度を SHAP/gain で分析し、
ノイズ特徴量を特定するためのモジュール。

LightGBM 4.6 の pred_contrib=True でネイティブ TreeSHAP 値を取得。
外部 shap パッケージは不要。

permutation_importance (sklearn) による特徴量評価も追加対応。
全モデル(Stage1/Win2Stage/Place2Stage/EVCorrection)の包括的監査を可能にする。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss, roc_auc_score

logger = logging.getLogger(__name__)


def analyze_feature_importance(
    model: lgb.Booster,
    features_df: pd.DataFrame,
    *,
    top_n: int = 0,
) -> pd.DataFrame:
    """SHAP/gainベースの特徴量重要度ランキングを生成。

    Args:
        model: 学習済み lgb.Booster (WinTwoStageModel.hit_model)
        features_df: 特徴量DataFrame (モデル入力と同じ列)
        top_n: 上位n件のみ返す (0=全件)

    Returns:
        DataFrame with columns ['feature', 'gain', 'mean_abs_shap']
        sorted by mean_abs_shap descending
    """
    feature_names = model.feature_name()

    # 1. Gain-based importance (高速、組込)
    gain = model.feature_importance(importance_type="gain")

    # 2. SHAP values via pred_contrib
    # IMPORTANT: shape [n_samples, n_features + 1] -- 最後の列はexpected value (base value)
    shap_matrix = model.predict(features_df, pred_contrib=True)
    shap_cols = shap_matrix.shape[1]
    expected_cols = len(feature_names) + 1
    if shap_cols != expected_cols:
        raise ValueError(
            f"pred_contrib returned {shap_cols} columns, "
            f"expected {expected_cols} (n_features + 1 for base value). "
            f"Model features: {len(feature_names)}"
        )

    # 期待値列を除外
    shap_values = shap_matrix[:, :-1]
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    result = pd.DataFrame({
        "feature": feature_names,
        "gain": gain,
        "mean_abs_shap": mean_abs_shap,
    })
    result = result.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    if top_n > 0:
        result = result.head(top_n).reset_index(drop=True)

    return result


def identify_noise_features(
    importance_df: pd.DataFrame,
    *,
    shap_threshold: float = 0.001,
    gain_threshold: float = 0.0,
) -> list[str]:
    """SHAP/gain寄与が閾値未満のノイズ特徴量を特定。

    ノイズ条件: mean_abs_shap < shap_threshold AND gain <= gain_threshold

    Args:
        importance_df: analyze_feature_importance の戻り値
        shap_threshold: SHAP寄与の閾値 (デフォルト 0.001)
        gain_threshold: gain寄与の閾値 (デフォルト 0.0)

    Returns:
        ノイズ特徴量名のリスト
    """
    noise_mask = (importance_df["mean_abs_shap"] < shap_threshold) & (
        importance_df["gain"] <= gain_threshold
    )
    return importance_df.loc[noise_mask, "feature"].tolist()


def compute_permutation_importance(
    model: lgb.Booster,
    features_df: pd.DataFrame,
    y: np.ndarray,
    *,
    n_repeats: int = 5,
    random_state: int = 42,
    max_samples: int = 5000,
    scoring: str = "auto",
) -> pd.DataFrame:
    """sklearn permutation_importanceベースの特徴量重要度計算。

    Args:
        model: 学習済み lgb.Booster
        features_df: 特徴量DataFrame (モデル入力と同じ列)
        y: ターゲット配列
        n_repeats: permutation試行回数
        random_state: 乱数シード
        max_samples: サブサンプリング上限 (超過時はランダム抽出)
        scoring: 評価指標 ("auto"でbinary/regression自動判定,
                  "neg_log_loss", "neg_mean_absolute_error"等)

    Returns:
        DataFrame with columns ['feature', 'perm_importance_mean', 'perm_importance_std']
    """
    feature_names = model.feature_name()

    # scoring自動判定: yが{0,1}のみ→binary, それ以外→regression
    if scoring == "auto":
        unique_vals = np.unique(y[~np.isnan(y)])
        if set(unique_vals.tolist()).issubset({0.0, 1.0}):
            scoring_str = "neg_log_loss"
        else:
            scoring_str = "neg_mean_absolute_error"
    else:
        scoring_str = scoring

    # sklearn互換のpredict function wrapper
    def predict_fn(x_data: np.ndarray | pd.DataFrame) -> np.ndarray:
        result: Any = model.predict(x_data)
        return np.asarray(result)

    # サブサンプリング
    rng = np.random.default_rng(random_state)
    if len(features_df) > max_samples:
        sample_idx = rng.choice(len(features_df), size=max_samples, replace=False)
        x_sample = features_df.iloc[sample_idx].reset_index(drop=True)
        y_sample = y[sample_idx]
    else:
        x_sample = features_df.reset_index(drop=True)
        y_sample = y

    result = permutation_importance(
        predict_fn,
        x_sample,
        y_sample,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring_str,
    )

    return pd.DataFrame({
        "feature": feature_names,
        "perm_importance_mean": result.importances_mean,
        "perm_importance_std": result.importances_std,
    })


def compute_all_model_importance(
    models: dict[str, lgb.Booster],
    features_df: pd.DataFrame,
    targets: dict[str, np.ndarray],
    *,
    n_repeats: int = 5,
    random_state: int = 42,
    max_samples: int = 5000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """全モデルのgain + permutation重要度を一括計算。

    Args:
        models: モデル名→Boosterのdict (例: {"stage1": booster, "win_hit": booster, ...})
        features_df: 特徴量DataFrame (全特徴量を含む)
        targets: モデル名→ターゲット配列のdict
        n_repeats: permutation試行回数
        random_state: 乱数シード
        max_samples: サブサンプリング上限

    Returns:
        (pivot_df, metadata_dict)
        pivot_df: CSV出力用。columns = ["feature", "<model>_gain", "<model>_perm", ...]
        metadata_dict: JSON出力用。モデル別gain/permutation重要度とメタデータ
    """
    all_importances: dict[str, pd.DataFrame] = {}

    for model_name, model in models.items():
        feature_names = model.feature_name()

        # gain importance
        gain = model.feature_importance(importance_type="gain")

        # permutation importance
        model_features = features_df[feature_names] if all(
            c in features_df.columns for c in feature_names
        ) else features_df[[c for c in feature_names if c in features_df.columns]]

        y = targets.get(model_name)
        if y is not None and len(y) == len(model_features):
            perm_df = compute_permutation_importance(
                model,
                model_features,
                y,
                n_repeats=n_repeats,
                random_state=random_state,
                max_samples=max_samples,
            )
            perm_mean_dict = dict(zip(perm_df["feature"], perm_df["perm_importance_mean"]))
            perm_std_dict = dict(zip(perm_df["feature"], perm_df["perm_importance_std"]))
        else:
            # ターゲットが利用できない場合はpermutationをスキップ
            perm_mean_dict = {f: float("nan") for f in feature_names}
            perm_std_dict = {f: float("nan") for f in feature_names}
            logger.warning(
                "モデル '%s' のターゲットが利用できないためpermutation importanceをスキップ",
                model_name,
            )

        all_importances[model_name] = pd.DataFrame({
            "feature": feature_names,
            "gain": gain,
            "perm_mean": [perm_mean_dict.get(f, float("nan")) for f in feature_names],
            "perm_std": [perm_std_dict.get(f, float("nan")) for f in feature_names],
        })

    # pivot_dfの構築: 全特徴量の和集合
    all_features: set[str] = set()
    for imp_df in all_importances.values():
        all_features.update(imp_df["feature"].tolist())
    sorted_features = sorted(all_features)

    pivot_data: dict[str, list[Any]] = {"feature": sorted_features}
    for model_name, imp_df in all_importances.items():
        gain_map = dict(zip(imp_df["feature"], imp_df["gain"]))
        perm_map = dict(zip(imp_df["feature"], imp_df["perm_mean"]))
        pivot_data[f"{model_name}_gain"] = [gain_map.get(f, float("nan")) for f in sorted_features]
        pivot_data[f"{model_name}_perm"] = [perm_map.get(f, float("nan")) for f in sorted_features]

    pivot_df = pd.DataFrame(pivot_data)

    # metadata_dictの構築
    metadata: dict[str, Any] = {
        "models": {},
        "metadata": {
            "n_samples": len(features_df),
            "n_repeats": n_repeats,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
    }
    for model_name, imp_df in all_importances.items():
        gain_series = imp_df.set_index("feature")["gain"]
        perm_mean_series = imp_df.set_index("feature")["perm_mean"]
        perm_std_series = imp_df.set_index("feature")["perm_std"]
        metadata["models"][model_name] = {
            "gain": {k: float(v) for k, v in gain_series.items()},
            "perm_mean": {k: float(v) for k, v in perm_mean_series.items()},
            "perm_std": {k: float(v) for k, v in perm_std_series.items()},
        }

    return pivot_df, metadata


def validate_noise_removal(
    original_model: lgb.Booster,
    df: pd.DataFrame,
    noise_features: list[str],
    target_col: str = "kakuteijyuni",
    num_threads: int = 0,
) -> dict[str, float]:
    """ノイズ特徴量除外前後のlogloss/AUCを比較。

    時系列順にtrain/valid (80/20) に分割し、
    両モデルのメトリクスをvalidデータで評価する。
    同一データでの学習・評価によるバイアスを防ぐ。

    Args:
        original_model: 元の学習済み lgb.Booster
        df: 特徴量 + ターゲット列を含むDataFrame (race_date順にソート済みであること)
        noise_features: 除外する特徴量名のリスト
        target_col: ターゲット列名 (default: kakuteijyuni)
        num_threads: LightGBM スレッド数

    Returns:
        dict with keys: original_logloss, new_logloss, original_auc, new_auc
    """
    feature_names = original_model.feature_name()
    remaining_features = [f for f in feature_names if f not in noise_features]

    # ターゲット (1着 = 1, それ以外 = 0)
    y = (df[target_col] == 1).astype(int).values

    # 時系列順にtrain/valid (80/20) に分割 -- look-ahead bias防止
    n = len(df)
    split = int(n * 0.8)

    # 元モデルの予測 (validデータのみで評価)
    orig_features = df[feature_names]
    orig_pred = original_model.predict(orig_features)
    orig_pred_valid = orig_pred[split:]
    y_valid = y[split:]

    # logloss/AUC の計算 (NaNを含む場合はフィルタ)
    valid_mask = ~(np.isnan(orig_pred_valid) | np.isnan(y_valid.astype(float)))
    if valid_mask.sum() < 2:
        logger.warning("Too few valid predictions for comparison")
        return {
            "original_logloss": float("nan"),
            "new_logloss": float("nan"),
            "original_auc": float("nan"),
            "new_auc": float("nan"),
        }

    original_logloss = float(log_loss(y_valid[valid_mask], orig_pred_valid[valid_mask]))
    original_auc = float(roc_auc_score(y_valid[valid_mask], orig_pred_valid[valid_mask]))

    # 新モデルをノイズ除外特徴量で学習 (trainデータのみ)
    new_features_df = df[remaining_features]
    train_features = new_features_df.iloc[:split]
    train_y = y[:split]
    valid_features = new_features_df.iloc[split:]
    valid_y = y[split:]

    train_data = lgb.Dataset(train_features, label=train_y)

    # 元モデルのハイパーパラメータを継承 (比較の公平性確保)
    base_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_threads": num_threads,
        "verbose": -1,
    }
    if hasattr(original_model, "params") and isinstance(original_model.params, dict):
        # 元モデルから主要ハイパーパラメータを抽出
        for key in ("num_leaves", "learning_rate", "min_child_samples",
                     "feature_fraction", "bagging_fraction", "lambda_l1", "lambda_l2",
                     "max_depth", "min_data_in_leaf"):
            if key in original_model.params:
                base_params[key] = original_model.params[key]

    new_model = lgb.train(
        base_params,
        train_data,
        num_boost_round=100,
    )

    new_pred_valid = new_model.predict(valid_features)
    valid_mask_new = ~(np.isnan(new_pred_valid) | np.isnan(valid_y.astype(float)))

    new_logloss = float(log_loss(valid_y[valid_mask_new], new_pred_valid[valid_mask_new]))
    new_auc = float(roc_auc_score(valid_y[valid_mask_new], new_pred_valid[valid_mask_new]))

    # logloss悪化が0.5%超の場合に警告
    if original_logloss > 0 and (new_logloss - original_logloss) / original_logloss > 0.005:
        logger.warning(
            "Noise removal degraded logloss by %.2f%%: %.6f -> %.6f",
            (new_logloss - original_logloss) / original_logloss * 100,
            original_logloss,
            new_logloss,
        )

    return {
        "original_logloss": original_logloss,
        "new_logloss": new_logloss,
        "original_auc": original_auc,
        "new_auc": new_auc,
    }
