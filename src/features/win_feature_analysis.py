"""win_feature_analysis.py — 単勝モデル特徴量重要度分析 (SHAP + gain)

WinTwoStageModel.hit_model の特徴量重要度を SHAP/gain で分析し、
ノイズ特徴量を特定するためのモジュール。

LightGBM 4.6 の pred_contrib=True でネイティブ TreeSHAP 値を取得。
外部 shap パッケージは不要。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import lightgbm as lgb
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
    assert shap_matrix.shape[1] == len(feature_names) + 1, (
        f"pred_contrib returned {shap_matrix.shape[1]} columns, "
        f"expected {len(feature_names) + 1} (n_features + 1)"
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

    new_model = lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_threads": num_threads,
            "verbose": -1,
            "num_leaves": 31,
        },
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
