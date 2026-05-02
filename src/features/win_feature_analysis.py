"""win_feature_analysis.py — 単勝モデル特徴量重要度分析 (SHAP + gain)

WinTwoStageModel.hit_model の特徴量重要度を SHAP/gain で分析し、
ノイズ特徴量を特定するためのモジュール。

LightGBM 4.6 の pred_contrib=True でネイティブ TreeSHAP 値を取得。
外部 shap パッケージは不要。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb


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
