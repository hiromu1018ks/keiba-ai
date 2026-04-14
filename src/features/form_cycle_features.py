"""フォームサイクル特徴量 — 好調/不調トレンド

過去出走の着順データから以下を計算:
- form_trend: 正規化着順の線形回帰傾き (正=好調)
- form_consistency: 正規化着順の標準偏差 (低い=安定)
- form_peak_flag: 直近2走が全体より良い場合 1.0

HorseHistoryFeatures のループ内で呼び出される。
"""

from __future__ import annotations

import numpy as np

FEATURE_COLS: list[str] = [
    "form_trend",
    "form_consistency",
    "form_peak_flag",
]


def compute_form_features(
    kakuteijyuni: np.ndarray, syussotosu: np.ndarray
) -> tuple[float, float, float]:
    """過去出走の着順からフォームサイクル特徴量を計算。

    Args:
        kakuteijyuni: 過去N走の着順 (idx=0 が最新)
        syussotosu:   過去N走の出走頭数

    Returns:
        (form_trend, form_consistency, form_peak_flag)
        データ不足時は全て NaN。
    """
    valid = ~np.isnan(kakuteijyuni) & ~np.isnan(syussotosu) & (syussotosu > 1)
    n = int(valid.sum())

    if n < 2:
        return float("nan"), float("nan"), float("nan")

    fp = kakuteijyuni[valid].astype(float)
    fs = syussotosu[valid].astype(float)

    # 正規化: (pos-1)/(size-1)。低いほど良い [0, 1]
    norm = (fp - 1) / np.maximum(fs - 1, 1)

    # form_trend: 線形回帰の傾きを反転 (正=着順改善=好調)
    x = np.arange(n, dtype=float)
    slope = float(np.polyfit(x, norm, 1)[0])
    form_trend = -slope

    # form_consistency: 正規化着順の標準偏差
    form_consistency = float(np.std(norm))

    # form_peak_flag: 直近2走が全体平均より良い → 1.0
    # 修正: norm[-2:] (最新2走) を使用。norm[:2] は最古2走だった。
    if n >= 3:
        recent_avg = float(norm[-2:].mean())
        overall_avg = float(norm.mean())
        form_peak_flag = 1.0 if recent_avg < overall_avg else 0.0
    else:
        form_peak_flag = float("nan")

    return form_trend, form_consistency, form_peak_flag
