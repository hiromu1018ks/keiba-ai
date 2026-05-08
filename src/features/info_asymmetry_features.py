"""カテゴリE: 情報非対称性特徴量（履歴ベース）

expanding().shift(1) で未来情報リークを完全遮断 (Rule 18)。
各行は自分より前のデータのみから履歴統計を計算する。

特徴量:
- hist_hit_rate_topk: 同条件で上位K頭の過去的中率
- hist_roi_topk: 同条件で上位K頭の過去ROI
- hist_positive_return_ratio: 正のリターンだったレースの割合
- hist_win_rate_same_condition: 同条件の過去的中率
- hist_market_entropy_avg: 同条件の過去平均エントロピー
"""

from __future__ import annotations

import pandas as pd


def compute_hist_features(df: pd.DataFrame) -> pd.DataFrame:
    """履歴特徴量を expanding().shift(1) でリークフリーに計算

    **重要: レースレベルDataFrameで使用すること (1行=1レース)。**
    馬レベルDataFrameではレース単位の expanding window が正しく動作しない。
    呼び出し元は TrainingPipelineV5._build_race_level_features() (Phase E)。

    Args:
        df: race_date, surface, distance_band, market_entropy,
            topk_hit, topk_roi, positive_return, is_winner を含むDataFrame
            race_date でソート済みであること (1行=1レース)

    Returns:
        hist_hit_rate_topk, hist_roi_topk, hist_positive_return_ratio,
        hist_win_rate_same_condition, hist_market_entropy_avg 列が追加されたDataFrame
    """
    df = df.copy()

    if "race_date" not in df.columns:
        df["hist_hit_rate_topk"] = float("nan")
        df["hist_roi_topk"] = float("nan")
        df["hist_positive_return_ratio"] = float("nan")
        df["hist_win_rate_same_condition"] = float("nan")
        df["hist_market_entropy_avg"] = float("nan")
        return df

    # 全体の expanding 統計 (shift(1) で未来情報を遮断)
    df["hist_hit_rate_topk"] = df["topk_hit"].expanding().mean().shift(1)
    df["hist_roi_topk"] = df["topk_roi"].expanding().mean().shift(1)
    df["hist_positive_return_ratio"] = (
        df["positive_return"].astype(float).expanding().mean().shift(1)
    )

    # 同条件 (surface + distance_band) の expanding 統計
    # groupby + expanding + shift の組み合わせは MultiIndex 上で shift が
    # グループ境界をまたぐため、transform + lambda でグループ内で完結させる
    df["_condition"] = df["surface"] + "_" + df["distance_band"]

    df["hist_win_rate_same_condition"] = (
        df.groupby("_condition", observed=True)["is_winner"].transform(
            lambda s: s.expanding().mean().shift(1)
        )
    )

    df["hist_market_entropy_avg"] = (
        df.groupby("_condition", observed=True)["market_entropy"].transform(
            lambda s: s.expanding().mean().shift(1)
        )
    )

    # 作業列を削除
    df = df.drop(columns=["_condition"])

    return df
