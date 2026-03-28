"""カテゴリD: 市場歪み特徴量

市場の効率性・歪み度合いを表す特徴量を計算:
- p_market_win_adj: 正規化市場確率 (Σ=1)
- market_entropy: シャノンエントロピー (拮抗度の指標)
- overround: 胴元控除率 (Σ(p_raw) - 1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_market_bias(df: pd.DataFrame) -> pd.DataFrame:
    """市場歪み特徴量を計算

    Args:
        df: race_id, umaban, tan_odds を含むDataFrame

    Returns:
        p_market_win_adj, market_entropy, overround 列が追加されたDataFrame
    """
    df = df.copy()

    if "tan_odds" not in df.columns:
        df["p_market_win_adj"] = np.nan
        df["market_entropy"] = np.nan
        df["overround"] = np.nan
        return df

    # 生の含み確率
    p_raw = 1.0 / df["tan_odds"].replace(0, np.nan)

    # Overround: 胴元控除率 (正=控除あり, 負=非現実的)
    overround = p_raw.groupby(df["race_id"]).transform("sum") - 1.0
    df["overround"] = overround

    # 正規化確率 (Σ=1)
    p_sum = p_raw.groupby(df["race_id"]).transform("sum")
    df["p_market_win_adj"] = p_raw / p_sum.replace(0, np.nan)

    # シャノンエントロピー: H = -Σ(p_i * ln(p_i))
    def _calc_entropy(group: pd.Series) -> float:
        p = group.values.astype(float)
        p = p[p > 0]  # log(0) を回避
        if len(p) == 0:
            return 0.0
        return float(-np.sum(p * np.log(p)))

    entropy = df.groupby("race_id")["p_market_win_adj"].transform(_calc_entropy)
    df["market_entropy"] = entropy

    return df


def compute_flb_slope(race_feat_df: pd.DataFrame) -> pd.Series:
    """Favorite-Longshot Bias の傾きをレースごとに計算

    log(odds) → 実際勝率 の回帰傾きを算出。
    傾きが大きい (=1に近い) ほど市場は効率的。
    傾きが小さいほど FLB が強い (=人気馬が割安)。

    Args:
        race_feat_df: race_id, tan_odds, finish_pos, field_size を含む
                      レース集計 DataFrame

    Returns:
        flb_slope Series (race_id ごとに1値)
    """
    if "tan_odds" not in race_feat_df.columns or "finish_pos" not in race_feat_df.columns:
        return pd.Series(0.0, index=race_feat_df.index, name="flb_slope")

    # 馬単位からレース単位で FLB slope を計算
    def _race_flb(group: pd.DataFrame) -> float:
        if len(group) < 3:
            return 0.0
        log_odds = np.log(group["tan_odds"].replace(0, np.nan).values.astype(float))
        # 実際勝率: 1着=1, それ以外=0 (単純化)
        win = (group["finish_pos"] == 1).astype(float).values
        valid = ~np.isnan(log_odds)
        if valid.sum() < 3:
            return 0.0
        # log(odds) でソートして累積勝率を計算 → 傾き
        order = np.argsort(log_odds[valid])
        sorted_log_odds = log_odds[valid][order]
        sorted_win = win[valid][order]
        if len(sorted_log_odds) < 3:
            return 0.0
        slope = float(np.polyfit(sorted_log_odds, sorted_win, 1)[0])
        return float(slope)

    slopes = race_feat_df.groupby("race_id").apply(_race_flb, include_groups=False)
    slopes.name = "flb_slope"

    # 元の DataFrame にマップ
    result = race_feat_df["race_id"].map(slopes)
    result.name = "flb_slope"
    return result.fillna(0.0)
