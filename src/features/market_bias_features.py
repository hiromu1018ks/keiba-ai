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
        df: race_id, umaban, tanodds を含むDataFrame

    Returns:
        p_market_win_adj, market_entropy, overround 列が追加されたDataFrame
    """
    df = df.copy()

    if "tanodds" not in df.columns:
        df["p_market_win_adj"] = np.nan
        df["market_entropy"] = np.nan
        df["overround"] = np.nan
        return df

    # 生の含み確率
    p_raw = 1.0 / df["tanodds"].replace(0, np.nan)

    # Overround: 胴元控除率
    # JRAプール方式では払戻率約70-80%のため、sum(1/tanodds) < 1.0 が正常。
    # overround = sum(1/tanodds) - 1.0 ≈ -0.20 ~ -0.30 が典型的。
    # 極端な負値(min=-0.476)は長距離・少頭数レース等で自然に発生。
    # 固定オッズ方式(英国)の正値overroundとは意味が異なる点に注意。
    overround = p_raw.groupby(df["race_id"], observed=True).transform("sum") - 1.0
    df["overround"] = overround

    # 正規化確率 (Σ=1)
    p_sum = p_raw.groupby(df["race_id"], observed=True).transform("sum")
    df["p_market_win_adj"] = p_raw / p_sum.replace(0, np.nan)

    # シャノンエントロピー: H = -Σ(p_i * ln(p_i))
    def _calc_entropy(group: pd.Series) -> float:
        p = group.values.astype(float)
        p = p[p > 0]  # log(0) を回避
        if len(p) == 0:
            return 0.0
        return float(-np.sum(p * np.log(p)))

    entropy = df.groupby("race_id", observed=True)["p_market_win_adj"].transform(_calc_entropy)
    df["market_entropy"] = entropy

    return df


def compute_flb_slope(race_feat_df: pd.DataFrame) -> pd.DataFrame:
    """オッズ分布の形状指標をレースごとに計算 (kakuteijyuni 不使用)

    Returns:
        odds_skewness, implied_prob_hhi 列を持つ DataFrame
    """
    result = pd.DataFrame(index=race_feat_df.index)
    if "tanodds" not in race_feat_df.columns:
        result["odds_skewness"] = 0.0
        result["implied_prob_hhi"] = 0.0
        return result

    def _race_shape(group):
        if len(group) < 2:
            return 0.0, 0.0
        odds = group["tanodds"].replace(0, np.nan).dropna().values.astype(float)
        if len(odds) < 2:
            return 0.0, 0.0
        skewness = float(pd.Series(odds).skew()) or 0.0
        inv_odds = 1.0 / odds
        total = inv_odds.sum()
        if total == 0:
            return skewness, 0.0
        p = inv_odds / total
        hhi = float(np.sum(p ** 2))
        return skewness, hhi

    shapes = race_feat_df.groupby("race_id", observed=True).apply(_race_shape, include_groups=False)
    result["odds_skewness"] = race_feat_df["race_id"].map(shapes.map(lambda x: x[0])).fillna(0.0)
    result["implied_prob_hhi"] = race_feat_df["race_id"].map(shapes.map(lambda x: x[1])).fillna(0.0)
    return result
