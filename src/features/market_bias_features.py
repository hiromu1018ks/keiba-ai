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
