"""カテゴリE: レース難易度スコア

difficulty_score = grade_weight × field_factor × (1 - entropy_normalized)

- grade_weight: G1=1.0, G2=0.8, G3=0.6, 重賞(D)=0.4, 特別(E)=0.2, 一般(_)=0.1
- field_factor: field_size / 18 (最大18頭で正規化)
- entropy_normalized: market_entropy / ln(field_size) (0〜1、高いほど拮抗)
"""

from __future__ import annotations

import math

import pandas as pd

_GRADE_WEIGHTS: dict[str, float] = {
    "A": 1.0,   # G1
    "B": 0.8,   # G2
    "C": 0.6,   # G3
    "D": 0.4,   # 重賞
    "E": 0.2,   # 特別
    "_": 0.1,   # 一般
}

_MAX_FIELD_SIZE = 18


def compute_difficulty_score(df: pd.DataFrame) -> pd.DataFrame:
    """レース難易度スコアを計算

    Args:
        df: race_id, field_size, grade_cd, market_entropy を含むDataFrame

    Returns:
        difficulty_score 列が追加されたDataFrame (0.0〜1.0)
    """
    df = df.copy()

    # グレード重み
    df["_grade_weight"] = df["grade_cd"].map(_GRADE_WEIGHTS).fillna(0.1)

    # 頭数係数 (正規化)
    df["_field_factor"] = (df["field_size"] / _MAX_FIELD_SIZE).clip(upper=1.0)

    # エントロピ正規化 (0〜1、高いほど拮抗)
    max_entropy = df["field_size"].apply(lambda n: math.log(n) if n > 1 else 1.0)
    df["_entropy_norm"] = (df["market_entropy"] / max_entropy.replace(0, 1.0)).clip(0, 1)

    # 難易度スコア: 高グレード × 大頭数 × 高拮抗 = 高難易度
    df["difficulty_score"] = (
        df["_grade_weight"] * df["_field_factor"] * df["_entropy_norm"]
    ).clip(0, 1)

    # 作業列を削除
    df = df.drop(columns=["_grade_weight", "_field_factor", "_entropy_norm"])

    return df
