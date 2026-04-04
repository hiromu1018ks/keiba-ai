"""Group E: 交互作用特徴量"""

from __future__ import annotations

import pandas as pd


def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    脚質×距離/馬場 + 体重×距離 の交互作用特徴量を追加。
    LightGBMカテゴリとして扱うため、文字列結合 → astype("category")。
    """
    df = df.copy()

    # 脚質×距離bin (カテゴリ積)
    # LEAK防止: kyakusitukubun_cd (過去) のみ使用。kyakusitukubun (現在=ポスト) は不可。
    if "kyakusitukubun_cd" in df.columns and "distance_bin" in df.columns:
        df["kyakusitu_x_distance"] = (
            df["kyakusitukubun_cd"].astype(str) + "_" + df["distance_bin"].astype(str)
        ).astype("category")

    # 脚質×馬場 (カテゴリ積)
    if "kyakusitukubun_cd" in df.columns and "surface" in df.columns:
        df["kyakusitu_x_surface"] = (
            df["kyakusitukubun_cd"].astype(str) + "_" + df["surface"].astype(str)
        ).astype("category")

    # 馬体重列名の解決 (weight_absolute または ba_taijyu)
    weight_col = "weight_absolute" if "weight_absolute" in df.columns else "bataijyu"

    # 馬体重×距離 (数値積)
    # NaNポリシー: いずれかがNaNなら結果もNaN (fillna(0)は使わない)
    if weight_col in df.columns and "kyori" in df.columns:
        df["weight_x_distance"] = (df[weight_col] * df["kyori"]).where(
            df[weight_col].notna() & df["kyori"].notna(),
            other=float("nan"),
        )

    return df
