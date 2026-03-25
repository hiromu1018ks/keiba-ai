"""カテゴリC: オッズ変化率特徴量

オッズ時系列データから以下を計算 (設計書 §2.3 WinTwoStageModel.FEATURE_COLS):
- odds_drop_rate_60_10: t-60min → t-10min のオッズ変化率
- odds_drop_rate_30_10: t-30min → t-10min のオッズ変化率
- odds_velocity: 単位時間あたりのオッズ変化量（線形回帰の傾き）
- odds_volatility: 連続するオッズ変化量の標準偏差
- popularity_change_30_10: t-30min → t-10min の人気順位変化

計算方式:
- 時系列DataFrameを happyo_time でソート後、先頭を t-60min、末尾を t-10min とみなす
- 中間地点を t-30min として 30-10 変化率を計算
- popularity_change は時系列の ninki 列が必要
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_odds_dynamics(
    df: pd.DataFrame,
    odds_ts: pd.DataFrame | None,
) -> pd.DataFrame:
    """オッズ変化率特徴量を計算

    Args:
        df: race_id, umaban を含むベースDataFrame
        odds_ts: race_id, happyo_time, umaban, tan_odds を含む時系列DataFrame
                 ninki 列がある場合は popularity_change も計算

    Returns:
        odds_drop_rate_60_10, odds_drop_rate_30_10, odds_velocity,
        odds_volatility, popularity_change_30_10 列が追加されたDataFrame
    """
    df = df.copy()

    nan_cols = [
        "odds_drop_rate_60_10",
        "odds_drop_rate_30_10",
        "odds_velocity",
        "odds_volatility",
        "popularity_change_30_10",
    ]

    if odds_ts is None or odds_ts.empty:
        for col in nan_cols:
            df[col] = np.nan
        return df

    ts = odds_ts.sort_values(["race_id", "umaban", "happyo_time"]).copy()
    grouped = ts.groupby(["race_id", "umaban"])

    # --- 変化率: (early_odds - late_odds) / early_odds ---
    first_odds = grouped["tan_odds"].first()
    last_odds = grouped["tan_odds"].last()

    # 60→10: 先頭(=t-60) → 末尾(=t-10)
    drop_60_10 = (first_odds - last_odds) / first_odds.replace(0, np.nan)
    drop_60_10.name = "odds_drop_rate_60_10"

    # 30→10: 中間(=t-30) → 末尾(=t-10)
    def _get_mid_odds(group: pd.DataFrame) -> float:
        n = len(group)
        if n < 3:
            return np.nan
        mid_idx = n // 2
        return float(group["tan_odds"].iloc[mid_idx])

    mid_odds = grouped.apply(_get_mid_odds, include_groups=False)
    mid_odds.name = "_mid_odds"
    drop_30_10 = (mid_odds - last_odds) / mid_odds.replace(0, np.nan)
    drop_30_10.name = "odds_drop_rate_30_10"

    # --- 速度: 線形回帰の傾き ---
    def _calc_velocity(group: pd.DataFrame) -> float:
        if len(group) < 2:
            return np.nan
        x = np.arange(len(group), dtype=float)
        y = group["tan_odds"].values.astype(float)
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    velocity = grouped.apply(_calc_velocity, include_groups=False)
    velocity.name = "odds_velocity"

    # --- ボラティリティ: 連続変化量の標準偏差 ---
    def _calc_volatility(group: pd.DataFrame) -> float:
        if len(group) < 2:
            return np.nan
        changes = group["tan_odds"].diff().dropna()
        return float(changes.std()) if len(changes) > 0 else np.nan

    volatility = grouped.apply(_calc_volatility, include_groups=False)
    volatility.name = "odds_volatility"

    # --- 人気変化: t-30 → t-10 ---
    if "ninki" in ts.columns:

        def _get_mid_ninki(group: pd.DataFrame) -> float:
            n = len(group)
            if n < 3:
                return np.nan
            mid_idx = n // 2
            return float(group["ninki"].iloc[mid_idx])

        mid_ninki = grouped.apply(_get_mid_ninki, include_groups=False)
        mid_ninki.name = "_mid_ninki"
        pop_change = mid_ninki - grouped["ninki"].last()
        pop_change.name = "popularity_change_30_10"
    else:
        pop_change = pd.Series(np.nan, index=df.index, name="popularity_change_30_10")

    # groupby 結果を df にマージ (left join on race_id, umaban)
    merge_cols = [drop_60_10, drop_30_10, velocity, volatility, pop_change]
    for series in merge_cols:
        merged = series.reset_index()
        df = df.merge(merged, on=["race_id", "umaban"], how="left")

    return df
