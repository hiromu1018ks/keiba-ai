"""カテゴリC: オッズ変化率特徴量

オッズ時系列データから以下を計算 (設計書 §2.3 WinTwoStageModel.FEATURE_COLS):
- odds_drop_rate_60_10: t-60min → t-10min のオッズ変化率
- odds_drop_rate_30_10: t-30min → t-10min のオッズ変化率
- odds_velocity: 単位時間あたりのオッズ変化量（線形回帰の傾き）
- odds_volatility: 連続するオッズ変化量の標準偏差
- popularity_change_30_10: t-30min → t-10min の人気順位変化

計算方式:
- 時系列DataFrameを happyotime でソート後、先頭を t-60min、末尾を t-10min とみなす
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
        odds_ts: race_id, happyotime, umaban, tanodds を含む時系列DataFrame
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

    ts = odds_ts.sort_values(["race_id", "umaban", "happyotime"]).copy()
    grouped = ts.groupby(["race_id", "umaban"])

    # --- 変化率: (early_odds - late_odds) / early_odds ---
    first_odds = grouped["tanodds"].first()
    last_odds = grouped["tanodds"].last()

    # 60→10: 先頭(=t-60) → 末尾(=t-10)
    drop_60_10 = (first_odds - last_odds) / first_odds.replace(0, np.nan)
    drop_60_10.name = "odds_drop_rate_60_10"

    # 30→10: 中間(=t-30) → 末尾(=t-10)
    def _get_mid_odds(group: pd.DataFrame) -> float:
        n = len(group)
        if n < 3:
            return np.nan
        mid_idx = n // 2
        return float(group["tanodds"].iloc[mid_idx])

    mid_odds = grouped.apply(_get_mid_odds, include_groups=False)
    mid_odds.name = "_mid_odds"
    drop_30_10 = (mid_odds - last_odds) / mid_odds.replace(0, np.nan)
    drop_30_10.name = "odds_drop_rate_30_10"

    # --- 速度: 線形回帰の傾き ---
    def _calc_velocity(group: pd.DataFrame) -> float:
        if len(group) < 2:
            return np.nan
        x = np.arange(len(group), dtype=float)
        y = group["tanodds"].values.astype(float)
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    velocity = grouped.apply(_calc_velocity, include_groups=False)
    velocity.name = "odds_velocity"

    # --- ボラティリティ: 連続変化量の標準偏差 ---
    def _calc_volatility(group: pd.DataFrame) -> float:
        if len(group) < 2:
            return np.nan
        changes = group["tanodds"].diff().dropna()
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


def compute_rolling_volatility(
    race_feat_df: pd.DataFrame,
    window: int = 200,
    min_periods: int = 50,
) -> pd.Series:
    """レースレベルの rolling オッズボラティリティを計算

    各レース内の odds_volatility の平均をレースレベル集約とし、
    それを rolling window で平滑化。

    Args:
        race_feat_df: race_id, odds_volatility を含む DataFrame
        window: rolling window size
        min_periods: rolling の最小サンプル数

    Returns:
        odds_volatility_rolling_mean Series
    """
    # odds_volatility 列が無ければ NaN Series を返す
    if "odds_volatility" not in race_feat_df.columns:
        return pd.Series(np.nan, index=race_feat_df.index, name="odds_volatility_rolling_mean")

    # レースごとの odds_volatility 平均
    race_vol = race_feat_df.groupby("race_id")["odds_volatility"].mean()

    # rolling 平均 (時系列順)
    if "race_date" in race_feat_df.columns:
        date_map = race_feat_df.groupby("race_id")["race_date"].first()
        race_vol = race_vol.to_frame("odds_volatility")
        race_vol["race_date"] = date_map
        race_vol = race_vol.sort_values("race_date")
        rolling_mean = (
            race_vol["odds_volatility"].rolling(window=window, min_periods=min_periods).mean()
        )
    else:
        rolling_mean = race_vol.rolling(window=window, min_periods=min_periods).mean()

    # レースごとの rolling 値を元 DataFrame にマップ
    result = race_feat_df["race_id"].map(rolling_mean)
    result.name = "odds_volatility_rolling_mean"
    return result


def compute_roi_ema(
    race_feat_df: pd.DataFrame,
    span: int = 50,
    min_periods: int = 50,
) -> pd.DataFrame:
    """人気層別 ROI EMA を計算

    各レースの人気層 (favorite/mid/longshot) ごとの ROI を
    指数移動平均 (EMA) で平滑化。

    Args:
        race_feat_df: race_id, kakuteijyuni, tanodds, popularity_rank を含む DataFrame
        span: EMA の span
        min_periods: 計算に必要な最小サンプル数

    Returns:
        favorite_roi_ema, mid_roi_ema, longshot_roi_ema 列を追加した DataFrame
    """
    df = race_feat_df.copy()

    # 必須列チェック
    required = {"kakuteijyuni", "tanodds", "popularity_rank", "race_id"}
    if not required.issubset(df.columns):
        df["favorite_roi_ema"] = 0.0
        df["mid_roi_ema"] = 0.0
        df["longshot_roi_ema"] = 0.0
        return df

    # 各馬の ROI (= odds × win) を計算
    df["is_win"] = (df["kakuteijyuni"] == 1).astype(float)
    df["roi"] = df["tanodds"] * df["is_win"]

    # 人気層分類: favorite (1-3), mid (4-8), longshot (9+)
    df["pop_band"] = pd.cut(
        df["popularity_rank"],
        bins=[0, 3, 8, float("inf")],
        labels=["favorite", "mid", "longshot"],
    )

    # レースごと・人気層ごとの平均 ROI
    race_band_roi = (
        df.groupby(["race_id", "pop_band"], observed=False)["roi"].mean().unstack(fill_value=0.0)
    )

    # 時系列ソート
    if "race_date" in df.columns:
        date_map = df.groupby("race_id")["race_date"].first()
        race_band_roi["race_date"] = date_map
        race_band_roi = race_band_roi.sort_values("race_date")
        race_band_roi = race_band_roi.drop(columns=["race_date"])

    # EMA 計算
    ema_cols: dict[str, pd.Series] = {}
    for band in ["favorite", "mid", "longshot"]:
        if band in race_band_roi.columns:
            series = race_band_roi[band]
        else:
            series = pd.Series(0.0, index=race_band_roi.index)
        ema = series.ewm(span=span, min_periods=min_periods).mean()
        ema_cols[f"{band}_roi_ema"] = ema

    # レースごとの EMA 値を元 DataFrame にマップ
    for col_name, ema_series in ema_cols.items():
        df[col_name] = df["race_id"].map(ema_series).fillna(0.0)

    # 作業列を除去
    df = df.drop(columns=["is_win", "roi", "pop_band"], errors="ignore")
    return df
