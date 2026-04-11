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

    # 合理的オッズ範囲外を NaN にする (1.0-999.9)
    ts.loc[ts["tanodds"] < 1.0, "tanodds"] = np.nan
    ts.loc[ts["tanodds"] > 999.9, "tanodds"] = np.nan

    # jodds_tanpuku の tanninki を ninki に正規化 (旧time_series互換)
    # Int64 (nullable int) の pd.NA を np.nan に変換 — float(pd.NA) が失敗するため
    if "tanninki" in ts.columns and "ninki" not in ts.columns:
        ts["ninki"] = ts["tanninki"].astype(float)

    # 大量時系列データのメモリ削減: 各(race_id, umaban)につき直近MAX_POINTSのみ保持
    # 特徴量は t-60min → t-10min の変化率等を計算するため、直近60ポイント(≈60分)で十分
    max_points = 60
    if len(ts) > 1_000_000:
        ts = ts.groupby(["race_id", "umaban"], as_index=False).tail(max_points)

    grouped = ts.groupby(["race_id", "umaban"])

    # --- 変化率: (early_odds - late_odds) / early_odds ---
    first_odds = grouped["tanodds"].first()
    last_odds = grouped["tanodds"].last()

    # 60→10: 先頭(=t-60) → 末尾(=t-10)
    drop_60_10 = (first_odds - last_odds) / first_odds.replace(0, np.nan)
    drop_60_10.name = "odds_drop_rate_60_10"

    # --- 中間位置特定の準備 (cumcount + group size) ---
    ts["_pos"] = ts.groupby(["race_id", "umaban"]).cumcount()
    ts["_size"] = ts.groupby(["race_id", "umaban"])["_pos"].transform("max") + 1
    ts["_mid_idx"] = ts["_size"] // 2
    mid_mask = ts["_pos"] == ts["_mid_idx"]

    # 30→10: 中間(=t-30) → 末尾(=t-10)
    # グループサイズ >= 3 の中間行のみ抽出
    mid_ts = ts[mid_mask & (ts["_size"] >= 3)]
    mid_odds = mid_ts.set_index(["race_id", "umaban"])["tanodds"]
    mid_odds.name = "_mid_odds"
    drop_30_10 = (mid_odds - last_odds) / mid_odds.replace(0, np.nan)
    drop_30_10.name = "odds_drop_rate_30_10"

    # --- 速度: 線形回帰の傾き (ベクトル化) ---
    # slope = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x^2)
    ts["_xy"] = ts["_pos"] * ts["tanodds"]
    ts["_x2"] = ts["_pos"] ** 2
    vel_stats = ts.groupby(["race_id", "umaban"]).agg(
        n=("tanodds", "count"),
        sum_x=("_pos", "sum"),
        sum_y=("tanodds", "sum"),
        sum_xy=("_xy", "sum"),
        sum_x2=("_x2", "sum"),
    )
    n = vel_stats["n"]
    denom = n * vel_stats["sum_x2"] - vel_stats["sum_x"] ** 2
    velocity = pd.Series(
        np.where(
            n >= 2,
            (n * vel_stats["sum_xy"] - vel_stats["sum_x"] * vel_stats["sum_y"])
            / denom.replace(0, np.nan),
            np.nan,
        ),
        index=vel_stats.index,
        name="odds_velocity",
    )

    # --- ボラティリティ: 連続変化量の標準偏差 (ベクトル化) ---
    ts["_odds_diff"] = ts.groupby(["race_id", "umaban"])["tanodds"].diff()
    volatility = ts.groupby(["race_id", "umaban"])["_odds_diff"].std()
    # グループサイズ < 2 の場合は NaN (diff が 0 個 → std は NaN)
    sizes = grouped.size()
    volatility[sizes < 2] = np.nan
    volatility.name = "odds_volatility"

    # --- 人気変化: t-30 → t-10 (ベクトル化) ---
    if "ninki" in ts.columns:
        mid_ninki = mid_ts.set_index(["race_id", "umaban"])["ninki"]
        mid_ninki.name = "_mid_ninki"
        pop_change = mid_ninki - grouped["ninki"].last()
        pop_change.name = "popularity_change_30_10"
    else:
        pop_change = pd.Series(np.nan, index=df.index, name="popularity_change_30_10")

    # groupby 結果を1つのDataFrameにまとめてから df にマージ (1回のmerge)
    agg_df = pd.concat(
        [s.reset_index() for s in [drop_60_10, drop_30_10, velocity, volatility, pop_change]],
        axis=1,
    )
    # reset_index() で重複した race_id/umaban 列を削除
    agg_df = agg_df.loc[:, ~agg_df.columns.duplicated()]
    df = df.merge(agg_df, on=["race_id", "umaban"], how="left")

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
    """オッズベース市場指標の EMA を計算 (kakuteijyuni 不使用)"""
    df = race_feat_df.copy()
    required = {"tanodds", "popularity_rank", "race_id"}
    if not required.issubset(df.columns):
        df["favorite_implied_prob_ema"] = 0.0
        df["overround_ema"] = 0.0
        df["entropy_ema"] = 0.0
        return df

    # Overround: sum(1/tanodds) - 1 (レース単位)
    p_raw = 1.0 / df["tanodds"].replace(0, np.nan)
    race_overround = p_raw.groupby(df["race_id"]).sum() - 1.0
    race_overround.name = "overround"

    # Entropy: H = -sum(p_i * ln(p_i)) (レース単位)
    p_norm = p_raw.groupby(df["race_id"]).transform(lambda x: x / x.sum())

    def _entropy(group: pd.Series) -> float:
        p = group.dropna().values.astype(float)
        p = p[p > 0]
        return float(-np.sum(p * np.log(p))) if len(p) > 0 else 0.0

    race_entropy = p_norm.groupby(df["race_id"]).apply(_entropy, include_groups=False)
    race_entropy.name = "entropy"

    # 1番人気の implied probability
    fav_df = df.loc[df["popularity_rank"] == 1, ["race_id", "tanodds"]].copy()
    fav_df["implied_prob"] = 1.0 / fav_df["tanodds"].replace(0, np.nan)
    race_fav_prob = fav_df.groupby("race_id")["implied_prob"].first()
    race_fav_prob.name = "favorite_implied_prob"

    # レース単位 DataFrame (列名を明示的に指定)
    race_stats = pd.DataFrame({
        "favorite_implied_prob": race_fav_prob,
        "overround": race_overround,
        "entropy": race_entropy,
    })

    if "race_date" in df.columns:
        date_map = df.groupby("race_id")["race_date"].first()
        race_stats["_sort"] = date_map
        race_stats = race_stats.sort_values("_sort").drop(columns=["_sort"])

    # EMA (列名で明示的にアクセス)
    for ema_col, src_col in [
        ("favorite_implied_prob_ema", "favorite_implied_prob"),
        ("overround_ema", "overround"),
        ("entropy_ema", "entropy"),
    ]:
        ema = race_stats[src_col].fillna(0.0).ewm(span=span, min_periods=min_periods).mean()
        df[ema_col] = df["race_id"].map(ema).fillna(0.0)

    return df
