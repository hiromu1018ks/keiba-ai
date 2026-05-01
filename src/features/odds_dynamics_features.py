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


def _build_snapshot_datetimes(ts: pd.DataFrame) -> pd.Series:
    if "race_date" in ts.columns:
        race_date = pd.to_datetime(ts["race_date"], errors="coerce")
    else:
        race_date = pd.Series(pd.NaT, index=ts.index)
    if race_date.notna().any():
        year_str = (
            race_date.dt.year.astype("Int64").astype(str).str.replace("<NA>", "", regex=False)
        )
    else:
        year_from_race = ts["race_id"].astype(str).str[:4]
        year_str = np.where(year_from_race.str.fullmatch(r"\d{4}"), year_from_race, "2000")
        year_str = pd.Series(year_str, index=ts.index)

    ht_raw = ts["happyotime"].astype(str).str.zfill(8)
    parsed = pd.to_datetime(year_str + ht_raw, format="%Y%m%d%H%M", errors="coerce")
    if parsed.notna().any():
        return parsed

    order = ts.groupby(["race_id", "umaban"]).cumcount()
    base = pd.Timestamp("2000-01-01")
    return pd.Series(base + pd.to_timedelta(order, unit="m"), index=ts.index)


def _build_post_time_map(df: pd.DataFrame) -> pd.Series:
    if "race_id" not in df.columns or "hassotime" not in df.columns:
        return pd.Series(dtype="datetime64[ns]")

    race_info = df[["race_id", "hassotime"]].drop_duplicates(subset=["race_id"]).copy()
    if "race_date" in df.columns:
        race_date_map = (
            df[["race_id", "race_date"]]
            .drop_duplicates(subset=["race_id"])
            .set_index("race_id")["race_date"]
        )
        race_date = pd.to_datetime(
            race_date_map,
            errors="coerce",
        )
        race_info["race_date"] = race_info["race_id"].map(race_date)
    else:
        race_info["race_date"] = pd.to_datetime(
            race_info["race_id"].astype(str).str[:8],
            format="%Y%m%d",
            errors="coerce",
        )

    valid = (
        race_info["race_date"].notna()
        & race_info["hassotime"].notna()
        & (race_info["hassotime"] != 0)
    )
    if not valid.any():
        return pd.Series(dtype="datetime64[ns]")

    ht = race_info.loc[valid, "hassotime"].astype(int).astype(str).str.zfill(4)
    date_str = race_info.loc[valid, "race_date"].dt.strftime("%Y%m%d")
    post_time = pd.to_datetime(date_str + ht, format="%Y%m%d%H%M", errors="coerce")
    post_time.index = race_info.loc[valid, "race_id"]
    return post_time


def _pick_target_snapshot(
    ts: pd.DataFrame,
    value_col: str,
    *,
    target_minutes: float,
    tolerance_minutes: float,
) -> pd.Series:
    if value_col not in ts.columns:
        return pd.Series(dtype=float)

    diff = (ts["_mins_before_anchor"] - target_minutes).abs()
    valid = diff.notna() & ts[value_col].notna()
    if not valid.any():
        return pd.Series(dtype=float)

    subset = ts.loc[valid, ["race_id", "umaban", "_ts_datetime", value_col]].copy()
    subset["_diff"] = diff.loc[valid].values
    preferred = subset["_diff"] <= tolerance_minutes
    if preferred.any():
        subset = subset.loc[preferred]
    subset = subset.sort_values(
        ["race_id", "umaban", "_diff", "_ts_datetime"],
        ascending=[True, True, True, False],
    )
    picked = subset.drop_duplicates(subset=["race_id", "umaban"], keep="first")
    return picked.set_index(["race_id", "umaban"])[value_col]


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

    # 各(race_id, umaban)につき直近MAX_POINTSのみ保持 (PT/BT 一致のため無条件)
    max_points = 60
    ts = ts.groupby(["race_id", "umaban"], as_index=False).tail(max_points)
    ts["_ts_datetime"] = _build_snapshot_datetimes(ts)

    grouped = ts.groupby(["race_id", "umaban"])

    post_time_map = _build_post_time_map(df)
    if not post_time_map.empty:
        ts["post_datetime"] = ts["race_id"].map(post_time_map)
    else:
        ts["post_datetime"] = pd.NaT
    fallback_post = grouped["_ts_datetime"].transform("max") + pd.Timedelta(minutes=10)
    ts["post_datetime"] = ts["post_datetime"].fillna(fallback_post)
    ts["_mins_before_anchor"] = (
        (ts["post_datetime"] - ts["_ts_datetime"]) / pd.Timedelta(minutes=1)
    ).astype(float)

    odds_10 = _pick_target_snapshot(ts, "tanodds", target_minutes=10.0, tolerance_minutes=15.0)
    odds_30 = _pick_target_snapshot(ts, "tanodds", target_minutes=30.0, tolerance_minutes=15.0)
    odds_60 = _pick_target_snapshot(ts, "tanodds", target_minutes=60.0, tolerance_minutes=20.0)
    base_index = pd.MultiIndex.from_frame(df[["race_id", "umaban"]])
    odds_10 = odds_10.reindex(base_index)
    odds_30 = odds_30.reindex(base_index)
    odds_60 = odds_60.reindex(base_index)

    # --- 変化率: (early_odds - late_odds) / early_odds ---
    drop_60_10 = (odds_60 - odds_10) / odds_60.replace(0, np.nan)
    drop_60_10.name = "odds_drop_rate_60_10"

    drop_30_10 = (odds_30 - odds_10) / odds_30.replace(0, np.nan)
    drop_30_10.name = "odds_drop_rate_30_10"

    # --- 速度: 線形回帰の傾き (ベクトル化) ---
    # slope = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x^2)
    vel_ts = ts[ts["tanodds"].notna()].copy()
    first_time = vel_ts.groupby(["race_id", "umaban"])["_ts_datetime"].transform("min")
    vel_ts["_elapsed_minutes"] = (
        (vel_ts["_ts_datetime"] - first_time) / pd.Timedelta(minutes=1)
    ).astype(float)
    vel_ts["_xy"] = vel_ts["_elapsed_minutes"] * vel_ts["tanodds"]
    vel_ts["_x2"] = vel_ts["_elapsed_minutes"] ** 2
    vel_stats = vel_ts.groupby(["race_id", "umaban"]).agg(
        n=("tanodds", "count"),
        sum_x=("_elapsed_minutes", "sum"),
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
        ninki_10 = _pick_target_snapshot(ts, "ninki", target_minutes=10.0, tolerance_minutes=15.0)
        ninki_30 = _pick_target_snapshot(ts, "ninki", target_minutes=30.0, tolerance_minutes=15.0)
        ninki_10 = ninki_10.reindex(base_index)
        ninki_30 = ninki_30.reindex(base_index)
        pop_change = ninki_30 - ninki_10
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
    race_stats = pd.DataFrame(
        {
            "favorite_implied_prob": race_fav_prob,
            "overround": race_overround,
            "entropy": race_entropy,
        }
    )

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
