"""発走N分前オッズスナップショット抽出モジュール.

run_paper_trading / BacktestEngine で共用するオッズ抽出ロジック。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd


def extract_pre_post_odds(
    odds_ts_df: pd.DataFrame,
    race_df: pd.DataFrame,
    minutes_before: int = 5,
    max_staleness_minutes: int = 60,
    *,
    _now: datetime | None = None,
) -> pd.DataFrame:
    """各レースの発走N分前時点のオッズスナップショットを抽出.

    Parameters
    ----------
    odds_ts_df : DataFrame
        時系列オッズ。happyotime (str "MMDDHHmm"), year, umaban 等を含む。
    race_df : DataFrame
        レース情報。hassotime (int "hhmm"), race_id 等を含む。
    minutes_before : int
        発走何分前のオッズを使うか (デフォルト: 5)。
    max_staleness_minutes : int
        cutoff から何分以上前のスナップショットを除外するか (デフォルト: 60)。
    _now : datetime, optional
        現在時刻のオーバーライド (テスト用)。未指定時は datetime.now()。

    Returns
    -------
    DataFrame
        build_all() と互換のスキーマ:
        race_id, umaban, tanodds, fukuoddslow, tanninki
    """
    if odds_ts_df.empty or race_df.empty:
        return pd.DataFrame(columns=["race_id", "umaban", "tanodds", "fukuoddslow", "tanninki"])

    # 1. race_id -> post_datetime のマッピング (vectorized)
    race_info = race_df.drop_duplicates(subset=["race_id"]).set_index("race_id")
    hassotime = race_info["hassotime"]
    valid_ht = hassotime.notna() & (hassotime != 0)
    if not valid_ht.any():
        return pd.DataFrame(columns=["race_id", "umaban", "tanodds", "fukuoddslow", "tanninki"])

    valid_rids = valid_ht[valid_ht].index
    ht_values = hassotime[valid_rids].astype(int).astype(str).str.zfill(4)
    rid_dates = pd.Series(valid_rids, index=valid_rids)

    # YYYYMMDD + HHMM → datetime (vectorized)
    year = rid_dates.str[:4].astype(int)
    month = rid_dates.str[4:6].astype(int)
    day = rid_dates.str[6:8].astype(int)
    hour = ht_values.str[:2].astype(int)
    minute = ht_values.str[2:4].astype(int)

    post_dt_str = (
        year.astype(str) + month.astype(str).str.zfill(2)
        + day.astype(str).str.zfill(2) + hour.astype(str).str.zfill(2)
        + minute.astype(str).str.zfill(2)
    )
    post_time_map = pd.to_datetime(post_dt_str, format="%Y%m%d%H%M")
    post_time_map.index = valid_rids

    if post_time_map.empty:
        return pd.DataFrame(columns=["race_id", "umaban", "tanodds", "fukuoddslow", "tanninki"])

    # 2. happyotime → datetime (vectorized)
    odds_ts_df = odds_ts_df.copy()
    ht_raw = odds_ts_df["happyotime"].astype(str).str.zfill(8)
    year_str = odds_ts_df["year"].astype(int).astype(str)
    datetime_str = year_str + ht_raw.str[:4] + ht_raw.str[4:]
    odds_ts_df["_ht_datetime"] = pd.to_datetime(datetime_str, format="%Y%m%d%H%M", errors="coerce")
    odds_ts_df = odds_ts_df[odds_ts_df["_ht_datetime"].notna()]

    if odds_ts_df.empty:
        return pd.DataFrame(columns=["race_id", "umaban", "tanodds", "fukuoddslow", "tanninki"])

    # 3. cutoff フィルタ (vectorized)
    now = _now or datetime.now()
    now_pd = pd.Timestamp(now)

    # post_datetime を merge で付与
    post_time_series = post_time_map.rename("post_datetime")
    odds_ts_df = odds_ts_df.merge(post_time_series, left_on="race_id", right_index=True, how="inner")

    cutoff = odds_ts_df["post_datetime"] - pd.Timedelta(minutes=minutes_before)
    min_cutoff = cutoff - pd.Timedelta(minutes=max_staleness_minutes)

    # cutoff時刻に達していないレースは除外
    valid = (
        (cutoff <= now_pd)
        & (odds_ts_df["_ht_datetime"] >= min_cutoff)
        & (odds_ts_df["_ht_datetime"] <= cutoff)
    )
    filtered = odds_ts_df[valid]

    if filtered.empty:
        return pd.DataFrame(columns=["race_id", "umaban", "tanodds", "fukuoddslow", "tanninki"])

    # 4. (race_id, umaban) ごとに最新エントリを取得
    idx = filtered.groupby(["race_id", "umaban"])["_ht_datetime"].idxmax()
    snapshot = filtered.loc[idx]

    # 5. build_all() と互換のスキーマで返す
    result = snapshot[["race_id", "umaban", "tanodds", "fukuoddslow", "tanninki"]].copy()
    result = result.reset_index(drop=True)
    return result
