"""発走N分前オッズスナップショット抽出モジュール.

run_paper_trading / BacktestEngine / TrainingPipeline で共用するオッズ抽出ロジック。
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

_OUTPUT_COLS = ["race_id", "umaban", "tanodds", "fukuoddslow", "tanninki"]


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
        return pd.DataFrame(columns=_OUTPUT_COLS)

    # 列名正規化: ninki / tanninki 両対応
    ninki_col: str | None = None
    if "tanninki" in odds_ts_df.columns:
        ninki_col = "tanninki"
    elif "ninki" in odds_ts_df.columns:
        ninki_col = "ninki"

    required = ["race_id", "umaban", "tanodds", "fukuoddslow", ninki_col, "year", "happyotime"]
    if ninki_col is None or any(c not in odds_ts_df.columns for c in required):
        return pd.DataFrame(columns=_OUTPUT_COLS)

    # 1. race_id -> post_datetime のマッピング (vectorized)
    race_info = race_df.drop_duplicates(subset=["race_id"]).set_index("race_id")
    hassotime = race_info["hassotime"]
    valid_ht = hassotime.notna() & (hassotime != 0)
    if not valid_ht.any():
        return pd.DataFrame(columns=_OUTPUT_COLS)

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
        return pd.DataFrame(columns=_OUTPUT_COLS)

    # 2. 必要列だけ選択してから処理 (メモリ削減: 全列コピーを避ける)
    ts = odds_ts_df[required].copy()

    # happyotime → datetime (vectorized)
    ht_raw = ts["happyotime"].astype(str).str.zfill(8)
    year_str = ts["year"].astype(int).astype(str)
    datetime_str = year_str + ht_raw.str[:4] + ht_raw.str[4:]
    ts["_ht_datetime"] = pd.to_datetime(datetime_str, format="%Y%m%d%H%M", errors="coerce")
    ts = ts[ts["_ht_datetime"].notna()]

    if ts.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    # 3. cutoff フィルタ (vectorized)
    now = _now or datetime.now()
    now_pd = pd.Timestamp(now)

    # post_datetime を merge で付与
    post_time_series = post_time_map.rename("post_datetime")
    ts = ts.merge(post_time_series, left_on="race_id", right_index=True, how="inner")

    cutoff = ts["post_datetime"] - pd.Timedelta(minutes=minutes_before)
    min_cutoff = cutoff - pd.Timedelta(minutes=max_staleness_minutes)

    # cutoff時刻に達していないレースは除外
    valid = (
        (cutoff <= now_pd)
        & (ts["_ht_datetime"] >= min_cutoff)
        & (ts["_ht_datetime"] <= cutoff)
    )
    filtered = ts[valid]

    if filtered.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    # 4. (race_id, umaban) ごとに最新エントリを取得
    idx = filtered.groupby(["race_id", "umaban"], observed=True)["_ht_datetime"].idxmax()
    snapshot = filtered.loc[idx]

    # 5. 列名正規化して返す (ninki → tanninki)
    result = snapshot.rename(columns={ninki_col: "tanninki"})
    result = result[_OUTPUT_COLS].reset_index(drop=True)
    return result
