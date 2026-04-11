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

    # 1. race_id -> post_datetime のマッピング
    post_time_map: dict[str, datetime] = {}
    for _, r in race_df.iterrows():
        ht = r.get("hassotime")
        if pd.isna(ht) or str(ht).strip() == "":
            continue
        ht_str = f"{int(ht):04d}"  # 930 -> "0930"
        # race_id の先頭8桁 = YYYYMMDD
        rid = r["race_id"]
        race_date_str = rid[:8]
        post_time_map[rid] = datetime(
            int(race_date_str[:4]),
            int(race_date_str[4:6]),
            int(race_date_str[6:8]),
            int(ht_str[:2]),
            int(ht_str[2:]),
        )

    # 2. odds_ts_df の各行について happyotime -> datetime
    def _parse_happyotime(row: pd.Series) -> datetime | None:
        ht = row.get("happyotime")
        if pd.isna(ht):
            return None
        ht = str(ht).zfill(8)  # "4110930" -> "04110930"
        if len(ht) != 8:
            return None
        year = int(row["year"])
        month = int(ht[:2])
        day = int(ht[2:4])
        hour = int(ht[4:6])
        minute = int(ht[6:8])
        return datetime(year, month, day, hour, minute)

    odds_ts_df = odds_ts_df.copy()
    odds_ts_df["_ht_datetime"] = odds_ts_df.apply(_parse_happyotime, axis=1)
    odds_ts_df = odds_ts_df[odds_ts_df["_ht_datetime"].notna()]

    # 3. 各行に cutoff を付与し、cutoff 以前のエントリのみ残す
    now = _now or datetime.now()

    def _is_before_cutoff(row: pd.Series) -> bool:
        post_time = post_time_map.get(row["race_id"])
        if post_time is None:
            return False
        cutoff = post_time - timedelta(minutes=minutes_before)
        # cutoff時刻に達していないレースはまだオッズが確定していない → 除外
        if cutoff > now:
            return False
        min_cutoff = cutoff - timedelta(minutes=max_staleness_minutes)
        ht_dt = row["_ht_datetime"]
        return min_cutoff <= ht_dt <= cutoff

    mask = odds_ts_df.apply(_is_before_cutoff, axis=1)
    valid = odds_ts_df[mask]

    if valid.empty:
        return pd.DataFrame(columns=["race_id", "umaban", "tanodds", "fukuoddslow", "tanninki"])

    # 4. (race_id, umaban) ごとに最新エントリを取得
    idx = valid.groupby(["race_id", "umaban"])["_ht_datetime"].idxmax()
    snapshot = valid.loc[idx]

    # 5. build_all() と互換のスキーマで返す
    result = snapshot[["race_id", "umaban", "tanodds", "fukuoddslow", "tanninki"]].copy()
    result = result.reset_index(drop=True)
    return result
