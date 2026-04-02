"""Parquet読み取りヘルパー。

型変換・リネームは基本的にしないが、旧ETL互換のため
race_date の datetime 変換と数値列の型強制を行う。
新ETLで書き出されたParquetではこれらは既に正しい型。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from db.parquet_store import ParquetStore

# 旧ETL互換: Parquet内で文字列として保存されている可能性のある数値列
# ETLの _TABLE_TYPE_RULES と同一のカラムセット
_INT_COLS: set[str] = {
    "trackcd", "kyori", "tenkocd", "syussotosu", "honsyokin",
    "umaban", "kakuteijyuni", "ninki", "kyakusitukubun",
    "jyuni1c", "jyuni4c", "zogenfugo", "tanninki",
}
_FLOAT_COLS: set[str] = {
    "time", "bataijyu", "zogensa", "harontimel3", "timediff",
}

# _coerce_typesで数値変換しない文字列固有列
_STRING_COLUMNS: set[str] = {
    "race_id", "kettonum", "bamei", "kisyucode", "chokyosicode",
    "banusicode", "recordspec", "datakubun", "makedate",
    "hondai", "fukudai", "kakko", "hondaieng", "fukudaieng",
    "kakkoeng", "ryakusyo10", "ryakusyo6", "ryakusyo3",
    "jyokenname", "chokyosiryakusyo", "banusiname",
    "kisyuryakusyo", "kisyuryakusyobefore",
    "kumi",  # ワイドオッズの馬番組み合わせ (e.g. "0102")
    "surface",  # ETL派生列: "turf"/"dirt"/"other" (文字列)
}


def _to_dt(yyyymmdd: str) -> datetime:
    return datetime.strptime(yyyymmdd, "%Y%m%d")


def _date_filters(start: str, end: str) -> list[tuple]:
    s, e = _to_dt(start), _to_dt(end)
    return [("race_date", ">=", s), ("race_date", "<=", e)]


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """旧ETL互換: 文字列型の数値列を適切な型に変換する。

    race_dateはdatetimeに、それ以外のobject型列はpd.to_numericで
    数値に変換できるもののみ変換（文字列固有の列はそのまま）。
    また、ETLで計算される派生列（surface, track_condition_code）が
    存在しない場合はフォールバックで計算する。
    """
    if "race_date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["race_date"]):
        df["race_date"] = pd.to_datetime(df["race_date"])

    for col in df.columns:
        if df[col].dtype == object and col not in _STRING_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ETL派生列のフォールバック（旧Parquet互換）
    if "surface" not in df.columns and "trackcd" in df.columns:
        df["surface"] = df["trackcd"].apply(
            lambda x: "turf" if 10 <= x <= 22 else "dirt" if 23 <= x <= 29 else "other"
        )

    if "track_condition_code" not in df.columns and "sibababacd" in df.columns and "trackcd" in df.columns:
        import numpy as np

        is_turf = df["trackcd"].between(10, 22)
        df["track_condition_code"] = np.where(
            is_turf, df["sibababacd"], df["dirtbabacd"]
        )

    return df


def _exclude_steeple(df: pd.DataFrame) -> pd.DataFrame:
    """障害レース除外（trackcd 51-59）。trackcd列がなければそのまま返す。"""
    if "trackcd" not in df.columns:
        return df
    return df[~df["trackcd"].between(51, 59)].copy()


def load_races(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
    df = store.read("raw", "races", filters=_date_filters(start, end))
    df = _coerce_types(df)
    return _exclude_steeple(df)


def load_entries(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
    df = store.read("raw", "entries", filters=_date_filters(start, end))
    df = _coerce_types(df)
    return _exclude_steeple(df)


def load_odds_snapshots(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
    df = store.read("odds", "odds_tanpuku", filters=_date_filters(start, end))
    return _coerce_types(df)


def load_odds_time_series_range(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
    s, e = _to_dt(start), _to_dt(end)
    filters = [
        ("year", ">=", s.year),
        ("year", "<=", e.year),
        ("race_date", ">=", s),
        ("race_date", "<=", e),
    ]
    # time_series (旧ETL, 高粒度) を優先、なければ jodds_tanpuku (新ETL) を使用
    subpath = "time_series" if store.exists("odds", "time_series") else "jodds_tanpuku"
    df = store.read("odds", subpath, filters=filters)
    df = _coerce_types(df)
    # 旧time_seriesの列名を生カラム名に正規化
    rename_ts = {"happyo_time": "happyotime", "tan_odds": "tanodds", "fuku_odds": "fukuoddslow"}
    existing = {k: v for k, v in rename_ts.items() if k in df.columns and v not in df.columns}
    if existing:
        df = df.rename(columns=existing)
    return df


def load_odds_time_series(store: ParquetStore, race_id: str) -> pd.DataFrame:
    subpath = "time_series" if store.exists("odds", "time_series") else "jodds_tanpuku"
    df = store.read("odds", subpath, filters=[("race_id", "==", race_id)])
    df = _coerce_types(df)
    rename_ts = {"happyo_time": "happyotime", "tan_odds": "tanodds", "fuku_odds": "fukuoddslow"}
    existing = {k: v for k, v in rename_ts.items() if k in df.columns and v not in df.columns}
    if existing:
        df = df.rename(columns=existing)
    return df


def load_wide_odds(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
    df = store.read("odds", "odds_wide", filters=_date_filters(start, end))
    return _coerce_types(df)


def load_payouts(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
    df = store.read("raw", "payouts", filters=_date_filters(start, end))
    return _coerce_types(df)


def load_history_entries(store: ParquetStore, lookback_years: int = 5) -> pd.DataFrame:
    cutoff = datetime.now() - timedelta(days=lookback_years * 365)
    df = store.read("raw", "entries", filters=[("race_date", ">=", cutoff)])
    return _coerce_types(df)


def load_history_races(store: ParquetStore, lookback_years: int = 5) -> pd.DataFrame:
    cutoff = datetime.now() - timedelta(days=lookback_years * 365)
    df = store.read("raw", "races", filters=[("race_date", ">=", cutoff)])
    return _coerce_types(df)


def load_horses(store: ParquetStore) -> pd.DataFrame:
    df = store.read("raw", "horses")
    return _coerce_types(df)


def load_jockey_stats(store: ParquetStore) -> pd.DataFrame:
    df = store.read("raw", "kisyu_seiseki")
    return _coerce_types(df)


def load_trainer_stats(store: ParquetStore) -> pd.DataFrame:
    df = store.read("raw", "chokyo_seiseki")
    return _coerce_types(df)


def load_features(store: ParquetStore, start: str, end: str) -> pd.DataFrame | None:
    if not store.exists("features", "horse_features"):
        return None
    df = store.read("features", "horse_features", filters=_date_filters(start, end))
    return _coerce_types(df)


def save_features(store: ParquetStore, df: pd.DataFrame) -> None:
    store.write("features", "horse_features", df)


def save_predictions(store: ParquetStore, df: pd.DataFrame) -> None:
    store.write("predictions", "predictions", df)


def save_bets(store: ParquetStore, df: pd.DataFrame) -> None:
    store.write("bets", "bets", df)
