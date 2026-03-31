"""Parquet読み取りヘルパー。型変換・リネームは一切しない。

前提: racedテーブルのParquetには race_id, race_date, surface が
ETLで事前計算されて含まれていること。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from db.parquet_store import ParquetStore


def _to_dt(yyyymmdd: str) -> datetime:
    return datetime.strptime(yyyymmdd, "%Y%m%d")


def _date_filters(start: str, end: str) -> list[tuple]:
    s, e = _to_dt(start), _to_dt(end)
    return [("race_date", ">=", s), ("race_date", "<=", e)]


def _exclude_steeple(df: pd.DataFrame) -> pd.DataFrame:
    """障害レース除外（trackcd 51-59）。trackcd列がなければそのまま返す。"""
    if "trackcd" not in df.columns:
        return df
    return df[~df["trackcd"].between(51, 59)].copy()


def load_races(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
    df = store.read("raw", "races", filters=_date_filters(start, end))
    return _exclude_steeple(df)


def load_entries(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
    df = store.read("raw", "entries", filters=_date_filters(start, end))
    return _exclude_steeple(df)


def load_odds_snapshots(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
    return store.read("odds", "odds_tanpuku", filters=_date_filters(start, end))


def load_odds_time_series_range(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
    s, e = _to_dt(start), _to_dt(end)
    filters = [
        ("year", ">=", s.year),
        ("year", "<=", e.year),
        ("race_date", ">=", s),
        ("race_date", "<=", e),
    ]
    return store.read("odds", "jodds_tanpuku", filters=filters)


def load_odds_time_series(store: ParquetStore, race_id: str) -> pd.DataFrame:
    return store.read("odds", "jodds_tanpuku", filters=[("race_id", "==", race_id)])


def load_wide_odds(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
    return store.read("odds", "odds_wide", filters=_date_filters(start, end))


def load_payouts(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
    return store.read("raw", "payouts", filters=_date_filters(start, end))


def load_history_entries(store: ParquetStore, lookback_years: int = 5) -> pd.DataFrame:
    cutoff = datetime.now() - timedelta(days=lookback_years * 365)
    return store.read("raw", "entries", filters=[("race_date", ">=", cutoff)])


def load_history_races(store: ParquetStore, lookback_years: int = 5) -> pd.DataFrame:
    cutoff = datetime.now() - timedelta(days=lookback_years * 365)
    return store.read("raw", "races", filters=[("race_date", ">=", cutoff)])


def load_horses(store: ParquetStore) -> pd.DataFrame:
    return store.read("raw", "horses")


def load_jockey_stats(store: ParquetStore) -> pd.DataFrame:
    return store.read("raw", "kisyu_seiseki")


def load_trainer_stats(store: ParquetStore) -> pd.DataFrame:
    return store.read("raw", "chokyo_seiseki")


def load_features(store: ParquetStore, start: str, end: str) -> pd.DataFrame | None:
    if not store.exists("features", "horse_features"):
        return None
    return store.read("features", "horse_features", filters=_date_filters(start, end))


def save_features(store: ParquetStore, df: pd.DataFrame) -> None:
    store.write("features", "horse_features", df)


def save_predictions(store: ParquetStore, df: pd.DataFrame) -> None:
    store.write("predictions", "predictions", df)


def save_bets(store: ParquetStore, df: pd.DataFrame) -> None:
    store.write("bets", "bets", df)
