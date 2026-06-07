"""Parquet読み取りヘルパー。

型変換・リネームは基本的にしないが、旧ETL互換のため
race_date の datetime 変換と数値列の型強制を行う。
新ETLで書き出されたParquetではこれらは既に正しい型。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd

from db.etl import _apply_type_conversions, _compute_race_date, _compute_race_id
from db.parquet_store import ParquetStore

if TYPE_CHECKING:
    from db.everydb2_queries import EveryDB2Queries

logger = logging.getLogger(__name__)

# 旧ETL互換: Parquet内で文字列として保存されている可能性のある数値列
# ETLの _TABLE_TYPE_RULES と同一のカラムセット
_INT_COLS: set[str] = {
    "trackcd",
    "kyori",
    "tenkocd",
    "syussotosu",
    "honsyokin",
    "umaban",
    "kakuteijyuni",
    "ninki",
    "kyakusitukubun",
    "jyuni1c",
    "jyuni4c",
    "jyuni2c",  # ETL-03: corner position 2
    "jyuni3c",  # ETL-03: corner position 3
    "zogenfugo",
    "tanninki",
}
_FLOAT_COLS: set[str] = {
    "time",
    "bataijyu",
    "zogensa",
    "harontimel3",
    "harontimel4",  # ETL-01: SE table haron time L4
    "timediff",
    *[f"laptime{i}" for i in range(1, 26)],  # ETL-02: RA table lap times
}

# coerce_typesで数値変換しない文字列固有列
_STRING_COLUMNS: set[str] = {
    "race_id",
    "kettonum",
    "bamei",
    "kisyucode",
    "chokyosicode",
    "banusicode",
    "recordspec",
    "datakubun",
    "makedate",
    "hondai",
    "fukudai",
    "kakko",
    "hondaieng",
    "fukudaieng",
    "kakkoeng",
    "ryakusyo10",
    "ryakusyo6",
    "ryakusyo3",
    "jyokenname",
    "chokyosiryakusyo",
    "banusiname",
    "kisyuryakusyo",
    "kisyuryakusyobefore",
    "kumi",  # ワイドオッズの馬番組み合わせ (e.g. "0102")
    "surface",  # ETL派生列: "turf"/"dirt"/"other" (文字列)
    "happyotime",  # 時系列オッズの発走時刻コード (e.g. "03101500")
    "gradecd",  # レースグレードコード (A=G1, B=G2, C=G3, D=Listed, E=OP等)
    "gradecdbefore",  # 前回レースグレードコード
    "ketto3infohansyokunum1",  # 種牡馬血統番号 (horses)
    "ketto3infohansyokunum2",
    "ketto3infohansyokunum3",
    "ketto3infohansyokunum4",
    "ketto3infohansyokunum5",  # 母父馬
    "ketto3infohansyokunum6",
    "ketto3infohansyokunum7",
    "ketto3infohansyokunum8",
    "ketto3infohansyokunum9",
    "ketto3infohansyokunum10",
    "ketto3infohansyokunum11",
    "ketto3infohansyokunum12",
    "ketto3infohansyokunum13",
    "ketto3infohansyokunum14",
    "hansyokunum",  # 現行keito/hansyokuの繁殖登録番号
    "keitoid",  # 現行keitoの系統ID
    "keitoname",  # 現行keitoの系統名
    "keitoex",  # 現行keitoの系統説明
    "keitoucode",  # 血統番号 (keito)
    "keitousystemcd",  # 系統コード (keito, e.g. "SS"=サンデーサイレンス系)
}


def load_races_from_db(db: EveryDB2Queries, ymd: str) -> pd.DataFrame:
    """EveryDB2 からレース情報を読み込む。"""
    raw = db.get_races(ymd)
    if raw.empty:
        return raw
    df = _apply_type_conversions(raw, "races")
    df = _compute_race_date(df)
    df = _compute_race_id(df)
    df = coerce_types(df)
    return _exclude_steeple(df)


def load_entries_from_db(db: EveryDB2Queries, ymd: str) -> pd.DataFrame:
    """EveryDB2 から出走馬を読み込む。"""
    raw = db.get_entries(ymd)
    if raw.empty:
        return raw
    df = _apply_type_conversions(raw, "entries")
    df = _compute_race_date(df)
    df = _compute_race_id(df)
    df = coerce_types(df)
    return _exclude_steeple(df)


def load_odds_snapshots_from_db(db: EveryDB2Queries, ymd: str) -> pd.DataFrame:
    """EveryDB2 から単勝・複勝オッズスナップショットを読み込む。"""
    raw = db.get_odds_snapshots(ymd)
    if raw.empty:
        return raw
    df = _apply_type_conversions(raw, "odds_tanpuku")
    df = _compute_race_date(df)
    df = _compute_race_id(df)
    return coerce_types(df)


def load_odds_time_series_from_db(db: EveryDB2Queries, ymd: str) -> pd.DataFrame:
    """EveryDB2 から時系列オッズを読み込む。"""
    raw = db.get_odds_time_series(ymd)
    if raw.empty:
        return raw
    df = _apply_type_conversions(raw, "jodds_tanpuku")
    df = _compute_race_date(df)
    df = _compute_race_id(df)
    return coerce_types(df)


def _to_dt(yyyymmdd: str) -> datetime:
    return datetime.strptime(yyyymmdd, "%Y%m%d")


def date_filters(start: str, end: str) -> list[tuple]:
    s, e = _to_dt(start), _to_dt(end)
    return [("race_date", ">=", s), ("race_date", "<=", e)]


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """旧ETL互換: 文字列型の数値列を適切な型に変換する。

    race_dateはdatetimeに、それ以外のobject型列はpd.to_numericで
    数値に変換できるもののみ変換（文字列固有の列はそのまま）。
    また、ETLで計算される派生列（surface, track_condition_code）が
    存在しない場合はフォールバックで計算する。

    新ETL形式では既に正しい型のため、早期returnで高速化する。
    """
    if df.empty:
        return df

    # 新ETL形式では既に正しい型 → 早期return
    # race_dateがdatetimeで、object型の数値列が存在しない場合はスキップ
    if "race_date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["race_date"]):
        object_needing_coercion = [
            c for c in df.columns
            if df[c].dtype == object and c not in _STRING_COLUMNS
        ]
        if not object_needing_coercion:
            return df

    if "race_date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["race_date"]):
        df["race_date"] = pd.to_datetime(df["race_date"])

    for col in df.columns:
        if df[col].dtype == object and col not in _STRING_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ETL派生列のフォールバック（旧Parquet互換）
    if "surface" not in df.columns and "trackcd" in df.columns:
        import numpy as np

        trackcd = df["trackcd"]
        df["surface"] = np.where(
            trackcd.isna(), "other",
            np.where(trackcd.between(10, 22), "turf",
                     np.where(trackcd.between(23, 29), "dirt", "other"))
        )

    if (
        "track_condition_code" not in df.columns
        and "sibababacd" in df.columns
        and "trackcd" in df.columns
    ):
        import numpy as np

        is_turf = df["trackcd"].between(10, 22)
        df["track_condition_code"] = np.where(is_turf, df["sibababacd"], df["dirtbabacd"])

    return df


def _exclude_steeple(df: pd.DataFrame) -> pd.DataFrame:
    """障害レース除外（trackcd 51-59）。trackcd列がなければそのまま返す。"""
    if "trackcd" not in df.columns:
        return df
    return df[~df["trackcd"].between(51, 59)].copy()


def load_races(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
    df = store.read("raw", "races", filters=date_filters(start, end))
    df = coerce_types(df)
    return _exclude_steeple(df)


def load_entries(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
    df = store.read("raw", "entries", filters=date_filters(start, end))
    df = coerce_types(df)
    return _exclude_steeple(df)


def load_odds_snapshots(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
    df = store.read("odds", "odds_tanpuku", filters=date_filters(start, end))
    return coerce_types(df)


def load_odds_time_series_range(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
    s, e = _to_dt(start), _to_dt(end)
    filters = [
        ("year", ">=", s.year),
        ("year", "<=", e.year),
        ("race_date", ">=", s),
        ("race_date", "<=", e),
    ]
    # jodds_tanpuku (新ETL) を優先: 100% JRA, 2015-2026, 重複なし
    # time_series (旧ETL) は 2015-2024 のみで4x重複・NAR混入あり
    subpath = "jodds_tanpuku" if store.exists("odds", "jodds_tanpuku") else "time_series"
    df = store.read("odds", subpath, filters=filters)
    if df.empty and subpath == "jodds_tanpuku" and store.exists("odds", "time_series"):
        logger.debug("jodds_tanpuku empty for %s-%s, falling back to time_series", start, end)
        df = store.read("odds", "time_series", filters=filters)
    df = coerce_types(df)
    # 旧time_seriesの列名を生カラム名に正規化
    rename_ts = {"happyo_time": "happyotime", "tan_odds": "tanodds", "fuku_odds": "fukuoddslow"}
    existing = {k: v for k, v in rename_ts.items() if k in df.columns and v not in df.columns}
    if existing:
        df = df.rename(columns=existing)
    return df


def load_odds_time_series(store: ParquetStore, race_id: str) -> pd.DataFrame:
    subpath = "jodds_tanpuku" if store.exists("odds", "jodds_tanpuku") else "time_series"
    df = store.read("odds", subpath, filters=[("race_id", "==", race_id)])
    if df.empty and subpath == "jodds_tanpuku" and store.exists("odds", "time_series"):
        logger.debug("jodds_tanpuku empty for %s, falling back to time_series", race_id)
        df = store.read("odds", "time_series", filters=[("race_id", "==", race_id)])
    df = coerce_types(df)
    rename_ts = {"happyo_time": "happyotime", "tan_odds": "tanodds", "fuku_odds": "fukuoddslow"}
    existing = {k: v for k, v in rename_ts.items() if k in df.columns and v not in df.columns}
    if existing:
        df = df.rename(columns=existing)
    return df


def load_wide_odds(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
    df = store.read("odds", "odds_wide", filters=date_filters(start, end))
    return coerce_types(df)


def load_payouts(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
    df = store.read("raw", "payouts", filters=date_filters(start, end))
    return coerce_types(df)


def load_history_entries(store: ParquetStore, lookback_years: int = 5) -> pd.DataFrame:
    cutoff = datetime.now() - timedelta(days=lookback_years * 365)
    df = store.read("raw", "entries", filters=[("race_date", ">=", cutoff)])
    return coerce_types(df)


def load_history_races(store: ParquetStore, lookback_years: int = 5) -> pd.DataFrame:
    cutoff = datetime.now() - timedelta(days=lookback_years * 365)
    df = store.read("raw", "races", filters=[("race_date", ">=", cutoff)])
    return coerce_types(df)


def load_horses(store: ParquetStore) -> pd.DataFrame:
    df = store.read("raw", "horses")
    return coerce_types(df)


def load_jockey_stats(store: ParquetStore) -> pd.DataFrame:
    df = store.read("raw", "kisyu_seiseki")
    return coerce_types(df)


def load_trainer_stats(store: ParquetStore) -> pd.DataFrame:
    df = store.read("raw", "chokyo_seiseki")
    return coerce_types(df)


def load_career_stats(store: ParquetStore) -> pd.DataFrame:
    """Point-in-time キャリア統計を読み込む。"""
    if not store.exists("raw", "horse_career_stats"):
        return pd.DataFrame()
    df = store.read("raw", "horse_career_stats")
    return coerce_types(df)


def load_keito(store: ParquetStore) -> pd.DataFrame:
    """系統コードマスタを読み込む。"""
    if not store.exists("raw", "keito"):
        return pd.DataFrame()
    df = store.read("raw", "keito")
    return coerce_types(df)


def load_sire_stats(store: ParquetStore) -> pd.DataFrame:
    """種牡馬産駎累積統計を読み込む。"""
    if not store.exists("raw", "sire_career_stats"):
        return pd.DataFrame()
    df = store.read("raw", "sire_career_stats")
    return coerce_types(df)


def load_horse_track_aptitude(store: ParquetStore) -> pd.DataFrame:
    """馬場条件適性統計を読み込む。"""
    if not store.exists("raw", "horse_track_aptitude"):
        return pd.DataFrame()
    df = store.read("raw", "horse_track_aptitude")
    return coerce_types(df)


def load_features(store: ParquetStore, start: str, end: str) -> pd.DataFrame | None:
    if not store.exists("features", "horse_features"):
        return None
    df = store.read("features", "horse_features", filters=date_filters(start, end))
    return coerce_types(df)


def save_features(store: ParquetStore, df: pd.DataFrame) -> None:
    store.write("features", "horse_features", df)


def save_predictions(store: ParquetStore, df: pd.DataFrame) -> None:
    store.write("predictions", "predictions", df)


def save_bets(store: ParquetStore, df: pd.DataFrame) -> None:
    store.write("bets", "bets", df)
