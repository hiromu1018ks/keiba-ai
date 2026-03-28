"""MLパイプラインの唯一のデータアクセス窓口。

将来DuckDB/Polarsへの移行を妨げないよう、この層が唯一のアクセス経路。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from db.parquet_store import ParquetStore


def _to_dt(yyyymmdd: str) -> datetime:
    """'YYYYMMDD' 文字列 → datetime"""
    return datetime.strptime(yyyymmdd, "%Y%m%d")


def _date_filters(start: str, end: str) -> list[tuple]:
    """pyarrow述語プッシュダウン用フィルタを生成。"""
    s, e = _to_dt(start), _to_dt(end)
    return [("race_date", ">=", s), ("race_date", "<=", e)]


def _exclude_steeple(df: pd.DataFrame) -> pd.DataFrame:
    """障害レース除外（track_cd 51-59）。track_cd列がなければそのまま返す。"""
    if "track_cd" not in df.columns:
        return df
    return df[~df["track_cd"].between(51, 59)].copy()


class DataRepository:
    """MLパイプラインのデータアクセス窓口。"""

    def __init__(self, store: ParquetStore) -> None:
        self.store = store

    # --- 読み取り（pyarrow filtersでプッシュダウン） ---

    def load_races(self, start: str, end: str) -> pd.DataFrame:
        df = self.store.read("raw", "races", filters=_date_filters(start, end))
        return _exclude_steeple(df)

    def load_entries(self, start: str, end: str) -> pd.DataFrame:
        df = self.store.read("raw", "entries", filters=_date_filters(start, end))
        return _exclude_steeple(df)

    def load_odds_snapshots(self, start: str, end: str) -> pd.DataFrame:
        return self.store.read("odds", "snapshots", filters=_date_filters(start, end))

    def load_odds_time_series_range(self, start: str, end: str) -> pd.DataFrame:
        """オッズ時系列（日付範囲）— パーティションテーブル"""
        return self.store.read("odds", "time_series", filters=_date_filters(start, end))

    def load_odds_time_series(self, race_id: str) -> pd.DataFrame:
        """オッズ時系列（単一レース）"""
        return self.store.read("odds", "time_series", filters=[("race_id", "==", race_id)])

    def load_wide_odds(self, start: str, end: str) -> pd.DataFrame:
        return self.store.read("odds", "wide", filters=_date_filters(start, end))

    def load_payouts(self, start: str, end: str) -> pd.DataFrame:
        return self.store.read("raw", "payouts", filters=_date_filters(start, end))

    # --- 全履歴参照（HorseHistoryFeatures用） ---

    def load_history_entries(self, lookback_years: int = 5) -> pd.DataFrame:
        """過去N年のentriesをロード。lookback_yearsでメモリ制御。

        注意: 障害レースを含む。HorseHistoryFeaturesが全成績を評価するため。
        """
        cutoff = datetime.now() - timedelta(days=lookback_years * 365)
        return self.store.read("raw", "entries", filters=[("race_date", ">=", cutoff)])

    def load_history_races(self, lookback_years: int = 5) -> pd.DataFrame:
        """過去N年のracesをロード。障害レースを含む（HorseHistoryFeatures用）。"""
        cutoff = datetime.now() - timedelta(days=lookback_years * 365)
        return self.store.read("raw", "races", filters=[("race_date", ">=", cutoff)])

    # --- 特徴量キャッシュ ---

    def load_features(self, start: str, end: str) -> pd.DataFrame | None:
        """特徴量キャッシュがあれば返す、なければNone。"""
        if not self.store.exists("features", "horse_features"):
            return None
        return self.store.read("features", "horse_features", filters=_date_filters(start, end))

    def save_features(self, df: pd.DataFrame) -> None:
        self.store.write("features", "horse_features", df)

    # --- 予測・馬券 ---

    def save_predictions(self, df: pd.DataFrame) -> None:
        self.store.write("predictions", "predictions", df)

    def save_bets(self, df: pd.DataFrame) -> None:
        self.store.write("bets", "bets", df)
