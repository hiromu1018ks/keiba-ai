"""DataRepository — MLパイプラインのデータアクセス窓口。

ParquetStore経由で各種ParquetファイルからDataFrameをロードする。
将来的にreaders.pyの関数群をここに統合する計画 (D-01)。
"""

from __future__ import annotations

import pandas as pd

from db.parquet_store import ParquetStore
from db.readers import coerce_types, date_filters


class DataRepository:
    """MLパイプラインのデータアクセス窓口。

    Args:
        store: ParquetStoreインスタンス。未指定時はデフォルト(data/)で生成。
    """

    def __init__(self, store: ParquetStore | None = None) -> None:
        self._store: ParquetStore = store if store is not None else ParquetStore()

    def load_trio_odds(self, start: str, end: str) -> pd.DataFrame:
        """三連複オッズを読み込む。

        Args:
            start: 開始日 (YYYYMMDD)
            end: 終了日 (YYYYMMDD)

        Returns:
            フィルタ・型変換済みのDataFrame
        """
        df = self._store.read("odds", "odds_sanren", filters=date_filters(start, end))
        return coerce_types(df)

    def load_exacta_odds(self, start: str, end: str) -> pd.DataFrame:
        """馬連オッズを読み込む。

        Args:
            start: 開始日 (YYYYMMDD)
            end: 終了日 (YYYYMMDD)

        Returns:
            フィルタ・型変換済みのDataFrame
        """
        df = self._store.read("odds", "odds_umaren", filters=date_filters(start, end))
        return coerce_types(df)

    def load_trifecta_odds(self, start: str, end: str) -> pd.DataFrame:
        """三連単オッズを読み込む。

        Args:
            start: 開始日 (YYYYMMDD)
            end: 終了日 (YYYYMMDD)

        Returns:
            フィルタ・型変換済みのDataFrame
        """
        df = self._store.read("odds", "odds_sanrentan", filters=date_filters(start, end))
        return coerce_types(df)

    def load_wide_odds(self, start: str, end: str) -> pd.DataFrame:
        """ワイドオッズを読み込む。

        Args:
            start: 開始日 (YYYYMMDD)
            end: 終了日 (YYYYMMDD)

        Returns:
            フィルタ・型変換済みのDataFrame
        """
        df = self._store.read("odds", "odds_wide", filters=date_filters(start, end))
        return coerce_types(df)

    def load_track_conditions(self, start: str, end: str) -> pd.DataFrame:
        """含水率・クッション値のtrack conditionデータを読み込む。

        Args:
            start: 開始日 (YYYYMMDD)
            end: 終了日 (YYYYMMDD)

        Returns:
            フィルタ・型変換済みのDataFrame。
            parquetが存在しない場合は空DataFrameを返す。
        """
        if not self._store.exists("raw", "track_conditions"):
            return pd.DataFrame()
        df = self._store.read("raw", "track_conditions", filters=date_filters(start, end))
        return coerce_types(df)
