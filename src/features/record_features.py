"""record_features.py -- Group B-3: コースレコード特徴量 (DATA-02)

n_record (コースレコードマスタ) からコースレコードタイム特徴量を生成する。

主な特徴量:
  - course_record_time: コースレコードタイム (秒)

PIT安全性:
  n_record 全48列がPRE (静的履歴データ)。レース結果は含まれない。
  RecInfoKubun=1 はコースレコード、RecInfoKubun=2 はGIレコード。
  コースレコードはレース前に公開されている情報。

注意:
  course_record_time はレースレベル特徴量 (同じレースの全馬に同じ値)。
  (jyocd, trackcd, kyori) で一意に定まる。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from db.parquet_store import ParquetStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLS: list[str] = ["course_record_time"]


class RecordFeatures:
    """コースレコード特徴量を生成。

    n_record から各 (jyocd, trackcd, kyori) のコースレコードタイムを取得する。
    """

    def __init__(self, store: ParquetStore) -> None:
        self.store = store
        self._record_cache: pd.DataFrame | None = None

    def _load_records(self) -> pd.DataFrame:
        """record.parquet を読み込みキャッシュする"""
        if self._record_cache is None:
            if self.store.exists("raw", "record"):
                self._record_cache = self.store.read("raw", "record")
            else:
                self._record_cache = pd.DataFrame()
        return self._record_cache

    @staticmethod
    def _parse_rectime(rectime: str | pd.Series) -> float | pd.Series:
        """RecTime (varchar) を秒数に変換。

        RecTime format: 4文字 "msss"
        char 0 = 分 (0-9)
        chars 1-3 = ss.s (秒.小数)

        例: "1553" → 1*60 + 55.3 = 115.3秒
        例: "0550" → 0*60 + 55.0 = 55.0秒
        例: "0000" or invalid → NaN
        """
        if isinstance(rectime, str):
            if len(rectime) < 4:
                return np.nan
            try:
                minutes = int(rectime[0])
                seconds = float(rectime[1:4]) / 10.0  # chars 1-3 = ss.s
                total = minutes * 60 + seconds
                if total <= 0:
                    return np.nan
                return total
            except (ValueError, IndexError):
                return np.nan
        # Series 版
        s = rectime.astype(str)
        minutes = pd.to_numeric(s.str[0], errors="coerce")
        seconds_part = pd.to_numeric(s.str[1:4], errors="coerce") / 10.0
        total = minutes * 60 + seconds_part
        total = total.where(total > 0, np.nan)
        return total

    def compute(self, race_df: pd.DataFrame) -> pd.DataFrame:
        """race_df (race_id, jyocd, trackcd, kyori) -> コースレコード特徴量 DataFrame。

        Args:
            race_df: race_id, jyocd, trackcd, kyori 列を持つ DataFrame

        Returns:
            race_id + FEATURE_COLS を持つ DataFrame
        """
        record_df = self._load_records()

        if record_df.empty:
            return race_df[["race_id"]].assign(
                **{c: float("nan") for c in FEATURE_COLS}
            )

        # 列名正規化 (小文字)
        record_df = record_df.copy()
        record_df.columns = record_df.columns.str.lower()

        # RecInfoKubun=1 (コースレコード) のみフィルタ
        if "recinfokubun" in record_df.columns:
            record_df = record_df[record_df["recinfokubun"] == "1"]

        if record_df.empty:
            return race_df[["race_id"]].assign(
                **{c: float("nan") for c in FEATURE_COLS}
            )

        # 必要列の確認
        required = {"jyocd", "trackcd", "kyori", "rectime"}
        if not required.issubset(set(record_df.columns)):
            return race_df[["race_id"]].assign(
                **{c: float("nan") for c in FEATURE_COLS}
            )

        # RecTime を秒数に変換
        record_df = record_df.copy()
        record_df["course_record_time"] = self._parse_rectime(record_df["rectime"])

        # makedate があれば最新レコードを選択
        keys = ["jyocd", "trackcd", "kyori"]
        if "makedate" in record_df.columns:
            record_df = (
                record_df.sort_values("makedate")
                .groupby(keys, sort=False, observed=True)
                .last()
                .reset_index()
            )
        else:
            record_df = record_df.groupby(keys, sort=False, observed=True).first().reset_index()

        # ルックアップテーブル作成
        lookup = record_df[keys + ["course_record_time"]].drop_duplicates(
            subset=keys, keep="last"
        )

        # race_df の各列を正規化してマージ
        result = race_df[["race_id"]].copy()
        # race_df のキー列を文字列に統一
        for col in keys:
            if col in race_df.columns:
                result[col] = race_df[col].astype(str)
            else:
                result[col] = ""

        lookup_dedup = lookup.copy()
        for col in keys:
            lookup_dedup[col] = lookup_dedup[col].astype(str)

        result = result.merge(
            lookup_dedup[keys + ["course_record_time"]],
            on=keys,
            how="left",
        )

        # レースレベル特徴量: 同じレースの全馬に同じ値がマッピングされるため
        # race_id で一意にする (呼び出し側が on=["race_id"] で merge する前提)
        result = result[["race_id"] + FEATURE_COLS].drop_duplicates(subset=["race_id"])
        return result
