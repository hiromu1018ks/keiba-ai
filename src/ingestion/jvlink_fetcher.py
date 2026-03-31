"""JV-Link データ取得インタフェース (F-4a)

JRA-VAN の JV-Link からレースカード・結果・オッズを取得する。
実際の JV-Link SDK は Windows COM コンポーネントのため、
ParquetStore + db.readers 経由でデータにアクセスする設計。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from db.readers import load_entries, load_odds_time_series, load_races
from domain.models import Entry, Race

if TYPE_CHECKING:
    from db.parquet_store import ParquetStore

logger = logging.getLogger(__name__)


class JVLinkFetcher:
    """レースデータ・オッズの取得インタフェース

    JV-Link SDK (Windows COM) または ParquetStore 経由で
    データを取得する。テストでは mock store を注入可能。
    """

    def __init__(self, store: ParquetStore) -> None:
        self.store = store

    def fetch_race_cards(self, date: str) -> list[Race]:
        """指定日のレースカードを取得

        Args:
            date: 日付 (YYYY-MM-DD)

        Returns:
            Race リスト。レースがない日は空リスト。
        """
        date_compact = date.replace("-", "")
        logger.debug("Fetching race cards for %s", date)
        df = load_races(self.store, date_compact, date_compact)
        if df.empty:
            logger.debug("No races found for %s", date)
            return []
        races = [self._row_to_race(row) for _, row in df.iterrows()]
        logger.debug("Found %d races for %s", len(races), date)
        return races

    def fetch_results(self, date: str) -> list[Entry]:
        """指定日の出走馬結果を取得

        Args:
            date: 日付 (YYYY-MM-DD)

        Returns:
            Entry リスト。
        """
        date_compact = date.replace("-", "")
        df = load_entries(self.store, date_compact, date_compact)
        if df.empty:
            return []
        return [self._row_to_entry(row) for _, row in df.iterrows()]

    def fetch_odds_snapshot(self, race_id: str) -> dict[int, float]:
        """特定レースの最新オッズスナップショットを取得

        Args:
            race_id: レースID

        Returns:
            horse_no → tan_odds の dict。
        """
        df = load_odds_time_series(self.store, race_id)
        if df.empty:
            return {}
        # 最新時刻の行のみ使用
        latest_time = df["happyotime"].max()
        latest = df[df["happyotime"] == latest_time]
        return dict(zip(latest["umaban"].astype(int), latest["tanodds"].astype(float)))

    def _row_to_race(self, row: pd.Series) -> Race:
        return Race(
            year=int(row["year"]),
            month_day=str(row["monthday"]),
            jyo_cd=str(row["jyocd"]),
            kaiji=str(row["kaiji"]),
            nichiji=str(row["nichiji"]),
            race_num=str(row["racenum"]),
            track_cd=int(row["trackcd"]),
            distance=int(row["kyori"]),
            tenko_cd=int(row["tenkocd"]),
            baba_cd=int(row["babacd"]),
            syubetu_cd=str(row["syubetucd"]),
            jyoken_cd=str(row["jyokencd"]),
            grade_cd=str(row["gradecd"]),
            field_size=int(row["syussotosu"]),
        )

    def _row_to_entry(self, row: pd.Series) -> Entry:
        return Entry(
            race_id=str(row["race_id"]),
            umaban=int(row["umaban"]),
            ketto_num=str(row["kettonum"]),
            finish_pos=int(row["kakuteijyuni"]),
            win_odds_actual=float(row["odds"]),
            popularity_rank=int(row["ninki"]),
            running_style=int(row["kyakusitukubun"]),
            ba_taijyu=float(row["bataijyu"]),
            zogen_fugo=int(row["zogenfugo"]),
            zogen_sa=float(row["zogensa"]),
            kisyu_code=str(row["kisyucode"]),
            chokyosi_code=str(row["chokyosicode"]),
        )
