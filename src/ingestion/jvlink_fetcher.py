"""JV-Link データ取得インタフェース (F-4a)

JRA-VAN の JV-Link からレースカード・結果・オッズを取得する。
実際の JV-Link SDK は Windows COM コンポーネントのため、
DataRepository 経由でデータにアクセスする設計。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from domain.models import Entry, Race

if TYPE_CHECKING:
    from db.repository import DataRepository

logger = logging.getLogger(__name__)


class JVLinkFetcher:
    """レースデータ・オッズの取得インタフェース

    JV-Link SDK (Windows COM) または DataRepository 経由で
    データを取得する。テストでは mock repo を注入可能。
    """

    def __init__(self, repo: DataRepository) -> None:
        self.repo = repo

    def fetch_race_cards(self, date: str) -> list[Race]:
        """指定日のレースカードを取得

        Args:
            date: 日付 (YYYY-MM-DD)

        Returns:
            Race リスト。レースがない日は空リスト。
        """
        date_compact = date.replace("-", "")
        logger.debug("Fetching race cards for %s", date)
        df = self.repo.load_races(date_compact, date_compact)
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
        df = self.repo.load_entries(date_compact, date_compact)
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
        df = self.repo.load_odds_time_series(race_id)
        if df.empty:
            return {}
        # 最新時刻の行のみ使用
        latest_time = df["happyo_time"].max()
        latest = df[df["happyo_time"] == latest_time]
        return dict(zip(latest["umaban"].astype(int), latest["tan_odds"].astype(float)))

    def _row_to_race(self, row: pd.Series) -> Race:
        return Race(
            year=int(row["year"]),
            month_day=str(row["month_day"]),
            jyo_cd=str(row["jyo_cd"]),
            kaiji=str(row["kaiji"]),
            nichiji=str(row["nichiji"]),
            race_num=str(row["race_num"]),
            track_cd=int(row["track_cd"]),
            distance=int(row["distance"]),
            tenko_cd=int(row["tenko_cd"]),
            baba_cd=int(row["baba_cd"]),
            syubetu_cd=str(row["syubetu_cd"]),
            jyoken_cd=str(row["jyoken_cd"]),
            grade_cd=str(row["grade_cd"]),
            field_size=int(row["field_size"]),
        )

    def _row_to_entry(self, row: pd.Series) -> Entry:
        return Entry(
            race_id=str(row["race_id"]),
            umaban=int(row["umaban"]),
            ketto_num=str(row["ketto_num"]),
            finish_pos=int(row["finish_pos"]),
            win_odds_actual=float(row["win_odds_actual"]),
            popularity_rank=int(row["popularity_rank"]),
            running_style=int(row["running_style"]),
            ba_taijyu=float(row["ba_taijyu"]),
            zogen_fugo=int(row["zogen_fugo"]),
            zogen_sa=float(row["zogen_sa"]),
            kisyu_code=str(row["kisyu_code"]),
            chokyosi_code=str(row["chokyosi_code"]),
        )
