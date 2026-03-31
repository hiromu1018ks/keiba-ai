"""オッズ収集 + t-3min/t-2min スナップショット (F-4b)

設計書 §8: 5分間隔の定期収集 + t-3min判定用 + t-2minログ用。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

import pandas as pd

from db.readers import save_predictions

if TYPE_CHECKING:
    from db.parquet_store import ParquetStore

logger = logging.getLogger(__name__)


@runtime_checkable
class OddsFetcherProtocol(Protocol):
    def fetch_odds_snapshot(self, race_id: str) -> dict[int, float]: ...


class OddsCollector:
    """オッズの定期収集と発走直前スナップショット管理

    設計書 §8 のオッズ監視システム。
    t-3min スナップショットは LateMoneyFilter の判定に使用。
    t-2min スナップショットはログのみ（将来チューニング用）。
    """

    def __init__(
        self,
        fetcher: OddsFetcherProtocol,
        store: Optional[ParquetStore] = None,
    ) -> None:
        self.fetcher = fetcher
        self.store = store

    def collect_t3_snapshot(self, race_id: str) -> dict[int, float]:
        """発走3分前のオッズスナップショットを取得

        LateMoneyFilter.process_last_minute() の odds_t3_snapshot に使用。
        """
        snapshot = self.fetcher.fetch_odds_snapshot(race_id)
        logger.info(f"[t-3min] race={race_id} n_horses={len(snapshot)}")
        return snapshot

    def collect_t2_snapshot(self, race_id: str) -> dict[int, float]:
        """発走2分前のオッズスナップショットを取得（ログ用途）

        設計書 §8: 判定には使わない。将来のチューニングデータ。
        """
        snapshot = self.fetcher.fetch_odds_snapshot(race_id)
        logger.info(f"[t-2min LOG ONLY] race={race_id} n_horses={len(snapshot)}")
        return snapshot

    def store_snapshot(
        self,
        race_id: str,
        timing: str,
        snapshot: dict[int, float],
    ) -> None:
        """スナップショットを Parquet に保存

        Args:
            race_id: レースID
            timing: "t3" or "t2"
            snapshot: horse_no → odds
        """
        if self.store is None:
            return

        rows = []
        for horse_no, odds in snapshot.items():
            rows.append(
                {
                    "race_id": race_id,
                    "horse_no": horse_no,
                    "tan_odds": odds,
                    "timing": timing,
                }
            )
        df = pd.DataFrame(rows)
        save_predictions(self.store, df)
        logger.info(f"[{timing}] Saved {len(rows)} odds for race={race_id}")

    @staticmethod
    def get_odds_change_rate(
        odds_before: float,
        odds_after: float,
    ) -> Optional[float]:
        """オッズ変化率を計算 (正値 = 急落、負値 = 急騰)

        Returns:
            変化率。オッズが0の場合は None。
        """
        if odds_before <= 0 or odds_after <= 0:
            return None
        return (odds_before - odds_after) / odds_before
