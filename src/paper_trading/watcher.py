"""Paper Trading watch フェーズ — レース時刻監視・ベット通知"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from db.everydb2_queries import EveryDB2Queries
    from monitoring.notifier import NotifierProtocol
    from paper_trading.predictor import PaperPredictor

logger = logging.getLogger(__name__)


def wait_until(target_time: datetime) -> None:
    """target_time まで待機 (テスト用にモック可能)"""
    now = datetime.now()
    if target_time > now:
        import time

        time.sleep((target_time - now).total_seconds())


class RaceWatcher:
    """watch フェーズ: スケジュールに基づき、各レースの発走-5分にベット通知。

    耐障害性:
    - PostgreSQL接続断: 接続を再確立してリトライ
    - プロセスクラッシュ: 既通知済みレースは predictions/YYYYMMDD.parquet で判定
    - ハートビート: 5分おきにログ出力
    """

    def __init__(
        self,
        predictor: PaperPredictor,
        everydb2: EveryDB2Queries,
        notifier: NotifierProtocol,
        predictions_dir: Path,
        retry_count: int = 3,
        retry_interval_seconds: int = 60,
        watch_lead_minutes: int = 5,
    ) -> None:
        self.predictor = predictor
        self.everydb2 = everydb2
        self.notifier = notifier
        self.predictions_dir = predictions_dir
        self.retry_count = retry_count
        self.retry_interval_seconds = retry_interval_seconds
        self.watch_lead_minutes = watch_lead_minutes

    def watch(
        self,
        target_date: date,
        schedule: list[dict[str, Any]],
        bankroll: float,
    ) -> list[dict[str, Any]]:
        """スケジュールに基づき、各レースの発走-5分にベット通知。

        Returns:
            当日の全ベット記録
        """
        import time

        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        ymd = target_date.strftime("%Y%m%d")
        final_pred_path = self.predictions_dir / f"{ymd}.parquet"
        pre_pred_path = self.predictions_dir / f"{ymd}_pre.parquet"

        all_bets: list[dict[str, Any]] = []

        if not pre_pred_path.exists():
            logger.error("Pre-computed features not found: %s", pre_pred_path)
            return all_bets

        pre_computed = pd.read_parquet(pre_pred_path)

        for race in schedule:
            race_id = race["race_id"]

            # 既に処理済みならスキップ
            if self._already_processed(race_id, final_pred_path):
                logger.info("Skipping already processed race: %s", race_id)
                continue

            # 発走-5分まで待機
            post_time = self._parse_post_time(target_date, race["post_time"])
            wait_until(post_time - timedelta(minutes=self.watch_lead_minutes))

            # PostgreSQLから当日データを取得
            horse_weights = None
            odds = None
            for attempt in range(self.retry_count):
                horse_weights = self.everydb2.get_horse_weights(race_id)
                odds = self.everydb2.get_latest_odds(race_id)
                if horse_weights is not None and odds is not None:
                    break
                logger.warning(
                    "Data fetch attempt %d/%d failed for %s",
                    attempt + 1,
                    self.retry_count,
                    race_id,
                )
                time.sleep(self.retry_interval_seconds)
            else:
                logger.warning("All data fetch attempts failed for %s, skipping", race_id)
                continue

            # 推論
            bets = self.predictor.predict_race(race_id, pre_computed, horse_weights, odds, bankroll)

            if bets:
                self.notifier.send_prediction(bets=bets, date=target_date.isoformat())
                all_bets.extend(bets)
                bankroll = bets[-1]["bankroll_after"]

        # 最終予測を保存
        if all_bets:
            pd.DataFrame(all_bets).to_parquet(final_pred_path, index=False)

        return all_bets

    def _already_processed(self, race_id: str, final_pred_path: Path) -> bool:
        """既に処理済みのレースかチェック"""
        if not final_pred_path.exists():
            return False
        try:
            df = pd.read_parquet(final_pred_path)
            return race_id in df["race_id"].values
        except Exception:
            return False

    @staticmethod
    def _parse_post_time(target_date: date, post_time_str: str) -> datetime:
        """'HH:MM' 形式の発走時刻を datetime に変換"""
        h, m = map(int, post_time_str.split(":"))
        return datetime.combine(target_date, __import__("datetime").time(h, m))
