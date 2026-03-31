"""Paper Trading 結果照合・ROI計算"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from db.everydb2_queries import EveryDB2Queries
    from db.parquet_store import ParquetStore

logger = logging.getLogger(__name__)


class PaperReconciler:
    """reconcile フェーズ: 予測と結果を照合して ROI を追跡。

    冪等性: 同一 race_id + umaban のレコードが既に存在する場合はスキップ。
    """

    def __init__(
        self,
        store: ParquetStore,
        bets_path: Path,
        everydb2: EveryDB2Queries,
        monitor: Any | None = None,
    ) -> None:
        self.store = store
        self.bets_path = bets_path
        self.everydb2 = everydb2
        self.monitor = monitor

    def reconcile(self, target_date: date) -> dict[str, Any]:
        """当日のレース結果を取得し、予測と照合。

        Returns:
            日次結果サマリー (n_settled, n_wins, daily_roi, cumulative_roi, etc.)
        """
        # 1. 既存ベットを読み込み
        if self.bets_path.exists():
            bets_df = pd.read_parquet(self.bets_path)
        else:
            bets_df = pd.DataFrame()

        if bets_df.empty:
            logger.info("No bets to reconcile for %s", target_date)
            return self._empty_result(target_date)

        # 2. 当日の未確定ベットを抽出
        target_ts = pd.Timestamp(target_date)
        pending = bets_df[(bets_df["race_date"] == target_ts) & (bets_df["result"] == 0.0)]

        if pending.empty:
            logger.info("No pending bets for %s", target_date)
            return self._compute_summary(bets_df, target_date)

        # 3. レース結果を取得
        results_df = self.everydb2.get_race_results(target_date)
        if results_df.empty:
            logger.warning("No race results available for %s", target_date)
            return self._compute_summary(bets_df, target_date)

        # 4. 照合: race_id + umaban でマージ
        n_settled = 0
        n_wins = 0

        for _, bet_row in pending.iterrows():
            race_id = bet_row["race_id"]
            umaban = bet_row["umaban"]

            # 結果検索
            result_row = results_df[
                (results_df["race_id"] == race_id) & (results_df["umaban"] == umaban)
            ]
            if result_row.empty:
                continue

            finish_pos = int(result_row.iloc[0]["kakuteijyuni"])
            bet_type = bet_row["bet_type"]

            # 複勝的中判定
            payout = 0.0
            if bet_type == "place" and 1 <= finish_pos <= 3:
                payout = bet_row["stake"] * bet_row["odds"]
                n_wins += 1

            # 払戻を更新
            mask = (bets_df["race_id"] == race_id) & (bets_df["umaban"] == umaban)
            bets_df.loc[mask, "result"] = payout
            n_settled += 1

        # 5. 保存
        bets_df.to_parquet(self.bets_path, index=False)

        return self._compute_summary(bets_df, target_date, n_settled, n_wins)

    def _compute_summary(
        self,
        bets_df: pd.DataFrame,
        target_date: date,
        n_settled: int = 0,
        n_wins: int = 0,
    ) -> dict[str, Any]:
        """累積統計を計算"""
        total_bets = len(bets_df)
        total_stake = bets_df["stake"].sum()
        total_return = bets_df[bets_df["result"] > 0]["result"].sum()
        total_wins = int((bets_df["result"] > 0).sum())

        cumulative_roi = total_return / total_stake if total_stake > 0 else 0.0

        # Max drawdown
        bankroll_series = bets_df["bankroll_after"]
        peak = bankroll_series.cummax()
        dd = (peak - bankroll_series) / peak
        max_dd = dd.max() if not dd.empty else 0.0

        return {
            "date": target_date.isoformat(),
            "n_bets": total_bets,
            "n_wins": total_wins,
            "total_stake": float(total_stake),
            "total_return": float(total_return),
            "cumulative_roi": float(cumulative_roi),
            "max_dd": float(max_dd),
            "bankroll": float(bankroll_series.iloc[-1]) if not bankroll_series.empty else 100000.0,
            "n_settled": n_settled,
            "n_new_wins": n_wins,
        }

    def _empty_result(self, target_date: date) -> dict[str, Any]:
        return {
            "date": target_date.isoformat(),
            "n_bets": 0,
            "n_wins": 0,
            "total_stake": 0.0,
            "total_return": 0.0,
            "cumulative_roi": 0.0,
            "max_dd": 0.0,
            "bankroll": 100000.0,
            "n_settled": 0,
            "n_new_wins": 0,
        }
