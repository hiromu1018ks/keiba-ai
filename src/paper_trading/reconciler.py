"""Paper Trading 結果照合・ROI計算

3-column state model (D-03): settlement_status / outcome / payout
Win/Place settlement via shared payout_maps (D-09)
Correct ROI calculation including losses (D-05)
Retry mechanism (D-06)
Atomic Parquet writes (D-07)
Schema validation (D-20)
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from betting.payout_maps import build_payout_map, build_wide_payout_map, build_win_payout_map

if TYPE_CHECKING:
    from db.everydb2_queries import EveryDB2Queries

logger = logging.getLogger(__name__)


class PaperReconciler:
    """reconcile フェーズ: 予測と結果を照合して ROI を追跡。

    3-column state model (D-03):
        settlement_status: pending -> settled
        outcome: None -> won / lost / refunded / voided
        payout: None -> float (0.0=loss, >0=win, =stake=refunded/voided)

    累積 bets.parquet を精算状態の正本 (source of truth) とする (D-08)。
    """

    def __init__(
        self,
        bets_path: Path,
        everydb2: EveryDB2Queries,
        monitor: Any | None = None,
        retry_interval: int = 60,
        retry_timeout: int = 600,
    ) -> None:
        self.bets_path = bets_path
        self.everydb2 = everydb2
        self.monitor = monitor
        self.retry_interval = retry_interval
        self.retry_timeout = retry_timeout

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_bet_id(
        session_id: str, race_id: str, bet_type: str, umaban: int, umaban_b: int | None = None,
    ) -> str:
        """bet_id = SHA256(session_id|race_id|bet_type|umaban[:32] (D-02)
        Wide bets include umaban_b for uniqueness."""
        raw = f"{session_id}|{race_id}|{bet_type}|{umaban}"
        if umaban_b is not None:
            raw += f"|{umaban_b}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    @staticmethod
    def _atomic_write_parquet(df: pd.DataFrame, target: Path) -> None:
        """Atomic Parquet write via NamedTemporaryFile + replace (D-07)."""
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".parquet",
            dir=str(target.parent),
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
        df.to_parquet(tmp_path, index=False)
        # Windows: os.replace with retry loop for PermissionError when target is open
        max_retries = 3
        for attempt in range(max_retries):
            try:
                os.replace(str(tmp_path), str(target))
                return
            except PermissionError:
                if attempt == max_retries - 1:
                    raise
                time.sleep(0.1)

    @staticmethod
    def _normalize_bet_identifiers(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize legacy numeric race IDs before appending new string IDs."""
        if "race_id" not in df.columns:
            return df

        result = df.copy()

        def _to_string(value: object) -> object:
            if pd.isna(value):
                return pd.NA
            if isinstance(value, (int, float)) and float(value).is_integer():
                return str(int(value))
            return str(value)

        result["race_id"] = result["race_id"].map(_to_string).astype("string")
        return result

    @staticmethod
    def _validate_bet_schema_basic(df: pd.DataFrame) -> list[str]:
        """Basic schema validation: old-schema rejection + required columns.

        Used by both write-time validation (full) and read-time aggregation (basic).
        """
        errors: list[str] = []

        # Old schema rejection (D-18)
        if "result" in df.columns and "payout" not in df.columns:
            errors.append("Old schema detected: 'result' column present without 'payout'")
            return errors  # early return -- cannot validate further

        # Required columns
        for col in ("schema_version", "settlement_status", "outcome", "payout", "bet_id", "stake"):
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")

        return errors

    @staticmethod
    def _validate_bet_schema(df: pd.DataFrame) -> list[str]:
        """書き込み前整合性検証 (D-20). Returns list of error strings (empty = valid)."""
        errors = PaperReconciler._validate_bet_schema_basic(df)

        if errors:
            return errors

        # schema_version == 2
        if not (df["schema_version"] == 2).all():
            errors.append("schema_version must be 2 for all rows")

        # stake > 0
        if (df["stake"] <= 0).any():
            errors.append("stake must be > 0 for all rows")

        # bet_id: non-NULL and unique
        if df["bet_id"].isna().any():
            errors.append("bet_id must be non-NULL")
        if df["bet_id"].nunique() != len(df):
            errors.append("bet_id must be unique")

        # pending rows: outcome NULL, payout NULL
        pending_mask = df["settlement_status"] == "pending"
        if pending_mask.any():
            pending = df[pending_mask]
            if pending["outcome"].notna().any():
                errors.append("Pending rows must have outcome=NULL")
            if pending["payout"].notna().any():
                errors.append("Pending rows must have payout=NULL")

        # settled rows: outcome non-NULL, payout >= 0
        settled_mask = df["settlement_status"] == "settled"
        if settled_mask.any():
            settled = df[settled_mask]
            if settled["outcome"].isna().any():
                errors.append("Settled rows must have outcome != NULL")
            if (settled["payout"] < 0).any():
                errors.append("Settled rows must have payout >= 0")

        return errors

    # ------------------------------------------------------------------
    # Core reconciliation
    # ------------------------------------------------------------------

    def reconcile(self, target_date: date) -> dict[str, Any]:
        """当日のレース結果を取得し、予測と照合。

        Returns:
            日次結果サマリー (n_settled, n_wins, roi, etc.)
        """
        # 1. Load bets.parquet (source of truth, D-08)
        if not self.bets_path.exists():
            logger.info("No bets file for %s", target_date)
            return self._empty_result(target_date)

        bets_df = pd.read_parquet(self.bets_path)

        # Old schema rejection (D-18)
        if "result" in bets_df.columns and "payout" not in bets_df.columns:
            raise ValueError(
                "Old schema detected in bets.parquet: 'result' column present without 'payout'. "
                "Migration not supported -- recreate bets from predictions."
            )

        if bets_df.empty:
            logger.info("No bets to reconcile for %s", target_date)
            return self._empty_result(target_date)

        # 2. Filter pending bets for target_date
        target_ts = pd.Timestamp(target_date)
        date_mask = bets_df["race_date"] == target_ts
        pending_mask = date_mask & (bets_df["settlement_status"] == "pending")
        pending = bets_df[pending_mask]

        if pending.empty:
            logger.info("No pending bets for %s", target_date)
            return self._compute_roi(bets_df, target_date)

        # 3. Fetch payouts from EveryDB2
        ymd = target_date.strftime("%Y%m%d")
        payouts_df = self.everydb2.get_payouts(ymd)

        if payouts_df.empty:
            logger.warning("No payout data for %s -- races may not have finished yet", ymd)
            return self._compute_roi(bets_df, target_date)

        # 4. Build payout maps
        win_map = build_win_payout_map(payouts_df)
        place_map = build_payout_map(payouts_df)
        wide_map = build_wide_payout_map(payouts_df)

        # 5. Settlement logic (D-11 order)
        n_settled = 0
        n_wins = 0

        for idx in pending.index:
            row = bets_df.loc[idx]
            race_id = str(row["race_id"])
            umaban = int(row["umaban"])
            bet_type = str(row["bet_type"])
            stake = float(row["stake"])

            # Select appropriate payout map
            if bet_type == "win":
                pmap = win_map
            elif bet_type == "wide":
                # Wide settlement requires partner umaban_b
                umaban_b_raw = row.get("umaban_b", 0)
                umaban_b = int(umaban_b_raw) if pd.notna(umaban_b_raw) else 0
                if umaban_b == 0:
                    logger.warning(
                        "Wide bet missing umaban_b for %s umaban=%d, keeping pending",
                        race_id, umaban,
                    )
                    continue
                lo, hi = min(umaban, umaban_b), max(umaban, umaban_b)
                wide_key = (race_id, lo, hi)
                # Check if race exists in payout data
                race_in_payouts = (
                    any(r == race_id for (r, *_rest) in wide_map.keys()) if wide_map else False
                )
                if not race_in_payouts:
                    race_in_payouts = any(r == race_id for (r, _) in win_map.keys()) if win_map else False
                if not race_in_payouts:
                    continue
                if wide_key in wide_map:
                    multiplier = wide_map[wide_key]
                    if multiplier <= 0:
                        logger.warning(
                            "Invalid wide payout multiplier %.4f for %s (%d,%d), keeping pending",
                            multiplier, race_id, lo, hi,
                        )
                        continue
                    payout = stake * multiplier
                    bets_df.at[idx, "outcome"] = "won"
                    bets_df.at[idx, "payout"] = payout
                    bets_df.at[idx, "settlement_status"] = "settled"
                    n_wins += 1
                else:
                    bets_df.at[idx, "outcome"] = "lost"
                    bets_df.at[idx, "payout"] = 0.0
                    bets_df.at[idx, "settlement_status"] = "settled"
                n_settled += 1
                continue
            else:
                pmap = place_map

            # Check if race exists in payout data
            race_in_payouts = any(r == race_id for (r, _) in pmap.keys()) if pmap else False
            # Also check win_map for race existence
            if not race_in_payouts:
                race_in_payouts = any(r == race_id for (r, _) in win_map.keys()) if win_map else False

            if not race_in_payouts:
                # Race not in payout data -> keep pending
                continue

            # Check if horse is in payout map
            key = (race_id, umaban)
            if key in pmap:
                multiplier = pmap[key]
                # Validate payout value (D-11 item 6: invalid payout -> keep pending)
                if multiplier <= 0:
                    logger.warning(
                        "Invalid payout multiplier %.4f for %s umaban=%d, keeping pending",
                        multiplier, race_id, umaban,
                    )
                    continue
                payout = stake * multiplier
                bets_df.at[idx, "outcome"] = "won"
                bets_df.at[idx, "payout"] = payout
                bets_df.at[idx, "settlement_status"] = "settled"
                n_wins += 1
            else:
                # Horse not in payout map -> lost
                bets_df.at[idx, "outcome"] = "lost"
                bets_df.at[idx, "payout"] = 0.0
                bets_df.at[idx, "settlement_status"] = "settled"

            n_settled += 1

        # 6. Atomic write (D-07)
        if n_settled > 0:
            self._atomic_write_parquet(bets_df, self.bets_path)
            logger.info("Settled %d bets (%d wins) for %s", n_settled, n_wins, target_date)

        return self._compute_roi(bets_df, target_date, n_settled, n_wins)

    def retry_pending(
        self, target_date: date, last_race_time: str | None = None
    ) -> dict[str, Any]:
        """Retry pending bets at intervals until timeout (D-06).

        Args:
            target_date: 対象日
            last_race_time: 最終レース発走時刻 (HH:MM) or None

        Returns:
            reconcile() result with exit_code=2 if pending remain.
        """
        start_epoch = time.monotonic()

        while True:
            result = self.reconcile(target_date)

            # Check if pending remain for this date
            if self.bets_path.exists():
                bets_df = pd.read_parquet(self.bets_path)
                target_ts = pd.Timestamp(target_date)
                pending_count = int(
                    ((bets_df["race_date"] == target_ts) & (bets_df["settlement_status"] == "pending")).sum()
                )
            else:
                pending_count = 0

            if pending_count == 0:
                result["exit_code"] = 0
                return result

            elapsed = time.monotonic() - start_epoch
            if elapsed >= self.retry_timeout:
                logger.warning(
                    "Retry timeout: %d bets still pending after %.0fs for %s",
                    pending_count, elapsed, target_date,
                )
                result["exit_code"] = 2
                result["n_pending"] = pending_count
                return result

            logger.info(
                "Retry: %d bets still pending for %s, sleeping %ds (elapsed %.0fs)",
                pending_count, target_date, self.retry_interval, elapsed,
            )
            time.sleep(self.retry_interval)

    # ------------------------------------------------------------------
    # ROI calculation
    # ------------------------------------------------------------------

    def _compute_roi(
        self,
        bets_df: pd.DataFrame,
        target_date: date,
        n_settled: int = 0,
        n_wins: int = 0,
        n_refunded: int = 0,
        n_voided: int = 0,
    ) -> dict[str, Any]:
        """累積統計を計算 (D-05: effective_stake = won + lost only)."""
        if bets_df.empty:
            return self._empty_result(target_date)

        total_bets = len(bets_df)

        # D-05: effective_stake excludes refunded/voided
        decidable = bets_df[bets_df["outcome"].isin(["won", "lost"])]
        effective_stake = float(decidable["stake"].sum()) if not decidable.empty else 0.0
        total_return = float(decidable["payout"].sum()) if not decidable.empty else 0.0
        cumulative_roi = total_return / effective_stake if effective_stake > 0 else 0.0
        net_profit = total_return - effective_stake

        total_wins = int((bets_df["outcome"] == "won").sum())
        n_pending = int((bets_df["settlement_status"] == "pending").sum())
        n_refunded = int((bets_df["outcome"] == "refunded").sum())
        n_voided = int((bets_df["outcome"] == "voided").sum())

        # Max drawdown (from bankroll_after if available)
        max_dd = 0.0
        bankroll_val = 100000.0
        if "bankroll_after" in bets_df.columns and not bets_df["bankroll_after"].empty:
            bankroll_series = bets_df["bankroll_after"]
            bankroll_val = float(bankroll_series.iloc[-1])
            peak = bankroll_series.cummax()
            dd = (peak - bankroll_series) / peak
            max_dd = float(dd.max()) if not dd.empty else 0.0

        return {
            "date": target_date.isoformat(),
            "n_bets": total_bets,
            "n_wins": total_wins,
            "n_pending": n_pending,
            "n_refunded": n_refunded,
            "n_voided": n_voided,
            "total_stake": float(bets_df["stake"].sum()),
            "effective_stake": effective_stake,
            "total_return": total_return,
            "net_profit": net_profit,
            "cumulative_roi": cumulative_roi,
            "max_dd": max_dd,
            "bankroll": bankroll_val,
            "n_settled": n_settled,
            "n_new_wins": n_wins,
        }

    def _empty_result(self, target_date: date) -> dict[str, Any]:
        return {
            "date": target_date.isoformat(),
            "n_bets": 0,
            "n_wins": 0,
            "n_pending": 0,
            "n_refunded": 0,
            "n_voided": 0,
            "total_stake": 0.0,
            "effective_stake": 0.0,
            "total_return": 0.0,
            "net_profit": 0.0,
            "cumulative_roi": 0.0,
            "max_dd": 0.0,
            "bankroll": 100000.0,
            "n_settled": 0,
            "n_new_wins": 0,
        }
