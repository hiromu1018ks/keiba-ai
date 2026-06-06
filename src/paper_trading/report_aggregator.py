"""PaperTradingReportAggregator: daily/weekly/target statistics from bets.parquet (D-11)."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _iso_week_range(iso_year: int, iso_week: int) -> tuple[date, date]:
    """Compute Monday-Sunday date range for an ISO week.

    Uses Jan 4 based calculation: Jan 4 is always in ISO week 1.
    """
    jan4 = date(iso_year, 1, 4)
    start_of_week1 = jan4 - timedelta(days=jan4.weekday())
    week_start = start_of_week1 + timedelta(weeks=iso_week - 1)
    week_end = week_start + timedelta(days=6)
    return week_start, week_end


def _atomic_write_json(data: dict[str, Any], target: Path) -> None:
    """Atomic JSON write via tempfile.mkstemp + os.replace with Windows retry."""
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, indent=2, ensure_ascii=False, default=str)

    fd, tmp_path = tempfile.mkstemp(
        suffix=".json",
        prefix=".tmp_report_",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                os.replace(tmp_path, str(target))
                return
            except PermissionError:
                if attempt == max_retries - 1:
                    raise
                time.sleep(0.1)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _common_fields(
    period: str,
    session_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Generate common JSON output fields per D-13."""
    from datetime import datetime

    return {
        "schema_version": "2.0",
        "period": period,
        "generated_at": datetime.now().isoformat(),
        "session_ids": session_ids or [],
    }


def _validate_bet_schema(df: pd.DataFrame) -> list[str]:
    """Validate bets.parquet schema (reject old v1 schema)."""
    errors: list[str] = []
    if "result" in df.columns and "payout" not in df.columns:
        errors.append("Old schema detected: 'result' column present without 'payout'")
        return errors
    required = ("schema_version", "settlement_status", "outcome", "payout", "bet_id", "stake")
    for col in required:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    return errors


class PaperTradingReportAggregator:
    """Aggregate paper trading statistics from bets.parquet (D-11).

    Reads bets.parquet as the sole cumulative history source (D-10).
    Produces daily, weekly, and per-target JSON summaries.
    """

    def __init__(
        self,
        bets_path: Path,
        output_dir: Path,
        session_manifest: Any | None = None,
    ) -> None:
        self._bets_path = bets_path
        self._output_dir = output_dir
        self._session_manifest = session_manifest

    def _model_identity(self) -> dict[str, str]:
        """Extract model identity from session_manifest per D-19."""
        if self._session_manifest is None:
            return {}
        return {
            "model_run_id": getattr(self._session_manifest, "model_run_id", ""),
            "training_start": getattr(self._session_manifest, "training_start", ""),
            "training_end": getattr(self._session_manifest, "training_end", ""),
            "manifest_hash": getattr(self._session_manifest, "manifest_hash", ""),
        }

    def _load_bets(self) -> pd.DataFrame:
        """Load bets.parquet with schema validation."""
        if not self._bets_path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(self._bets_path)
        if df.empty:
            return df
        errors = _validate_bet_schema(df)
        if errors:
            raise ValueError(f"Schema validation failed: {'; '.join(errors)}")
        return df

    def _base_stats(self, bets_df: pd.DataFrame) -> dict[str, Any]:
        """Compute base statistics from a filtered bets DataFrame.

        D-05: effective_stake = won + lost only (excludes refunded/voided).
        """
        if bets_df.empty:
            return {
                "total_bets": 0,
                "n_won": 0,
                "n_lost": 0,
                "n_pending": 0,
                "n_refunded": 0,
                "n_voided": 0,
                "effective_stake": 0.0,
                "total_return": 0.0,
                "roi": 0.0,
                "hit_rate": 0.0,
                "decidable_count": 0,
            }

        total_bets = len(bets_df)
        n_won = int((bets_df["outcome"] == "won").sum())
        n_lost = int((bets_df["outcome"] == "lost").sum())
        n_pending = int((bets_df["settlement_status"] == "pending").sum())
        n_refunded = int((bets_df["outcome"] == "refunded").sum())
        n_voided = int((bets_df["outcome"] == "voided").sum())

        # D-05: effective_stake = won + lost only
        decidable = bets_df[bets_df["outcome"].isin(["won", "lost"])]
        effective_stake = float(decidable["stake"].sum()) if not decidable.empty else 0.0
        total_return = float(decidable["payout"].sum()) if not decidable.empty else 0.0
        decidable_count = len(decidable)

        roi = total_return / effective_stake if effective_stake > 0 else 0.0
        hit_rate = n_won / decidable_count if decidable_count > 0 else 0.0

        return {
            "total_bets": total_bets,
            "n_won": n_won,
            "n_lost": n_lost,
            "n_pending": n_pending,
            "n_refunded": n_refunded,
            "n_voided": n_voided,
            "effective_stake": effective_stake,
            "total_return": total_return,
            "roi": roi,
            "hit_rate": hit_rate,
            "decidable_count": decidable_count,
        }

    def _get_session_ids(self, bets_df: pd.DataFrame) -> list[str]:
        """Extract unique session_ids from bets DataFrame."""
        if bets_df.empty or "session_id" not in bets_df.columns:
            return []
        return sorted(bets_df["session_id"].unique().tolist())

    def aggregate_daily(self, target_date: date) -> dict[str, Any]:
        """Aggregate statistics for a single day."""
        bets_df = self._load_bets()
        target_ts = pd.Timestamp(target_date)
        daily = bets_df[bets_df["race_date"] == target_ts] if not bets_df.empty else pd.DataFrame()

        stats = self._base_stats(daily)

        # D-14: pending_count, unsettled_stake, data_completeness
        pending_count = stats["n_pending"]
        unsettled_stake = (
            float(daily[daily["settlement_status"] == "pending"]["stake"].sum())
            if not daily.empty and "settlement_status" in daily.columns
            else 0.0
        )

        if daily.empty:
            data_completeness = "no_data"
        elif pending_count == 0:
            data_completeness = "complete"
        else:
            data_completeness = "partial"

        result: dict[str, Any] = {
            **_common_fields(str(target_date), self._get_session_ids(daily)),
            **stats,
            "pending_count": pending_count,
            "unsettled_stake": unsettled_stake,
            "data_completeness": data_completeness,
            "model_identity": self._model_identity(),
        }
        return result

    def aggregate_weekly(self, iso_year: int, iso_week: int) -> dict[str, Any]:
        """Aggregate statistics for an ISO week."""
        week_start, week_end = _iso_week_range(iso_year, iso_week)
        bets_df = self._load_bets()

        if not bets_df.empty:
            start_ts = pd.Timestamp(week_start)
            end_ts = pd.Timestamp(week_end)
            weekly = bets_df[(bets_df["race_date"] >= start_ts) & (bets_df["race_date"] <= end_ts)]
        else:
            weekly = pd.DataFrame()

        stats = self._base_stats(weekly)
        period = f"{iso_year}-W{iso_week:02d}"

        result: dict[str, Any] = {
            **_common_fields(period, self._get_session_ids(weekly)),
            **stats,
            "week_start": str(week_start),
            "week_end": str(week_end),
            "model_identity": self._model_identity(),
        }
        return result

    def aggregate_by_target(self, target_date: date | None = None) -> dict[str, Any]:
        """Aggregate statistics grouped by bet_type (win/place)."""
        bets_df = self._load_bets()

        if target_date is not None and not bets_df.empty:
            target_ts = pd.Timestamp(target_date)
            bets_df = bets_df[bets_df["race_date"] == target_ts]

        session_ids = self._get_session_ids(bets_df)

        targets: dict[str, dict[str, Any]] = {}
        if not bets_df.empty and "bet_type" in bets_df.columns:
            for bet_type, group in bets_df.groupby("bet_type"):
                targets[bet_type] = self._base_stats(group)

        result: dict[str, Any] = {
            **_common_fields(
                str(target_date) if target_date else "all",
                session_ids,
            ),
            "targets": targets,
            "model_identity": self._model_identity(),
        }
        return result

    def aggregate_all(self, target_date: date) -> dict[str, Any]:
        """Run all three aggregations for a target date."""
        iso_cal = target_date.isocalendar()
        return {
            "daily": self.aggregate_daily(target_date),
            "weekly": self.aggregate_weekly(iso_cal[0], iso_cal[1]),
            "target": self.aggregate_by_target(target_date),
        }

    def save_outputs(self, target_date: date) -> dict[str, Path]:
        """Save all aggregation outputs as JSON files per D-13.

        Returns dict of {type: path} for verification.
        """
        all_results = self.aggregate_all(target_date)
        iso_cal = target_date.isocalendar()

        paths: dict[str, Path] = {}

        # daily_summary/YYYY/YYYY-MM-DD.json
        daily_dir = self._output_dir / "daily_summary" / str(target_date.year)
        daily_path = daily_dir / f"{target_date}.json"
        _atomic_write_json(all_results["daily"], daily_path)
        paths["daily"] = daily_path

        # weekly_summary/{iso_year}/W{iso_week:02d}.json
        weekly_dir = self._output_dir / "weekly_summary" / str(iso_cal[0])
        weekly_path = weekly_dir / f"W{iso_cal[1]:02d}.json"
        _atomic_write_json(all_results["weekly"], weekly_path)
        paths["weekly"] = weekly_path

        # target_summary/YYYY-MM-DD.json
        target_dir = self._output_dir / "target_summary"
        target_path = target_dir / f"{target_date}.json"
        _atomic_write_json(all_results["target"], target_path)
        paths["target"] = target_path

        # target_summary/latest.json
        latest_path = target_dir / "latest.json"
        _atomic_write_json(all_results["target"], latest_path)
        paths["latest"] = latest_path

        return paths
