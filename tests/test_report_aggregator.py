"""Tests for PaperTradingReportAggregator (Task 2)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from paper_trading.report_aggregator import (
    PaperTradingReportAggregator,
    _iso_week_range,
)


def _make_bets_df(
    rows: list[dict] | None = None,
) -> pd.DataFrame:
    """Create a v2-schema bets DataFrame for testing."""
    if rows is None:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "race_date" in df.columns and not df.empty:
        df["race_date"] = pd.to_datetime(df["race_date"])
    return df


def _sample_bets(target_date: str = "2024-01-15") -> pd.DataFrame:
    """Standard test bets covering win/place, won/lost/pending."""
    return _make_bets_df(
        [
            {
                "bet_id": "b1",
                "race_id": "r1",
                "race_date": target_date,
                "bet_type": "win",
                "umaban": 1,
                "stake": 100.0,
                "odds": 5.0,
                "session_id": "s1",
                "schema_version": "2.0",
                "settlement_status": "settled",
                "outcome": "won",
                "payout": 500.0,
            },
            {
                "bet_id": "b2",
                "race_id": "r2",
                "race_date": target_date,
                "bet_type": "win",
                "umaban": 3,
                "stake": 100.0,
                "odds": 3.0,
                "session_id": "s1",
                "schema_version": "2.0",
                "settlement_status": "settled",
                "outcome": "lost",
                "payout": 0.0,
            },
            {
                "bet_id": "b3",
                "race_id": "r3",
                "race_date": target_date,
                "bet_type": "place",
                "umaban": 5,
                "stake": 100.0,
                "odds": 2.0,
                "session_id": "s1",
                "schema_version": "2.0",
                "settlement_status": "pending",
                "outcome": "pending",
                "payout": 0.0,
            },
            {
                "bet_id": "b4",
                "race_id": "r4",
                "race_date": target_date,
                "bet_type": "place",
                "umaban": 7,
                "stake": 100.0,
                "odds": 1.5,
                "session_id": "s2",
                "schema_version": "2.0",
                "settlement_status": "settled",
                "outcome": "won",
                "payout": 150.0,
            },
        ]
    )


class TestAggregateDaily:
    """Daily aggregation tests."""

    def test_daily_returns_roi_hit_rate_bet_counts(self, tmp_path: Path) -> None:
        """Test 1: aggregate_daily returns ROI, hit_rate, bet counts, pending count."""
        bets_path = tmp_path / "bets.parquet"
        df = _sample_bets()
        df.to_parquet(bets_path, index=False)

        agg = PaperTradingReportAggregator(bets_path, tmp_path / "out")
        result = agg.aggregate_daily(date(2024, 1, 15))

        assert result["total_bets"] == 4
        assert result["n_won"] == 2
        assert result["n_lost"] == 1
        assert result["n_pending"] == 1
        assert result["roi"] == pytest.approx(650.0 / 300.0)
        assert result["hit_rate"] == pytest.approx(2.0 / 3.0)

    def test_daily_effective_stake_excludes_refunded(self, tmp_path: Path) -> None:
        """Test 2: effective_stake = won + lost only (excludes refunded/voided, per D-05)."""
        bets_path = tmp_path / "bets.parquet"
        df = _make_bets_df(
            [
                {
                    "bet_id": "b1",
                    "race_id": "r1",
                    "race_date": "2024-01-15",
                    "bet_type": "win",
                    "umaban": 1,
                    "stake": 100.0,
                    "odds": 5.0,
                    "session_id": "s1",
                    "schema_version": "2.0",
                    "settlement_status": "settled",
                    "outcome": "won",
                    "payout": 500.0,
                },
                {
                    "bet_id": "b2",
                    "race_id": "r2",
                    "race_date": "2024-01-15",
                    "bet_type": "win",
                    "umaban": 2,
                    "stake": 100.0,
                    "odds": 3.0,
                    "session_id": "s1",
                    "schema_version": "2.0",
                    "settlement_status": "settled",
                    "outcome": "refunded",
                    "payout": 0.0,
                },
                {
                    "bet_id": "b3",
                    "race_id": "r3",
                    "race_date": "2024-01-15",
                    "bet_type": "win",
                    "umaban": 3,
                    "stake": 100.0,
                    "odds": 3.0,
                    "session_id": "s1",
                    "schema_version": "2.0",
                    "settlement_status": "settled",
                    "outcome": "voided",
                    "payout": 0.0,
                },
            ]
        )
        df.to_parquet(bets_path, index=False)

        agg = PaperTradingReportAggregator(bets_path, tmp_path / "out")
        result = agg.aggregate_daily(date(2024, 1, 15))

        # effective_stake should only count won+lost (100), not refunded/voided
        assert result["effective_stake"] == pytest.approx(100.0)
        assert result["n_refunded"] == 1
        assert result["n_voided"] == 1

    def test_daily_empty_returns_zero_stats_no_data(self, tmp_path: Path) -> None:
        """Test 3: empty bets returns zero-stats with data_completeness='no_data'."""
        agg = PaperTradingReportAggregator(tmp_path / "nonexistent.parquet", tmp_path / "out")
        result = agg.aggregate_daily(date(2024, 1, 15))

        assert result["total_bets"] == 0
        assert result["roi"] == 0.0
        assert result["hit_rate"] == 0.0
        assert result["data_completeness"] == "no_data"


class TestAggregateWeekly:
    """Weekly aggregation tests."""

    def test_weekly_iso_week_range(self) -> None:
        """Test 4: aggregate_weekly with ISO week 2024-W01 returns correct Monday-Sunday range."""
        start, end = _iso_week_range(2024, 1)
        # ISO week 1 of 2024 starts on Monday 2024-01-01
        assert start == date(2024, 1, 1)
        assert end == date(2024, 1, 7)

    def test_weekly_aggregation(self, tmp_path: Path) -> None:
        """Weekly aggregation returns correct stats for ISO week range."""
        bets_path = tmp_path / "bets.parquet"
        # 2024-W01 = Jan 1-7
        df = _make_bets_df(
            [
                {
                    "bet_id": "b1",
                    "race_id": "r1",
                    "race_date": "2024-01-03",
                    "bet_type": "win",
                    "umaban": 1,
                    "stake": 100.0,
                    "odds": 5.0,
                    "session_id": "s1",
                    "schema_version": "2.0",
                    "settlement_status": "settled",
                    "outcome": "won",
                    "payout": 500.0,
                },
            ]
        )
        df.to_parquet(bets_path, index=False)

        agg = PaperTradingReportAggregator(bets_path, tmp_path / "out")
        result = agg.aggregate_weekly(2024, 1)

        assert result["total_bets"] == 1
        assert result["n_won"] == 1
        assert result["period"] == "2024-W01"

    def test_weekly_year_end_boundary(self) -> None:
        """Test 8: 2025-12-30 belongs to ISO week 1 of 2026."""
        # 2025-12-30 is a Tuesday, and it's in ISO week 1 of 2026
        d = date(2025, 12, 30)
        iso_cal = d.isocalendar()
        assert iso_cal[0] == 2026
        assert iso_cal[1] == 1

        start, end = _iso_week_range(2026, 1)
        assert start == date(2025, 12, 29)  # Monday
        assert end == date(2026, 1, 4)  # Sunday


class TestAggregateByTarget:
    """Per-target (win/place) aggregation tests."""

    def test_by_target_separates_win_place(self, tmp_path: Path) -> None:
        """Test 5: aggregate_by_target separates win and place ROI/hit_rate/bet_count."""
        bets_path = tmp_path / "bets.parquet"
        df = _sample_bets()
        df.to_parquet(bets_path, index=False)

        agg = PaperTradingReportAggregator(bets_path, tmp_path / "out")
        result = agg.aggregate_by_target(target_date=date(2024, 1, 15))

        assert "win" in result["targets"]
        assert "place" in result["targets"]

        win_stats = result["targets"]["win"]
        place_stats = result["targets"]["place"]

        assert win_stats["total_bets"] == 2  # b1 (won) + b2 (lost)
        assert place_stats["total_bets"] == 2  # b3 (pending) + b4 (won)

    def test_by_target_includes_model_identity(self, tmp_path: Path) -> None:
        """Test 6: aggregate_all includes model_identity from session_manifest."""
        bets_path = tmp_path / "bets.parquet"
        df = _sample_bets()
        df.to_parquet(bets_path, index=False)

        manifest = SimpleNamespace(
            model_run_id="mlflow-run-123",
            training_start="2020-01-01",
            training_end="2023-12-31",
            manifest_hash="abc123",
        )
        agg = PaperTradingReportAggregator(bets_path, tmp_path / "out", session_manifest=manifest)
        result = agg.aggregate_all(date(2024, 1, 15))

        assert result["daily"]["model_identity"]["model_run_id"] == "mlflow-run-123"
        assert result["weekly"]["model_identity"]["model_run_id"] == "mlflow-run-123"
        assert result["target"]["model_identity"]["manifest_hash"] == "abc123"


class TestDailyPending:
    """Daily pending/unsettled fields tests."""

    def test_daily_includes_pending_fields(self, tmp_path: Path) -> None:
        """Test 7: aggregate_daily includes pending_count, unsettled_stake, data_completeness."""
        bets_path = tmp_path / "bets.parquet"
        df = _sample_bets()
        df.to_parquet(bets_path, index=False)

        agg = PaperTradingReportAggregator(bets_path, tmp_path / "out")
        result = agg.aggregate_daily(date(2024, 1, 15))

        assert result["pending_count"] == 1
        assert result["unsettled_stake"] == pytest.approx(100.0)
        assert result["data_completeness"] == "partial"


class TestSaveOutputs:
    """JSON file output tests."""

    def test_save_outputs_creates_correct_structure(self, tmp_path: Path) -> None:
        """Test 9: save_outputs creates correct directory structure and files."""
        bets_path = tmp_path / "bets.parquet"
        df = _sample_bets()
        df.to_parquet(bets_path, index=False)

        output_dir = tmp_path / "reports"
        agg = PaperTradingReportAggregator(bets_path, output_dir)
        paths = agg.save_outputs(date(2024, 1, 15))

        # daily_summary/YYYY/YYYY-MM-DD.json
        assert paths["daily"] == output_dir / "daily_summary" / "2024" / "2024-01-15.json"
        assert paths["daily"].exists()

        # weekly_summary/{iso_year}/W{iso_week:02d}.json
        # 2024-01-15 is ISO week 3
        assert paths["weekly"].exists()
        assert "weekly_summary" in str(paths["weekly"])

        # target_summary/YYYY-MM-DD.json
        assert paths["target"] == output_dir / "target_summary" / "2024-01-15.json"
        assert paths["target"].exists()

        # target_summary/latest.json
        assert paths["latest"] == output_dir / "target_summary" / "latest.json"
        assert paths["latest"].exists()

    def test_all_json_outputs_have_common_fields(self, tmp_path: Path) -> None:
        """Test 10: All JSON outputs include schema_version, period, generated_at, session_ids."""
        bets_path = tmp_path / "bets.parquet"
        df = _sample_bets()
        df.to_parquet(bets_path, index=False)

        output_dir = tmp_path / "reports"
        agg = PaperTradingReportAggregator(bets_path, output_dir)
        paths = agg.save_outputs(date(2024, 1, 15))

        for name, path in paths.items():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            assert "schema_version" in data, f"{name} missing schema_version"
            assert data["schema_version"] == "2.0", f"{name} wrong schema_version"
            assert "period" in data, f"{name} missing period"
            assert "generated_at" in data, f"{name} missing generated_at"
            assert "session_ids" in data, f"{name} missing session_ids"
