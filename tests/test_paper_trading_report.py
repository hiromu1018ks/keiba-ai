"""Tests for PaperTradingReport as pure HTML renderer (Plan 03, D-12)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest


def _make_aggregate_results(
    *,
    has_pending: bool = False,
    data_completeness: str = "complete",
    roi: float = 1.15,
    n_bets: int = 10,
    hit_rate: float = 0.4,
) -> dict:
    """Build a mock aggregate_results dict matching Aggregator.aggregate_all() output."""
    daily = {
        "schema_version": "2.0",
        "period": "2026-06-06",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session_ids": ["sess-001"],
        "total_bets": n_bets,
        "n_won": int(n_bets * hit_rate),
        "n_lost": n_bets - int(n_bets * hit_rate),
        "n_pending": 2 if has_pending else 0,
        "n_refunded": 0,
        "n_voided": 0,
        "effective_stake": 1000.0,
        "total_return": 1000.0 * roi,
        "roi": roi,
        "hit_rate": hit_rate,
        "decidable_count": n_bets - (2 if has_pending else 0),
        "pending_count": 2 if has_pending else 0,
        "unsettled_stake": 200.0 if has_pending else 0.0,
        "data_completeness": data_completeness,
        "model_identity": {
            "model_run_id": "mlflow-run-abc123",
            "training_start": "2020-01-01",
            "training_end": "2023-12-31",
            "manifest_hash": "sha256hash123",
        },
    }
    weekly = {
        "schema_version": "2.0",
        "period": "2026-W23",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session_ids": ["sess-001"],
        "total_bets": n_bets,
        "roi": roi,
        "hit_rate": hit_rate,
        "effective_stake": 1000.0,
        "total_return": 1000.0 * roi,
        "week_start": "2026-06-01",
        "week_end": "2026-06-07",
        "model_identity": daily["model_identity"],
    }
    target = {
        "schema_version": "2.0",
        "period": "2026-06-06",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session_ids": ["sess-001"],
        "targets": {
            "win": {
                "total_bets": 6,
                "n_won": 2,
                "n_lost": 4,
                "n_pending": 0,
                "effective_stake": 600.0,
                "total_return": 690.0,
                "roi": 1.15,
                "hit_rate": 0.333,
            },
            "place": {
                "total_bets": 4,
                "n_won": 2,
                "n_lost": 2,
                "n_pending": 0,
                "effective_stake": 400.0,
                "total_return": 460.0,
                "roi": 1.15,
                "hit_rate": 0.5,
            },
        },
        "model_identity": daily["model_identity"],
    }
    return {"daily": daily, "weekly": weekly, "target": target}


def _make_bets() -> list[dict]:
    """Build mock bets list with new schema fields."""
    return [
        {
            "race_date": "2026-06-06",
            "race_id": "202606060611",
            "horse_name": "Horse A",
            "umaban": 1,
            "odds": 5.3,
            "ev": 1.15,
            "stake": 100.0,
            "settlement_status": "settled",
            "outcome": "won",
            "payout": 530.0,
            "bet_type": "win",
            "bet_id": "abc123",
        },
        {
            "race_date": "2026-06-06",
            "race_id": "202606060611",
            "horse_name": "Horse B",
            "umaban": 3,
            "odds": 2.1,
            "ev": 1.05,
            "stake": 100.0,
            "settlement_status": "settled",
            "outcome": "lost",
            "payout": 0.0,
            "bet_type": "place",
            "bet_id": "def456",
        },
        {
            "race_date": "2026-06-06",
            "race_id": "202606060812",
            "horse_name": "Horse C",
            "umaban": 7,
            "odds": 8.5,
            "ev": 1.20,
            "stake": 100.0,
            "settlement_status": "pending",
            "outcome": "",
            "payout": 0.0,
            "bet_type": "win",
            "bet_id": "ghi789",
        },
    ]


class TestReportRendererNewSchema:
    """Tests for PaperTradingReport as pure HTML renderer (D-12)."""

    def test_generate_with_aggregator_output_no_keyerror(self, tmp_path: Path) -> None:
        """Test 1: generate() with Aggregator output dict renders HTML without KeyError."""
        from paper_trading.report import PaperTradingReport

        report = PaperTradingReport(tmp_path)
        aggregate_results = _make_aggregate_results()
        bets = _make_bets()

        result_path = report.generate(aggregate_results=aggregate_results, bets=bets)

        assert result_path.exists()
        html = result_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html
        assert "Paper Trading Report" in html

    def test_old_methods_removed(self, tmp_path: Path) -> None:
        """Test 2: _derive_fields and _compute_monthly_stats are removed."""
        from paper_trading.report import PaperTradingReport

        report = PaperTradingReport(tmp_path)
        assert not hasattr(report, "_derive_fields"), (
            "_derive_fields should be removed (D-12)"
        )
        assert not hasattr(report, "_compute_monthly_stats"), (
            "_compute_monthly_stats should be removed (D-12)"
        )

    def test_html_contains_cumulative_roi(self, tmp_path: Path) -> None:
        """Test 3: HTML contains Cumulative ROI KPI from aggregator daily stats."""
        from paper_trading.report import PaperTradingReport

        report = PaperTradingReport(tmp_path)
        aggregate_results = _make_aggregate_results(roi=1.25)
        bets = _make_bets()

        result_path = report.generate(aggregate_results=aggregate_results, bets=bets)
        html = result_path.read_text(encoding="utf-8")

        assert "Cumulative ROI" in html
        # 125.0% = 1.25
        assert "125.0%" in html

    def test_html_contains_model_identity(self, tmp_path: Path) -> None:
        """Test 4: HTML contains model identity section (D-19)."""
        from paper_trading.report import PaperTradingReport

        report = PaperTradingReport(tmp_path)
        aggregate_results = _make_aggregate_results()
        bets = _make_bets()

        result_path = report.generate(aggregate_results=aggregate_results, bets=bets)
        html = result_path.read_text(encoding="utf-8")

        # D-19: model identity visible in report footer
        assert "mlflow-run-abc123" in html
        assert "2020-01-01" in html
        assert "2023-12-31" in html
        assert "sha256hash123" in html

    def test_html_bet_table_uses_new_schema(self, tmp_path: Path) -> None:
        """Test 5: HTML bet table uses settlement_status and outcome columns."""
        from paper_trading.report import PaperTradingReport

        report = PaperTradingReport(tmp_path)
        aggregate_results = _make_aggregate_results()
        bets = _make_bets()

        result_path = report.generate(aggregate_results=aggregate_results, bets=bets)
        html = result_path.read_text(encoding="utf-8")

        # New schema: settlement_status and outcome, not old "result"
        assert "settled" in html
        assert "won" in html
        assert "lost" in html
        assert "pending" in html
        # P/L column with profit/loss values from payout - stake
        assert "430" in html or "530" in html  # payout for won bet

    def test_html_shows_pending_count(self, tmp_path: Path) -> None:
        """Test 6: HTML shows pending count and unsettled stake when present."""
        from paper_trading.report import PaperTradingReport

        report = PaperTradingReport(tmp_path)
        aggregate_results = _make_aggregate_results(has_pending=True)
        bets = _make_bets()

        result_path = report.generate(aggregate_results=aggregate_results, bets=bets)
        html = result_path.read_text(encoding="utf-8")

        # pending_count=2 from mock data
        assert "Pending" in html
        assert "200" in html  # unsettled_stake

    def test_generate_returns_path(self, tmp_path: Path) -> None:
        """Test 7: generate() writes to report.html and returns Path."""
        from paper_trading.report import PaperTradingReport

        report = PaperTradingReport(tmp_path)
        aggregate_results = _make_aggregate_results()
        bets = _make_bets()

        result_path = report.generate(aggregate_results=aggregate_results, bets=bets)

        assert result_path == tmp_path / "report.html"
        assert result_path.exists()
        assert result_path.stat().st_size > 0

    def test_generate_empty_results(self, tmp_path: Path) -> None:
        """Test 8: generate() with empty aggregator results (no_data) renders without error."""
        from paper_trading.report import PaperTradingReport

        report = PaperTradingReport(tmp_path)
        aggregate_results = _make_aggregate_results(
            roi=0.0, n_bets=0, hit_rate=0.0, data_completeness="no_data"
        )

        result_path = report.generate(aggregate_results=aggregate_results, bets=[])

        assert result_path.exists()
        html = result_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html
