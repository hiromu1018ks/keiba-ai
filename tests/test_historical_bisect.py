"""HistoricalBisect unit tests (BISECT-01 per D-05)

Tests the auxiliary v1.7->v2.0 historical artifact comparison that performs
lightweight analysis of when ROI degradation started between Phase 34 and Phase 38.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from backtest.historical_bisect import HistoricalBisect, HistoricalBisectResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_multi_year_result() -> dict:
    """Synthetic multi_year_result.json."""
    return {
        "overall": {
            "total_bets": 6662,
            "roi": 0.9244,
            "best_year": 2024,
            "worst_year": 2025,
        },
        "years": {
            "2024": {
                "total_bets": 3327,
                "roi": 0.9372,
                "hit_rate": 0.28,
            },
            "2025": {
                "total_bets": 3335,
                "roi": 0.9116,
                "hit_rate": 0.26,
            },
        },
    }


def _make_oof_df(n_rows: int = 100) -> pd.DataFrame:
    """Synthetic OOF predictions DataFrame."""
    np.random.seed(42)
    return pd.DataFrame({
        "race_id": [f"2024{i % 20:02d}010101" for i in range(n_rows)],
        "umaban": [i % 18 + 1 for i in range(n_rows)],
        "p_win_oof": np.random.uniform(0.01, 0.5, n_rows),
        "kakuteijyuni": np.random.randint(1, 18, n_rows),
        "surface": ["turf" if i % 2 == 0 else "dirt" for i in range(n_rows)],
        "fold_year": [2024] * n_rows,
    })


def _setup_input_dir(
    tmp_path: Path,
    *,
    multi_year_result: dict | None = None,
    oof_df: pd.DataFrame | None = None,
) -> Path:
    """Set up input directory with fixture data."""
    input_dir = tmp_path / "backtest_input"
    input_dir.mkdir(parents=True, exist_ok=True)

    if multi_year_result is None:
        multi_year_result = _make_multi_year_result()

    (input_dir / "multi_year_result.json").write_text(
        json.dumps(multi_year_result, indent=2), encoding="utf-8"
    )

    if oof_df is None:
        oof_df = _make_oof_df()

    oof_path = tmp_path / "oof" / "oof_predictions.parquet"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_parquet(oof_path, index=False)

    return input_dir


# ---------------------------------------------------------------------------
# Test 1: Initialization and git tag detection
# ---------------------------------------------------------------------------


class TestInitialization:
    """Test HistoricalBisect initializes and identifies available tags."""

    @patch("subprocess.run")
    def test_initializes_and_identifies_tags(self, mock_run, tmp_path: Path) -> None:
        # Mock git tag -l to return v1.7 and v2.0
        mock_run.return_value = MagicMock(
            stdout="v1.0\nv1.7\nv2.0\nv2.1\n",
            returncode=0,
        )

        input_dir = _setup_input_dir(tmp_path)
        hb = HistoricalBisect(
            input_dir=input_dir,
            oof_path=tmp_path / "oof" / "oof_predictions.parquet",
        )

        assert "v1.7" in hb.available_tags
        assert "v2.0" in hb.available_tags

    @patch("subprocess.run")
    def test_handles_no_git(self, mock_run, tmp_path: Path) -> None:
        """Gracefully handles missing git or no tags."""
        mock_run.side_effect = FileNotFoundError("git not found")

        input_dir = _setup_input_dir(tmp_path)
        hb = HistoricalBisect(
            input_dir=input_dir,
            oof_path=tmp_path / "oof" / "oof_predictions.parquet",
        )

        assert hb.available_tags == []


# ---------------------------------------------------------------------------
# Test 2: compare_phase_artifacts()
# ---------------------------------------------------------------------------


class TestComparePhaseArtifacts:
    """Test compare_phase_artifacts() extracts metrics from local artifacts."""

    def test_extracts_roi_bet_count_hit_rate(self, tmp_path: Path) -> None:
        input_dir = _setup_input_dir(tmp_path)
        hb = HistoricalBisect(
            input_dir=input_dir,
            oof_path=tmp_path / "oof" / "oof_predictions.parquet",
        )

        result = hb.compare_phase_artifacts()

        assert "current_baseline" in result
        assert result["current_baseline"]["overall_roi"] == 0.9244
        assert result["current_baseline"]["total_bets"] == 6662

    def test_includes_per_year_breakdown(self, tmp_path: Path) -> None:
        input_dir = _setup_input_dir(tmp_path)
        hb = HistoricalBisect(
            input_dir=input_dir,
            oof_path=tmp_path / "oof" / "oof_predictions.parquet",
        )

        result = hb.compare_phase_artifacts()

        assert "per_year" in result
        assert "2024" in result["per_year"]
        assert result["per_year"]["2024"]["roi"] == 0.9372


# ---------------------------------------------------------------------------
# Test 3: compare_oof_metrics()
# ---------------------------------------------------------------------------


class TestCompareOOFMetrics:
    """Test compare_oof_metrics() loads OOF and computes per-fold metrics."""

    def test_computes_ic_brier_ece(self, tmp_path: Path) -> None:
        input_dir = _setup_input_dir(tmp_path)
        hb = HistoricalBisect(
            input_dir=input_dir,
            oof_path=tmp_path / "oof" / "oof_predictions.parquet",
        )

        result = hb.compare_oof_metrics()

        assert "current_oof" in result
        assert "ic" in result["current_oof"]
        assert "brier" in result["current_oof"]
        assert "ece" in result["current_oof"]


# ---------------------------------------------------------------------------
# Test 4: run_historical_comparison()
# ---------------------------------------------------------------------------


class TestRunHistoricalComparison:
    """Test run_historical_comparison() returns complete HistoricalBisectResult."""

    @patch("subprocess.run")
    def test_returns_historical_bisect_result(
        self, mock_run, tmp_path: Path
    ) -> None:
        # Mock git log between tags
        mock_run.return_value = MagicMock(
            stdout="abc1234 feat(35): add haron features\ndef5678 fix(36): "
            "MarketModel wiring fix\nghi9012 feat(37): OOF health validator\n",
            returncode=0,
        )

        input_dir = _setup_input_dir(tmp_path)
        hb = HistoricalBisect(
            input_dir=input_dir,
            oof_path=tmp_path / "oof" / "oof_predictions.parquet",
        )
        result = hb.run_historical_comparison()

        assert isinstance(result, HistoricalBisectResult)
        assert result.baseline_metrics is not None
        assert result.v17_reference_roi == 0.978
        assert result.total_degradation > 0
        assert isinstance(result.estimated_degradation_phase, str)
        assert result.confidence in ("LOW", "MEDIUM", "HIGH")
        assert isinstance(result.auxiliary_findings, list)

    @patch("subprocess.run")
    def test_estimates_degradation_phase(self, mock_run, tmp_path: Path) -> None:
        mock_run.return_value = MagicMock(
            stdout=(
                "abc1234 feat(35): haron features\n"
                "def5678 fix(36): MarketModel fix\n"
                "ghi9012 feat(37): OOF health\n"
            ),
            returncode=0,
        )

        input_dir = _setup_input_dir(tmp_path)
        hb = HistoricalBisect(
            input_dir=input_dir,
            oof_path=tmp_path / "oof" / "oof_predictions.parquet",
        )
        result = hb.run_historical_comparison()

        # Should estimate degradation phase (v1.7 Phase 34 -> v2.0 Phase 38)
        assert "Phase" in result.estimated_degradation_phase or "phase" in result.estimated_degradation_phase.lower()
