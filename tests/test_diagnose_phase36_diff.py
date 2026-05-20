"""Unit tests for scripts/diagnose_phase36_diff.py

Tests the v1.7 vs current backtest diff diagnostic logic:
- load_backtest_results: CSV reading with column standardization
- find_common_races: race_id intersection/difference
- compute_horse_overlap: same-horse vs different-horse classification
- compute_roi_breakdown: ROI/win_rate calculation
- generate_report: full report generation
- Edge cases: empty DataFrames
- Phase36 contribution decomposition
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from scripts.diagnose_phase36_diff import (
    compute_horse_overlap,
    compute_roi_breakdown,
    find_common_races,
    generate_report,
    load_backtest_results,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic backtest CSV data
# ---------------------------------------------------------------------------

@pytest.fixture()
def baseline_csv(tmp_path: Path) -> Path:
    """Small synthetic baseline (v1.7) backtest CSV."""
    data = textwrap.dedent("""\
        race_id,umaban,stake,odds,result,ev,edge,regime,quality_passed,bet_type
        202401010101,1,100,5.2,0,0.85,0.10,aggressive,True,win
        202401010101,3,100,8.1,810,1.20,0.25,aggressive,True,win
        202401010102,2,100,3.5,350,1.05,0.05,conservative,True,win
        202401010102,5,100,12.0,0,0.90,0.15,conservative,True,win
        202401010103,1,100,4.0,0,0.80,0.05,aggressive,True,win
        202401010104,4,100,6.5,650,1.15,0.20,aggressive,True,win
    """)
    p = tmp_path / "baseline.csv"
    p.write_text(data, encoding="utf-8")
    return p


@pytest.fixture()
def current_csv(tmp_path: Path) -> Path:
    """Small synthetic current backtest CSV.

    Differences from baseline:
    - Race 0101: same horse 1, same horse 3, added horse 5
    - Race 0102: same horse 2, removed horse 5
    - Race 0103: removed (not in current)
    - Race 0105: new race (not in baseline)
    """
    data = textwrap.dedent("""\
        race_id,umaban,stake,odds,result,ev,edge,regime,quality_passed,bet_type
        202401010101,1,100,5.2,0,0.85,0.10,aggressive,True,win
        202401010101,3,100,8.1,810,1.20,0.25,aggressive,True,win
        202401010101,5,100,15.0,1500,1.80,0.50,aggressive,True,win
        202401010102,2,100,3.5,350,1.05,0.05,conservative,True,win
        202401010104,4,100,6.5,650,1.15,0.20,aggressive,True,win
        202401010105,2,100,9.0,900,1.60,0.30,aggressive,True,win
    """)
    p = tmp_path / "current.csv"
    p.write_text(data, encoding="utf-8")
    return p


@pytest.fixture()
def empty_csv(tmp_path: Path) -> Path:
    """CSV with only headers, no data rows."""
    data = "race_id,umaban,stake,odds,result,ev,edge,regime,quality_passed,bet_type\n"
    p = tmp_path / "empty.csv"
    p.write_text(data, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Test 1: load_backtest_results
# ---------------------------------------------------------------------------

class TestLoadBacktestResults:
    def test_reads_csv_and_returns_dataframe(self, baseline_csv: Path) -> None:
        df = load_backtest_results(str(baseline_csv))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6

    def test_column_names_lowercase(self, baseline_csv: Path) -> None:
        df = load_backtest_results(str(baseline_csv))
        # All column names should be lowercase
        for col in df.columns:
            assert col == col.lower(), f"Column '{col}' is not lowercase"

    def test_required_columns_exist(self, baseline_csv: Path) -> None:
        df = load_backtest_results(str(baseline_csv))
        assert "race_id" in df.columns
        assert "umaban" in df.columns

    def test_race_id_is_string(self, baseline_csv: Path) -> None:
        df = load_backtest_results(str(baseline_csv))
        assert df["race_id"].dtype == object  # string dtype

    def test_empty_csv_produces_empty_df(self, empty_csv: Path) -> None:
        df = load_backtest_results(str(empty_csv))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Test 2: find_common_races
# ---------------------------------------------------------------------------

class TestFindCommonRaces:
    def test_identifies_intersection(self, baseline_csv: Path, current_csv: Path) -> None:
        base_df = load_backtest_results(str(baseline_csv))
        curr_df = load_backtest_results(str(current_csv))
        common, base_only, curr_only = find_common_races(base_df, curr_df)

        # Races 0101, 0102, 0104 are in both
        expected_common = {"202401010101", "202401010102", "202401010104"}
        assert common == expected_common

    def test_baseline_only_races(self, baseline_csv: Path, current_csv: Path) -> None:
        base_df = load_backtest_results(str(baseline_csv))
        curr_df = load_backtest_results(str(current_csv))
        _, base_only, _ = find_common_races(base_df, curr_df)

        assert base_only == {"202401010103"}

    def test_current_only_races(self, baseline_csv: Path, current_csv: Path) -> None:
        base_df = load_backtest_results(str(baseline_csv))
        curr_df = load_backtest_results(str(current_csv))
        _, _, curr_only = find_common_races(base_df, curr_df)

        assert curr_only == {"202401010105"}

    def test_empty_baseline(self, empty_csv: Path, current_csv: Path) -> None:
        base_df = load_backtest_results(str(empty_csv))
        curr_df = load_backtest_results(str(current_csv))
        common, base_only, curr_only = find_common_races(base_df, curr_df)

        assert common == set()
        assert base_only == set()
        assert len(curr_only) > 0

    def test_empty_current(self, baseline_csv: Path, empty_csv: Path) -> None:
        base_df = load_backtest_results(str(baseline_csv))
        curr_df = load_backtest_results(str(empty_csv))
        common, base_only, curr_only = find_common_races(base_df, curr_df)

        assert common == set()
        assert len(base_only) > 0
        assert curr_only == set()


# ---------------------------------------------------------------------------
# Test 3: compute_horse_overlap
# ---------------------------------------------------------------------------

class TestComputeHorseOverlap:
    def test_same_horse_identified(self, baseline_csv: Path, current_csv: Path) -> None:
        base_df = load_backtest_results(str(baseline_csv))
        curr_df = load_backtest_results(str(current_csv))
        common, _, _ = find_common_races(base_df, curr_df)
        overlap = compute_horse_overlap(base_df, curr_df, common)

        same = overlap["same_horse_df"]
        # Race 0101: horses 1, 3 are same in both
        # Race 0102: horse 2 is same in both
        # Race 0104: horse 4 is same in both
        assert len(same) == 4

    def test_baseline_only_horse(self, baseline_csv: Path, current_csv: Path) -> None:
        base_df = load_backtest_results(str(baseline_csv))
        curr_df = load_backtest_results(str(current_csv))
        common, _, _ = find_common_races(base_df, curr_df)
        overlap = compute_horse_overlap(base_df, curr_df, common)

        bl_only = overlap["baseline_only_horse_df"]
        # Race 0102: horse 5 only in baseline
        assert len(bl_only) == 1

    def test_current_only_horse(self, baseline_csv: Path, current_csv: Path) -> None:
        base_df = load_backtest_results(str(baseline_csv))
        curr_df = load_backtest_results(str(current_csv))
        common, _, _ = find_common_races(base_df, curr_df)
        overlap = compute_horse_overlap(base_df, curr_df, common)

        cur_only = overlap["current_only_horse_df"]
        # Race 0101: horse 5 only in current
        assert len(cur_only) == 1

    def test_overlap_keys_present(self, baseline_csv: Path, current_csv: Path) -> None:
        base_df = load_backtest_results(str(baseline_csv))
        curr_df = load_backtest_results(str(current_csv))
        common, _, _ = find_common_races(base_df, curr_df)
        overlap = compute_horse_overlap(base_df, curr_df, common)

        assert "same_horse_df" in overlap
        assert "baseline_only_horse_df" in overlap
        assert "current_only_horse_df" in overlap
        assert "n_same_horse" in overlap
        assert "n_baseline_only" in overlap
        assert "n_current_only" in overlap


# ---------------------------------------------------------------------------
# Test 4: compute_roi_breakdown
# ---------------------------------------------------------------------------

class TestComputeRoiBreakdown:
    def test_roi_calculation(self, baseline_csv: Path) -> None:
        df = load_backtest_results(str(baseline_csv))
        # Filter to race 0101 only for deterministic test
        race_df = df[df["race_id"] == "202401010101"]
        breakdown = compute_roi_breakdown(race_df)

        assert breakdown["n_bets"] == 2
        assert breakdown["total_stake"] == 200.0
        assert breakdown["total_return"] == 810.0
        assert abs(breakdown["roi"] - 810.0 / 200.0) < 1e-6

    def test_win_rate_calculation(self, baseline_csv: Path) -> None:
        df = load_backtest_results(str(baseline_csv))
        race_df = df[df["race_id"] == "202401010101"]
        breakdown = compute_roi_breakdown(race_df)

        # 1 win out of 2 bets
        assert abs(breakdown["win_rate"] - 0.5) < 1e-6

    def test_empty_dataframe_returns_zeros(self) -> None:
        df = pd.DataFrame()
        breakdown = compute_roi_breakdown(df)

        assert breakdown["n_bets"] == 0
        assert breakdown["total_stake"] == 0.0
        assert breakdown["total_return"] == 0.0
        assert breakdown["roi"] == 0.0
        assert breakdown["win_rate"] == 0.0


# ---------------------------------------------------------------------------
# Test 5: generate_report
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_report_has_required_fields(
        self, baseline_csv: Path, current_csv: Path
    ) -> None:
        report = generate_report(str(baseline_csv), str(current_csv))

        required_keys = [
            "n_common_races",
            "n_baseline_only_races",
            "n_current_only_races",
            "same_horse",
            "baseline_only_horse",
            "current_only_horse",
            "phase36_contribution",
        ]
        for key in required_keys:
            assert key in report, f"Missing key: {key}"

    def test_report_common_race_count(
        self, baseline_csv: Path, current_csv: Path
    ) -> None:
        report = generate_report(str(baseline_csv), str(current_csv))
        assert report["n_common_races"] == 3  # 0101, 0102, 0104

    def test_report_baseline_only_count(
        self, baseline_csv: Path, current_csv: Path
    ) -> None:
        report = generate_report(str(baseline_csv), str(current_csv))
        assert report["n_baseline_only_races"] == 1  # 0103

    def test_report_current_only_count(
        self, baseline_csv: Path, current_csv: Path
    ) -> None:
        report = generate_report(str(baseline_csv), str(current_csv))
        assert report["n_current_only_races"] == 1  # 0105

    def test_same_horse_roi_is_dict(
        self, baseline_csv: Path, current_csv: Path
    ) -> None:
        report = generate_report(str(baseline_csv), str(current_csv))
        same = report["same_horse"]
        assert isinstance(same, dict)
        assert "n_bets" in same
        assert "roi" in same


# ---------------------------------------------------------------------------
# Test 6: Edge case - empty DataFrames
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_baseline_produces_empty_report(
        self, empty_csv: Path, current_csv: Path
    ) -> None:
        report = generate_report(str(empty_csv), str(current_csv))
        assert report["n_common_races"] == 0
        assert report["same_horse"]["n_bets"] == 0

    def test_empty_current_produces_empty_report(
        self, baseline_csv: Path, empty_csv: Path
    ) -> None:
        report = generate_report(str(baseline_csv), str(empty_csv))
        assert report["n_common_races"] == 0
        assert report["same_horse"]["n_bets"] == 0

    def test_empty_df_no_crash(self) -> None:
        df = pd.DataFrame()
        breakdown = compute_roi_breakdown(df)
        assert breakdown["roi"] == 0.0


# ---------------------------------------------------------------------------
# Test 7: Phase36 contribution decomposition
# ---------------------------------------------------------------------------

class TestPhase36Contribution:
    def test_phase36_contribution_keys(
        self, baseline_csv: Path, current_csv: Path
    ) -> None:
        report = generate_report(str(baseline_csv), str(current_csv))
        p36 = report["phase36_contribution"]

        expected_keys = [
            "roi_same",
            "roi_new",
            "roi_removed",
            "n_new_horses",
            "n_removed_horses",
            "net_contribution_pct",
        ]
        for key in expected_keys:
            assert key in p36, f"Missing phase36_contribution key: {key}"

    def test_phase36_new_horses_count(
        self, baseline_csv: Path, current_csv: Path
    ) -> None:
        report = generate_report(str(baseline_csv), str(current_csv))
        p36 = report["phase36_contribution"]

        # Race 0101 horse 5 is new in current (current_only in common races)
        assert p36["n_new_horses"] == 1

    def test_phase36_removed_horses_count(
        self, baseline_csv: Path, current_csv: Path
    ) -> None:
        report = generate_report(str(baseline_csv), str(current_csv))
        p36 = report["phase36_contribution"]

        # Race 0102 horse 5 is removed from current (baseline_only in common races)
        assert p36["n_removed_horses"] == 1

    def test_phase36_roi_values_are_numbers(
        self, baseline_csv: Path, current_csv: Path
    ) -> None:
        report = generate_report(str(baseline_csv), str(current_csv))
        p36 = report["phase36_contribution"]

        assert isinstance(p36["roi_same"], float)
        assert isinstance(p36["roi_new"], float)
        assert isinstance(p36["roi_removed"], float)
        assert isinstance(p36["net_contribution_pct"], float)

    def test_phase36_screener_fix_count(
        self, baseline_csv: Path, current_csv: Path
    ) -> None:
        """Report includes screener_fix_count (quality_passed changed)."""
        report = generate_report(str(baseline_csv), str(current_csv))
        p36 = report["phase36_contribution"]
        assert "screener_fix_count" in p36
        assert isinstance(p36["screener_fix_count"], int)

    def test_phase36_ev_tail_count(
        self, baseline_csv: Path, current_csv: Path
    ) -> None:
        """Report includes ev_tail_count (current_only with EV >= 1.5)."""
        report = generate_report(str(baseline_csv), str(current_csv))
        p36 = report["phase36_contribution"]
        assert "ev_tail_count" in p36
        assert isinstance(p36["ev_tail_count"], int)
