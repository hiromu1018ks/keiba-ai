# tests/test_horse_track_aptitude.py
"""Tests for horse_track_aptitude.py — PIT-safe track condition aptitude precompute.

Covers: expanding window + shift(1) PIT-safety, hit/starts counting,
condition classification, versatility, prev values, APTITUDE_COLS, empty input.
"""

from __future__ import annotations

import pandas as pd
import pytest

from features.horse_track_aptitude import APTITUDE_COLS, precompute_track_aptitude

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entries(rows: list[dict]) -> pd.DataFrame:
    """Build entries DataFrame with defaults."""
    defaults = {
        "race_id": "20200101010101",
        "kettonum": "12345",
        "kakuteijyuni": 1,
        "race_date": pd.Timestamp("2020-01-01"),
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _make_races(rows: list[dict]) -> pd.DataFrame:
    """Build races DataFrame with defaults."""
    defaults = {
        "race_id": "20200101010101",
        "trackcd": 24,  # dirt by default
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _make_track_conditions(rows: list[dict]) -> pd.DataFrame:
    """Build track_conditions DataFrame with defaults."""
    defaults = {
        "race_id": "20200101010101",
        "race_date": pd.Timestamp("2020-01-01"),
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


# ---------------------------------------------------------------------------
# Test 1: Output has 14 columns
# ---------------------------------------------------------------------------

def test_output_has_14_columns():
    """precompute_track_aptitude produces 14 columns."""
    entries = _make_entries([
        {"race_id": "R1", "kettonum": "H1", "kakuteijyuni": 1,
         "race_date": pd.Timestamp("2020-01-01")},
        {"race_id": "R2", "kettonum": "H1", "kakuteijyuni": 2,
         "race_date": pd.Timestamp("2020-02-01")},
    ])
    races = _make_races([
        {"race_id": "R1", "trackcd": 24},
        {"race_id": "R2", "trackcd": 24},
    ])
    tc = _make_track_conditions([
        {"race_id": "R1", "dirt_moisture": 15.0, "turf_cushion": float("nan")},
        {"race_id": "R2", "dirt_moisture": 5.0, "turf_cushion": float("nan")},
    ])
    result = precompute_track_aptitude(entries, races, tc)
    expected_cols = {
        "race_id", "kettonum",
        "horse_dirt_wet_hit_rate", "horse_dirt_dry_hit_rate",
        "horse_cushion_hard_hit_rate", "horse_cushion_soft_hit_rate",
        "horse_dirt_wet_starts_count", "horse_dirt_dry_starts_count",
        "horse_cushion_hard_starts_count", "horse_cushion_soft_starts_count",
        "horse_condition_versatility", "horse_condition_type",
        "prev_dirt_moisture", "prev_turf_cushion",
    }
    assert set(result.columns) == expected_cols
    assert len(result.columns) == 14


# ---------------------------------------------------------------------------
# Test 2: PIT-safe — first start has NaN rates and 0 starts_count
# ---------------------------------------------------------------------------

def test_pit_safe_first_start():
    """First start for a horse has NaN hit rates and 0 starts_count."""
    entries = _make_entries([
        {"race_id": "R1", "kettonum": "H1", "kakuteijyuni": 1,
         "race_date": pd.Timestamp("2020-01-01")},
    ])
    races = _make_races([
        {"race_id": "R1", "trackcd": 24},
    ])
    tc = _make_track_conditions([
        {"race_id": "R1", "dirt_moisture": 15.0, "turf_cushion": float("nan")},
    ])
    result = precompute_track_aptitude(entries, races, tc)
    assert len(result) == 1
    row = result.iloc[0]
    # First start: all hit rates NaN (no past data)
    assert pd.isna(row["horse_dirt_wet_hit_rate"])
    assert pd.isna(row["horse_dirt_dry_hit_rate"])
    assert pd.isna(row["horse_cushion_hard_hit_rate"])
    assert pd.isna(row["horse_cushion_soft_hit_rate"])
    # Starts counts are 0 (no past races)
    assert row["horse_dirt_wet_starts_count"] == 0
    assert row["horse_dirt_dry_starts_count"] == 0
    assert row["horse_cushion_hard_starts_count"] == 0
    assert row["horse_cushion_soft_starts_count"] == 0


# ---------------------------------------------------------------------------
# Test 3: Hit definition: kakuteijyuni <= 3 counts as hit
# ---------------------------------------------------------------------------

def test_hit_definition():
    """kakuteijyuni <= 3 counts as hit; <= 0 or NaN excluded from denominator."""
    entries = _make_entries([
        # Race 1: wet dirt (moisture=15), finish 2nd -> hit
        {"race_id": "R1", "kettonum": "H1", "kakuteijyuni": 2,
         "race_date": pd.Timestamp("2020-01-01")},
        # Race 2: dry dirt (moisture=2), finish 5th -> not hit, but counts as start
        {"race_id": "R2", "kettonum": "H1", "kakuteijyuni": 5,
         "race_date": pd.Timestamp("2020-02-01")},
        # Race 3: wet dirt (moisture=15), finish 3rd -> hit (boundary)
        {"race_id": "R3", "kettonum": "H1", "kakuteijyuni": 3,
         "race_date": pd.Timestamp("2020-03-01")},
    ])
    races = _make_races([
        {"race_id": "R1", "trackcd": 24},
        {"race_id": "R2", "trackcd": 24},
        {"race_id": "R3", "trackcd": 24},
    ])
    tc = _make_track_conditions([
        {"race_id": "R1", "dirt_moisture": 15.0, "turf_cushion": float("nan")},
        {"race_id": "R2", "dirt_moisture": 2.0, "turf_cushion": float("nan")},
        {"race_id": "R3", "dirt_moisture": 15.0, "turf_cushion": float("nan")},
    ])
    result = precompute_track_aptitude(entries, races, tc)
    # Row for R2: after R1 (wet dirt, hit)
    row_r2 = result[result["race_id"] == "R2"].iloc[0]
    assert row_r2["horse_dirt_wet_starts_count"] == 1  # R1 was wet dirt
    assert row_r2["horse_dirt_wet_hit_rate"] == pytest.approx(1.0)  # 1 hit / 1 start
    assert row_r2["horse_dirt_dry_starts_count"] == 0  # R1 was not dry

    # Row for R3: after R1 (wet, hit) + R2 (dry, not hit)
    row_r3 = result[result["race_id"] == "R3"].iloc[0]
    assert row_r3["horse_dirt_wet_starts_count"] == 1  # R1 only
    assert row_r3["horse_dirt_wet_hit_rate"] == pytest.approx(1.0)
    assert row_r3["horse_dirt_dry_starts_count"] == 1  # R2 only
    assert row_r3["horse_dirt_dry_hit_rate"] == pytest.approx(0.0)  # 0 hits / 1 start


def test_excluded_from_denominator():
    """kakuteijyuni <= 0 or NaN excluded from denominator."""
    entries = _make_entries([
        {"race_id": "R1", "kettonum": "H1", "kakuteijyuni": 0,
         "race_date": pd.Timestamp("2020-01-01")},  # excluded
        {"race_id": "R2", "kettonum": "H1", "kakuteijyuni": 2,
         "race_date": pd.Timestamp("2020-02-01")},
    ])
    races = _make_races([
        {"race_id": "R1", "trackcd": 24},
        {"race_id": "R2", "trackcd": 24},
    ])
    tc = _make_track_conditions([
        {"race_id": "R1", "dirt_moisture": 15.0, "turf_cushion": float("nan")},
        {"race_id": "R2", "dirt_moisture": 15.0, "turf_cushion": float("nan")},
    ])
    result = precompute_track_aptitude(entries, races, tc)
    # Row for R2: R1 excluded from denominator
    row_r2 = result[result["race_id"] == "R2"].iloc[0]
    assert row_r2["horse_dirt_wet_starts_count"] == 0  # R1 excluded
    assert pd.isna(row_r2["horse_dirt_wet_hit_rate"])  # 0 starts -> NaN


# ---------------------------------------------------------------------------
# Test 4: horse_condition_type classification
# ---------------------------------------------------------------------------

def test_condition_type_wet_good():
    """wet_rate >= 0.3 AND dry_rate < 0.3 with min_starts=3 -> wet_good."""
    # Build entries where horse has 4 wet starts with 2 hits (rate=0.5)
    # and 4 dry starts with 0 hits (rate=0.0)
    rows_entries = []
    rows_races = []
    rows_tc = []
    for i in range(4):
        rid = f"W{i+1}"
        rows_entries.append({
            "race_id": rid, "kettonum": "H1", "kakuteijyuni": 1,  # hit
            "race_date": pd.Timestamp(f"2020-0{i+1}-01"),
        })
        rows_races.append({"race_id": rid, "trackcd": 24})
        rows_tc.append({"race_id": rid, "dirt_moisture": 15.0, "turf_cushion": float("nan")})
    for i in range(4):
        rid = f"D{i+1}"
        rows_entries.append({
            "race_id": rid, "kettonum": "H1", "kakuteijyuni": 8,  # not hit
            "race_date": pd.Timestamp(f"2020-{i+5:02d}-01"),
        })
        rows_races.append({"race_id": rid, "trackcd": 24})
        rows_tc.append({"race_id": rid, "dirt_moisture": 2.0, "turf_cushion": float("nan")})

    entries = _make_entries(rows_entries)
    races = _make_races(rows_races)
    tc = _make_track_conditions(rows_tc)
    result = precompute_track_aptitude(entries, races, tc)

    # Last row (D4): after all 4 wet (4 hits) + 3 dry (0 hits) = wet_good
    last_row = result.iloc[-1]
    assert last_row["horse_condition_type"] == "wet_good"


def test_condition_type_dry_good():
    """dry_rate >= 0.3 AND wet_rate < 0.3 -> dry_good."""
    rows_entries = []
    rows_races = []
    rows_tc = []
    # 4 wet starts, 0 hits
    for i in range(4):
        rid = f"W{i+1}"
        rows_entries.append({
            "race_id": rid, "kettonum": "H1", "kakuteijyuni": 8,
            "race_date": pd.Timestamp(f"2020-0{i+1}-01"),
        })
        rows_races.append({"race_id": rid, "trackcd": 24})
        rows_tc.append({"race_id": rid, "dirt_moisture": 15.0, "turf_cushion": float("nan")})
    # 4 dry starts, 2 hits
    for i in range(4):
        rid = f"D{i+1}"
        kakuteijyuni = 1 if i < 2 else 8
        rows_entries.append({
            "race_id": rid, "kettonum": "H1", "kakuteijyuni": kakuteijyuni,
            "race_date": pd.Timestamp(f"2020-{i+5:02d}-01"),
        })
        rows_races.append({"race_id": rid, "trackcd": 24})
        rows_tc.append({"race_id": rid, "dirt_moisture": 2.0, "turf_cushion": float("nan")})

    entries = _make_entries(rows_entries)
    races = _make_races(rows_races)
    tc = _make_track_conditions(rows_tc)
    result = precompute_track_aptitude(entries, races, tc)

    last_row = result.iloc[-1]
    assert last_row["horse_condition_type"] == "dry_good"


def test_condition_type_balanced():
    """Both rates >= 0.3 -> balanced."""
    rows_entries = []
    rows_races = []
    rows_tc = []
    # 4 wet, 2 hits
    for i in range(4):
        rid = f"W{i+1}"
        kakuteijyuni = 1 if i < 2 else 8
        rows_entries.append({
            "race_id": rid, "kettonum": "H1", "kakuteijyuni": kakuteijyuni,
            "race_date": pd.Timestamp(f"2020-0{i+1}-01"),
        })
        rows_races.append({"race_id": rid, "trackcd": 24})
        rows_tc.append({"race_id": rid, "dirt_moisture": 15.0, "turf_cushion": float("nan")})
    # 4 dry, 2 hits
    for i in range(4):
        rid = f"D{i+1}"
        kakuteijyuni = 1 if i < 2 else 8
        rows_entries.append({
            "race_id": rid, "kettonum": "H1", "kakuteijyuni": kakuteijyuni,
            "race_date": pd.Timestamp(f"2020-{i+5:02d}-01"),
        })
        rows_races.append({"race_id": rid, "trackcd": 24})
        rows_tc.append({"race_id": rid, "dirt_moisture": 2.0, "turf_cushion": float("nan")})

    entries = _make_entries(rows_entries)
    races = _make_races(rows_races)
    tc = _make_track_conditions(rows_tc)
    result = precompute_track_aptitude(entries, races, tc)

    last_row = result.iloc[-1]
    assert last_row["horse_condition_type"] == "balanced"


def test_condition_type_unknown_insufficient_starts():
    """Insufficient starts (< min_starts=3) -> unknown."""
    entries = _make_entries([
        {"race_id": "R1", "kettonum": "H1", "kakuteijyuni": 1,
         "race_date": pd.Timestamp("2020-01-01")},
        {"race_id": "R2", "kettonum": "H1", "kakuteijyuni": 1,
         "race_date": pd.Timestamp("2020-02-01")},
    ])
    races = _make_races([
        {"race_id": "R1", "trackcd": 24},
        {"race_id": "R2", "trackcd": 24},
    ])
    tc = _make_track_conditions([
        {"race_id": "R1", "dirt_moisture": 15.0, "turf_cushion": float("nan")},
        {"race_id": "R2", "dirt_moisture": 15.0, "turf_cushion": float("nan")},
    ])
    result = precompute_track_aptitude(entries, races, tc)
    # Row for R2: only 1 past wet start (< 3)
    row_r2 = result[result["race_id"] == "R2"].iloc[0]
    assert row_r2["horse_condition_type"] == "unknown"


# ---------------------------------------------------------------------------
# Test 5: horse_condition_versatility
# ---------------------------------------------------------------------------

def test_versatility():
    """versatility = mean(wet_rate, dry_rate) * (1 - |wet_rate - dry_rate|)."""
    rows_entries = []
    rows_races = []
    rows_tc = []
    # 4 wet, 2 hits (rate=0.5)
    for i in range(4):
        rid = f"W{i+1}"
        kakuteijyuni = 1 if i < 2 else 8
        rows_entries.append({
            "race_id": rid, "kettonum": "H1", "kakuteijyuni": kakuteijyuni,
            "race_date": pd.Timestamp(f"2020-0{i+1}-01"),
        })
        rows_races.append({"race_id": rid, "trackcd": 24})
        rows_tc.append({"race_id": rid, "dirt_moisture": 15.0, "turf_cushion": float("nan")})
    # 4 dry, 1 hit (rate=0.25)
    for i in range(4):
        rid = f"D{i+1}"
        kakuteijyuni = 1 if i == 0 else 8
        rows_entries.append({
            "race_id": rid, "kettonum": "H1", "kakuteijyuni": kakuteijyuni,
            "race_date": pd.Timestamp(f"2020-{i+5:02d}-01"),
        })
        rows_races.append({"race_id": rid, "trackcd": 24})
        rows_tc.append({"race_id": rid, "dirt_moisture": 2.0, "turf_cushion": float("nan")})

    entries = _make_entries(rows_entries)
    races = _make_races(rows_races)
    tc = _make_track_conditions(rows_tc)
    result = precompute_track_aptitude(entries, races, tc)

    last_row = result.iloc[-1]
    # After all 8 races: wet_rate = 2/4 = 0.5, dry_rate = 1/3 = 0.333...
    # (D4 itself excluded by shift(1), so dry = 3 past starts, 1 hit)
    wet_rate = 0.5
    dry_rate = 1.0 / 3.0
    mean_rate = (wet_rate + dry_rate) / 2
    expected_versatility = mean_rate * (1 - abs(wet_rate - dry_rate))
    assert last_row["horse_condition_versatility"] == pytest.approx(expected_versatility)


def test_versatility_nan_when_insufficient():
    """Versatility is NaN when either component rate is NaN."""
    entries = _make_entries([
        {"race_id": "R1", "kettonum": "H1", "kakuteijyuni": 1,
         "race_date": pd.Timestamp("2020-01-01")},
    ])
    races = _make_races([
        {"race_id": "R1", "trackcd": 24},
    ])
    tc = _make_track_conditions([
        {"race_id": "R1", "dirt_moisture": 15.0, "turf_cushion": float("nan")},
    ])
    result = precompute_track_aptitude(entries, races, tc)
    # First start: rates are NaN -> versatility is NaN
    assert pd.isna(result.iloc[0]["horse_condition_versatility"])


# ---------------------------------------------------------------------------
# Test 6: prev_dirt_moisture / prev_turf_cushion
# ---------------------------------------------------------------------------

def test_prev_dirt_moisture():
    """prev_dirt_moisture captures previous race's dirt_moisture via shift(1)."""
    entries = _make_entries([
        {"race_id": "R1", "kettonum": "H1", "kakuteijyuni": 1,
         "race_date": pd.Timestamp("2020-01-01")},
        {"race_id": "R2", "kettonum": "H1", "kakuteijyuni": 2,
         "race_date": pd.Timestamp("2020-02-01")},
        {"race_id": "R3", "kettonum": "H1", "kakuteijyuni": 3,
         "race_date": pd.Timestamp("2020-03-01")},
    ])
    races = _make_races([
        {"race_id": "R1", "trackcd": 24},
        {"race_id": "R2", "trackcd": 24},
        {"race_id": "R3", "trackcd": 24},
    ])
    tc = _make_track_conditions([
        {"race_id": "R1", "dirt_moisture": 10.0, "turf_cushion": float("nan")},
        {"race_id": "R2", "dirt_moisture": 15.0, "turf_cushion": float("nan")},
        {"race_id": "R3", "dirt_moisture": 5.0, "turf_cushion": float("nan")},
    ])
    result = precompute_track_aptitude(entries, races, tc)
    # R1: first start -> prev is NaN
    row_r1 = result[result["race_id"] == "R1"].iloc[0]
    assert pd.isna(row_r1["prev_dirt_moisture"])
    # R2: prev is R1's moisture
    row_r2 = result[result["race_id"] == "R2"].iloc[0]
    assert row_r2["prev_dirt_moisture"] == pytest.approx(10.0)
    # R3: prev is R2's moisture
    row_r3 = result[result["race_id"] == "R3"].iloc[0]
    assert row_r3["prev_dirt_moisture"] == pytest.approx(15.0)


def test_prev_turf_cushion():
    """prev_turf_cushion captures previous race's turf_cushion via shift(1)."""
    entries = _make_entries([
        {"race_id": "R1", "kettonum": "H1", "kakuteijyuni": 1,
         "race_date": pd.Timestamp("2020-01-01")},
        {"race_id": "R2", "kettonum": "H1", "kakuteijyuni": 2,
         "race_date": pd.Timestamp("2020-02-01")},
    ])
    races = _make_races([
        {"race_id": "R1", "trackcd": 11},  # turf
        {"race_id": "R2", "trackcd": 11},
    ])
    tc = _make_track_conditions([
        {"race_id": "R1", "dirt_moisture": float("nan"), "turf_cushion": 9.5},
        {"race_id": "R2", "dirt_moisture": float("nan"), "turf_cushion": 8.0},
    ])
    result = precompute_track_aptitude(entries, races, tc)
    row_r2 = result[result["race_id"] == "R2"].iloc[0]
    assert row_r2["prev_turf_cushion"] == pytest.approx(9.5)


# ---------------------------------------------------------------------------
# Test 7: APTITUDE_COLS constant
# ---------------------------------------------------------------------------

def test_aptitude_cols_constant():
    """APTITUDE_COLS lists all 14 output column names."""
    assert len(APTITUDE_COLS) == 14
    expected = {
        "race_id", "kettonum",
        "horse_dirt_wet_hit_rate", "horse_dirt_dry_hit_rate",
        "horse_cushion_hard_hit_rate", "horse_cushion_soft_hit_rate",
        "horse_dirt_wet_starts_count", "horse_dirt_dry_starts_count",
        "horse_cushion_hard_starts_count", "horse_cushion_soft_starts_count",
        "horse_condition_versatility", "horse_condition_type",
        "prev_dirt_moisture", "prev_turf_cushion",
    }
    assert set(APTITUDE_COLS) == expected


# ---------------------------------------------------------------------------
# Test 8: Empty input returns empty DataFrame
# ---------------------------------------------------------------------------

def test_empty_input():
    """Empty input returns empty DataFrame."""
    entries = pd.DataFrame(
        columns=["race_id", "kettonum", "kakuteijyuni", "race_date"]
    )
    races = pd.DataFrame(columns=["race_id", "trackcd"])
    tc = pd.DataFrame(columns=["race_id", "race_date", "dirt_moisture", "turf_cushion"])
    result = precompute_track_aptitude(entries, races, tc)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# Test: Turf cushion classification (hard/soft)
# ---------------------------------------------------------------------------

def test_turf_cushion_classification():
    """Turf races: cushion >= 10 is hard, < 8 is soft."""
    entries = _make_entries([
        {"race_id": "R1", "kettonum": "H1", "kakuteijyuni": 1,
         "race_date": pd.Timestamp("2020-01-01")},
        {"race_id": "R2", "kettonum": "H1", "kakuteijyuni": 2,
         "race_date": pd.Timestamp("2020-02-01")},
        {"race_id": "R3", "kettonum": "H1", "kakuteijyuni": 1,
         "race_date": pd.Timestamp("2020-03-01")},
    ])
    races = _make_races([
        {"race_id": "R1", "trackcd": 11},  # turf
        {"race_id": "R2", "trackcd": 11},
        {"race_id": "R3", "trackcd": 11},
    ])
    tc = _make_track_conditions([
        {"race_id": "R1", "dirt_moisture": float("nan"), "turf_cushion": 11.0},  # hard
        {"race_id": "R2", "dirt_moisture": float("nan"), "turf_cushion": 7.0},   # soft
        {"race_id": "R3", "dirt_moisture": float("nan"), "turf_cushion": 11.0},  # hard
    ])
    result = precompute_track_aptitude(entries, races, tc)

    # R2: after R1 (hard, hit)
    row_r2 = result[result["race_id"] == "R2"].iloc[0]
    assert row_r2["horse_cushion_hard_starts_count"] == 1
    assert row_r2["horse_cushion_hard_hit_rate"] == pytest.approx(1.0)
    assert row_r2["horse_cushion_soft_starts_count"] == 0

    # R3: after R1 (hard, hit) + R2 (soft, hit)
    row_r3 = result[result["race_id"] == "R3"].iloc[0]
    assert row_r3["horse_cushion_hard_starts_count"] == 1
    assert row_r3["horse_cushion_hard_hit_rate"] == pytest.approx(1.0)
    assert row_r3["horse_cushion_soft_starts_count"] == 1
    assert row_r3["horse_cushion_soft_hit_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test: Multiple horses don't cross-contaminate
# ---------------------------------------------------------------------------

def test_multiple_horses_isolation():
    """Different horses (kettonum) have independent stats."""
    entries = _make_entries([
        {"race_id": "R1", "kettonum": "H1", "kakuteijyuni": 1,
         "race_date": pd.Timestamp("2020-01-01")},
        {"race_id": "R1", "kettonum": "H2", "kakuteijyuni": 5,
         "race_date": pd.Timestamp("2020-01-01")},
        {"race_id": "R2", "kettonum": "H1", "kakuteijyuni": 2,
         "race_date": pd.Timestamp("2020-02-01")},
        {"race_id": "R2", "kettonum": "H2", "kakuteijyuni": 1,
         "race_date": pd.Timestamp("2020-02-01")},
    ])
    races = _make_races([
        {"race_id": "R1", "trackcd": 24},
        {"race_id": "R2", "trackcd": 24},
    ])
    tc = _make_track_conditions([
        {"race_id": "R1", "dirt_moisture": 15.0, "turf_cushion": float("nan")},
        {"race_id": "R2", "dirt_moisture": 15.0, "turf_cushion": float("nan")},
    ])
    result = precompute_track_aptitude(entries, races, tc)

    # H1: R1 was wet hit -> R2 has 1 wet start, rate=1.0
    h1_r2 = result[(result["kettonum"] == "H1") & (result["race_id"] == "R2")].iloc[0]
    assert h1_r2["horse_dirt_wet_starts_count"] == 1
    assert h1_r2["horse_dirt_wet_hit_rate"] == pytest.approx(1.0)

    # H2: R1 was wet not-hit (5th) -> R2 has 1 wet start, rate=0.0
    h2_r2 = result[(result["kettonum"] == "H2") & (result["race_id"] == "R2")].iloc[0]
    assert h2_r2["horse_dirt_wet_starts_count"] == 1
    assert h2_r2["horse_dirt_wet_hit_rate"] == pytest.approx(0.0)
