"""Tests for _verify_coverage in run_etl.py."""

import logging
from unittest.mock import MagicMock

import pandas as pd
import pytest


def _make_store(
    exists_side_effect: object = True,
    df: pd.DataFrame | None = None,
) -> MagicMock:
    """Create a mock ParquetStore with configurable exists/read behaviour."""
    store = MagicMock()
    store.exists.side_effect = (
        exists_side_effect if callable(exists_side_effect) else lambda *_: exists_side_effect
    )
    if df is not None:
        store.read.return_value = df
    return store


def _make_df(
    years: list[int] | None = None,
    n_rows: int = 100,
    extra_cols: dict[str, list] | None = None,
) -> pd.DataFrame:
    """Build a minimal DataFrame with a race_date column spanning *years*."""
    if years is None:
        years = list(range(2015, 2026))
    dates: list[pd.Timestamp] = []
    per_year = max(n_rows // len(years), 1)
    for y in years:
        for m in range(1, min(per_year + 1, 13)):
            dates.append(pd.Timestamp(year=y, month=m, day=1))
    data: dict = {"race_date": dates[:n_rows]}
    if extra_cols:
        for col, vals in extra_cols.items():
            data[col] = vals
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# 1. Full coverage -- info log only, no warnings
# ---------------------------------------------------------------------------
def test_full_coverage_info_only(caplog: pytest.LogCaptureFixture) -> None:
    """All years present, no missing data -> single info-level coverage log."""
    from scripts.run_etl import _verify_coverage

    df = _make_df(years=list(range(2015, 2026)), n_rows=120)
    store = _make_store(exists_side_effect=True, df=df)
    tables = ["odds_sanren"]

    with caplog.at_level(logging.DEBUG, logger="scripts.run_etl"):
        _verify_coverage(store, tables, 2015, 2025)

    assert any("Coverage odds_sanren" in r.message for r in caplog.records)
    # No WARNING-level messages expected
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == []


# ---------------------------------------------------------------------------
# 2. Missing years -- warning log
# ---------------------------------------------------------------------------
def test_missing_years_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Years 2018 and 2019 absent -> warning about missing years."""
    from scripts.run_etl import _verify_coverage

    df = _make_df(years=[2015, 2016, 2017, 2020, 2021, 2022, 2023, 2024, 2025], n_rows=100)
    store = _make_store(exists_side_effect=True, df=df)
    tables = ["odds_umaren"]

    with caplog.at_level(logging.DEBUG, logger="scripts.run_etl"):
        _verify_coverage(store, tables, 2015, 2025)

    warn_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("missing years" in m for m in warn_msgs)


# ---------------------------------------------------------------------------
# 3. High missing rate (>30%) -- warning log
# ---------------------------------------------------------------------------
def test_high_missing_rate_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Column with >30% nulls triggers missing-rate warning."""
    from scripts.run_etl import _verify_coverage

    df = _make_df(n_rows=50)
    half = len(df) // 2
    df["value"] = [None] * half + [1.0] * (len(df) - half)
    store = _make_store(exists_side_effect=True, df=df)
    tables = ["odds_sanrentan"]

    with caplog.at_level(logging.DEBUG, logger="scripts.run_etl"):
        _verify_coverage(store, tables, 2015, 2025)

    warn_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("missing rate" in m and "exceeds 30%" in m for m in warn_msgs)


# ---------------------------------------------------------------------------
# 4. Nonexistent file -- skip gracefully with warning
# ---------------------------------------------------------------------------
def test_nonexistent_file_skip(caplog: pytest.LogCaptureFixture) -> None:
    """When store.exists() returns False, table is skipped with a warning."""
    from scripts.run_etl import _verify_coverage

    store = _make_store(exists_side_effect=False)
    tables = ["odds_umaren_head"]

    with caplog.at_level(logging.DEBUG, logger="scripts.run_etl"):
        _verify_coverage(store, tables, 2015, 2025)

    warn_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("SKIP" in m and "odds_umaren_head" in m for m in warn_msgs)
    # read() should never be called for missing files
    store.read.assert_not_called()


# ---------------------------------------------------------------------------
# 5. Empty DataFrame -- handled without error
# ---------------------------------------------------------------------------
def test_empty_dataframe_no_error(caplog: pytest.LogCaptureFixture) -> None:
    """Empty DataFrame (0 rows) should not raise and should log coverage."""
    from scripts.run_etl import _verify_coverage

    df = pd.DataFrame({"race_date": pd.Series(dtype="datetime64[ns]")})
    store = _make_store(exists_side_effect=True, df=df)
    tables = ["odds_sanrentan_head"]

    with caplog.at_level(logging.DEBUG, logger="scripts.run_etl"):
        _verify_coverage(store, tables, 2015, 2025)

    # Should log coverage with 0 rows (info level acceptable)
    assert any("Coverage" in r.message and "0 rows" in r.message for r in caplog.records)
