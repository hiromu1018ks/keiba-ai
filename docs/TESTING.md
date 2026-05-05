<!-- generated-by: gsd-doc-writer -->

# Testing

## Test Framework and Setup

The project uses **pytest 8.0+** with **pytest-cov 5.0+** for test execution and coverage reporting. Both are declared as production dependencies in `pyproject.toml`.

**Test configuration** is defined in `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
testpaths = ["tests"]
pythonpath = [".", "src"]
```

- `testpaths = ["tests"]` -- all test files live in the top-level `tests/` directory.
- `pythonpath = [".", "src"]` -- imports resolve against `src/` so that `from domain.types import ...` and `from db.parquet_store import ParquetStore` work without `sys.path` hacks.

No global `conftest.py` exists. Each test file is self-contained: fixtures are defined locally within the file that uses them.

**Prerequisites before running tests:**

```bash
pip install -e ".[dev]"
```

This installs pytest, pytest-cov, ruff, mypy, and all runtime dependencies. No database connection is required -- all tests use `unittest.mock` to isolate from PostgreSQL and external services.

## Running Tests

### Full test suite

```bash
python -m pytest tests/ -v
```

### Single test file

```bash
python -m pytest tests/test_domain.py -v
```

### Specific test class or method

```bash
python -m pytest tests/test_backtest_engine.py::TestBacktestResult::test_result_structure -v
```

### With coverage report

```bash
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

This reports line-level coverage for all modules under `src/` and shows which lines are not covered.

### Filter by keyword

```bash
python -m pytest tests/ -v -k "leakage"
```

This runs only tests whose names match the substring "leakage" (e.g., `test_leakage.py`, `test_oof_leakage.py`).

No watch mode script is configured. Use pytest-watch (`ptw`) externally if desired.

## Writing New Tests

### File naming convention

All test files follow the pattern `test_<module_name>.py` in the `tests/` directory. The module name matches the source module being tested:

| Test file | Source module |
|---|---|
| `test_domain.py` | `src/domain/models.py`, `src/domain/types.py` |
| `test_feature_engine.py` | `src/features/feature_engine.py` |
| `test_parquet_store.py` | `src/db/parquet_store.py` |
| `test_backtest_engine.py` | `src/backtest/engine.py` |
| `test_stacked_ensemble.py` | `src/models/stacked_ensemble.py` |

### Test structure

Tests are organized into classes by concern (`class TestXxx:`), using descriptive method names in Japanese or English. Example pattern:

```python
"""src/domain モジュールのテスト"""

import numpy as np
import pytest

from domain.types import BetType, Surface


class TestEnums:
    def test_surface_values(self):
        assert Surface.TURF.value == "turf"
        assert Surface.DIRT.value == "dirt"
```

### Fixture patterns

Fixtures are defined locally in each test file (no shared `conftest.py`). Common patterns:

- **Data fixtures** -- `@pytest.fixture` functions that return `pd.DataFrame` instances with realistic column names (using raw column names from Parquet such as `trackcd`, `kyori`, `umaban`):

```python
@pytest.fixture
def sample_race_df() -> pd.DataFrame:
    return pd.DataFrame({
        "race_id": ["2024032405030208"] * 18,
        "trackcd": [11] * 18,
        "kyori": [1600] * 18,
        # ...
    })
```

- **Mock fixtures** -- `MagicMock` objects with `spec=` to enforce interface contracts:

```python
@pytest.fixture
def mock_models() -> MagicMock:
    models = MagicMock(spec=TrainedModelsV5)
    models.regime_detector = MagicMock()
    return models
```

- **Filesystem fixtures** -- pytest's built-in `tmp_path` for Parquet I/O tests:

```python
@pytest.fixture
def store(tmp_path: Path) -> ParquetStore:
    return ParquetStore(data_dir=str(tmp_path))
```

### Mocking conventions

All tests are database-free. The project standard is `unittest.mock` (imported as `from unittest.mock import MagicMock, patch`). No external mocking libraries are used.

- Use `MagicMock(spec=ClassName)` to lock mocks to the real interface.
- Use `patch("module.path.ClassName")` as a context manager or decorator for module-level patches.
- Helper functions for creating mock objects follow the `_make_mock_*` naming pattern:

```python
def _make_mock_store() -> MagicMock:
    mock_store = MagicMock()
    return mock_store
```

## Coverage Requirements

No minimum coverage threshold is configured. The project has `pytest-cov` available and the recommended invocation produces a coverage report:

```bash
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

There is no `fail_under` setting in `pyproject.toml` and no `.coveragerc` file. Coverage is informational, not enforced.

## CI Integration

No CI/CD pipeline is detected in this repository. There are no `.github/workflows/` files present.

Tests are run locally before commits using the commands above. The project relies on developer discipline rather than automated CI enforcement for test execution.

## Test Suite Overview

The test suite contains **96 test files** with approximately **1,255 test methods** organized into **287 test classes**. Tests cover all major subsystems:

| Area | Example test files | Focus |
|---|---|---|
| Domain models | `test_domain.py` | Enums, dataclasses, type definitions |
| Feature engineering | `test_feature_engine.py`, `test_horse_history_features.py`, `test_odds_dynamics_features.py` | Feature generation, column mapping, historical aggregation |
| ML models | `test_stacked_ensemble.py`, `test_ev_correction.py`, `test_two_stage_return_model.py` | Model training, prediction, calibration |
| Backtesting | `test_backtest_engine.py`, `test_backtest_report.py`, `test_walk_forward_cv.py` | Backtest logic, multi-year validation, reporting |
| Betting strategy | `test_win_strategy.py`, `test_place_strategy.py`, `test_wide_strategy.py`, `test_stake_calculator.py` | Bet selection, stake sizing, drawdown control |
| Data layer | `test_parquet_store.py`, `test_db.py`, `test_readers.py`, `test_etl.py` | Parquet I/O, database queries, ETL pipeline |
| Paper trading | `test_paper_predictor.py`, `test_paper_trading_guards.py`, `test_paper_reconciler.py` | Simulated trading, guardrails, reconciliation |
| Monitoring | `test_model_monitor.py`, `test_diagnostic_logger.py` | Model drift detection, logging |
| Automation | `test_scheduler.py`, `test_auto_retrain_trigger.py`, `test_race_watcher.py` | Scheduling, triggers, race event handling |
| Config validation | `test_settings.py`, `test_leakage.py`, `test_oof_leakage.py` | Settings integrity, data leakage prevention |
