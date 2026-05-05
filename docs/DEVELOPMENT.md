<!-- generated-by: gsd-doc-writer -->

# Development Guide

## Local Setup

1. **Install runtime** — Python 3.11 is required, managed via [mise](https://mise.jdx.dev/):

   ```bash
   mise install
   mise activate
   ```

2. **Clone and install** (editable mode with dev dependencies):

   ```bash
   git clone <repository-url>
   cd keiba-ai
   pip install -e ".[dev]"
   ```

   This installs `ruff`, `mypy`, and `ipykernel` as development tools. For scraping functionality, use `pip install -e ".[scraping]"` to add `playwright`.

3. **Configure environment** — Copy the example env file and set your PostgreSQL password:

   ```bash
   cp .env.example .env
   # Edit .env and set PGPASSWORD=<your-password>
   ```

   The `.env` file only requires `PGPASSWORD` for ETL and paper trading. Slack notifications (`SLACK_WEBHOOK_URL`) are optional.

4. **Verify installation** — Run the test suite to confirm everything works:

   ```bash
   python -m pytest tests/ -v
   ```

   All tests use mocks and require no database connection.

## Build Commands

### Testing and Quality

| Command | Description |
|---------|-------------|
| `python -m pytest tests/ -v` | Run the full test suite |
| `python -m pytest tests/test_domain.py -v` | Run a single test file |
| `python -m pytest tests/ -v --cov=src --cov-report=term-missing` | Run tests with coverage report |
| `ruff check src/ tests/` | Lint source and test files |
| `ruff format --check src/ tests/` | Check formatting without writing |
| `ruff format src/ tests/` | Apply formatting fixes |
| `mypy src/` | Run type checking on source code |

### Pipeline Scripts

| Command | Description |
|---------|-------------|
| `python scripts/run_etl.py --start 20140101 --end 20231231` | ETL: PostgreSQL (EveryDB2) to Parquet (~10 min) |
| `python scripts/run_train.py --start 20200101 --end 20231231` | Train ML models (~44 min) |
| `python scripts/run_backtest.py --train-start 20200101 --train-end 20231231 --test-start 20240101 --test-end 20241231` | Single-year backtest (~57 min) |
| `python scripts/run_backtest.py --years 2023 2024 2025 --train-window 4` | Multi-year backtest |
| `python scripts/run_wf_validation.py` | Walk-forward validation (2-fold: 2024, 2025 test) |
| `python scripts/run_tuning.py --model win_hit --start 20200101 --end 20231231 --trials 50` | Optuna hyperparameter tuning (models: `win_hit`, `win_return`, `place_hit`, `place_return`, `ability`) |
| `python scripts/run_paper_trading.py --mode setup --date 2026-04-05` | Paper trading: check race schedule for the day |
| `python scripts/run_paper_trading.py --mode predict --date 2026-04-05` | Paper trading: feature generation, inference, bet saving, Slack notification |
| `python scripts/run_paper_trading.py --mode reconcile --date 2026-04-05` | Paper trading: reconcile results with race outcomes, compute ROI |
| `python scripts/run_paper_trading.py --mode dry-run --date 2024-07-13` | Paper trading: dry-run on historical data (no DB required) |
| `python scripts/run_paper_trading.py --mode diagnose --start 2024-07-01 --end 2024-07-31` | Paper trading: diagnostic inference on Parquet data (EveryDB2 bypass) |
| `python scripts/run_strategy_optimization.py --n-trials 100` | Strategy parameter optimization via Optuna TPE |

### Analysis and Utility Scripts

| Command | Description |
|---------|-------------|
| `python scripts/analyze_feature_importance.py` | Feature importance analysis |
| `python scripts/analyze_odds_movement.py` | Odds movement analysis |
| `python scripts/compare_bt_pt_features.py` | Compare backtest vs paper-trading features |
| `python scripts/precompute_career_stats.py` | Pre-compute horse career statistics |
| `python scripts/precompute_sire_stats.py` | Pre-compute sire statistics |

## Code Style

### Linting and Formatting

- **Ruff** — Combined linter and formatter. Configuration in `pyproject.toml`:
  - Target: Python 3.11
  - Line length: 100 characters
  - Enabled rules: `E`, `F`, `I`, `N`, `W` (pycodestyle errors, pyflakes, isort, pep8-naming, pycodestyle warnings)
  - Run lint: `ruff check src/ tests/`
  - Run format: `ruff format src/ tests/`

### Type Checking

- **mypy** — Static type checker with strict settings. Configuration in `pyproject.toml`:
  - `disallow_untyped_defs = true` — All functions must have type annotations
  - `warn_return_any = true`
  - `warn_unused_configs = true`
  - Run: `mypy src/`

### Test Conventions

- Tests are in the `tests/` directory, named `test_*.py`
- All tests use `unittest.mock` — no database connection required
- Python path is configured via `pyproject.toml`: `pythonpath = [".", "src"]`
- Import from source uses paths like `from db.parquet_store import ParquetStore`, `from db.readers import load_races`, or `from domain.types import BetType` (not `from src.db...`)

## Branch Conventions

The main development branch is `main`. Based on git history:

- Feature branches: `feat/<description>` (e.g., `feat/parquet-migration`, `feat/phase-b-features`, `feat/etl-realdata-validation`)
- Performance branches: `perf/<description>` (e.g., `perf/pipeline-optimization`)
- Branch prefixes follow Conventional Commits categories

## PR Process

No `.github/PULL_REQUEST_TEMPLATE.md` or formal review process is documented. General guidelines:

- Follow **Conventional Commits** format for commit messages. Japanese descriptions are acceptable:
  - `feat(scope): description` — new feature
  - `fix(scope): description` — bug fix
  - `docs(scope): description` — documentation
  - `test(scope): description` — tests
  - `chore: description` — maintenance
  - `refactor(scope): description` — code refactoring
  - Scopes are optional but commonly used (e.g., `ensemble`, `backtest`, `betting`, `phase-13`)
- Ensure all tests pass: `python -m pytest tests/ -v`
- Run lint and type checks: `ruff check src/ tests/` and `mypy src/`
- Keep PRs focused on a single concern
- Include test coverage for new functionality
