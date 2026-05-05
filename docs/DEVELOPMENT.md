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

3. **Configure environment** — Copy the example env file and set your PostgreSQL password:

   ```bash
   cp .env.example .env
   # Edit .env and set PGPASSWORD=<your-password>
   ```

4. **Verify installation** — Run the test suite to confirm everything works:

   ```bash
   python -m pytest tests/ -v
   ```

   All tests use mocks and require no database connection.

## Build Commands

| Command | Description |
|---------|-------------|
| `python -m pytest tests/ -v` | Run the full test suite |
| `python -m pytest tests/test_domain.py -v` | Run a single test file |
| `python -m pytest tests/ -v --cov=src --cov-report=term-missing` | Run tests with coverage report |
| `ruff check src/ tests/` | Lint source and test files |
| `ruff format --check src/ tests/` | Check formatting without writing |
| `ruff format src/ tests/` | Apply formatting fixes |
| `mypy src/` | Run type checking on source code |
| `python scripts/run_etl.py --start 20140101 --end 20231231` | ETL: PostgreSQL (EveryDB2) to Parquet (~10 min) |
| `python scripts/run_train.py --start 20200101 --end 20231231` | Train ML models (~44 min) |
| `python scripts/run_backtest.py --train-start 20200101 --train-end 20231231 --test-start 20240101 --test-end 20241231` | Single-year backtest (~57 min) |
| `python scripts/run_backtest.py --years 2023 2024 2025 --train-window 4` | Multi-year backtest |
| `python scripts/run_wf_validation.py` | Walk-forward validation |
| `python scripts/run_tuning.py` | Optuna hyperparameter tuning |
| `python scripts/run_paper_trading.py` | Paper trading simulation |
| `python scripts/run_strategy_optimization.py` | Strategy parameter optimization |

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

## Branch Conventions

The main development branch is `main`. Based on existing branch patterns:

- Feature branches: `feat/<description>` (e.g., `feat/parquet-migration`, `feat/phase-b-features`)
- Performance branches: `perf/<description>` (e.g., `perf/pipeline-optimization`)
- Fix branches follow Conventional Commits prefixes (`fix/`, `docs/`, `chore/`, `test/`)

## PR Process

No `.github/PULL_REQUEST_TEMPLATE.md` or formal review process is documented. General guidelines:

- Follow **Conventional Commits** format for commit messages (Japanese descriptions are acceptable):
  - `feat(scope): description`
  - `fix(scope): description`
  - `docs(scope): description`
  - `test(scope): description`
  - `chore: description`
- Ensure all tests pass: `python -m pytest tests/ -v`
- Run lint and type checks: `ruff check src/ tests/` and `mypy src/`
- Keep PRs focused on a single concern
- Include test coverage for new functionality
