<!-- generated-by: gsd-doc-writer -->

# Getting Started

This guide walks you through setting up the keiba-ai prediction system from a fresh clone to running your first backtest.

## Prerequisites

- **Python >= 3.11** — The project targets Python 3.11, managed via [mise](https://mise.jdx.dev/) (recommended) or any Python 3.11+ installation.
- **mise** (optional but recommended) — Runtime version manager. Install from <https://mise.jdx.dev/>.
- **PostgreSQL** with EveryDB2/JRA-VAN DataLab data loaded — Required for the ETL step only. The ML pipeline and tests run entirely on Parquet files and do not need a database connection.
- **pip** — Python package installer (included with Python).

## Installation Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/hiromu1018ks/keiba-ai.git
   cd keiba-ai
   ```

2. **Set up Python 3.11 with mise (recommended)**

   ```bash
   mise install
   mise activate
   ```

   The `mise.toml` file pins Python to version 3.11. If you prefer not to use mise, ensure `python` resolves to Python 3.11+ on your system.

3. **Install the project with dev dependencies**

   ```bash
   pip install -e ".[dev]"
   ```

   This installs all runtime dependencies (pandas, LightGBM, scikit-learn, MLflow, etc.) plus dev tools (ruff, mypy, ipykernel).

4. **Configure environment variables**

   Copy `.env.example` to `.env` and fill in the database password:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and set `PGPASSWORD` to your PostgreSQL password. This variable is only needed for the ETL step (`scripts/run_etl.py`). The ML pipeline and tests do not require it.

   Alternatively, export it directly:

   ```bash
   export PGPASSWORD=<your-password>
   ```

5. **Verify installation**

   Run the test suite to confirm everything is set up correctly:

   ```bash
   python -m pytest tests/ -v
   ```

   All tests use mocks and do not require a database connection.

## First Run

### Option A: Run tests (no database required)

The fastest way to verify the project is working:

```bash
python -m pytest tests/ -v
```

### Option B: Run a full pipeline (requires Parquet data)

If you have Parquet data files in the `data/` directory, you can run the ML pipeline:

```bash
# Step 1: ETL — Only if you need to extract fresh data from PostgreSQL
python scripts/run_etl.py --start 20140101 --end 20231231

# Step 2: Train the model
python scripts/run_train.py --start 20200101 --end 20231231

# Step 3: Run a backtest
python scripts/run_backtest.py \
  --train-start 20200101 --train-end 20231231 \
  --test-start 20240101 --test-end 20241231
```

## Common Setup Issues

### 1. `python` resolves to the wrong version

If `python --version` shows anything other than 3.11+, the installed packages may fail. Use mise (`mise install && mise activate`) or create a virtual environment with the correct version:

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
pip install -e ".[dev]"
```

### 2. `PGPASSWORD` not set when running ETL

The `run_etl.py` script connects to PostgreSQL at `localhost:5432/everydb2`. If `PGPASSWORD` is empty, the connection will fail. Set it in `.env` or export it before running the script. This is not needed for training, backtesting, or tests.

### 3. Missing Parquet data files

The ML pipeline expects Parquet files under `data/raw/`, `data/odds/`, and `data/features/`. If these directories are empty, run the ETL step first, or obtain pre-built Parquet data files. Running tests does not require any data files.

### 4. `pip install` fails on LightGBM or psycopg2-binary

On some platforms, LightGBM or psycopg2-binary may require system-level build tools. On Windows, install the Visual C++ Build Tools. On Linux, install `libpq-dev` and `build-essential`.

## Next Steps

- **ARCHITECTURE.md** — Understand the system components and data flow.
- **CONFIGURATION.md** — Learn about all configuration options in `config/settings.yaml`.
- **DEVELOPMENT.md** — Set up your development environment with linting, formatting, and type checking.
