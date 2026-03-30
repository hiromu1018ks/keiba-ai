# Backtest Report Generator Design

Date: 2026-03-30
Status: Draft

## Problem

Backtest results currently output only 7 aggregate metrics to `backtest_result.json`. The
`bet_history` list (per-bet detail with race_id, bet_type, umaban, stake, odds, result) is
computed in memory but discarded during serialization. This makes it impossible to verify
correctness or understand WHERE profit/loss comes from.

## Goal

Generate a self-contained HTML report from backtest results that shows:
1. Summary KPI cards (ROI, win rate, profit, max DD, final bankroll)
2. Bankroll trajectory chart with drawdown overlay
3. Monthly dashboard (ROI, bet count, win rate per month)
4. Condition analysis (surface × distance band, popularity band, EV band)
5. Full bet detail table (sortable, filterable, paginated)

## Approach

- **Data flow:** `run_backtest.py --report` → save `bet_history.json` → `BacktestReportGenerator` → HTML
- **HTML generation:** Jinja2 template + embedded data (no external file dependencies)
- **Charts:** Chart.js (CDN) for line/bar charts
- **Tables:** DataTables (CDN) for sortable/filterable bet detail

## Data Flow

```
run_backtest.py --report
  → BacktestEngine.run()
    → BacktestResult (aggregate + bet_history with enriched fields)
  → data/backtest/bet_history.json  (NEW: full bet history)
  → data/backtest/backtest_result.json  (MOVED from project root)
  → BacktestReportGenerator.generate(result, bet_history)
    → data/backtest/backtest_report.html  (NEW: self-contained report)
```

**Output consolidation:** When `--report` is active, ALL output files go to `data/backtest/`:
- `data/backtest/bet_history.json`
- `data/backtest/backtest_result.json`
- `data/backtest/backtest_report.html`

## bet_history Schema Extension

Current fields in each bet dict (populated in engine loop):
- `race_id` (str): e.g. "20240101010111"
- `bet_type` (str): "place" (currently only type generated)
- `umaban` (int): horse number
- `stake` (float): amount wagered (always 100.0 in backtest)
- `odds` (float): odds at bet time
- `result` (float): payout amount (0.0 if lost)

Fields to ADD **inside the engine loop** (at `bet_history.append()` time, where
`race_df_single`, `surface_key`, and row data are all still available):
- `surface` (str): `"turf"` or `"dirt"` — from `surface_key` in engine loop (Surface enum values)
- `distance` (int): race distance in meters — from `race_df_single` row
- `ev` (float): expected value at bet time — from `_generate_bets()` local variable
- `popularity` (int): from `race_df_single["popularity_rank"]` (already computed by FeatureEngine)
- `bankroll_after` (float): bankroll balance after this bet settles — from engine loop's `bankroll` variable

Fields derived by `BacktestReportGenerator._derive_fields()` (post-hoc, no data access needed):
- `race_date` (str): extracted from first 8 chars of `race_id`, format "YYYY-MM-DD"
- `profit` (float): `result - stake` (positive for wins, negative for losses)
- `is_win` (bool): `result > 0`

**Key design choice:** The engine loop is the ONLY place where `surface`, `distance`,
`popularity`, and `ev` are simultaneously available. Adding them at `bet_history.append()`
time avoids re-querying data or re-computing features.

## CLI Interface

```bash
# Without report (existing behavior, unchanged)
python scripts/run_backtest.py \
  --train-start 20200101 --train-end 20231231 \
  --test-start 20240101 --test-end 20241231

# With report (NEW)
python scripts/run_backtest.py \
  --train-start 20200101 --train-end 20231231 \
  --test-start 20240101 --test-end 20241231 \
  --report
```

The `--report` flag triggers:
1. Saving `bet_history.json` to `data/backtest/`
2. Moving `backtest_result.json` output to `data/backtest/`
3. Generating `backtest_report.html` to `data/backtest/`
4. Printing the report path to stdout

## HTML Report Structure

Single HTML file with 5 sections. Charts require internet (CDN); fallback message shown
if CDN resources cannot load.

### Section 1: Summary KPI Cards
- ROI (%), Win rate (%), Total profit (¥), Max drawdown (%), Final bankroll (¥)
- Test period, train period, generation timestamp

### Section 2: Bankroll Trajectory Chart
- Chart.js line chart
- X-axis: date, Y-axis: cumulative bankroll
- Drawdown area overlay (red shading below peak)
- Data: `bankroll_after` field from each bet (already captured in engine loop)

### Section 3: Monthly Dashboard
- Side-by-side charts:
  - Monthly ROI bar chart (green/red based on positive/negative)
  - Monthly bet count bar chart
- Table below: month, bets, wins, win rate, stake, return, ROI
- Note: `BacktestResult.monthly_returns` is always empty `{}` — monthly stats are
  computed by `BacktestReportGenerator` from `bet_history`

### Section 4: Condition Analysis
- Cross-tabulation tables:
  - Surface × distance band: ROI, win rate, bet count per cell
    - Surface values: `"turf"` / `"dirt"` (displayed as "芝" / "ダート" in HTML via lookup)
  - Popularity band (1-3, 4-6, 7+): ROI, win rate, avg payout
  - EV band (<1.0, 1.0-1.2, 1.2-1.5, 1.5+): ROI, win rate, avg payout
- Heatmap coloring for ROI cells

### Section 5: Bet Detail Table
- DataTables-powered sortable/filterable/paginated table
- Columns: date, race_id, umaban, surface, distance, popularity, EV, odds, stake, payout, profit, result (win/loss)
- Color coding: green rows for wins, default for losses
- Default sort: by date descending
- Search/filter: by race_id, umaban, surface, result

### Footer
- Generation command, commit hash, Python version

## File Structure (New Files)

```
src/backtest/report.py              # BacktestReportGenerator class
src/backtest/templates/report.html  # Jinja2 template
tests/test_backtest_report.py       # Unit tests
```

Modified files:
```
scripts/run_backtest.py              # Add --report flag, call generator, move output dir
src/backtest/engine.py               # Extend bet_history with surface/distance/ev/popularity/bankroll_after
pyproject.toml                       # Add jinja2 to dependencies
```

## BacktestReportGenerator API

```python
class BacktestReportGenerator:
    def __init__(self, output_dir: Path) -> None:
        """Initialize with output directory for HTML report."""

    def generate(self, result: BacktestResult, bet_history: list[dict]) -> Path:
        """Generate HTML report from backtest result and bet history.

        Returns path to generated HTML file.
        """

    def _derive_fields(self, bet_history: list[dict]) -> list[dict]:
        """Add derived fields: race_date, profit, is_win.
        Does NOT re-query data — only derives from existing fields."""

    def _compute_monthly_stats(self, bets: list[dict]) -> list[dict]:
        """Aggregate bets by month: ROI, win rate, bet count, stake, return."""

    def _compute_condition_stats(self, bets: list[dict]) -> dict:
        """Aggregate by surface×distance, popularity, EV bands."""

    def _compute_bankroll_series(self, bets: list[dict]) -> list[dict]:
        """Extract bankroll_after per date for charting.
        Uses bankroll_after already captured in engine loop."""
```

## Dependencies

- **jinja2**: Must be added explicitly to `pyproject.toml` dependencies (do not rely on
  transitive dependency via MLflow)
- **Chart.js**: CDN only (https://cdn.jsdelivr.net/npm/chart.js)
- **DataTables**: CDN only (https://cdn.datatables.net)

**Limitation:** Charts and table interactivity require internet access for CDN resources.
If offline, the HTML structure and data still render, but charts/interactivity are unavailable.

## Testing

Test file: `tests/test_backtest_report.py`

All tests use mocks (no DB, no files on disk for unit tests):

| Test | What it verifies |
|------|-----------------|
| `test_derive_fields` | race_date extraction, profit/is_win computation |
| `test_compute_monthly_stats` | Monthly aggregation correctness |
| `test_compute_monthly_stats_empty` | Edge: zero bets → empty result |
| `test_compute_monthly_stats_all_losses` | Edge: 100% loss month |
| `test_compute_condition_stats` | Surface/distance/popularity/EV band analysis |
| `test_compute_bankroll_series` | Cumulative bankroll chart data |
| `test_compute_bankroll_series_single_bet` | Edge: single bet |
| `test_html_generation` | Template renders without error, contains key sections |
| `test_bet_history_serialization` | JSON save/load round-trip |
| `test_cli_report_flag` | --report flag triggers report generation |
| `test_engine_bet_history_enrichment` | Engine loop populates surface/distance/ev/popularity/bankroll_after |

## Out of Scope

- Notebook analysis cells (Phase 2, separate task)
- Win/Wide bet analysis (backtest only generates place bets currently)
- Real-time monitoring or dashboards
- Comparison of multiple backtest runs
- PDF export
