# Odds Movement Analysis Script Design

**Date:** 2026-04-16
**Status:** Approved
**Approach:** Standalone analysis script (Approach A)

## 1. Objective

レース直前のオッズ変動（特に急落・急騰）が複勝率や回収率にどう影響するかを統計的に分析するスタンドアロンスクリプトを作成する。

## 2. Background

### Research Findings (EXa MCP Survey)

1. **ヒノくん (note.com, 2025)**: 締切2分前〜確定でオッズが下がった馬は勝率13.1%/複勝率37.0%、上がった馬は勝率5.5%/複勝率17.3%。ただし「オッズ半減＝期待値半減」のジレンマ（タコが自分の足を食う現象）も指摘
2. **BloodHorse**: 大型競馬場で発走5分前に10倍台の馬が人気馬になることはほぼない。2-3ポイントの変動ですら「大きな動き」
3. **Punter2Pro (CLV)**: 59,281件でClosing Lineを上回るベット群は+19.62% ROI。「最後まで短縮する馬を最大化」が鍵
4. **ちゃんわ**: 売上シェアが小さい券種ほど急落しやすい → 複勝は単勝より安定

### Project Data Availability

| Source | File | Key Columns |
|--------|------|-------------|
| Time series odds | `data/odds/jodds_tanpuku/` (partitioned) | race_id, umaban, tanodds, fukuoddslow, tanninki, happyotime, year, month |
| Race results | `data/raw/entries.parquet` | race_id, umaban, kakuteijyuni, ninki, kisyucode, chokyosicode, surface, ... |
| Place payouts | `data/raw/payouts.parquet` | race_id, payfukusyoumaban1-5, payfukusyopay1-5, datakubun |

Time granularity: `happyotime` = "MMDDHHmm" (8 chars), minute-level snapshots.
Typical range: t-60min (first point) to t-10min (last regular point), with late-money at t-3min/t-2min.

## 3. Scope

- **Period:** 2023-2025 (jodds_tanpuku available)
- **Target:** JRA races only (NAR: jyocd >= 30 excluded)
- **Focus:** Place betting (複勝) — win odds movement as signal for place outcome
- **Output:** Console tables + CSV files

## 4. Architecture

```
┌─────────────────────┐
│ jodds_tanpuku       │  Time series odds (partitioned parquet)
│  year=2023..2025    │
└─────────┬───────────┘
          │ per (race_id, umaban): sorted by happyotime
          ▼
┌─────────────────────────────────┐
│ Movement Feature Computation     │
│                                  │
│  odds_drop_60_10   = (t-60 - t-10) / t-60   │
│  odds_drop_30_10   = (t-30 - t-10) / t-30    │
│  odds_drop_10_final = (t-10 - final) / t-10  │
│  pop_change_30_10  = popularity rank change   │
│                                  │
│  Classify: steamer(>=20% drop) / stable / drifter(>=20% rise) │
└─────────┬───────────────────────┘
          │ LEFT JOIN
          ├──────────────────┐
          ▼                  ▼
┌──────────────────┐  ┌──────────────────┐
│ entries.parquet  │  │ payouts.parquet  │
│ (finishing pos,  │  │ (place payout)   │
│  jockey, trainer,│  │                   │
│  surface, etc.)  │  │                   │
└────────┬─────────┘  └────────┬─────────┘
         │                    │
         ▼                    ▼
┌────────────────────────────────────────┐
│ Cross-tabulation Analysis              │
│                                         │
│  1. Basic stats: by movement bucket     │
│     place rate, ROI, win rate           │
│  2. Jockey/Trainer: drop frequency x    │
│     performance                         │
│  3. Race conditions: turf/dirt, venue,  │
│     field size, distance                │
└────────────────┬───────────────────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Console + CSV   │
        └─────────────────┘
```

## 5. Time Window Definition

Since exact post times are not always available in the data, use **quantile-based** time points per (race_id, umaban):

| Label | Definition | Approximate time |
|-------|-----------|------------------|
| `early` | First data point | ~t-60 min |
| `mid` | 50th percentile index | ~t-30 min |
| `late` | 90th percentile index | ~t-10 min |
| `final` | Last data point | Pre-post (closest to race start) |

Minimum 5 data points required per horse (`--min-points 5`).

## 6. Classification Thresholds

Primary threshold: **20%** change (based on research consensus).

Additional reporting at **15%** and **25%** for sensitivity analysis.

| Category | Condition |
|----------|-----------|
| Steamer (strong drop) | odds_drop >= 40% |
| Moderate drop | 25% <= odds_drop < 40% |
| Mild drop | 15% <= odds_drop < 25% |
| Stable | -15% < odds_drop < 15% |
| Mild rise | 15% <= odds_rise < 25% |
| Moderate rise | 25% <= odds_rise < 40% |
| Drifter (strong rise) | odds_rise >= 40% |

## 7. Analysis Dimensions

### 7-1. Basic Statistics (Required)

**Table A: Performance by Movement Bucket**

Columns: bucket, count, place_rate, avg_place_odds, place_roi, win_rate

**Table B: Popularity Segment x Movement Type Cross-tab**

Segments: 1-3rd favorite, 4-7th, 8th+

**Table C: Time Window Predictive Power Comparison**

Compare 60->10min, 30->10min, 10->final windows by optimal threshold and ROI.

### 7-2. Jockey/Trainer Analysis

**Table D: Top/Bottom 20 Jockeys by Late Drop Pattern**

Columns: jockey_code, rides, drop_rate(%), drop_place_rate, stable_place_rate, diff

Identifies: "jockeys whose drops are trustworthy" vs "jockeys whose drops are overreactions"

Same table structure for trainers.

### 7-3. Race Condition Analysis

**Table E: Condition Matrix**

Dimensions:
- Surface: turf / dirt
- Venue group: Tokyo/Hanshin / others
- Distance: short (<=1400m) / mid (1400-2000m) / long (>=2000m)
- Field size: <=8 / 9-12 / 13+
- Track condition: good / slightly soft / soft / heavy

## 8. Output Structure

```
output/odds_movement_analysis_YYYYMMDD/
├── summary_main.csv          # Tables A + B + C
├── by_jockey.csv             # Table D (jockey)
├── by_trainer.csv            # Table D (trainer)
├── by_race_condition.csv     # Table E
├── detail_records.csv        # Full per-horse detail (with --detail flag)
└── analysis_log.txt          # Execution log with parameters
```

Console output uses pandas display or tabulate for readable tables.

## 9. Technical Implementation

### 9-1. Script Structure

```python
# scripts/analyze_odds_movement.py

def main()                           # CLI arg parsing -> pipeline orchestration
def load_time_series(start, end)     # Read jodds_tanpuku via pyarrow Dataset API
def load_entries()                   # Read entries.parquet
def load_payouts()                   # Read payouts.parquet
def compute_movement_features(ts_df) # Core: vectorized odds movement calculation
def classify_movement(df)            # Bucket assignment
def join_results(df, entries, payouts)# Merge with results and payouts
def analyze_basic_stats(df)          # Section 7-1
def analyze_jockey_trainer(df)       # Section 7-2
def analyze_race_conditions(df)      # Section 7-3
def print_summary(results)            # Console output
def save_csv(results, output_dir)     # CSV file output
```

### 9-2. Key Implementation Details

**Efficient time series loading:**
Use pyarrow Dataset API with Hive partitioning predicate pushdown:

```python
import pyarrow.dataset as ds
dataset = ds.dataset("data/odds/jodds_tanpuku/", partitioning="hive")
ts_df = dataset.to_table(
    where=(ds.field("year") >= 2023) & (ds.field("year") <= 2025)
).to_pandas()
```

**Vectorized movement computation (no iteration):**
```python
g = ts_sorted.groupby(["race_id", "umaban"], sort=False)
features = g.agg(
    early_odds=("tanodds", "first"),
    mid_odds=("tanodds", lambda x: x.iloc[len(x)//2]),
    late_odds=("tanodds", lambda x: x.iloc[int(len(x)*0.9)]),
    final_odds=("tanodds", "last"),
    early_pop=("tanninki", "first"),
    late_pop=("tanninki", lambda x: x.iloc[int(len(x)*0.9)]),
    n_points=("tanodds", "count"),
).reset_index()
```

**Place ROI from actual payouts:**
```python
# Use payouts.parquet actual payment amounts
# Match horse number to payfukusyoumaban1-5 columns
def get_place_payout(row, payouts):
    for i in range(1, 6):
        if row["umaban"] == payouts[f"payfukusyoumaban{i}"]:
            return payouts[f"payfukusyopay{i}"]
    return 0.0
```

### 9-3. Data Quality Filters

- Exclude NAR races: `jyocd >= 30`
- Minimum data points: `n_points >= 5` (configurable)
- Confirmed results only: `datakubun == '0'`
- Exclude NaN odds values

### 9-4. CLI Interface

```bash
# Basic execution
python scripts/analyze_odds_movement.py --start 2023 --end 2025

# With options
python scripts/analyze_odds_movement.py \
  --start 2024 --end 2025 \
  --output-dir output/odds_analysis \
  --drop-threshold 0.20 \
  --min-points 5 \
  --detail
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--start` | 2023 | Start year (inclusive) |
| `--end` | 2025 | End year (inclusive) |
| `--output-dir` | `output/odds_movement_analysis_{date}` | Output directory |
| `--drop-threshold` | 0.20 | Primary classification threshold |
| `--min-points` | 5 | Minimum time series points per horse |
| `--detail` | False | Output full detail CSV |

## 10. Dependencies

- Python 3.11+ (project standard)
- pandas, pyarrow (already in project dependencies)
- tabulate (for console table formatting, add if not present)
- No database connection required
- No ML framework required

## 11. Non-Goals

- Real-time/live odds monitoring (separate concern handled by OddsCollector)
- Win bet optimization (focus is place betting)
- Integration into ML pipeline training features (can be added later)
- Web UI or interactive visualization (console/CSV only)
