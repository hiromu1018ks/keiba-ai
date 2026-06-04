# Phase 41: Shadow Comparison Framework - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-28
**Phase:** 41-Shadow Comparison Framework
**Areas discussed:** Comparison runner architecture, LightGBM LambdaRank shadow, Metrics output format, Baseline model strategy

---

## Comparison Runner Architecture

| Option | Description | Selected |
|--------|-------------|----------|
| Standalone script | scripts/run_shadow_comparison.py. Independent entry point with --baseline-dir/--shadow-dir. No changes to existing scripts. | |
| New class + thin CLI | ShadowComparisonFramework class in src/backtest/shadow_comparison.py + scripts/run_shadow_comparison.py wrapper. Testable class, clear operator entry point. | ✓ |
| run_backtest.py extension | Add --shadow-comparison flag to existing CLI. Would bloat run_backtest.py. | |

**User's choice:** New class + thin CLI. Detailed CLI args specified: --baseline-root, --shadow-root, --folds, --train-window, --betting-target, --output-dir, --report.

**Notes:** User emphasized keeping run_backtest.py focused on single-pipeline backtests. Framework should load both models, run BacktestEngine twice, collect metrics, align diagnostics, emit JSON + diff artifacts.

### Model Loading

| Option | Description | Selected |
|--------|-------------|----------|
| Pre-trained only | Both models must be pre-trained. ModelLoader.load_from_dir() × 2. | ✓ |
| Training included | Optionally train inside comparison. Doubles runtime, harder to debug. | |

**User's choice:** Pre-trained only. Training and evaluation should not be mixed. Training inside comparison runner makes failures harder to debug.

### Per-race Alignment

| Option | Description | Selected |
|--------|-------------|----------|
| Post-hoc merge | Run BacktestEngine twice, then merge by race_id + umaban. | ✓ |
| Result-only comparison | Compare aggregate BacktestResult only, no per-race alignment. | |

**User's choice:** Post-hoc alignment at two levels: race-level (selected horse comparison) and horse-level (full diagnostic row merge). If bet_history lacks required columns, Phase 41 should add diagnostic output explicitly.

### Fixed Fold Definition

| Option | Description | Selected |
|--------|-------------|----------|
| 2-fold fixed (2024/2025) | Fold 2024: train 2020-2023/test 2024. Fold 2025: train 2021-2024/test 2025. | ✓ |
| Train-window rolling | --years 2024 2025 --train-window 4. More realistic but less controlled. | |

**User's choice:** 2-fold fixed matching WF validation. Explicit fold definition prevents accidental changes. CLI: --folds 2024 2025 with --baseline-root/--shadow-root resolving year subdirectories.

---

## LightGBM LambdaRank Shadow

| Option | Description | Selected |
|--------|-------------|----------|
| Implement in Phase 41 | 3-way comparison: Ridge vs LambdaRank vs baseline. Significant scope increase. | |
| Defer to v2.2+ | Ridge vs baseline only. Framework N-way capable for future. | ✓ |
| Skeleton only | Training code skeleton without actual training. | |

**User's choice:** Defer to v2.2+. Phase 41 focuses on comparison infrastructure correctness. Adding a nonlinear ranker at the same time makes failures ambiguous. Framework designed for N-way comparison.

---

## Metrics Output Format

### Output Artifacts

| Option | Description | Selected |
|--------|-------------|----------|
| JSON + HTML + CSV | JSON metrics + HTML report + CSV diff. Follows existing BT report pattern. | ✓ |
| JSON only | Minimal output. No HTML report. | |
| JSON + Markdown | Lightweight Markdown instead of HTML template. | |

**User's choice:** JSON + Parquet/CSV + HTML. Five artifacts specified: shadow_comparison_result.json, shadow_race_diff.parquet + .csv, shadow_horse_diff.parquet, shadow_comparison_report.html, shadow_manifest.json.

### Aggregation Granularity

| Option | Description | Selected |
|--------|-------------|----------|
| Standard breakdown | overall + fold/year + surface + odds_band + prob_rank_band. | ✓ |
| Minimal breakdown | overall + fold/year only. | |

**User's choice:** Standard breakdown plus value_score_band and selected_changed vs unchanged. CLV computed when both betting_line_odds and closing_odds available, formula: closing_odds/betting_odds - 1, diagnostic only, null with clv_available=false if missing.

### HTML Report Generation

| Option | Description | Selected |
|--------|-------------|----------|
| Jinja2 HTML (existing pattern) | Dedicated template following BacktestReportGenerator pattern. | ✓ |
| Simple HTML (no template) | Python-generated HTML. No Jinja2 dependency. | |

**User's choice:** Jinja2 with dedicated shadow_comparison_report.html template. Side-by-side baseline vs shadow summary, fold/year breakdown, surface/odds/prob-rank/value-band tables, selected_changed examples with score decomposition, calibration metrics section.

---

## Baseline Model Strategy

### Baseline Definition

| Option | Description | Selected |
|--------|-------------|----------|
| Pre-Phase39 artifact | Historical model trained before MAWC/ranker changes. | |
| Feature flag disable | Current pipeline with MAWC/ranker explicitly disabled. | ✓ |

**User's choice:** Feature flags for reproducible baseline from current code. Baseline = current pipeline with enable_market_aware_calibrator=false + enable_race_level_ranker=false. Definition recorded in shadow_manifest.json.

### Flag Implementation

| Option | Description | Selected |
|--------|-------------|----------|
| None fallback (existing pattern) | MAWC/ranker None → automatic fallback. | |
| Explicit feature flags | enable_market_aware_calibrator / enable_race_level_ranker bools. | ✓ |

**User's choice:** Explicit feature flags with None fallback as safety only. If flag=false, RacePredictor skips component even if artifact exists. If flag=true but artifact missing or not deployable, fail or fall back according to explicit config. Flag states recorded in manifest.

---

## Claude's Discretion

- ShadowComparisonFramework internal method design and data flow
- ShadowComparisonResult / ShadowRunConfig dataclass field design
- Jinja2 template layout (follow existing report patterns)
- Feature flag injection mechanism in RacePredictor
- Per-race diff table column selection
- Statistical comparison utilities if needed

## Deferred Ideas

- LightGBM LambdaRank shadow variant → v2.2+
- Training orchestration inside comparison framework → future milestone
- Deployment gate automation (DEP-01) → v2.2+
