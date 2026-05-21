---
phase: 30-residual-ic-evaluation-framework
plan: 02
status: complete
---

# Plan 30-02 Summary: OOF Save Hook + CLI Script

## What was done

Added OOF prediction Parquet save hook to `TrainingPipeline.run()` at the point after `full_features_df` is saved to features parquet. The save is purely additive — no model training logic modified.

Created `scripts/run_ic_eval.py` — thin CLI wrapper for offline IC evaluation accepting OOF Parquet path, with optional MLflow logging.

## Files created/modified

- `src/pipelines/training_pipeline.py` (modified) — added 5 lines after line 249 for OOF Parquet save
- `scripts/run_ic_eval.py` (new) — 78 lines

## Verification

- Syntax check pass
- Pipeline import verified (existing import structure)
- ruff check clean on both files
- No existing tests broken

## Key decisions

- OOF save happens at the combined `full_features_df` level (after all surface submodels are trained)
- Save path: `data/oof/oof_predictions.parquet` (per D-03)
- CLI script follows `run_etl.py` pattern (sys.path.insert, argparse, logging)
- `--mlflow` flag is opt-in (no MLflow dependency when not used)
