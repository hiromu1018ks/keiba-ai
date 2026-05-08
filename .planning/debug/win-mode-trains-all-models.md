---
slug: win-mode-trains-all-models
status: resolved
trigger: user report — place_hit_ensemble consuming 1,037s (40% of training) in win-only mode
created: 2026-05-08
---

# Debug: Win mode trains all models

## Symptoms
- TrainingPipelineV5 trains all models (win, place, wide) regardless of `betting_target`
- `place_hit_ensemble` takes 1,037s even when running `--betting-target win`
- No `betting_target` parameter exists on `TrainingPipelineV5.run()` or `_train_submodel()`

## Root Cause (confirmed)
`TrainingPipelineV5._train_submodel()` unconditionally trains all model types:
- Line 582-613: PlaceTwoStageModel (hit + return) always trained
- Line 618-652: Benter/Isotonic/Temperature for place always trained
- Line 817-821: PlaceEVCorrectionModel always trained
- Line 823-831: WideTwoStageModel always trained
- Line 833-845: Confidence calibration always includes place columns
- Line 847-854: PlaceSelectionGate always trained

`run_backtest.py` line 451 calls `pipeline.run()` without any `betting_target` argument.

## Fix Plan
1. Add `betting_target` param to `TrainingPipelineV5.run()` and `_train_submodel()`
2. Skip place-specific training when `betting_target == "win"`
3. Skip wide-specific training when `betting_target != "wide"`
4. Guard race_predictor.py inference steps with model existence checks
5. Create placeholder/dummy models for skipped components to avoid AttributeError

## Resolution
Fixed by adding `betting_target` parameter throughout the pipeline:

1. `TrainingPipelineV5.run()` — added `betting_target: str = "place"` param
2. `TrainingPipelineV5._train_submodel()` — added `betting_target` param, skips:
   - PlaceTwoStageModel (including place_hit_ensemble) when `betting_target == "win"`
   - PlaceEVCorrectionModel when `betting_target == "win"`
   - Place Benter/Isotonic/Temperature when `betting_target == "win"`
   - PlaceSelectionGate when `betting_target == "win"`
   - WideTwoStageModel when `betting_target != "wide"`
3. `race_predictor.py` — guarded all place/wide inference with None checks
4. `run_backtest.py` — passes `--betting-target` to `pipeline.run()`
5. MLflow logging + model saving — guarded against None models

Expected win-only training time reduction: ~40% (place_hit_ensemble 1,037s eliminated)

## Evidence
- 2026-05-08: Read `src/pipelines/training_pipeline.py` — no `betting_target` parameter
- 2026-05-08: Read `scripts/run_backtest.py` line 451 — `pipeline.run()` has no target filtering
- 2026-05-08: Read `src/backtest/race_predictor.py` — always calls place/wide models in inference
- 2026-05-09: Implemented fix — all 1,356 tests pass
