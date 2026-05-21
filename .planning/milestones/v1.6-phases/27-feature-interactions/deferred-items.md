# Deferred Items -- Phase 27 Plan 01

## Pre-existing test failures (out of scope)

3 pipeline tests fail with `record_df has duplicate race_ids: 3600`:
- tests/test_training_pipeline.py::TestTrainingPipelineV5::test_run_returns_trained_models_v5
- tests/test_training_pipeline.py::TestTrainingPipelineV5::test_pipeline_trains_per_surface
- tests/test_training_pipeline.py::TestTrainingPipelineV5::test_pipeline_logs_to_mlflow

Root cause: RecordFeatures.compute() returns duplicate race_ids in mock test setup.
This is a pre-existing issue unrelated to INTER-01 changes.
File: src/pipelines/training_pipeline.py line 481 (assertion in record_features section).
