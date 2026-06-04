# Deferred Items — Phase 39 Plan 02

## Out-of-Scope Test Failures

These tests fail because they reference `win_benter`/`win_segment_calibrator` in ModelLoader/RacePredictor,
which are updated in Plan 39-03.

- `tests/test_model_loader.py` — 4 tests: ModelLoader still uses `win_benter`, `win_isotonic_calibrator`, `win_temperature_scaler`, `win_segment_calibrator` keyword arguments in SubmodelSet construction. Fixed in Plan 39-03.
- `tests/test_ensemble_gate_propagation.py` — 1 test: Same ModelLoader issue. Fixed in Plan 39-03.
- `tests/test_win_profit_selector.py` — 1 test: RacePredictor/ModelLoader chain still references old fields. Fixed in Plan 39-03.
- `tests/test_backtest_engine.py::TestBacktestOptimizationStages::test_observed_true_on_all_groupby` — Pre-existing failure from Phase 38 (unobserved groupby calls in feature_frame.py). Not related to Plan 39-02 changes.
