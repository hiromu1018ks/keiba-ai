# Testing Patterns

**Analysis Date:** 2026-05-02

## Test Framework

**Runner:**
- pytest >= 8.0 (declared in `pyproject.toml` dependencies)
- Config: `pyproject.toml` `[tool.pytest.ini_options]`

**Assertion Library:**
- pytest built-in assertions with `assert` statements
- `numpy.testing.assert_allclose()` for numerical precision
- `pytest.approx()` for floating-point tolerance: `assert bet.edge == pytest.approx(0.033)`
- `pd.testing.assert_frame_equal()` for DataFrame comparison

**Run Commands:**
```bash
python -m pytest tests/ -v                           # Run all tests
python -m pytest tests/test_domain.py -v              # Single file
python -m pytest tests/ -v --cov=src --cov-report=term-missing  # Coverage
```

**Configuration:**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = [".", "src"]
```

## Test File Organization

**Location:**
- All tests in flat `tests/` directory (not co-located with source)
- 81 test files, 219 test classes, 931 test methods, 43 standalone test functions

**Naming:**
- Test files: `test_<source_module_name>.py` (e.g., `test_ev_correction.py` for `ev_correction_model.py`)
- Test classes: `Test<ComponentName>` (e.g., `TestRacePredictor`, `TestBacktestResult`, `TestEnums`)
- Test methods: `test_<descriptive_behavior>()` (e.g., `test_predict_returns_dataframe_with_ev_columns`)
- Test functions (outside classes): `test_<descriptive_behavior>()` (43 standalone functions in 10 files)

**Structure:**
```
tests/
  __init__.py
  test_domain.py              # domain/models.py + domain/types.py
  test_feature_engine.py      # features/feature_engine.py
  test_race_predictor.py      # backtest/race_predictor.py
  test_backtest_engine.py     # backtest/engine.py
  test_ev_correction.py       # models/ev_correction_model.py
  test_training_pipeline.py   # pipelines/training_pipeline.py
  ... (81 files total)
```

## Test Structure

**Suite Organization:**

Two patterns coexist in the codebase:

Pattern 1 -- Class-based (dominant, 931 methods across 219 classes):
```python
class TestRacePredictor:
    def test_predict_returns_dataframe_with_ev_columns(self, mock_models: MagicMock) -> None:
        from backtest.race_predictor import RacePredictor
        predictor = RacePredictor(models=mock_models)
        # ... test body

    def test_predict_skips_unknown_surface(self, mock_models: MagicMock) -> None:
        # ... test body
```

Pattern 2 -- Standalone functions (43 functions in 10 files):
```python
def test_place_selection_gate_trains_and_scores() -> None:
    from models.place_selection_gate import PlaceSelectionGateModel
    # ... test body

def test_settings_has_required_sections(settings):
    # ... test body
```

**Class grouping conventions:**
- Related tests grouped in named classes: `TestRacePredictor`, `TestRacePredictorEVEdge`
- Domain model tests split by entity: `TestEnums`, `TestRace`, `TestEntry`, `TestBet`, `TestOddsSnapshot`, `TestDDState`
- All test methods have return type annotation `-> None` when inside typed test classes

**Patterns:**
- Setup via `@pytest.fixture` (68 fixtures across 39 files)
- No teardown pattern -- tests are stateless
- Assertion via plain `assert` with descriptive messages

## Mocking

**Framework:** `unittest.mock` (stdlib, not pytest-mock or other third-party)

**Import pattern:**
```python
from unittest.mock import MagicMock, patch
```

**Mock creation patterns:**

1. Spec-based mock for type safety:
```python
models = MagicMock(spec=TrainedModelsV5)
models.submodels = {"turf": MagicMock(spec=SubmodelSet)}
```

2. Helper functions for complex mock construction:
```python
def _make_submodel_mock() -> MagicMock:
    """SubmodelSet の各フィールドを MagicMock で構成するヘルパー"""
    sm = MagicMock()
    sm.market = MagicMock()
    sm.stage1 = MagicMock()
    sm.place_ability = MagicMock()
    sm.win = MagicMock()
    sm.ev_corrector = MagicMock()
    sm.place = MagicMock()
    sm.wide = MagicMock()
    sm.confidence = MagicMock()
    sm.place_selection_gate = None
    sm.benter_combo = None
    sm.isotonic_calibrator = None
    return sm
```
(From `tests/test_race_predictor.py`)

3. Stub classes for interface testing:
```python
class StubGate:
    is_trained = True

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        scored = df.copy()
        scored["place_gate_score"] = [0.2, 1.3, 0.1]
        scored["place_gate_pass"] = [False, True, False]
        return scored
```
(From `tests/test_race_predictor.py`)

4. Fake classes for replacing heavy dependencies:
```python
class _FakeHistFeatures:
    """HorseHistoryFeatures のスタブ (DB不要)"""
    def __init__(self, *args, **kwargs):  # noqa: ARG002
        pass
    def compute(self, race_df, entry_df, target_race_ids=None):  # noqa: ARG002
        return pd.DataFrame(columns=["race_id", "umaban"])
```
(From `tests/test_training_pipeline.py`)

5. Return value chaining for pipeline mock:
```python
submodel = mock_models.submodels["turf"]
submodel.market.predict_and_calc_error.return_value = race_df.copy()
submodel.stage1.add_ability_probs.return_value = race_df.copy()
submodel.place_ability.predict.return_value = race_df.copy()
submodel.win.predict_ev.return_value = race_df.copy()
submodel.ev_corrector.correct_ev.return_value = race_df.copy()
submodel.place.predict_ev.return_value = result_df  # includes p_place_pred
submodel.confidence.predict_lower_bound.return_value = (
    result_df.copy(),
    pd.DataFrame({"EV_lower_place": [1.5]}),
)
```

**What to Mock:**
- All ML models (LightGBM, XGBoost, etc.) -- use `MagicMock` with `return_value`
- Database connections (`ParquetStore`, `DataRepository`) -- use `MagicMock`
- External services (MLflow, Slack) -- use `patch` or `MagicMock`
- File system paths -- use `tmp_path` pytest fixture

**What NOT to Mock:**
- Domain model instantiation (`Race`, `Entry`, `Bet`) -- construct real objects
- Feature computation functions (`compute_intra_race_features`, etc.) -- test directly with DataFrames
- Pandas operations -- test with real DataFrames
- Enum values -- use real enum instances

**`patch` usage:**
```python
@patch("pipelines.training_pipeline.HorseHistoryFeatures", _FakeHistFeatures)
@patch("pipelines.training_pipeline.PlaceAbilityModel", _FakePlaceAbilityModel)
```

## Fixtures and Factories

**Test Data:**

Fixtures construct realistic DataFrames matching the production column schema:
```python
@pytest.fixture
def sample_race_df() -> pd.DataFrame:
    """1レース分の race データ (18頭立て) -- 生カラム名"""
    return pd.DataFrame({
        "race_id": ["2024032405030208"] * 18,
        "race_date": [pd.Timestamp("2024-03-24")] * 18,
        "trackcd": [11] * 18,
        "kyori": [1600] * 18,
        "surface": ["turf"] * 18,
        ...
    })
```
(From `tests/test_feature_engine.py`)

**Fixture categories:**
- `mock_models` -- Pre-configured `TrainedModelsV5` mock (in `test_race_predictor.py`, `test_backtest_engine.py`)
- `pre_ev_df` -- Realistic pre-correction DataFrame (in `test_ev_correction.py`)
- `sample_race_df`, `sample_entry_df`, `sample_odds_df` -- Feature engine test data
- `store` (with `tmp_path`) -- `ParquetStore` backed by temporary directory
- `settings` -- Loaded `config/settings.yaml`

**Location:**
- Fixtures defined in the test file that uses them (not in `conftest.py`)
- Shared via function-scoped or module-scoped `@pytest.fixture`
- No `conftest.py` file exists -- each test file is self-contained

## Coverage

**Requirements:** Not enforced by CI (no `.github/` CI configuration)

**View Coverage:**
```bash
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

**Current State (estimated from file counts):**
- 81 test files covering 73 source modules
- Most source modules have a 1:1 test file correspondence
- Source files without dedicated test files: `src/db/schema.py`, `src/domain/__init__.py`, various `__init__.py`

## Test Types

**Unit Tests:**
- Primary test type (nearly all 974 test functions)
- Each test constructs inputs, calls a function/method, asserts output
- No DB, no network, no file I/O (except `tmp_path` for `ParquetStore` tests)
- Typical pattern:
  ```python
  def test_predict_skips_unknown_surface(self, mock_models: MagicMock) -> None:
      predictor = RacePredictor(models=mock_models)
      race_df = pd.DataFrame({"surface": ["unknown"], ...})
      result = predictor.predict(race_df)
      assert result.empty
  ```

**Integration Tests:**
- Not a formal category in this codebase
- `tests/test_parquet_store.py` uses real file I/O via `tmp_path` (closest to integration)
- `tests/test_etl.py` tests ETL pipeline with mocked DB

**E2E Tests:**
- Not present
- Full pipeline validation done via `scripts/run_backtest.py` (manual, not automated tests)

## Common Patterns

**Async Testing:**
- Not applicable -- no async code in the codebase

**Error Testing:**
```python
def test_alpha_validation_rejects_out_of_range(self, mock_models: MagicMock) -> None:
    """alpha outside [0, 1] should raise ValueError."""
    from backtest.race_predictor import RacePredictor

    with pytest.raises(ValueError, match="alpha must be in"):
        RacePredictor(models=mock_models, alpha=1.5)

    with pytest.raises(ValueError, match="alpha must be in"):
        RacePredictor(models=mock_models, alpha=-0.1)
```

**Numerical Precision Testing:**
```python
# numpy allclose for arrays
np.testing.assert_allclose(result["edge_place"].values, expected_edge, rtol=1e-6)

# pytest.approx for scalars
assert bets[0].edge == pytest.approx(0.08, abs=1e-3)
assert abs(result["edge_place"].iloc[0] - 0.05) < 1e-10
```

**Empty Input / Edge Case Testing:**
```python
def test_predict_skips_unknown_surface(self, mock_models: MagicMock) -> None:
    # Unknown surface returns empty DataFrame
    ...

def test_predict_empty_input(self, ...) -> None:
    # Empty DataFrame passed through
    ...
```

**Race DataFrame Construction:**
Tests construct minimal but realistic DataFrames with the exact column names used in production:
```python
race_df = pd.DataFrame({
    "race_id": ["R1", "R1", "R1"],
    "umaban": [1, 2, 3],
    "surface": ["turf", "turf", "turf"],
    "fukuoddslow": [4.0, 5.0, 10.0],
    "place_selection_prob": [0.42, 0.30, 0.12],
    "place_selection_edge": [0.09, 0.08, 0.01],
    ...
})
```

## How Tests Avoid DB Dependency

All tests run without PostgreSQL or any external service:

1. **Mock DataRepository/ParquetStore:** Database reads replaced with `MagicMock`
2. **Mock ML models:** LightGBM predictions replaced with `MagicMock.return_value`
3. **Real DataFrames:** Data construction uses `pd.DataFrame({...})` directly
4. **`tmp_path` fixture:** File I/O tests use pytest's temporary directory
5. **Fake/Stubs:** Heavy classes replaced with lightweight implementations (`_FakeHistFeatures`, `_FakePlaceAbilityModel`)
6. **No conftest.py imports:** Each test file is self-contained with its own fixtures

## Test Execution Details

**Total test count:** ~974 (931 class methods + 43 standalone functions) across 81 files

**Largest test files (by line count):**
- `tests/test_horse_history_features.py` (1213 lines)
- `tests/test_backtest_engine.py` (1198 lines)
- `tests/test_race_predictor.py` (1193 lines)
- `tests/test_history_features_v2.py` (718 lines)
- `tests/test_training_pipeline.py` (714 lines)

**No markers or parametrize:** Tests use plain `assert` statements; no `@pytest.mark.parametrize` or custom markers observed.

## Test Coverage Gaps

**Modules without dedicated test files:**
- `src/db/schema.py` -- Schema definitions
- `src/paper_trading/config.py` -- Paper trading configuration
- `src/monitoring/__init__.py`, `src/automation/__init__.py` -- Package inits (low priority)

**Limited error path testing:**
- Only 8 `pytest.raises` usages across 7 files out of 81 test files
- Exception handling in production code (e.g., `except Exception` in `race_predictor.py`) may have untested error paths

**No conftest.py:**
- Common mock construction (e.g., `_make_submodel_mock()`) is duplicated across test files
- A shared `conftest.py` could reduce repetition for `mock_models`, `_make_submodel_mock`, etc.

---

*Testing analysis: 2026-05-02*
