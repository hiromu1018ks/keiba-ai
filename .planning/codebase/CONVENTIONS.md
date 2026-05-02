# Coding Conventions

**Analysis Date:** 2026-05-02

## Naming Patterns

**Files:**
- Python modules use `snake_case.py`: `ev_correction_model.py`, `race_predictor.py`, `parquet_store.py`
- Test files follow `test_<module_name>.py`: `test_ev_correction.py`, `test_race_predictor.py`
- Directories use `snake_case`: `features/`, `backtest/`, `betting/`, `paper_trading/`
- Package `__init__.py` files may contain docstrings or be minimal

**Classes:**
- PascalCase: `RacePredictor`, `FeatureEngine`, `ParquetStore`, `PlaceSelectionGateModel`
- Model container classes use descriptive compound names: `TrainedModelsV5`, `SubmodelSet`, `TwoStageConfig`
- Enum classes inherit from `(str, Enum)`: `Surface(str, Enum)`, `BetType(str, Enum)`, `RegimeState(str, Enum)`
- Dataclasses use `@dataclass` or `@dataclass(frozen=True)` for immutability: `Race(frozen=True)`, `Bet`, `Entry`

**Functions:**
- `snake_case`: `build_place_selection_ev()`, `compute_intra_race_features()`, `_map_basic_features()`
- Private methods use single underscore prefix: `_build_place_selection_ev()`, `_prepare_training_frame()`
- Module-level helper functions use underscore prefix when internal: `_numeric_or_nan()`, `_quantile_edges()`, `_compute_class_level()`
- Static methods use `@staticmethod` for stateless operations: `_favorite_implied_prob()`, `_pass_mask()`

**Variables:**
- `snake_case` throughout: `race_df`, `submodel`, `edge_threshold`, `regime_params`
- Boolean variables/masks use descriptive names: `hard_mask`, `soft_mask`, `gate_enabled`, `is_trained`
- Constants use UPPER_SNAKE_CASE at module level: `_GRADE_LEVEL_MAP`, `SCORE_COL`, `PASS_COL`
- Class-level constants use UPPER_SNAKE_CASE: `PlaceSelectionGateModel.SCORE_COL`, `PlaceSelectionGateModel.SOFT_PROB_BUFFER`

**Types:**
- Type aliases use PascalCase: `Surface`, `BetType`, `RegimeState`
- Type hints use modern Python 3.11 syntax: `pd.DataFrame | None`, `list[Bet]`, `dict[str, Any]`
- Generic types use lowercase builtins: `list[...]`, `dict[...]`, `tuple[...]`

## Code Style

**Formatting:**
- Tool: Ruff (target py311, line-length=100)
- Config in `pyproject.toml` `[tool.ruff]`
- Rules selected: E (pycodestyle errors), F (pyflakes), I (isort), N (pep8-naming), W (pycodestyle warnings)
- Run: `ruff check src/ tests/` and `ruff format --check src/ tests/`

**Linting:**
- Tool: Ruff lint (`[tool.ruff.lint]`)
- Selected rule categories: `["E", "F", "I", "N", "W"]`
- No custom ignore rules configured

**Type Checking:**
- Tool: mypy (strict: `disallow_untyped_defs = true`)
- Config in `pyproject.toml` `[tool.mypy]`
- `python_version = "3.11"`, `warn_return_any = true`, `warn_unused_configs = true`
- All function signatures must have complete type annotations (parameters and return types)
- Run: `mypy src/`

## Import Organization

**Order (enforced by Ruff isort - `I` rule):**
1. `from __future__ import annotations` (present in all 73 source files)
2. Standard library: `import logging`, `import math`, `from pathlib import Path`
3. Third-party: `import numpy as np`, `import pandas as pd`, `import pytest`
4. Local/application: `from domain.models import ...`, `from features.feature_engine import ...`

**Path Aliases:**
- `pythonpath = [".", "src"]` configured in `pyproject.toml` `[tool.pytest.ini_options]`
- Source imports use bare module paths: `from domain.models import Race`, `from features.feature_engine import FeatureEngine`
- Never use `src.` prefix in imports: use `from db.parquet_store import ParquetStore`, not `from src.db.parquet_store import ...`
- Scripts also resolve via `sys.path` manipulation with `_PROJECT_ROOT`

**Conditional imports:**
- `TYPE_CHECKING` guard for type-only imports to avoid circular dependencies:
  ```python
  from typing import TYPE_CHECKING
  if TYPE_CHECKING:
      from betting.drawdown_controller import DrawdownController
  ```
- Deferred imports inside method bodies for heavy/optional modules (seen in `feature_engine.py`):
  ```python
  def build_all(self, ...):
      from features.intra_race_features import compute_intra_race_features
  ```

**Conventions:**
- Every source file starts with `from __future__ import annotations` (100% of 73 source files)
- Third-party aliases: `numpy as np`, `pandas as pd` (universal)
- `from unittest.mock import MagicMock, patch` in test files (not `from unittest import mock`)

## Error Handling

**Patterns:**
- Guard clauses return early for empty/invalid inputs:
  ```python
  if race_df.empty:
      return race_df
  if surface_key not in self.models.submodels:
      return pd.DataFrame()
  ```
- Pandas numeric coercion uses `errors="coerce"` to convert invalid values to NaN:
  ```python
  pd.to_numeric(df["fukuoddslow"], errors="coerce")
  ```
- ValueError for validation failures with descriptive messages:
  ```python
  raise ValueError(f"alpha must be in [0, 1], got {alpha}")
  raise ValueError(f"Unsupported filter operator: {op}")
  ```
- Exception logging with traceback for prediction pipeline:
  ```python
  except Exception as e:
      import traceback
      logger.error("Market prediction failed: %s\n%s", e, traceback.format_exc())
      return pd.DataFrame()
  ```
- NaN-safe Series construction:
  ```python
  pd.Series(np.nan, index=df.index, dtype=float)
  ```

## Logging

**Framework:** Python stdlib `logging`

**Logger creation pattern (module-level):**
```python
import logging
logger = logging.getLogger(__name__)
```
Used consistently across 20+ source files including `backtest/race_predictor.py`, `betting/orchestrator.py`, `pipelines/training_pipeline.py`, `db/etl.py`.

**Configuration:**
- Defined in `config/settings.yaml`:
  ```yaml
  logging:
    level: "INFO"
    format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
  ```

**Log level usage:**
- `logger.info()` for pipeline step completion, timing measurements
- `logger.warning()` for fallback behavior (e.g., `popularity_rank` fallback in `feature_engine.py`)
- `logger.error()` for prediction failures with traceback
- `logger.debug()` for skipped conditions (e.g., unknown surface)

**Timing:**
- `TimingContext` context manager in `src/utils/timing.py`:
  ```python
  with TimingContext("build_all/intra_race"):
      df = compute_intra_race_features(df)
  ```
- `@timed("step_name")` decorator for function-level timing

## Comments

**When to Comment:**
- Pipeline step numbering: `# 1. race + entry を race_id で結合`, `# 2. odds を (race_id, umaban) で結合`
- Design rationale: `# LEAK修正: entries.odds は確定オッズ (レース後)。特徴量計算では tanodds を優先使用する。`
- TODO items with context: `# TODO: compute from predictions`
- Version notes: `# v5: hit_leaves 31->15, hit_rounds 500->300 -- 過学習抑制`

**JSDoc/TSDoc:**
- Google-style docstrings on public classes and methods:
  ```python
  def build_all(
      self,
      race_df: pd.DataFrame,
      entry_df: pd.DataFrame,
      odds_df: pd.DataFrame,
      ...
  ) -> pd.DataFrame:
      """バッチ特徴量生成（TrainingPipelineV5 から呼ばれる）

      Args:
          race_df: レースメタデータ (load_races() の出力)
          entry_df: 出走馬データ (load_entries_with_results() の出力)
          ...

      Returns:
          全馬の特徴量を含むDataFrame (1行 = 1馬)
      """
  ```
- Module-level docstrings on every source file:
  ```python
  """特徴量エンジン v5.3 -- メインオーケストレータ"""
  """1レース分の推論パイプライン (BacktestEngine と PaperPredictor の共通コンポーネント)"""
  ```
- Class docstrings describe responsibility and callers:
  ```python
  class PlaceSelectionGateModel:
      """OOF-learned gate and reranker for final place bet selection."""
  ```

## Function Design

**Size:** Functions vary widely. Some methods exceed 100 lines (e.g., `PlaceSelectionGateModel.train()` at ~80 lines, `RacePredictor.predict()` at ~130 lines). Complex methods are decomposed into private helpers.

**Parameters:**
- Keyword-only parameters after `*` separator:
  ```python
  def __init__(self, *, n_bins: int = 6, prior_weight: float = 24.0) -> None:
  ```
- Optional dependencies use `| None` with default `None`:
  ```python
  def predict(self, race_df: pd.DataFrame, hist_features: pd.DataFrame | None = None) -> pd.DataFrame:
  ```
- `**kwargs` avoided; explicit parameter lists preferred

**Return Values:**
- All functions have explicit return type annotations
- Pandas-heavy code returns `pd.DataFrame` or `pd.Series`
- Factory/helper functions return typed collections: `list[Bet]`, `dict[str, Any]`

## Module Design

**Exports:**
- No barrel `__init__.py` re-exports; each module is imported directly
- `__init__.py` files contain only docstrings or are empty

**Barrel Files:**
- Not used. Direct imports from leaf modules:
  ```python
  from models.place_selection_gate import PlaceSelectionGateModel
  from features.feature_engine import FeatureEngine
  from backtest.race_predictor import RacePredictor
  ```

**Data Classes:**
- Domain objects use `@dataclass` (not Pydantic, not attrs)
- Frozen dataclasses for immutable value objects: `Race(frozen=True)`, `SafetyCheckResult(frozen=True)`
- Mutable dataclasses for stateful objects: `Bet`, `Entry`, `DDState`
- Properties for computed values: `Race.surface`, `Race.race_id`, `Bet.is_valid`, `Entry.is_winner`

## Configuration Conventions

**Settings file:** `config/settings.yaml`
- Sections: `database`, `paths`, `logging`, `feature_engine`, `late_money`, `submodel`
- Loaded via `yaml.safe_load()` (see `tests/test_settings.py`)
- Database password overridden by `PGPASSWORD` environment variable (never hardcoded)

**Constants:**
- Model hyperparameter defaults in dataclass fields: `TwoStageConfig(hit_rounds=300)`
- Class-level constants as class attributes: `PlaceSelectionGateModel.SCORE_COL = "place_gate_score"`
- Module-level private constants: `_GRADE_LEVEL_MAP`, `_race_entry_shared`

## Commit Message Format

- Conventional Commits (日本語 body)
- Examples from git log:
  ```
  feat: OOF rerankerとpruningを強化
  feat: learned gateで複勝選定を改善
  feat: ROI改善とベット選定を調整
  feat(v5): モデル根本改善 -- セグメント除外全削除, 過学習抑制, キャリブレーション改善
  ```
- Prefix: `feat:`, `fix:`, `refactor:` followed by Japanese description
- Scope optional: `feat(v5):`

## Project-Specific Conventions

**ML Feature Column Names:**
- Snake_case, descriptive: `p_place_pred`, `ev_win`, `edge_place`, `fukuoddslow`, `popularity_rank`
- Japanese-derived column names from EveryDB2 schema preserved as-is: `kakuteijyuni`, `umaban`, `bataijyu`, `kyori`
- Feature names added by the pipeline are English: `distance_bin`, `surface`, `field_size`

**Model Architecture:**
- 2-stage model: P(hit) x E(odds|hit) -- never combine into single model
- Submodel split by surface only: `{"turf": SubmodelSet, "dirt": SubmodelSet}`
- EV correction split into P-correction and E-correction models
- Benter combination for probability fusion: `p_combined = logit_inv(alpha*logit(p_fund) + beta*logit(p_market) + gamma)`

**Type Safety:**
- `from __future__ import annotations` in every file (enables PEP 604 syntax)
- `disallow_untyped_defs = true` means every function needs annotations
- `TYPE_CHECKING` guard to prevent circular imports
- `@runtime_checkable` Protocol classes for dependency injection: `StakeCalculatorProtocol`, `GateKeeperProtocol`

**Avoid:**
- Do not import from `src.` prefix -- use bare module paths
- Do not read `.env` files or credentials in code
- Do not use `pytest.mark.integration` or real DB connections in tests
- Do not use `*args` without `**kwargs` -- prefer explicit parameters

---

*Convention analysis: 2026-05-02*
