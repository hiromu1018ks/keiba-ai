---
phase: 01-feature-analysis-enhancement
reviewed: 2026-05-02T00:00:00Z
depth: standard
files_reviewed: 8
files_reviewed_list:
  - scripts/analyze_feature_importance.py
  - src/features/horse_history_features.py
  - src/features/win_feature_analysis.py
  - src/models/two_stage_return_model.py
  - src/pipelines/training_pipeline.py
  - tests/test_horse_history_features.py
  - tests/test_two_stage_return_model.py
  - tests/test_win_feature_analysis.py
findings:
  critical: 3
  warning: 7
  info: 4
  total: 14
status: issues_found
---

# Phase 1: Code Review Report

**Reviewed:** 2026-05-02
**Depth:** standard
**Files Reviewed:** 8
**Status:** issues_found

## Summary

Reviewed 8 source files for Phase 1 (feature analysis and enhancement). Found 3 critical issues and 7 warnings across the implementation. The critical issues include: (1) a train-on-evaluate-same-data leak in `validate_noise_removal()` that will always underestimate the harm of noise removal, (2) `PlaceTwoStageModel` silently drops `odds_to_ability_ratio` at inference time because it lacks the fallback computation that `WinTwoStageModel` has, and (3) `remove_noise_features()` mutates a class-level list in-place without synchronization, which is unsafe in the concurrent training pipeline.

## Critical Issues

### CR-01: validate_noise_removal trains and evaluates on the same data -- inflated metrics mask degradation

**File:** `src/features/win_feature_analysis.py:139-158`
**Issue:** `validate_noise_removal()` trains the new model on the full `df`, then evaluates logloss/AUC on the exact same data (line 155: `new_pred = new_model.predict(new_features_df)`). The "original" model metrics are also computed on training data (line 123). Comparing training-set metrics of two models tells you nothing reliable about generalization -- the new model may appear equally good or better in-sample while being worse out-of-sample. The function is explicitly documented as the gate-check for removing features (line 98: "validate_noise_removal()でlogloss/AUCへの影響を検証した後に呼び出すこと"). A biased validation will approve harmful noise removals.

**Fix:** Use a hold-out split or cross-validation. At minimum, use the same chronological train/valid split as `_train_valid_split`:
```python
# Split data chronologically
n = len(df)
split = int(n * 0.8)
train_features = new_features_df.iloc[:split]
train_y = y[:split]
valid_features = new_features_df.iloc[split:]
valid_y = y[split:]

train_data = lgb.Dataset(train_features, label=train_y)
new_model = lgb.train(params, train_data, num_boost_round=100)
new_pred = new_model.predict(valid_features)
new_logloss = float(log_loss(valid_y, new_pred))
new_auc = float(roc_auc_score(valid_y, new_pred))
```
The original model metrics should also be computed on the same validation split for a fair comparison.

### CR-02: PlaceTwoStageModel silently drops odds_to_ability_ratio at inference time

**File:** `src/models/two_stage_return_model.py:346-360`
**Issue:** `WinTwoStageModel._prepare_features()` (line 115-137) contains a fallback that computes `odds_to_ability_ratio` from `p_market_win_adj` and `p_ability_win` when the column is missing from the DataFrame. However, `PlaceTwoStageModel._prepare_features()` (line 346-360) has no such fallback -- it simply filters to `available_cols`, silently dropping `odds_to_ability_ratio` if it was not pre-computed. Since `odds_to_ability_ratio` is listed in `PlaceTwoStageModel.RETURN_FEATURE_COLS` (line 337), the place return model will train with this feature but silently lose it at inference time, creating a train/test feature mismatch that degrades predictions silently.

**Fix:** Add the same fallback computation to `PlaceTwoStageModel._prepare_features()`:
```python
def _prepare_features(
    self, df: pd.DataFrame, *, use_cols: list[str] | None = None
) -> pd.DataFrame:
    cols = use_cols or self.FEATURE_COLS
    # FEAT-02: inference-time fallback for odds_to_ability_ratio
    if (
        "odds_to_ability_ratio" in cols
        and "odds_to_ability_ratio" not in df.columns
    ):
        if "p_market_win_adj" in df.columns and "p_ability_win" in df.columns:
            df = df.copy()
            p_market = df["p_market_win_adj"].clip(lower=1e-6)
            p_ability = df["p_ability_win"].clip(lower=1e-6)
            df["odds_to_ability_ratio"] = (p_market / p_ability).clip(0.1, 10.0)
    available_cols = [c for c in cols if c in df.columns]
    # ... rest unchanged
```

### CR-03: remove_noise_features mutates class-level mutable list -- unsafe for concurrent use

**File:** `src/models/two_stage_return_model.py:94-113`
**Issue:** `remove_noise_features()` is a `@classmethod` that reassigns `cls.FEATURE_COLS` via list comprehension (line 105). Since `FEATURE_COLS` is a class variable (mutable `list[str]`), this mutation is visible to all threads in the same process. The training pipeline (`training_pipeline.py:207-216`) uses `ThreadPoolExecutor(max_workers=2)` to train turf and dirt submodels in parallel. If `remove_noise_features` is called during or before parallel training, one thread may see a partially-constructed list or a list that was modified mid-iteration. Even outside parallel training, the mutation is persistent across calls -- a second invocation of the CLI script in the same process will operate on an already-trimmed list, potentially removing too many features.

**Fix:** Either make `FEATURE_COLS` immutable (use `@property` returning a frozen copy) or ensure `remove_noise_features` is called only in a controlled single-threaded context. For the concurrent case, compute the filtered feature list as a local variable rather than mutating the class:
```python
@classmethod
def get_filtered_feature_cols(cls, noise_features: list[str]) -> list[str]:
    return [f for f in cls.FEATURE_COLS if f not in noise_features]
```
If in-place mutation must be kept, document the thread-safety requirement and add a lock.

## Warnings

### WR-01: odds_to_ability_ratio not computed for PlaceTwoStageModel during training

**File:** `src/pipelines/training_pipeline.py:411-417`
**Issue:** `odds_to_ability_ratio` is computed only once in the training pipeline, on `df_oof` at line 417. This value is then inherited by both `WinTwoStageModel` and `PlaceTwoStageModel` through `df_oof`. However, the code path is correct for training -- the feature is computed before either model trains. The issue is that the computation happens before `PlaceAbilityModel.predict()` (line 426) which adds `p_ability_place`. If `odds_to_ability_ratio` were ever to depend on place probabilities, this ordering would be wrong. Currently it only uses `p_ability_win`, so it is not a bug today, but the placement of the computation is fragile and should be documented or moved closer to where it is consumed.

**Fix:** Add a comment clarifying the intentional dependency ordering, or move the computation closer to `WinTwoStageModel.train_hit_model()`.

### WR-02: history_mask uses only kakuteijyuni > 0 filter, but hist_start/hist_idx use different valid_mask

**File:** `src/features/horse_history_features.py:618-631`
**Issue:** The `history_mask` at line 618-622 uses only `kakuteijyuni > 0` as the filter, but the `valid_mask` used for most features (line 603) also requires `syussotosu >= 8`. The `hist_idx`/`hist_start` derived from `history_mask` are used for `class_move`, `blinker_change`, `distance_change`, `surface_change`, and other FEAT-02 features (lines 913-997, 1004-1081). This means these features can use past races with field sizes under 8 (which the CLAUDE.md explicitly marks as unreliable: "8頭未満レース" are filtered in `_norm_finish_logit`). The inconsistency could cause noise in these features.

**Fix:** Use the same `valid_mask` criteria for `history_mask`:
```python
history_mask = (
    (horse_arrs["kakuteijyuni"] > 0)
    & (horse_arrs.get("valid_field", np.zeros(n, dtype=bool)) == 1)
) if horse_arrs is not None and "kakuteijyuni" in horse_arrs else np.array([], dtype=bool)
```

### WR-03: class_drop_bounce condition uses avg_recent_b > 0.5 which is inverted

**File:** `src/features/horse_history_features.py:1033-1037`
**Issue:** The `class_drop_bounce` feature computes `norm_recent_b = (kj - 1) / (ss - 1)` for recent races. When `avg_recent_b > 0.5`, the horse finished in the back half of the field (0 = first, 1 = last). The condition `if avg_recent_b > 0.5` correctly triggers when the horse had poor recent form (finished in the back half). However, the resulting value is `abs(class_move) * avg_recent_b`, which means: the higher the average finish position (worse performance), the higher the bounce score. This is the intended behavior (worse recent form = higher bounce expectation). The logic is correct but counterintuitive -- a comment would improve readability.

**Fix:** Add a clarifying comment:
```python
# avg_recent_b > 0.5 means horse finished in back half of field (poor form)
# Higher avg = worse form = stronger bounce signal
class_drop_bounce: float = (
    min(float(abs(class_move)) * avg_recent_b, 10.0)
    if avg_recent_b > 0.5  # poor recent form expected for bounce
    else 0.0
)
```

### WR-04: win_dominance returns 0.0 for horses with history but no wins -- semantically ambiguous

**File:** `src/features/horse_history_features.py:1054-1055`
**Issue:** When a horse has past races (`n_past > 0`) but never won (`win_mask.any()` is False), `win_dominance` returns `0.0`. When a horse has no history at all (`n_past == 0`), it returns `float("nan")`. This makes `0.0` semantically ambiguous -- it could mean "never won" or could be confused with a legitimate dominance score. LightGBM can handle NaN, so returning `0.0` for a horse that has never won loses information that could be useful (the model cannot distinguish "never won" from "won in tiny fields").

**Fix:** Return `float("nan")` for horses with no wins (consistent with no-history case), or use a sentinel value like `-1.0` to distinguish "no wins" from legitimate dominance scores.

### WR-05: analyze_feature_importance.shap_matrix assertion may crash on unexpected model versions

**File:** `src/features/win_feature_analysis.py:47-49`
**Issue:** The `assert` statement at line 47-49 will raise an `AssertionError` with no useful error message if the SHAP matrix shape does not match expectations. This could happen with different LightGBM versions or multiclass models. An assertion with a bare message provides no actionable information.

**Fix:** Replace with a descriptive error:
```python
shap_cols = shap_matrix.shape[1]
expected_cols = len(feature_names) + 1
if shap_cols != expected_cols:
    raise ValueError(
        f"pred_contrib returned {shap_cols} columns, "
        f"expected {expected_cols} (n_features + 1 for base value). "
        f"Model features: {len(feature_names)}"
    )
```

### WR-06: _load_features_for_analysis fallback creates meaningless SHAP values

**File:** `scripts/analyze_feature_importance.py:176-179`
**Issue:** When ParquetStore loading fails, the fallback creates a zero-filled DataFrame (line 179). SHAP values computed on all-zeros data are meaningless and will produce misleading results. The script continues to run and save a report, which a user might treat as valid analysis.

**Fix:** Exit with an error instead of silently producing invalid results:
```python
logger.error("実データの読み込みに失敗しました。ダミーデータでの分析は無意味です。")
sys.exit(1)
```

### WR-07: _find_model_file picks arbitrary surface model without user control

**File:** `scripts/analyze_feature_importance.py:128-145`
**Issue:** The function searches for models in order `turf`, `dirt`, then falls back to `glob("win_hit_*.lgb")`. It always picks `matches[0]` from glob, which depends on filesystem ordering and is not deterministic. The user has no way to specify which surface model to analyze. Since turf and dirt models have different feature distributions, the SHAP analysis could differ significantly between them.

**Fix:** Add a `--surface` argument to let the user choose:
```python
parser.add_argument("--surface", choices=["turf", "dirt"], default="turf",
                    help="解析対象のサーフェス (default: turf)")
```

## Info

### IN-01: print() statements in horse_history_features.py

**File:** `src/features/horse_history_features.py:590-593, 1119`
**Issue:** `print()` calls for progress reporting instead of using `logger.info()`. The project convention uses `logging` throughout.

**Fix:** Replace `print(...)` with `logger.info("HorseHistoryFeatures: %d/%d (%.0f%%)", i, total, i / max(total, 1) * 100)`.

### IN-02: Test test_removes_specified_features mutates class state

**File:** `tests/test_win_feature_analysis.py:169-177`
**Issue:** `test_removes_specified_features` removes features from `WinTwoStageModel.FEATURE_COLS` and restores them manually at line 177. If the test fails before line 177, the class state is corrupted for subsequent tests. The `test_no_duplicate_entries` test at line 213 also depends on the class state being intact.

**Fix:** Use a pytest fixture with cleanup:
```python
@pytest.fixture(autouse=True)
def restore_feature_cols(self):
    original = list(WinTwoStageModel.FEATURE_COLS)
    yield
    WinTwoStageModel.FEATURE_COLS = original
```

### IN-03: Unused import in horse_history_features.py

**File:** `src/features/horse_history_features.py:25`
**Issue:** `from features.form_cycle_features import compute_form_features` is imported at the module level. It is used at line 902, so this is not actually unused. However, the circular-ish import pattern (importing from the same package at module level) could cause issues if `form_cycle_features` imports anything from this module.

**Fix:** The import is functional. Consider using a local import inside `compute()` to match the pattern used elsewhere in this file (e.g., line 309-311 `from db.readers import load_history_entries`).

### IN-04: Magic number 10.0 as class_drop_bounce clip ceiling

**File:** `src/features/horse_history_features.py:1033`
**Issue:** `min(float(abs(class_move)) * avg_recent_b, 10.0)` uses `10.0` as a clip ceiling without explanation. This is a magic number that should be a named constant.

**Fix:** Extract to a constant: `CLASS_DROP_BOUNCE_MAX = 10.0`.

---

_Reviewed: 2026-05-02_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
