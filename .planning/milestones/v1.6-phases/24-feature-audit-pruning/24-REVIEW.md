---
phase: 24-feature-audit-pruning
reviewed: 2026-05-12T12:00:00Z
depth: standard
files_reviewed: 8
files_reviewed_list:
  - src/features/win_feature_analysis.py
  - src/features/feature_engine.py
  - scripts/analyze_feature_importance.py
  - scripts/prune_noise_features.py
  - tests/test_tier_report_cli.py
  - tests/test_win_feature_analysis.py
  - tests/test_feature_engine.py
  - tests/test_prune_noise_features.py
findings:
  critical: 3
  warning: 6
  info: 4
  total: 13
status: fixes_applied
---

# Phase 24: Code Review Report

**Reviewed:** 2026-05-12T12:00:00Z
**Depth:** standard
**Files Reviewed:** 8
**Status:** issues_found

## Summary

Reviewed 8 files comprising Phase 24 (Feature Audit & Pruning): tier classification, code-hash cache invalidation, CLI tier reports, and an integrated pruning pipeline with OOF safety gating and rollback.

Three critical issues were found: (1) `subprocess.run` with `bt_command.split()` breaks on command paths containing spaces, (2) `_edit_feature_cols_in_file` fails to handle one-line list definitions that contain removable features, and (3) closure variable capture in `_get_feature_values()` inside a loop captures the wrong `gain_col`/`perm_col` variables. Additionally, six warnings cover edge cases in percentile computation, incomplete dry-run test, regex brittleness, and subprocess error handling.

## Critical Issues

### CR-01: subprocess.run splits command string naively -- breaks on paths with spaces

**File:** `scripts/prune_noise_features.py:500-501`
**Issue:** `subprocess.run(bt_command.split(), ...)` splits the command string on whitespace. If the default command or any user-supplied `--bt-command` value contains paths with spaces (e.g., a Windows user directory like `C:\Users\John Doe\...`), the command will be tokenized incorrectly and fail. On Windows this is a realistic scenario since the project uses `C:\Users\hirom\develop\keiba-ai`.
**Fix:**
```python
import shlex
# Replace line 500-501:
result = subprocess.run(
    shlex.split(bt_command),
    capture_output=True,
    text=True,
    cwd=ROOT,
)
```
Even better, pass the command as a pre-tokenized list to avoid the split entirely.

### CR-02: _edit_feature_cols_in_file skips features inside single-line list definitions

**File:** `scripts/prune_noise_features.py:265-272`
**Issue:** When the FEATURE_COLS definition is on a single line (e.g., `FEATURE_COLS = ["feat_a", "feat_b", "feat_c"]`), the function detects `]` in the `after_eq` portion and immediately sets `in_target_list = False`. This means it never enters the list-parsing branch and never removes any features from single-line definitions. The code appends the line unmodified and continues. This is the exact case used in the test fixture `test_backup_created_on_apply` (line 261: `'class AbilityModel:\n    FEATURE_COLS = ["a", "b"]\n'`) but that test only checks backup creation, not that features are actually removed from single-line definitions. The `test_per_model_independent_pruning` test uses multi-line definitions and so passes.
**Fix:** When a single-line list is detected, parse and filter its contents inline:
```python
if "]" in after_eq:
    # Single-line list -- parse inline
    list_content = after_eq[after_eq.index("[") + 1:after_eq.rindex("]")]
    items = re.findall(r'"([^"]+)"', list_content)
    filtered = [f'"{item}"' for item in items if item not in remove_set]
    removed_count += len(items) - len(filtered)
    new_after_eq = f' [{".".join(filtered + ["]"])}'
    # Reconstruct line with filtered content
    new_line = line[:line.index("=") + 1] + f" [{''.join(f'{q},' for q in filtered).rstrip(',')}]"
    new_lines.append(new_line)
    in_target_list = False
    continue
```
Alternatively, normalize the input to multi-line format before editing.

### CR-03: Closure in _get_feature_values captures loop variable by reference

**File:** `src/features/win_feature_analysis.py:359-371`
**Issue:** The function `_get_feature_values` is defined inside a `for` loop over `tier_result.items()` and references `gain_col` and `perm_col` from the enclosing scope. In Python, closures capture variables by reference, not by value. Since `gain_col` and `perm_col` are reassigned on each loop iteration, all closures created during the loop will reference the final values of `gain_col` and `perm_col` (i.e., the last model's columns). This means every model's detail report uses the gain/perm columns from the last model processed, producing incorrect gain/perm values for all models except the last one.

This is a real bug because `generate_tier_report()` is called from `_run_tier_report()` in the CLI and produces JSON reports with wrong per-model gain/perm values.

**Fix:** Pass `gain_col` and `perm_col` as default arguments to capture their values:
```python
def _get_feature_values(
    feature_name: str,
    _gain_col: str = gain_col,
    _perm_col: str = perm_col,
) -> dict[str, float | None]:
    row_mask = pivot_df["feature"] == feature_name
    if row_mask.any():
        row = pivot_df.loc[row_mask].iloc[0]
        gain_val = row.get(_gain_col)
        perm_val = row.get(_perm_col)
        return {
            "gain": float(gain_val) if pd.notna(gain_val) else None,
            "perm": float(perm_val) if pd.notna(perm_val) else None,
        }
    return {"gain": None, "perm": None}
```

## Warnings

### WR-01: Tier 2 percentile produces degenerate results for small feature sets

**File:** `src/features/win_feature_analysis.py:311-316`
**Issue:** When a model has very few features with nonzero gain (e.g., 1-10 features), `np.percentile(values, 10)` returns a value very close to the minimum. This means almost all features with nonzero gain can end up classified as Tier 2 (since `g <= threshold` catches all features at or below the 10th percentile, which in a set of 10 values includes the bottom 1-2 features). For a model with only 1-2 nonzero-gain features, the percentile is just the minimum value itself, marking all non-Tier-1 features as Tier 2. This is logically questionable since Tier 2 is supposed to be "low importance" but could classify nearly all features in a small model.
**Fix:** Add a minimum count guard: only compute Tier 2 if there are enough features (e.g., `if len(nonzero_gains) >= 5`), otherwise leave Tier 2 empty.

### WR-02: Dry-run test does not actually verify pruning is skipped

**File:** `tests/test_prune_noise_features.py:114-151`
**Issue:** `test_dry_run_does_not_modify_files` creates mock files and a `tier_result`, patches `apply_pruning` but never actually invokes any pruning code path. The `with patch.object(...)` block does nothing (just `pass`), and the test only verifies that the file content didn't change -- which is trivially true because no code ran. The test gives a false sense of coverage without verifying the actual dry-run behavior of the main flow.
**Fix:** Actually invoke the relevant code path (e.g., `apply_pruning` with `--apply` absent) and verify it does not call `_edit_feature_cols_in_file` or modify files.

### WR-03: subprocess.run does not check return code for backtest failure

**File:** `scripts/prune_noise_features.py:500-505`
**Issue:** After running the full backtest via `subprocess.run`, the code stores `result.returncode` in the comparison dict but does not check whether the backtest actually succeeded before reading the result JSON. If the backtest crashes (non-zero return code), the code will still try to read `backtest_result.json`, potentially reading a stale file from a previous run, producing misleading ROI comparison results.
**Fix:**
```python
if result.returncode != 0:
    logger.error(
        "バックテスト実行失敗 (returncode=%d): %s",
        result.returncode, result.stderr,
    )
    return {
        "error": "backtest execution failed",
        "bt_returncode": result.returncode,
        "roi_improved": False,
        ...
    }
```

### WR-04: _edit_feature_cols_in_file regex does not handle trailing comments on feature lines with ]

**File:** `scripts/prune_noise_features.py:284`
**Issue:** The regex `r'^\s*"([^"]+)"\s*,?\s*(#.*)?$'` only matches lines that contain a single quoted string optionally followed by a comment. If a feature line has a trailing comma and closing bracket (e.g., `"feature_name",  # some comment ]`), the regex would still match and extract the feature name correctly. However, if a line contains something like `"feature_name"  # ]` (a comment with `]`), the list-end detection on line 277 would be skipped because it checks `stripped.startswith("]")`, and the regex on 284 would still match. The real concern is that non-standard formatting (trailing type hints, continuation characters) could silently skip features that should be removed. The text-editing approach is inherently fragile.
**Fix:** Consider using AST-based editing instead of regex for production code. For now, document the known limitations in the docstring.

### WR-05: _parse_model_filename double-parse ambiguity in main()

**File:** `scripts/prune_noise_features.py:740-743`
**Issue:** In the main flow, `model_key` has the form `"name_surface"` (e.g., `"win_hit_turf"`). The code does `parts = model_key.rsplit("_", 1)` to split into `["win_hit", "turf"]`, then calls `_parse_model_filename(parts[0])` which is `_parse_model_filename("win_hit")`. But `_parse_model_filename` expects a basename that ends with `_turf` or `_dirt` to extract the surface. Since `"win_hit"` does not end with `_turf` or `_dirt`, `_parse_model_filename` returns `(None, "")`. The fallback `name = model_key` on line 744 then uses the full key `"win_hit_turf"` as the model name for `_model_type()`, which checks membership in `BINARY_MODELS`. Since `"win_hit_turf"` is not in `BINARY_MODELS`, it is classified as `"regression"`, and the OOF safety check is incorrectly skipped for binary models with a surface suffix.
**Fix:** Use the model key directly or parse it correctly. For example:
```python
name, surface = _parse_model_filename(model_key.rsplit("_", 1)[0] + "_" + model_key.rsplit("_", 1)[1]) if "_" in model_key else (model_key, "")
```
Or simpler: use `parts[0]` directly as the model name for `_model_type()`:
```python
parts = model_key.rsplit("_", 1)
name = parts[0] if len(parts) > 1 else model_key
safety = run_oof_safety_check(name, ...)
```
Since `_model_type` checks `BINARY_MODELS` which contains `"win_hit"` (not `"win_hit_turf"`), using `parts[0]` directly would work.

### WR-06: Rollback deletes backup files, preventing repeated rollback

**File:** `scripts/prune_noise_features.py:367-368`
**Issue:** `rollback_files()` copies the backup over the current file and then deletes the backup (`os.remove(backup_path)`). If the rollback itself fails partway through (e.g., disk error on one file) or if the user needs to re-run rollback, the backup files are already gone. A safer approach would be to keep the backup files until explicit confirmation, or at least only delete them after all files are confirmed restored.
**Fix:** Collect all backup paths, copy all files first, then delete backups only after confirming all copies succeeded.

## Info

### IN-01: Duplicate _save_json helper across two scripts

**File:** `scripts/analyze_feature_importance.py:357-376` and `scripts/prune_noise_features.py:190-210`
**Issue:** Both scripts contain nearly identical `_save_json` functions for numpy-to-Python type conversion. This is a maintenance burden.
**Fix:** Extract to a shared utility module (e.g., `src/utils/json_helpers.py`).

### IN-02: Duplicate _parse_model_filename and _FILE_PREFIX_TO_DISPLAY across scripts

**File:** `scripts/analyze_feature_importance.py:422-451` and `scripts/prune_noise_features.py:130-149`
**Issue:** Both scripts have their own copies of `_parse_model_filename` and the prefix-to-display mapping. Divergence risk.
**Fix:** Extract shared model file utilities into a common module.

### IN-03: generate_tier_report defines closure inside loop -- code smell

**File:** `src/features/win_feature_analysis.py:359-371`
**Issue:** Beyond the closure bug (CR-03), defining a named function inside a loop is a code smell. Even with the fix, it creates a new function object on every iteration.
**Fix:** Move `_get_feature_values` outside the loop and pass `gain_col`/`perm_col` as explicit parameters.

### IN-04: Test fixture uses pathlib.Path(str(tmp_path)) pattern repeatedly

**File:** `tests/test_tier_report_cli.py:72`, `tests/test_feature_engine.py:656,669,686,778,796,863,867`
**Issue:** The pattern `pathlib.Path(str(tmp_path))` is used to convert `tmp_path` (which may be typed as `object` in the fixture signature). Using `pathlib.Path` directly would be cleaner.
**Fix:** Type the `tmp_path` fixture parameter correctly or use `from pathlib import Path` and `Path(tmp_path)`.

---

## Fixes Applied

All 9 in-scope findings (3 Critical, 6 Warning) were fixed in iteration 1:

- **CR-01** `bda364e`: `shlex.split()` for command tokenization
- **CR-02** `34a3b55`: Single-line FEATURE_COLS inline parsing
- **CR-03** `8a474a7`: Closure variable capture via default arguments
- **WR-01** `17f146a`: Tier 2 minimum count guard (`>= 5`)
- **WR-02** `c500621`: Dry-run test with `assert_not_called()`
- **WR-03** `99eccff`: Return code check after `subprocess.run`
- **WR-05** `d660630`: Direct model name from `rsplit` parts
- **WR-06** `aedb0c4`: Two-phase rollback (copy all then delete all)

Full test suite: 1419 passed, 1 skipped, 0 failures.

---

_Reviewed: 2026-05-12T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
