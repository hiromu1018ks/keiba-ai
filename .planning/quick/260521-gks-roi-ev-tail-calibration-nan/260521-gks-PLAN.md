---
phase: quick
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/backtest/race_predictor.py
  - src/models/stacked_ensemble.py
  - src/models/stage1_ability_model.py
  - src/models/two_stage_return_model.py
  - src/models/ev_correction_model.py
  - src/models/place_ability_model.py
  - src/models/regime_detector.py
  - src/models/conformal_ev_model.py
  - src/models/wide_two_stage_model.py
  - src/models/gpd_diagnostics.py
autonomous: true
requirements: [RTG-04, BUG-01, BUG-02]

must_haves:
  truths:
    - "EV Tail Calibration bypassed — raw edge used for win candidate sorting"
    - "Correlation penalty disabled — corr_penalty_weight defaults to 0.0"
    - "3 high-NaN features removed from ALL model FEATURE_COLS lists"
  artifacts:
    - path: "src/backtest/race_predictor.py"
      provides: "get_win_candidates() without EV Tail Calibration"
      contains: "raw_edge"
    - path: "src/models/stacked_ensemble.py"
      provides: "corr_penalty_weight=0.0 default"
      contains: "corr_penalty_weight"
  key_links:
    - from: "src/backtest/race_predictor.py"
      to: "src/betting/ev_tail_calibration.py"
      via: "import removed / bypassed"
      pattern: "NOT.*ev_tail_calibration"
---

<objective>
ROI劣化要因3件の修正: EV Tail Calibration無効化 + 相関ペナルティ無効化 + 高NaN特徴量削除

Purpose: BT ROI 97.8% -> 79.8% 劣化の主因を取り除き、ROI回復の基盤を整える
Output: 3つの独立したバグ修正 (各タスクでコミット)
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/backtest/race_predictor.py
@src/betting/ev_tail_calibration.py
@src/models/stacked_ensemble.py
@src/models/stage1_ability_model.py
@src/models/two_stage_return_model.py
@src/models/ev_correction_model.py
@src/models/place_ability_model.py
@src/models/regime_detector.py
@src/models/conformal_ev_model.py
@src/models/wide_two_stage_model.py
@src/models/gpd_diagnostics.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Disable EV Tail Calibration in get_win_candidates()</name>
  <files>src/backtest/race_predictor.py</files>
  <action>
In `src/backtest/race_predictor.py`, method `get_win_candidates()` (line 521-608):

1. Remove the entire EV tail calibration block (lines 557-571):
   - Delete `from betting.ev_tail_calibration import EVTtailCalibrator`
   - Delete the `_calibrator = EVTtailCalibrator()` instantiation
   - Delete the `calibrated_edges` loop that iterates candidates and calls `_calibrator.calibrate()`
   - Delete the `candidates["_calibrated_edge"] = calibrated_edges` assignment
   - Delete the `sort_edge_col = "_calibrated_edge"` assignment

2. Replace the sort logic to use the original `edge_col` (`win_selection_edge`) directly:
   - Replace `sort_edge_col` with `edge_col` everywhere in the sort block (lines 580-602).
   - The variable `sort_edge_col` no longer exists, so the sort becomes `candidates.sort_values([..., edge_col, ...])`.
   - Remove the `candidates.drop(columns=[sort_edge_col])` line at line 605 (no temporary column to drop).

3. Delete the `_n_ev_excluded = 0` and `candidates.attrs["n_ev_excluded"]` lines (lines 548-551) as they were only used for EV exclusion diagnostics which is already disabled. Replace with `n_ev_excluded = 0` comment only if needed for log continuity.

The result: candidates are sorted by raw `win_selection_edge` (descending) without the 0.70x/0.85x/1.05x scaling factors that were suppressing profitable bets.
  </action>
  <verify>
    <automated>cd C:\Users\hirom\develop\keiba-ai && python -m pytest tests/test_race_predictor.py -v -x 2>$null; if ($LASTEXITCODE -ne 0) { python -m pytest tests/ -v -x -k "race_predictor" }</automated>
  </verify>
  <done>get_win_candidates() sorts by raw edge without EV tail calibration. No import of ev_tail_calibration in race_predictor.py. All existing tests pass.</done>
</task>

<task type="auto">
  <name>Task 2: Disable correlation penalty in StackedEnsemble</name>
  <files>src/models/stacked_ensemble.py</files>
  <action>
In `src/models/stacked_ensemble.py`, class `StackedEnsemble.__init__()` (line 38-55):

1. Change the default value of `corr_penalty_weight` from `0.5` to `0.0` (line 42):
   ```python
   corr_penalty_weight: float = 0.0,
   ```

This single-line change disables the correlation penalty. The `_compute_corr_penalty()` static method already returns 0.0 when `weight <= 0` (line 375). No other changes needed — the logging in `_tune_hyperparams` (lines 279-289) is already gated by `self.corr_penalty_weight > 0`.
  </action>
  <verify>
    <automated>cd C:\Users\hirom\develop\keiba-ai && python -m pytest tests/ -v -x -k "ensemble" 2>$null; if ($LASTEXITCODE -ne 0) { python -c "from models.stacked_ensemble import StackedEnsemble; m = StackedEnsemble(); assert m.corr_penalty_weight == 0.0, f'Expected 0.0, got {m.corr_penalty_weight}'; print('PASS: corr_penalty_weight=0.0')" }</automated>
  </verify>
  <done>StackedEnsemble.corr_penalty_weight defaults to 0.0. AUC is no longer sacrificed for unattainable diversity. Existing ensemble tests pass.</done>
</task>

<task type="auto">
  <name>Task 3: Remove 3 high-NaN features from all model FEATURE_COLS</name>
  <files>
    src/models/stage1_ability_model.py,
    src/models/two_stage_return_model.py,
    src/models/ev_correction_model.py,
    src/models/place_ability_model.py,
    src/models/regime_detector.py,
    src/models/conformal_ev_model.py,
    src/models/wide_two_stage_model.py,
    src/models/gpd_diagnostics.py
  </files>
  <action>
Remove these 3 feature names from every FEATURE_COLS list they appear in:
- `pace_ratio_zscore`
- `pace_ratio_trend`
- `pace_adj_finish_avg`

Files and exact locations to edit:

**src/models/stage1_ability_model.py** — Remove 3 lines (lines 171, 173, 174)

**src/models/two_stage_return_model.py** — 3 separate FEATURE_COLS lists:
- `FEATURE_COLS` (around line 174, 176, 177) — the WinTwoStageModel outer list
- `HIT_FEATURE_COLS` (around line 513, 515, 516) — the hit submodel list
- `RETURN_FEATURE_COLS` (around line 675, 677, 678) — the return submodel list

**src/models/ev_correction_model.py** — 2 separate lists:
- `FEATURE_COLS` of WinEVCorrectionModel (around line 220, 222, 223)
- `FEATURE_COLS` of PlaceEVCorrectionModel (around line 529, 531, 532)

**src/models/place_ability_model.py** — Remove 3 lines (lines 139, 141, 142)

**src/models/regime_detector.py** — Remove 3 lines (lines 114, 116, 117)

**src/models/conformal_ev_model.py** — Remove 3 lines (lines 183, 185, 186)

**src/models/wide_two_stage_model.py** — `SHARED_FEATURE_COLS` (lines 101, 103, 104)

**src/models/gpd_diagnostics.py** — Remove 3 entries from the feature family dict (lines 149, 169, 170):
- `"pace_adj_finish_avg": "fundamental",`
- `"pace_ratio_zscore": "fundamental",`
- `"pace_ratio_trend": "fundamental",`

Do NOT remove these features from `src/features/horse_history_features.py` BASE_COLS — the features are still computed and stored, they are just not used as model inputs. This avoids data pipeline changes.

Do NOT remove `pace_ratio_avg` (the base average feature, NaN rate is acceptable at ~15-20%).
  </action>
  <verify>
    <automated>cd C:\Users\hirom\develop\keiba-ai && python -m pytest tests/ -v -x 2>$null; python -c "import ast, pathlib; targets=['pace_ratio_zscore','pace_ratio_trend','pace_adj_finish_avg']; files=['src/models/stage1_ability_model.py','src/models/two_stage_return_model.py','src/models/ev_correction_model.py','src/models/place_ability_model.py','src/models/regime_detector.py','src/models/conformal_ev_model.py','src/models/wide_two_stage_model.py']; found=False; [found:=True for f in files for t in targets if t in pathlib.Path(f).read_text(encoding='utf-8').split('FEATURE_COLS')[1].split(']')[0] if '\"' + t + '\"' in pathlib.Path(f).read_text(encoding='utf-8').split('FEATURE_COLS')[1].split(']')[0]]; print('PASS: all 3 features removed from FEATURE_COLS' if not found else 'FAIL: features still present')"</automated>
  </verify>
  <done>All 3 high-NaN features (pace_ratio_zscore, pace_ratio_trend, pace_adj_finish_avg) removed from 8 model files (11 FEATURE_COLS lists total). Features remain in horse_history_features.py for data pipeline continuity. All tests pass.</done>
</task>

</tasks>

<verification>
1. `python -m pytest tests/ -v -x` — all tests pass
2. `python -c "from models.stacked_ensemble import StackedEnsemble; assert StackedEnsemble().corr_penalty_weight == 0.0"` — penalty disabled
3. `grep -r "ev_tail_calibration" src/backtest/race_predictor.py` — returns empty (calibration bypassed)
4. `grep -c "pace_ratio_zscore\|pace_ratio_trend\|pace_adj_finish_avg" src/models/stage1_ability_model.py src/models/two_stage_return_model.py src/models/ev_correction_model.py src/models/place_ability_model.py src/models/regime_detector.py src/models/conformal_ev_model.py src/models/wide_two_stage_model.py` — all return 0
</verification>

<success_criteria>
- EV Tail Calibration bypassed in get_win_candidates() — raw edge for sorting
- corr_penalty_weight defaults to 0.0
- 3 high-NaN features removed from all 11 FEATURE_COLS across 8 model files
- All existing tests pass
- 3 atomic commits (one per task)
</success_criteria>

<output>
Create `.planning/quick/260521-gks-roi-ev-tail-calibration-nan/260521-gks-SUMMARY.md` when done
</output>
