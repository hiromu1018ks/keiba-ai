---
phase: 45-structural-fix
verified: 2026-05-31T15:30:00Z
status: passed
score: 10/10 must-haves verified
overrides_applied: 0
---

# Phase 45: Structural Fix Verification Report

**Phase Goal:** MAWC保守的仕様への構造変更（FIX-01: 交互作用項削除+強正則化C探索、FIX-02: OOF品質ゲート確認）により、Phase 44で特定された確率過度圧縮問題を構造的に修正し、Phase 46でShadow Comparison可能な保守的variantを生成する。
**Verified:** 2026-05-31T15:30:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### ROADMAP Success Criteria

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | 診断・ビセクションで特定された構造的欠陥(特徴量ルーティング、キャリブレーション設定等)に対する修正が実装されている | VERIFIED | `MawcConservativeRetrainer` removes 15 logit_model_x_* interactions (lines 140-159), reduces 51->36 dim features, uses conservative C grid [0.003, 0.005, 0.01, 0.03]. Structural fix addresses MAWC beta_market=0.90 dominance and ECE 3x degradation in odds 1-3 found in Phase 44. |
| 2 | 修正がOOF/WF指標で説明可能であることが確認されている | VERIFIED | `evaluate_quality_gates()` computes overall Brier/logloss/ECE + favorite band guard (odds 1-3) ECE/p-compression/EV pass rate per C value. All metrics derived from OOF predictions vs ground truth. |
| 3 | 修正内容が2024/2025固有係数に依存せず、OOF指標で汎化性が確認されている | VERIFIED | Year-level non-degradation check (lines 415-439) computes per-year Brier/logloss/ECE and verifies each year within tolerance. Strong regularization (C grid max 0.03 vs original max 3.0) prevents year-specific overfitting. |

### Observable Truths (Plan 01 Must-Haves)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | OOF予測データから36-dim特徴量行列（15のlogit_model_x_*交互作用項削除）が正しく構築される | VERIFIED | `build_conservative_feature_matrix()` produces 36 features: 6 main + 15 segment + 15 logit_market_x_*. Assertion at line 310-311 validates dim. Test `test_produces_36_dim_matrix` passes. |
| 2 | C grid [0.003, 0.005, 0.01, 0.03]の各C値でLogisticRegressionがfitされ、Brier/logloss/ECE + favorite band guard が評価される | VERIFIED | `retrain_with_c_grid()` iterates CONSERVATIVE_C_GRID, fits LogisticRegression per C, calls `evaluate_quality_gates()` with Brier/logloss/ECE + FavoriteBandGuard. Test `test_fits_each_c_value` passes (4 candidates). |
| 3 | 品質ゲート通過候補の中で最小Cが選択される（全不適格ならnot_deployed記録） | VERIFIED | `select_best_c()` returns `min(passing, key=lambda c: c.c_value)` or None. Tests `test_returns_minimum_c_among_passing` and `test_returns_none_when_all_fail` pass. |
| 4 | odds 1-3帯のp過度圧縮チェック（mean(p_conservative/p_model) >= 0.90）が実行される | VERIFIED | Lines 372-376: `p_compression_ratio = mean(p_cons_fav / max(p_base_fav, 1e-10))`, checked against P_COMPRESSION_FLOOR=0.90. Test `test_fails_on_p_over_compression` validates ratio < 0.90 correctly flagged. |
| 5 | 保守的variantが既存モデル全ファイルコピー + MAWC joblib差し替えでdata/models-backtest-mawc-conservative/{year}/に保存される | VERIFIED | `create_conservative_variant()` uses `shutil.copytree()` (line 648) then replaces MAWC joblib for deployed surfaces. Test `test_copies_all_files_and_replaces_mawc` verifies meta.json preserved, other files copied, MAWC replaced. |
| 6 | manifest JSONにsource_model_dir/mawc_fix_version/C_grid/removed_interactions/guard_resultsが記録される | VERIFIED | `generate_manifest()` produces dict with all required keys (lines 740-753). Uses per_year_surface structure to preserve year-specific data. Test `test_produces_complete_manifest` validates all keys. |

### Observable Truths (Plan 02 Must-Haves)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 7 | CLI から run_mawc_conservative_retrain.py が実行でき、conservative variant が生成される | VERIFIED | CLI `--help` exits 0 with all flags: `--oof-path`, `--source-model-dir`, `--target-root`, `--years`, `--report`. Import `MawcConservativeRetrainer` wired at line 95-98. Tests `test_cli_help_exits_zero` and `test_cli_dry_run_help` pass. |
| 8 | manifest.json に source_model_dir/mawc_fix_version/C_grid/removed_interactions/per_surface results が構造化出力される | VERIFIED | `save_retrain_results()` writes manifest.json via `json_mod.dump()`. Test `test_save_manifest_json` verifies all keys present. |
| 9 | retrain_summary.md に C grid 結果・品質ゲート判定・favorite band guard 結果・推奨C値が記録される | VERIFIED | `_write_retrain_summary()` produces 6 sections: Configuration, Per-Surface Results, Quality Gate Details, Favorite Band Guard, C Grid Candidates, Phase 46 Next Steps. Test `test_save_retrain_summary_md` validates all sections. |
| 10 | HTML レポートに baseline vs conservative メトリクス比較と品質ゲート結果が表示される | VERIFIED | `MawcConservativeReportGenerator` in separate module renders Jinja2 template with 5 sections. Template has delta highlighting (.negative/.positive CSS). Tests `test_report_generator_html` and `test_report_has_css_classes` pass. |

**Score:** 10/10 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/models/mawc_conservative_retrainer.py` | MawcConservativeRetrainer class + dataclasses + save_retrain_results | VERIFIED | 1041 lines, exports: MawcConservativeRetrainer, ConservativeRetrainResult, QualityGateResult, CGridCandidateResult, FavoriteBandGuardResult, save_retrain_results, _compute_ece |
| `tests/test_mawc_conservative_retrainer.py` | Comprehensive unit tests | VERIFIED | 972 lines, 33 tests passing, min_lines=300 exceeded |
| `scripts/run_mawc_conservative_retrain.py` | CLI entry point | VERIFIED | 167 lines, exports build_parser + main, --help works |
| `src/models/mawc_conservative_report.py` | MawcConservativeReportGenerator | VERIFIED | 72 lines, exports MawcConservativeReportGenerator, separate module |
| `src/models/templates/mawc_conservative_report.html` | Jinja2 HTML template | VERIFIED | 369 lines, 5 sections + Phase 46 footer, self-contained CSS |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `mawc_conservative_retrainer.py` | `market_aware_win_calibrator.py` | `from models.market_aware_win_calibrator import MarketAwareWinCalibrator` | WIRED | Line 27 import. Used for encoding helpers, load(), save(), build_feature_matrix(). |
| `mawc_conservative_retrainer.py` | `oof_predictions.parquet` | `pd.read_parquet` | WIRED | Line 200: `df = pd.read_parquet(oof_path)` in prepare_oof_data(). |
| `mawc_conservative_retrainer.py` | `models-backtest-mawc-conservative` | `shutil.copytree` | WIRED | Line 648: copytree, line 753: target_variant_dir in manifest. |
| `run_mawc_conservative_retrain.py` | `mawc_conservative_retrainer.py` | `from models.mawc_conservative_retrainer import` | WIRED | Line 95-98: imports MawcConservativeRetrainer + save_retrain_results. |
| `run_mawc_conservative_retrain.py` | `models-backtest-mawc-conservative` | CLI default + pipeline output | WIRED | Line 75: default target-root. Line 110-115: run_full_pipeline writes to target. |
| `run_mawc_conservative_retrain.py` | `mawc_conservative_report.py` | `from models.mawc_conservative_report import` | WIRED | Line 124: conditional import when --report flag set. |
| `mawc_conservative_report.py` | `templates/mawc_conservative_report.html` | Jinja2 FileSystemLoader | WIRED | Line 38: template_dir = Path(__file__).parent / "templates". Line 58: get_template. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| `build_conservative_feature_matrix()` | X (36-dim ndarray) | DataFrame columns: p_model, p_market, tanodds, popularity_rank, field_size | Yes -- logit transforms + one-hot + interaction multiplication | FLOWING |
| `retrain_with_c_grid()` | p_conservative | LogisticRegression.predict_proba(X)[:,1] on 36-dim features | Yes -- fitted LR produces real probabilities | FLOWING |
| `evaluate_quality_gates()` | cons_brier/logloss/ece | sklearn.metrics on p_conservative vs y | Yes -- real metric computation | FLOWING |
| `generate_manifest()` | per_year_surface dict | ConservativeRetrainResult.quality_gate data | Yes -- populated from gate results | FLOWING |
| `create_conservative_variant()` | target_dir | shutil.copytree + mawc.save() | Yes -- real file I/O with joblib serialization | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 33 unit tests pass | `python -m pytest tests/test_mawc_conservative_retrainer.py -v` | 33 passed in 2.20s | PASS |
| CLI --help exits 0 | `python scripts/run_mawc_conservative_retrain.py --help` | Exit 0, shows all flags | PASS |
| Module import works | `python -c "from models.mawc_conservative_retrainer import MawcConservativeRetrainer"` | Import OK | PASS |
| Report import works | `python -c "from models.mawc_conservative_report import MawcConservativeReportGenerator"` | Import OK | PASS |

### Probe Execution

Step 7c: SKIPPED -- no probe scripts declared for this phase.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|------------|------------|-------------|--------|----------|
| FIX-01 | 45-01, 45-02 | ビセクション・診断結果に基づき、OOF/WFで説明できる構造的欠陥(特徴量ルーティング、キャリブレーション設定等)を修正する | SATISFIED | 36-dim feature matrix (15 logit_model_x_* removed), conservative C grid [0.003-0.03], quality gates with favorite band guard |
| FIX-02 | 45-01, 45-02 | 修正内容が2024/2025固有係数に依存せず、汎化可能であることをOOF指標で確認する | SATISFIED | Year-level non-degradation checks, strong regularization prevents overfitting, C grid max 0.03 vs original max 3.0 |

No orphaned requirements found. REQUIREMENTS.md maps only FIX-01 and FIX-02 to Phase 45, both declared in PLAN frontmatter.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | - | - | No TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER markers found in any Phase 45 files |

### Human Verification Required

None -- all must-haves are programmatically verifiable. No visual appearance, real-time behavior, or external service dependencies.

### Gaps Summary

No gaps found. All 10 must-have truths verified with substantive implementations and correct wiring. Both ROADMAP success criteria and both requirement IDs (FIX-01, FIX-02) are satisfied.

---

_Verified: 2026-05-31T15:30:00Z_
_Verifier: Claude (gsd-verifier)_
