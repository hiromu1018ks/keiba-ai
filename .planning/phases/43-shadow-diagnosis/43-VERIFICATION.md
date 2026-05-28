---
phase: 43-shadow-diagnosis
verified: 2026-05-29T16:30:00Z
status: human_needed
score: 9/9 must-haves verified
overrides_applied: 0
human_verification:
  - test: "HTML レポートの visual review — Step 1/2/3 セクションが正しく表示される"
    expected: "3ステップ段階的分析がセクション化表示され、delta悪化が赤色表示される"
    why_human: "HTML の視覚的レイアウト・スタイリングは grep で完全検証不可"
  - test: "summary.md の内容レビュー — 主要な劣化次元が記録されている"
    expected: "Probability Quality / Selection Pattern / Calibration Gaps / Missing Inputs セクションが完備"
    why_human: "Markdown の可読性・フォーマット品質は人間判断"
---

# Phase 43: Shadow Diagnosis Verification Report

**Phase Goal:** baseline vs shadowの確率品質・選定パターン・キャリブレーション乖離を完全に特定し、劣化の次元を明らかにする
**Verified:** 2026-05-29T16:30:00Z
**Status:** human_needed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria + Plan Must-Haves)

ROADMAP Success Criteria (contract truths):

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC-1 | 2024/2025固定foldのBrier/logloss/ECEがbaseline vs shadowで数値比較され、劣化次元が特定されている | VERIFIED | `_step1_probability_quality()` が `_compute_prob_metrics()` 経由で Brier/logloss/ECE/APR を算出。Delta値も計算。JSON schema に `step1_probability_quality` セクションあり。テスト `test_step1_probability_quality` で数値検証済み |
| SC-2 | RaceLevelRankerの選定パターン差分がレポート化されている | VERIFIED | `_step2_selection_pattern()` が `selected_changed` で changed/unchanged グループに分割し ROI/HR/avg_odds/APR を計算。HTML template に "Step 2: Selection Pattern" セクションあり。テスト `test_step2_selection_pattern` で検証 |
| SC-3 | surface/odds_band/popularity_band/probability_rank_band/selected_changed別のactual/predicted比率乖離箇所が特定されている | VERIFIED | `_step3_calibration_by_segment()` + `_add_segment_columns()` で5セグメント別APR/ECE計算。テスト `test_step3_calibration` で `probability_rank_band` / `selected_changed` が必ず含まれることを検証 |

Plan 01 must-haves:

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| P1-1 | 全馬ベースでbaseline vs shadowのBrier/logloss/ECE/actual_predicted_ratioが数値比較される | VERIFIED | `_compute_prob_metrics()` (lines 201-245) で np.mean Brier/logloss + `_compute_ece` + APR 計算 |
| P1-2 | selected_changed vs unchangedレースでROI/的中率/avg_odds/actual_predicted_ratioの差分が計算される | VERIFIED | `_step2_selection_pattern()` (lines 335-357) + `_compute_group_metrics()` (lines 276-333) |
| P1-3 | surface/odds_band/popularity_band/probability_rank_band/selected_changed別にactual/predicted比率とECEが比較される | VERIFIED | `_step3_calibration_by_segment()` + `_add_segment_columns()` で5セグメント定義。segment_cols = ["popularity_band", "probability_rank_band", "odds_band", "surface", "selected_changed"] |
| P1-4 | Phase 41成果物に不足列があればmissing_inputsとして記録される | VERIFIED | `_detect_missing_inputs()` (lines 171-180) + `_add_segment_columns()` 内で動的追加。テスト `test_missing_inputs_detection` で検証 |

Plan 02 must-haves:

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| P2-1 | CLIからrun_shadow_diagnosis.pyが実行でき、shadow_diagnosis_result.jsonが生成される | VERIFIED | `scripts/run_shadow_diagnosis.py` に `build_parser()` + `main()` 実装。`--help` が exit code 0 で完了。`save_diagnosis_results()` でJSON出力 |
| P2-2 | HTMLレポートに3ステップ段階的分析がセクション化表示される | VERIFIED | `shadow_diagnosis_report.html` に "Step 1: Probability Quality" / "Step 2: Selection Pattern" / "Step 3: Calibration by Segment" セクション。テスト `test_report_generator_html` で検証 |
| P2-3 | shadow_diagnosis_summary.mdに主要な劣化次元とPhase 44/45への推奨が記録される | VERIFIED | `_build_summary_md()` が 5セクション構成MD生成。"Recommendations for Phase 44/45" セクション含む。テスト `test_save_diagnosis_summary_md` で検証 |
| P2-4 | missing_inputsが3出力ファイル全てに含まれる | VERIFIED | JSON: `_result_to_dict()` の "missing_inputs" key。HTML: `{% if missing_inputs %}` block。MD: "Missing Inputs" section |

**Score:** 9/9 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/backtest/shadow_diagnosis.py` | ShadowDiagnosis class + 6 dataclasses + output functions | VERIFIED | 794行。6 dataclass, ShadowDiagnosis, save_diagnosis_results, ShadowDiagnosisReportGenerator, _result_to_dict, _build_summary_md 全て実装 |
| `tests/test_shadow_diagnosis.py` | ユニットテスト (Plan 01: 7, Plan 02: 5 = 12) | VERIFIED | 543行。12テスト全て PASS |
| `scripts/run_shadow_diagnosis.py` | CLI entry point | VERIFIED | build_parser + main + --input-dir/--output-dir/--report |
| `src/backtest/templates/shadow_diagnosis_report.html` | Jinja2 HTML template | VERIFIED | 213行。3ステップセクション + missing_inputs |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/backtest/shadow_diagnosis.py` | `src/backtest/shadow_comparison.py` | `from backtest.shadow_comparison import ShadowComparisonFramework` (line 21) | WIRED | `_compute_ece()` static method を再利用 |
| `src/backtest/shadow_diagnosis.py` | `shadow_horse_diff.parquet` | `pd.read_parquet(input_dir / "shadow_horse_diff.parquet")` (line 146) | WIRED | Phase 41 成果物の horse_diff 読み込み |
| `scripts/run_shadow_diagnosis.py` | `src/backtest/shadow_diagnosis.py` | `from backtest.shadow_diagnosis import ShadowDiagnosis, ...` (line 70) | WIRED | CLI から ShadowDiagnosis + save_diagnosis_results + ShadowDiagnosisReportGenerator を import |
| `src/backtest/shadow_diagnosis.py` | `src/backtest/templates/shadow_diagnosis_report.html` | `env.get_template("shadow_diagnosis_report.html")` (line 774) | WIRED | Jinja2 FileSystemLoader 経由 |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|-------------------|--------|
| `ShadowDiagnosis._step1_probability_quality()` | `p_vals` (from horse_diff) | `self.horse_diff[f"{baseline_name}_p_win_final"]` | YES -- p_win_final から Brier/logloss/ECE を計算 | FLOWING |
| `ShadowDiagnosis._step2_selection_pattern()` | `group_race` | `self.race_diff` filtered by `selected_changed` | YES -- race_diff から ROI/HR/avg_odds を計算 | FLOWING |
| `ShadowDiagnosis._step3_calibration_by_segment()` | `seg_df` | `horse_work` grouped by segment columns | YES -- セグメント別 APR/ECE を計算 | FLOWING |
| `save_diagnosis_results()` | `result_dict` | `_result_to_dict(diagnosis_result)` | YES -- ShadowDiagnosisResult を JSON にシリアライズ | FLOWING |
| `ShadowDiagnosisReportGenerator.generate()` | `context` | `diagnosis_result.step1/step2/step3.segments` | YES -- Jinja2 template に全データを渡す | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Tests pass | `python -m pytest tests/test_shadow_diagnosis.py -v` | 12 passed in 0.80s | PASS |
| Lint clean | `ruff check src/backtest/shadow_diagnosis.py scripts/run_shadow_diagnosis.py tests/test_shadow_diagnosis.py` | All checks passed! | PASS |
| CLI --help | `python scripts/run_shadow_diagnosis.py --help` | exit 0, shows --input-dir/--output-dir/--report | PASS |
| Type check | `mypy src/backtest/shadow_diagnosis.py` | 2 errors (pandas-stubs + import-untyped) -- pre-existing project-level issue, not Phase 43 code | INFO |

### Probe Execution

Step 7c: SKIPPED -- no probe scripts declared or expected for this phase.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-----------|-------------|--------|----------|
| DIAG-01 | 43-01, 43-02 | Shadow Comparison で Brier/logloss/ECE を比較、劣化次元特定 | SATISFIED | `_step1_probability_quality()` 実装。Brier/logloss/ECE/APR baseline vs shadow + Delta値。JSON/HTML/MD 全出力に含む |
| DIAG-02 | 43-01, 43-02 | 選定パターン差分 (baseline vs shadow) のレポート化 | SATISFIED | `_step2_selection_pattern()` 実装。changed/unchanged ROI/HR/avg_odds/APR。HTML "Step 2: Selection Pattern" セクション |
| DIAG-03 | 43-01, 43-02 | surface/odds_band/popularity_band/probability_rank_band/selected_changed 別の actual/predicted 比率乖離特定 | SATISFIED | `_step3_calibration_by_segment()` + 5 segment definitions。segment_cols に全5次元。odds_band は closing_win_odds + baseline_tanodds フォールバック付き |

Orphaned requirements: None. REQUIREMENTS.md maps DIAG-01/02/03 to Phase 43 and both plans declare all three.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/backtest/shadow_diagnosis.py` | 169 | `return []` (空リスト返却) | Info | `_resolve_variant_names` の正常系 (manifest に variants が無い場合) |
| `src/backtest/shadow_diagnosis.py` | 457-459 | `self.horse_diff` ではなく `horse_work` をチェックすべき (CR-01) | Warning | closing_win_odds フォールバック成功後も missing_inputs に誤報追加される可能性。数値結果には影響なし |

Debt marker gate: No TBD/FIXME/XXX markers found. PASS.

**CR-01 詳細分析:** Line 457 は `self.horse_diff` (元の不変 DataFrame) をチェックしている。フォールバックで `horse_work` に `closing_win_odds` がマージ成功しても、元の `self.horse_diff` には無いため missing_inputs に追加される。これは**誤報**だが、odds_band の計算自体は正常に完了するため、診断の数値結果への影響はない。出力レポートの missing_inputs セクションが不正確になるのみ。

**CR-02 詳細分析:** Line 474-481 で `selected_changed` を merge する際、`horse_work` に既に同名列が存在すると `_x`/`_y` サフィックスが付いて KeyError が発生する可能性。ただし Phase 41 の出力構造では `selected_changed` は race_diff のみに存在するため、実運用ではトリガーされない。エッジケース防御としては drop columns が推奨されるが Blocker ではない。

### Human Verification Required

#### 1. HTML レポートの Visual Review

**Test:** HTML レポートファイルをブラウザで開き、Step 1/2/3 セクションが正しく表示されることを確認する
**Expected:** 3ステップ段階的分析がセクション化表示され、delta悪化(Brier/logloss/ECE増加、APR減少)が赤色表示される。Missing Inputs セクションに不足列が表示される。
**Why human:** HTML の視覚的レイアウト、CSS スタイリング、テーブル描画、赤色ハイライトは grep で完全検証不可

#### 2. Markdown Summary の可読性確認

**Test:** shadow_diagnosis_summary.md を開き、5セクションのフォーマットと内容を確認する
**Expected:** Probability Quality / Selection Pattern / Top Calibration Gaps / Missing Inputs / Recommendations for Phase 44/45 が表形式で表示される。上位5キャリブレーションギャップが |delta_apr|+|delta_ece| 降順で表示される。
**Why human:** Markdown テーブルの可読性・フォーマット品質は人間判断

### Gaps Summary

**Critical gaps:** None.

**Warnings:**

1. **CR-01 (Warning):** `_add_segment_columns()` Line 457 で `self.horse_diff` をチェックすべきところを `self.horse_diff` (元の不変DataFrame) を見ているため、closing_win_odds フォールバック成功後も missing_inputs に誤報が追加される。数値結果への影響は無いが出力の正確性に影響。推奨修正: `self.horse_diff` を `horse_work` に変更。

2. **CR-02 (Warning):** `_add_segment_columns()` Line 474-478 で `selected_changed` の merge 前に既存列の drop 処理が無い。Phase 41 出力では horse_diff に selected_changed は含まれないため実害は無いが、防御的コードとしては改善推奨。

**Both warnings are non-blocking:** テストは全12件 PASS、lint は clean、診断ロジックは正しく動作する。Phase 44/45 への成果物提供に支障は無い。

---

_Verified: 2026-05-29T16:30:00Z_
_Verifier: Claude (gsd-verifier)_
