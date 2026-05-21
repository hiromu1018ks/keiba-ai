---
phase: 18-validation-freeze
verified: 2026-05-07T12:00:00Z
status: human_needed
score: 9/10 must-haves verified
overrides_applied: 0
human_verification:
  - test: "run_backtest.py --ensemble --strategy-manifest PATH でバックテストを実行し、ROI>100%かつ100+ベットを確認する"
    expected: "data/validation/validation_report.json が生成され、validation_result=PASS になる"
    why_human: "PostgreSQL接続(EveryDB2)が必要。実データでのバックテスト実行はmock不可。D-09でHuman UAT指定"
---

# Phase 18: Validation & Freeze Verification Report

**Phase Goal:** アンサンブルバックテストで年間100+ベットかつROI>100%が確認され、最適化済みパラメータが改ざん検知付きで固定されている状態になる
**Verified:** 2026-05-07
**Status:** human_needed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | BacktestEngine.run()がmanifest_pathを受け取った場合、run()先頭でverify_strategy_manifest()によるSHA256検証を実行する | VERIFIED | engine.py:518-521 -- `if self._manifest_path is not None: verify_strategy_manifest(self._manifest_path)` を確認。テスト test_manifest_path_triggers_verify_strategy_manifest PASS |
| 2 | BacktestEngine.run()がmanifest_pathを受け取った場合、PFP freeze()をrun()先頭で実行し、PFP verify()をrun()末尾(BacktestResult返却前)で実行する | VERIFIED | engine.py:524-525 で freeze()、engine.py:1219-1223 で verify()。早期returnパス4箇所でも `_verify_pfp()` で verify 実行(536, 562, 569, 579)。テスト test_manifest_path_triggers_pfp_freeze_and_verify PASS |
| 3 | PFP verify失敗時(RuntimeError)またはSHA256不一致時(ValueError)にバックテストが即時停止する | VERIFIED | engine.py:1221-1222 で `raise RuntimeError(pfp_result["message"])`。verify_strategy_manifest は ValueError/FileNotFoundError 送出。テスト test_pfp_verify_failure_raises_runtime_error + test_manifest_missing_path_raises_file_not_found PASS |
| 4 | run_backtest.pyがmanifest_pathをBacktestEngineコンストラクタに渡す | VERIFIED | run_backtest.py:473(_run_single_year), :616(_run_multi_year) -- `manifest_path=Path(args.strategy_manifest) if args.strategy_manifest else None` を確認 |
| 5 | --strategy-manifest単独(--ensembleなし)指定時に引数エラーとなる | VERIFIED | run_backtest.py:117-118 -- `if args.strategy_manifest and not args.ensemble: parser.error(...)` を確認 |
| 6 | バックテスト完了後に検証結果JSON(validation_report.json)がdata/validation/に出力される | VERIFIED | engine.py:1255-1285 -- try/except内で generate_validation_report() -> data/validation/validation_report.json へ書き出し。run_backtest.py:721-760 でマルチ年度版 multi_year_validation_report.json も出力 |
| 7 | 検証結果JSONにROI、ベット数、テスト期間、PFP検証結果(PASS/FAIL)、年別内訳が含まれる | VERIFIED | validation_report.py:93-111 -- report dictにvalidation_timestamp, test_period, train_period, manifest, pfp_verification, roi(total_roi/total_bets/total_stake/total_return/target_roi/target_bets/passed), yearly_breakdown, validation_result, cause_analysis を含む |
| 8 | ROI>100%かつ100+ベットの場合validation_result=PASS、そうでなければFAILとなる | VERIFIED | validation_report.py:22-34 -- `evaluate_validation()`: `if roi > 1.0 and total_bets >= 100: return "PASS"` テスト3件(PASS/FAIL-ROI/FAIL-bets)全てPASS |
| 9 | ROI<100%の場合、オッズバンド別ROI、レジーム別ROI、EV診断別ROI、芝ダート別ROIを含む原因分析が自動生成される | VERIFIED | validation_report.py:114-248 -- generate_cause_analysis() が odds_band_roi(4バンド), regime_roi, ev_diagnosis(over/under), bet_count_sufficiency, surface_roi を含むdict返却。validation_report.py:88-89 で `total_roi <= 1.0` 時のみ cause_analysis 設定。テスト5件(odds_bands/regime/empty/missing_fields)全てPASS |
| 10 | アンサンブルバックテストの結果が実際に年間100+ベットかつROI>100%を達成している | HUMAN_NEEDED | コード上の判定ロジックは正しいが、実データでのバックテスト実行結果はHuman UATが必要(PostgreSQL依存)。D-09で指定済み |

**Score:** 9/10 truths verified (1 requires human verification)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/backtest/engine.py` | BacktestEngine PFP freeze/verify + manifest SHA256再検証 + 検証レポート出力 | VERIFIED | manifest_path引数(372), PFP freeze(524-525), _verify_pfp helper(490-499), verify in run()(1219-1223), validation report output(1255-1285) |
| `scripts/run_backtest.py` | manifest_path渡し + validate_args拡張 + マルチ年度検証レポート | VERIFIED | validate_args(117-118), _run_single_year(473), _run_multi_year(616, 721-760) |
| `src/backtest/validation_report.py` | 検証結果JSON生成 + 原因分析レポート生成 | VERIFIED | 3公開関数: evaluate_validation(22-34), generate_validation_report(37-111), generate_cause_analysis(114-248)。283行 |
| `tests/test_backtest_validation.py` | VAL-01/VAL-02検証テスト(全mockベース) | VERIFIED | TestValidationReport クラス8テスト全PASS |
| `tests/test_backtest_engine.py` | PFP統合テスト | VERIFIED | TestBacktestEnginePFPIntegration クラス5テスト全PASS |
| `data/validation/` | 検証結果JSON出力先ディレクトリ | VERIFIED | engine.py:1274-1275 で mkdir(parents=True, exist_ok=True) により実行時自動作成 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| scripts/run_backtest.py | src/backtest/engine.py | BacktestEngine(manifest_path=...) | WIRED | 473行, 616行, 748行でmanifest_path渡し確認 |
| src/backtest/engine.py | src/backtest/parameter_freeze_protocol.py | verify_strategy_manifest + ParameterFreezeProtocol | WIRED | import(17-20), run()先頭(518-525), run()末尾(1219-1223) |
| src/backtest/engine.py | src/backtest/validation_report.py | run()末尾でgenerate_validation_report() | WIRED | import(1257), call(1264-1272), file output(1274-1282) |
| scripts/run_backtest.py | src/backtest/validation_report.py | マルチ年度全体検証レポート | WIRED | import(724), call(730-749), file output(751-758) |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| engine.py run() | backtest_result (BacktestResult) | race loop内でtotal_bets/stake/return/bet_history集計 | Yes (live計算) | FLOWING |
| engine.py validation output | report (dict) | generate_validation_report(backtest_result, ...) | Yes (backtest_resultから派生) | FLOWING |
| validation_report.py | yearly_breakdown | bet_history[].race_date[:4] | Yes (bet_historyから年別集計) | FLOWING |
| validation_report.py | cause_analysis | bet_history[].final_odds/regime/surface/ev | Yes (bet_historyから分析) | FLOWING |
| run_backtest.py multi_year | multi_report (dict) | generate_validation_report(aggregate BacktestResult) | Yes (all_resultsから集計) | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| evaluate_validation PASS | `python -c "from backtest.validation_report import evaluate_validation; print(evaluate_validation(1.05, 200))"` | PASS | PASS |
| evaluate_validation FAIL (ROI) | `python -c "from backtest.validation_report import evaluate_validation; print(evaluate_validation(0.89, 200))"` | FAIL | PASS |
| evaluate_validation FAIL (bets) | `python -c "from backtest.validation_report import evaluate_validation; print(evaluate_validation(1.05, 50))"` | FAIL | PASS |
| PFP integration 5テスト | `python -m pytest tests/test_backtest_engine.py -v -k PFPIntegration` | 5 passed | PASS |
| validation_report 8テスト | `python -m pytest tests/test_backtest_validation.py -v` | 8 passed | PASS |
| 全テストスイート | `python -m pytest tests/ -q` | 1327 passed, 1 skipped | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| VAL-01 | 18-02 | アンサンブルバックテストで年間100+ベットかつROI>100%を達成することを確認する | PARTIAL | 判定ロジック(evaluate_validation)と出力インフラ(generate_validation_report)は実装済み。実際のバックテスト実行結果確認はHuman UATが必要 |
| VAL-02 | 18-01, 18-02 | ParameterFreezeProtocolで最適化済みパラメータを固定し、SHA256改ざん検知を適用する | VERIFIED | BacktestEngine.run()内でverify_strategy_manifest(SHA256)+PFP freeze/verify二重検証を実行。5テスト全PASS |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (なし) | - | - | - | スキャン結果: TODO/FIXME/PLACEHOLDER/空実装なし |

### Human Verification Required

### 1. アンサンブルバックテストROI検証

**Test:** `python scripts/run_backtest.py --ensemble --strategy-manifest data/tuning/strategy_manifest.json --years 2024 2025 --train-window 4`
**Expected:** data/validation/multi_year_validation_report.json が生成され、validation_result="PASS" (ROI>1.0 かつ total_bets>=100)
**Why human:** PostgreSQL接続(localhost:5432/everydb2)が必要。実データでのバックテストはCI環境では実行不可。D-09決定でHuman UAT指定

### 2. 検証レポート内容確認

**Test:** 出力された validation_report.json / multi_year_validation_report.json の内容を目視確認
**Expected:** ROI、ベット数、年別内訳、PFP検証結果、(ROI<=1.0の場合) cause_analysisが含まれる
**Why human:** JSONファイルの内容確認は目視が必要

### Gaps Summary

Phase 18の目標達成に必要なコードインフラは全て実装されている:

1. **PFP二重検証(VAL-02):** BacktestEngine.run()の先頭でSHA256検証+PFP freeze、末尾でPFP verify。失敗時は即時停止(RuntimeError)。5テスト全PASS。完全実装。

2. **検証レポート生成(VAL-01):** evaluate_validation()でD-06基準(ROI>1.0, bets>=100)判定。generate_validation_report()でJSON出力。generate_cause_analysis()で5項目原因分析。8テスト全PASS。完全実装。

3. **配線:** run_backtest.pyの_single_year/_multi_year両方でmanifest_pathをengineに渡す。validate_args()で--strategy-manifest+--ensemble組み合わせバリデーション。完全実装。

4. **テスト:** 13新規テスト(5+8)全PASS。既存1327テスト回帰なし。

唯一、ROADMAP SC 1「実際にROI>100%+100+ベットを達成している」は、実データでのバックテスト実行結果に依存するためHuman UATが必要。D-09で明示的に指定されている通り。

---

_Verified: 2026-05-07T12:00:00Z_
_Verifier: Claude (gsd-verifier)_
