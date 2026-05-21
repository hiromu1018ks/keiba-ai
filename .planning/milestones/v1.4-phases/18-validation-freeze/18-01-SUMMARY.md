---
phase: 18-validation-freeze
plan: 01
subsystem: backtest
tags: [pfp, sha256, manifest, freeze-verify, val-02]
dependency_graph:
  requires: [phase-17-optuna-optimization]
  provides: [pfp-dual-verification-in-engine, manifest-path-wiring]
  affects: [src/backtest/engine.py, scripts/run_backtest.py]
tech_stack:
  added: []
  patterns: [PFP freeze/verify lifecycle, SHA256 manifest verification, helper method for DRY verify]
key_files:
  created: []
  modified:
    - src/backtest/engine.py
    - scripts/run_backtest.py
    - tests/test_backtest_engine.py
decisions:
  - PFP verifyを全returnパスで実行するため_verify_pfp()ヘルパーメソッドを追加
metrics:
  duration: 227s
  completed: "2026-05-06T22:55:27Z"
  tasks: 2
  files: 3
---

# Phase 18 Plan 01: PFP二重検証統合 + manifest配線 Summary

BacktestEngineにPFP freeze/verify二重検証を統合し、run_backtest.pyからmanifest_pathをengineに渡す配線を完了。SHA256 manifest検証 + ParameterFreezeProtocol検証の二重検証により、OOS期間中のモデル不変性を保証。

## Changes

### Task 1: BacktestEngineにPFP freeze/verify二重検証を統合 (TDD)

**RED/GREEN:**
- `TestBacktestEnginePFPIntegration` クラスに5つのテストを追加
- `BacktestEngine.__init__()` に `manifest_path: Path | None = None` 引数を追加
- `run()` 先頭で `verify_strategy_manifest()` SHA256検証 + `ParameterFreezeProtocol.freeze()` を実行
- `_verify_pfp()` ヘルパーメソッドで全returnパスでPFP verifyを実行
- PFP verify失敗時にRuntimeError送出 (D-04)
- manifest_path=Noneの場合はPFP関連コードが一切実行されない (後方互換)

**Deviation:** 計画ではrun()末尾のBacktestResult返却前のみにPFP verifyを配置していたが、実装時にrun()内に4つの早期returnパス(empty race_df, empty odds_ts, missing hassotime, empty pre_post_odds)が存在することを発見。全returnパスでPFP verifyを実行する必要があったため、`_verify_pfp()` ヘルパーメソッドを導入してDRYにした。(Rule 1 - 実装上のバグ修正)

### Task 2: run_backtest.pyにmanifest_path渡し + validate_args拡張

- `validate_args()` に `--strategy-manifest requires --ensemble` チェックを追加
- `_run_single_year()` のBacktestEngineコンストラクタに `manifest_path` を追加
- `_run_multi_year()` のBacktestEngineコンストラクタに `manifest_path` を追加

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] PFP verifyが早期returnパスで実行されない問題を修正**
- **Found during:** Task 1 GREEN phase
- **Issue:** 計画ではrun()末尾のみにPFP verifyを配置していたが、run()内に4つの早期returnパス(empty data)が存在。これらのパスではPFP verifyがスキップされ、OOS期間中のモデル不変性検証が抜け落ちていた
- **Fix:** `_verify_pfp()` プライベートヘルパーメソッドを導入し、全returnパス(早期return 4箇所 + 末尾 1箇所 = 計5箇所)で呼び出すように変更
- **Files modified:** src/backtest/engine.py
- **Commit:** 756480c

## Test Results

```
tests/test_backtest_engine.py::TestBacktestEnginePFPIntegration - 5 passed
tests/test_backtest_engine.py (全体) - 59 passed
```

## Verification

| 項目 | 結果 |
|------|------|
| PFP integration 5テスト全PASS | PASS |
| `manifest_path` in engine.py >= 4件 | 5件 |
| `manifest_path` in run_backtest.py >= 3件 | 6件 |
| validate_argsにstrategy_manifest/ensembleチェック含む | 含む(行117) |
| 既存59テスト回帰なし | 59 passed |

## Key Commits

| Commit | Description |
|--------|-------------|
| 756480c | feat(18-01): PFP freeze/verify二重検証をBacktestEngineに統合 |
| b4b7b73 | feat(18-01): run_backtest.pyにmanifest_path渡し + validate_args拡張 |

## Self-Check: PASSED

All files and commits verified present.
