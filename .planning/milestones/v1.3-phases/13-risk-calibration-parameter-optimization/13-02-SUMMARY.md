---
phase: 13-risk-calibration-parameter-optimization
plan: 02
subsystem: models, betting, backtest
tags: [regime-detector, meta-switcher, parameter-freeze, sha256, optuna-preset]

# Dependency graph
requires:
  - phase: 11-bet-selection-filters
    provides: RegimeDetector, MetaSwitcher, ParameterFreezeProtocol baseline
provides:
  - RegimeDetector override_params injection for Optuna optimization
  - MetaSwitcher _default_params aligned with RegimeDetector values
  - Strategy manifest JSON save/verify/load (SHA256 tamper detection)
  - TestStrategyManifest 8-test suite
affects: [13-03 (strategy_optimizer), backtest validation pipeline]

# Tech tracking
tech-stack:
  added: []
  patterns: [constructor-injection-for-params, json-manifest-sha256]

key-files:
  created: []
  modified:
    - src/models/regime_detector.py
    - src/betting/meta_switcher.py
    - src/backtest/parameter_freeze_protocol.py
    - tests/test_parameter_freeze.py

key-decisions:
  - "RegimeDetector get_strategy_params() override via _get_base_params() + shim pattern (backward compatible)"
  - "MetaSwitcher values aligned to RegimeDetector as single source of truth"
  - "Strategy manifest uses sort_keys=True + indent=2 for deterministic SHA256 across save/verify"

patterns-established:
  - "Constructor injection pattern: override_params dict per regime for parameter externalization"
  - "JSON manifest + SHA256 pattern for parameter integrity during test-period backtest"

requirements-completed: [VAL-01]

# Metrics
duration: 4min
completed: 2026-05-05
---

# Phase 13 Plan 02: Parameter Externalization Summary

**RegimeDetector主要3パラメータ(fractional_kelly, ev_threshold, edge_threshold)のコンストラクタ注入外部化、MetaSwitcher値乖離解消、戦略パラメータJSON manifest + SHA256改ざん検知機能の実装**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-05T01:37:51Z
- **Completed:** 2026-05-05T01:42:41Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- RegimeDetector.get_strategy_params() の主要3パラメータを override_params で外部上書き可能に (Optuna最適化の前提)
- MetaSwitcher._default_params() の5箇所の値乖離を解消 (ev_threshold/edge_threshold を RegimeDetector と完全一致)
- save_strategy_manifest() / verify_strategy_manifest() / load_and_freeze_strategy() の3関数を追加 (JSON + SHA256)
- TestStrategyManifest 8テスト追加、既存テスト36件全PASS

## Task Commits

Each task was committed atomically:

1. **Task 1: RegimeDetector外部化 + MetaSwitcher値揃え** - `c5b356b` (feat)
2. **Task 2: ParameterFreezeProtocol JSON manifest + SHA256 + テスト** - `13493b8` (feat)

## Files Created/Modified
- `src/models/regime_detector.py` - override_params注入、_get_base_params()リファクタ
- `src/betting/meta_switcher.py` - ev_threshold/edge_threshold値をRegimeDetectorに揃えた
- `src/backtest/parameter_freeze_protocol.py` - save/verify/load_strategy_manifest 追加
- `tests/test_parameter_freeze.py` - TestStrategyManifest 8テスト追加

## Decisions Made
- 既存のget_strategy_params()本体を_get_base_params()にリネームし、get_strategy_params()をオーバーライドシムにする設計（既存コードに影響なし）
- SHA256計算は sort_keys=True + indent=2 で統一（save/verify間のdeterministic保証）
- manifest関数はモジュールレベル関数として追加（既存ParameterFreezeProtocolクラスに変更なし）

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] SHA256 hash mismatch between save and verify**
- **Found during:** Task 2 (TestStrategyManifest tests)
- **Issue:** save_strategy_manifest() は `indent=2` でハッシュ計算していたが、verify_strategy_manifest() は `indent` なしで再計算していたためハッシュ不一致
- **Fix:** verify_strategy_manifest() の json.dumps に `indent=2` を追加して save 側と一致させた
- **Files modified:** src/backtest/parameter_freeze_protocol.py
- **Verification:** 全14テストPASS（3テストがFAIL→PASSに解消）
- **Committed in:** 13493b8 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** SHA256照合の正確性修正。スコープ変更なし。

## Issues Encountered
None - 計画通りに実行完了

## Next Phase Readiness
- RegimeDetector外部化完了 → Plan 03 (strategy_optimizer) でOptuna最適化パラメータ注入可能
- JSON manifest保存/検証完了 → Plan 03 で最適化結果の凍結・不変性保証可能
- 既存テスト全36件PASS、リグレッションなし

---
*Phase: 13-risk-calibration-parameter-optimization*
*Completed: 2026-05-05*

## Self-Check: PASSED

- [x] src/models/regime_detector.py EXISTS
- [x] src/betting/meta_switcher.py EXISTS
- [x] src/backtest/parameter_freeze_protocol.py EXISTS
- [x] tests/test_parameter_freeze.py EXISTS
- [x] 13-02-SUMMARY.md EXISTS
- [x] Commit c5b356b FOUND
- [x] Commit 13493b8 FOUND
