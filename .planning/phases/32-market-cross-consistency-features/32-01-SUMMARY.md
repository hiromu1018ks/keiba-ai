---
phase: 32-market-cross-consistency-features
plan: 01
subsystem: features
tags: [harville, market-cross-consistency, wide-odds, trio-odds, lightgbm, parquet]

# Dependency graph
requires:
  - phase: 29-etl-extensions
    provides: "DataRepository基盤 + wide/trio Parquetファイル"
  - phase: 31-race-level-aggregation-features
    provides: "race_level_features.pyパターン (submodule, single/multi-race branching)"
provides:
  - "compute_market_cross_features() 5特徴量 (MCF-01~06)"
  - "DataRepository.load_wide_odds() メソッド"
  - "MCF_COLS エクスポートリスト"
affects: [32-02, feature_engine, model FEATURE_COLS]

# Tech tracking
tech-stack:
  added: []
  patterns: [harville-formula, market-cross-consistency, submodule-pattern]

key-files:
  created:
    - src/features/market_cross_features.py
    - tests/test_market_cross_features.py
  modified:
    - src/db/repository.py

key-decisions:
  - "groupby.apply(include_groups=False)でrace_idがgroupから除外されるためfor-loopでname/groupを直接処理"
  - "_get_prob_for_umaban()でインプライド確率をumaban→index逆引きで取得"
  - "Index.iloc[]不存在のためIndex[]直接アクセスに修正"

patterns-established:
  - "Harville formula pattern: epsilon=1e-10分母保護 + 両順序和(ワイド)/6順列和(三連複)"
  - "ninki正規化パターン: pd.to_numeric(ninki, errors='coerce')後==1フィルタで文字列/整数両対応"
  - "market_crossサブモジュールパターン: race_level_features.pyと同じsingle/multi-race分岐"

requirements-completed: [MCF-01, MCF-02, MCF-03, MCF-04, MCF-05, MCF-06]

# Metrics
duration: 29min
completed: 2026-05-18
---

# Phase 32 Plan 01: Market Cross-Consistency Features Summary

**Harville理論オッズによる馬券種クロス整合性5特徴量 + DataRepository.load_wide_odds()を実装**

## Performance

- **Duration:** 29 min
- **Started:** 2026-05-18T10:04:10Z
- **Completed:** 2026-05-18T10:33:32Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 3

## Accomplishments
- compute_market_cross_features()で5つのMCF特徴量を実装 (rl_favorite_in_wide_top1, rl_trio_overlap, rl_market_consistency, rl_trio_odds_ratio, rl_wide_harville_ratio)
- Harville公式のワイド(両順序和)・三連複(6順列和)確率計算エンジンを実装
- epsilon=1e-10によるdivision-by-zero保護 (T-32-02)
- DataRepository.load_wide_odds()を追加しwideオッズの中央集権アクセスを実現 (D-05)
- 16テスト全て通過 (None/empty fallback, 正常ケース, Harville比率, エッジケース, ninki型違い, DataRepository mock)

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): failing tests for MCF features** - `300bff1` (test)
2. **Task 1 (GREEN): implement market cross-consistency features** - `ef333aa` (feat)

_Note: TDD task with RED/GREEN cycle. REFACTOR not needed (clean implementation)._

## TDD Gate Compliance

- RED gate: `300bff1` - test(32-01): add failing test for market cross-consistency features (16 tests failing with ModuleNotFoundError)
- GREEN gate: `ef333aa` - feat(32-01): implement market cross-consistency features with Harville formula (16 tests passing)
- REFACTOR gate: Not needed - implementation was clean after GREEN

## Files Created/Modified
- `src/features/market_cross_features.py` - Harville理論オッズ計算エンジン + 5MCF特徴量 (新規作成, ~500行)
- `src/db/repository.py` - load_wide_odds()メソッド追加 (既存ファイル更新)
- `tests/test_market_cross_features.py` - 16テストケース (新規作成, ~340行)

## Decisions Made
- groupby.apply(include_groups=False)でrace_idがgroup DataFrameから除外される問題に対し、for-loopで(name, group)を直接処理する方式を採用 -- pandas API制約への対応
- _get_prob_for_umaban()でインプライド確率をumaban→index逆引きで取得 -- kumi文字列パース後の馬番照合に必要
- Indexオブジェクトに.iloc[]が存在しないため、[]直接アクセスに修正 -- ruff lint対応

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed groupby.apply race_id exclusion with include_groups=False**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** groupby("race_id").apply() with include_groups=False はgroupからrace_id列を除外するため、_process_race内でgroup["race_id"]がKeyError
- **Fix:** groupby.applyの代わりにfor name, groupループでrace_idを直接取得する方式に変更
- **Files modified:** src/features/market_cross_features.py
- **Verification:** 全16テスト通過
- **Committed in:** ef333aa (Task 1 GREEN commit)

**2. [Rule 1 - Bug] Fixed Index.iloc AttributeError**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** valid.sort_values().index はIndexオブジェクトを返し、.iloc[]を持たないためAttributeError
- **Fix:** sorted_idx.iloc[rank-1] → sorted_idx[rank-1] に修正
- **Files modified:** src/features/market_cross_features.py
- **Verification:** 全16テスト通過
- **Committed in:** ef333aa (Task 1 GREEN commit)

**3. [Rule 3 - Blocking] Fixed line-length violations**
- **Found during:** Task 1 (GREEN phase, lint check)
- **Issue:** 3行がline-length=100制限を超過 (最大136文字)
- **Fix:** 長い三項演算子を変数抽出に分割、タプルを複数行に分割
- **Files modified:** src/features/market_cross_features.py
- **Verification:** ruff check -- 全checks passed
- **Committed in:** ef333aa (Task 1 GREEN commit)

---

**Total deviations:** 3 auto-fixed (2 bug, 1 blocking)
**Impact on plan:** pandas API互換性とlint準拠の修正のみ。スコープクリープなし。

## Issues Encountered
- groupby.applyのinclude_groups=False挙動がrace_level_features.pyパターンと異なるため、for-loop方式に切り替えが必要だった。race_level_features.pyはgroupby.apply内でgroup["race_id"]にアクセスしていないため問題が顕在化していなかった。

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- compute_market_cross_features()とDataRepository.load_wide_odds()が利用可能
- Plan 02 (build_all統合, FEATURE_COLS更新, POST_RACEテスト追加, manifest再生成) への準備完了
- wide_df/trio_dfのロード方法はPlan 02でbuild_all()内に統合予定

---
*Phase: 32-market-cross-consistency-features*
*Completed: 2026-05-18*

## Self-Check: PASSED

- [x] src/features/market_cross_features.py: FOUND
- [x] src/db/repository.py: FOUND
- [x] tests/test_market_cross_features.py: FOUND
- [x] Commit 300bff1 (test): FOUND
- [x] Commit ef333aa (feat): FOUND
