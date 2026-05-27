---
phase: 38-investmentfeatureframe
fixed_at: 2026-05-27T14:30:00Z
review_path: .planning/phases/38-investmentfeatureframe/38-REVIEW.md
iteration: 1
findings_in_scope: 6
fixed: 6
skipped: 0
status: all_fixed
---

# Phase 38: Code Review Fix Report

**Fixed at:** 2026-05-27T14:30:00Z
**Source review:** .planning/phases/38-investmentfeatureframe/38-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 6
- Fixed: 6
- Skipped: 0

## Fixed Issues

### CR-01: `if_ability_race_rank` outputs raw probability, not a race rank

**Files modified:** `src/investment/schema_registry.py`
**Commit:** eb2218c
**Applied fix:** Renamed spec from `if_ability_race_rank` to `if_p_ability_win` and updated description to "AbilityModel単勝確率 (順位化はPhase 39で実装)". The spec passes raw `p_ability_win` probability without any rank transformation; the new name accurately reflects the actual content. Rank transformation is deferred to Phase 39 as designed.

### CR-02: `if_odds_to_ability_dispersion` outputs raw ratio, not race-level dispersion

**Files modified:** `src/investment/schema_registry.py`
**Commit:** 0730fb5
**Applied fix:** Renamed spec from `if_odds_to_ability_dispersion` to `if_odds_ability_ratio_dup` and updated description to "オッズ/能力比 (分散化はPhase 39で実装, model_market_gapと同ソース)". The spec passes raw `odds_to_ability_ratio` without any race-level std transformation; the new name and description accurately reflect the actual content. Also renamed `missing_indicator` from `if_odds_to_ability_dispersion_missing` to `if_odds_ability_ratio_dup_missing`.

### WR-01: `if_odds_band_id` description misleading

**Files modified:** `src/investment/schema_registry.py`
**Commit:** bd8e074
**Applied fix:** Updated description from "オッズ帯ID" to "単勝オッズ (バンド変換はPhase 39で実装)" to accurately reflect that the spec passes raw tanodds without banding.

### WR-02: `load_or_compute` cache docstring missing warning

**Files modified:** `src/investment/cache.py`
**Commit:** da3d859
**Applied fix:** Added an "Important" note to the `load_or_compute` docstring warning that `source_artifact_hash` is the only guard against stale cache hits when source data changes but the column schema stays the same. Callers must ensure the hash changes whenever source data content changes.

### WR-03: `builder_version` parameter unused

**Files modified:** `src/investment/feature_frame.py`
**Commit:** 6964bd7
**Applied fix:** Removed the unused `builder_version` parameter from `InvestmentFeatureFrameBuilder.build_frame()` method and the module-level `build_frame()` function. The `build_train_frame` and `build_inference_frame` wrappers pass kwargs through so they needed no changes. The class attribute `BUILDERS_VERSION` is retained for external consumers.

### WR-04: Integration test may silently pass without verification

**Files modified:** `tests/test_investment_integration.py`
**Commit:** b7b80b7
**Applied fix:** Added `pytest.fail("No optional spec with non-required sources found to verify")` after the for loop in `test_optional_missing_produces_nan_with_indicator`. If no optional spec with non-required sources exists, the test now fails explicitly instead of passing silently.

---

_Fixed: 2026-05-27T14:30:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
