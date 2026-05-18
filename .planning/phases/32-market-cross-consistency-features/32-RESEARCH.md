# Phase 32: Market Cross-Consistency Features - Research

**Researched:** 2026-05-18
**Domain:** Harville theoretical odds / cross-betting-market consistency features
**Confidence:** HIGH

## Summary

Phase 32 adds 5 market cross-consistency features (MCF-02~06) that capture structural contradictions between win, wide (quinella), and trio (trifecta-box) betting markets. These features are computed by comparing actual odds against Harville theoretical odds derived from normalized implied win probabilities. The core mathematical engine is the Harville formula (Harville 1973), which predicts multi-horse outcome probabilities from single-horse win probabilities under an independence-of-conditionals assumption.

The implementation follows the Phase 31 `race_level_features.py` submodule pattern: a new `src/features/market_cross_features.py` module with `compute_market_cross_features(df, wide_df, trio_df)` as the entry point, a `MCF_COLS` export list, single-race/multi-race branching, and tuple-return for groupby.apply. Wide odds data loading is consolidated into DataRepository via a new `load_wide_odds()` method, eliminating training/backtest duplication (MCF-07).

**Primary recommendation:** Follow the Phase 31 submodule pattern exactly. Compute Harville theoretical odds from normalized `1/tanodds` implied probabilities per race. Compare against actual ninki=1 wide/trio odds. Merge wide/trio data inside `build_all()` after race_level_features and before SAFE-01 POST_RACE stripping.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Wide odds single-value = midpoint `(oddslow + oddshigh) / 2`
- **D-02:** `rl_wide_harville_ratio` targets ninki=1 combination only
- **D-03:** `rl_trio_odds_ratio` targets ninki=1 combination only
- **D-04:** Wide/trio odds load-and-merge method is Claude's discretion, but must satisfy MCF-07 (training/backtest dedup), architectural consistency, and build_all()/build_features() parity
- **D-05:** Data access uses DataRepository (Phase 29). Add `load_wide_odds()` alongside existing `load_trio_odds()`, `load_exacta_odds()`, `load_trifecta_odds()`
- **D-06:** Missing wide/trio odds -> NaN features. LightGBM handles NaN natively
- **D-07:** Missing data monitoring is Claude's discretion. Logging recommended
- **D-08:** Harville formula standard formulation:
  - Wide: P(i,j) = P(i) * P(j) / (1 - P(i)) (unordered combination, summed for both orderings)
  - Trio: P(i,j,k) = P(i) * P(j)/(1-P(i)) * P(k)/(1-P(i)-P(j)) (unordered combination, summed for all 6 orderings)
  - P(i) = (1/tanodds_i) / sum(1/tanodds_j) (normalized implied probability per race)
- **D-09:** Theoretical odds = 1 / theoretical probability. Ratio > 1.0 = market underestimates, < 1.0 = overestimates
- **D-10:** New module `src/features/market_cross_features.py`, following Phase 31 `race_level_features.py` pattern:
  - Main entry: `compute_market_cross_features(df, wide_df, trio_df)`
  - MCF_COLS list export
  - build_all()/build_features() parity

### Claude's Discretion
- build_all() wide/trio data merge integration method
- market_cross_features.py internal function structure
- Edge case handling (small fields, odds missing, kumi string parsing)
- Test case design
- Missing data monitoring method (logging etc.)
- Harville numerical stability (P(i) sum != 1 etc.)

### Deferred Ideas (OUT OF SCOPE)
- Top-3 combination individual deviation evaluation (ninki=1,2,3) -- future phase with trifecta-based MCF-08
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| MCF-01 | Harville formula theoretical wide odds calculation | Harville formula verified (D-08). P(i) from normalized implied probabilities. Standard implementation. |
| MCF-02 | rl_favorite_in_wide_top1 -- is favorite in wide ninki=1 combo (0/1) | Parse kumi 4-digit string, check if popularity_rank=1 horse is in the pair. |
| MCF-03 | rl_trio_overlap -- trio ninki=1 combo overlap with top-3 favorites (0-3) | Parse kumi 6-digit string, count overlap with top-3 by tanodds rank. |
| MCF-04 | rl_market_consistency -- is favorite in trio ninki=1 combo (0/1) | Parse kumi 6-digit string, check if popularity_rank=1 horse is in the triple. |
| MCF-05 | rl_trio_odds_ratio -- actual trio ninki=1 odds / Harville theoretical trio odds | Harville trio formula + ratio computation. ninki=1 only per D-03. |
| MCF-06 | rl_wide_harville_ratio -- actual wide ninki=1 odds / Harville theoretical wide odds | Harville wide formula + ratio computation. ninki=1 only per D-02. Midpoint odds per D-01. |
| MCF-07 | Wide odds merge into build_all() (training/backtest dedup) | DataRepository.load_wide_odds() + merge inside build_all() after race_level_features. |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Harville probability computation | Feature Engine (Python) | -- | Pure mathematical transformation from tanodds; no DB/network dependency |
| Wide/trio odds loading | DataRepository | -- | Centralized data access (D-05); ParquetStore-based |
| Feature integration into pipeline | FeatureEngine.build_all() / build_features() | -- | Submodule pattern; TimingContext integration |
| POST_RACE safety verification | Test suite | -- | AST source scan + whitelist + output verification (3-layer) |
| FEATURE_COLS update | Model classes | -- | All 12 model FEATURE_COLS lists need 5 new features |
| Manifest regeneration | freeze_feature_manifest.py | -- | SHA256 hash update for determinism |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | (existing) | Numerical computation for Harville formulas | Already in project; vectorized operations |
| pandas | (existing) | DataFrame manipulation, groupby, merge | Already in project; established groupby pattern |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| ParquetStore | (existing) | Wide/trio odds Parquet file reading | DataRepository.load_wide_odds() |
| DataRepository | (existing) | Centralized odds data loading | build_all() passes wide_df/trio_df to compute function |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual Harville computation | Pre-built library | No established Python Harville library exists. Manual implementation is simple (3-5 lines per formula) and auditable. |

**No new packages required.** This phase uses only existing numpy, pandas, and project-internal modules.

## Package Legitimacy Audit

No external packages are installed in this phase. All implementation uses existing project dependencies (numpy, pandas, pyarrow).

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
build_all(race_df, entry_df, odds_df, ...)
  |
  v
[1. race/entry/odds merge] --> result_df (per-horse rows with tanodds)
  |
  v
[2. compute_race_level_features(result_df)] --> rl_* columns
  |
  v
[3. Load wide/trio odds via DataRepository]
  |  wide_df = repo.load_wide_odds(start, end)    [ninki=1 rows]
  |  trio_df = repo.load_trio_odds(start, end)     [ninki=1 rows]
  |
  v
[4. compute_market_cross_features(result_df, wide_df, trio_df)]
  |  Parse tanodds -> implied probabilities -> Harville P(i), P(i,j), P(i,j,k)
  |  Parse kumi strings -> horse numbers
  |  Compare actual odds vs theoretical -> ratios
  |  Check favorite presence in combos -> binary features
  |
  v
[5. MCF features merged to result_df]
  |
  v
[6. SAFE-01: strip POST_RACE cols] --> final output
```

### Recommended Project Structure
```
src/
  features/
    market_cross_features.py    # NEW: compute_market_cross_features(), MCF_COLS
    race_level_features.py      # Phase 31 pattern template
    feature_engine.py           # MODIFIED: add MCF integration + wide/trio merge
  db/
    repository.py               # MODIFIED: add load_wide_odds()
  models/
    stage1_ability_model.py     # MODIFIED: FEATURE_COLS += 5 MCF features
    ... (10 more model files)   # MODIFIED: same
tests/
  test_market_cross_features.py # NEW: unit tests for MCF computation
  test_post_race_leakage.py     # MODIFIED: add TestMarketCrossFeatures class
scripts/
  freeze_feature_manifest.py    # RE-RUN: regenerate manifest
data/
  feature_freeze_manifest.json  # REGENERATED
```

### Pattern 1: Submodule Pattern (from Phase 31)
**What:** Independent compute_*() function in its own module, called from feature_engine.py with TimingContext
**When to use:** All new feature groups
**Example:**
```python
# src/features/market_cross_features.py
MCF_COLS: list[str] = [
    "rl_favorite_in_wide_top1",
    "rl_trio_overlap",
    "rl_market_consistency",
    "rl_trio_odds_ratio",
    "rl_wide_harville_ratio",
]

def compute_market_cross_features(
    df: pd.DataFrame,
    wide_df: pd.DataFrame | None = None,
    trio_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Market cross-consistency features (MCF-01~06)"""
    df = df.copy()
    if wide_df is None or trio_df is None:
        for col in MCF_COLS:
            df[col] = np.nan
        return df
    # ... dispatch to _compute_for_single_race / _compute_for_multi_race
```

```python
# src/features/feature_engine.py (build_all, after race_level_features)
from features.market_cross_features import compute_market_cross_features
with TimingContext("build_all/market_cross"):
    result_df = compute_market_cross_features(result_df, wide_df, trio_df)
```

### Pattern 2: Tuple Return for groupby.apply
**What:** Return tuples instead of dicts from groupby.apply to avoid scalar-unpacking bugs with single-group DataFrames
**When to use:** Any groupby("race_id").apply that returns multiple values
**Example:**
```python
# From race_level_features.py (_rank_features)
def _rank_features(group: pd.Series) -> tuple[float, float, float]:
    sorted_odds = np.sort(group.dropna().values)
    # ... compute
    return fav1, top3_gap, rank_gap

rank_results = tanodds_valid.groupby(race_ids, observed=True).apply(
    _rank_features, include_groups=False
)
df["rl_top1_odds"] = race_ids.map(rank_results.map(lambda x: x[0]))
```

### Pattern 3: DataRepository Method Addition
**What:** Add load_wide_odds() following existing load_trio_odds() pattern
**When to use:** Any new data source
**Example:**
```python
# src/db/repository.py
def load_wide_odds(self, start: str, end: str) -> pd.DataFrame:
    df = self._store.read("odds", "odds_wide", filters=date_filters(start, end))
    return coerce_types(df)
```

### Anti-Patterns to Avoid
- **Dict return from groupby.apply:** When only one group exists, pandas unpacks the dict values to scalars, causing KeyError. Use tuples instead. [VERIFIED: Phase 31 race_level_features.py implementation]
- **Mutating input DataFrame:** Always `df = df.copy()` at the start of compute_*() functions. [VERIFIED: race_level_features.py line 291]
- **Computing Harville for all ninki values:** Only ninki=1 is in scope (D-02, D-03). Computing for all ninki would be wasteful and out of scope.
- **Forgetting to normalize implied probabilities:** P(i) = (1/tanodds_i) / sum(1/tanodds_j) per race. Raw 1/tanodds values do not sum to 1.0 (overround).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Wide odds loading | Custom Parquet reader | DataRepository.load_wide_odds() | Centralized access, date filtering, type coercion |
| Trio odds loading | Custom Parquet reader | DataRepository.load_trio_odds() | Already exists from Phase 29 |
| Feature column tracking | Manual list management | MCF_COLS constant + FEATURE_COLS in model classes | Single source of truth, manifest compatibility |
| POST_RACE safety | Manual column audit | AST source scan + whitelist test | Established 3-layer pattern from Phase 31 |

**Key insight:** The Harville formula IS simple enough to implement directly (no library needed). The formula is ~5 lines per odds type. The complexity is in data parsing (kumi strings) and edge case handling (small fields, missing odds), not the math itself.

## Common Pitfalls

### Pitfall 1: kumi String Parsing Format Mismatch
**What goes wrong:** Wide kumi is 4 digits ("0405"), trio kumi is 6 digits ("030405"). Using the wrong parser gives wrong horse numbers.
**Why it happens:** Both are zero-padded numeric strings but different lengths.
**How to avoid:** Use separate parsers: `int(kumi[0:2]), int(kumi[2:4])` for wide; `int(kumi[0:2]), int(kumi[2:4]), int(kumi[4:6])` for trio.
**Warning signs:** Horse numbers > 18 (max field size in JRA) indicate wrong parsing.

### Pitfall 2: Harville Division by Zero
**What goes wrong:** When P(i) is close to 1.0 (extreme favorite), `1 - P(i)` approaches 0, causing division by zero in conditional probability terms.
**Why it happens:** Strong favorites in small fields. tanodds=1.0 gives P(i) near 1.0 after normalization.
**How to avoid:** Clamp denominators: use `max(1 - P(i), epsilon)` where epsilon = 1e-10. Return NaN when theoretical probability is not computable.
**Warning signs:** Features containing inf or extremely large values (>1e6).

### Pitfall 3: ninki Type Mismatch Between Wide and Trio
**What goes wrong:** Wide ninki is stored as string object ("001", "002"), trio ninki as Int64 (1, 2). Filtering `ninki == 1` fails for wide data.
**Why it happens:** ETL type coercion differences between tables.
**How to avoid:** Normalize ninki before filtering: `pd.to_numeric(df["ninki"], errors="coerce")`, then filter `== 1`.
**Warning signs:** Empty DataFrame after ninki=1 filter on wide data.

### Pitfall 4: Wide Odds < 1.0 (Sub-Yen Payouts)
**What goes wrong:** JRA wide odds can be less than 1.0 (e.g., 0.33) for extremely popular combinations. These are actual payout multipliers, not errors.
**Why it happens:** JRA wide bets have a minimum payout of 100 yen per 100 yen wagered, but the displayed odds can show less than 1.0 for the multiplier component.
**How to avoid:** Do NOT filter out or clamp odds < 1.0. These are valid data points. The Harville ratio will correctly reflect that actual < theoretical for these cases.
**Warning signs:** Wide odds values < 1.0 should NOT be treated as missing or erroneous.

### Pitfall 5: Harville Wide Formula Must Sum Both Orderings
**What goes wrong:** Using only P(i)*P(j)/(1-P(i)) without also adding P(j)*P(i)/(1-P(j)) gives asymmetric results.
**Why it happens:** The Harville formula for ordered pairs is P(i then j) = P(i)*P(j)/(1-P(i)). For unordered combinations (wide/quinella), both orderings must be summed.
**How to avoid:** P_wide(i,j) = P(i)*P(j)/(1-P(i)) + P(j)*P(i)/(1-P(j)) for i != j. This simplifies to P(i)*P(j) * (1/(1-P(i)) + 1/(1-P(j))).
**Warning signs:** Harville theoretical wide odds that are exactly 2x what they should be.

### Pitfall 6: build_all() Receives No Wide/Trio Data by Default
**What goes wrong:** Adding compute_market_cross_features() to build_all() but not loading wide/trio data inside build_all() means all features will be NaN.
**Why it happens:** build_all() signature only accepts race_df, entry_df, odds_df. Wide/trio data must be loaded separately.
**How to avoid:** Load wide/trio data inside build_all() via DataRepository (or accept as optional parameters). The build_features() path (inference) may not have wide/trio data -- handle gracefully with NaN fallback.
**Warning signs:** All MCF features are NaN in training output.

## Code Examples

### Harville Probability Computation
```python
# Source: Harville 1973 / standard formulation per D-08
def _harville_wide_prob(p_i: float, p_j: float) -> float:
    """Harville theoretical probability for wide (quinella) - unordered pair.
    
    P(i,j) = P(i)*P(j)/(1-P(i)) + P(j)*P(i)/(1-P(j))
           = P(i)*P(j) * (1/(1-P(i)) + 1/(1-P(j)))
    """
    eps = 1e-10
    denom_i = max(1.0 - p_i, eps)
    denom_j = max(1.0 - p_j, eps)
    return p_i * p_j * (1.0 / denom_i + 1.0 / denom_j)


def _harville_trio_prob(p_i: float, p_j: float, p_k: float) -> float:
    """Harville theoretical probability for trio (trifecta box) - unordered triple.
    
    P(i,j,k) = sum over all 6 permutations of:
        P(first) * P(second)/(1-P(first)) * P(third)/(1-P(first)-P(second))
    """
    eps = 1e-10
    perms = [(p_i, p_j, p_k), (p_i, p_k, p_j),
             (p_j, p_i, p_k), (p_j, p_k, p_i),
             (p_k, p_i, p_j), (p_k, p_j, p_i)]
    total = 0.0
    for a, b, c in perms:
        denom1 = max(1.0 - a, eps)
        denom2 = max(1.0 - a - b, eps)
        total += a * (b / denom1) * (c / denom2)
    return total
```

### kumi String Parsing
```python
# Source: verified against data/odds/odds_wide.parquet and odds_sanren.parquet
def _parse_wide_kumi(kumi: str) -> tuple[int, int]:
    """Parse 4-digit wide kumi string to horse numbers.
    "0405" -> (4, 5)
    """
    return int(kumi[0:2]), int(kumi[2:4])


def _parse_trio_kumi(kumi: str) -> tuple[int, int, int]:
    """Parse 6-digit trio kumi string to horse numbers.
    "030405" -> (3, 4, 5)
    """
    return int(kumi[0:2]), int(kumi[2:4]), int(kumi[4:6])
```

### Implied Probability Normalization (per-race)
```python
# Source: race_level_features.py established pattern
tanodds = pd.to_numeric(df["tanodds"], errors="coerce").replace(0, np.nan)
valid_mask = tanodds.notna() & (tanodds > 0)
inv_odds = 1.0 / tanodds[valid_mask]
total = inv_odds.sum()
p_norm = inv_odds / total  # Normalized implied probabilities
```

### Wide Odds Midpoint (per D-01)
```python
# Source: CONTEXT.md D-01
wide_mid = (wide_df["oddslow"] + wide_df["oddshigh"]) / 2
```

### ninki Normalization Before Filtering
```python
# Source: verified against data -- wide ninki is string, trio ninki is Int64
wide_df["ninki_num"] = pd.to_numeric(wide_df["ninki"], errors="coerce")
wide_n1 = wide_df[wide_df["ninki_num"] == 1]

trio_df["ninki_num"] = pd.to_numeric(trio_df["ninki"], errors="coerce")
trio_n1 = trio_df[trio_df["ninki_num"] == 1]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Harville exacta pricing | Same formula, still standard | Harville 1973 | Standard academic approach, no newer replacement |
| Direct odds loading via readers.py | DataRepository centralized access | Phase 29 (v1.7) | All data access goes through single class |
| Per-script odds merge | build_all() integrated merge | Phase 32 (this phase) | Eliminates training/backtest duplication |

**Deprecated/outdated:**
- readers.py direct usage: Being replaced by DataRepository (D-05). readers.py functions remain as implementation references but should not be called directly from feature_engine.py.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Wide odds midpoint (oddslow+oddshigh)/2 is the correct single-value derivation per D-01 | Standard Stack | Low -- locked decision |
| A2 | Harville unordered wide formula sums both orderings | Code Examples | Medium -- if only one ordering used, theoretical odds are 2x too small |
| A3 | Harville unordered trio formula sums all 6 permutations | Code Examples | Medium -- if only one ordering used, theoretical odds are 6x too small |
| A4 | Wide ninki is string dtype, trio ninki is Int64 dtype | Common Pitfalls | Low -- verified against actual Parquet data |
| A5 | build_features() (inference path) will not have wide/trio data available, so MCF features will be NaN | Architecture Patterns | Low -- acceptable per D-06, LightGBM handles NaN |
| A6 | Wide odds values can be < 1.0 (sub-yen payouts) and are valid data | Common Pitfalls | Low -- verified against actual Parquet data (min oddslow ~0.3) |

## Open Questions

1. **How should build_all() obtain the date range for wide/trio data loading?**
   - What we know: build_all() receives race_df which has race_date column. DataRepository.load_wide_odds() requires start/end date strings.
   - What's unclear: Whether to extract date range from race_df inside build_all(), or pass it as a parameter, or have the caller (TrainingPipeline/BacktestEngine) load and pass the data.
   - Recommendation: Extract from race_df inside build_all() (simplest, self-contained). The race_df always has race_date when called from training/backtest paths.

2. **Should compute_market_cross_features() filter wide/trio to ninki=1 internally, or receive pre-filtered data?**
   - What we know: Only ninki=1 is needed (D-02, D-03).
   - What's unclear: Whether filtering happens at load time or compute time.
   - Recommendation: Filter at compute time (inside the function). This keeps DataRepository generic and makes the ninki=1 constraint visible in one place.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| data/odds/odds_wide.parquet | Wide odds loading | Verified | 3.67M rows | -- |
| data/odds/odds_sanren.parquet | Trio odds loading | Verified | 15.58M rows | -- |
| DataRepository | load_wide_odds() | Yes | Phase 29 | -- |
| ParquetStore | DataRepository backend | Yes | Existing | -- |
| numpy | Harville computation | Yes | Existing | -- |
| pandas | DataFrame operations | Yes | Existing | -- |

**Missing dependencies with no fallback:**
- None

**Missing dependencies with fallback:**
- None

## Sources

### Primary (HIGH confidence)
- Phase 31 `race_level_features.py` -- submodule pattern, groupby.apply tuple-return, single/multi race branching
- Phase 29 `repository.py` -- DataRepository method pattern (load_trio_odds, load_exacta_odds, load_trifecta_odds)
- `feature_engine.py` lines 345-348, 460-462 -- integration points for race_level_features
- `data/odds/odds_wide.parquet` -- verified schema: kumi (4-digit string), oddslow/oddshigh (float64), ninki (string object)
- `data/odds/odds_sanren.parquet` -- verified schema: kumi (6-digit string), odds (float64), ninki (Int64)
- CONTEXT.md D-01~D-10 -- locked decisions

### Secondary (MEDIUM confidence)
- Harville 1973 formula -- standard academic reference, verified via multiple sources
- `tests/test_post_race_leakage.py` -- established 3-layer test pattern

### Tertiary (LOW confidence)
- None -- all findings verified against codebase or locked decisions

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no new packages, all existing infrastructure
- Architecture: HIGH - follows established Phase 31 pattern exactly
- Pitfalls: HIGH - verified against actual Parquet data and established patterns
- Harville formula: HIGH - standard academic formula verified from multiple sources

**Research date:** 2026-05-18
**Valid until:** 2026-06-18 (stable - no external dependency changes expected)
