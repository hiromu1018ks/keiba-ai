# Phase 24: Feature Audit & Pruning - Research

**Researched:** 2026-05-12
**Domain:** Feature importance audit, noise pruning, cache invalidation
**Confidence:** HIGH

## Summary

Phase 24 builds directly on Phase 23's audit infrastructure to quantify feature effectiveness across all 7 model classes, identify noise features via Tier 1 (Gain=0 AND Perm<=0) criteria, and implement code-hash based cache invalidation. The codebase is well-prepared: `compute_all_model_importance()`, `identify_noise_features()`, and `validate_noise_removal()` are already implemented in `win_feature_analysis.py`. The total feature landscape spans 139 unique features across 7 models, with ConformalEVModel having the most at 131 features. The existing cache mechanism in `feature_engine.py` uses SHA-256 of source parquet paths + date ranges, but lacks code-hash awareness.

**Primary recommendation:** Extend `identify_noise_features()` to support Tier 1/Tier 2 classification with permutation importance, then run the audit, apply per-model pruning via FEATURE_COLS edits, validate with OOF logloss/AUC via `validate_noise_removal()`, and add code-hash to `compute_cache_key()`.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** 多段階プルーニング。Tier 1（Gain=0 AND Permutation<=0）を自動除外、Tier 2（低重要度）をレポート出力してユーザー判断に委ねる。
- **D-02:** 適用単位はモデル別個別プルーニング。各モデルのFEATURE_COLSを独立に最適化する。
- **D-03:** Tier 1除外の安全性確認はOOF logloss/AUC比較で実施。フルバックテストの前に高速な品質チェック。
- **D-04:** 段階的ROI検証 — Step 1: OOF logloss/AUC → Step 2: 通過したらフルBT。v1.5ベースライン ROI 84.4%。
- **D-05:** フルBTでROI悪化時は即座にロールバック + 原因分析レポート出力。自動再試行なし。
- **D-06:** コードハッシュ方式。キャッシュキー計算に`src/features/`配下の全.pyファイルの内容ハッシュを結合して含める。
- **D-07:** 無効化された古いキャッシュファイルは自動削除。

### Claude's Discretion
- Tier 1/Tier 2の具体的な閾値設定（Tier 2の「低重要度」の定義）
- 各モデルでどの特徴量がTier 1に該当するかの特定
- 監査レポートの出力形式とファイル配置
- OOF logloss/AUC比較の具体的な実装
- キャッシュキー計算の具体的なハッシュ対象ファイルリスト
- ロールバック時の原因分析レポートのフォーマット
- プルーニング適用後のフルBT実行コマンド構成

### Deferred Ideas (OUT OF SCOPE)
None
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| AUDIT-01 | 全モデルのpermutation重要度をOOFデータで計算し、Tier 1 + Tier 2の多段階レポートを出力 | `compute_all_model_importance()` already computes gain+perm for all models. Need to add Tier classification logic and generate structured report. |
| AUDIT-02 | Tier 1ノイズ特徴量をモデル別にFEATURE_COLSから除外し、OOF logloss/AUC比較→フルBT ROI検証 | `validate_noise_removal()` provides logloss/AUC comparison. Per-model FEATURE_COLS edits required. `run_backtest.py --ensemble --calibration-bt --report` for ROI verification. |
| AUDIT-03 | src/features/配下の全.pyファイルのコードハッシュをキャッシュキーに含め、自動キャッシュ無効化+削除 | `compute_cache_key()` at line 37 needs code-hash addition. 22 .py files in src/features/, ~210KB total. Auto-deletion of stale cache files from `data/features/cache/`. |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Feature importance computation (gain+perm) | ML Pipeline | — | OOF data from TrainingPipeline, computed by sklearn+LightGBM |
| Tier 1/2 classification | ML Analysis | — | Pure logic based on importance scores |
| Per-model FEATURE_COLS pruning | Model classes | — | Each model class owns its FEATURE_COLS |
| OOF logloss/AUC safety check | ML Pipeline | — | validate_noise_removal() operates on OOF data |
| Full backtest ROI verification | Backtest Engine | — | BacktestEngine.run() produces ROI |
| Code-hash cache invalidation | Feature Engine | — | FeatureEngine.build_all() owns cache read/write |
| Stale cache auto-deletion | Feature Engine | — | Part of cache lifecycle in is_cache_valid() |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scikit-learn | installed | permutation_importance, log_loss, roc_auc_score | Already used in win_feature_analysis.py |
| LightGBM | installed | gain importance via feature_importance(), pred_contrib for SHAP | Core model framework |
| numpy | installed | Array operations for importance computation | Standard numeric library |
| pandas | installed | DataFrame manipulation for pivot tables | Standard data library |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| hashlib | stdlib | SHA-256 code hash for cache invalidation | compute_cache_key() extension |
| json | stdlib | Metadata serialization for audit reports | Report output |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| sklearn permutation_importance | SHAP permutation | sklearn already integrated, no new dependency needed |
| Manual Tier classification | Boruta algorithm | Boruta is more complex; Tier 1 (Gain=0 AND Perm<=0) is conservative and safe |

**Installation:** No new dependencies needed — all libraries already in the project.

## Architecture Patterns

### System Architecture Diagram

```
[Phase 23 Audit Scripts]
         |
         v
[analyze_feature_importance.py --all-models]
         |
         v
[compute_all_model_importance()]  →  pivot_df (CSV) + metadata (JSON)
         |
         v
[Tier Classification Logic]  ── NEW ──
    ├── Tier 1: Gain=0 AND Perm≤0 → auto-remove per-model
    └── Tier 2: Low importance → report for user decision
         |
         v
[Per-Model FEATURE_COLS Edit]  (7 model classes)
         |
         v
[OOF Safety Check] ─ validate_noise_removal() per model
    ├── PASS → proceed to full BT
    └── FAIL → revert FEATURE_COLS, report degradation
         |
         v
[Full Backtest] ─ run_backtest.py --ensemble --report
    ├── ROI ≥ 84.4% → commit pruned FEATURE_COLS
    └── ROI < 84.4% → rollback + cause analysis report

Separate flow (AUDIT-03):
[src/features/*.py] → code hash → [compute_cache_key()] → cache key includes code
                                                    → stale cache auto-deletion
```

### Recommended Project Structure
```
src/features/
├── feature_engine.py          # MODIFY: compute_cache_key() + is_cache_valid()
├── win_feature_analysis.py    # MODIFY: add classify_feature_tiers()
data/
├── features/cache/            # MODIFY: auto-cleanup of stale files
├── audit/                     # NEW: audit reports directory
│   ├── feature_importance_pivot.csv
│   ├── feature_importance_metadata.json
│   ├── tier_report.json
│   └── noise_removal_validation.json
scripts/
├── analyze_feature_importance.py  # MODIFY: add --tier-report flag
```

### Pattern 1: Tier Classification
**What:** Classify features into Tier 1 (auto-remove) and Tier 2 (flag for user) based on gain and permutation importance.
**When to use:** After compute_all_model_importance() produces per-model importance scores.
**Example:**
```python
# Source: Extension of win_feature_analysis.py
def classify_feature_tiers(
    pivot_df: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    tier2_percentile: float = 10.0,  # bottom 10% = Tier 2
) -> dict[str, dict[str, Any]]:
    """Tier 1: Gain=0 AND Perm<=0. Tier 2: bottom percentile."""
    tiers: dict[str, dict[str, Any]] = {}
    for model_name, model_data in metadata["models"].items():
        gain = model_data["gain"]
        perm = model_data["perm_mean"]
        tier1 = [f for f in gain if gain[f] == 0 and perm.get(f, 0) <= 0]
        # Tier 2: features not in Tier 1 but in bottom percentile
        ...
        tiers[model_name] = {"tier1": tier1, "tier2": tier2}
    return tiers
```

### Pattern 2: Code-Hash Cache Invalidation
**What:** Include source code hash in cache key so that feature module changes invalidate the cache.
**When to use:** In compute_cache_key() for all feature cache operations.
**Example:**
```python
# Source: Extension of feature_engine.py:37
def compute_cache_key(
    input_paths: list[Path],
    date_range: tuple[str, str] | None,
    feature_type: str,
    code_hash: str | None = None,  # NEW parameter
) -> str:
    payload = json.dumps({
        "paths": [str(p) for p in sorted(input_paths)],
        "start": date_range[0] if date_range else "",
        "end": date_range[1] if date_range else "",
        "type": feature_type,
        "code_hash": code_hash or "",  # NEW
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

def compute_code_hash(features_dir: str = "src/features") -> str:
    """Hash all .py files in features directory."""
    h = hashlib.sha256()
    for py_file in sorted(Path(features_dir).glob("*.py")):
        h.update(py_file.read_bytes())
    return h.hexdigest()[:16]
```

### Anti-Patterns to Avoid
- **Global pruning across all models:** Must NOT remove a feature from all models if it's only noise in one. Per-model FEATURE_COLS are independent.
- **Mutating FEATURE_COLS via remove_noise_features():** Deprecated, not thread-safe. Edit the class-level list directly in source code or use `get_filtered_feature_cols()`.
- **Running full BT before OOF safety check:** Always validate with logloss/AUC first; full BT takes ~57 min/year.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Permutation importance | Custom permutation logic | sklearn.inspection.permutation_importance | Already integrated, handles edge cases |
| SHAP values | Custom SHAP computation | LightGBM pred_contrib=True | Native TreeSHAP, no extra dependency |
| Cache key hashing | Custom hash algorithm | hashlib.sha256 | Already used in compute_cache_key() |
| Tier 1 detection | Complex statistical test | Simple Gain=0 AND Perm<=0 | Conservative, no false positives |
| OOF logloss/AUC comparison | Custom validation | validate_noise_removal() | Already implemented, tested |

**Key insight:** Phase 23 built all the infrastructure. Phase 24 is about executing it, adding Tier classification logic, and applying the results.

## Runtime State Inventory

> Not a rename/refactor phase — omitting.

## Common Pitfalls

### Pitfall 1: Permutation Importance Computation Cost
**What goes wrong:** Computing permutation importance for ConformalEVModel (131 features) is very slow.
**Why it happens:** sklearn permutation_importance is O(n_features * n_repeats * n_samples).
**How to avoid:** Use max_samples=5000 (already in compute_permutation_importance) and consider reducing n_repeats for large models. The CLI already supports `--n-repeats`.
**Warning signs:** Script runs >30 min without progress.

### Pitfall 2: Return/E-correction Models Lack Binary Target
**What goes wrong:** Permutation importance requires a target. Return models (win_return, place_return) use regression targets that aren't easily available from the CLI's `_load_features_for_analysis()`.
**Why it happens:** The CLI constructs targets only for hit models (binary). Return and E-correction models are skipped in the target construction.
**How to avoid:** For models without targets, compute gain importance only (already handled by compute_all_model_importance with NaN perm values). Tier 1 for these models uses gain=0 only.
**Warning signs:** Permutation importance shows NaN for return/E-correction models.

### Pitfall 3: Cache Invalidation Does Not Detect Feature Module Changes
**What goes wrong:** After editing `src/features/*.py`, the cached `horse_features.parquet` is still used because compute_cache_key() only hashes source parquet paths and date ranges.
**Why it happens:** Current cache key has no awareness of the code that generated the features.
**How to avoid:** Add code hash of `src/features/*.py` to the cache key computation (D-06).
**Warning signs:** Model trains on stale features after code change.

### Pitfall 4: FEATURE_COLS Edit Breaks Downstream Models
**What goes wrong:** Removing a feature from AbilityModel.FEATURE_COLS that ConformalEVModel.FEATURE_COLS also references. If the feature is removed from the build_all() output, it becomes NaN for ConformalEVModel.
**Why it happens:** Features are shared across models. build_all() computes features for all models, and individual FEATURE_COLS select subsets.
**How to avoid:** Tier 1 removal from a model's FEATURE_COLS only affects that model's training input, NOT build_all() output. build_all() continues to compute all features. Each model's `_prepare_features()` filters by its own FEATURE_COLS. This is safe by design.
**Warning signs:** None — the architecture naturally isolates per-model feature selection.

### Pitfall 5: Stale Cache Files Accumulate
**What goes wrong:** Multiple cache files (`feat_*.parquet`) accumulate in `data/features/cache/` as date ranges change or code updates. Currently 6 files totaling ~63MB.
**Why it happens:** No cleanup mechanism exists. Each new cache key creates a new file without removing old ones.
**How to avoid:** Implement auto-deletion: when cache_key changes, delete files matching old keys before writing new one. Or delete all files in cache/ before writing a new cache (since only the latest cache is useful).
**Warning signs:** Disk usage grows linearly over time.

### Pitfall 6: OOF Safety Check Insufficient for Return Models
**What goes wrong:** `validate_noise_removal()` uses `target_col="kakuteijyuni"` (binary hit/no-hit). Return models predict continuous values (odds), not binary outcomes.
**Why it happens:** The function was designed for hit model validation.
**How to avoid:** For return models, either skip OOF validation (rely on full BT) or implement MAE-based comparison. CONTEXT.md says OOF logloss/AUC for safety check, which applies primarily to hit/classification models.
**Warning signs:** validate_noise_removal() produces meaningless metrics for return models.

## Code Examples

### Current Feature Counts Per Model
```python
# [VERIFIED: codebase runtime check 2026-05-12]
AbilityModel:               80 features
WinTwoStageModel:           38 features
PlaceTwoStageModel.HIT:     45 features
PlaceTwoStageModel.RETURN:  43 features
EVCorrectionModel:          24 features
PlaceEVCorrectionModel:     24 features
ConformalEVModel:          131 features

# Total unique features across all models: 139
# Features used by 2+ models: 123
# Features used by 3+ models: 66
```

### Current compute_cache_key() — Modification Target
```python
# Source: src/features/feature_engine.py:37-52
def compute_cache_key(
    input_paths: list[Path],
    date_range: tuple[str, str] | None,
    feature_type: str,
) -> str:
    """キャッシュキーを計算: 入力パス + 日付範囲 + 特徴量種別 -> SHA-256先頭16文字"""
    payload = json.dumps(
        {
            "paths": [str(p) for p in sorted(input_paths)],
            "start": date_range[0] if date_range else "",
            "end": date_range[1] if date_range else "",
            "type": feature_type,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
```

### Current Cache Files (6 stale files, ~63MB)
```
feat_00927f0176aac090.parquet  20.5MB  2026-05-04
feat_3c1663e6f1aaeb1a.parquet   5.7MB  2026-05-07
feat_6ffc76c95f8cb794.parquet  20.5MB  2026-05-05
feat_99ab6bc8eae465bc.parquet   5.7MB  2026-05-06
feat_a030ef24bc1aba22.parquet   5.6MB  2026-05-04
feat_da40350bb11ea0e6.parquet   5.6MB  2026-05-07
```

### validate_noise_removal() — Existing Safety Check
```python
# Source: src/features/win_feature_analysis.py:273-376
# Returns: dict with original_logloss, new_logloss, original_auc, new_auc
# Uses time-series 80/20 split (look-ahead bias prevention)
# Logs warning if logloss degrades >0.5%
# Retrains model with reduced features and compares
```

### Feature Importance Analysis CLI — Existing
```python
# Source: scripts/analyze_feature_importance.py
# Usage:
#   python scripts/analyze_feature_importance.py --all-models --format both
#   python scripts/analyze_feature_importance.py --model win_hit --surface turf
# Output: feature_importance_report.csv + feature_importance_report.json
# Supports: --n-repeats, --output, --output-json
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| SHAP-only noise detection | Gain + Permutation + SHAP (Phase 23) | Phase 23 | Multi-metric noise detection more robust |
| Global feature removal | Per-model pruning (D-02) | Phase 24 CONTEXT | Feature can be noise in one model but useful in another |
| Timestamp-based cache | Code-hash based cache (D-06) | Phase 24 | Source code changes now properly invalidate cache |
| remove_noise_features() | Direct FEATURE_COLS editing | Phase 24 | Avoid deprecated thread-unsafe mutation |

**Deprecated/outdated:**
- `WinTwoStageModel.remove_noise_features()`: Deprecated with DeprecationWarning. Use `get_filtered_feature_cols()` or direct FEATURE_COLS editing.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | ConformalEVModel uses 131 features — pruning effect highest there | Feature Counts | If many are already zero-gain, actual removable count may be small |
| A2 | OOF data (horse_features.parquet) contains all features needed for permutation importance | AUDIT-01 | If features are missing, audit will show NaN perm values |
| A3 | Tier 1 features (Gain=0 AND Perm<=0) are rare enough that removal won't destabilize models | Tier Classification | If too many features qualify, model capacity may drop significantly |
| A4 | `src/features/` contains 22 .py files totaling ~210KB — hashing is fast (<100ms) | Cache Invalidation | Negligible performance impact |

**If this table is empty:** All claims in this research were verified or cited — no user confirmation needed.

## Open Questions

1. **Return model Tier 1 definition**
   - What we know: Return models (win_return, place_return) lack permutation targets in the current CLI. Gain importance is available.
   - What's unclear: Should Tier 1 for return models be "Gain=0 only" or should we construct regression targets for permutation importance?
   - Recommendation: Use Gain=0 only for return/E-correction models. Gain=0 is already a strong signal of zero contribution.

2. **Tier 2 threshold**
   - What we know: CONTEXT.md leaves Tier 2 threshold to Claude's discretion.
   - What's unclear: Percentile-based (bottom 10%) vs absolute threshold (perm < some_value AND gain < some_value).
   - Recommendation: Percentile-based (bottom 10%) is more adaptive across models with different feature counts.

3. **Cache deletion strategy**
   - What we know: 6 stale cache files exist. CONTEXT.md says auto-delete stale files.
   - What's unclear: Delete-all-before-write vs delete-only-stale-key.
   - Recommendation: Delete all `feat_*.parquet` in cache/ when writing a new cache. Only the latest cache is useful (earlier date ranges produce different keys and won't match anyway).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.11 (mise) | All | ✓ | 3.11.x | — |
| PostgreSQL | OOF data generation | ✓ | localhost:5432 | — |
| LightGBM | Model training | ✓ | installed | — |
| scikit-learn | permutation_importance | ✓ | installed | — |
| src/features/*.py (22 files) | Code hash computation | ✓ | — | — |
| data/features/cache/ | Cache storage | ✓ | 6 files | — |
| data/models/*.lgb | Model files for audit | ✗ | — | Need training run first |

**Missing dependencies with no fallback:**
- Model files (`.lgb`): The audit requires trained models. These are generated by `run_train.py`. The `data/models/` directory currently only has `place_ability_turf.joblib`. A training run or backtest must be executed before the audit can run on real data.

**Missing dependencies with fallback:**
- None — all other dependencies are in place.

## Validation Architecture

> nyquist_validation is explicitly `false` in `.planning/config.json`. Skipping this section.

## Security Domain

> This is a data analysis and code modification phase with no security-sensitive operations. Omitting security domain section.

## Sources

### Primary (HIGH confidence)
- `src/features/win_feature_analysis.py` — Full file read, all 377 lines. Contains compute_all_model_importance(), identify_noise_features(), validate_noise_removal().
- `src/features/feature_engine.py` — Full file read, 517 lines. Contains compute_cache_key(), is_cache_valid(), build_all().
- `src/models/stage1_ability_model.py` — FEATURE_COLS at line 28, 80 features.
- `src/models/two_stage_return_model.py` — FEATURE_COLS for Win (line 48, 38), Place HIT (line 289, 45), Place RETURN (line 345, 43).
- `src/models/ev_correction_model.py` — FEATURE_COLS for Win EV (line 151, 24), Place EV (line 405, 24).
- `src/models/conformal_ev_model.py` — FEATURE_COLS at line 81, 131 features.
- `scripts/analyze_feature_importance.py` — Full CLI, 478 lines.
- `src/pipelines/training_pipeline.py` — OOF flow at lines 440-570.
- `src/backtest/validation_report.py` — Full file read, ROI PASS/FAIL logic.
- `tests/test_win_feature_analysis.py` — Full test file, 316 lines.
- [VERIFIED: runtime] Feature counts via Python import and count.

### Secondary (MEDIUM confidence)
- `data/features/cache/` — 6 stale cache files, ~63MB total. Verified by directory listing.
- `.planning/ROADMAP.md` — Phase 24 success criteria.
- `.planning/config.json` — nyquist_validation: false confirmed.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in project, verified by codebase reading
- Architecture: HIGH — all 7 model classes and their FEATURE_COLS read in full, cross-model analysis verified by runtime check
- Pitfalls: HIGH — identified from direct code reading and runtime verification
- Cache mechanism: HIGH — full feature_engine.py read, cache directory verified

**Research date:** 2026-05-12
**Valid until:** 2026-06-12 (stable codebase, no fast-moving dependencies)
