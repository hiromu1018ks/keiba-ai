# Phase 9: Win Reporting - Research

**Researched:** 2026-05-04
**Domain:** Backtest reporting, diagnostic analytics, odds band analysis
**Confidence:** HIGH

## Summary

Phase 9 extends the existing backtest reporting system (report.py, engine.py, run_backtest.py) to produce win-specific bet history, ROI diagnostics, and odds-band analysis. The architecture is already in place from place/wide reporting; this phase adds win-specific fields and analysis sections while preserving backward compatibility for place/wide modes.

The primary change surface is three files: (1) `src/backtest/engine.py` lines 758-812 where bet_history dicts are constructed -- win-specific fields from race_predictor's DataFrame need to be extracted here; (2) `src/backtest/report.py` where BacktestReportGenerator needs betting_target-aware condition branching in `_compute_condition_stats()` and `generate()`; (3) `scripts/run_backtest.py` where `display_single_year_result()` needs win-specific output formatting.

The existing `_compute_condition_stats()` already implements popularity bands (1-3, 4-6, 7+) and EV bands (<1.0, 1.0-1.2, 1.2-1.5, 1.5+) using a reusable `_band_stats()` helper. Adding odds multiplier bands follows the identical pattern. The regime breakdown (D-07) requires adding `regime` to the bet_history dict in engine.py since it is currently only logged to diag_logger, not stored in bet_history.

**Primary recommendation:** Extend the existing reporting pipeline through three targeted injection points, using the established `_band_stats()` helper pattern for all new band analyses.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** 人気順位バンド(1-3/4-6/7+) + オッズ倍率バンドの2次元分析を実装
- **D-02:** 人気順位バンドは既存 `_compute_condition_stats()` の popularity bands (1-3, 4-6, 7+) をそのまま利用
- **D-03:** オッズ倍率バンドの区分はClaude裁量で最適化。JRA控除率25%と実用的な投資リスク区分を考慮
- **D-04:** 既存 `BacktestReportGenerator` を拡張し、betting_targetで条件分岐。新規クラスは作らない
- **D-05:** win指定時は単勝専用セクションを出力。place/wide時は既存ロジックを維持
- **D-06:** 2層出力: (1) 人間向け HTML + CLI標準出力、(2) AI分析向け 構造化JSON
- **D-07:** 包括的診断: 月別推移, 表面x距離別, Regime別, EVバンド別, オッズバンド別
- **D-08:** AI分析用JSONに改善点自動特定データを含める
- **D-09:** bet_historyに win_selection_ev, win_selection_edge, win_selection_prob, win_gate_score, conformal_confidence_score, tanoddslow, kakuteijyuni, popularity を記録
- **D-10:** これらのフィールドによりスコア成分のROI寄与分析を可能にする

### Claude's Discretion
- オッズ倍率バンドの具体的な区分
- AI分析用JSONのスキーマ詳細
- HTMLレポートの視覚デザイン
- CLI標準出力のフォーマット
- bet_historyフィールドのengine.pyでの取得方法
- MultiYearReportGeneratorへの対応範囲

### Deferred Ideas (OUT OF SCOPE)
None
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| RPT-01 | バックテスト結果のベット履歴に単勝ベットの馬番・オッズ・EV・結果を記録できる | engine.py bet_history拡張パターン (lines 758-812), result_dfからwin_selection_*列を取得する既存パターン (lines 799-808のp_place_predパターンを流用) |
| RPT-02 | 単勝ROI・回収率・的中率・ベット数の集計診断を出力できる | report.py _compute_monthly_stats()パターン, _compute_condition_stats()パターン, 新規regime/stats計算を同じパターンで追加 |
| RPT-03 | オッズバンド別(人気・中穴・大穴)のROI内訳を分析・表示できる | report.py _band_stats() helper (lines 189-223) にodds_multiplier bandを追加, 既存popularity_bands (1-3, 4-6, 7+) をそのまま利用 |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Win bet history field extraction | Backend (engine.py) | -- | Data lives in result_df from race_predictor; engine.py already extracts fields from this DataFrame |
| ROI diagnostics aggregation | Backend (report.py) | -- | Statistical computation belongs in report generator, same tier as existing _compute_condition_stats() |
| Odds band analysis | Backend (report.py) | -- | Band bucketization and groupby logic follows existing _band_stats() pattern |
| CLI display formatting | CLI (run_backtest.py) | -- | Terminal output formatting; existing display_single_year_result() pattern |
| HTML report rendering | Frontend (templates) | Backend (report.py) | Jinja2 templates consume data dicts; report.py prepares the data |
| AI diagnostic JSON | Backend (report.py) | -- | JSON schema construction and improvement-point identification |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | in-project | DataFrame operations for band analysis | Already used throughout report.py for groupby/aggregation |
| jinja2 | in-project | HTML template rendering | Already used in BacktestReportGenerator.generate() |
| json (stdlib) | in-project | AI diagnostic JSON output | Already used in save_bet_history() |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| dataclasses | in-project | BacktestResult, Bet dataclasses | Extending BacktestResult if needed for win-specific fields |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Extending BacktestReportGenerator | New WinReportGenerator class | D-04 explicitly locks this: extend existing class, no new class |

**Installation:**
No new packages needed. All dependencies already in project.

## Architecture Patterns

### System Architecture Diagram

```
race_predictor.predict()
    |
    v
result_df (DataFrame with win_selection_ev, win_selection_edge,
           win_selection_prob, win_gate_score, conformal_confidence_score,
           tanoddslow, kakuteijyuni, popularity_rank, regime)
    |
    +---> engine.py bet_history construction (lines 758-812)
    |         |
    |         v  [NEW: extract win_selection_*, conformal_confidence_score,
    |             tanoddslow, regime from result_df/horse_rows]
    |         |
    |         v  bet_history list[dict] (enriched with win fields)
    |
    +---> report.py BacktestReportGenerator.generate()
              |
              +---> _derive_fields() [existing: race_date, profit, is_win]
              +---> _compute_monthly_stats() [existing pattern]
              +---> _compute_condition_stats() [EXTEND: add regime_bands,
              |     odds_multiplier_bands via _band_stats() helper]
              +---> _compute_regime_stats() [NEW: same pattern as monthly]
              +---> Jinja2 template rendering [EXTEND: win-specific sections]
              |
              +---> save_bet_history() [existing JSON]
              +---> save_ai_diagnostics() [NEW: structured diagnostic JSON]

CLI:
run_backtest.py display_single_year_result()
    |
    +---> [EXTEND: win-specific KPI output (win rate, avg odds, edge stats)]

HTML:
templates/report.html [EXTEND: regime section, odds multiplier band section]
templates/multi_year_report.html [EXTEND: win-specific columns]
```

### Recommended Project Structure
```
src/backtest/
    engine.py              # [MODIFY] bet_history dict: add win fields + regime
    report.py              # [MODIFY] BacktestReportGenerator: betting_target branch,
                           #         _compute_condition_stats() extension,
                           #         _compute_regime_stats() new method,
                           #         save_ai_diagnostics() new method
    templates/
        report.html        # [MODIFY] win-specific sections
        multi_year_report.html  # [MODIFY] win-specific columns

scripts/
    run_backtest.py        # [MODIFY] display_single_year_result() win output
```

### Pattern 1: Band Statistics Helper (Established)
**What:** Reusable `_band_stats()` function in `_compute_condition_stats()` that takes a key function and band order list, returning standardized band analysis dicts.
**When to use:** Every band analysis (popularity, EV, odds multiplier, regime) follows this exact pattern.
**Example:**
```python
# Source: src/backtest/report.py lines 189-248
def _band_stats(
    bets_list: list[dict[str, Any]],
    key_fn: Any,
    band_order: list[str],
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, float]] = defaultdict(
        lambda: {"bets": 0, "wins": 0, "total_payout": 0.0, "total_stake": 0.0}
    )
    for b in bets_list:
        band = key_fn(b)
        groups[band]["bets"] += 1
        groups[band]["total_stake"] += b["stake"]
        if b["result"] > 0:
            groups[band]["wins"] += 1
            groups[band]["total_payout"] += b["result"]
    result = []
    for band in band_order:
        if band not in groups:
            continue
        g = groups[band]
        n = g["bets"]
        result.append({
            "band": band,
            "bets": n,
            "wins": int(g["wins"]),
            "win_rate": g["wins"] / n if n > 0 else 0.0,
            "avg_payout": g["total_payout"] / g["wins"] if g["wins"] > 0 else 0.0,
            "roi": g["total_payout"] / g["total_stake"] if g["total_stake"] > 0 else 0.0,
        })
    return result
```

### Pattern 2: bet_history Field Extraction (Established)
**What:** Extracting values from `result_df` / `horse_rows` DataFrame into bet_history dict.
**When to use:** Adding any new field to bet_history from the prediction pipeline output.
**Example:**
```python
# Source: src/backtest/engine.py lines 799-808
"p_place_pred": (
    float(horse_rows.iloc[0].get("p_place_pred", 0))
    if not horse_rows.empty
    else 0.0
),
```

### Pattern 3: Monthly Stats Aggregation (Established)
**What:** groupby aggregation pattern using defaultdict, computing bets/wins/stake/total_return per time period.
**When to use:** regime stats, monthly stats, any time-based or category-based aggregation.
**Example:**
```python
# Source: src/backtest/report.py lines 106-135
monthly: dict[str, dict[str, float]] = defaultdict(
    lambda: {"bets": 0, "wins": 0, "stake": 0.0, "total_return": 0.0}
)
for b in bets:
    month = b["race_date"][:7]
    monthly[month]["bets"] += 1
    monthly[month]["stake"] += b["stake"]
    if b["result"] > 0:
        monthly[month]["wins"] += 1
        monthly[month]["total_return"] += b["result"]
```

### Anti-Patterns to Avoid
- **Creating a new WinReportGenerator class:** D-04 explicitly locks extending the existing class. Do not create new classes.
- **Duplicating band analysis logic:** The `_band_stats()` helper exists precisely to avoid duplication. Every new band analysis must use it.
- **Accessing win-specific columns without checking betting_target:** report.py must guard all win-specific output with `if betting_target == "win"` checks (D-05).
- **Modifying place/wide report behavior:** All changes must be additive or guarded by betting_target conditionals. Existing place/wide output must remain unchanged.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Band bucketization | Custom bucketize logic per band | `_band_stats()` helper with key_fn lambda | Already handles all edge cases (empty groups, zero division, ordering) |
| Monthly/grouped aggregation | Custom groupby per category | `_compute_monthly_stats()` pattern with defaultdict | Already handles zero-division and empty inputs |
| JSON serialization | Custom JSON builder | `json.dumps()` with `ensure_ascii=False` | Already used in `save_bet_history()` |
| HTML generation | String concatenation | Jinja2 templates | Already established pattern in templates/ directory |

**Key insight:** The reporting pipeline has excellent internal reuse through the `_band_stats()` helper. Adding new band analyses (regime, odds multiplier) is a ~10-line addition each, not a new module.

## Common Pitfalls

### Pitfall 1: Missing regime in bet_history
**What goes wrong:** D-07 requires regime-level breakdown, but `regime` is currently only passed to `diag_logger.log_race()` (engine.py:676), never stored in the bet_history dict. Without it, the report generator cannot compute per-regime ROI.
**Why it happens:** regime was added for diagnostic logging (engine.py:584, 676) but never propagated to the bet_history construction path.
**How to avoid:** Add `"regime": str(regime)` to the bet_history dict at engine.py line 758-812. The `regime` variable is already in scope at that point (computed at line 539-542).
**Warning signs:** KeyError when trying to access `b["regime"]` in `_compute_condition_stats()`.

### Pitfall 2: win_selection_* columns missing from result_df
**What goes wrong:** The win_selection_* columns are only guaranteed to exist when the WinSelectionGate is trained and betting_target == "win". In place mode or with an untrained gate, these columns may be absent.
**Why it happens:** `ensure_win_selection_columns()` (win_selection_gate.py:33-54) creates these columns with fallback logic, but they may contain NaN or not exist if the gate model is not trained.
**How to avoid:** Use the same `horse_rows.iloc[0].get("col_name", default)` pattern as p_place_pred (engine.py:799), with a fallback to 0.0 for numeric fields.
**Warning signs:** KeyError or AttributeError when accessing win_selection_ev on horse_rows.

### Pitfall 3: Odds multiplier band definitions for JRA
**What goes wrong:** Using arbitrary band boundaries that don't align with JRA's 25% takeout rate and typical odds distributions.
**Why it happens:** JRA odds have a very different distribution than US/UK racing. The 25% takeout means the fair-value cutoff (odds where EV=1.0 given typical model edge) differs from other markets.
**How to avoid:** Use JRA-informed band boundaries (see Odds Multiplier Band Recommendations below). The key insight is that with 25% takeout, a horse at 1.0x implied probability needs odds > ~3.3 to break even on win bets (3.3 * 0.303 = 1.0), so bands should reflect practical betting tiers.
**Warning signs:** Bands with zero bets or all bets concentrated in one band.

### Pitfall 4: HTML template breaking for place/wide mode
**What goes wrong:** Adding win-specific HTML sections that reference win-only data keys (like `condition_stats.regime_bands`), causing Jinja2 UndefinedError when running place/wide backtest.
**Why it happens:** The template is shared across all betting_target modes.
**How to avoid:** Wrap all new sections in `{% if condition_stats.regime_bands %}` guards. The existing template already uses this pattern (lines 184, 206).
**Warning signs:** Template rendering error when running `--betting-target place`.

### Pitfall 5: bet_history field name collision
**What goes wrong:** The existing bet_history already has `"odds"` (engine.py:764, which is bet.odds = the odds used for bet decision) and `"popularity"` (engine.py:773, from popularity_rank). Adding `"tanoddslow"` and `"popularity"` again would overwrite existing values.
**Why it happens:** The bet.odds field stores different values for win vs place mode. For win, it stores tanoddslow; for place, fukuoddslow. The existing "odds" key in bet_history already captures this correctly.
**How to avoid:** Only add truly new fields (win_selection_ev, win_selection_edge, etc.). The D-09 "tanoddslow" field should be added as a separate key to capture the confirmed final odds distinct from the decision-time odds.
**Warning signs:** Double-checking that "popularity" is already populated (line 773) so don't re-add it.

## Code Examples

### Adding win fields to bet_history (engine.py)
```python
# Source: established pattern from engine.py:799-808
# Add after the existing "top3_finishers": _top3, line in bet_history dict

# --- Win-specific fields (D-09, RPT-01) ---
"win_selection_ev": (
    float(horse_rows.iloc[0].get("win_selection_ev", 0.0))
    if not horse_rows.empty
    else 0.0
),
"win_selection_edge": (
    float(horse_rows.iloc[0].get("win_selection_edge", 0.0))
    if not horse_rows.empty
    else 0.0
),
"win_selection_prob": (
    float(horse_rows.iloc[0].get("win_selection_prob", 0.0))
    if not horse_rows.empty
    else 0.0
),
"win_gate_score": (
    float(horse_rows.iloc[0].get("win_gate_score", float("nan")))
    if not horse_rows.empty
    else float("nan")
),
"conformal_confidence_score": (
    float(horse_rows.iloc[0].get("conformal_confidence_score", 0.0))
    if not horse_rows.empty
    else 0.0
),
"tanoddslow": (
    float(horse_rows.iloc[0].get("tanoddslow", 0.0))
    if not horse_rows.empty
    else 0.0
),
"regime": str(regime),  # D-07: regime is in scope from line 539-542
```

### Adding odds multiplier bands to _compute_condition_stats()
```python
# Source: pattern from report.py:230-242 (ev_bands)
odds_multiplier_bands = _band_stats(
    bets,
    lambda b: (
        "1.0-3.0" if b.get("tanoddslow", 0) < 3.0
        else "3.0-5.0" if b.get("tanoddslow", 0) < 5.0
        else "5.0-10.0" if b.get("tanoddslow", 0) < 10.0
        else "10.0-30.0" if b.get("tanoddslow", 0) < 30.0
        else "30.0+"
    ),
    ["1.0-3.0", "3.0-5.0", "5.0-10.0", "10.0-30.0", "30.0+"],
)
```

### Adding regime stats (new method, follows _compute_monthly_stats pattern)
```python
def _compute_regime_stats(self, bets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Regime別集計: ROI, 的中率, ベット数"""
    if not bets:
        return []
    regime_data: dict[str, dict[str, float]] = defaultdict(
        lambda: {"bets": 0, "wins": 0, "stake": 0.0, "total_return": 0.0}
    )
    for b in bets:
        regime = b.get("regime", "unknown")
        regime_data[regime]["bets"] += 1
        regime_data[regime]["stake"] += b["stake"]
        if b["result"] > 0:
            regime_data[regime]["wins"] += 1
            regime_data[regime]["total_return"] += b["result"]
    result = []
    for regime in ["aggressive", "conservative", "collapsed"]:
        if regime not in regime_data:
            continue
        s = regime_data[regime]
        n = s["bets"]
        result.append({
            "regime": regime,
            "bets": n,
            "wins": int(s["wins"]),
            "win_rate": s["wins"] / n if n > 0 else 0.0,
            "roi": s["total_return"] / s["stake"] if s["stake"] > 0 else 0.0,
        })
    return result
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Separate report classes per bet type | Single BacktestReportGenerator with betting_target branching | Phase 9 (D-04) | All reporting logic in one class; reduces duplication |
| Popularity bands only (1-3/4-6/7+) | Popularity + odds multiplier 2D bands | Phase 9 (D-01) | Multi-dimensional revenue structure analysis |
| Diagnostic logging to file only | Structured diagnostic JSON for AI analysis | Phase 9 (D-06) | Enables automated improvement identification |

**Deprecated/outdated:**
- None for this phase. All existing patterns remain valid.

## Odds Multiplier Band Recommendations

Based on JRA's 25% takeout rate and practical betting tiers [CITED: grokipedia.com/page/Horse_racing_in_Japan]:

| Band | Range | Rationale |
|------|-------|-----------|
| Favorite (人気) | 1.0 - 3.0 | High-probability, low-return. JRA favorites typically 1.2-3.0. Break-even at ~3.3 with 25% takeout. |
| Middle (中穴) | 3.0 - 10.0 | Medium-probability, medium-return. The "value zone" where model edge matters most. |
| Longshot (大穴) | 10.0 - 30.0 | Low-probability, high-return. Model edge critical for profitability here. |
| Extreme (超穴) | 30.0+ | Very low probability. Typically few bets; useful for diagnostic purposes. |

These four bands provide meaningful segmentation for JRA win betting analysis. The 3.0 and 10.0 boundaries are natural breakpoints in JRA odds distributions.

**Confidence:** MEDIUM - Based on JRA takeout rate documentation and domain knowledge of JRA odds distributions. Should be validated against actual backtest data distribution. [ASSUMED: specific boundary values; the general framework (favorite/middle/longshot) is standard in horse racing analytics]

## AI Diagnostic JSON Schema

The AI diagnostic JSON (D-06, D-08) should contain:

```python
{
    "meta": {
        "betting_target": "win",
        "generated_at": "ISO-8601",
        "commit": "git hash"
    },
    "summary": {
        "roi": float,
        "win_rate": float,
        "total_bets": int,
        "total_stake": float,
        "total_return": float,
        "avg_edge": float,
        "profit": float
    },
    "monthly_trend": [...],  # from _compute_monthly_stats()
    "regime_breakdown": [...],  # from _compute_regime_stats()
    "odds_multiplier_bands": [...],  # from _band_stats()
    "popularity_bands": [...],  # existing
    "ev_bands": [...],  # existing
    "surface_distance": [...],  # existing
    "highlights": {
        "best_band": {"name": str, "roi": float, "bets": int},
        "worst_band": {"name": str, "roi": float, "bets": int},
        "monthly_trend": "improving|declining|stable",
        "regime_best": str,
        "overperforming_conditions": [...],
        "underperforming_conditions": [...]
    }
}
```

The `highlights` section enables automated improvement identification by comparing each band's ROI against the overall baseline ROI.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Odds multiplier band boundaries (1.0-3.0, 3.0-5.0, 5.0-10.0, 10.0-30.0, 30.0+) are appropriate for JRA | Odds Multiplier Band Recommendations | Bands may be too granular or too coarse for actual data distribution |
| A2 | regime variable is in scope at engine.py line 758-812 and can be added to bet_history | Code Examples | Variable scope issue would require passing regime through a different mechanism |
| A3 | conformal_confidence_score is available in result_df after race_predictor.predict() returns | Code Examples | Column may not exist if confidence estimator is not configured |
| A4 | MultiYearReportGenerator uses _single_gen._compute_condition_stats() internally, so extending that method automatically propagates to multi-year reports | Architecture Patterns | May need separate handling in MultiYearReportGenerator.generate() |

## Open Questions

1. **Should odds multiplier bands be computed from `tanoddslow` (final odds) or `bet.odds` (decision odds)?**
   - What we know: Both are available. `bet.odds` is the decision-time odds; `tanoddslow` is the confirmed final odds.
   - Recommendation: Use `tanoddslow` for retrospective analysis (which odds category produced the ROI) since it reflects the actual payout multiplier. The decision-time odds (`bet.odds`) may differ if odds shifted between decision and race start.

2. **Should the regime field be added to bet_history for ALL betting_target modes or only win?**
   - What we know: regime is computed for all modes (engine.py:539-542). Adding it universally would benefit place/wide analysis too.
   - Recommendation: Add universally. It costs nothing and benefits future place/wide diagnostic improvements.

3. **How should the AI diagnostic JSON handle edge cases (e.g., only 1 bet in a band, zero bets in a month)?**
   - What we know: The existing `_band_stats()` skips empty bands. The highlights section needs to handle "no data" gracefully.
   - Recommendation: Skip empty bands in highlights. Require minimum 5 bets for a band to be highlighted (statistical significance).

## Environment Availability

Step 2.6: SKIPPED (no external dependencies identified -- all changes are to in-project Python code using existing libraries)

## Sources

### Primary (HIGH confidence)
- Codebase analysis: src/backtest/report.py (full file read)
- Codebase analysis: src/backtest/engine.py (lines 46-99, 530-879)
- Codebase analysis: src/backtest/race_predictor.py (lines 40-222, 408-470)
- Codebase analysis: src/models/win_selection_gate.py (lines 25-54, 160-239, 982-1001)
- Codebase analysis: scripts/run_backtest.py (lines 80-100, 215-370, 480-542)
- Codebase analysis: src/backtest/templates/report.html (full file)
- Codebase analysis: src/backtest/templates/multi_year_report.html (full file)
- Codebase analysis: tests/test_backtest_report.py (full file)
- Codebase analysis: tests/test_multi_year_report.py (full file)

### Secondary (MEDIUM confidence)
- [grokipedia.com/page/Horse_racing_in_Japan] - JRA 25% takeout rate confirmation [VERIFIED: web search]
- [japanracing.jp/en/jpn-racing/guide/pdf/goracing_en_04.pdf] - JRA official odds explanation [VERIFIED: web search]

### Tertiary (LOW confidence)
- Odds multiplier band boundary values (1.0-3.0, 3.0-10.0, 10.0-30.0, 30.0+) - based on domain knowledge, not verified against actual JRA data distribution

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no new libraries, all existing patterns
- Architecture: HIGH - extension points clearly identified in codebase
- Pitfalls: HIGH - all pitfalls verified against actual code
- Odds band boundaries: MEDIUM - domain knowledge, needs data validation

**Research date:** 2026-05-04
**Valid until:** 2026-06-04 (stable codebase, no external API dependencies)
