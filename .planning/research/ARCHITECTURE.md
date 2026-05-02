# Architecture Patterns: Win Model Improvement

**Domain:** Horse racing prediction system -- win (単勝) model optimization
**Researched:** 2026-05-02

## Recommended Architecture

The win model improvement should follow a **targeted enhancement** strategy within the existing 2-stage decomposition, rather than a wholesale replacement. The core architecture (P(hit) x E(odds|hit)) is theoretically sound. The improvement path focuses on three layers: (1) better probability estimation for the win-specific case, (2) Benter-style fundamental+market combination applied to win probabilities, and (3) win-specific betting strategy with proper Kelly sizing.

### Current State Analysis

The existing system has a **critical asymmetry**: the Place model pipeline is mature (Benter combination, isotonic calibration, temperature scaling, PlaceSelectionGate), while the Win model pipeline is comparatively bare. Specifically:

```
PLACE pipeline (mature):
  AbilityModel -> PlaceAbilityModel -> PlaceTwoStageModel
    -> PlaceEVCorrectionModel -> BenterCombination -> IsotonicRegression
    -> TemperatureScaling -> RobustConfidenceEstimator -> PlaceSelectionGate
    -> select_bets (gate-based, regime-adaptive)

WIN pipeline (minimal):
  AbilityModel -> WinTwoStageModel -> EVCorrectionModel
    -> (no Benter for win)
    -> (no calibration for win)
    -> (no gate for win)
    -> WinStrategy.generate (simple threshold)
```

The Win path gets the 2-stage model and EV correction, but lacks every downstream refinement that Place receives. This is the primary architectural gap to close.

## Target Architecture

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                     Shared Model Layer (unchanged)                       │
│  MarketModel -> AbilityModel (OOF) -> PlaceAbilityModel                 │
├──────────────────────────────────────────────────────────────────────────┤
│                     Win-Specific Enhancement Layer                       │
│                                                                          │
│  ┌─────────────────┐    ┌──────────────────────┐    ┌────────────────┐  │
│  │ WinTwoStageModel │───>│ WinEVCorrectionModel │───>│ WinBenterGate  │  │
│  │ (improved feats) │    │ (existing, improved)  │    │ (NEW)          │  │
│  └─────────────────┘    └──────────────────────┘    └───────┬────────┘  │
│                                                              │           │
│  ┌──────────────────────────────┐    ┌────────────────────┐  │           │
│  │ WinCalibrationPipeline (NEW) │<───┘                    │  │           │
│  │  -> IsotonicRegression       │                         │  │           │
│  │  -> TemperatureScaling       │                         │  │           │
│  └──────────┬───────────────────┘                         │  │           │
│             │                                              │  │           │
│  ┌──────────▼───────────────────┐    ┌────────────────────┘  │           │
│  │ WinConfidenceEstimator (NEW) │<───┘                      │           │
│  │  -> EV lower bound for win   │                           │           │
│  └──────────┬───────────────────┘                           │           │
│             │                                               │           │
│  ┌──────────▼───────────────────┐                           │           │
│  │ WinSelectionGate (NEW)       │<── WinBenterGate output   │           │
│  │  -> learned yes/no filter    │                           │           │
│  └──────────┬───────────────────┘                           │           │
│             │                                               │           │
├─────────────┼───────────────────────────────────────────────────────────┤
│  Betting Layer                                                          │
│  ┌──────────▼───────────────────┐                                       │
│  │ WinStrategy (enhanced)        │                                       │
│  │  -> Kelly stake sizing        │                                       │
│  │  -> Regime-adaptive params    │                                       │
│  │  -> Pool-size-aware capping   │                                       │
│  └──────────────────────────────┘                                       │
└──────────────────────────────────────────────────────────────────────────┘
```

### Component Boundaries

| Component | Responsibility | Communicates With | Status |
|-----------|----------------|-------------------|--------|
| WinTwoStageModel | P(win) x E(odds\|win) estimation with improved feature set | Upstream: AbilityModel, MarketModel. Downstream: WinEVCorrectionModel | Exists, enhance |
| WinEVCorrectionModel | P/E decomposition correction for win probabilities | Upstream: WinTwoStageModel. Downstream: WinBenterGate | Exists (EVCorrectionModel), minor changes |
| WinBenterGate | Combine fundamental P(win) with market-implied P(win) via logit-space weighting; also acts as the selection gate | Upstream: WinEVCorrectionModel + market odds. Downstream: WinStrategy | **NEW** |
| WinCalibrationPipeline | Isotonic regression + temperature scaling for win probabilities | Upstream: WinBenterGate. Downstream: WinConfidenceEstimator | **NEW** |
| WinConfidenceEstimator | Confidence intervals on EV_win for conservative bet sizing | Upstream: WinCalibrationPipeline. Downstream: WinStrategy | **NEW** |
| WinSelectionGate | Learned binary gate: should this horse receive a win bet? | Upstream: WinConfidenceEstimator. Downstream: WinStrategy | **NEW** |
| WinStrategy (enhanced) | Kelly-based stake sizing, regime-adaptive thresholds, pool-size awareness | Upstream: WinSelectionGate, RegimeDetector, DDController. Downstream: BacktestEngine | Exists, enhance |

### Data Flow

```text
1. Shared Layer (existing, unchanged):
   Parquet -> FeatureEngine -> AbilityModel (OOF) -> p_ability_win
                                       -> PlaceAbilityModel -> p_ability_place
                                       -> MarketModel -> signed_log_error_win, abs_log_error_win

2. Win-Specific Model Chain (enhanced):
   p_ability_win + market_errors + race_features
     -> WinTwoStageModel (improved FEATURE_COLS)
       -> p_win_pred, e_return_win_pred, ev_win

   p_win_pred + e_return_win_pred + interaction_features
     -> EVCorrectionModel (P-correction + E-correction)
       -> p_win_corrected, e_return_win_corrected, ev_win_corrected

   p_win_corrected + 1/tanodds (market implied)
     -> WinBenterGate (logit combination)
       -> p_win_combined (unbiased probability estimate)

   p_win_combined + validation actuals
     -> IsotonicRegression + TemperatureScaling
       -> p_win_calibrated

   p_win_calibrated + variance features
     -> WinConfidenceEstimator
       -> EV_lower_win (conservative EV bound)

   EV_lower_win + horse_features + race_features
     -> WinSelectionGate (learned binary filter)
       -> pass/reject decision

3. Betting Decision:
   For each horse that passes WinSelectionGate:
     edge = p_win_calibrated * tanodds - 1.0
     stake = kelly(edge, tanodds, bankroll) * fractional_kelly
     stake = min(stake, pool_size_limit)
     if edge >= regime_params.win_edge_threshold:
         emit Bet(bet_type=WIN)
```

## Key Architectural Decisions

### Decision 1: WinBenterGate is a Single Combined Component

**What:** Instead of separate BenterCombination + PlaceSelectionGate (as in Place), the Win path uses a single `WinBenterGate` that combines fundamental+market probabilities AND makes the selection decision.

**Why:** Win betting has lower frequency but higher variance than place. The Place pipeline separates combination (BenterCombination) from selection (PlaceSelectionGate) because place candidates are numerous and need nuanced ranking. Win candidates are sparse (typically 0-2 per race), so a unified component reduces complexity without losing expressiveness. The Benter-combined probability is the selection signal.

**When to split:** If win betting volume grows to justify a separate selection gate (e.g., when betting multiple horses per race becomes viable), refactor WinBenterGate into separate WinBenterCombination + WinSelectionGate.

### Decision 2: Win-Specific Benter Combination (Not Reusing Place Benter)

**What:** The existing `BenterCombination` is fitted on place data (`p_place_pred` vs `p_market_place`). A new fit is needed for win (`p_win_corrected` vs `1/tanodds`).

**Why:** Benter (1994) emphasizes that the combined model must be fitted for each bet type independently. The fundamental model's bias characteristics differ for win vs place. Benter's Tables 3-4 show that fundamental model bias direction depends on whether p_model > p_market or vice versa, and this relationship differs for win probabilities (spikier, higher variance) vs place probabilities (smoother, lower variance).

**Implementation:** Reuse the `BenterCombination` class but fit new alpha/beta/gamma parameters on win validation data. The key data: `_val_p_win` (validation P(win) predictions), `_val_tanodds` (validation win odds), `_val_y_win` (actual win=1/loss=0).

### Decision 3: Race-Level Normalization to Sum=1

**What:** Win probabilities within a race must sum to 1.0 (exactly one horse wins). This is a constraint that place probabilities do not have (multiple horses can place).

**Why:** The existing `_normalize_probability_by_race()` with `target_sum=1.0` is already used in `EVCorrectionModel` for P(win). This must be preserved and extended to the Benter combination step. After combining fundamental + market probabilities, re-normalize within each race so probabilities sum to 1.0.

**Critical:** Benter's original paper uses the multinomial logit formulation which inherently produces probabilities summing to 1. Our LightGBM binary approach does not have this property, making explicit normalization essential at every step.

### Decision 4: Feature Enhancement in WinTwoStageModel

**What:** Add win-specific features to `WinTwoStageModel.FEATURE_COLS`.

**Why:** The current `FEATURE_COLS` for the Win hit model (27 features) is a subset of the Place hit model (45+ features). Missing features that matter for win prediction:
- `norm_finish_logit_avg`, `harontimel5_zscore` (raw ability metrics, not just rank)
- `jockey_wr_overall`, `trainer_wr_overall`, `jt_combo_place_rate` (human factors)
- `race_mean_fuku_odds`, `odds_gap_fav12` (race context)
- `form_trend`, `form_consistency` (recent form signals)
- `blood_surface_wr`, `blood_distance_wr` (pedigree signals)

However, feature addition must be careful: adding too many features to a model with sparse positive labels (~8% win rate for a 12-horse field) risks overfitting. The feature set should be expanded selectively based on feature importance analysis.

### Decision 5: Win-Specific Calibration Pipeline

**What:** Add IsotonicRegression + TemperatureScaling fitted on win predictions, mirroring the Place pipeline's calibration.

**Why:** The current Place pipeline benefits from calibration (though Isotonic was disabled in v5.6 for being too aggressive). For win, calibration is arguably MORE important because:
1. Win probability estimation errors have outsized EV impact (EV = p * odds, and odds are high)
2. The 2-stage decomposition introduces compounding errors that calibration can correct
3. Benter's Tables 5-7 demonstrate that combined model probabilities show good calibration after the second-stage logit combination

The v5.6 lesson (isotonic overcorrection) should be heeded: fit calibration only on Benter-combined probabilities, not on raw model outputs.

### Decision 6: Pool-Size-Aware Kelly Betting

**What:** Extend `WinStrategy` to account for pari-mutuel pool size effects on dividend.

**Why:** Benter (1994) demonstrates that in pari-mutuel markets, the bettor's own wager reduces the dividend, creating a maximum profitable bet size. For a horse with p=0.06, odds=20, the maximum expected profit bet is only $416 despite the 20% edge. This is critical for JRA pools which may be smaller than HKJC pools.

The formula: `max_bet = pool_size * (p * odds - 1) / (odds - 1 + pool_size_fraction)`

**Implementation:** Use the `overround` feature (already available) as a proxy for pool sophistication, and apply a conservative bet cap based on estimated pool size.

## Patterns to Follow

### Pattern 1: Benter Second-Stage Combination

**What:** Combine fundamental model probability with market-implied probability via logit-space weighting.

**When:** After the fundamental model (WinTwoStageModel + EVCorrection) produces p_win_corrected, but before betting decisions.

**Example:**
```python
# WinBenterGate (new component in SubmodelSet)
class WinBenterGate:
    """Win-specific Benter combination + selection gate."""

    def __init__(self, benter: BenterCombination, gate_model: lgb.Booster | None = None):
        self.benter = benter
        self.gate_model = gate_model  # learned binary gate

    def combine_and_select(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Market implied probability
        p_market_win = np.where(
            df["tanodds"] > 0, 1.0 / df["tanodds"], np.nan
        )
        # 2. Benter logit combination
        df["p_win_combined"] = self.benter.combine(
            df["p_win_corrected"].values,
            np.clip(p_market_win, 0.01, 0.99),
        )
        # 3. Race normalization (sum=1.0)
        df["p_win_combined"] = _normalize_probability_by_race(
            df, "p_win_combined", target_sum=1.0
        )
        # 4. Edge calculation
        df["edge_win"] = df["p_win_combined"] * df["tanodds"] - 1.0
        return df
```

### Pattern 2: Win-Regime Parameter Separation

**What:** Add win-specific parameters to `RegimeDetector.get_strategy_params()`.

**When:** Whenever the regime changes, win betting thresholds should adjust independently of place.

**Example:**
```python
# Add to regime params:
if regime == RegimeState.AGGRESSIVE:
    return {
        # ... existing place params ...
        "win_edge_threshold": 0.05,
        "win_max_odds": 30.0,
        "win_max_bets_per_race": 1,
    }
elif regime == RegimeState.CONSERVATIVE:
    return {
        # ... existing place params ...
        "win_edge_threshold": 0.08,
        "win_max_odds": 20.0,
        "win_max_bets_per_race": 1,
    }
```

### Pattern 3: Selective Feature Expansion

**What:** Add features to WinTwoStageModel based on importance analysis, not blanket inclusion.

**When:** During Phase 1 (feature analysis), identify which of the ~100 existing features matter for win prediction specifically.

**Approach:**
1. Run LightGBM feature importance on the existing Win hit model
2. Identify high-importance features already present
3. Add Place-only features that have theoretical win relevance
4. Validate via time-series cross-validation (WalkForwardCV)

### Pattern 4: Dual Pipeline Coexistence

**What:** Win and Place pipelines share the upstream models (AbilityModel, MarketModel) but diverge at the 2-stage model level.

**When:** Always. The shared upstream ensures consistency while allowing bet-type-specific optimization downstream.

**Implementation in SubmodelSet:**
```python
@dataclass
class SubmodelSet:
    # Shared (existing)
    market: MarketModel
    stage1: AbilityModel
    place_ability: PlaceAbilityModel

    # Win pipeline (enhanced)
    win: WinTwoStageModel
    ev_corrector: EVCorrectionModel           # existing, serves win
    win_benter_gate: WinBenterGate | None = None        # NEW
    win_isotonic_calibrator: IsotonicRegression | None = None  # NEW (win-specific)
    win_temperature_scaler: TemperatureScaling | None = None   # NEW (win-specific)
    win_selection_gate: lgb.Booster | None = None              # NEW

    # Place pipeline (existing, unchanged)
    place: PlaceTwoStageModel
    place_ev_corrector: PlaceEVCorrectionModel
    place_selection_gate: PlaceSelectionGateModel | None = None
    benter_combo: BenterCombination | None = None      # Place Benter
    isotonic_calibrator: IsotonicRegression | None = None     # Place isotonic
    temperature_scaler: TemperatureScaling | None = None      # Place temp

    # Shared
    wide: WideTwoStageModel
    confidence: RobustConfidenceEstimator
    use_ensemble: bool = False
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Reusing Place Calibration for Win

**What:** Using the same IsotonicRegression / TemperatureScaling fitted on place data for win predictions.
**Why bad:** Place probabilities have a much higher base rate (~18-35% place rate vs ~8% win rate). Calibration fitted on place data will systematically overcorrect win probabilities toward the place mean.
**Instead:** Fit separate calibration models on win validation data. The Benter + isotonic + temperature pipeline must be independently fitted for each bet type.

### Anti-Pattern 2: Place-Style Aggressive Multi-Bet on Win

**What:** Allowing 2-3 win bets per race (as place does) with relaxed thresholds.
**Why bad:** Win bets have ~3x the variance of place bets. Multiple win bets per race dramatically increase risk without proportional EV gain. Benter's operation bet "all positive expectation bets" but in pools with $10M+ turnover. JRA pools are smaller, making the dividend impact of multiple bets worse.
**Instead:** Default to max 1 win bet per race. Only consider a second bet when both horses have edge > 2x the threshold AND the probability sum is < 0.25 (to avoid correlated losses).

### Anti-Pattern 3: Full Kelly Betting on Win

**What:** Using the raw Kelly fraction for win bet sizing.
**Why bad:** Benter (1994) explicitly warns: "betting the full amount recommended by the Kelly formula is unwise... if one overestimates the advantage by more than a factor of two, Kelly betting will cause a negative rate of capital growth." Win edge estimation is inherently less precise than place due to the sparser signal (only 1 winner per race).
**Instead:** Use fractional Kelly (1/3 to 1/2 Kelly). The existing `StakeCalculator` already uses half-Kelly with a 0.25 cap, which is appropriate for place but aggressive for win. Consider a separate `win_fractional_kelly = 0.25` (quarter-Kelly) for win bets.

### Anti-Pattern 4: Modifying RacePredictor to Be Win-First

**What:** Changing `RacePredictor.predict()` to prioritize win inference over place.
**Why bad:** `RacePredictor` is a shared component used by both BacktestEngine and PaperPredictor. Changing its internal ordering breaks the place/wide pipeline that other parts of the system depend on.
**Instead:** Add win-specific methods to RacePredictor (e.g., `get_win_candidates()`, `select_win_bets()`) that run after the existing predict chain. The predict chain should remain place-first; win inference reuses its intermediate results (p_win_pred, ev_win_corrected already computed in predict()).

### Anti-Pattern 5: Training Win Models on Different Data Than Place

**What:** Filtering training data differently for win vs place (e.g., excluding certain races from win training but including them for place).
**Why bad:** Creates inconsistency between model predictions. If win and place models see different training data, the relationship between p_win and p_place breaks down, making the Benter combination's probability normalization invalid.
**Instead:** Use identical training data for all bet types within a surface. Filter at the betting decision layer, not the training layer.

## Component Integration: RacePredictor Changes

The `RacePredictor` requires targeted additions without disrupting existing place/wide flow.

### Current predict() flow (lines 88-191 of race_predictor.py):
```
market.predict_and_calc_error -> stage1.add_ability_probs -> place_ability.predict
  -> win.predict_ev -> ev_corrector.correct_ev -> place.predict_ev
  -> place_ev_corrector.correct_ev -> confidence.predict_lower_bound
  -> Benter combination (place) -> place_selection_gate
```

### Proposed additions to predict():
```
... existing flow unchanged ...

# After ev_corrector.correct_ev (line 113), the following win outputs exist:
#   p_win_pred, e_return_win_pred, ev_win
#   p_win_corrected, e_return_win_corrected, ev_win_corrected

# --- NEW: Win Benter + Calibration ---
if submodel.win_benter_gate is not None:
    df = submodel.win_benter_gate.combine_and_select(df)

    # Win-specific calibration (if fitted)
    if submodel.win_isotonic_calibrator is not None:
        df["p_win_calibrated"] = submodel.win_isotonic_calibrator.transform(
            df["p_win_combined"].values
        )
    else:
        df["p_win_calibrated"] = df["p_win_combined"]

    if submodel.win_temperature_scaler is not None:
        df["p_win_calibrated"] = submodel.win_temperature_scaler.transform(
            df["p_win_calibrated"].values
        )

    # Recalculate edge with calibrated probability
    df["edge_win_calibrated"] = df["p_win_calibrated"] * df["tanodds"] - 1.0
```

### New method: get_win_candidates()
```python
def get_win_candidates(
    self,
    race_df: pd.DataFrame,
    *,
    regime_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Select win bet candidates from predicted race data."""
    # Use p_win_calibrated if available, else p_win_corrected
    # Apply edge threshold from regime_params
    # Apply max_odds filter
    # Apply win_selection_gate if trained
    # Return sorted candidates (max 1-2 per race)
```

### New method: select_win_bets()
```python
def select_win_bets(
    self,
    race_df: pd.DataFrame,
    bankroll: float,
    *,
    candidates: pd.DataFrame | None = None,
) -> list[Bet]:
    """Generate win Bet objects with Kelly stake sizing."""
    # Get candidates
    # Calculate Kelly stake (quarter-Kelly for win)
    # Apply pool-size cap
    # Return Bet list
```

## Component Integration: TrainingPipeline Changes

### Current _train_submodel flow (lines 282-608 of training_pipeline.py):
```
market -> ability_oof -> place_ability -> win_2stage -> ev_correction
  -> jockey/trainer/jt context -> place_2stage -> benter+isotonic+temp
  -> place_ev_correction -> wide -> confidence -> place_selection_gate
```

### Proposed insertion point for win components:
```
... existing flow through ev_corrector.correct_ev (line 479) ...

# --- NEW: Win Benter + Calibration ---
if hasattr(win_2s, "_val_p_raw") and len(win_2s._val_p_raw) >= 500:
    # Win-specific Benter combination
    val_p_win = win_2s._val_p_raw  # validation P(win) from hit model
    val_p_market_win = np.where(
        df_oof["tanodds"] > 0, 1.0 / df_oof["tanodds"], 0.5
    )
    val_y_win = (df_oof["kakuteijyuni"] == 1).astype(int).values

    win_benter = BenterCombination.fit(val_p_win, val_p_market_win, val_y_win)

    # Win Isotonic calibration
    val_p_win_combined = win_benter.combine(val_p_win, val_p_market_win)
    win_isotonic = IsotonicRegression(out_of_bounds="clip")
    win_isotonic.fit(val_p_win_combined, val_y_win)

    # Win Temperature Scaling
    val_p_win_isotonic = win_isotonic.transform(val_p_win_combined)
    win_temp_scaler = TemperatureScaling.fit(val_p_win_isotonic, val_y_win)

# ... continue with existing place pipeline ...
```

### SubmodelSet construction changes (line 593):
```python
return SubmodelSet(
    # ... existing fields ...
    win_benter_gate=WinBenterGate(win_benter) if win_benter else None,
    win_isotonic_calibrator=win_isotonic if win_isotonic else None,
    win_temperature_scaler=win_temp_scaler if win_temp_scaler else None,
    # win_selection_gate trained after confidence estimation
)
```

## Component Integration: BacktestEngine Changes

### Current flow (engine.py line 420+):
The backtest loop currently only calls `get_place_candidates()` and `select_bets()` (which only returns place + wide bets). Win bets are NOT generated in the backtest.

### Proposed changes:
```python
# After line 596 (existing bets = self._race_predictor.select_bets(...)):
# Add win bet generation
win_bets = self._race_predictor.select_win_bets(
    result_df, bankroll,
    regime_params=regime_params,
)
all_bets = bets + win_bets
```

### Settlement changes:
Win bet settlement already works generically (Bet.result = odds * stake if horse finishes 1st). The existing settlement logic in `BacktestEngine._settle_bets()` handles this via `kakuteijyuni == 1` check.

## Scalability Considerations

| Concern | Current (Place-focused) | After Win Enhancement | Notes |
|---------|------------------------|----------------------|-------|
| Model count per surface | 10 models in SubmodelSet | 13-14 models (+3-4 win-specific) | Memory impact negligible; models are <50MB each |
| Training time | ~57 min/year | ~65 min/year (+10-15%) | Win Benter fitting is fast (<1 min); gate training adds ~3 min |
| Backtest inference per race | ~5 ms | ~6 ms (+20%) | Win Benter + calibration adds ~1 ms per race |
| Bet frequency | 9,074 place bets/year | ~2,000-3,000 win bets/year | Win is sparser; total volume depends on threshold tuning |
| Bankroll at risk per race | 2% (place cap) | 2% total (place + win combined) | Race exposure cap already handles this |

## Build Order (Dependency Chain)

The win model improvement should be built in this order, where each phase builds on the previous:

```
Phase 1: Feature Analysis & Win Feature Enhancement
  ├─ Analyze feature importance for existing Win hit model
  ├─ Expand WinTwoStageModel.FEATURE_COLS with validated features
  ├─ Run WalkForwardCV to measure improvement
  └─ No architectural changes needed

Phase 2: Win Benter Combination + Calibration
  ├─ Add WinBenterGate component
  ├─ Fit BenterCombination on win validation data
  ├─ Add Win IsotonicRegression + TemperatureScaling
  ├─ Integrate into TrainingPipeline._train_submodel()
  ├─ Integrate into RacePredictor.predict()
  └─ Run backtest to measure improvement

Phase 3: Win Selection Gate + Confidence Estimation
  ├─ Train win-specific selection gate (like PlaceSelectionGate)
  ├─ Add WinConfidenceEstimator (EV lower bound for win)
  ├─ Integrate into SubmodelSet
  └─ Run backtest to measure improvement

Phase 4: Win Betting Strategy Enhancement
  ├─ Add win-specific regime parameters
  ├─ Implement pool-size-aware Kelly sizing in WinStrategy
  ├─ Add get_win_candidates() + select_win_bets() to RacePredictor
  ├─ Integrate win bets into BacktestEngine
  └─ Run final backtest for ROI measurement

Phase 5: Validation & Hardening
  ├─ WalkForwardCV across multiple years
  ├─ Sensitivity analysis on edge thresholds
  ├─ Overfitting check (train vs test ROI gap)
  └─ Paper trading validation
```

### Dependency rationale:
- Phase 1 is independent of all others and has the highest expected ROI (better features -> better P(win))
- Phase 2 depends on Phase 1 because Benter combination quality depends on fundamental model quality
- Phase 3 depends on Phase 2 because the selection gate needs calibrated probabilities as input
- Phase 4 depends on Phase 2-3 because betting strategy needs calibrated edge estimates
- Phase 5 depends on all previous phases

### Risk mitigation:
- Each phase produces a measurable backtest ROI change
- If a phase shows negative ROI, it can be reverted independently
- The Place pipeline remains completely untouched throughout

## Sources

- Benter (1994), "Computer Based Horse Race Handicapping and Wagering Systems: A Report" -- annotated version at [Acta Machina](http://actamachina.com/posts/annotated-benter-paper). HIGH confidence for architectural patterns.
- Bolton & Chapman (1986), "The Searching Problem in Probabilistic Simulation of Horse Racing" -- multinomial logit model for horse racing. HIGH confidence.
- Existing codebase: `src/models/two_stage_return_model.py`, `src/models/benter_combination.py`, `src/models/ev_correction_model.py`, `src/backtest/race_predictor.py`, `src/pipelines/training_pipeline.py`. PRIMARY confidence.
- [StableBet](https://stablebet.co.uk/betting/strategies/ai-prediction-models/) -- LightGBM + isotonic calibration pattern for racing. MEDIUM confidence.
- [seven-seas-punter](https://github.com/levonrush/seven-seas-punter) -- time-based CV + calibrated probabilities + value betting backtest. MEDIUM confidence.

---
*Architecture research: 2026-05-02*
