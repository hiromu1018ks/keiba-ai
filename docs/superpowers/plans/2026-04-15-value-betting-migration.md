# Value Betting Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate Place bet selection from EV-based criterion (`p × e_return >= 1.1`) to Value Betting (`p_model - p_market >= edge_threshold`), eliminating the e_return overestimation problem.

**Architecture:** Replace the EV bet-selection criterion with edge = p_place_pred - 1/fukuoddslow. The model's predicted probability (p_place_pred) is compared against the market's implied probability (1/fukuoddslow). Horses where the model identifies positive edge become bet candidates. Kelly sizing uses edge directly: `kelly = (edge × odds) / (odds - 1)`. PlaceEVCorrectionModel remains in the pipeline (its output stays available) but bet selection no longer depends on e_return or ev_place_corrected.

**Tech Stack:** Python 3.11, pandas, numpy, pytest, LightGBM (unchanged)

**Supersedes:** `docs/superpowers/plans/2026-04-15-place-ev-overestimation-fix.md` — Value Betting makes e_return accuracy irrelevant for bet selection.

---

## Background: Why This Change

The current system computes `EV = p_place_pred × e_return_place_pred` and bets when EV ≥ 1.1. When e_return was 2.2× overestimated, the system accidentally approximated Value Betting (high-p horses got higher EV). After fixing e_return accuracy, EV flattened to ~1.0 for all horses, eliminating bet candidates.

Value Betting solves this by directly comparing model probability vs market probability:
- `edge = p_model - p_market` — how much the model disagrees with the market
- Bet when `edge ≥ threshold` — the model sees at least N% advantage
- Size with Kelly: `kelly = (edge × odds) / (odds - 1)` — the standard formula

---

## File Structure

### Modified files:

| File | Responsibility | Change summary |
|------|---------------|----------------|
| `src/domain/models.py:145-169` | Bet dataclass | Add `edge: float = 0.0` field |
| `src/backtest/race_predictor.py:38-178` | Bet selection | Add edge computation in `predict()`, migrate `select_bets()` to edge-based |
| `src/betting/stake_calculator.py:1-123` | Kelly sizing | Change `calc_stake()` signature: `ev_lower` → `edge`, new Kelly formula |
| `src/models/regime_detector.py:180-202` | Regime params | Add `edge_threshold` alongside existing `ev_threshold` |
| `src/betting/meta_switcher.py:41-64` | Regime params | Mirror RegimeDetector: add `edge_threshold` |
| `src/betting/place_strategy.py:1-88` | Orchestrator | Use `edge_place` instead of `ev_lower_place` |
| `src/betting/gate_keeper.py:1-41` | Bet filter | Filter on `edge` instead of `ev_lower_corrected` |
| `src/backtest/engine.py:1-676` | Backtest | Pass edge to Bet, track in history, fix n_candidates |
| `src/betting/orchestrator.py:1-221` | Live betting | Update Protocols + process_race() call sites |
| `src/paper_trading/predictor.py` | Paper trading | Log `bet.edge` in bet history |

### Test files modified:

| File | Change |
|------|--------|
| `tests/test_race_predictor.py` | Add edge computation test, update select_bets tests |
| `tests/test_stake_calculator.py` | New Value Betting Kelly tests |
| `tests/test_regime_detector.py` | Add edge_threshold param tests |
| `tests/test_meta_switcher.py` | Add edge_threshold param tests |
| `tests/test_place_strategy.py` | Update to use edge-based filtering |
| `tests/test_gate_keeper.py` | Update to use edge-based filtering |

---

## Task 1: Add `edge` field to Bet dataclass

**Files:**
- Modify: `src/domain/models.py` (Bet dataclass, ~line 145-169)
- Test: `tests/test_domain.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_domain.py`:

```python
def test_bet_has_edge_field():
    """Bet dataclass should have an edge field for Value Betting."""
    bet = Bet(
        race_id="20240101T11R01",
        umaban=1,
        bet_type=BetType.PLACE,
        odds=1.5,
        ev_lower_corrected=0.0,
        stake=100.0,
        edge=0.033,
    )
    assert bet.edge == pytest.approx(0.033)


def test_bet_edge_defaults_to_zero():
    """Bet edge should default to 0.0 for backward compatibility."""
    bet = Bet(
        race_id="20240101T11R01",
        umaban=1,
        bet_type=BetType.PLACE,
        odds=1.5,
        ev_lower_corrected=1.2,
        stake=100.0,
    )
    assert bet.edge == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_domain.py::test_bet_has_edge_field tests/test_domain.py::test_bet_edge_defaults_to_zero -v`
Expected: FAIL — `Bet.__init__()` got an unexpected keyword argument `edge`

- [ ] **Step 3: Add `edge` field to Bet dataclass**

In `src/domain/models.py`, find the `Bet` dataclass and add `edge` field before `final_odds`:

```python
@dataclass
class Bet:
    race_id: str
    umaban: int
    bet_type: BetType
    odds: float
    ev_lower_corrected: float
    stake: float
    edge: float = 0.0              # Value Betting edge (p_model - p_market)
    final_odds: float = 0.0
    result: Optional[float] = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_domain.py::test_bet_has_edge_field tests/test_domain.py::test_bet_edge_defaults_to_zero -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/domain/models.py tests/test_domain.py
git commit -m "feat: Bet dataclass に edge フィールドを追加 (Value Betting 用)"
```

---

## Task 2: Add `edge_threshold` to RegimeDetector and MetaSwitcher

**Files:**
- Modify: `src/models/regime_detector.py:180-202` (`get_strategy_params()`)
- Modify: `src/betting/meta_switcher.py:41-64` (`_default_params()`)
- Test: `tests/test_regime_detector.py`
- Test: `tests/test_meta_switcher.py`

- [ ] **Step 1: Write the failing test for RegimeDetector**

Add to `tests/test_regime_detector.py`:

```python
def test_get_strategy_params_contains_edge_threshold():
    """RegimeDetector should return edge_threshold for Value Betting."""
    detector = RegimeDetector()
    for regime in [RegimeState.AGGRESSIVE, RegimeState.CONSERVATIVE, RegimeState.COLLAPSED]:
        params = detector.get_strategy_params(regime)
        assert "edge_threshold" in params
        assert isinstance(params["edge_threshold"], float)
        assert params["edge_threshold"] > 0


def test_edge_threshold_values_by_regime():
    """Edge thresholds should increase from AGGRESSIVE to COLLAPSED."""
    detector = RegimeDetector()
    agg = detector.get_strategy_params(RegimeState.AGGRESSIVE)
    con = detector.get_strategy_params(RegimeState.CONSERVATIVE)
    col = detector.get_strategy_params(RegimeState.COLLAPSED)
    assert agg["edge_threshold"] < con["edge_threshold"] < col["edge_threshold"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_regime_detector.py::test_get_strategy_params_contains_edge_threshold tests/test_regime_detector.py::test_edge_threshold_values_by_regime -v`
Expected: FAIL — `AssertionError: 'edge_threshold' not in params`

- [ ] **Step 3: Add `edge_threshold` to RegimeDetector.get_strategy_params()**

In `src/models/regime_detector.py`, modify `get_strategy_params()` to include `edge_threshold` alongside existing `ev_threshold`:

```python
def get_strategy_params(self, regime: RegimeState) -> dict[str, object]:
    if regime == RegimeState.AGGRESSIVE:
        return {
            "ev_threshold": 1.10,
            "edge_threshold": 0.03,       # 3% edge — Value Betting
            "score_threshold": 0.010,
            "max_bets_per_race": 3,
        }
    elif regime == RegimeState.CONSERVATIVE:
        return {
            "ev_threshold": 1.30,
            "edge_threshold": 0.05,       # 5% edge — more selective
            "score_threshold": 0.020,
            "max_bets_per_race": 2,
        }
    else:  # COLLAPSED
        return {
            "ev_threshold": 1.50,
            "edge_threshold": 0.08,       # 8% edge — near halt
            "score_threshold": 0.050,
            "max_bets_per_race": 1,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_regime_detector.py::test_get_strategy_params_contains_edge_threshold tests/test_regime_detector.py::test_edge_threshold_values_by_regime -v`
Expected: PASS

- [ ] **Step 5: Write the failing test for MetaSwitcher**

Add to `tests/test_meta_switcher.py`:

```python
def test_default_params_contains_edge_threshold():
    """MetaSwitcher should mirror edge_threshold from RegimeDetector."""
    mock_rd = MagicMock()
    mock_rd.current_regime = RegimeState.AGGRESSIVE
    switcher = MetaSwitcher(mock_rd)
    params = switcher.get_strategy_params()
    assert "edge_threshold" in params
    assert params["edge_threshold"] == 0.03
```

- [ ] **Step 6: Run test to verify it fails**

Run: `python -m pytest tests/test_meta_switcher.py::test_default_params_contains_edge_threshold -v`
Expected: FAIL

- [ ] **Step 7: Add `edge_threshold` to MetaSwitcher._default_params()**

In `src/betting/meta_switcher.py`, modify `_default_params()` to mirror the RegimeDetector values:

```python
def _default_params(self, regime: RegimeState) -> dict[str, object]:
    if regime == RegimeState.AGGRESSIVE:
        return {
            "ev_threshold": 1.10,
            "edge_threshold": 0.03,
            "score_threshold": 0.010,
            "max_bets_per_race": 3,
        }
    elif regime == RegimeState.CONSERVATIVE:
        return {
            "ev_threshold": 1.30,
            "edge_threshold": 0.05,
            "score_threshold": 0.020,
            "max_bets_per_race": 2,
        }
    else:  # COLLAPSED
        return {
            "ev_threshold": 1.50,
            "edge_threshold": 0.08,
            "score_threshold": 0.050,
            "max_bets_per_race": 1,
        }
```

- [ ] **Step 8: Run all regime-related tests**

Run: `python -m pytest tests/test_regime_detector.py tests/test_meta_switcher.py -v`
Expected: ALL PASS

- [ ] **Step 9: Commit**

```bash
git add src/models/regime_detector.py src/betting/meta_switcher.py tests/test_regime_detector.py tests/test_meta_switcher.py
git commit -m "feat: RegimeDetector/MetaSwitcher に edge_threshold を追加 (Value Betting 閾値)"
```

---

## Task 3: Add edge computation to RacePredictor.predict()

**Files:**
- Modify: `src/backtest/race_predictor.py` (`predict()` method, after step 9)
- Test: `tests/test_race_predictor.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_race_predictor.py`:

```python
def test_predict_computes_edge_place():
    """predict() should compute edge_place = p_place_pred - 1/fukuoddslow."""
    submodel = _make_submodel_mock()
    models = _make_models_mock(submodel)

    # Configure all mock returns (same pattern as test_predict_returns_dataframe_with_ev_columns)
    # ... (follow existing mock setup pattern in the file)

    race_df = pd.DataFrame({
        "race_id": ["R01"],
        "umaban": [1],
        "surface": ["turf"],
        "kyori": [1600],
        "distance_bin": [16],
        "popularity_rank": [1],
        "ninki": [1],
        "fukuoddslow": [1.5],
        "kakuteijyuni": [1],
        "kettonum": [1],
        "odds": [2.0],
        "bataijyu": [500.0],
        "field_size": [16],
        "track_condition_code": [1],
        "grade_code": [0],
        "p_place_pred": [0.70],
    })
    # ... configure all mock returns as existing tests do

    predictor = RacePredictor(models)
    result = predictor.predict(race_df)

    assert "edge_place" in result.columns
    # edge = 0.70 - 1/1.5 = 0.70 - 0.667 = 0.033
    expected_edge = 0.70 - 1.0 / 1.5
    assert result["edge_place"].iloc[0] == pytest.approx(expected_edge, abs=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_race_predictor.py::test_predict_computes_edge_place -v`
Expected: FAIL — `AssertionError: 'edge_place' not in result.columns`

- [ ] **Step 3: Add edge computation to predict()**

In `src/backtest/race_predictor.py`, at the end of the `predict()` method (after confidence lower bound computation, before the `return df` statement), add:

```python
        # --- Value Betting: edge = p_model - p_market ---
        # p_market = 1 / fukuoddslow (market implied probability)
        # edge > 0 means the model thinks the horse is undervalued
        p_market = np.where(
            df["fukuoddslow"] > 0,
            1.0 / df["fukuoddslow"],
            np.nan,
        )
        df["edge_place"] = df["p_place_pred"] - p_market
```

**IMPORTANT:** `race_predictor.py` currently does NOT import numpy. Add `import numpy as np` at the top of the file alongside the existing `import pandas as pd`. Alternatively, use pandas-only: `p_market = 1.0 / df["fukuoddslow"].where(df["fukuoddslow"] > 0)` — both work, but `np.where` is more explicit about the NaN case.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_race_predictor.py::test_predict_computes_edge_place -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtest/race_predictor.py tests/test_race_predictor.py
git commit -m "feat: RacePredictor.predict() に edge_place 計算を追加"
```

---

## Task 4: Migrate select_bets() to edge-based selection

**Files:**
- Modify: `src/backtest/race_predictor.py` (`select_bets()` method, ~lines 120-178)
- Test: `tests/test_race_predictor.py`

This is the core change. The method currently filters by `ev_place_corrected >= ev_threshold` and sorts by EV. We change it to filter by `edge_place >= edge_threshold` and sort by edge.

- [ ] **Step 1: Write the failing test for edge-based selection**

Add to `tests/test_race_predictor.py`:

```python
def test_select_bets_uses_edge_not_ev():
    """select_bets() should filter by edge_place, not ev_place_corrected."""
    submodel = _make_submodel_mock()
    models = _make_models_mock(submodel)
    predictor = RacePredictor(models)

    race_df = pd.DataFrame({
        "race_id": ["R01", "R01", "R01"],
        "umaban": [1, 2, 3],
        "fukuoddslow": [1.5, 3.0, 10.0],
        # Horse 1: high EV but negative edge (model < market) → NO BET
        # Horse 2: low EV but positive edge → BET
        # Horse 3: high EV but zero edge → NO BET
        "p_place_pred": [0.60, 0.40, 0.10],
        "edge_place": [0.60 - 1/1.5, 0.40 - 1/3.0, 0.10 - 1/10.0],  # [-0.067, 0.067, 0.000]
        "ev_place_corrected": [1.8, 0.9, 2.0],  # EV says bet horse 1 and 3
        "EV_lower_place": [1.5, 0.8, 1.8],
        "kakuteijyuni": [1, 2, 5],
        "popularity_rank": [1, 3, 8],
        "field_size": [10],
        "surface": ["turf"],
        "distance_bin": [16],
        "track_condition_code": [1],
        "grade_code": [0],
    })

    # Force AGGRESSIVE regime with edge_threshold=0.03
    models.regime_detector.get_strategy_params.return_value = {
        "ev_threshold": 1.10,
        "edge_threshold": 0.03,
        "max_bets_per_race": 3,
    }
    models.regime_detector.current_regime = "AGGRESSIVE"

    bets = predictor.select_bets(race_df, bankroll=100000)
    # Only horse 2 should be selected: edge=0.067 > 0.03
    # Horse 1: edge=-0.067 < 0.03 → NO
    # Horse 3: edge=0.000 < 0.03 → NO
    assert len(bets) == 1
    assert bets[0].umaban == 2
    assert bets[0].edge == pytest.approx(0.40 - 1.0/3.0, abs=1e-4)


def test_select_bets_edge_threshold_respects_regime():
    """Horse should NOT be selected when edge < regime edge_threshold."""
    submodel = _make_submodel_mock()
    models = _make_models_mock(submodel)
    predictor = RacePredictor(models)

    race_df = pd.DataFrame({
        "race_id": ["R01"],
        "umaban": [1],
        "fukuoddslow": [1.5],
        "p_place_pred": [0.69],  # edge = 0.69 - 0.667 = 0.023
        "edge_place": [0.023],
        "ev_place_corrected": [1.8],
        "EV_lower_place": [1.5],
        "kakuteijyuni": [1],
        "popularity_rank": [1],
        "field_size": [10],
        "surface": ["turf"],
        "distance_bin": [16],
        "track_condition_code": [1],
        "grade_code": [0],
    })

    # CONSERVATIVE: edge_threshold=0.05 → 0.023 < 0.05 → NO BET
    models.regime_detector.get_strategy_params.return_value = {
        "ev_threshold": 1.30,
        "edge_threshold": 0.05,
        "max_bets_per_race": 2,
    }
    bets = predictor.select_bets(race_df, bankroll=100000)
    assert len(bets) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_race_predictor.py::test_select_bets_uses_edge_not_ev tests/test_race_predictor.py::test_select_bets_edge_threshold_respects_regime -v`
Expected: FAIL — wrong horses selected (old EV-based logic)

- [ ] **Step 3: Rewrite select_bets() to use edge**

In `src/backtest/race_predictor.py`, rewrite the `select_bets()` method:

```python
def select_bets(
    self,
    race_df: pd.DataFrame,
    bankroll: float,
) -> list[Bet]:
    """Value Betting: select horses where edge = p_model - p_market >= threshold."""
    regime = self.models.regime_detector.current_regime
    regime_params = self.models.regime_detector.get_strategy_params(regime)
    edge_threshold = regime_params.get("edge_threshold", 0.03)
    max_bets = regime_params.get("max_bets_per_race", 3)

    # Filter by edge (Value Betting criterion)
    candidates = race_df[race_df["edge_place"].fillna(0) >= edge_threshold]
    candidates = candidates.nlargest(max_bets, "edge_place")

    bets: list[Bet] = []
    for _, row in candidates.iterrows():
        edge_val = float(row["edge_place"])
        odds_val = float(row["fukuoddslow"])

        if self.stake_calc is not None:
            # Kelly sizing with Value Betting edge
            stake = self.stake_calc.calc_stake(
                edge=edge_val,
                odds=odds_val,
                bankroll=bankroll,
                bet_type=BetType.PLACE,
            )
            if self.dd_ctrl is not None:
                stake = self.dd_ctrl.adjust_stake(stake, bankroll)
                stake = max(0, math.floor(stake / 100) * 100)
        else:
            # Flat mode: fixed 100 yen
            stake = 100.0

        if stake < 100:
            continue

        bets.append(
            Bet(
                race_id=row["race_id"],
                umaban=int(row["umaban"]),
                bet_type=BetType.PLACE,
                odds=odds_val,
                ev_lower_corrected=float(row.get("ev_place_corrected", 0)),
                stake=stake,
                edge=edge_val,
            )
        )

    return bets
```

**Key changes from old version:**
- `ev_col = "ev_place_corrected"` → `edge_col = "edge_place"`
- `ev_threshold` → `edge_threshold`
- `candidates.nlargest(max_bets, ev_col)` → `candidates.nlargest(max_bets, "edge_place")`
- `calc_stake(ev_lower=..., odds=..., ...)` → `calc_stake(edge=..., odds=..., ...)`
- Bet construction: add `edge=edge_val`

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_race_predictor.py::test_select_bets_uses_edge_not_ev tests/test_race_predictor.py::test_select_bets_edge_threshold_respects_regime -v`
Expected: PASS

- [ ] **Step 5: Update existing select_bets tests**

Update the existing `test_select_bets_returns_list` and `test_select_bets_flat_mode_uses_100_yen` tests:

1. Add `"edge_place"` column to test DataFrames with appropriate values (e.g., `[0.05]` for a horse that should pass the threshold)
2. Update the mock `get_strategy_params` return value to include `"edge_threshold": 0.03`
3. The existing fixture `_make_models_mock` (or however the mock `regime_detector` is configured) must return `edge_threshold` in its `get_strategy_params` return value

Example update for mock setup:
```python
models.regime_detector.get_strategy_params.return_value = {
    "ev_threshold": 1.20,
    "edge_threshold": 0.03,   # NEW
    "max_bets_per_race": 3,
}
```

- [ ] **Step 6: Run all race_predictor tests**

Run: `python -m pytest tests/test_race_predictor.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add src/backtest/race_predictor.py tests/test_race_predictor.py
git commit -m "feat: select_bets() を EV ベースから Value Betting (edge) に移行"
```

---

## Task 5: Migrate StakeCalculator to Value Betting Kelly

**Files:**
- Modify: `src/betting/stake_calculator.py:30-77` (`calc_stake()`)
- Test: `tests/test_stake_calculator.py`

The current formula: `kelly = (ev_lower - 1) / (odds - 1)` (EV-based Kelly)
The new formula: `kelly = (edge × odds) / (odds - 1)` (Value Betting Kelly)

These are mathematically equivalent when `ev_lower = p × odds`, but the new version takes edge directly as input, making the intent clear.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_stake_calculator.py`:

```python
def test_calc_stake_value_betting_positive_edge():
    """Value Betting: positive edge should produce a stake."""
    calc = StakeCalculator()
    # edge=0.033, odds=1.5, bankroll=100000
    # kelly = (0.033 * 1.5) / (1.5 - 1) = 0.0495 / 0.5 = 0.099
    # half-kelly: 0.099 * 0.5 = 0.0495
    # cap check: min(0.0495, 0.125) = 0.0495
    # raw_stake = 100000 * 0.0495 = 4950
    # rounded: 4900
    stake = calc.calc_stake(edge=0.033, odds=1.5, bankroll=100_000, bet_type=BetType.PLACE)
    assert stake == 4900.0


def test_calc_stake_value_betting_zero_edge():
    """Value Betting: zero edge should return 0."""
    calc = StakeCalculator()
    stake = calc.calc_stake(edge=0.0, odds=1.5, bankroll=100_000, bet_type=BetType.PLACE)
    assert stake == 0.0


def test_calc_stake_value_betting_negative_edge():
    """Value Betting: negative edge should return 0."""
    calc = StakeCalculator()
    stake = calc.calc_stake(edge=-0.05, odds=1.5, bankroll=100_000, bet_type=BetType.PLACE)
    assert stake == 0.0


def test_calc_stake_value_betting_high_edge_respects_cap():
    """Value Betting: high edge should still respect the Kelly cap."""
    calc = StakeCalculator()
    # edge=0.20, odds=1.5, bankroll=100000
    # kelly = (0.20 * 1.5) / 0.5 = 0.6
    # half-kelly: 0.3, capped at 0.125
    # raw_stake = 100000 * 0.125 = 12500 → capped at 10000
    stake = calc.calc_stake(edge=0.20, odds=1.5, bankroll=100_000, bet_type=BetType.PLACE)
    assert stake == 10000.0


def test_calc_stake_value_betting_formula_equivalence():
    """Verify VB Kelly = (edge * odds) / (odds - 1) matches standard Kelly."""
    calc = StakeCalculator()
    # p_model = 0.70, odds = 1.5, edge = 0.70 - 1/1.5 = 0.0333
    # Standard Kelly: f* = (p*b - q) / b where b=odds-1, p=p_model, q=1-p_model
    # f* = (0.70*0.5 - 0.30) / 0.5 = (0.35 - 0.30) / 0.5 = 0.10
    # VB Kelly: f* = (edge * odds) / (odds - 1) = (0.0333 * 1.5) / 0.5 = 0.10
    edge = 0.70 - 1.0 / 1.5  # 0.0333...
    odds = 1.5
    standard_kelly = (0.70 * (odds - 1) - (1 - 0.70)) / (odds - 1)
    vb_kelly = (edge * odds) / (odds - 1)
    assert abs(standard_kelly - vb_kelly) < 1e-10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stake_calculator.py::test_calc_stake_value_betting_positive_edge -v`
Expected: FAIL — `TypeError: calc_stake() got an unexpected keyword argument 'edge'`

- [ ] **Step 3: Rewrite calc_stake() for Value Betting**

In `src/betting/stake_calculator.py`, replace the `calc_stake()` method:

```python
MIN_EDGE_THRESHOLD: float = 0.005   # Minimum edge to consider betting (0.5%)
FRACTIONAL_KELLY: float = 0.5       # Half-Kelly
KELLY_FRACTION_CAP: float = 0.25    # Max Kelly fraction (1/4 of full Kelly)
RACE_EXPOSURE_CAP: float = 0.02     # Max 2% of bankroll per race
MIN_STAKE: int = 100                # Minimum bet unit (100 yen)
MAX_STAKE: int = 10000              # Absolute maximum stake (10K yen)


def calc_stake(
    self, edge: float, odds: float, bankroll: float, bet_type: BetType
) -> float:
    """Calculate Kelly-optimal stake for Value Betting.

    Args:
        edge: Value Betting edge = p_model - p_market (= p_model - 1/odds)
        odds: Decimal odds (e.g., 1.5 means 1.5x return)
        bankroll: Current bankroll in yen
        bet_type: Type of bet (PLACE, WIN, WIDE)

    Returns:
        Stake in yen (multiple of 100), or 0.0 if no bet recommended.
    """
    if bankroll <= 0 or odds <= 1.0:
        return 0.0
    if math.isnan(edge) or math.isnan(odds):
        return 0.0
    if edge < self.MIN_EDGE_THRESHOLD:
        return 0.0

    # Value Betting Kelly: f* = (edge * odds) / (odds - 1)
    kelly_fraction = (edge * odds) / (odds - 1.0)

    # Fractional Kelly (half-Kelly for safety)
    kelly_fraction *= self.FRACTIONAL_KELLY

    # Effective cap: max fraction of bankroll
    effective_cap = self.KELLY_FRACTION_CAP * self.FRACTIONAL_KELLY  # 0.125
    kelly_fraction = min(kelly_fraction, effective_cap)

    # Compute stake
    raw_stake = bankroll * kelly_fraction
    stake = max(0, math.floor(raw_stake / self.MIN_STAKE) * self.MIN_STAKE)

    # Absolute cap
    stake = min(stake, self.MAX_STAKE)

    return float(stake)
```

**Key changes:**
- Parameter: `ev_lower: float` → `edge: float`
- Guard: `ev_lower < MIN_EV_THRESHOLD (1.05)` → `edge < MIN_EDGE_THRESHOLD (0.005)`
- Formula: `(ev_lower - 1.0) / (odds - 1.0)` → `(edge * odds) / (odds - 1.0)`
- Remove `MIN_EV_THRESHOLD = 1.05`, add `MIN_EDGE_THRESHOLD = 0.005`

- [ ] **Step 4: Run new tests to verify they pass**

Run: `python -m pytest tests/test_stake_calculator.py::test_calc_stake_value_betting_positive_edge tests/test_stake_calculator.py::test_calc_stake_value_betting_zero_edge tests/test_stake_calculator.py::test_calc_stake_value_betting_negative_edge tests/test_stake_calculator.py::test_calc_stake_value_betting_high_edge_respects_cap tests/test_stake_calculator.py::test_calc_stake_value_betting_formula_equivalence -v`
Expected: ALL PASS

- [ ] **Step 5: Update existing StakeCalculator tests**

The existing tests use `ev_lower` parameter. Update them to use `edge`:
- `test_calc_stake_positive_ev` → convert ev_lower to edge equivalent
- `test_calc_stake_below_ev_threshold` → use edge < MIN_EDGE_THRESHOLD
- `test_calc_stake_rounds_to_100` → use edge-based call
- `test_calc_stake_higher_ev_larger_stake` → use edge-based comparison
- `test_calc_stake_fractional_kelly_halves_stake` → update formula in comments
- `test_calc_stake_effective_cap` → use high edge
- `test_calc_stake_max_stake_cap` → use very high edge
- `test_calc_stake_zero_bankroll` → edge-based call with bankroll=0

For each test, the conversion is:
```python
# Old: calc_stake(ev_lower=1.20, odds=2.0, bankroll=100000, bet_type=BetType.PLACE)
# New: calc_stake(edge=0.10, odds=2.0, bankroll=100000, bet_type=BetType.PLACE)
# where edge = (ev_lower - 1) / odds * (odds - 1) / odds ... actually:
# ev_lower = p * odds, edge = p - 1/odds
# So if ev_lower = 1.20 and odds = 2.0: p = 0.60, edge = 0.60 - 0.50 = 0.10
```

- [ ] **Step 6: Run all stake_calculator tests**

Run: `python -m pytest tests/test_stake_calculator.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add src/betting/stake_calculator.py tests/test_stake_calculator.py
git commit -m "feat: StakeCalculator を Value Betting Kelly に移行 (edge ベース)"
```

---

## Task 6: Migrate orchestrator path (PlaceStrategy + GateKeeper)

**Files:**
- Modify: `src/betting/place_strategy.py` (full file, ~88 lines)
- Modify: `src/betting/gate_keeper.py` (full file, ~41 lines)
- Test: `tests/test_place_strategy.py`
- Test: `tests/test_gate_keeper.py`

### PlaceStrategy

- [ ] **Step 1: Write the failing test for PlaceStrategy**

Add to `tests/test_place_strategy.py`:

```python
def test_generate_uses_edge_not_ev():
    """PlaceStrategy should filter by edge_place, not ev_lower_place."""
    strategy = PlaceStrategy()
    feats = {
        "edge_place": [0.01, 0.05, -0.02],  # horse 1: low, horse 2: good, horse 3: negative
        "ev_lower_place": [1.5, 0.8, 2.0],   # old EV says horse 1 and 3
        "place_odds": [1.5, 3.0, 10.0],
        "umaban": [1, 2, 3],
    }
    bets = strategy.generate(feats, bankroll=100_000, ev_threshold=1.10, max_bets=3)
    # Only horse 2 has edge > 0.03 threshold
    assert len(bets) == 1
    assert bets[0]["umaban"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_place_strategy.py::test_generate_uses_edge_not_ev -v`
Expected: FAIL

- [ ] **Step 3: Rewrite PlaceStrategy.generate() to use edge**

In `src/betting/place_strategy.py`, modify the `generate()` method:

The key changes:
1. Replace `ev_lower_list = feats["ev_lower_place"]` with `edge_list = feats["edge_place"]`
2. Replace `if ev_lower >= ev_threshold:` with `if edge >= edge_threshold:`
3. Use edge for sorting and Kelly calculation
4. The method signature changes: add `edge_threshold` parameter (keep `ev_threshold` for backward compat but unused)

```python
def generate(
    self,
    feats: dict,
    bankroll: float,
    ev_threshold: float = 1.10,      # kept for API compat, not used
    max_bets: int = 3,
    edge_threshold: float = 0.03,    # NEW: Value Betting threshold
) -> list[dict]:
```

Replace the filtering logic:
```python
    edge_list = feats["edge_place"]
    odds_list = feats["place_odds"]

    candidates = []
    for i, edge in enumerate(edge_list):
        if pd.isna(edge) or edge < edge_threshold:
            continue
        odds = odds_list[i]
        if pd.isna(odds) or odds <= 1.0:
            continue
        # Value Betting Kelly
        kelly_fraction = min((edge * odds) / (odds - 1.0), 0.25)
        kelly_fraction *= 0.5  # half-Kelly
        raw_stake = bankroll * kelly_fraction
        stake = max(0, math.floor(raw_stake / 100) * 100)
        stake = min(stake, 10000)

        if stake >= 100:
            candidates.append({
                "umaban": int(feats["umaban"][i]),
                "bet_type": "PLACE",
                "odds": float(odds),
                "edge": float(edge),
                "stake": stake,
            })

    candidates.sort(key=lambda x: x["edge"], reverse=True)
    return candidates[:max_bets]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_place_strategy.py::test_generate_uses_edge_not_ev -v`
Expected: PASS

### GateKeeper

- [ ] **Step 5: Write the failing test for GateKeeper**

Add to `tests/test_gate_keeper.py`:

```python
def test_filter_bets_uses_edge():
    """GateKeeper should filter on edge, not ev_lower_corrected."""
    gk = GateKeeper()
    bets = [
        Bet(race_id="R1", umaban=1, bet_type=BetType.PLACE, odds=1.5,
            ev_lower_corrected=0.0, stake=100.0, edge=0.05),
        Bet(race_id="R1", umaban=2, bet_type=BetType.PLACE, odds=3.0,
            ev_lower_corrected=0.0, stake=100.0, edge=0.01),
    ]
    filtered = gk.filter_bets(bets, edge_threshold=0.03)
    assert len(filtered) == 1
    assert filtered[0].umaban == 1
```

- [ ] **Step 6: Run test to verify it fails**

Run: `python -m pytest tests/test_gate_keeper.py::test_filter_bets_uses_edge -v`
Expected: FAIL

- [ ] **Step 7: Rewrite GateKeeper.filter_bets() to use edge**

In `src/betting/gate_keeper.py`, modify `filter_bets()`:

```python
def filter_bets(self, bets: list[Bet], edge_threshold: float = 0.03) -> list[Bet]:
    """Filter bets by Value Betting edge threshold."""
    return [b for b in bets if b.edge >= edge_threshold]
```

- [ ] **Step 8: Run all gate_keeper tests**

Run: `python -m pytest tests/test_gate_keeper.py -v`
Expected: ALL PASS

- [ ] **Step 9: Update existing place_strategy and gate_keeper tests**

Update any remaining tests that use the old `ev_threshold` or `ev_lower_corrected` parameters.

- [ ] **Step 10: Run all orchestrator-related tests**

Run: `python -m pytest tests/test_place_strategy.py tests/test_gate_keeper.py tests/test_meta_switcher.py -v`
Expected: ALL PASS

- [ ] **Step 11: Commit**

```bash
git add src/betting/place_strategy.py src/betting/gate_keeper.py tests/test_place_strategy.py tests/test_gate_keeper.py
git commit -m "feat: PlaceStrategy/GateKeeper を Value Betting (edge) に移行"
```

---

## Task 6.5: Update BettingOrchestrator Protocols and call sites

**Files:**
- Modify: `src/betting/orchestrator.py` (Protocols + `process_race()`)
- Test: `tests/test_orchestrator.py`

This is a critical fix — the orchestrator defines **Protocol interfaces** and **direct call sites** that will break after the StakeCalculator and GateKeeper signature changes.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_orchestrator.py`:

```python
def test_process_race_uses_edge_for_stake_calculation():
    """Orchestrator should pass bet.edge (not ev_lower_corrected) to StakeCalculator."""
    mock_sc = MagicMock()
    mock_sc.calc_stake.return_value = 500.0
    # ... set up full orchestrator mock ...

    # After processing, verify calc_stake was called with edge parameter
    call_args = mock_sc.calc_stake.call_args
    assert "edge" in call_args.kwargs or isinstance(call_args[0][0], float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_orchestrator.py::test_process_race_uses_edge_for_stake_calculation -v`
Expected: FAIL (current code passes ev_lower_corrected)

- [ ] **Step 3: Update StakeCalculatorProtocol**

In `src/betting/orchestrator.py`, update the Protocol (~line 16-22):

```python
class StakeCalculatorProtocol(Protocol):
    def calc_stake(
        self, edge: float, odds: float, bankroll: float, bet_type: BetType
    ) -> float: ...
```

- [ ] **Step 4: Update GateKeeperProtocol**

In `src/betting/orchestrator.py`, update the Protocol (~line 31-36):

```python
class GateKeeperProtocol(Protocol):
    def filter_bets(self, bets: list[Bet], edge_threshold: float = 0.03) -> list[Bet]: ...
```

- [ ] **Step 5: Update BetStrategyProtocol**

In `src/betting/orchestrator.py`, update the Protocol (~line 46-53):

```python
class BetStrategyProtocol(Protocol):
    def generate(
        self,
        feats: dict,
        bankroll: float,
        ev_threshold: float = 1.10,
        max_bets: int = 3,
        edge_threshold: float = 0.03,
    ) -> list[dict]: ...
```

- [ ] **Step 6: Update process_race() call sites**

In `src/betting/orchestrator.py`, update `process_race()` (~lines 152-221):

1. **Parameter extraction** (~line 152-155):
```python
edge_threshold = params.get("edge_threshold", 0.03)  # NEW
# Keep ev_threshold for Win/Wide backward compatibility
```

2. **PlaceStrategy.generate() call** (~line 170):
```python
place_bets = self.place_strategy.generate(
    feats, bankroll, ev_threshold, max_bets,
    edge_threshold=edge_threshold,  # NEW
)
```

3. **GateKeeper.filter_bets() call** (~line 194):
```python
all_bets = self.gate_keeper.filter_bets(all_bets, edge_threshold=edge_threshold)
```

4. **StakeCalculator.calc_stake() call** (~line 198-203):
```python
base_stake = self.stake_calculator.calc_stake(
    bet.edge,       # CHANGED from bet.ev_lower_corrected
    bet.odds,
    bankroll,
    bet.bet_type,
)
```

- [ ] **Step 7: Update paper_trading/predictor.py**

In `src/paper_trading/predictor.py`, find the bet history logging (~line 185) and add `edge`:

```python
"edge": float(bet.edge),  # Value Betting edge
```

Keep the existing `"ev": float(bet.ev_lower_corrected)` for backward compatibility.

- [ ] **Step 8: Run all orchestrator tests**

Run: `python -m pytest tests/test_orchestrator.py tests/test_paper_predictor.py -v`
Expected: ALL PASS

- [ ] **Step 9: Commit**

```bash
git add src/betting/orchestrator.py src/paper_trading/predictor.py tests/test_orchestrator.py tests/test_paper_predictor.py
git commit -m "feat: BettingOrchestrator の Protocol/呼び出しサイトを Value Betting に更新"
```

---

## Task 7: Update BacktestEngine for edge tracking

**Files:**
- Modify: `src/backtest/engine.py` (bet creation in `run()` method)
- Test: existing tests should pass without changes (edge defaults to 0.0)

The BacktestEngine creates Bet objects at ~line 402 via `select_bets()`. Since select_bets() now returns Bets with `edge` set, the engine just needs to pass edge through to bet history. Also fix the `n_candidates` diagnostic to use `edge_place` instead of `ev_place`.

- [ ] **Step 1: Fix n_candidates diagnostic to use edge_place**

In `src/backtest/engine.py`, find the `n_candidates` calculation (~lines 369-374):

```python
# OLD (uses ev_place — will always show 0 candidates after migration):
ev_threshold = regime_params.get("ev_threshold", 1.10)
n_candidates = (
    int((result_df["ev_place"].fillna(0) >= ev_threshold).sum())
    if "ev_place" in result_df.columns
    else 0
)

# NEW (uses edge_place — correct for Value Betting):
edge_threshold = regime_params.get("edge_threshold", 0.03)
n_candidates = (
    int((result_df["edge_place"].fillna(0) >= edge_threshold).sum())
    if "edge_place" in result_df.columns
    else 0
)
```

- [ ] **Step 2: Update diag_logger.log_race() calls**

Find `diag_logger.log_race()` calls (~lines 380, 418) and update the `ev_threshold` parameter to also pass `edge_threshold`:

```python
diag_logger.log_race(
    ...,
    ev_threshold=ev_threshold,           # keep for backward compat
    edge_threshold=edge_threshold,        # NEW
    n_candidates=n_candidates,
    ...
)
```

- [ ] **Step 3: Verify edge is propagated in bet history**

In `src/backtest/engine.py`, find where bet history is recorded (~lines 480-510 in the `run()` method). Verify that `bet.edge` is included in the bet history dictionary. If the existing code uses `dataclasses.asdict(bet)` or similar, edge will be included automatically.

If the bet history is constructed manually, add `"edge": bet.edge` to the dictionary.

- [ ] **Step 4: Update ROI reporting to show edge statistics**

In the reporting section of `run()` (~lines 546-563), add edge statistics:

```python
# Edge statistics for Value Betting
if bet_history:
    edges = [b["edge"] for b in bet_history if "edge" in b]
    if edges:
        result_data["avg_edge"] = sum(edges) / len(edges)
        result_data["min_edge"] = min(edges)
        result_data["max_edge"] = max(edges)
```

- [ ] **Step 5: Run existing backtest tests**

Run: `python -m pytest tests/ -v -k "backtest or engine"`
Expected: ALL PASS

- [ ] **Step 6: (Optional) Update diagnostic_logger.py field name**

In `src/backtest/diagnostic_logger.py`, the `RaceDiagnostic` dataclass has an `ev_threshold` field. Consider adding `edge_threshold: float = 0.0` alongside it. This is cosmetic and can be deferred.

- [ ] **Step 7: Commit**

```bash
git add src/backtest/engine.py
git commit -m "feat: BacktestEngine に edge 統計を追加 (Value Betting レポート)"
```

---

## Task 8: Full test suite validation

**Files:** None (validation only)

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS — if any failures, fix before proceeding.

- [ ] **Step 2: Run linter**

Run: `ruff check src/ tests/`
Expected: No errors

- [ ] **Step 3: Run type checker**

Run: `mypy src/`
Expected: No errors (or same errors as before migration)

---

## Task 9: Validation backtest

**Files:** None (validation only)

Run a backtest with the new Value Betting logic to validate that it produces reasonable results:

```bash
# Single-year backtest (2024 test, 4-year training)
python scripts/run_backtest.py \
  --train-start 20200101 --train-end 20231231 \
  --test-start 20240101 --test-end 20241231 \
  --ensemble

# Multi-year backtest
python scripts/run_backtest.py \
  --years 2023 2024 2025 \
  --train-window 4 \
  --ensemble \
  --report
```

**Expected behavior:**
- Bets should be concentrated on favorites/popular horses (high p_place_pred, low-mid odds)
- Edge filtering should eliminate zero-edge longshots that old EV system bet on
- Total bet count may decrease (stricter criterion)
- ROI should be positive and ideally comparable to or better than the previous 209.0% result

**Comparison baseline (from memory):**
- Previous best: ROI 209.0% across 2023-2025 (4yr training, ensemble, flat, JRA-only)
- If ROI drops significantly, investigate edge thresholds (Task 2 thresholds may need tuning)

- [ ] **Step 1: Run single-year backtest (2024)**

```bash
python scripts/run_backtest.py \
  --train-start 20200101 --train-end 20231231 \
  --test-start 20240101 --test-end 20241231 \
  --ensemble
```

- [ ] **Step 2: Analyze results — check edge distribution and bet patterns**

Review the output for:
- Number of bets (compare with ~9,074 baseline)
- Average edge of bet selections
- ROI percentage
- Maximum drawdown

- [ ] **Step 3: If results are poor, tune edge thresholds**

Adjust `edge_threshold` values in `src/models/regime_detector.py`:
- If too few bets: lower AGGRESSIVE threshold from 0.03 to 0.02
- If too many losing bets: raise thresholds
- Re-run backtest after each adjustment

- [ ] **Step 4: Run multi-year backtest for final validation**

```bash
python scripts/run_backtest.py \
  --years 2023 2024 2025 \
  --train-window 4 \
  --ensemble \
  --report
```

- [ ] **Step 5: Document results and commit**

Save the backtest results to memory. Update `MEMORY.md` with the new backtest entry.

---

## Summary of Changes

| Component | Old (EV-based) | New (Value Betting) |
|-----------|---------------|---------------------|
| **Bet selection** | `ev_place_corrected >= 1.1` | `edge_place >= 0.03` |
| **Edge formula** | N/A | `p_place_pred - 1/fukuoddslow` |
| **Kelly formula** | `(ev_lower - 1) / (odds - 1)` | `(edge × odds) / (odds - 1)` |
| **Sort criterion** | `ev_place_corrected` DESC | `edge_place` DESC |
| **AGGRESSIVE threshold** | EV ≥ 1.10 | Edge ≥ 0.03 |
| **CONSERVATIVE threshold** | EV ≥ 1.30 | Edge ≥ 0.05 |
| **COLLAPSED threshold** | EV ≥ 1.50 | Edge ≥ 0.08 |
| **StakeCalculator guard** | `ev_lower < 1.05` | `edge < 0.005` |
| **Depends on e_return?** | Yes (EV = p × e_return) | **No** (edge = p - p_market) |

### What stays the same:
- PlaceTwoStageModel (HIT_FEATURE_COLS / RETURN_FEATURE_COLS separation kept)
- PlaceEVCorrectionModel (still runs, output available but unused for selection)
- RobustConfidenceEstimator (still runs for monitoring, unused for selection)
- Win and Wide strategies (still use EV, can be migrated later)
- Feature engineering, data pipeline, ETL — all unchanged

### Known tech debt (do NOT address in this plan):
- `engine.py` has a legacy `_generate_bets()` method (~lines 613-650) that still uses EV-based logic. It delegates to `RacePredictor.select_bets()` which is now edge-based, so it works — but the method body comments reference EV. Leave as-is; it's dead code.
- `diagnostic_logger.py` has `ev_threshold` field name. Add `edge_threshold` field in a future cleanup.
