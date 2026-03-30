# Ablation Results (2026-03-30)

## p_ability_win Leakage Ablation

Removed `p_ability_win` from FEATURE_COLS in both `PlaceAbilityModel` and `WinTwoStageModel` (which `PlaceTwoStageModel` inherits). The column still exists in the DataFrame for post-processing constraints but is not used as an input feature for any model.

### Results

| Pattern | Stage1 | p_ability_win | ROI | Bets | Stake | Return | Max DD |
|---------|--------|--------------|-----|------|-------|--------|--------|
| C (current) | 30 cols | in-sample | 143.3% | 6,134 | 613,400 | 879,110 | 3.4% |
| B (ablation) | 29 cols | removed | 54.3% | 2,188 | 218,800 | 118,820 | 100.0% |

### Interpretation

Removing `p_ability_win` from downstream model features caused:

- **ROI: 143.3% -> 54.3%** (-89.0 percentage points)
- **Bets: 6,134 -> 2,188** (-64% fewer bets placed)
- **Max drawdown: 3.4% -> 100.0%** (near-total bankroll wipeout)
- **Final bankroll: 365,710 -> 20 yen** (starting from 1,000 yen)

The p_ability_win feature (Stage1 ability model output, computed with full in-sample labels) leaks future information into the cascade. The downstream PlaceAbilityModel and WinTwoStageModel/PlaceTwoStageModel use it as their strongest feature, inflating both hit-rate and EV estimates.

### Files Modified (then reverted)

- `src/models/place_ability_model.py` line 60: commented out `"p_ability_win"`
- `src/models/two_stage_return_model.py` line 26: commented out `"p_ability_win"`

### Next Steps

Replace in-sample `p_ability_win` with out-of-fold (OOF) predictions to eliminate cascade leakage while preserving the model cascade architecture.
