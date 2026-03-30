# Ablation Results (2026-03-30)

## p_ability_win Leakage Ablation

Removed `p_ability_win` from FEATURE_COLS in both `PlaceAbilityModel` and `WinTwoStageModel` (which `PlaceTwoStageModel` inherits). The column still exists in the DataFrame for post-processing constraints but is not used as an input feature for any model.

### Results

| Pattern | Stage1 | p_ability_win | ROI | Bets | Stake | Return | Max DD |
|---------|--------|--------------|-----|------|-------|--------|--------|
| C (current) | 30 cols | in-sample | 143.3% | 6,134 | 613,400 | 879,110 | 3.4% |
| B (ablation) | 29 cols | removed | 54.3% | 2,188 | 218,800 | 118,820 | 100.0% |
| D (OOF fix) | 30 cols | OOF (3-fold) | 136.5% | 6,045 | 604,500 | 825,180 | 4.2% |

### Interpretation

Removing `p_ability_win` from downstream model features caused:

- **ROI: 143.3% -> 54.3%** (-89.0 percentage points)
- **Bets: 6,134 -> 2,188** (-64% fewer bets placed)
- **Max drawdown: 3.4% -> 100.0%** (near-total bankroll wipeout)
- **Final bankroll: 365,710 -> 20 yen** (starting from 1,000 yen)

The p_ability_win feature (Stage1 ability model output, computed with full in-sample labels) leaks future information into the cascade. The downstream PlaceAbilityModel and WinTwoStageModel/PlaceTwoStageModel use it as their strongest feature, inflating both hit-rate and EV estimates.

### OOF Fix Results

Replacing in-sample `p_ability_win` with 3-fold expanding window OOF predictions:

- **ROI: 143.3% -> 136.5%** (-6.8 percentage points from in-sample, well above 100%)
- **Bets: 6,134 -> 6,045** (comparable bet volume)
- **Max drawdown: 3.4% -> 4.2%** (slightly higher but still very controlled)
- **Final bankroll: 365,710 -> 320,680 yen** (starting from 1,000 yen)
- **Train time: 989s** (~16.5 min, vs 17 min in-sample)

### Verdict: PASS (Features valid, production-ready)

OOF ROI of 136.5% exceeds the 110% threshold by a wide margin. The cascade architecture is validated:
the model cascade with OOF predictions preserves most of the predictive power while eliminating
in-sample leakage. The 6.8pp ROI drop from in-sample is a reasonable cost for leakage elimination.

### Files Modified

- `src/models/stage1_ability_model.py`: added `train_oof()` method (K-fold expanding window)
- `src/pipelines/training_pipeline.py`: `_train_submodel()` uses `train_oof()` instead of `train()` + `add_ability_probs()`
- `src/models/place_ability_model.py`: fixed numpy index-out-of-bounds with non-contiguous indices
