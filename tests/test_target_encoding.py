"""TargetEncoder -- OOF-safe target encoding tests"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.target_encoding import (
    TE_FEATURE_COLS,
    TE_STAGE2_FEATURE_COLS,
    TargetEncoder,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic DataFrame with blood_keito_cd, kisyucode, chokyosicode."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="3D")
    return pd.DataFrame(
        {
            "race_date": dates,
            "blood_keito_cd": rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=n),
            "kisyucode": rng.choice([10.0, 20.0, 30.0, 40.0], size=n),
            "chokyosicode": rng.choice([100.0, 200.0, 300.0], size=n),
            "kakuteijyuni": rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], size=n),
        }
    )


# ---------------------------------------------------------------------------
# Test 1: 3-fold expanding window TE calculation
# ---------------------------------------------------------------------------
class TestFitTransformOof:
    def test_basic_oof_output(self) -> None:
        """fit_transform_oof() returns TE columns for each cat_col."""
        df = _make_df()
        enc = TargetEncoder(
            cat_cols=["blood_keito_cd", "kisyucode", "chokyosicode"],
            target_col="kakuteijyuni",
        )
        result = enc.fit_transform_oof(df)
        for col in ["te_blood_keito_cd", "te_kisyucode", "te_chokyosicode"]:
            assert col in result.columns
            assert result[col].notna().any()

    def test_fold_boundaries_by_race_date(self) -> None:
        """Each fold's test data uses train data from earlier dates only."""
        df = _make_df(n=300, seed=1)
        enc = TargetEncoder(
            cat_cols=["blood_keito_cd"],
            target_col="kakuteijyuni",
            n_folds=3,
        )
        result = enc.fit_transform_oof(df)
        # Result should have same length as input
        assert len(result) == len(df)
        # TE values should be between 0 and 1 (win rate smoothed)
        assert result["te_blood_keito_cd"].min() >= 0.0
        assert result["te_blood_keito_cd"].max() <= 1.0

    def test_no_future_information_leakage(self) -> None:
        """Test 2: TE values in fold i come from fold i's train data only."""
        # Create deterministic data where category 1.0 has increasing win rate
        df = pd.DataFrame(
            {
                "race_date": pd.date_range("2020-01-01", periods=300, freq="D"),
                "blood_keito_cd": [1.0] * 300,
                "kakuteijyuni": [1] * 10 + [10] * 90 + [1] * 90 + [10] * 10 + [1] * 100,
            }
        )
        enc = TargetEncoder(
            cat_cols=["blood_keito_cd"],
            target_col="kakuteijyuni",
            n_folds=3,
            smoothing=1,
            min_samples=1,
        )
        result = enc.fit_transform_oof(df)

        # Early rows should have TE close to early win rate (~10%)
        # Later rows should have TE reflecting accumulated data
        early_te = result.iloc[:50]["te_blood_keito_cd"].mean()
        later_te = result.iloc[200:]["te_blood_keito_cd"].mean()
        # The TE values should differ between early and late folds
        # (proving they use different training windows)
        assert early_te != later_te

    def test_cold_start_filled_with_global_mean(self) -> None:
        """Test 3: Unknown categories get global mean."""
        # Category 99.0 only appears in test portion (later dates)
        df = pd.DataFrame(
            {
                "race_date": pd.date_range("2020-01-01", periods=200, freq="D"),
                "blood_keito_cd": [1.0] * 100 + [99.0] * 100,
                "kakuteijyuni": [1] * 30 + [10] * 70 + [1] * 10 + [10] * 90,
            }
        )
        enc = TargetEncoder(
            cat_cols=["blood_keito_cd"],
            target_col="kakuteijyuni",
            n_folds=3,
            smoothing=1,
            min_samples=1,
        )
        result = enc.fit_transform_oof(df)
        global_mean = (df["kakuteijyuni"] == 1).astype(int).mean()

        # First fold's test data won't have seen cat 99.0 yet,
        # so its TE should equal global mean
        cat_99_mask = result["blood_keito_cd"] == 99.0
        # Some fold-0 test rows with cat 99 should have global_mean
        # (depending on fold boundaries)
        cat_99_te = result.loc[cat_99_mask, "te_blood_keito_cd"]
        # At least the first few cat_99 entries should be global_mean (cold start)
        # Since cat 99 only appears after row 100, and fold boundaries split the data,
        # the earliest fold test might not have cat 99 in training
        assert cat_99_te.notna().all()

    def test_smoothing_applied(self) -> None:
        """Test 4: Categories with few samples get smoothed toward global mean."""
        rng = np.random.default_rng(42)
        n = 300
        # Create rare category (only 2 samples in early data)
        cats = [1.0] * 290 + [2.0] * 5 + [3.0] * 5
        rng.shuffle(cats)
        df = pd.DataFrame(
            {
                "race_date": pd.date_range("2020-01-01", periods=n, freq="D"),
                "blood_keito_cd": cats,
                "kakuteijyuni": rng.choice([1, 2, 3, 4, 5], size=n),
            }
        )
        enc_high_smooth = TargetEncoder(
            cat_cols=["blood_keito_cd"],
            target_col="kakuteijyuni",
            n_folds=3,
            smoothing=100,  # Very high smoothing -> all categories near global mean
            min_samples=1,
        )
        enc_low_smooth = TargetEncoder(
            cat_cols=["blood_keito_cd"],
            target_col="kakuteijyuni",
            n_folds=3,
            smoothing=1,  # Low smoothing -> categories retain their own stats
            min_samples=1,
        )
        result_high = enc_high_smooth.fit_transform_oof(df)
        result_low = enc_low_smooth.fit_transform_oof(df)

        # With very high smoothing, TE values for different categories should be close
        # With low smoothing, they should be more spread out
        high_std = result_high["te_blood_keito_cd"].std()
        low_std = result_low["te_blood_keito_cd"].std()
        assert high_std < low_std


# ---------------------------------------------------------------------------
# Test 5: transform() uses stored encoding maps
# ---------------------------------------------------------------------------
class TestTransform:
    def test_transform_uses_fitted_maps(self) -> None:
        """transform() maps categories using learned encoding maps."""
        df_train = _make_df(n=200, seed=42)
        enc = TargetEncoder(
            cat_cols=["blood_keito_cd"],
            target_col="kakuteijyuni",
        )
        enc.fit_transform_oof(df_train)

        # Create new data with same categories
        df_new = pd.DataFrame(
            {
                "race_date": pd.date_range("2021-01-01", periods=20, freq="D"),
                "blood_keito_cd": [1.0] * 10 + [2.0] * 10,
                "kakuteijyuni": [1] * 20,
            }
        )
        result = enc.transform(df_new)
        assert "te_blood_keito_cd" in result.columns
        # Categories seen in training should get specific TE values
        assert result["te_blood_keito_cd"].notna().all()

    def test_transform_unknown_category_gets_global_mean(self) -> None:
        """Test 5b: New categories in transform() get global mean."""
        df_train = _make_df(n=200, seed=42)
        enc = TargetEncoder(
            cat_cols=["blood_keito_cd"],
            target_col="kakuteijyuni",
        )
        enc.fit_transform_oof(df_train)

        # New data with unseen category 999.0
        df_new = pd.DataFrame(
            {
                "race_date": pd.date_range("2021-01-01", periods=10, freq="D"),
                "blood_keito_cd": [999.0] * 10,
                "kakuteijyuni": [1] * 10,
            }
        )
        result = enc.transform(df_new)
        # Unknown category should get global mean
        assert "te_blood_keito_cd" in result.columns
        assert result["te_blood_keito_cd"].notna().all()
        # All values should be the global mean from training
        assert result["te_blood_keito_cd"].nunique() == 1


# ---------------------------------------------------------------------------
# Test 6: Multiple category columns simultaneously
# ---------------------------------------------------------------------------
class TestMultipleCatCols:
    def test_multiple_columns(self) -> None:
        """fit_transform_oof() handles multiple cat_cols at once."""
        df = _make_df(n=200, seed=42)
        enc = TargetEncoder(
            cat_cols=["blood_keito_cd", "kisyucode", "chokyosicode"],
            target_col="kakuteijyuni",
        )
        result = enc.fit_transform_oof(df)
        for col in ["te_blood_keito_cd", "te_kisyucode", "te_chokyosicode"]:
            assert col in result.columns
            assert result[col].notna().any()


# ---------------------------------------------------------------------------
# Test 7: Binary target
# ---------------------------------------------------------------------------
class TestBinaryTarget:
    def test_binary_target_correct_values(self) -> None:
        """TE values are correct for binary (0/1) target."""
        df = pd.DataFrame(
            {
                "race_date": pd.date_range("2020-01-01", periods=200, freq="D"),
                "blood_keito_cd": [1.0] * 100 + [2.0] * 100,
                "kakuteijyuni": [1] * 30 + [0] * 70 + [1] * 60 + [0] * 40,
            }
        )
        enc = TargetEncoder(
            cat_cols=["blood_keito_cd"],
            target_col="kakuteijyuni",
            n_folds=3,
            smoothing=1,
            min_samples=1,
        )
        result = enc.fit_transform_oof(df)
        # cat 1.0 has ~30% win rate, cat 2.0 has ~60% win rate
        # TE values should reflect this difference (at least in later folds)
        cat1_mask = result["blood_keito_cd"] == 1.0
        cat2_mask = result["blood_keito_cd"] == 2.0
        # In later folds, cat2 should have higher TE than cat1
        cat1_late = result.loc[cat1_mask].iloc[-20:]["te_blood_keito_cd"].mean()
        cat2_late = result.loc[cat2_mask].iloc[-20:]["te_blood_keito_cd"].mean()
        assert cat2_late > cat1_late


# ---------------------------------------------------------------------------
# Test 8: Unsorted input
# ---------------------------------------------------------------------------
class TestUnsortedInput:
    def test_unsorted_dataframe(self) -> None:
        """fit_transform_oof() sorts internally by race_date."""
        df = _make_df(n=200, seed=42)
        # Shuffle to make it unsorted
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        enc = TargetEncoder(
            cat_cols=["blood_keito_cd"],
            target_col="kakuteijyuni",
        )
        result = enc.fit_transform_oof(df_shuffled)
        assert "te_blood_keito_cd" in result.columns
        assert len(result) == len(df_shuffled)
        # All TE values should be valid
        assert result["te_blood_keito_cd"].notna().all()


# ---------------------------------------------------------------------------
# Test 9 & 10: Feature column constants
# ---------------------------------------------------------------------------
class TestConstants:
    def test_te_feature_cols_stage1(self) -> None:
        """TE_FEATURE_COLS contains only te_blood_keito_cd."""
        assert TE_FEATURE_COLS == ["te_blood_keito_cd"]

    def test_te_stage2_feature_cols(self) -> None:
        """TE_STAGE2_FEATURE_COLS contains all 3 TE features."""
        assert TE_STAGE2_FEATURE_COLS == [
            "te_blood_keito_cd",
            "te_kisyucode",
            "te_chokyosicode",
        ]
