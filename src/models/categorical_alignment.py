"""LightGBM inference-time categorical metadata alignment."""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd


def align_lightgbm_categories(
    features: pd.DataFrame,
    model: lgb.Booster | object,
) -> pd.DataFrame:
    """Align category vocabularies with those recorded by a LightGBM model.

    LightGBM requires both the categorical column count and each column's
    category vocabulary to match training.  ``pandas_categorical`` stores the
    vocabularies but not their column names, so names are recovered from the
    category-typed input columns in model feature order.
    """
    if not isinstance(model, lgb.Booster):
        return features

    vocabularies = getattr(model, "pandas_categorical", None)
    if not vocabularies:
        return features

    model_feature_names = model.feature_name()
    missing = [col for col in model_feature_names if col not in features.columns]
    if missing:
        raise ValueError(
            "LightGBM feature metadata mismatch: "
            f"missing model features={missing[:5]}"
        )

    # FEATURE_COLS can evolve after a model has been trained. LightGBM checks
    # every category-typed column in the input, including columns unknown to
    # the saved model, so select and order the exact training feature set first.
    result = features[model_feature_names].copy()
    categorical_columns = [
        col
        for col in model_feature_names
        if isinstance(result[col].dtype, pd.CategoricalDtype)
    ]
    if len(categorical_columns) != len(vocabularies):
        raise ValueError(
            "LightGBM categorical metadata mismatch: "
            f"input category columns={categorical_columns}, "
            f"stored vocabularies={len(vocabularies)}"
        )

    for col, vocabulary in zip(categorical_columns, vocabularies, strict=True):
        result[col] = pd.Categorical(result[col], categories=vocabulary)
    return result
