"""Investment feature frame package.

Public API exports for schema registry and leakage detection.
"""

from investment.leakage import (
    IN_SAMPLE_ONLY_COLS,
    validate_no_post_race_leakage,
    validate_oof_safe_sources,
    validate_schema_identity,
)
from investment.schema_registry import (
    ALL_IF_COLUMNS,
    CATEGORY_ORDER,
    FEATURE_SPECS,
    InvestmentFeatureSpec,
)

__all__ = [
    "ALL_IF_COLUMNS",
    "CATEGORY_ORDER",
    "FEATURE_SPECS",
    "IN_SAMPLE_ONLY_COLS",
    "InvestmentFeatureSpec",
    "validate_no_post_race_leakage",
    "validate_oof_safe_sources",
    "validate_schema_identity",
]
