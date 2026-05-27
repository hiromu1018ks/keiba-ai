"""Investment feature frame package.

Public API exports for schema registry and leakage detection.
"""

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
    "InvestmentFeatureSpec",
]
