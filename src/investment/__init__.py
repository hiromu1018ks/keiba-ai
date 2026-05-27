"""Investment feature frame package.

Public API exports for schema registry, leakage detection, manifest,
cache, and feature frame builder.
"""

from investment.cache import InvestmentFrameCache
from investment.feature_frame import (
    InvestmentFeatureFrameBuilder,
    build_frame,
)
from investment.leakage import (
    IN_SAMPLE_ONLY_COLS,
    validate_no_post_race_leakage,
    validate_oof_safe_sources,
    validate_schema_identity,
)
from investment.manifest import (
    compute_investment_schema_hash,
    generate_investment_manifest,
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
    "InvestmentFeatureFrameBuilder",
    "InvestmentFeatureSpec",
    "InvestmentFrameCache",
    "build_frame",
    "compute_investment_schema_hash",
    "generate_investment_manifest",
    "validate_no_post_race_leakage",
    "validate_oof_safe_sources",
    "validate_schema_identity",
]
