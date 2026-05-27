"""InvestmentFeatureSpec frozen dataclass + FEATURE_SPECS dict のテスト (38-01 Task 1)

TDD RED: 以下テストは実装前に実行すると全て失敗する。
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from domain.types import POST_RACE_COLS
from investment.schema_registry import (
    ALL_IF_COLUMNS,
    CATEGORY_ORDER,
    FEATURE_SPECS,
    InvestmentFeatureSpec,
)


class TestInvestmentFeatureSpecFrozen:
    """Test 1: InvestmentFeatureSpec is frozen."""

    def test_frozen_raises_on_assignment(self) -> None:
        spec = InvestmentFeatureSpec(
            name="if_test",
            category="model_prob",
            dtype="float64",
            train_sources=("p_win_oof",),
            infer_sources=("p_win_pred",),
            required=True,
            default_value=None,
            missing_indicator=None,
            leakage_class="safe",
            description="test spec",
        )
        with pytest.raises(FrozenInstanceError):
            spec.name = "if_modified"  # type: ignore[misc]


class TestFeatureSpecsCategories:
    """Test 2: FEATURE_SPECS contains entries for all 9 categories."""

    EXPECTED_CATEGORIES = (
        "model_prob",
        "market_prob",
        "model_market_gap",
        "race_relative",
        "odds_band",
        "late_odds",
        "ability_form",
        "course_pace",
        "uncertainty",
    )

    def test_feature_specs_has_all_categories(self) -> None:
        categories = {spec.category for spec in FEATURE_SPECS.values()}
        for cat in self.EXPECTED_CATEGORIES:
            assert cat in categories, f"Missing category: {cat}"

    def test_feature_specs_count_in_range(self) -> None:
        """90-130 specs total."""
        count = len(FEATURE_SPECS)
        assert 90 <= count <= 130, f"Expected 90-130 specs, got {count}"

    def test_category_column_counts(self) -> None:
        """Each category has the expected number of specs per D-05."""
        cat_counts: dict[str, int] = {}
        for spec in FEATURE_SPECS.values():
            cat_counts[spec.category] = cat_counts.get(spec.category, 0) + 1

        expected_ranges = {
            "model_prob": (8, 12),
            "market_prob": (6, 10),
            "model_market_gap": (10, 16),
            "race_relative": (12, 18),
            "odds_band": (6, 10),
            "late_odds": (8, 12),
            "ability_form": (15, 25),
            "course_pace": (10, 18),
            "uncertainty": (10, 16),
        }
        for cat, (lo, hi) in expected_ranges.items():
            count = cat_counts.get(cat, 0)
            assert lo <= count <= hi, (
                f"Category '{cat}' has {count} specs, expected {lo}-{hi}"
            )


class TestFeatureSpecsUniqueNames:
    """Test 3: Every spec has unique name starting with 'if_'."""

    def test_all_names_start_with_if_prefix(self) -> None:
        for spec in FEATURE_SPECS.values():
            assert spec.name.startswith("if_"), (
                f"Spec name '{spec.name}' does not start with 'if_'"
            )

    def test_all_names_unique(self) -> None:
        names = [spec.name for spec in FEATURE_SPECS.values()]
        assert len(names) == len(set(names)), "Duplicate spec names found"


class TestCategoryOrder:
    """Test 4: CATEGORY_ORDER lists all 9 categories in correct order."""

    EXPECTED_ORDER = (
        "model_prob",
        "market_prob",
        "model_market_gap",
        "race_relative",
        "odds_band",
        "late_odds",
        "ability_form",
        "course_pace",
        "uncertainty",
    )

    def test_category_order_has_9_entries(self) -> None:
        assert len(CATEGORY_ORDER) == 9

    def test_category_order_matches_expected(self) -> None:
        assert CATEGORY_ORDER == self.EXPECTED_ORDER


class TestNoPostRaceOverlap:
    """Test 5: No spec has a name that appears in POST_RACE_COLS."""

    def test_no_spec_name_in_post_race_cols(self) -> None:
        post_race_set = set(POST_RACE_COLS)
        for spec in FEATURE_SPECS.values():
            assert spec.name not in post_race_set, (
                f"Spec '{spec.name}' overlaps with POST_RACE_COLS"
            )


class TestAllIfColumns:
    """Test 6: ALL_IF_COLUMNS returns deduplicated list matching all spec names."""

    def test_all_if_columns_matches_specs(self) -> None:
        spec_names = [spec.name for spec in FEATURE_SPECS.values()]
        assert len(ALL_IF_COLUMNS) == len(set(spec_names))
        assert set(ALL_IF_COLUMNS) == set(spec_names)

    def test_all_if_columns_follows_category_order(self) -> None:
        """ALL_IF_COLUMNS follows CATEGORY_ORDER then spec definition order."""
        expected_order: list[str] = []
        for cat in CATEGORY_ORDER:
            cat_specs = [
                s for s in FEATURE_SPECS.values() if s.category == cat
            ]
            expected_order.extend(s.name for s in cat_specs)
        assert ALL_IF_COLUMNS == expected_order


class TestSpecMetadata:
    """Test 7: Required features have default_value=None and missing_indicator=None;
    optional features have non-None default_value."""

    def test_required_features_have_no_defaults(self) -> None:
        for spec in FEATURE_SPECS.values():
            if spec.required:
                assert spec.default_value is None, (
                    f"Required spec '{spec.name}' has default_value={spec.default_value}"
                )
                assert spec.missing_indicator is None, (
                    f"Required spec '{spec.name}' has missing_indicator={spec.missing_indicator}"
                )

    def test_optional_features_have_defaults(self) -> None:
        for spec in FEATURE_SPECS.values():
            if not spec.required:
                assert spec.default_value is not None, (
                    f"Optional spec '{spec.name}' has default_value=None"
                )

    def test_all_specs_have_10_fields(self) -> None:
        """InvestmentFeatureSpec has exactly 10 fields per D-16."""
        spec = next(iter(FEATURE_SPECS.values()))
        from dataclasses import fields

        field_names = {f.name for f in fields(spec)}
        expected = {
            "name",
            "category",
            "dtype",
            "train_sources",
            "infer_sources",
            "required",
            "default_value",
            "missing_indicator",
            "leakage_class",
            "description",
        }
        assert field_names == expected


class TestInitExports:
    """Test: __init__.py exports the correct public API."""

    def test_init_exports(self) -> None:
        import investment

        assert hasattr(investment, "InvestmentFeatureSpec")
        assert hasattr(investment, "FEATURE_SPECS")
        assert hasattr(investment, "CATEGORY_ORDER")
        assert hasattr(investment, "ALL_IF_COLUMNS")
