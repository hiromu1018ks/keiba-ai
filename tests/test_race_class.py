from __future__ import annotations

import pandas as pd
import pytest

from features.race_class import compute_race_class_features


def test_effective_jyoken_uses_populated_age_specific_column() -> None:
    df = pd.DataFrame(
        {
            "race_date": pd.to_datetime(["2024-06-08", "2024-06-09", "2024-10-01"]),
            "gradecd": ["", "", ""],
            "jyokencd1": ["000", "000", "000"],
            "jyokencd2": ["703", "005", "000"],
            "jyokencd3": ["000", "005", "999"],
            "jyokencd4": ["000", "005", "999"],
            "jyokencd5": ["703", "005", "999"],
            "honsyokin1": [55_000, 80_000, 220_000],
        }
    )

    result = compute_race_class_features(df)

    assert result["effective_jyokencd"].tolist() == ["703", "005", "999"]
    assert result["class_level_current"].tolist() == [1.0, 2.0, 5.0]
    assert result["class_level_source_flag"].tolist() == [1.0, 1.0, 1.0]
    assert result["class_regime_after_202406"].tolist() == [1.0, 1.0, 1.0]


def test_grade_overrides_condition_code() -> None:
    df = pd.DataFrame(
        {
            "gradecd": ["A", "B", "C", "L", "E"],
            "jyokencd5": ["999", "999", "999", "999", "016"],
        }
    )

    result = compute_race_class_features(df)

    assert result["class_level_current"].tolist() == pytest.approx([8.0, 7.0, 6.0, 5.5, 5.0])
    assert result["class_level_source_flag"].tolist() == [3.0] * 5


def test_000_to_999_does_not_become_raw_999_jump() -> None:
    df = pd.DataFrame(
        {
            "gradecd": ["", ""],
            "jyokencd1": ["000", "000"],
            "jyokencd5": ["005", "999"],
        }
    )

    result = compute_race_class_features(df)

    assert result["class_level_current"].tolist() == [2.0, 5.0]
    assert result["class_level_current"].diff().iloc[1] == 3.0
