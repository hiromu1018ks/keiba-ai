"""Race class normalization utilities.

EveryDB2 stores race condition in several age-specific ``jyokencd`` columns.
Using one raw column as an ordinal feature breaks around class-system changes,
so this module centralizes the conversion to stable numeric features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GRADE_LEVEL_MAP: dict[str, float] = {
    "A": 8.0,  # G1
    "B": 7.0,  # G2
    "C": 6.0,  # G3
    "D": 5.5,  # non-graded stakes
    "L": 5.5,  # Listed
    "E": 5.0,  # Open/special
    "G": 5.5,  # jump graded/listed-like codes in JV data
    "H": 5.0,
}

SOURCE_MISSING = 0.0
SOURCE_JYOKEN = 1.0
SOURCE_PRIZE = 2.0
SOURCE_GRADE = 3.0


def _to_code(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        if "." in text:
            num = float(text)
            if np.isfinite(num) and num.is_integer():
                text = str(int(num))
    except ValueError:
        pass
    if text.isdigit():
        return text.zfill(3)
    return text


def class_level_from_values(
    grade_code: object,
    jyoken_code: object,
    honsyokin1: object | None = None,
) -> float:
    """Return a stable class level from grade and condition codes.

    Scale:
      1 = debut/maiden, 2 = 1-win, 3 = 2-win, 4 = 3-win,
      5 = open, 5.5 = listed/non-graded stakes, 6/7/8 = G3/G2/G1.
    """
    grade = str(grade_code).strip() if pd.notna(grade_code) else ""
    if grade in GRADE_LEVEL_MAP:
        return GRADE_LEVEL_MAP[grade]

    code = _to_code(jyoken_code)
    if code in {"701", "702", "703"}:
        return 1.0
    if code == "999":
        return 5.0
    if code and code != "000":
        numeric = pd.to_numeric(pd.Series([code]), errors="coerce").iloc[0]
        if pd.notna(numeric):
            value = float(numeric)
            if 1.0 <= value <= 5.0:
                return 2.0
            if 6.0 <= value <= 10.0:
                return 3.0
            if 11.0 <= value <= 16.0:
                return 4.0
            if value > 16.0:
                return 4.5

    if honsyokin1 is not None:
        prize = pd.to_numeric(pd.Series([honsyokin1]), errors="coerce").iloc[0]
        if pd.notna(prize) and float(prize) > 0:
            prize_val = float(prize)
            if prize_val < 70_000:
                return 1.0
            if prize_val < 100_000:
                return 2.0
            if prize_val < 160_000:
                return 3.0
            if prize_val < 220_000:
                return 4.0
            return 5.0

    return float("nan")


def effective_jyoken_code(df: pd.DataFrame) -> pd.Series:
    """Select the active race condition code from ``jyokencd1`` ... ``jyokencd5``.

    ``jyokencd5`` is the youngest eligible condition and is populated with the
    effective code for the JRA flat races used here. If it is blank/000, fall
    back through age-specific columns.
    """
    if df.empty:
        return pd.Series(dtype=object)
    result = pd.Series("000", index=df.index, dtype=object)
    priority = ["jyokencd5", "jyokencd4", "jyokencd3", "jyokencd2", "jyokencd1"]
    for col in priority:
        if col not in df.columns:
            continue
        codes = df[col].map(_to_code)
        mask = result.eq("000") & codes.ne("") & codes.ne("000")
        result.loc[mask] = codes.loc[mask]
    return result


def compute_race_class_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add stable class features to ``df`` and return it."""
    result = df.copy()
    if "effective_jyokencd" not in result.columns:
        result["effective_jyokencd"] = effective_jyoken_code(result)

    grade_source = (
        result["grade_code"]
        if "grade_code" in result.columns
        else result.get("gradecd", pd.Series("", index=result.index))
    )
    prize_source = result.get("honsyokin1", pd.Series(np.nan, index=result.index))

    levels = [
        class_level_from_values(grade, jyoken, prize)
        for grade, jyoken, prize in zip(grade_source, result["effective_jyokencd"], prize_source)
    ]
    result["class_level_current"] = pd.Series(levels, index=result.index, dtype=float)

    grade_text = grade_source.fillna("").astype(str).str.strip()
    has_grade = grade_text.isin(GRADE_LEVEL_MAP)
    has_jyoken = result["effective_jyokencd"].ne("000")
    has_prize = pd.to_numeric(prize_source, errors="coerce").fillna(0) > 0
    source = pd.Series(SOURCE_MISSING, index=result.index, dtype=float)
    source = source.mask(has_prize, SOURCE_PRIZE)
    source = source.mask(has_jyoken, SOURCE_JYOKEN)
    source = source.mask(has_grade, SOURCE_GRADE)
    result["class_level_source_flag"] = source

    bins = [-np.inf, 1.5, 2.5, 3.5, 4.5, 5.25, 5.75, 6.5, 7.5, np.inf]
    labels = [
        "maiden",
        "one_win",
        "two_win",
        "three_win",
        "open",
        "listed",
        "g3",
        "g2",
        "g1",
    ]
    result["class_bucket"] = pd.cut(
        result["class_level_current"],
        bins=bins,
        labels=labels,
    ).astype("category")
    if "race_date" in result.columns:
        race_date = pd.to_datetime(result["race_date"], errors="coerce")
        result["class_regime_after_202406"] = (
            race_date >= pd.Timestamp("2024-06-01")
        ).astype(float)
    else:
        result["class_regime_after_202406"] = 0.0
    return result
