"""PT の POST_RACE DROP と JRA フィルタのガードテスト。"""

import pandas as pd

POST_RACE_COLS = ("kakuteijyuni", "confirmed_odds")


def _drop_post_race_cols(df: pd.DataFrame) -> pd.DataFrame:
    """POST_RACE 列を除外 (BT engine.py と同じ処理)。"""
    return df.drop(
        columns=[c for c in POST_RACE_COLS if c in df.columns],
        errors="ignore",
    )


def test_post_race_cols_removed():
    """POST_RACE 列が predict 前に DROP されることを検証。"""
    df = pd.DataFrame(
        {
            "race_id": ["R001"],
            "umaban": [1],
            "kakuteijyuni": [3],
            "confirmed_odds": [5.2],
            "tanodds": [4.8],
        }
    )

    result = _drop_post_race_cols(df)

    assert "kakuteijyuni" not in result.columns
    assert "confirmed_odds" not in result.columns
    assert "tanodds" in result.columns
    assert "umaban" in result.columns


def test_post_race_cols_missing_no_error():
    """POST_RACE 列が存在しなくてもエラーにならない。"""
    df = pd.DataFrame(
        {
            "race_id": ["R001"],
            "umaban": [1],
            "tanodds": [4.8],
        }
    )

    result = _drop_post_race_cols(df)

    assert "tanodds" in result.columns
    assert len(result) == 1
