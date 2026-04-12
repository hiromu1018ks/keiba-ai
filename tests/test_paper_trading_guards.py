"""PT の POST_RACE DROP と JRA フィルタのガードテスト。"""

import pandas as pd

POST_RACE_COLS = (
    "kakuteijyuni",
    "confirmed_odds",
    "ninki",
    "kyakusitukubun",
    "time",
    "timediff",
    "harontimel3",
    "harontimel4",
    "jyuni1c",
    "jyuni2c",
    "jyuni3c",
    "jyuni4c",
    "honsyokin",
    "chakusacd",
    "dmjyuni",
    "dmtime",
)


def _drop_post_race_cols(df: pd.DataFrame) -> pd.DataFrame:
    """POST_RACE 列を除外 (BT engine.py と同じ処理)。"""
    return df.drop(
        columns=[c for c in POST_RACE_COLS if c in df.columns],
        errors="ignore",
    )


def _apply_jra_filter(feat_df: pd.DataFrame) -> pd.DataFrame:
    """JRAフィルタ: NARレース (jyocd >= 30) を除外 (BT engine.py と同じ処理)。"""
    if "jyocd" not in feat_df.columns:
        return feat_df
    jyocd_int = pd.to_numeric(feat_df["jyocd"], errors="coerce")
    return feat_df[jyocd_int.between(1, 10)]


def test_post_race_cols_removed():
    """POST_RACE 列が predict 前に DROP されることを検証。"""
    df = pd.DataFrame(
        {
            "race_id": ["R001"],
            "umaban": [1],
            "kakuteijyuni": [3],
            "confirmed_odds": [5.2],
            "ninki": [5],
            "kyakusitukubun": [2],
            "time": [65.3],
            "timediff": [0.5],
            "harontimel3": [34.2],
            "honsyokin": [500],
            "tanodds": [4.8],
        }
    )

    result = _drop_post_race_cols(df)

    assert "kakuteijyuni" not in result.columns
    assert "confirmed_odds" not in result.columns
    assert "ninki" not in result.columns
    assert "kyakusitukubun" not in result.columns
    assert "time" not in result.columns
    assert "timediff" not in result.columns
    assert "harontimel3" not in result.columns
    assert "honsyokin" not in result.columns
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


def test_jra_filter_removes_nar():
    """jyocd >= 30 の NAR レースが除外される。"""
    df = pd.DataFrame(
        {
            "race_id": ["R001", "R002", "R003"],
            "jyocd": [5, 30, 8],
            "umaban": [1, 2, 3],
        }
    )

    result = _apply_jra_filter(df)

    assert len(result) == 2
    assert set(result["race_id"]) == {"R001", "R003"}


def test_jra_filter_preserves_all_jra():
    """jyocd 1-10 は全て保持される。"""
    df = pd.DataFrame(
        {
            "race_id": [f"R{i:03d}" for i in range(10)],
            "jyocd": list(range(1, 11)),
            "umaban": [1] * 10,
        }
    )

    result = _apply_jra_filter(df)

    assert len(result) == 10


def test_jra_filter_handles_missing_jyocd():
    """jyocd 列がない場合はフィルタをスキップ。"""
    df = pd.DataFrame(
        {
            "race_id": ["R001"],
            "umaban": [1],
        }
    )

    _apply_jra_filter(df)

    assert len(df) == 1
