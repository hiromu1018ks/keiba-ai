"""src/features/leakage_validators.py のテスト"""

import pandas as pd
import pytest

from features.leakage_validators import validate_no_future_leakage


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """リークなしの正しいDataFrame"""
    return pd.DataFrame({
        "race_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "hist_value": [float("nan"), 10.0, 15.0],  # expanding().shift(1) で計算
    })


@pytest.fixture
def leaky_df() -> pd.DataFrame:
    """リークありのDataFrame（3行目に未来の値が混入）"""
    return pd.DataFrame({
        "race_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "hist_value": [float("nan"), 10.0, 25.0],  # 25.0 は未来データを含む
    })


@pytest.fixture
def source_df() -> pd.DataFrame:
    """hist_value の計算元データ（正しい値は expanding mean of source_col）"""
    return pd.DataFrame({
        "race_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "source_col": [10.0, 20.0, 15.0],
    })


class TestLeakageValidators:
    def test_clean_data_passes(self, clean_df: pd.DataFrame, source_df: pd.DataFrame):
        """リークなしのデータはバリデーションをパス"""
        issues = validate_no_future_leakage(
            clean_df, source_df, hist_cols=["hist_value"],
            source_cols=["source_col"],
        )
        assert issues == []

    def test_leaky_data_detected(self, leaky_df: pd.DataFrame, source_df: pd.DataFrame):
        """リークありのデータは検出される"""
        issues = validate_no_future_leakage(
            leaky_df, source_df, hist_cols=["hist_value"],
            source_cols=["source_col"],
        )
        assert len(issues) > 0
        assert any("hist_value" in issue for issue in issues)

    def test_nan_first_row_is_ok(self, clean_df: pd.DataFrame, source_df: pd.DataFrame):
        """最初の行が NaN でもエラーにならない"""
        issues = validate_no_future_leakage(
            clean_df, source_df, hist_cols=["hist_value"],
            source_cols=["source_col"],
        )
        assert issues == []

    def test_all_nan_column_passes(self):
        """全NaNの列はバリデーションをパス（計算不能 = リークなし）"""
        df = pd.DataFrame({
            "race_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "hist_value": [float("nan"), float("nan")],
        })
        src = pd.DataFrame({
            "race_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "source_col": [10.0, 20.0],
        })
        issues = validate_no_future_leakage(
            df, src, hist_cols=["hist_value"], source_cols=["source_col"],
        )
        assert issues == []

    def test_multiple_columns(self):
        """複数列を同時にバリデーション"""
        df = pd.DataFrame({
            "race_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "hist_a": [float("nan"), 10.0, 999.0],  # リークあり
            "hist_b": [float("nan"), 10.0, 15.0],   # OK
        })
        src = pd.DataFrame({
            "race_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "source_a": [10.0, 20.0, 15.0],
            "source_b": [10.0, 20.0, 15.0],
        })
        issues = validate_no_future_leakage(
            df, src,
            hist_cols=["hist_a", "hist_b"],
            source_cols=["source_a", "source_b"],
        )
        assert any("hist_a" in issue for issue in issues)
        assert not any("hist_b" in issue for issue in issues)
