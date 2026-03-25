"""未来情報リーク検証モジュール

expanding 系特徴量 (hist_*) に未来情報が含まれていないことを検証する。
設計書 Rule 18: hist系特徴量は expanding().shift(1) で未来情報リークを完全遮断。

使い方:
    issues = validate_no_future_leakage(
        df=feat_df,          # 検証対象のDataFrame
        source_df=source_df, # 計算元データのDataFrame
        hist_cols=["hist_hit_rate_topk", ...],
        source_cols=["topk_hit", ...],
    )
    assert issues == [], f"リーク検出: {issues}"
"""

from __future__ import annotations

import pandas as pd


def validate_no_future_leakage(
    df: pd.DataFrame,
    source_df: pd.DataFrame,
    hist_cols: list[str],
    source_cols: list[str],
    date_col: str = "race_date",
    tolerance: float = 1e-10,
) -> list[str]:
    """expanding 系特徴量の未来情報リークを検証

    各行の hist 値が、その行より前の source データのみから計算されているか確認。
    expanding().shift(1) のセマンティクスに準拠。

    Args:
        df: 検証対象DataFrame (race_date + hist_cols を含む)
        source_df: 計算元DataFrame (race_date + source_cols を含む)
        hist_cols: 検証する履歴特徴量列名のリスト
        source_cols: hist_cols に対応する計算元列名のリスト (同じ順序)
        date_col: 日付列名
        tolerance: 浮動小数点誤差の許容範囲

    Returns:
        リークが検出された列のエラーメッセージリスト (空=問題なし)
    """
    issues: list[str] = []

    if len(hist_cols) != len(source_cols):
        issues.append(
            f"hist_cols ({len(hist_cols)}) と source_cols ({len(source_cols)}) の数が不一致"
        )
        return issues

    for hist_col, source_col in zip(hist_cols, source_cols):
        if hist_col not in df.columns or source_col not in source_df.columns:
            continue

        merged = pd.merge(
            df[[date_col, hist_col]],
            source_df[[date_col, source_col]],
            on=date_col,
            how="inner",
        )
        merged = merged.sort_values(date_col).reset_index(drop=True)

        # 全行NaNの場合はスキップ（計算不能 = リークなし）
        if merged[hist_col].isna().all():
            continue

        for i in range(len(merged)):
            actual = merged.iloc[i][hist_col]

            # NaN はスキップ（最初の行など）
            if pd.isna(actual):
                continue

            # i行目より前のデータのみで expanding mean を計算
            past_values = merged.iloc[:i][source_col].dropna()
            if len(past_values) == 0:
                # 過去データがないのに値がある = リーク
                issues.append(
                    f"{hist_col}: 行{i} に値 {actual} があるが過去データが不存在"
                )
                continue

            expected = past_values.mean()

            if abs(actual - expected) > tolerance:
                issues.append(
                    f"{hist_col}: 行{i} に未来情報リークの疑い "
                    f"(actual={actual:.10f}, expected={expected:.10f})"
                )

    return issues
