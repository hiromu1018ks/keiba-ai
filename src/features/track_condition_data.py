"""track_condition_data.py — ダート含水率・芝クッション値CSV→race-level Parquet変換

外部CSV (entry_id, value) を読み込み、18桁entry_idの先頭16桁をrace_idとして
race-levelに集約し、単一Parquet (track_conditions.parquet) として出力する。

設計判断:
  D-04: entry_id(18桁) = race_id(先頭16桁) + umaban(末尾2桁)
  D-05: 同一race_id内で非NaN値が複数種類ある場合はValueError
  D-06: NaNと非NaNの混在時は非NaN値を採用
  D-08: 含水率/クッション値のNaNはそのまま保持。統計的補完は行わない
  D-09: 物理的異常値のみNaN化 — 含水率は 0 < value < 100、クッション値は value > 0
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from db.parquet_store import ParquetStore

logger = logging.getLogger(__name__)


def parse_track_condition_csv(
    csv_path: str | Path,
    value_name: str,
) -> pd.DataFrame:
    """ヘッダなしCSV (entry_id, value) を読み込み、race_id/umaban/race_dateを派生させる。

    Args:
        csv_path: CSVファイルパス (ヘッダなし、2列: entry_id, value)
        value_name: 値列の名前 (例: "dirt_moisture", "turf_cushion")

    Returns:
        DataFrame with columns: [entry_id, race_id, umaban, race_date, {value_name}]
    """
    df = pd.read_csv(
        csv_path,
        header=None,
        names=["entry_id", value_name],
        dtype={"entry_id": str},
    )
    # entry_id が18桁であることを確認
    if (df["entry_id"].str.len() != 18).any():
        bad = df.loc[df["entry_id"].str.len() != 18, "entry_id"].head(5).tolist()
        raise ValueError(f"entry_id must be 18 digits, got: {bad}")

    df["race_id"] = df["entry_id"].str[:16]
    df["umaban"] = df["entry_id"].str[16:18]
    df["race_date"] = pd.to_datetime(df["race_id"].str[:8], format="%Y%m%d")
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")

    return df[["entry_id", "race_id", "umaban", "race_date", value_name]]


def aggregate_to_race_level(
    df: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:
    """entry-level DataFrameをrace-levelに集約する。

    集約ルール:
      - 同一race_id内で非NaN値が1種類のみ → その値
      - NaNと非NaNの混在 → 非NaN値を採用
      - 複数種類の非NaN値 → ValueError (D-05)

    Args:
        df: parse_track_condition_csvの出力 (race_id, race_date, value_colを含む)
        value_col: 集約対象の値列名

    Returns:
        DataFrame with columns: [race_id, race_date, {value_col}]

    Raises:
        ValueError: 同一race_id内に複数種類の非NaN値が存在する場合
    """
    if df.empty:
        return pd.DataFrame(columns=["race_id", "race_date", value_col])

    def _agg_group(group: pd.DataFrame) -> float:
        non_null = group[value_col].dropna()
        if non_null.empty:
            return np.nan
        unique_vals = non_null.unique()
        if len(unique_vals) == 1:
            return float(unique_vals[0])
        raise ValueError(
            f"Multiple distinct non-NaN values in race_id={group['race_id'].iloc[0]}: "
            f"{unique_vals.tolist()}"
        )

    # race_id, race_dateでグループ化
    grouped = df.groupby(["race_id", "race_date"], sort=False, observed=True)
    result_rows: list[dict[str, object]] = []
    for (rid, rdate), group in grouped:
        val = _agg_group(group)
        result_rows.append({"race_id": rid, "race_date": rdate, value_col: val})

    result = pd.DataFrame(result_rows)
    if not result.empty:
        result["race_date"] = pd.to_datetime(result["race_date"])
    return result


def validate_physical_range(
    df: pd.DataFrame,
    col: str,
    low: float,
    high: float,
) -> tuple[pd.DataFrame, int]:
    """物理的異常値をNaNに置換する。

    Args:
        df: 対象DataFrame
        col: チェック対象の数値列
        low: 下限 (exclusive)。 NaNやinfはこの値未満とみなす
        high: 上限 (exclusive)。 np.infで上限なし

    Returns:
        (modified df, count_of_replaced_values)
    """
    df = df.copy()
    numeric = pd.to_numeric(df[col], errors="coerce")
    # NaNはそのまま、範囲外のみNaN化
    out_of_range = numeric.notna() & ~((numeric > low) & (numeric < high))
    count = int(out_of_range.sum())
    if count > 0:
        logger.info(
            "Replaced %d out-of-range values in '%s' (range: (%s, %s))",
            count, col, low, high,
        )
        df.loc[out_of_range, col] = np.nan
    return df, count


def convert_track_conditions(
    dirt_csv: str | Path,
    cushion_csv: str | Path,
    store: ParquetStore,
    races_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """ダート含水率・芝クッション値CSVを変換し、race-level Parquetを出力する。

    Args:
        dirt_csv: ダート含水率CSVパス
        cushion_csv: 芝クッション値CSVパス
        store: ParquetStore インスタンス
        races_df: 既存races.parquetのDataFrame (交差検証用、オプション)

    Returns:
        DataFrame with columns: [race_id, race_date, dirt_moisture, turf_cushion]
    """
    # --- ダート含水率 ---
    logger.info("Parsing dirt moisture CSV: %s", dirt_csv)
    dirt_df = parse_track_condition_csv(dirt_csv, "dirt_moisture")
    logger.info("  %d entry-level rows", len(dirt_df))

    dirt_df, dirt_replaced = validate_physical_range(dirt_df, "dirt_moisture", low=0.0, high=100.0)
    logger.info("  %d physical out-of-range values replaced", dirt_replaced)

    dirt_race = aggregate_to_race_level(dirt_df, "dirt_moisture")
    logger.info("  %d race-level rows", len(dirt_race))

    # --- 芝クッション値 ---
    logger.info("Parsing turf cushion CSV: %s", cushion_csv)
    turf_df = parse_track_condition_csv(cushion_csv, "turf_cushion")
    logger.info("  %d entry-level rows", len(turf_df))

    turf_df, turf_replaced = validate_physical_range(turf_df, "turf_cushion", low=0.0, high=np.inf)
    logger.info("  %d physical out-of-range values replaced", turf_replaced)

    turf_race = aggregate_to_race_level(turf_df, "turf_cushion")
    logger.info("  %d race-level rows", len(turf_race))

    # --- 外部結合で統合 ---
    merged = dirt_race.merge(turf_race, on=["race_id", "race_date"], how="outer")

    # race_dateの欠損を補完 (一方にしかないrace)
    if merged["race_date"].isna().any():
        # dirt と turf の race_date を結合して補完
        date_map = pd.concat([
            dirt_race[["race_id", "race_date"]],
            turf_race[["race_id", "race_date"]],
        ]).drop_duplicates(subset="race_id")
        merged = merged.drop(columns=["race_date"]).merge(date_map, on="race_id", how="left")

    # 列順を確定
    merged = merged[["race_id", "race_date", "dirt_moisture", "turf_cushion"]]
    logger.info("Merged: %d races total", len(merged))

    # --- 交差検証 (D-07) ---
    if races_df is not None and not races_df.empty:
        known_races = set(races_df["race_id"].unique())
        new_races = set(merged["race_id"].unique())
        unmatched = new_races - known_races
        if unmatched:
            logger.warning(
                "%d race_ids in track_conditions not found in races.parquet (first 10: %s)",
                len(unmatched),
                sorted(unmatched)[:10],
            )
        missing = known_races - new_races
        logger.info(
            "Cross-validation: %d track_condition races, %d races.parquet races, "
            "%d unmatched in track_conditions, %d missing from track_conditions",
            len(new_races),
            len(known_races),
            len(unmatched),
            len(missing),
        )

    # --- Parquet出力 ---
    store.write("raw", "track_conditions", merged)
    logger.info("Written to data/raw/track_conditions.parquet (%d rows)", len(merged))

    return merged
