"""オッズ急落・急騰分析スクリプト

レース直前のオッズ変動が複勝率や回収率に与える影響を統計的に分析する。

Usage:
    python scripts/analyze_odds_movement.py --start 20230101 --end 20251231
    python scripts/analyze_odds_movement.py --start 20240101 --end 20251231 --detail
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# ── プロジェクトルート設定 ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(ROOT) / "data"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="オッズ変動分析")
    parser.add_argument(
        "--start", type=str, default="20230101", help="開始日 YYYYMMDD (default: 20230101)"
    )
    parser.add_argument(
        "--end", type=str, default="20251231", help="終了日 YYYYMMDD (default: 20251231)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="出力ディレクトリ (default: output/odds_movement_analysis_{date})",
    )
    parser.add_argument(
        "--drop-threshold", type=float, default=0.20, help="分類閾値 (default: 0.20)"
    )
    parser.add_argument(
        "--min-points", type=int, default=5, help="1頭あたり最低データポイント数 (default: 5)"
    )
    parser.add_argument("--detail", action="store_true", help="詳細レコードCSVも出力")
    return parser


def load_time_series(start_year: int, end_year: int) -> pd.DataFrame:
    """jodds_tanpuku.parquet を読み込み、年フィルタ適用"""
    path = DATA_DIR / "odds" / "jodds_tanpuku.parquet"
    if not path.exists():
        raise FileNotFoundError(f"jodds_tanpuku.parquet not found at {path}")

    logger.info("Loading jodds_tanpuku.parquet (year %d-%d)...", start_year, end_year)
    df = pd.read_parquet(
        path,
        filters=[("year", ">=", str(start_year)), ("year", "<=", str(end_year))],
    )
    logger.info("Loaded %d rows", len(df))
    return df


def load_entries(start_date: str, end_date: str) -> pd.DataFrame:
    """entries.parquet を読み込み、日付フィルタ適用"""
    path = DATA_DIR / "raw" / "entries.parquet"
    df = pd.read_parquet(path)
    df["_race_date_str"] = pd.to_datetime(df["race_date"]).dt.strftime("%Y%m%d")
    df = df[(df["_race_date_str"] >= start_date) & (df["_race_date_str"] <= end_date)]
    df = df.drop(columns=["_race_date_str"])
    logger.info("Loaded %d entries", len(df))
    return df


def load_races(start_date: str, end_date: str) -> pd.DataFrame:
    """races.parquet を読み込み、日付フィルタ適用"""
    path = DATA_DIR / "raw" / "races.parquet"
    df = pd.read_parquet(path)
    df["_race_date_str"] = pd.to_datetime(df["race_date"]).dt.strftime("%Y%m%d")
    df = df[(df["_race_date_str"] >= start_date) & (df["_race_date_str"] <= end_date)]
    df = df.drop(columns=["_race_date_str"])
    logger.info("Loaded %d races", len(df))
    return df


def load_payouts(start_date: str, end_date: str) -> pd.DataFrame:
    """payouts.parquet を読み込み、確定結果のみ抽出"""
    path = DATA_DIR / "raw" / "payouts.parquet"
    df = pd.read_parquet(path)
    df["_race_date_str"] = pd.to_datetime(df["race_date"]).dt.strftime("%Y%m%d")
    df = df[(df["_race_date_str"] >= start_date) & (df["_race_date_str"] <= end_date)]
    df = df.drop(columns=["_race_date_str"])
    df = df[df["datakubun"] == "2"]
    logger.info("Loaded %d confirmed payouts", len(df))
    return df


def compute_movement_features(ts_df: pd.DataFrame) -> pd.DataFrame:
    """時系列オッズから各馬のオッズ変動特徴量をベクトル化計算

    Args:
        ts_df: jodds_tanpuku データ。必須列: race_id, umaban(str),
               happyotime(str), tanodds(float), tanninki(Int64), race_date(datetime)

    Returns:
        各(race_id, umaban)ごとに1行のDataFrame。
        列: race_id, umaban, early_odds, mid_odds, late_odds, final_odds,
            early_pop, late_pop, n_points,
            odds_drop_60_10, odds_drop_30_10, odds_drop_10_final,
            pop_change_30_10
    """
    # ── 前処理 ──
    df = ts_df.copy()

    # umaban を string → int に正規化（結合用）
    df["umaban_int"] = pd.to_numeric(df["umaban"], errors="coerce").astype("Int64")

    # tanninki の NaN を -1 で埋める
    df["tanninki"] = df["tanninki"].fillna(-1)

    # 有効なオッズのみ残す（ゼロとNaN除外）
    df = df[df["tanodds"].notna() & (df["tanodds"] > 0)]

    # NAR除外 (jyocdはobject型なので数値変換して比較)
    if "jyocd" in df.columns:
        jyocd_num = pd.to_numeric(df["jyocd"], errors="coerce")
        df = df[jyocd_num < 30]

    # ソート: (race_id, umaban) ごとに (race_date, happyotime) で昇順
    df = df.sort_values(["race_id", "umaban", "race_date", "happyotime"])

    # ── groupby agg ──
    def _first(series: pd.Series) -> object:
        return series.iloc[0]

    def _mid(series: pd.Series) -> object:
        idx = len(series) // 2
        return series.iloc[idx]

    def _late(series: pd.Series) -> object:
        idx = int(len(series) * 0.9)
        return series.iloc[idx]

    g = df.groupby(["race_id", "umaban"], sort=False)

    features = g.agg(
        early_odds=("tanodds", _first),
        mid_odds=("tanodds", _mid),
        late_odds=("tanodds", _late),
        final_odds=("tanodds", "last"),
        early_pop=("tanninki", _first),
        mid_pop=("tanninki", _mid),
        late_pop=("tanninki", _late),
        n_points=("tanodds", "count"),
    ).reset_index()

    # ── 変動率計算 ──
    features["odds_drop_60_10"] = (features["early_odds"] - features["late_odds"]) / features[
        "early_odds"
    ]
    features["odds_drop_30_10"] = (features["mid_odds"] - features["late_odds"]) / features[
        "mid_odds"
    ]
    features["odds_drop_10_final"] = (features["late_odds"] - features["final_odds"]) / features[
        "late_odds"
    ]
    features["pop_change_30_10"] = features["mid_pop"] - features["late_pop"]

    return features


def classify_movement(
    df: pd.DataFrame,
    threshold: float = 0.20,
) -> pd.DataFrame:
    """TODO: Task 3で実装"""
    raise NotImplementedError


def main() -> None:
    args = build_parser().parse_args()
    start_year = int(args.start[:4])
    end_year = int(args.end[:4])

    logger.info("=" * 60)
    logger.info("オッズ変動分析: %s ~ %s", args.start, args.end)
    logger.info("=" * 60)

    # 1. データ読み込み
    ts_df = load_time_series(start_year, end_year)
    entries_df = load_entries(args.start, args.end)  # noqa: F841
    races_df = load_races(args.start, args.end)  # noqa: F841
    payouts_df = load_payouts(args.start, args.end)  # noqa: F841

    # 2. 特徴量計算
    movement_df = compute_movement_features(ts_df)
    logger.info("Computed movement features for %d horses", len(movement_df))

    # TODO: 残りのステップで実装
    logger.info("Analysis complete (placeholder).")


if __name__ == "__main__":
    main()
