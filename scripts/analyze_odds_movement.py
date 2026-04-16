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
from datetime import datetime
from pathlib import Path

import numpy as np
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


if __name__ == "__main__":
    main()
