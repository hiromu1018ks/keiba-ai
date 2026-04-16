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
            early_pop, mid_pop, late_pop, n_points,
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
    """オッズ変動量に基づいて Steamer/Stable/Drifter 分類

    Args:
        df: compute_movement_features の出力
        threshold: 分類閾値（デフォルト20%）

    Returns:
        分類列 ('movement_class', 'movement_bucket') が追加されたDataFrame
    """
    df = df.copy()
    drop = df["odds_drop_30_10"]  # 主要指標: 30→10分の変動

    def _bucket(x: float) -> str:
        if x >= 0.40:
            return "strong_drop"
        elif x >= 0.25:
            return "moderate_drop"
        elif x >= threshold:
            return "mild_drop"
        elif x > -threshold:
            return "stable"
        elif x >= -0.25:
            return "mild_rise"
        elif x >= -0.40:
            return "moderate_rise"
        else:
            return "strong_rise"

    def _category(x: float) -> str:
        if x >= threshold:
            return "steamer"
        elif x > -threshold:
            return "stable"
        else:
            return "drifter"

    df["movement_bucket"] = drop.apply(_bucket)
    df["movement_class"] = drop.apply(_category)

    return df


def join_results(
    movement_df: pd.DataFrame,
    entries: pd.DataFrame,
    races: pd.DataFrame,
    payouts: pd.DataFrame,
    min_points: int = 5,
) -> pd.DataFrame:
    """オッズ変動特徴量に着順・払戻金・レース条件を結合

    Args:
        movement_df: classify_movement 後のDataFrame
        entries: entries.parquet 読み込み
        races: races.parquet 読み込み
        payouts: payouts.parquet 読み込み
        min_points: 最低データポイント数

    Returns:
        分析用完全結合DataFrame
    """
    df = movement_df.copy()

    # 最低ポイント数フィルタ
    df = df[df["n_points"] >= min_points].copy()
    logger.info("After min_points filter: %d horses", len(df))

    # umaban 型合わせ: movement側はstr → int
    df["umaban_int"] = pd.to_numeric(df["umaban"], errors="coerce").astype("Int64")

    # ── entries 結合 ──
    entry_cols = ["race_id", "umaban", "kakuteijyuni", "ninki", "kisyucode", "chokyosicode"]
    entries_sub = entries[entry_cols].copy()
    # 両側ともstringで結合（movement側のumabanはgroupbyからstring、entries側もobject）
    entries_sub["umaban"] = entries_sub["umaban"].astype(str)
    df["umaban"] = df["umaban"].astype(str)
    df = df.merge(entries_sub, on=["race_id", "umaban"], how="left")

    # ── races 結合（レース条件） ──
    race_cols = ["race_id", "kyori", "syussotosu", "trackcd"]
    # sibababacd / dirtbabacd があれば含める
    available_race_cols = [
        c for c in race_cols + ["sibababacd", "dirtbabacd"] if c in races.columns
    ]
    races_sub = races[available_race_cols].drop_duplicates("race_id")
    df = df.merge(races_sub, on="race_id", how="left")

    # surface マッピング（trackcd: 10-22=芝, 23-29=ダート）
    if "trackcd" in df.columns:

        def _map_surface(tc):
            if pd.isna(tc):
                return "other"
            tc_int = int(tc)
            if 10 <= tc_int <= 22:
                return "turf"
            elif 23 <= tc_int <= 29:
                return "dirt"
            return "other"

        df["surface"] = df["trackcd"].apply(_map_surface)

    # ── payouts 結合（複勝払戻金） ──
    pay_cols = (
        ["race_id"]
        + [f"payfukusyoumaban{i}" for i in range(1, 6)]
        + [f"payfukusyopay{i}" for i in range(1, 6)]
    )
    pay_available = [c for c in pay_cols if c in payouts.columns]
    payouts_sub = payouts[pay_available].drop_duplicates("race_id")
    df = df.merge(payouts_sub, on="race_id", how="left")

    # ── 複勝判定 & 払戻金取得 ──
    def _get_place_payout(row: pd.Series) -> float:
        if pd.isna(row.get("kakuteijyuni")) or row["kakuteijyuni"] == 0:
            return 0.0
        if row["kakuteijyuni"] > 3:
            return 0.0
        umaban_val = row.get("umaban_int", row.get("umaban"))
        if pd.isna(umaban_val):
            return 0.0
        try:
            umaban_int = int(umaban_val)
        except (ValueError, TypeError):
            return 0.0
        for i in range(1, 6):
            maban_col = f"payfukusyoumaban{i}"
            pay_col = f"payfukusyopay{i}"
            if maban_col not in row.index:
                continue
            maban = row[maban_col]
            if pd.notna(maban) and umaban_int == int(maban):
                payout = row[pay_col]
                return float(payout) if pd.notna(payout) else 0.0
        return 0.0

    df["place_payout"] = df.apply(_get_place_payout, axis=1)
    df["is_place"] = (df["place_payout"] > 0).astype(int)
    df["is_win"] = (df["kakuteijyuni"] == 1).astype(int)

    logger.info("Joined results: %d records (%d place hits)", len(df), df["is_place"].sum())
    return df


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
