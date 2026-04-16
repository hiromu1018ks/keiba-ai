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


def analyze_basic_stats(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """基本統計: テーブルA/B/C

    Returns:
        dict with keys: table_a, table_b, table_c
    """
    stake = 100  # 100円固定

    # ── テーブルA: バケット別成績 ──
    bucket_stats = (
        df.groupby("movement_bucket")
        .agg(
            count=("is_place", "count"),
            place_rate=("is_place", "mean"),
            win_rate=("is_win", "mean"),
            avg_final_odds=("final_odds", "mean"),
            total_payout=("place_payout", "sum"),
            total_bets=("is_place", "count"),
        )
        .reset_index()
    )
    bucket_stats["place_roi"] = (
        bucket_stats["total_payout"] / (bucket_stats["total_bets"] * stake) * 100
    )
    bucket_stats["place_rate"] = (bucket_stats["place_rate"] * 100).round(1)
    bucket_stats["win_rate"] = (bucket_stats["win_rate"] * 100).round(1)
    bucket_stats["place_roi"] = bucket_stats["place_roi"].round(1)

    # ── テーブルB: 人気セグメント × クラス クロス ──
    def _pop_segment(ninki: float) -> str:
        if pd.isna(ninki):
            return "unknown"
        ninki_val = float(ninki)
        if ninki_val <= 3:
            return "1-3番人気"
        elif ninki_val <= 7:
            return "4-7番人気"
        else:
            return "8番人気以降"

    df["pop_segment"] = df["ninki"].apply(_pop_segment)

    cross = (
        df.groupby(["pop_segment", "movement_class"])
        .agg(
            count=("is_place", "count"),
            place_rate=("is_place", "mean"),
        )
        .reset_index()
    )
    cross["place_rate"] = (cross["place_rate"] * 100).round(1)

    # ── テーブルC: 時間枠別予測力比較 ──
    windows = {
        "60->10min": "odds_drop_60_10",
        "30->10min": "odds_drop_30_10",
        "10->final": "odds_drop_10_final",
    }
    window_rows = []
    for label, col in windows.items():
        for thresh in [0.15, 0.20, 0.25]:
            mask = df[col] >= thresh
            sub = df[mask]
            if len(sub) > 0:
                window_rows.append(
                    {
                        "window": label,
                        "threshold": f"{thresh * 100:.0f}%",
                        "count": len(sub),
                        "place_rate": round(sub["is_place"].mean() * 100, 1),
                        "roi": round(sub["place_payout"].sum() / (len(sub) * stake) * 100, 1),
                    }
                )
    window_comparison = pd.DataFrame(window_rows)

    return {
        "table_a": bucket_stats,
        "table_b": cross,
        "table_c": window_comparison,
    }


def analyze_jockey_trainer(df: pd.DataFrame, top_n: int = 20) -> dict[str, pd.DataFrame]:
    """騎手・調教師別の急落傾向分析

    Returns:
        dict with keys: by_jockey, by_trainer
    """

    def _analyze_group(group_col: str) -> pd.DataFrame:
        grouped = (
            df.groupby(group_col, dropna=False)
            .agg(
                rides=("is_place", "count"),
                steam_count=(
                    "movement_class",
                    lambda x: (x == "steamer").sum(),  # type: ignore[no-any-return]
                ),
                steam_place_rate=(
                    "is_place",
                    lambda x: (
                        x[df.loc[x.index, "movement_class"] == "steamer"].mean()
                        if (df.loc[x.index, "movement_class"] == "steamer").any()
                        else float("nan")
                    ),
                ),
                stable_place_rate=(
                    "is_place",
                    lambda x: (
                        x[df.loc[x.index, "movement_class"] == "stable"].mean()
                        if (df.loc[x.index, "movement_class"] == "stable").any()
                        else float("nan")
                    ),
                ),
            )
            .reset_index()
        )

        grouped["steam_rate"] = (grouped["steam_count"] / grouped["rides"] * 100).round(1)
        grouped["diff"] = (grouped["steam_place_rate"] - grouped["stable_place_rate"]).round(1)

        grouped = grouped[grouped["rides"] >= 10]
        grouped = grouped.sort_values("steam_rate", ascending=False)
        return grouped.head(top_n)

    return {
        "by_jockey": _analyze_group("kisyucode"),
        "by_trainer": _analyze_group("chokyosicode"),
    }


def analyze_race_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """レース条件別のマトリックス分析 (Table E)"""

    def _distance_band(kyori: float) -> str:
        if pd.isna(kyori):
            return "unknown"
        kyori_val = float(kyori)
        if kyori_val <= 1400:
            return "短距離(<=1400m)"
        elif kyori_val <= 2000:
            return "中距離(1400-2000m)"
        else:
            return "長距離(>2000m)"

    def _field_size(syussotosu: float) -> str:
        if pd.isna(syussotosu):
            return "unknown"
        size_val = float(syussotosu)
        if size_val <= 8:
            return "8頭以下"
        elif size_val <= 12:
            return "9-12頭"
        else:
            return "13頭以上"

    df_analysis = df.copy()
    df_analysis["distance_band"] = df_analysis["kyori"].apply(_distance_band)
    df_analysis["field_size_cat"] = df_analysis["syussotosu"].apply(_field_size)

    dimensions = [
        ("surface", "surface"),
        ("distance_band", "distance_band"),
        ("field_size_cat", "field_size_cat"),
    ]

    rows = []
    for label, col in dimensions:
        for cls in ["steamer", "stable"]:
            sub = df_analysis[df_analysis["movement_class"] == cls]
            grouped = sub.groupby(col, dropna=False)
            for cat, grp in grouped:
                if len(grp) < 5:
                    continue
                rows.append(
                    {
                        "dimension": label,
                        "category": cat,
                        "movement_class": cls,
                        "count": len(grp),
                        "place_rate": round(grp["is_place"].mean() * 100, 1),
                        "roi": round(grp["place_payout"].sum() / (len(grp) * 100) * 100, 1),
                    }
                )

    return pd.DataFrame(rows)


def print_summary(results: dict, title: str = "") -> None:
    """コンソールに分析結果を表形式で出力"""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

    for name, df_val in results.items():
        if isinstance(df_val, pd.DataFrame) and len(df_val) > 0:
            print(f"\n--- {name} ---")
            print(df_val.to_string(index=False))


def save_csv(results: dict, output_dir: str, detail_df: pd.DataFrame | None = None) -> None:
    """CSVファイル群を出力"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, df_val in results.items():
        if isinstance(df_val, pd.DataFrame) and len(df_val) > 0:
            path = out / f"{name}.csv"
            df_val.to_csv(path, index=False)
            logger.info("Saved: %s (%d rows)", path, len(df_val))

    if detail_df is not None and len(detail_df) > 0:
        detail_path = out / "detail_records.csv"
        keep_cols = [
            "race_id",
            "umaban",
            "movement_class",
            "movement_bucket",
            "odds_drop_30_10",
            "final_odds",
            "ninki",
            "kakuteijyuni",
            "is_place",
            "place_payout",
            "kisyucode",
            "chokyosicode",
            "surface",
        ]
        available = [c for c in keep_cols if c in detail_df.columns]
        detail_df[available].to_csv(detail_path, index=False)
        logger.info("Saved: %s", detail_path)


def main() -> None:
    args = build_parser().parse_args()
    start_year = int(args.start[:4])
    end_year = int(args.end[:4])
    output_dir = (
        args.output_dir or f"output/odds_movement_analysis_{datetime.now().strftime('%Y%m%d')}"
    )

    logger.info("=" * 60)
    logger.info("オッズ変動分析: %s ~ %s", args.start, args.end)
    logger.info("=" * 60)

    # 1. データ読み込み
    ts_df = load_time_series(start_year, end_year)
    entries_df = load_entries(args.start, args.end)
    races_df = load_races(args.start, args.end)
    payouts_df = load_payouts(args.start, args.end)

    # 2. 特徴量計算
    movement_df = compute_movement_features(ts_df)
    logger.info("Computed movement features for %d horses", len(movement_df))

    # 3. 分類
    classified_df = classify_movement(movement_df, threshold=args.drop_threshold)
    logger.info("Classified movements")

    # 4. 結合
    joined_df = join_results(
        classified_df, entries_df, races_df, payouts_df, min_points=args.min_points
    )
    logger.info("Final dataset: %d records", len(joined_df))

    # 5. 分析
    basic = analyze_basic_stats(joined_df)
    jt = analyze_jockey_trainer(joined_df)
    rc = analyze_race_conditions(joined_df)

    # 6. 出力
    all_results = {**basic, **jt, "by_race_condition": rc}
    print_summary(all_results, title=f"オッズ変動分析結果 ({args.start} ~ {args.end})")
    save_csv(all_results, output_dir, detail_df=joined_df if args.detail else None)

    logger.info("Done. Output saved to: %s", output_dir)


if __name__ == "__main__":
    main()
