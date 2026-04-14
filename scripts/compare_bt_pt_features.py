"""BT(2025) vs PT(2026/4) 特徴量分布比較 — jyocdマージ衝突修正版"""

import sys
import os
import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.parquet_store import ParquetStore
from db.readers import load_entries, load_races, load_history_entries, load_history_races
from features.pace_aptitude_features import PaceAptitudeFeatures
from features.course_features import CourseFeatures


def compute_distance_bin(df: pd.DataFrame) -> pd.DataFrame:
    """kyori + surface → distance_bin"""
    if "distance_bin" not in df.columns and "kyori" in df.columns:
        is_turf = df["surface"] == "turf"
        dist = df["kyori"]
        df["distance_bin"] = "unknown"
        df.loc[is_turf & (dist > 2100), "distance_bin"] = "long"
        df.loc[is_turf & (dist <= 2100), "distance_bin"] = "intermediate"
        df.loc[is_turf & (dist <= 1700), "distance_bin"] = "mile"
        df.loc[is_turf & (dist <= 1400), "distance_bin"] = "sprint"
        df.loc[~is_turf & (dist > 1700), "distance_bin"] = "intermediate"
        df.loc[~is_turf & (dist <= 1700), "distance_bin"] = "mile"
        df.loc[~is_turf & (dist <= 1400), "distance_bin"] = "sprint"
    return df


def main():
    store = ParquetStore()

    # ================================================================
    # 1. BT 2025 特徴量読み込み
    # ================================================================
    print("=" * 70)
    print("Loading BT 2025 features...")
    bt_df = pd.read_parquet("data/backtest/bt_2025_horse_features.parquet")
    print(f"  BT 2025: {len(bt_df)} rows, {len(bt_df.columns)} cols")

    # ================================================================
    # 2. PT データ読み込み (2026/4/4, 5, 11, 12)
    # ================================================================
    pt_dates = ["20260404", "20260405", "20260411", "20260412"]
    print(f"\nLoading PT data for {pt_dates}...")

    pt_entries_list = []
    pt_races_list = []
    for d in pt_dates:
        e = load_entries(store, d, d)
        r = load_races(store, d, d)
        if not e.empty:
            pt_entries_list.append(e)
            print(f"  {d}: entries={len(e)}, races={len(r)}")
        if not r.empty:
            pt_races_list.append(r)

    if not pt_entries_list:
        print("ERROR: No PT entries found!")
        return

    pt_entries = pd.concat(pt_entries_list, ignore_index=True)
    pt_races = pd.concat(pt_races_list, ignore_index=True) if pt_races_list else pd.DataFrame()

    # ================================================================
    # 3. PT: entries + races マージ（jyocd 衝突回避）
    # ================================================================
    print("\nMerging PT entries + races...")

    # races 側から jyocd を除外（entries 側に既にあるため）
    race_merge_cols = ["race_id", "trackcd", "kyori", "surface", "track_condition_code", "syussotosu"]
    race_merge_cols = [c for c in race_merge_cols if c in pt_races.columns]
    pt_races_for_merge = pt_races[race_merge_cols].drop_duplicates("race_id")

    # entries 側から syussotosu を除外（races 側を使うため）
    pt_entries_for_merge = pt_entries.drop(columns=["syussotosu"], errors="ignore")

    pt_merged = pt_entries_for_merge.merge(
        pt_races_for_merge,
        on="race_id",
        how="left",
    )

    # syussotosu が欠落した場合、entries 側から復元
    if "syussotosu" not in pt_merged.columns and "syussotosu" in pt_entries.columns:
        pt_merged["syussotosu"] = pt_entries.set_index("race_id").loc[
            pt_merged["race_id"].drop_duplicates(), "syussotosu"
        ].values

    # distance_bin 計算
    pt_merged = compute_distance_bin(pt_merged)

    # syussotosu >= 8 フィルタ
    syussotosu_num = pd.to_numeric(pt_merged["syussotosu"], errors="coerce").fillna(-1)
    pt_merged = pt_merged[syussotosu_num >= 8].copy()
    print(f"  After merge+filter: {len(pt_merged)} rows")
    print(f"  Columns with jyocd: {'jyocd' in pt_merged.columns}")

    # ================================================================
    # 4. 新特徴量計算 (PaceAptitude + CourseFeatures)
    # ================================================================
    print("\nComputing PaceAptitudeFeatures for PT...")
    pace_feat = PaceAptitudeFeatures(store)
    pace_result = pace_feat.compute_batch(pt_merged)
    print(f"  pace_aptitude NaN rate: {pace_result['pace_aptitude'].isna().mean():.1%}")

    print("\nComputing CourseFeatures for PT...")
    course_feat = CourseFeatures(store)
    course_result = course_feat.compute_batch(pt_merged)
    print(f"  course_wr NaN rate: {course_result['course_wr'].isna().mean():1%}")

    # 結合
    _pace_cols = [c for c in ["pace_aptitude", "front_pace_wr", "closing_pace_wr"] if c in pace_result.columns]
    _course_cols = [c for c in ["course_wr", "course_distance_wr"] if c in course_result.columns]

    if _pace_cols:
        pt_merged = pt_merged.drop(columns=_pace_cols, errors="ignore").merge(
            pace_result[["kettonum", "race_id"] + _pace_cols], on=["kettonum", "race_id"], how="left"
        )
    if _course_cols:
        pt_merged = pt_merged.drop(columns=_course_cols, errors="ignore").merge(
            course_result[["kettonum", "race_id"] + _course_cols], on=["kettonum", "race_id"], how="left"
        )

    # ================================================================
    # 5. 分布比較
    # ================================================================
    print("\n" + "=" * 70)
    print("BT(2025) vs PT(2026/4) Feature Distribution Comparison")
    print("=" * 70)

    # 共通カラムを取得
    bt_cols = set(bt_df.columns)
    pt_cols = set(pt_merged.columns)
    common_cols = sorted(bt_cols & pt_cols)

    results = []
    for col in common_cols:
        # 数値カラムのみ処理（文字列カラムはスキップ）
        if not pd.api.types.is_numeric_dtype(bt_df[col]) or not pd.api.types.is_numeric_dtype(pt_merged[col]):
            continue

        bt_vals = bt_df[col].dropna()
        pt_vals = pt_merged[col].dropna()

        if len(bt_vals) < 10 or len(pt_vals) < 10:
            continue

        bt_mean = bt_vals.mean()
        bt_std = bt_vals.std()
        bt_med = bt_vals.median()
        bt_q25 = bt_vals.quantile(0.25)
        bt_q75 = bt_vals.quantile(0.75)

        pt_mean = pt_vals.mean()
        pt_std = pt_vals.std()
        pt_med = pt_vals.median()
        pt_q25 = pt_vals.quantile(0.25)
        pt_q75 = pt_vals.quantile(0.75)

        # Z-score (pooled std)
        pooled_std = np.sqrt((bt_std**2 + pt_std**2) / 2)
        z = (pt_mean - bt_mean) / pooled_std if pooled_std > 0 else 0

        bt_nan = bt_df[col].isna().mean()
        pt_nan = pt_merged[col].isna().mean()

        # カテゴリ分類
        category = "other (その他)"
        if col.startswith("pace_") or col.startswith("course_"):
            category = "NEW: pace/course (新特徴量)"
        elif col.startswith("sire_") or col.startswith("bms_"):
            category = "NEW: sire/bms (血統)"
        elif col in ("ev_win", "ev_place", "ev_win_corrected", "ev_place_corrected"):
            category = "EV (期待値)"
        elif col in ("p_win_pred", "p_win_corrected", "p_place_pred", "p_ability_win"):
            category = "prediction (予測値)"
        elif col in ("odds", "tanodds", "fukuoddslow"):
            category = "odds (オッズ)"

        results.append({
            "column": col,
            "bt_mean": bt_mean,
            "bt_std": bt_std,
            "bt_median": bt_med,
            "bt_25": bt_q25,
            "bt_75": bt_q75,
            "pt_mean": pt_mean,
            "pt_std": pt_std,
            "pt_median": pt_med,
            "pt_25": pt_q25,
            "pt_75": pt_q75,
            "z_score": z,
            "abs_z": abs(z),
            "bt_nan_rate": bt_nan,
            "pt_nan_rate": pt_nan,
            "category": category,
        })

    result_df = pd.DataFrame(results).sort_values("abs_z", ascending=False)

    # CSV保存
    out_path = "data/backtest/bt_pt_feature_comparison_v2.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # ================================================================
    # 6. 重要な特徴量の詳細表示
    # ================================================================
    print("\n" + "=" * 70)
    print("TOP SHIFTS (|Z| > 1.0): 分布が大きく異なる特徴量")
    print("=" * 70)

    big_shifts = result_df[result_df["abs_z"] > 1.0]
    if big_shifts.empty:
        print("No significant shifts found (|Z| > 1.0)")
    else:
        for _, row in big_shifts.iterrows():
            print(f"\n  [{row['category']}] {row['column']}")
            print(f"    BT: mean={row['bt_mean']:.4f} ± {row['bt_std']:.4f} med={row['bt_median']:.4f} NaN={row['bt_nan_rate']:.1%}")
            print(f"    PT: mean={row['pt_mean']:.4f} ± {row['pt_std']:.4f} med={row['pt_median']:.4f} NaN={row['pt_nan_rate']:.1%}")
            print(f"    Z-score: {row['z_score']:+.3f} (|Z|={row['abs_z']:.3f})")

    # 新特徴量の分布
    print("\n" + "=" * 70)
    print("NEW FEATURES (pace/course/sire/bms): 新特徴量の分布確認")
    print("=" * 70)

    new_feats = result_df[result_df["category"].str.startswith("NEW")]
    if new_feats.empty:
        print("No new features found in comparison!")
    else:
        for _, row in new_feats.iterrows():
            print(f"\n  {row['column']} [{row['category']}]")
            bt_n = len(bt_df[row["column"]].dropna())
            pt_n = len(pt_merged[row["column"]].dropna())
            print(f"    BT: mean={row['bt_mean']:.4f} med={row['bt_median']:.4f} NaN={row['bt_nan_rate']:.1%} (n={bt_n})")
            print(f"    PT: mean={row['pt_mean']:.4f} med={row['pt_median']:.4f} NaN={row['pt_nan_rate']:.1%} (n={pt_n})")
            print(f"    Z: {row['z_score']:+.3f}")

    # EV / 予測値 の分布
    print("\n" + "=" * 70)
    print("EV & PREDICTION DISTRIBUTIONS: 予測品質の確認")
    print("=" * 70)

    ev_preds = result_df[result_df["category"].isin(["EV (期待値)", "prediction (予測値)"])]
    for _, row in ev_preds.iterrows():
        print(f"\n  {row['column']}")
        print(f"    BT: mean={row['bt_mean']:.4f} med={row['bt_median']:.4f}")
        print(f"    PT: mean={row['pt_mean']:.4f} med={row['pt_median']:.4f}")
        print(f"    Z: {row['z_score']:+.3f} | BT_NaN={row['bt_nan_rate']:.1%} PT_NaN={row['pt_nan_rate']:.1%}")


if __name__ == "__main__":
    main()
