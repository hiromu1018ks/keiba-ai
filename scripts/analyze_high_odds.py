"""高オッズ的中パターンのハイブリッド分析スクリプト

Cohen's d 統計プロファイリング + LightGBM TreeSHAP による特徴量分析。
高オッズ帯(デフォルト20倍以上)の的中馬と非的中馬を比較し、
どの特徴量が高オッズ的中を予測するかを特定する。

使い方:
  python scripts/analyze_high_odds.py --help
  python scripts/analyze_high_odds.py --odds-threshold 20.0 --start 20200101 --end 20241231
  python scripts/analyze_high_odds.py --odds-threshold 10.0 --surface dirt
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys

# プロジェクトルートをパスに追加
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="高オッズ的中パターンのハイブリッド分析 (Cohen's d + TreeSHAP)",
    )
    parser.add_argument(
        "--odds-threshold",
        type=float,
        default=20.0,
        help="高オッズ判定閾値 (default: 20.0)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="分析開始日 (YYYYMMDD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="分析終了日 (YYYYMMDD)",
    )
    parser.add_argument(
        "--model-dir",
        default="data/models/",
        help="学習済みモデルのディレクトリ (default: data/models/)",
    )
    parser.add_argument(
        "--output",
        default="data/analysis/high_odds_analysis.json",
        help="出力JSONパス (default: data/analysis/high_odds_analysis.json)",
    )
    parser.add_argument(
        "--surface",
        choices=["turf", "dirt"],
        default="turf",
        help="解析対象のサーフェス (default: turf)",
    )
    args = parser.parse_args()

    import numpy as np
    import pandas as pd

    data_dir = os.path.join(ROOT, "data")

    # --- 1. データ読み込み ---
    logger.info("データ読み込み中...")
    from db.parquet_store import ParquetStore

    store = ParquetStore(data_dir=data_dir)

    races_df = store.load("raw", "races")
    entries_df = store.load("raw", "entries")

    # オッズデータ読み込み
    odds_path = os.path.join(data_dir, "odds", "snapshots.parquet")
    if not os.path.exists(odds_path):
        logger.error("オッズデータが見つかりません: %s", odds_path)
        sys.exit(1)
    odds_df = pd.read_parquet(odds_path)

    logger.info(
        "データ読み込み完了: races=%d, entries=%d, odds=%d",
        len(races_df),
        len(entries_df),
        len(odds_df),
    )

    # --- 2. 日付フィルタ ---
    if args.start:
        start_dt = pd.Timestamp(args.start)
        races_df = races_df[races_df["race_date"] >= start_dt]
    if args.end:
        end_dt = pd.Timestamp(args.end)
        races_df = races_df[races_df["race_date"] <= end_dt]

    # --- 3. 高オッズサンプル抽出 ---
    # tanodds 列の確認 (大文字小文字の可能性あり)
    odds_col = _find_column(odds_df, ["tanodds", "TAN_ODDS", "win_odds"])
    if odds_col is None:
        logger.error("オッズ列が見つかりません。利用可能列: %s", list(odds_df.columns))
        sys.exit(1)

    high_odds = odds_df[odds_df[odds_col] >= args.odds_threshold].copy()
    logger.info(
        "高オッズ(>=%.1f)サンプル: %d件", args.odds_threshold, len(high_odds)
    )

    # entries と join して着順情報を取得
    # race_id, umaban で join
    join_cols = _find_common_columns(high_odds, entries_df, ["race_id", "umaban"])
    if join_cols is None:
        logger.error("オッズと出走データのjoinキー(race_id, umaban)が見つかりません")
        sys.exit(1)

    merged = high_odds.merge(entries_df, on=join_cols, how="inner")

    # 的中判定: kakuteijyuni == 1 が的中
    kj_col = _find_column(merged, ["kakuteijyuni", "KAKUTEIJYUNI"])
    if kj_col is None:
        logger.error("確定着順列が見つかりません")
        sys.exit(1)

    hits = merged[merged[kj_col] == 1]
    misses = merged[merged[kj_col] > 1]

    n_hits = len(hits)
    n_misses = len(misses)
    logger.info(
        "高オッズ的中: %d件, 非的中: %d件 (閾値: %.1f)",
        n_hits,
        n_misses,
        args.odds_threshold,
    )

    # --- 4. サンプル不足チェック ---
    if n_hits < 50:
        logger.warning(
            "高オッズ的中サンプル < 50 (%d). "
            "--odds-threshold を 10 に下げることを検討してください。",
            n_hits,
        )

    if n_hits == 0 or n_misses == 0:
        logger.error("的中群または非的中群が0件です。分析を中止します。")
        sys.exit(1)

    # --- 5. 統計プロファイリング (Cohen's d) ---
    logger.info("Cohen's d 統計プロファイリング実行中...")

    # 数値列のみ抽出
    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    # ターゲット関連列を除外
    exclude_cols = {kj_col, odds_col, "umaban"}
    analysis_cols = [c for c in numeric_cols if c not in exclude_cols]

    cohens_d_results = []
    for col in analysis_cols:
        hit_vals = hits[col].dropna().values.astype(float)
        miss_vals = misses[col].dropna().values.astype(float)

        if len(hit_vals) < 2 or len(miss_vals) < 2:
            continue

        d = _compute_cohens_d(hit_vals, miss_vals)
        cohens_d_results.append({
            "feature": col,
            "cohens_d": d,
            "hit_mean": float(np.mean(hit_vals)),
            "miss_mean": float(np.mean(miss_vals)),
            "hit_n": len(hit_vals),
            "miss_n": len(miss_vals),
        })

    cohens_df = pd.DataFrame(cohens_d_results)
    if len(cohens_df) > 0:
        cohens_df["abs_d"] = cohens_df["cohens_d"].abs()
        cohens_df = cohens_df.sort_values("abs_d", ascending=False).drop(
            columns=["abs_d"]
        ).reset_index(drop=True)

    logger.info("Cohen's d 分析完了: %d特徴量", len(cohens_df))

    # --- 6. SHAP分析 (高オッズ馬限定) ---
    shap_df = pd.DataFrame()
    model = None
    try:
        model_path = _find_model_file(args.model_dir, preferred_surface=args.surface)
        if model_path is not None:
            import lightgbm as lgb

            logger.info("モデル読み込み: %s", model_path)
            model = lgb.Booster(model_file=model_path)

            # モデル特徴量に対応する列のみ抽出
            feature_names = model.feature_name()
            available_features = [f for f in feature_names if f in merged.columns]

            if len(available_features) == len(feature_names):
                features_for_shap = merged[feature_names].fillna(0)

                # pred_contrib=True で TreeSHAP
                shap_matrix = model.predict(features_for_shap, pred_contrib=True)
                # 最後の列は expected value (base value)
                shap_values = shap_matrix[:, :-1]
                mean_abs_shap = np.abs(shap_values).mean(axis=0)

                shap_df = pd.DataFrame({
                    "feature": feature_names,
                    "mean_abs_shap_high_odds": mean_abs_shap,
                })
                shap_df = shap_df.sort_values(
                    "mean_abs_shap_high_odds", ascending=False
                ).reset_index(drop=True)
                logger.info("SHAP分析完了: %d特徴量", len(shap_df))
            else:
                missing = set(feature_names) - set(available_features)
                logger.warning(
                    "モデル特徴量の一部がデータに存在しません (%d件): %s",
                    len(missing),
                    list(missing)[:5],
                )
        else:
            logger.warning(
                "モデルファイルが見つかりません: %s。SHAP分析をスキップします。",
                args.model_dir,
            )
    except Exception as e:
        logger.warning("SHAP分析でエラー発生、スキップ: %s", e)

    # --- 7. 結果統合・出力 ---
    if len(shap_df) > 0 and len(cohens_df) > 0:
        combined = cohens_df.merge(shap_df, on="feature", how="left")
    elif len(cohens_df) > 0:
        combined = cohens_df.copy()
        combined["mean_abs_shap_high_odds"] = float("nan")
    else:
        combined = shap_df.copy() if len(shap_df) > 0 else pd.DataFrame()
        if len(combined) > 0:
            combined["cohens_d"] = float("nan")
            combined["hit_mean"] = float("nan")
            combined["miss_mean"] = float("nan")
            combined["hit_n"] = 0
            combined["miss_n"] = 0

    # 上位20特徴量をMarkdownテーブルで出力
    if len(combined) > 0:
        top20 = combined.head(20)
        print("\n=== 高オッズ的中パターン分析 上位20特徴量 ===")
        print(top20.to_string(index=False))

    # JSON出力
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    analysis_records = combined.to_dict(orient="records") if len(combined) > 0 else []
    output_data = {
        "odds_threshold": args.odds_threshold,
        "surface": args.surface,
        "hit_count": n_hits,
        "miss_count": n_misses,
        "analysis": analysis_records,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info("分析結果保存: %s (%d特徴量)", output_path, len(analysis_records))
    print("\n完了")


def _compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d を計算 (pooled standard deviation)。

    d = (m1 - m2) / pooled_std
    """
    n1, n2 = len(group1), len(group2)
    m1, m2 = float(np.mean(group1)), float(np.mean(group2))
    var1, var2 = float(np.var(group1, ddof=1)), float(np.var(group2, ddof=1))

    # Pooled standard deviation
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    if pooled_var <= 0:
        return 0.0
    pooled_std = float(np.sqrt(pooled_var))

    return (m1 - m2) / pooled_std


def _find_model_file(model_dir: str, *, preferred_surface: str = "turf") -> str | None:
    """モデルディレクトリからwin_hitモデルファイルを検索。"""
    # 指定サーフェスを優先、次に他のサーフェスを試す
    seen: set[str] = set()
    for surface in [preferred_surface, "turf", "dirt"]:
        if surface in seen:
            continue
        seen.add(surface)
        pattern = os.path.join(model_dir, f"win_hit_{surface}.lgb")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]

    # フォールバック: 任意のwin_hitモデル
    pattern = os.path.join(model_dir, "win_hit_*.lgb")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]

    return None


def _find_column(
    df: "pd.DataFrame", candidates: list[str]
) -> str | None:
    """DataFrameから候補列名のいずれかを探す。"""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _find_common_columns(
    df1: "pd.DataFrame", df2: "pd.DataFrame", candidates: list[str]
) -> list[str] | None:
    """2つのDataFrameに共通して存在する候補列を探す。"""
    result = []
    for col in candidates:
        if col in df1.columns and col in df2.columns:
            result.append(col)
    return result if result else None


if __name__ == "__main__":
    main()
