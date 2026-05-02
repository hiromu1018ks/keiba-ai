"""特徴量重要度分析CLI

使い方:
  python scripts/analyze_feature_importance.py --help
  python scripts/analyze_feature_importance.py --model-dir data/models/
  python scripts/analyze_feature_importance.py --model-dir data/models/ --shap-threshold 0.001 --output report.csv
"""

from __future__ import annotations

import argparse
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
        description="WinTwoStageModel.hit_modelの特徴量重要度をSHAP/gainで分析",
    )
    parser.add_argument(
        "--model-dir",
        default="data/models/",
        help="学習済みモデルのディレクトリ (default: data/models/)",
    )
    parser.add_argument(
        "--shap-threshold",
        type=float,
        default=0.001,
        help="ノイズ判定SHAP閾値 (default: 0.001)",
    )
    parser.add_argument(
        "--gain-threshold",
        type=float,
        default=0.0,
        help="ノイズ判定gain閾値 (default: 0.0)",
    )
    parser.add_argument(
        "--output",
        default="feature_importance_report.csv",
        help="出力CSVパス (default: feature_importance_report.csv)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="表示する上位特徴量数 (0=全件, default: 0)",
    )
    parser.add_argument(
        "--auto-exclude",
        action="store_true",
        help="指定時、ノイズ特徴量を自動でFEATURE_COLSから除外して再学習検証を実行",
    )
    args = parser.parse_args()

    # モデル読み込み
    model_dir = args.model_dir.rstrip("/")
    model_path = _find_model_file(model_dir)
    if model_path is None:
        logger.error(
            "モデルファイルが見つかりません: %s/win_hit_*.lgb を確認してください",
            model_dir,
        )
        sys.exit(1)

    import lightgbm as lgb

    logger.info("モデル読み込み: %s", model_path)
    model = lgb.Booster(model_file=model_path)

    # 特徴量データの読み込み (ParquetStore経由)
    features_df = _load_features_for_analysis(model)
    if features_df is None:
        logger.error("特徴量データの読み込みに失敗しました")
        sys.exit(1)

    # 分析実行
    from features.win_feature_analysis import (
        analyze_feature_importance,
        identify_noise_features,
    )

    importance_df = analyze_feature_importance(model, features_df, top_n=args.top_n)
    noise_features = identify_noise_features(
        importance_df,
        shap_threshold=args.shap_threshold,
        gain_threshold=args.gain_threshold,
    )

    # is_noise列を追加
    importance_df["is_noise"] = importance_df["feature"].isin(noise_features)

    # 結果表示
    display_count = min(30, len(importance_df))
    print(f"\n=== 特徴量重要度ランキング (Top {display_count}) ===")
    print(importance_df.head(display_count).to_string(index=False))

    if noise_features:
        print(f"\n=== ノイズ特徴量 ({len(noise_features)}件) ===")
        noise_df = importance_df[importance_df["is_noise"]]
        print(noise_df.to_string(index=False))
    else:
        print("\nノイズ特徴量は検出されませんでした")

    # CSV保存
    importance_df.to_csv(args.output, index=False)
    logger.info("レポート保存: %s (%d行)", args.output, len(importance_df))

    # 自動除外モード
    if args.auto_exclude and noise_features:
        _auto_exclude_and_validate(model, features_df, noise_features)

    print("\n完了")


def _find_model_file(model_dir: str) -> str | None:
    """モデルディレクトリからwin_hitモデルファイルを検索。"""
    import glob

    # turf/dirt の両方を試す (turf優先)
    for surface in ["turf", "dirt"]:
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


def _load_features_for_analysis(model: "lgb.Booster") -> "pd.DataFrame | None":
    """分析用特徴量データを読み込む。

    モデルの特徴量名に対応する列を持つダミーDataFrameを生成。
    実データ分析にはParquetStoreからデータを読み込む必要があるが、
    モデル読み込み単体でも動作確認できるようにnull値でフォールバック。
    """
    import pandas as pd

    try:
        from db.parquet_store import ParquetStore

        store = ParquetStore()
        if store.exists("features", "horse_features"):
            feat_df = store.load("features", "horse_features")
            feature_names = model.feature_name()
            available = [c for c in feature_names if c in feat_df.columns]
            if len(available) == len(feature_names):
                logger.info("特徴量データ読み込み: %d行, %d列", len(feat_df), len(feature_names))
                return feat_df[feature_names]
            logger.warning(
                "特徴量データに一部列が欠落: %d/%d",
                len(available),
                len(feature_names),
            )
    except Exception as e:
        logger.warning("ParquetStoreからの特徴量読み込み失敗: %s", e)

    # フォールバック: モデル特徴量名で空DataFrame
    logger.info("ダミー特徴量を生成します (実データはParquetStoreから取得してください)")
    feature_names = model.feature_name()
    return pd.DataFrame(0.0, index=range(100), columns=feature_names)


def _auto_exclude_and_validate(
    model: "lgb.Booster",
    features_df: "pd.DataFrame",
    noise_features: list[str],
) -> None:
    """ノイズ除外 + 再学習検証。"""
    from features.win_feature_analysis import validate_noise_removal
    from models.two_stage_return_model import WinTwoStageModel

    print(f"\n=== 自動除外モード: {len(noise_features)}件のノイズ特徴量を除外 ===")

    # 再学習検証
    if "kakuteijyuni" in features_df.columns:
        metrics = validate_noise_removal(model, features_df, noise_features)
        print(f"  元モデル logloss: {metrics['original_logloss']:.6f}  AUC: {metrics['original_auc']:.6f}")
        print(f"  新モデル logloss: {metrics['new_logloss']:.6f}  AUC: {metrics['new_auc']:.6f}")

        logloss_diff = metrics["new_logloss"] - metrics["original_logloss"]
        auc_diff = metrics["new_auc"] - metrics["original_auc"]
        print(f"  差分: logloss {logloss_diff:+.6f}  AUC {auc_diff:+.6f}")

        if logloss_diff > 0:
            logger.warning("logloss悪化: ノイズ除外による精度低下の可能性があります")
    else:
        print("  kakuteijyuni列がないため再学習検証をスキップ")

    # FEATURE_COLSから除外
    WinTwoStageModel.remove_noise_features(noise_features)
    print(f"  FEATURE_COLS更新: {len(WinTwoStageModel.FEATURE_COLS)}特徴量")


if __name__ == "__main__":
    main()
