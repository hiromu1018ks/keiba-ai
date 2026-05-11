"""特徴量重要度分析CLI

使い方:
  python scripts/analyze_feature_importance.py --help
  python scripts/analyze_feature_importance.py --model-dir data/models/
  python scripts/analyze_feature_importance.py \\
      --model-dir data/models/ --shap-threshold 0.001 --output report.csv
  python scripts/analyze_feature_importance.py \\
      --model-dir data/models/ --all-models --format both
  python scripts/analyze_feature_importance.py \\
      --model-dir data/models/ --model win_hit --surface turf
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

# モデルファイル名パターンと表示名のマッピング
MODEL_PATTERNS: dict[str, list[str]] = {
    "stage1": ["stage1"],
    "win_hit": ["win_hit"],
    "win_return": ["win_ret"],
    "place_hit": ["place_hit"],
    "place_return": ["place_ret"],
    "ev_correction": ["ev_corrector_p"],
    "place_ev_correction": ["place_ev_corrector_p"],
}

# --model choices (7 models)
MODEL_CHOICES = [
    "stage1", "win_hit", "win_return",
    "place_hit", "place_return",
    "ev_correction", "place_ev_correction",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="特徴量重要度をSHAP/gain/permutationで分析",
    )
    # 既存引数 (後方互換)
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
    parser.add_argument(
        "--surface",
        choices=["turf", "dirt"],
        default="turf",
        help="解析対象のサーフェス (default: turf)",
    )
    # 新規引数
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="全モデルの重要度を一括計算",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default=None,
        help="単一モデル指定 (default: win_hit)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json", "both"],
        default="both",
        help="出力形式 (default: both)",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        help="permutation importance試行回数 (default: 5)",
    )
    parser.add_argument(
        "--output-json",
        default="feature_importance_report.json",
        help="JSON出力パス (default: feature_importance_report.json)",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.rstrip("/")

    if args.all_models:
        _run_all_models(args, model_dir)
    else:
        _run_single_model(args, model_dir)

    print("\n完了")


def _run_single_model(args: argparse.Namespace, model_dir: str) -> None:
    """単一モデルの分析 (既存機能、後方互換)。"""
    model_name = args.model or "win_hit"
    model_path = _find_model_file(model_dir, model_name=model_name, preferred_surface=args.surface)
    if model_path is None:
        logger.error(
            "モデルファイルが見つかりません: %s/%s_*.lgb を確認してください",
            model_dir, _get_file_prefix(model_name),
        )
        sys.exit(1)

    import lightgbm as lgb

    logger.info("モデル読み込み: %s", model_path)
    model = lgb.Booster(model_file=model_path)

    # 特徴量データの読み込み
    result = _load_features_for_analysis(model)
    if result is None:
        logger.error("特徴量データの読み込みに失敗しました")
        sys.exit(1)
    features_df, _ = result

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


def _run_all_models(args: argparse.Namespace, model_dir: str) -> None:
    """全モデルのgain+permutation重要度を一括計算。"""
    import lightgbm as lgb
    import numpy as np

    from features.win_feature_analysis import compute_all_model_importance

    # model_dir内の全.lgbファイルを検索
    lgb_files = glob.glob(os.path.join(model_dir, "*.lgb"))
    if not lgb_files:
        logger.error("モデルファイルが見つかりません: %s/*.lgb", model_dir)
        sys.exit(1)

    models: dict[str, lgb.Booster] = {}
    file_info: dict[str, tuple[str, str]] = {}  # model_key -> (model_name, surface)

    for lgb_file in sorted(lgb_files):
        basename = os.path.splitext(os.path.basename(lgb_file))[0]
        name, surface = _parse_model_filename(basename)
        if name is None:
            continue
        model_key = f"{name}_{surface}"
        try:
            booster = lgb.Booster(model_file=lgb_file)
            models[model_key] = booster
            file_info[model_key] = (name, surface)
            logger.info("モデル読み込み: %s -> %s (%s)", lgb_file, name, surface)
        except Exception as e:
            logger.warning("モデル読み込み失敗 %s: %s", lgb_file, e)

    if not models:
        logger.error("読み込み可能なモデルがありません")
        sys.exit(1)

    # 特徴量データの読み込み (全特徴量を含むDataFrame)
    # 最初のモデルを使ってParquetStoreから読み込み
    first_model = next(iter(models.values()))
    result = _load_features_for_analysis(first_model)
    if result is None:
        logger.error("特徴量データの読み込みに失敗しました")
        sys.exit(1)
    base_features_df, target_series = result

    # 各モデルの特徴量列をmodel.feature_name()で決定
    # base_features_dfに全特徴量が含まれている前提で拡張
    all_feature_cols: set[str] = set(base_features_df.columns.tolist())
    for booster in models.values():
        all_feature_cols.update(booster.feature_name())

    # 欠損列をNaNで補完
    for col in all_feature_cols:
        if col not in base_features_df.columns:
            base_features_df[col] = np.nan

    # ターゲット構築
    targets: dict[str, np.ndarray] = {}
    for model_key, booster in models.items():
        name, surface = file_info[model_key]
        # hit系モデル (binary): y=1/0
        if "hit" in name or name == "stage1":
            if target_series is not None and len(target_series) == len(base_features_df):
                targets[model_key] = (target_series == 1).astype(int).values
        # return系モデル (regression): 確定オッズ等
        elif "ret" in name:
            # 回帰モデルのターゲットはOOFデータから取得困難なためダミー
            # gain importanceのみ計算
            pass
        # ev_correction系: gainのみ
        elif "corrector" in name or "correction" in name:
            pass

    # compute_all_model_importance 呼び出し
    pivot_df, metadata = compute_all_model_importance(
        models,
        base_features_df,
        targets,
        n_repeats=args.n_repeats,
    )

    # 結果表示
    print(f"\n=== 全モデル特徴量重要度 ({len(models)}モデル) ===")
    display_cols = ["feature"] + [c for c in pivot_df.columns if c != "feature"][:6]
    print(pivot_df[display_cols].head(30).to_string(index=False))

    # 出力
    if args.format in ("csv", "both"):
        pivot_df.to_csv(args.output, index=False)
        logger.info("CSV保存: %s (%d行, %d列)", args.output, len(pivot_df), len(pivot_df.columns))

    if args.format in ("json", "both"):
        # numpy型をPython型に変換
        _save_json(metadata, args.output_json)
        logger.info("JSON保存: %s", args.output_json)


def _save_json(data: dict, path: str) -> None:
    """numpy型を再帰的にPython型に変換してJSON保存。"""

    def convert(obj: object) -> object:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert(data), f, indent=2, ensure_ascii=False)


def _get_file_prefix(model_name: str) -> str:
    """モデル表示名からファイルプレフィックスを取得。"""
    prefix_map = {
        "stage1": "stage1",
        "win_hit": "win_hit",
        "win_return": "win_ret",
        "place_hit": "place_hit",
        "place_return": "place_ret",
        "ev_correction": "ev_corrector_p",
        "place_ev_correction": "place_ev_corrector_p",
    }
    return prefix_map.get(model_name, model_name)


def _find_model_file(
    model_dir: str,
    *,
    model_name: str = "win_hit",
    preferred_surface: str = "turf",
) -> str | None:
    """モデルディレクトリから指定モデルファイルを検索。"""
    file_prefix = _get_file_prefix(model_name)

    # 指定サーフェスを優先、次に他のサーフェスを試す
    seen: set[str] = set()
    for surface in [preferred_surface, "turf", "dirt"]:
        if surface in seen:
            continue
        seen.add(surface)
        pattern = os.path.join(model_dir, f"{file_prefix}_{surface}.lgb")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]

    # フォールバック: 任意の該当モデル
    pattern = os.path.join(model_dir, f"{file_prefix}_*.lgb")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]

    return None


def _parse_model_filename(basename: str) -> tuple[str | None, str]:
    """モデルファイル名(basename拡張子なし)からモデル名とsurfaceを推定。

    例: "win_hit_turf" -> ("win_hit", "turf")
         "ev_corrector_p_turf" -> ("ev_corrector_p", "turf")
    """
    # surface候補で終わるかチェック
    for surface in ["turf", "dirt"]:
        if basename.endswith(f"_{surface}"):
            name_part = basename[: -len(f"_{surface}")]
            # 表示名にマッピング
            display_name = _file_prefix_to_display(name_part)
            return display_name, surface
    return None, ""


def _file_prefix_to_display(prefix: str) -> str:
    """ファイルプレフィックスから表示名に変換。"""
    reverse_map = {
        "stage1": "stage1",
        "win_hit": "win_hit",
        "win_ret": "win_return",
        "place_hit": "place_hit",
        "place_ret": "place_return",
        "ev_corrector_p": "ev_correction",
        "ev_corrector_e": "ev_correction_e",
        "place_ev_corrector_p": "place_ev_correction",
        "place_ev_corrector_e": "place_ev_correction_e",
    }
    return reverse_map.get(prefix, prefix)


def _load_features_for_analysis(
    model: "lgb.Booster",  # noqa: F821
) -> "tuple[pd.DataFrame, pd.Series | None] | None":  # noqa: F821
    """分析用特徴量データを読み込む。

    モデルの特徴量名に対応する列を持つDataFrameを生成。
    実データ分析にはParquetStoreからデータを読み込む必要があるが、
    モデル読み込み単体でも動作確認できるようにnull値でフォールバック。

    Returns:
        (features_df, target_series) のタプル。
        target_seriesはkakuteijyuni列(利用可能な場合)。
        読み込み失敗時はNone。
    """
    import pandas as pd  # noqa: F401

    try:
        from db.parquet_store import ParquetStore

        store = ParquetStore()
        if store.exists("features", "horse_features"):
            feat_df = store.load("features", "horse_features")
            feature_names = model.feature_name()
            available = [c for c in feature_names if c in feat_df.columns]
            if len(available) == len(feature_names):
                logger.info("特徴量データ読み込み: %d行, %d列", len(feat_df), len(feature_names))
                target = feat_df.get("kakuteijyuni")  # type: ignore[union-attr]
                return feat_df[feature_names], target
            logger.warning(
                "特徴量データに一部列が欠落: %d/%d",
                len(available),
                len(feature_names),
            )
    except Exception as e:
        logger.warning("ParquetStoreからの特徴量読み込み失敗: %s", e)

    # 実データが読み込めない場合、呼び出し元にNoneを返して処理を委ねる
    logger.error(
        "実データの読み込みに失敗しました。ダミーデータでの分析は無意味です。"
        "ParquetStoreに特徴量データが存在することを確認してください。"
    )
    return None


def _auto_exclude_and_validate(
    model: "lgb.Booster",  # noqa: F821
    features_df: "pd.DataFrame",  # noqa: F821
    noise_features: list[str],
) -> None:
    """ノイズ除外 + 再学習検証。"""
    from features.win_feature_analysis import validate_noise_removal
    from models.two_stage_return_model import WinTwoStageModel

    print(f"\n=== 自動除外モード: {len(noise_features)}件のノイズ特徴量を除外 ===")

    # 再学習検証
    if "kakuteijyuni" in features_df.columns:
        metrics = validate_noise_removal(model, features_df, noise_features)
        ol = metrics['original_logloss']
        oa = metrics['original_auc']
        nl = metrics['new_logloss']
        na = metrics['new_auc']
        print(f"  元モデル logloss: {ol:.6f}  AUC: {oa:.6f}")
        print(f"  新モデル logloss: {nl:.6f}  AUC: {na:.6f}")

        logloss_diff = metrics["new_logloss"] - metrics["original_logloss"]
        auc_diff = metrics["new_auc"] - metrics["original_auc"]
        print(f"  差分: logloss {logloss_diff:+.6f}  AUC {auc_diff:+.6f}")

        if logloss_diff > 0:
            logger.warning("logloss悪化: ノイズ除外による精度低下の可能性があります")
    else:
        print("  kakuteijyuni列がないため再学習検証をスキップ")

    # FEATURE_COLSから除外 (注意: クラス変数を破壊的変更 — プロセス全体に影響)
    logger.warning(
        "クラスレベルの FEATURE_COLS を破壊的に変更します "
        "(プロセス内の他以降の処理に影響): %d件除外 -> %d特徴量",
        len(noise_features),
        len(WinTwoStageModel.FEATURE_COLS) - len(noise_features),
    )
    WinTwoStageModel.remove_noise_features(noise_features)
    print(f"  FEATURE_COLS更新: {len(WinTwoStageModel.FEATURE_COLS)}特徴量")


if __name__ == "__main__":
    main()
