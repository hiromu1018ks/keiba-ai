"""Tier 1ノイズ特徴量プルーニング統合スクリプト (AUDIT-02)

Tier 1除外 -> OOF安全性確認 (binaryモデルのみ) -> フルBT ROI検証 -> ロールバック/確定の
段階的プルーニングフローを実行する。

使い方:
  # ドライラン (FEATURE_COLSは変更しない)
  python scripts/prune_noise_features.py --model-dir data/models

  # 実際にFEATURE_COLSを編集
  python scripts/prune_noise_features.py --model-dir data/models --apply

  # フルBT ROI検証付き
  python scripts/prune_noise_features.py --model-dir data/models --apply --full-bt

  # ROI悪化時自動ロールバック付き
  python scripts/prune_noise_features.py --model-dir data/models --apply --full-bt --rollback
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

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

# ---------------------------------------------------------------------------
# モデル分類定数 (D-02)
# ---------------------------------------------------------------------------

BINARY_MODELS: set[str] = {"stage1", "win_hit", "place_hit"}
REGRESSION_MODELS: set[str] = {
    "win_return",
    "place_return",
    "ev_correction",
    "place_ev_correction",
    "conformal_ev",
}

# モデル名 -> (ファイルパス, 変数名) のマッピング
# スクリプトからの相対パスで定義 (実行時に絶対パスに解決)
_MODEL_COL_MAP: dict[str, list[tuple[str, str]]] = {
    "stage1": [
        ("src/models/stage1_ability_model.py", "AbilityModel.FEATURE_COLS"),
    ],
    "win_hit": [
        ("src/models/two_stage_return_model.py", "WinTwoStageModel.FEATURE_COLS"),
    ],
    "win_return": [
        ("src/models/two_stage_return_model.py", "WinTwoStageModel.FEATURE_COLS"),
    ],
    "place_hit": [
        ("src/models/two_stage_return_model.py", "PlaceTwoStageModel.HIT_FEATURE_COLS"),
    ],
    "place_return": [
        ("src/models/two_stage_return_model.py", "PlaceTwoStageModel.RETURN_FEATURE_COLS"),
    ],
    "ev_correction": [
        ("src/models/ev_correction_model.py", "EVCorrectionModel.FEATURE_COLS"),
    ],
    "place_ev_correction": [
        ("src/models/ev_correction_model.py", "PlaceEVCorrectionModel.FEATURE_COLS"),
    ],
    "conformal_ev": [
        ("src/models/conformal_ev_model.py", "ConformalEVModel.FEATURE_COLS"),
    ],
}

# モデルファイル名プレフィックス (analyze_feature_importance.py と同一)
_MODEL_FILE_PREFIX: dict[str, str] = {
    "stage1": "stage1",
    "win_hit": "win_hit",
    "win_return": "win_ret",
    "place_hit": "place_hit",
    "place_return": "place_ret",
    "ev_correction": "ev_corrector_p",
    "place_ev_correction": "place_ev_corrector_p",
    "conformal_ev": "conformal_ev",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _model_type(model_name: str) -> str:
    """モデル名からbinary/regressionを判定する。"""
    if model_name in BINARY_MODELS:
        return "binary"
    return "regression"


def _find_model_file(model_dir: str, model_name: str) -> str | None:
    """モデルディレクトリから.lgbファイルを検索する。"""
    prefix = _MODEL_FILE_PREFIX.get(model_name, model_name)
    # 優先: turf -> dirt -> 任意
    for surface in ["turf", "dirt"]:
        pattern = os.path.join(model_dir, f"{prefix}_{surface}.lgb")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    # フォールバック: 任意の該当ファイル
    pattern = os.path.join(model_dir, f"{prefix}_*.lgb")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    return None


_FILE_PREFIX_TO_DISPLAY: dict[str, str] = {
    "stage1": "stage1",
    "win_hit": "win_hit",
    "win_ret": "win_return",
    "place_hit": "place_hit",
    "place_ret": "place_return",
    "ev_corrector_p": "ev_correction",
    "place_ev_corrector_p": "place_ev_correction",
    "conformal_ev": "conformal_ev",
}


def _parse_model_filename(basename: str) -> tuple[str | None, str]:
    """ファイル名からモデル名とsurfaceを推定する。"""
    for surface in ["turf", "dirt"]:
        if basename.endswith(f"_{surface}"):
            name_part = basename[: -len(f"_{surface}")]
            display_name = _FILE_PREFIX_TO_DISPLAY.get(name_part)
            return display_name, surface
    return None, ""


def _load_lgb_models(model_dir: str) -> dict[str, Any]:
    """model_dir から .lgb ファイルをロードする。"""
    import lightgbm as lgb  # noqa: F811

    lgb_files = glob.glob(os.path.join(model_dir, "*.lgb"))
    models: dict[str, Any] = {}
    for lgb_file in sorted(lgb_files):
        basename = os.path.splitext(os.path.basename(lgb_file))[0]
        name, surface = _parse_model_filename(basename)
        if name is None:
            continue
        model_key = f"{name}_{surface}"
        try:
            booster = lgb.Booster(model_file=lgb_file)
            models[model_key] = booster
            logger.info("モデル読み込み: %s -> %s (%s)", lgb_file, name, surface)
        except Exception as e:
            logger.warning("モデル読み込み失敗 %s: %s", lgb_file, e)
    return models


def _load_features(features_path: str) -> Any:
    """特徴量parquetを読み込む。"""
    import pandas as pd  # noqa: F811

    if not os.path.exists(features_path):
        logger.error("特徴量ファイルが見つかりません: %s", features_path)
        return None
    return pd.read_parquet(features_path)


def _ensure_output_dir(path: str) -> None:
    """出力先ディレクトリを自動作成する。"""
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def _save_json(data: dict[str, Any], path: str) -> None:
    """JSON出力 (numpy型をPython型に変換)。"""
    import numpy as np  # noqa: F811

    def convert(obj: Any) -> Any:
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

    _ensure_output_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert(data), f, indent=2, ensure_ascii=False)
    logger.info("JSON保存: %s", path)


# ---------------------------------------------------------------------------
# FEATURE_COLS ファイル編集 (--apply)
# ---------------------------------------------------------------------------


def _backup_file(file_path: str) -> str:
    """ファイルのバックアップを作成する。"""
    backup_path = file_path + ".backup"
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        logger.info("バックアップ作成: %s", backup_path)
    return backup_path


def _edit_feature_cols_in_file(
    file_path: str,
    var_name: str,
    features_to_remove: list[str],
) -> bool:
    """FEATURE_COLS定義から指定特徴量を削除する (行ベーステキスト編集)。

    Returns:
        True if any features were removed, False otherwise.
    """
    if not features_to_remove:
        return False

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    remove_set = set(features_to_remove)
    class_name, _, attr_name = var_name.partition(".")
    # 変数定義ブロックを探す
    # 例: "FEATURE_COLS: list[str] = [" または "FEATURE_COLS = ["
    # クラス変数の場合: クラス定義内の行
    in_target_class = False
    in_target_list = False
    new_lines: list[str] = []
    removed_count = 0

    for line in lines:
        stripped = line.strip()

        # クラス定義の追跡
        if class_name and re.match(rf"^class {class_name}\b", stripped):
            in_target_class = True
        elif class_name and re.match(r"^class \w+", stripped):
            in_target_class = False

        # リスト開始の検出
        if (in_target_class or not class_name) and not in_target_list:
            # "ATTR_NAME: list[str] = [" or "ATTR_NAME = ["
            if re.match(rf"^\s*{attr_name}\s*(:\s*\w+\[.*?\]\s*)?=\s*\[", stripped):
                in_target_list = True
                # = より後の部分に ] があれば1行リスト (型注釈の ] は = の前にある)
                eq_pos = stripped.find("=")
                after_eq = stripped[eq_pos + 1:] if eq_pos >= 0 else ""
                if "]" in after_eq:
                    # 1行リスト: インラインで特徴量をフィルタ
                    in_target_list = False
                    list_start = after_eq.index("[")
                    list_end = after_eq.rindex("]")
                    list_content = after_eq[list_start + 1:list_end]
                    items = re.findall(r'"([^"]+)"', list_content)
                    filtered = [f'"{item}"' for item in items if item not in remove_set]
                    removed_count += len(items) - len(filtered)
                    # 元の行のプレフィックスを保持して1行リストを再構築
                    line_prefix = line[:line.index("=") + 1]
                    new_line = line_prefix + " [" + ", ".join(filtered) + "]\n"
                    new_lines.append(new_line)
                    continue

        if in_target_list:
            # リスト終了の検出
            if stripped == "]" or stripped.startswith("]"):
                in_target_list = False
                new_lines.append(line)
                continue

            # 文字列リテラルから特徴量名を抽出
            # 例: "feature_name", または "feature_name"
            match = re.match(r'^\s*"([^"]+)"\s*,?\s*(#.*)?$', stripped)
            if match:
                feat_name = match.group(1)
                if feat_name in remove_set:
                    removed_count += 1
                    logger.info("  除外: %s", feat_name)
                    continue  # 行をスキップ = 特徴量を削除
            new_lines.append(line)
        else:
            new_lines.append(line)

    if removed_count > 0:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        logger.info("%s から %d 特徴量を除外しました", var_name, removed_count)

    return removed_count > 0


def apply_pruning(
    tier_result: dict[str, dict[str, Any]],
    safety_result: dict[str, dict[str, Any]],
) -> dict[str, bool]:
    """安全確認を通過したTier 1特徴量を各モデルのFEATURE_COLSから除外する。

    Returns:
        {model_name: applied} のdict。
    """
    applied: dict[str, bool] = {}

    for model_name, tier_data in tier_result.items():
        tier1_features = tier_data.get("tier1", [])
        if not tier1_features:
            applied[model_name] = False
            continue

        # 安全確認不通過のモデルはスキップ
        model_safety = safety_result.get(model_name, {})
        if not model_safety.get("safety_passed", False):
            logger.warning(
                "モデル '%s' は安全確認未通過のため除外をスキップ", model_name,
            )
            applied[model_name] = False
            continue

        # 対象ファイル + 変数名
        entries = _MODEL_COL_MAP.get(model_name, [])
        if not entries:
            logger.warning("モデル '%s' のファイルマッピングが未定義", model_name)
            applied[model_name] = False
            continue

        any_removed = False
        for rel_path, var_name in entries:
            file_path = os.path.join(ROOT, rel_path)
            if not os.path.exists(file_path):
                logger.warning("ファイルが見つかりません: %s", file_path)
                continue
            # バックアップ作成
            _backup_file(file_path)
            # 編集実行
            removed = _edit_feature_cols_in_file(file_path, var_name, tier1_features)
            if removed:
                any_removed = True

        applied[model_name] = any_removed

    return applied


def rollback_files() -> list[str]:
    """.backupファイルから元のFEATURE_COLSに復元する。

    Returns:
        復元されたファイルパスのリスト。
    """
    restored: list[str] = []
    # src/models/ 内の .backup ファイルを検索
    for rel_path_entries in _MODEL_COL_MAP.values():
        for rel_path, _ in rel_path_entries:
            file_path = os.path.join(ROOT, rel_path)
            backup_path = file_path + ".backup"
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, file_path)
                os.remove(backup_path)
                restored.append(file_path)
                logger.info("ロールバック: %s を復元", file_path)
    return restored


# ---------------------------------------------------------------------------
# OOF安全性確認 (D-03)
# ---------------------------------------------------------------------------


def run_oof_safety_check(
    model_name: str,
    model: Any,
    df: Any,
    tier1_features: list[str],
    safety_threshold: float = 0.005,
) -> dict[str, Any]:
    """単一モデルのOOF安全性確認を実行する。

    binaryモデル: validate_noise_removal() でlogloss/AUC比較。
    regressionモデル: スキップしてsafety_passed=True。

    Args:
        model_name: モデル名
        model: lgb.Booster
        df: 特徴量DataFrame
        tier1_features: Tier 1特徴量リスト
        safety_threshold: logloss悪化許容閾値 (割合)

    Returns:
        safety確認結果dict
    """
    mtype = _model_type(model_name)

    if mtype == "regression":
        return {
            "model_type": "regression",
            "oof_safety": None,
            "safety_skipped_reason": "regression model - safety validated via full BT",
            "safety_passed": True,
        }

    if not tier1_features:
        return {
            "model_type": "binary",
            "oof_safety": None,
            "safety_skipped_reason": "no Tier 1 features to validate",
            "safety_passed": True,
        }

    # binaryモデル: validate_noise_removal() を実行
    from features.win_feature_analysis import validate_noise_removal  # noqa: F811

    # モデルの特徴量名に含まれるTier 1特徴量のみを対象とする
    model_features = set(model.feature_name())
    relevant_noise = [f for f in tier1_features if f in model_features]

    if not relevant_noise:
        return {
            "model_type": "binary",
            "oof_safety": None,
            "safety_skipped_reason": "no Tier 1 features found in model",
            "safety_passed": True,
        }

    try:
        metrics = validate_noise_removal(model, df, relevant_noise)
    except Exception as e:
        logger.warning("OOF安全性確認エラー (%s): %s", model_name, e)
        return {
            "model_type": "binary",
            "oof_safety": None,
            "safety_skipped_reason": f"validation error: {e}",
            "safety_passed": False,
        }

    original_logloss = metrics["original_logloss"]
    new_logloss = metrics["new_logloss"]
    logloss_ratio = (
        (new_logloss - original_logloss) / abs(original_logloss)
        if original_logloss != 0
        else 0.0
    )

    safety_passed = logloss_ratio <= safety_threshold

    if not safety_passed:
        logger.warning(
            "モデル '%s' のOOF安全性確認FAIL: logloss悪化 %.2f%% "
            "(閾値 %.2f%%). 除外をスキップ: %s",
            model_name,
            logloss_ratio * 100,
            safety_threshold * 100,
            relevant_noise,
        )

    return {
        "model_type": "binary",
        "oof_safety": metrics,
        "safety_passed": safety_passed,
        "logloss_degradation_ratio": logloss_ratio,
    }


# ---------------------------------------------------------------------------
# フルBT ROI検証 (D-04 Step 2)
# ---------------------------------------------------------------------------


def run_full_bt_roi_check(
    baseline_roi: float = 0.844,
    bt_command: str | None = None,
) -> dict[str, Any]:
    """フルバックテストを実行し、ROI比較結果を返す。

    Args:
        baseline_roi: 比較基準ROI (default: 0.844 = 84.4%)
        bt_command: バックテスト実行コマンド (Noneならデフォルト)

    Returns:
        ROI比較結果dict
    """
    if bt_command is None:
        bt_command = (
            "python scripts/run_backtest.py "
            "--years 2024 --train-window 4 --ensemble --calibration-bt --report"
        )

    logger.info("フルBT実行: %s", bt_command)

    # subprocessでバックテストを実行
    result = subprocess.run(
        shlex.split(bt_command),
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    # バックテスト失敗時は即座にエラー結果を返す (古い結果JSONの誤読防止)
    if result.returncode != 0:
        logger.error(
            "バックテスト実行失敗 (returncode=%d): %s",
            result.returncode, result.stderr,
        )
        return {
            "error": "backtest execution failed",
            "bt_returncode": result.returncode,
            "roi_improved": False,
            "baseline_roi": baseline_roi,
            "pruned_roi": 0.0,
            "roi_delta": -baseline_roi,
        }

    # バックテスト結果JSONからROIを読み取り
    bt_result_path = os.path.join(ROOT, "backtest_result.json")
    pruned_roi = 0.0

    if os.path.exists(bt_result_path):
        with open(bt_result_path, "r", encoding="utf-8") as f:
            bt_data = json.load(f)
        pruned_roi = bt_data.get("total_roi", 0.0)
    else:
        logger.warning("バックテスト結果が見つかりません: %s", bt_result_path)

    roi_delta = pruned_roi - baseline_roi
    roi_improved = pruned_roi >= baseline_roi

    comparison = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline_roi": baseline_roi,
        "baseline_source": "v1.5 (Phase 22 result)",
        "pruned_roi": pruned_roi,
        "roi_delta": roi_delta,
        "roi_improved": roi_improved,
        "bt_command": bt_command,
        "bt_returncode": result.returncode,
    }

    # 結果をJSONに保存
    _save_json(comparison, os.path.join(ROOT, "data", "audit", "roi_comparison.json"))

    if roi_improved:
        logger.info(
            "ROI改善: %.4f -> %.4f (delta: +%.4f). プルーニング確定。",
            baseline_roi, pruned_roi, roi_delta,
        )
    else:
        logger.warning(
            "ROI悪化: %.4f -> %.4f (delta: %.4f). ロールバックが必要。",
            baseline_roi, pruned_roi, roi_delta,
        )

    return comparison


# ---------------------------------------------------------------------------
# ロールバック + 原因分析 (D-05)
# ---------------------------------------------------------------------------


def run_rollback_with_cause_analysis(
    roi_comparison: dict[str, Any],
    tier_result: dict[str, dict[str, Any]],
    safety_result: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """ROI悪化時のロールバック + 原因分析を実行する。

    Args:
        roi_comparison: run_full_bt_roi_check() の戻り値
        tier_result: Tier分類結果
        safety_result: OOF安全性確認結果

    Returns:
        原因分析レポートdict
    """
    import pandas as pd  # noqa: F811

    # Step R1: .backupから復元
    restored_files = rollback_files()
    logger.info("ロールバック完了: %dファイルを復元", len(restored_files))

    # Step R2: バックテストのbet_historyを読み込み
    bt_csv_pattern = os.path.join(ROOT, "data", "backtest", "bt_2024_*.csv")
    bt_csv_files = glob.glob(bt_csv_pattern)
    bet_history: list[dict[str, Any]] = []

    for csv_file in bt_csv_files:
        try:
            bt_df = pd.read_csv(csv_file)
            for _, row in bt_df.iterrows():
                bet_history.append(row.to_dict())
        except Exception as e:
            logger.warning("BT CSV読み込み失敗 %s: %s", csv_file, e)

    # Step R3: 原因分析
    from backtest.validation_report import generate_cause_analysis  # noqa: F811

    cause_analysis: dict[str, Any]
    if bet_history:
        cause_analysis = generate_cause_analysis(bet_history)
    else:
        cause_analysis = {"error": "No bet_history available for cause analysis"}

    # Step R4: 原因分析レポートを構築
    tier1_removed_per_model: dict[str, list[str]] = {}
    for model_name, tier_data in tier_result.items():
        model_safety = safety_result.get(model_name, {})
        if model_safety.get("safety_passed", False):
            tier1_removed_per_model[model_name] = tier_data.get("tier1", [])

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline_roi": roi_comparison.get("baseline_roi", 0.844),
        "pruned_roi": roi_comparison.get("pruned_roi", 0.0),
        "rollback_performed": True,
        "restored_files": restored_files,
        "cause_analysis": cause_analysis,
        "tier1_removed_per_model": tier1_removed_per_model,
        "recommendation": (
            "Tier 1 features may be providing regularization benefit. "
            "Consider keeping features with gain>0 in some models."
        ),
    }

    # Step R5: 保存 + ログ出力
    output_path = os.path.join(ROOT, "data", "audit", "cause_analysis.json")
    _save_json(report, output_path)
    logger.warning(
        "ロールバック完了。原因分析レポート: %s", output_path,
    )

    return report


# ---------------------------------------------------------------------------
# メインフロー
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tier 1ノイズ特徴量プルーニング統合スクリプト (AUDIT-02)",
    )
    parser.add_argument(
        "--model-dir",
        default="data/models",
        help="モデルファイルディレクトリ (default: data/models)",
    )
    parser.add_argument(
        "--features-path",
        default="data/features/horse_features.parquet",
        help="OOF特徴量parquetパス (default: data/features/horse_features.parquet)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="実際にFEATURE_COLSを編集する (dry-runモードがデフォルト)",
    )
    parser.add_argument(
        "--full-bt",
        action="store_true",
        help="フルバックテストでROI検証を実行する (D-04 Step 2)",
    )
    parser.add_argument(
        "--output",
        default="data/audit/pruning_validation.json",
        help="検証結果JSONの出力先 (default: data/audit/pruning_validation.json)",
    )
    parser.add_argument(
        "--safety-threshold",
        type=float,
        default=0.005,
        help="logloss悪化許容閾値 (default: 0.005 = 0.5%%)",
    )
    parser.add_argument(
        "--baseline-roi",
        type=float,
        default=0.844,
        help="ROI比較ベースライン (default: 0.844 = 84.4%%)",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="ROI悪化時に自動ロールバックする (--full-btと併用)",
    )
    args = parser.parse_args()

    model_dir = os.path.join(ROOT, args.model_dir)
    features_path = os.path.join(ROOT, args.features_path)
    output_path = os.path.join(ROOT, args.output)

    logger.info("=== Tier 1ノイズ特徴量プルーニング ===")
    logger.info("モデルディレクトリ: %s", model_dir)
    logger.info("特徴量ファイル: %s", features_path)
    logger.info("モード: %s", "apply" if args.apply else "dry-run")

    # Step 1: モデルファイルをロード
    models = _load_lgb_models(model_dir)
    if not models:
        logger.error("読み込み可能なモデルがありません: %s/*.lgb", model_dir)
        sys.exit(1)

    # Step 2: 特徴量データをロード
    import numpy as np  # noqa: F811

    df = _load_features(features_path)
    if df is None:
        logger.error("特徴量データの読み込みに失敗しました")
        sys.exit(1)

    # Step 3: compute_all_model_importance() で重要度を計算
    from features.win_feature_analysis import (  # noqa: F811
        classify_feature_tiers,
        compute_all_model_importance,
    )

    # ターゲット構築
    targets: dict[str, np.ndarray] = {}
    for model_key, booster in models.items():
        key_base = (
            model_key.rsplit("_", 1)[0] if "_" in model_key else model_key
        )
        name, _ = _parse_model_filename(key_base)
        if name in BINARY_MODELS:
            if "kakuteijyuni" in df.columns:
                targets[model_key] = (df["kakuteijyuni"] == 1).astype(int).values

    pivot_df, metadata = compute_all_model_importance(
        models, df, targets, n_repeats=5,
    )

    # Step 4: classify_feature_tiers() でTier 1/2を分類
    tier_result = classify_feature_tiers(pivot_df, metadata)

    # Tier レポートも出力
    _save_json(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "models": tier_result,
        },
        os.path.join(ROOT, "data", "audit", "tier_report.json"),
    )

    # Step 5: 各モデルのOOF安全性確認
    safety_result: dict[str, dict[str, Any]] = {}
    for model_key, booster in models.items():
        # model_key は "name_surface" 形式
        # rsplit("_", 1) でモデル名とサーフェスを分離し、
        # モデル名をそのまま _model_type() に渡す
        parts = model_key.rsplit("_", 1)
        name = parts[0] if len(parts) > 1 else model_key

        tier_data = tier_result.get(model_key, {})
        tier1_features = tier_data.get("tier1", [])

        safety = run_oof_safety_check(
            name, booster, df, tier1_features, args.safety_threshold,
        )
        safety_result[model_key] = safety

    # Step 7: 検証結果JSON出力
    validation_output: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": {},
        "total_tier1_removed": 0,
        "total_tier2_flagged": 0,
    }

    for model_key in models:
        tier_data = tier_result.get(model_key, {})
        safety = safety_result.get(model_key, {})
        tier1 = tier_data.get("tier1", [])
        tier2 = tier_data.get("tier2", [])

        validation_output["models"][model_key] = {
            "tier1_features": tier1,
            "tier2_features": tier2,
            "model_type": safety.get("model_type", "unknown"),
            "oof_safety": safety.get("oof_safety"),
            "safety_skipped_reason": safety.get("safety_skipped_reason"),
            "safety_passed": safety.get("safety_passed", False),
            "applied": False,
        }
        if safety.get("safety_passed", False):
            validation_output["total_tier1_removed"] += len(tier1)
        validation_output["total_tier2_flagged"] += len(tier2)

    _save_json(validation_output, output_path)

    # Step 8: --apply 指定時のみFEATURE_COLSを編集
    if args.apply:
        applied = apply_pruning(tier_result, safety_result)
        for model_key, was_applied in applied.items():
            if model_key in validation_output["models"]:
                validation_output["models"][model_key]["applied"] = was_applied
        # 更新版を保存
        _save_json(validation_output, output_path)

    # Step B1-B5: --full-bt 指定時のフルBT ROI検証
    if args.full_bt and args.apply:
        roi_comparison = run_full_bt_roi_check(args.baseline_roi)

        if not roi_comparison.get("roi_improved", False) and args.rollback:
            # D-05: ロールバック + 原因分析
            run_rollback_with_cause_analysis(
                roi_comparison, tier_result, safety_result,
            )

    logger.info("=== プルーニング完了 ===")
    logger.info("結果: %s", output_path)


if __name__ == "__main__":
    main()
