#!/usr/bin/env python3
"""特徴量凍結manifest生成スクリプト.

全モデルのFEATURE_COLSをJSON manifest + SHA256 hashで凍結する。
ParameterFreezeProtocol (Phase 13) のパターンを踏襲:
  - sort_keys=True + indent=2 で決定論的 (D-07)
  - SHA256 hashはモデル毎に記録 (D-08)

Usage:
    python scripts/freeze_feature_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# src/ をimport pathに追加 (他のscriptsと同じパターン)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.conformal_ev_model import ConformalEVModel
from models.ev_correction_model import EVCorrectionModel, PlaceEVCorrectionModel
from models.market_model import MarketModel
from models.place_ability_model import PlaceAbilityModel
from models.race_quality_screener import RaceQualityScreener
from models.regime_detector import RegimeDetector
from models.stage1_ability_model import AbilityModel
from models.two_stage_return_model import PlaceTwoStageModel, WinTwoStageModel
from models.wide_two_stage_model import WideTwoStageModel


def freeze_feature_manifest(output_path: Path) -> str:
    """全モデルのFEATURE_COLSを凍結manifestとして保存.

    各モデルのFEATURE_COLSをJSONにシリアライズし、SHA256 hashを生成。
    sort_keys=True + indent=2 で決定論的 (D-07)。

    Args:
        output_path: manifest保存先パス

    Returns:
        overall SHA256ハッシュ文字列
    """
    # 12モデルのFEATURE_COLSを収集 (plan 28-01指定のモデルリスト)
    model_specs: list[tuple[str, list[str]]] = [
        ("AbilityModel", AbilityModel.FEATURE_COLS),
        ("WinTwoStageModel", WinTwoStageModel.FEATURE_COLS),
        ("PlaceTwoStageModel.HIT", PlaceTwoStageModel.HIT_FEATURE_COLS),
        ("PlaceTwoStageModel.RETURN", PlaceTwoStageModel.RETURN_FEATURE_COLS),
        ("EVCorrectionModel", EVCorrectionModel.FEATURE_COLS),
        ("PlaceEVCorrectionModel", PlaceEVCorrectionModel.FEATURE_COLS),
        ("ConformalEVModel", ConformalEVModel.FEATURE_COLS),
        ("RegimeDetector", RegimeDetector.FEATURE_COLS),
        ("MarketModel", MarketModel.FEATURE_COLS),
        ("PlaceAbilityModel", PlaceAbilityModel.FEATURE_COLS),
        ("RaceQualityScreener", RaceQualityScreener.FEATURE_COLS),
        ("WideTwoStageModel.SHARED", WideTwoStageModel.SHARED_FEATURE_COLS),
    ]

    models_manifest: dict[str, dict] = {}
    for name, cols in model_specs:
        # sort_keys=True + indent=2 で決定論的 (D-07)
        cols_json = json.dumps(cols, sort_keys=True, indent=2)
        # SHA256 hashはモデル毎 (D-08)
        sha256 = hashlib.sha256(cols_json.encode()).hexdigest()
        models_manifest[name] = {
            "feature_count": len(cols),
            "features": cols,
            "sha256": sha256,
        }

    # 全体manifestのSHA256 (models部分のみでhash計算 → 全体JSONでoverall_sha256)
    full_manifest = {
        "version": "v1.7",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_sha256": "",  # placeholder, computed below
        "models": models_manifest,
    }

    # overall_sha256はversion+timestamp+models全体のJSONで計算
    # determinismのためoverall_sha256を除いた部分でhashを計算
    manifest_for_hash = {k: v for k, v in full_manifest.items() if k != "overall_sha256"}
    manifest_json = json.dumps(manifest_for_hash, sort_keys=True, indent=2, ensure_ascii=False)
    overall_sha256 = hashlib.sha256(manifest_json.encode()).hexdigest()
    full_manifest["overall_sha256"] = overall_sha256

    # 出力先ディレクトリを作成 (存在しない場合)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # JSON書き込み: sort_keys=True, indent=2, ensure_ascii=False (D-07: deterministic)
    output_path.write_text(
        json.dumps(full_manifest, sort_keys=True, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return overall_sha256


def main() -> None:
    """manifest生成のエントリポイント."""
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "data" / "feature_freeze_manifest.json"

    print("Feature Freeze Manifest Generator")
    print("=" * 50)
    print()

    overall_sha = freeze_feature_manifest(output_path)

    # manifest内容の概要を表示
    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    print(f"Output: {output_path}")
    print(f"Version: {manifest['version']}")
    print(f"Timestamp: {manifest['timestamp']}")
    print(f"Overall SHA256: {overall_sha[:12]}...")
    print(f"Models: {len(manifest['models'])}")
    print()

    for name, info in manifest["models"].items():
        print(f"  {name}: {info['feature_count']} features, sha256={info['sha256'][:8]}...")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
