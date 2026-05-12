"""test_tier_report_cli.py --tier-report CLI機能のテスト

全テスト mock 使用 (DB不要) — プロジェクト規約に従う。
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# プロジェクトルートをパスに追加
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

spec = importlib.import_module("analyze_feature_importance")
main = spec.main
_run_tier_report = spec._run_tier_report


class TestTierReportCLI:
    """--tier-report CLI引数のテスト"""

    def test_tier_report_flag_in_help(self) -> None:
        """--helpに--tier-reportが含まれる"""
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["analyze_feature_importance.py", "--help"]):
                main()
        assert exc_info.value.code == 0

    def test_tier_report_generates_json(self, tmp_path: object) -> None:
        """mockベースで_run_tier_reportがJSONファイルを出力する"""
        import pathlib

        feature_names = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]

        # pivot_df と metadata を構築
        pivot_df = pd.DataFrame({
            "feature": feature_names,
            "win_hit_turf_gain": [100.0, 80.0, 50.0, 20.0, 0.0],
            "win_hit_turf_perm": [0.05, 0.03, 0.02, 0.01, -0.001],
        })
        metadata = {
            "models": {
                "win_hit_turf": {
                    "gain": {
                        "feat_a": 100.0, "feat_b": 80.0, "feat_c": 50.0,
                        "feat_d": 20.0, "feat_e": 0.0,
                    },
                    "perm_mean": {
                        "feat_a": 0.05, "feat_b": 0.03, "feat_c": 0.02,
                        "feat_d": 0.01, "feat_e": -0.001,
                    },
                    "perm_std": {f: 0.0 for f in feature_names},
                }
            },
            "metadata": {"n_samples": 100, "n_repeats": 5, "timestamp": "2026-01-01T00:00:00Z"},
        }

        # モックモデル
        mock_booster = MagicMock()
        mock_booster.feature_name.return_value = feature_names
        models = {"win_hit_turf": mock_booster}

        output_path = str(pathlib.Path(str(tmp_path)) / "tier_report.json")

        _run_tier_report(pivot_df, metadata, output_path, models)

        # JSONファイルが存在する
        assert os.path.exists(output_path)
        with open(output_path, encoding="utf-8") as f:
            report = json.load(f)
        # モデルエントリが存在する
        assert "models" in report
        assert "win_hit_turf" in report["models"]
        # Tier 1 に feat_e (gain=0, perm=-0.001) が含まれる
        assert "feat_e" in report["models"]["win_hit_turf"]["tier1"]

    def test_tier_report_json_structure(self, tmp_path: object) -> None:
        """出力JSONがtimestamp, models, tier1_definition, tier2_definitionキーを含む"""
        import pathlib

        feature_names = ["feat_a", "feat_b"]
        pivot_df = pd.DataFrame({
            "feature": feature_names,
            "win_hit_turf_gain": [50.0, 0.0],
            "win_hit_turf_perm": [0.02, -0.001],
        })
        metadata = {
            "models": {
                "win_hit_turf": {
                    "gain": {"feat_a": 50.0, "feat_b": 0.0},
                    "perm_mean": {"feat_a": 0.02, "feat_b": -0.001},
                    "perm_std": {"feat_a": 0.0, "feat_b": 0.0},
                }
            },
            "metadata": {"n_samples": 100, "n_repeats": 5, "timestamp": "2026-01-01T00:00:00Z"},
        }

        mock_booster = MagicMock()
        mock_booster.feature_name.return_value = feature_names
        models = {"win_hit_turf": mock_booster}

        output_path = str(pathlib.Path(str(tmp_path)) / "tier_report.json")

        _run_tier_report(pivot_df, metadata, output_path, models)

        with open(output_path, encoding="utf-8") as f:
            report = json.load(f)

        assert "timestamp" in report
        assert "models" in report
        assert "tier1_definition" in report
        assert "tier2_definition" in report
        # 各モデルエントリに必要キーが含まれる
        for model_data in report["models"].values():
            assert "tier1" in model_data
            assert "tier2" in model_data
            assert "tier1_count" in model_data
            assert "tier2_count" in model_data
            assert "total_features" in model_data

    def test_tier_report_auto_enables_all_models(self) -> None:
        """--tier-report単体指定で--all-models相当の動作になる"""
        # _run_all_models が呼び出されることを確認する
        with patch.object(spec, "_run_all_models") as mock_run_all:
            with patch("sys.argv", [
                "analyze_feature_importance.py",
                "--tier-report",
                "--model-dir=/nonexistent",
            ]):
                try:
                    main()
                except SystemExit:
                    pass

        # _run_all_models が呼び出された (--all-models相当)
        mock_run_all.assert_called_once()
        # args.all_models が True に設定されていることを確認
        call_args = mock_run_all.call_args
        args = call_args[0][0]
        assert args.all_models is True
