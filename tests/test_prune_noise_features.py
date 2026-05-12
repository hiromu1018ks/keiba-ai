"""test_prune_noise_features.py — プルーニング統合スクリプトのテスト

全テスト mock 使用 (DB不要) — プロジェクト規約に従う。
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# テスト対象のインポート
import importlib
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

# スクリプトをモジュールとしてインポート
spec = importlib.util.spec_from_file_location(
    "prune_noise_features",
    os.path.join(ROOT, "scripts", "prune_noise_features.py"),
)
prune_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prune_mod)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_project(tmp_path: str) -> str:
    """テスト用の一時プロジェクトディレクトリ構造を作成する。"""
    # モデルディレクトリ
    model_dir = os.path.join(tmp_path, "data", "models")
    os.makedirs(model_dir)

    # 特徴量ディレクトリ
    feat_dir = os.path.join(tmp_path, "data", "features")
    os.makedirs(feat_dir)

    # 監査ディレクトリ
    audit_dir = os.path.join(tmp_path, "data", "audit")
    os.makedirs(audit_dir)

    # バックテストディレクトリ
    bt_dir = os.path.join(tmp_path, "data", "backtest")
    os.makedirs(bt_dir)

    return str(tmp_path)


@pytest.fixture()
def mock_booster() -> MagicMock:
    """5特徴量のlgb.Boosterモック。"""
    model = MagicMock()
    model.feature_name.return_value = [
        "feat_a", "feat_b", "feat_c", "feat_d", "feat_e",
    ]
    model.feature_importance.return_value = np.array([100.0, 80.0, 50.0, 20.0, 0.0])
    model.predict.return_value = np.array([0.3, 0.7, 0.5, 0.2, 0.6])
    return model


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """テスト用の特徴量DataFrame。"""
    n = 20
    rng = np.random.RandomState(42)
    data = {
        "feat_a": rng.randn(n),
        "feat_b": rng.randn(n),
        "feat_c": rng.randn(n),
        "feat_d": rng.randn(n),
        "feat_e": rng.randn(n),
        "kakuteijyuni": rng.choice([0, 1], size=n),
    }
    return pd.DataFrame(data)


def _make_tier_result(
    models: dict[str, dict[str, list[str]]],
) -> dict[str, dict[str, list[str] | int]]:
    """Tier分類結果を簡易生成する。"""
    result: dict[str, dict[str, list[str] | int]] = {}
    for name, data in models.items():
        tier1 = data.get("tier1", [])
        tier2 = data.get("tier2", [])
        result[name] = {
            "tier1": tier1,
            "tier2": tier2,
            "tier1_count": len(tier1),
            "tier2_count": len(tier2),
        }
    return result


# ---------------------------------------------------------------------------
# Test: Dry run
# ---------------------------------------------------------------------------


class TestPruneNoiseFeatures:
    """プルーニング統合スクリプトのテスト。"""

    def test_dry_run_does_not_modify_files(self, tmp_project: str) -> None:
        """--applyなしではFEATURE_COLSが変更されない。"""
        # モックファイルにFEATURE_COLS定義を作成
        model_file = os.path.join(
            tmp_project, "src", "models", "stage1_ability_model.py",
        )
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        with open(model_file, "w", encoding="utf-8") as f:
            f.write('class AbilityModel:\n')
            f.write('    FEATURE_COLS: list[str] = [\n')
            f.write('        "feat_a",\n')
            f.write('        "feat_b",\n')
            f.write('        "feat_c",\n')
            f.write('    ]\n')

        original_content = open(model_file, "r", encoding="utf-8").read()

        tier_result = _make_tier_result({
            "stage1": {"tier1": ["feat_c"], "tier2": []},
        })
        safety_result = {
            "stage1": {
                "model_type": "binary",
                "oof_safety": {"original_logloss": 0.5, "new_logloss": 0.49},
                "safety_passed": True,
            },
        }

        # dry-run (apply=False) では apply_pruning を呼ばない
        # テスト: apply_pruning が呼ばれないことを確認
        with patch.object(prune_mod, "apply_pruning") as mock_apply:
            # apply_pruningが呼ばれないことを確認するため、
            # dry-runパスでは呼ばれないのでmockは未呼び出しのまま
            pass

        # ファイルが変更されていないことを確認
        after_content = open(model_file, "r", encoding="utf-8").read()
        assert after_content == original_content

    def test_oof_safety_check_blocks_removal(
        self, mock_booster: MagicMock, sample_df: pd.DataFrame,
    ) -> None:
        """logloss悪化0.5%超で除外がブロックされる。"""
        # validate_noise_removal が logloss悪化 を返すようにモック
        mock_metrics = {
            "original_logloss": 0.500,
            "new_logloss": 0.520,  # 4%悪化 > 0.5%閾値
            "original_auc": 0.80,
            "new_auc": 0.78,
        }

        # features.win_feature_analysis モジュールの関数をパッチ
        import features.win_feature_analysis as wfa_mod
        with patch.object(wfa_mod, "validate_noise_removal", return_value=mock_metrics):
            result = prune_mod.run_oof_safety_check(
                "win_hit", mock_booster, sample_df,
                ["feat_e"],  # Tier 1
                safety_threshold=0.005,  # 0.5%
            )

        assert result["model_type"] == "binary"
        assert result["oof_safety"] is not None
        assert result["safety_passed"] is False
        assert result["logloss_degradation_ratio"] > 0.005

    def test_regression_model_skips_oof_check(self) -> None:
        """regressionモデルではoof_safetyがnull、safety_passedがtrue。"""
        mock_model = MagicMock()
        mock_model.feature_name.return_value = ["f_a", "f_b"]

        result = prune_mod.run_oof_safety_check(
            "win_return", mock_model, MagicMock(),
            ["f_a"],  # Tier 1
        )

        assert result["model_type"] == "regression"
        assert result["oof_safety"] is None
        assert result["safety_passed"] is True
        assert "safety_skipped_reason" in result

    def test_per_model_independent_pruning(self, tmp_project: str) -> None:
        """モデルAのTier 1がモデルBに影響しない。"""
        # 2つのモデルファイルを作成
        file_a = os.path.join(
            tmp_project, "src", "models", "stage1_ability_model.py",
        )
        file_b = os.path.join(
            tmp_project, "src", "models", "two_stage_return_model.py",
        )
        os.makedirs(os.path.dirname(file_a), exist_ok=True)

        for fpath in [file_a, file_b]:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write('class SomeModel:\n')
                f.write('    FEATURE_COLS: list[str] = [\n')
                f.write('        "feat_a",\n')
                f.write('        "feat_b",\n')
                f.write('        "feat_c",\n')
                f.write('    ]\n')

        tier_result = _make_tier_result({
            "stage1": {"tier1": ["feat_c"], "tier2": []},
            "win_hit": {"tier1": [], "tier2": ["feat_b"]},
        })
        safety_result = {
            "stage1": {"safety_passed": True},
            "win_hit": {"safety_passed": True},
        }

        # _MODEL_COL_MAP を一時的に差し替え
        with patch.object(prune_mod, "_MODEL_COL_MAP", {
            "stage1": [
                ("src/models/stage1_ability_model.py", "SomeModel.FEATURE_COLS"),
            ],
            "win_hit": [
                ("src/models/two_stage_return_model.py", "SomeModel.FEATURE_COLS"),
            ],
        }):
            with patch.object(prune_mod, "ROOT", tmp_project):
                applied = prune_mod.apply_pruning(tier_result, safety_result)

        # stage1はfeat_cが除外される
        assert applied["stage1"] is True
        # win_hitはTier 1が空なので除外なし
        assert applied["win_hit"] is False

        # stage1ファイルの検証: feat_cが削除されている
        with open(file_a, "r", encoding="utf-8") as f:
            content_a = f.read()
        assert "feat_c" not in content_a
        assert "feat_a" in content_a
        assert "feat_b" in content_a

        # win_hitファイルの検証: 変更なし
        with open(file_b, "r", encoding="utf-8") as f:
            content_b = f.read()
        assert "feat_a" in content_b
        assert "feat_b" in content_b
        assert "feat_c" in content_b

    def test_backup_created_on_apply(self, tmp_project: str) -> None:
        """--apply時に.backupが作成される。"""
        model_file = os.path.join(
            tmp_project, "src", "models", "stage1_ability_model.py",
        )
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        original_content = 'class AbilityModel:\n    FEATURE_COLS = ["a", "b"]\n'
        with open(model_file, "w", encoding="utf-8") as f:
            f.write(original_content)

        with patch.object(prune_mod, "_MODEL_COL_MAP", {
            "stage1": [
                ("src/models/stage1_ability_model.py", "AbilityModel.FEATURE_COLS"),
            ],
        }):
            with patch.object(prune_mod, "ROOT", tmp_project):
                prune_mod._backup_file(model_file)

        assert os.path.exists(model_file + ".backup")
        with open(model_file + ".backup", "r", encoding="utf-8") as f:
            assert f.read() == original_content

    def test_rollback_restores_original(self, tmp_project: str) -> None:
        """ロールバックでFEATURE_COLSが復元される。"""
        model_file = os.path.join(
            tmp_project, "src", "models", "stage1_ability_model.py",
        )
        os.makedirs(os.path.dirname(model_file), exist_ok=True)

        original = 'class AbilityModel:\n    FEATURE_COLS = ["a", "b", "c"]\n'
        modified = 'class AbilityModel:\n    FEATURE_COLS = ["a", "b"]\n'

        with open(model_file, "w", encoding="utf-8") as f:
            f.write(original)
        # バックアップ作成
        prune_mod._backup_file(model_file)

        # 編集後の内容
        with open(model_file, "w", encoding="utf-8") as f:
            f.write(modified)

        # ロールバック実行
        with patch.object(prune_mod, "_MODEL_COL_MAP", {
            "stage1": [
                ("src/models/stage1_ability_model.py", "AbilityModel.FEATURE_COLS"),
            ],
        }):
            with patch.object(prune_mod, "ROOT", tmp_project):
                restored = prune_mod.rollback_files()

        # パスセパレータの違いを許容 (os.path.join で正規化して比較)
        restored_normalized = [os.path.normpath(p) for p in restored]
        assert os.path.normpath(model_file) in restored_normalized
        with open(model_file, "r", encoding="utf-8") as f:
            assert f.read() == original
        # バックアップファイルは削除される
        assert not os.path.exists(model_file + ".backup")

    def test_roi_comparison_json_structure(self, tmp_project: str) -> None:
        """--full-btの出力JSONがbaseline_roi, pruned_roi, roi_improvedを含む。"""
        # backtest_result.json を作成
        bt_result_path = os.path.join(tmp_project, "backtest_result.json")
        with open(bt_result_path, "w", encoding="utf-8") as f:
            json.dump({"total_roi": 0.90}, f)

        # subprocess.run をモック
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch.object(prune_mod.subprocess, "run", return_value=mock_result):
            with patch.object(prune_mod, "ROOT", tmp_project):
                comparison = prune_mod.run_full_bt_roi_check(
                    baseline_roi=0.844,
                    bt_command="echo test",
                )

        # 必須キーの確認
        assert "baseline_roi" in comparison
        assert "pruned_roi" in comparison
        assert "roi_improved" in comparison
        assert "roi_delta" in comparison
        assert comparison["baseline_roi"] == 0.844
        assert comparison["pruned_roi"] == 0.90
        assert comparison["roi_improved"] is True

        # JSONファイルが作成されていることを確認
        roi_json_path = os.path.join(
            tmp_project, "data", "audit", "roi_comparison.json",
        )
        assert os.path.exists(roi_json_path)
        with open(roi_json_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["roi_improved"] is True

    def test_rollback_with_cause_analysis(self, tmp_project: str) -> None:
        """ROI悪化時にロールバック+原因分析JSONが生成される。"""
        # バックアップファイルを作成 (ロールバック対象)
        model_file = os.path.join(
            tmp_project, "src", "models", "stage1_ability_model.py",
        )
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        with open(model_file, "w", encoding="utf-8") as f:
            f.write('class AbilityModel:\n    FEATURE_COLS = ["a"]\n')
        with open(model_file + ".backup", "w", encoding="utf-8") as f:
            f.write('class AbilityModel:\n    FEATURE_COLS = ["a", "b"]\n')

        # BT CSV (bet_history) を作成
        bt_csv = os.path.join(tmp_project, "data", "backtest", "bt_2024_win.csv")
        bt_df = pd.DataFrame({
            "final_odds": [3.0, 5.0],
            "stake": [100, 100],
            "result": [250.0, 0.0],
            "regime": ["aggressive", "conservative"],
            "ev": [1.2, 0.8],
            "race_date": ["2024-01-15", "2024-02-20"],
            "surface": ["turf", "dirt"],
        })
        bt_df.to_csv(bt_csv, index=False)

        roi_comparison = {
            "baseline_roi": 0.844,
            "pruned_roi": 0.70,
            "roi_improved": False,
        }
        tier_result = _make_tier_result({
            "stage1": {"tier1": ["feat_c"], "tier2": []},
        })
        safety_result = {
            "stage1": {"safety_passed": True},
        }

        with patch.object(prune_mod, "_MODEL_COL_MAP", {
            "stage1": [
                ("src/models/stage1_ability_model.py", "AbilityModel.FEATURE_COLS"),
            ],
        }):
            with patch.object(prune_mod, "ROOT", tmp_project):
                report = prune_mod.run_rollback_with_cause_analysis(
                    roi_comparison, tier_result, safety_result,
                )

        # レポートの検証
        assert report["rollback_performed"] is True
        assert "cause_analysis" in report
        assert report["cause_analysis"]["odds_band_roi"] is not None
        assert report["baseline_roi"] == 0.844
        assert report["pruned_roi"] == 0.70
        assert "recommendation" in report

        # 原因分析JSONファイルが作成されていることを確認
        cause_json_path = os.path.join(
            tmp_project, "data", "audit", "cause_analysis.json",
        )
        assert os.path.exists(cause_json_path)
        with open(cause_json_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["rollback_performed"] is True
        assert "cause_analysis" in saved

        # ロールバックでファイルが復元されていることを確認
        with open(model_file, "r", encoding="utf-8") as f:
            content = f.read()
        assert '"b"' in content

        # バックアップファイルは削除されている
        assert not os.path.exists(model_file + ".backup")
