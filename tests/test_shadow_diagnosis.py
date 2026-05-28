"""ShadowDiagnosis ユニットテスト (DIAG-01~03)

Phase 41 成果物 (shadow_comparison_result.json, shadow_race_diff.parquet,
shadow_horse_diff.parquet, shadow_manifest.json) から3ステップ段階的除外診断を検証する。
save_diagnosis_results, ShadowDiagnosisReportGenerator も検証。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.shadow_diagnosis import (
    CalibrationResult,
    ProbabilityQualityResult,
    SegmentCalibration,
    SelectionGroupMetrics,
    SelectionPatternResult,
    ShadowDiagnosis,
    ShadowDiagnosisReportGenerator,
    ShadowDiagnosisResult,
    save_diagnosis_results,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_BASELINE = "baseline"
_DEFAULT_SHADOW = "shadow_mawc_ranker"


def _make_horse_df(
    *,
    shadow_name: str = _DEFAULT_SHADOW,
    n_races: int = 4,
    horses_per_race: int = 6,
    include_popularity: bool = True,
    include_surface: bool = True,
    include_tanodds: bool = True,
) -> pd.DataFrame:
    """合成 horse_diff DataFrame を生成."""
    rows: list[dict] = []
    np.random.seed(42)

    for r in range(n_races):
        race_id = f"2024{r + 1:02d}010101"
        for h in range(horses_per_race):
            umaban = h + 1
            row: dict = {
                "race_id": race_id,
                "umaban": umaban,
                "kakuteijyuni": h + 1,
                f"{_DEFAULT_BASELINE}_p_win_final": round(
                    np.random.uniform(0.05, 0.5), 4
                ),
                f"{shadow_name}_p_win_final": round(
                    np.random.uniform(0.05, 0.5), 4
                ),
                f"{_DEFAULT_BASELINE}_investment_score": round(
                    np.random.uniform(0.1, 0.9), 4
                ),
                f"{_DEFAULT_BASELINE}_stake": 100.0 if h == 0 else 0.0,
                f"{_DEFAULT_BASELINE}_win_market_selection_score": round(
                    np.random.uniform(0.2, 0.8), 4
                ),
                f"{_DEFAULT_BASELINE}_selected": h == 0,
                f"{shadow_name}_investment_score": round(
                    np.random.uniform(0.1, 0.9), 4
                ),
                f"{shadow_name}_stake": 100.0 if h == 1 else 0.0,
                f"{shadow_name}_win_market_selection_score": round(
                    np.random.uniform(0.2, 0.8), 4
                ),
                f"{shadow_name}_selected": h == 1,
            }
            if include_popularity:
                row["popularity"] = h + 1
            if include_surface:
                row["surface"] = "turf" if r % 2 == 0 else "dirt"
            if include_tanodds:
                row["tanodds"] = round(2.0 + h * 3.0, 1)
                row["closing_win_odds"] = round(2.1 + h * 3.0, 1)
            rows.append(row)

    return pd.DataFrame(rows)


def _make_race_df(
    *,
    n_races: int = 4,
) -> pd.DataFrame:
    """合成 race_diff DataFrame を生成."""
    rows: list[dict] = []
    for r in range(n_races):
        race_id = f"2024{r + 1:02d}010101"
        changed = r % 2 == 0  # 交互に changed/unchanged
        rows.append({
            "race_id": race_id,
            "baseline_selected_umaban": 1,
            "shadow_selected_umaban": 2 if changed else 1,
            "selected_changed": changed,
            "baseline_result": 150.0 if r == 0 else 0.0,
            "shadow_result": 0.0 if r == 0 else 0.0,
            "baseline_stake": 100.0,
            "shadow_stake": 100.0,
            "baseline_tanodds": 3.5,
            "shadow_tanodds": 5.2,
            "baseline_p_win_final": 0.25,
            "shadow_p_win_final": 0.20,
            "baseline_closing_win_odds": 3.6,
            "shadow_closing_win_odds": 5.3,
        })
    return pd.DataFrame(rows)


def _make_manifest(
    *,
    baseline_name: str = _DEFAULT_BASELINE,
    shadow_name: str = _DEFAULT_SHADOW,
) -> dict:
    """合成 manifest JSON."""
    return {
        "generated_at": "2026-01-01T00:00:00Z",
        "framework_version": "1.0",
        "variants": [
            {
                "variant_name": baseline_name,
                "model_dir": f"data/models/{baseline_name}",
                "flag_states": {
                    "enable_market_aware_calibrator": False,
                    "enable_race_level_ranker": False,
                },
            },
            {
                "variant_name": shadow_name,
                "model_dir": f"data/models/{shadow_name}",
                "flag_states": {
                    "enable_market_aware_calibrator": True,
                    "enable_race_level_ranker": True,
                },
            },
        ],
        "folds": [
            {
                "year": 2024,
                "train_start": "2020-01-01",
                "train_end": "2023-12-31",
                "test_start": "2024-01-01",
                "test_end": "2024-12-31",
            },
        ],
        "artifacts": {},
    }


def _setup_input_dir(
    tmp_path: Path,
    *,
    horse_df: pd.DataFrame | None = None,
    race_df: pd.DataFrame | None = None,
    manifest: dict | None = None,
    shadow_name: str = _DEFAULT_SHADOW,
) -> Path:
    """input_dir に JSON/Parquet ファイルを書き出すヘルパー."""
    input_dir = tmp_path / "shadow_input"
    input_dir.mkdir(parents=True, exist_ok=True)

    if horse_df is None:
        horse_df = _make_horse_df(shadow_name=shadow_name)
    if race_df is None:
        race_df = _make_race_df()
    if manifest is None:
        manifest = _make_manifest(shadow_name=shadow_name)

    # shadow_comparison_result.json (minimal)
    result_json: dict = {
        "generated_at": "2026-01-01T00:00:00Z",
        "folds": {},
        "overall": {"metrics": {}},
    }
    (input_dir / "shadow_comparison_result.json").write_text(
        json.dumps(result_json, indent=2), encoding="utf-8"
    )

    horse_df.to_parquet(input_dir / "shadow_horse_diff.parquet", index=False)
    race_df.to_parquet(input_dir / "shadow_race_diff.parquet", index=False)
    (input_dir / "shadow_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    return input_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestShadowDiagnosis:
    """ShadowDiagnosis クラスのテスト."""

    # Test 1: 疎通テスト
    def test_run_returns_result(self, tmp_path: Path) -> None:
        """ShadowDiagnosis.run() が例外なく ShadowDiagnosisResult を返す."""
        input_dir = _setup_input_dir(tmp_path)
        diag = ShadowDiagnosis(input_dir)
        result = diag.run()

        assert isinstance(result, ShadowDiagnosisResult)
        assert isinstance(result.step1, ProbabilityQualityResult)
        assert isinstance(result.step2, SelectionPatternResult)
        assert isinstance(result.step3, CalibrationResult)
        assert isinstance(result.missing_inputs, list)
        assert isinstance(result.variant_names, list)
        assert len(result.variant_names) == 2
        assert result.generated_at != ""

    # Test 2: 確率品質ステップ
    def test_step1_probability_quality(self, tmp_path: Path) -> None:
        """step1 で baseline vs shadow の Brier/logloss/ECE/actual_predicted_ratio を検証."""
        input_dir = _setup_input_dir(tmp_path)
        diag = ShadowDiagnosis(input_dir)
        result = diag.run()
        step1 = result.step1

        # baseline と shadow のメトリクスが計算されている
        assert step1.baseline_brier > 0
        assert step1.shadow_brier > 0
        assert step1.baseline_logloss > 0
        assert step1.shadow_logloss > 0
        assert step1.baseline_ece >= 0
        assert step1.shadow_ece >= 0
        assert step1.baseline_n_horses > 0
        assert step1.shadow_n_horses == step1.baseline_n_horses

        # actual_predicted_ratio が計算されている
        assert step1.baseline_apr > 0 or step1.baseline_apr == 0.0
        assert step1.shadow_apr > 0 or step1.shadow_apr == 0.0

        # Delta 値
        assert isinstance(step1.delta_brier, float)
        assert isinstance(step1.delta_logloss, float)
        assert isinstance(step1.delta_ece, float)
        assert isinstance(step1.delta_apr, float)

        # Brier の直接的数値検証:
        # Brier = mean((p - y)^2) で y は kakuteijyuni==1
        # seed=42 のランダム確率に対して Brier は 0 < Brier < 1
        assert 0 < step1.baseline_brier < 1

    # Test 3: 選定パターンステップ
    def test_step2_selection_pattern(self, tmp_path: Path) -> None:
        """step2 で changed/unchanged グループの ROI/HR/avg_odds を検証."""
        input_dir = _setup_input_dir(tmp_path)
        diag = ShadowDiagnosis(input_dir)
        result = diag.run()
        step2 = result.step2

        # changed と unchanged 両方のグループが存在
        assert isinstance(step2.changed, SelectionGroupMetrics)
        assert isinstance(step2.unchanged, SelectionGroupMetrics)

        # changed グループのメトリクスが計算されている
        assert step2.n_changed_races > 0
        assert step2.n_unchanged_races > 0
        assert step2.changed.bet_count >= 0
        assert step2.unchanged.bet_count >= 0

        # Delta 値
        assert isinstance(step2.delta_roi, float)
        assert isinstance(step2.delta_hit_rate, float)

    # Test 4: キャリブレーションステップ
    def test_step3_calibration(self, tmp_path: Path) -> None:
        """step3 でセグメント別 actual/predicted 比率と ECE を検証."""
        input_dir = _setup_input_dir(tmp_path)
        diag = ShadowDiagnosis(input_dir)
        result = diag.run()
        step3 = result.step3

        assert isinstance(step3, CalibrationResult)
        assert len(step3.segments) > 0

        # セグメント名の確認
        segment_names = {s.segment_name for s in step3.segments}
        # probability_rank_band は常に計算可能
        assert "probability_rank_band" in segment_names
        # selected_changed は常に計算可能
        assert "selected_changed" in segment_names

        # 各セグメントの値を検証
        for seg in step3.segments:
            assert isinstance(seg, SegmentCalibration)
            assert seg.n_samples > 0
            assert seg.segment_name != ""
            assert seg.segment_value != ""

    # Test 5: missing_inputs 検出
    def test_missing_inputs_detection(self, tmp_path: Path) -> None:
        """horse_diff に popularity/surface/tanodds 列が無い場合、missing_inputs に含まれる."""
        horse_df = _make_horse_df(
            include_popularity=False,
            include_surface=False,
            include_tanodds=False,
        )
        input_dir = _setup_input_dir(tmp_path, horse_df=horse_df)
        diag = ShadowDiagnosis(input_dir)
        result = diag.run()

        assert "popularity" in result.missing_inputs
        assert "surface" in result.missing_inputs

    # Test 6: セグメントフォールバック (空セグメントスキップ)
    def test_segment_fallback_empty_group(self, tmp_path: Path) -> None:
        """小規模データで空セグメントがスキップされる (unknown にフォールバックしない)."""
        # 出走頭数2のレース1つだけ → probability_rank_band "7+" は空
        horse_df = _make_horse_df(n_races=1, horses_per_race=2)
        race_df = _make_race_df(n_races=1)
        input_dir = _setup_input_dir(tmp_path, horse_df=horse_df, race_df=race_df)
        diag = ShadowDiagnosis(input_dir)
        result = diag.run()
        step3 = result.step3

        prob_segments = [
            s for s in step3.segments if s.segment_name == "probability_rank_band"
        ]
        segment_values = {s.segment_value for s in prob_segments}
        # "7+" は存在しない (空なのでスキップ)
        assert "7+" not in segment_values
        # "unknown" にもフォールバックしない
        assert "unknown" not in segment_values

    # Test 7: variant 名の動的取得
    def test_variant_name_from_manifest(self, tmp_path: Path) -> None:
        """manifest から variant 名を取得し、列名プレフィックスが正しく解決される."""
        custom_shadow = "custom_shadow_v2"
        manifest = _make_manifest(shadow_name=custom_shadow)
        horse_df = _make_horse_df(shadow_name=custom_shadow)
        input_dir = _setup_input_dir(
            tmp_path,
            horse_df=horse_df,
            manifest=manifest,
            shadow_name=custom_shadow,
        )
        diag = ShadowDiagnosis(input_dir)
        result = diag.run()

        assert _DEFAULT_BASELINE in result.variant_names
        assert custom_shadow in result.variant_names

        # step1 が計算できている = 列名プレフィックス解決が正しい
        assert result.step1.shadow_brier > 0


# ---------------------------------------------------------------------------
# Plan 02 Tests: save_diagnosis_results + CLI + Report
# ---------------------------------------------------------------------------


def _make_synthetic_result() -> ShadowDiagnosisResult:
    """テスト用の合成 ShadowDiagnosisResult を生成."""
    return ShadowDiagnosisResult(
        step1=ProbabilityQualityResult(
            baseline_brier=0.15,
            shadow_brier=0.16,
            baseline_logloss=0.45,
            shadow_logloss=0.47,
            baseline_ece=0.03,
            shadow_ece=0.04,
            baseline_apr=1.05,
            shadow_apr=0.95,
            baseline_n_horses=100,
            shadow_n_horses=100,
            delta_brier=0.01,
            delta_logloss=0.02,
            delta_ece=0.01,
            delta_apr=-0.10,
        ),
        step2=SelectionPatternResult(
            changed=SelectionGroupMetrics(
                roi=-0.05, hit_rate=0.15, avg_odds=8.5,
                actual_predicted_ratio=0.90, bet_count=20, n_races=20,
            ),
            unchanged=SelectionGroupMetrics(
                roi=0.10, hit_rate=0.25, avg_odds=5.2,
                actual_predicted_ratio=1.10, bet_count=20, n_races=20,
            ),
            n_changed_races=20,
            n_unchanged_races=20,
            delta_roi=-0.15,
            delta_hit_rate=-0.10,
        ),
        step3=CalibrationResult(
            segments=[
                SegmentCalibration(
                    segment_name="popularity_band",
                    segment_value="1-3",
                    n_samples=30,
                    actual_predicted_ratio_baseline=1.05,
                    actual_predicted_ratio_shadow=0.95,
                    ece_baseline=0.03,
                    ece_shadow=0.05,
                    delta_apr=-0.10,
                    delta_ece=0.02,
                ),
                SegmentCalibration(
                    segment_name="probability_rank_band",
                    segment_value="top1",
                    n_samples=25,
                    actual_predicted_ratio_baseline=1.00,
                    actual_predicted_ratio_shadow=1.02,
                    ece_baseline=0.02,
                    ece_shadow=0.03,
                    delta_apr=0.02,
                    delta_ece=0.01,
                ),
            ],
        ),
        missing_inputs=["popularity", "surface"],
        variant_names=["baseline", "shadow_mawc_ranker"],
        generated_at="2026-05-28T22:00:00+00:00",
    )


class TestSaveDiagnosisResults:
    """save_diagnosis_results 関数のテスト."""

    def test_save_diagnosis_results_json(self, tmp_path: Path) -> None:
        """save_diagnosis_results が JSON ファイルを生成し必要キーが含まれる."""
        result = _make_synthetic_result()
        output_dir = tmp_path / "output"
        paths = save_diagnosis_results(result, output_dir)

        assert "result_json" in paths
        assert "summary_md" in paths

        # JSON 検証
        json_path = paths["result_json"]
        assert json_path.exists()
        data = json.loads(json_path.read_text(encoding="utf-8"))

        # トップレベルキー
        assert "step1_probability_quality" in data
        assert "step2_selection_pattern" in data
        assert "step3_calibration" in data
        assert "missing_inputs" in data
        assert "recommendations" in data

        # step1 構造
        s1 = data["step1_probability_quality"]
        assert "baseline" in s1
        assert "shadow" in s1
        assert "delta" in s1
        assert s1["baseline"]["brier"] == 0.15
        assert s1["delta"]["brier"] == 0.01

        # step2 構造
        s2 = data["step2_selection_pattern"]
        assert "changed" in s2
        assert "unchanged" in s2
        assert "delta" in s2
        assert s2["delta"]["roi"] == -0.15

        # step3 構造
        s3 = data["step3_calibration"]
        assert "segments" in s3
        assert len(s3["segments"]) == 2
        assert s3["segments"][0]["segment_name"] == "popularity_band"

        # missing_inputs
        assert data["missing_inputs"] == ["popularity", "surface"]

    def test_save_diagnosis_summary_md(self, tmp_path: Path) -> None:
        """summary.md が生成され、主要セクションが含まれる."""
        result = _make_synthetic_result()
        output_dir = tmp_path / "output"
        paths = save_diagnosis_results(result, output_dir)

        md_path = paths["summary_md"]
        assert md_path.exists()
        content = md_path.read_text(encoding="utf-8")

        assert "Probability Quality" in content
        assert "Selection Pattern" in content
        assert "Calibration Gaps" in content
        assert "Missing Inputs" in content
        assert "popularity" in content
        assert "surface" in content
        assert "0.1500" in content  # baseline_brier


class TestCLIDryRun:
    """CLI スクリプトのテスト."""

    def test_cli_dry_run(self) -> None:
        """--help が exit code 0 で完了する."""
        result = subprocess.run(
            [sys.executable, "scripts/run_shadow_diagnosis.py", "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 0
        assert "Shadow Diagnosis" in result.stdout


class TestReportGenerator:
    """ShadowDiagnosisReportGenerator のテスト."""

    def test_report_generator_html(self, tmp_path: Path) -> None:
        """ShadowDiagnosisReportGenerator が HTML を生成し主要セクションが含まれる."""
        result = _make_synthetic_result()
        output_dir = tmp_path / "report_output"
        gen = ShadowDiagnosisReportGenerator(output_dir)
        report_path = gen.generate(result)

        assert report_path.exists()
        html = report_path.read_text(encoding="utf-8")

        assert "Step 1: Probability Quality" in html
        assert "Step 2: Selection Pattern" in html
        assert "Step 3: Calibration" in html
        assert "0.1500" in html  # baseline_brier

    def test_report_missing_inputs_section(self, tmp_path: Path) -> None:
        """missing_inputs が HTML に正しく表示される."""
        result = _make_synthetic_result()
        output_dir = tmp_path / "report_output2"
        gen = ShadowDiagnosisReportGenerator(output_dir)
        report_path = gen.generate(result)

        html = report_path.read_text(encoding="utf-8")
        assert "popularity" in html
        assert "surface" in html
