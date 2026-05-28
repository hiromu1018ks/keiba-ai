"""ShadowDiagnosis — 3ステップ段階的除外診断 (DIAG-01~03)

Phase 41 成果物 (shadow_comparison_result.json, shadow_race_diff.parquet,
shadow_horse_diff.parquet, shadow_manifest.json) を読み込み、baseline vs shadow の
確率品質・選定パターン・キャリブレーション乖離を全面比較する診断エンジン。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from backtest.shadow_comparison import ShadowComparisonFramework

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# セグメント定数 (D-03)
# ---------------------------------------------------------------------------

POPULARITY_BAND_EDGES: list[float] = [0, 3, 6, 9, 14, float("inf")]
POPULARITY_BAND_NAMES: list[str] = ["1-3", "4-6", "7-9", "10-14", "15+"]

PROB_RANK_BAND_EDGES: list[float] = [0, 1, 3, 6, float("inf")]
PROB_RANK_BAND_NAMES: list[str] = ["top1", "2-3", "4-6", "7+"]

ODDS_BAND_EDGES: list[float] = [0, 3, 5, 10, 30, float("inf")]
ODDS_BAND_NAMES: list[str] = ["1-3", "3-5", "5-10", "10-30", "30+"]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbabilityQualityResult:
    """DIAG-01: 確率品質比較結果."""

    baseline_brier: float = 0.0
    shadow_brier: float = 0.0
    baseline_logloss: float = 0.0
    shadow_logloss: float = 0.0
    baseline_ece: float = 0.0
    shadow_ece: float = 0.0
    baseline_apr: float = 0.0  # actual_predicted_ratio
    shadow_apr: float = 0.0
    baseline_n_horses: int = 0
    shadow_n_horses: int = 0
    delta_brier: float = 0.0
    delta_logloss: float = 0.0
    delta_ece: float = 0.0
    delta_apr: float = 0.0


@dataclass(frozen=True)
class SelectionGroupMetrics:
    """選定グループ (changed/unchanged) のメトリクス."""

    roi: float = 0.0
    hit_rate: float = 0.0
    avg_odds: float = 0.0
    actual_predicted_ratio: float = 0.0
    bet_count: int = 0
    n_races: int = 0


@dataclass(frozen=True)
class SelectionPatternResult:
    """DIAG-02: 選定パターン差分結果."""

    changed: SelectionGroupMetrics = field(default_factory=SelectionGroupMetrics)
    unchanged: SelectionGroupMetrics = field(default_factory=SelectionGroupMetrics)
    n_changed_races: int = 0
    n_unchanged_races: int = 0
    delta_roi: float = 0.0
    delta_hit_rate: float = 0.0


@dataclass(frozen=True)
class SegmentCalibration:
    """単一セグメントのキャリブレーション結果."""

    segment_name: str = ""
    segment_value: str = ""
    n_samples: int = 0
    actual_predicted_ratio_baseline: float = 0.0
    actual_predicted_ratio_shadow: float = 0.0
    ece_baseline: float = 0.0
    ece_shadow: float = 0.0
    delta_apr: float = 0.0
    delta_ece: float = 0.0


@dataclass
class CalibrationResult:
    """DIAG-03: セグメント別キャリブレーション乖離結果."""

    segments: list[SegmentCalibration] = field(default_factory=list)


@dataclass
class ShadowDiagnosisResult:
    """ShadowDiagnosis の全体結果."""

    step1: ProbabilityQualityResult = field(default_factory=ProbabilityQualityResult)
    step2: SelectionPatternResult = field(default_factory=SelectionPatternResult)
    step3: CalibrationResult = field(default_factory=CalibrationResult)
    missing_inputs: list[str] = field(default_factory=list)
    variant_names: list[str] = field(default_factory=list)
    generated_at: str = ""


# ---------------------------------------------------------------------------
# ShadowDiagnosis
# ---------------------------------------------------------------------------


class ShadowDiagnosis:
    """Phase 41 成果物から3ステップ段階的除外診断を実行 (DIAG-01~03).

    Args:
        input_dir: Phase 41 成果物が格納されたディレクトリ.
    """

    def __init__(self, input_dir: Path) -> None:
        self.input_dir = input_dir
        self.missing_inputs: list[str] = []

        # Phase 41 成果物の読み込み
        self.result_json: dict[str, Any] = json.loads(
            (input_dir / "shadow_comparison_result.json").read_text(encoding="utf-8")
        )
        self.race_diff: pd.DataFrame = pd.read_parquet(
            input_dir / "shadow_race_diff.parquet"
        )
        self.horse_diff: pd.DataFrame = pd.read_parquet(
            input_dir / "shadow_horse_diff.parquet"
        )
        self.manifest: dict[str, Any] = json.loads(
            (input_dir / "shadow_manifest.json").read_text(encoding="utf-8")
        )

        # variant 名を動的に取得 (Pitfall 5)
        self.variant_names: list[str] = self._resolve_variant_names()
        self.baseline_name: str = self.variant_names[0] if self.variant_names else "baseline"
        self.shadow_name: str = (
            self.variant_names[1] if len(self.variant_names) > 1 else "shadow"
        )

        # 不足列の検出
        self._detect_missing_inputs()

    def _resolve_variant_names(self) -> list[str]:
        """manifest から variant 名を動的に取得."""
        variants = self.manifest.get("variants", [])
        if variants:
            return [v["variant_name"] for v in variants]
        # フォールバック: folds から推測は不可なので空リスト
        return []

    def _detect_missing_inputs(self) -> None:
        """horse_diff/race_diff の列をチェックし、不足列を検出."""
        horse_cols = set(self.horse_diff.columns)
        race_cols = set(self.race_diff.columns)

        # DIAG-03 で必要な列
        for col in ["popularity", "surface", "tanodds", "closing_win_odds"]:
            if col not in horse_cols and col not in race_cols:
                if col not in self.missing_inputs:
                    self.missing_inputs.append(col)

    def run(self) -> ShadowDiagnosisResult:
        """3ステップ段階的除外診断を実行して ShadowDiagnosisResult を返す."""
        step1 = self._step1_probability_quality()
        step2 = self._step2_selection_pattern()
        step3 = self._step3_calibration_by_segment()

        return ShadowDiagnosisResult(
            step1=step1,
            step2=step2,
            step3=step3,
            missing_inputs=list(self.missing_inputs),
            variant_names=list(self.variant_names),
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    # ------------------------------------------------------------------
    # Step 1: 確率品質次元 (DIAG-01)
    # ------------------------------------------------------------------

    def _compute_prob_metrics(
        self,
        p_col: str,
    ) -> tuple[float, float, float, float, int]:
        """p_col と kakuteijyuni から Brier/logloss/ECE/APR を計算."""
        if self.horse_diff.empty or p_col not in self.horse_diff.columns:
            return 0.0, 0.0, 0.0, 0.0, 0

        p_vals = pd.to_numeric(self.horse_diff[p_col], errors="coerce")

        if "kakuteijyuni" not in self.horse_diff.columns:
            return 0.0, 0.0, 0.0, 0.0, 0

        is_win = (self.horse_diff["kakuteijyuni"] == 1).astype(float)

        p_vals = p_vals.reset_index(drop=True)
        is_win = is_win.reset_index(drop=True)
        valid_mask = p_vals.notna() & (p_vals > 0) & (p_vals < 1)

        n_valid = int(valid_mask.sum())
        if n_valid == 0:
            return 0.0, 0.0, 0.0, 0.0, 0

        p_valid = p_vals[valid_mask].values
        y_valid = is_win[valid_mask].values

        # Brier
        brier = float(np.mean((p_valid - y_valid) ** 2))

        # Logloss
        eps = 1e-15
        p_clipped = np.clip(p_valid, eps, 1 - eps)
        logloss = float(
            -np.mean(y_valid * np.log(p_clipped) + (1 - y_valid) * np.log(1 - p_clipped))
        )

        # ECE
        ece = ShadowComparisonFramework._compute_ece(p_valid, y_valid, n_bins=10)

        # Actual/predicted ratio
        mean_actual = float(np.mean(y_valid))
        mean_pred = float(np.mean(p_valid))
        apr = mean_actual / mean_pred if mean_pred > 0 else 0.0

        return brier, logloss, ece, apr, n_valid

    def _step1_probability_quality(self) -> ProbabilityQualityResult:
        """全馬ベースで baseline vs shadow の確率品質を比較 (DIAG-01)."""
        bl_p_col = f"{self.baseline_name}_p_win_final"
        sh_p_col = f"{self.shadow_name}_p_win_final"

        bl_brier, bl_logloss, bl_ece, bl_apr, bl_n = self._compute_prob_metrics(bl_p_col)
        sh_brier, sh_logloss, sh_ece, sh_apr, sh_n = self._compute_prob_metrics(sh_p_col)

        return ProbabilityQualityResult(
            baseline_brier=bl_brier,
            shadow_brier=sh_brier,
            baseline_logloss=bl_logloss,
            shadow_logloss=sh_logloss,
            baseline_ece=bl_ece,
            shadow_ece=sh_ece,
            baseline_apr=bl_apr,
            shadow_apr=sh_apr,
            baseline_n_horses=bl_n,
            shadow_n_horses=sh_n,
            delta_brier=sh_brier - bl_brier,
            delta_logloss=sh_logloss - bl_logloss,
            delta_ece=sh_ece - bl_ece,
            delta_apr=sh_apr - bl_apr,
        )

    # ------------------------------------------------------------------
    # Step 2: 選定パターン差分 (DIAG-02)
    # ------------------------------------------------------------------

    def _compute_group_metrics(
        self,
        group_race_ids: set[str],
    ) -> SelectionGroupMetrics:
        """レースグループの ROI/HR/avg_odds/APR を計算."""
        if not group_race_ids or self.race_diff.empty:
            return SelectionGroupMetrics()

        group_race = self.race_diff[self.race_diff["race_id"].isin(group_race_ids)]
        n_races = len(group_race)
        if n_races == 0:
            return SelectionGroupMetrics()

        # ROI: total_result / total_stake - 1
        bl_stake = group_race.get("baseline_stake", pd.Series(dtype=float))
        bl_result = group_race.get("baseline_result", pd.Series(dtype=float))

        total_stake = float(pd.to_numeric(bl_stake, errors="coerce").fillna(0).sum())
        total_return = float(pd.to_numeric(bl_result, errors="coerce").fillna(0).sum())

        roi = total_return / total_stake - 1.0 if total_stake > 0 else 0.0

        # Hit rate
        n_hits = int((pd.to_numeric(bl_result, errors="coerce").fillna(0) > 0).sum())
        bet_count = int((pd.to_numeric(bl_stake, errors="coerce").fillna(0) > 0).sum())
        hit_rate = n_hits / bet_count if bet_count > 0 else 0.0

        # Average odds
        bl_odds = pd.to_numeric(
            group_race.get("baseline_tanodds", pd.Series(dtype=float)),
            errors="coerce",
        )
        avg_odds = float(bl_odds.mean()) if bl_odds.notna().any() else 0.0

        # Actual/predicted ratio from horse_diff
        apr = 0.0
        if not self.horse_diff.empty and group_race_ids:
            group_horse = self.horse_diff[
                self.horse_diff["race_id"].isin(group_race_ids)
            ]
            p_col = f"{self.baseline_name}_p_win_final"
            if "kakuteijyuni" in group_horse.columns and p_col in group_horse.columns:
                p_vals = pd.to_numeric(group_horse[p_col], errors="coerce")
                y_vals = (group_horse["kakuteijyuni"] == 1).astype(float)
                valid = p_vals.notna() & (p_vals > 0) & (p_vals < 1)
                if valid.sum() > 0:
                    mean_actual = float(y_vals[valid].mean())
                    mean_pred = float(p_vals[valid].mean())
                    apr = mean_actual / mean_pred if mean_pred > 0 else 0.0

        return SelectionGroupMetrics(
            roi=roi,
            hit_rate=hit_rate,
            avg_odds=avg_odds,
            actual_predicted_ratio=apr,
            bet_count=bet_count,
            n_races=n_races,
        )

    def _step2_selection_pattern(self) -> SelectionPatternResult:
        """selected_changed/unchanged レースの差分を計算 (DIAG-02)."""
        if self.race_diff.empty or "selected_changed" not in self.race_diff.columns:
            return SelectionPatternResult()

        changed_ids = set(
            self.race_diff[self.race_diff["selected_changed"] == True]["race_id"]  # noqa: E712
        )
        unchanged_ids = set(
            self.race_diff[self.race_diff["selected_changed"] == False]["race_id"]  # noqa: E712
        )

        changed_metrics = self._compute_group_metrics(changed_ids)
        unchanged_metrics = self._compute_group_metrics(unchanged_ids)

        return SelectionPatternResult(
            changed=changed_metrics,
            unchanged=unchanged_metrics,
            n_changed_races=len(changed_ids),
            n_unchanged_races=len(unchanged_ids),
            delta_roi=changed_metrics.roi - unchanged_metrics.roi,
            delta_hit_rate=changed_metrics.hit_rate - unchanged_metrics.hit_rate,
        )

    # ------------------------------------------------------------------
    # Step 3: キャリブレーション乖離 (DIAG-03)
    # ------------------------------------------------------------------

    def _compute_segment_apr_ece(
        self,
        segment_df: pd.DataFrame,
    ) -> tuple[float, float, float, float]:
        """セグメント内の baseline/shadow APR と ECE を計算."""
        bl_p_col = f"{self.baseline_name}_p_win_final"
        sh_p_col = f"{self.shadow_name}_p_win_final"

        def _apr_ece(p_col: str) -> tuple[float, float]:
            if segment_df.empty or p_col not in segment_df.columns:
                return 0.0, 0.0
            p_vals = pd.to_numeric(segment_df[p_col], errors="coerce")
            if "kakuteijyuni" not in segment_df.columns:
                return 0.0, 0.0
            y_vals = (segment_df["kakuteijyuni"] == 1).astype(float)
            valid = p_vals.notna() & (p_vals > 0) & (p_vals < 1)
            if valid.sum() == 0:
                return 0.0, 0.0
            p_valid = p_vals[valid].values
            y_valid = y_vals[valid].values
            mean_actual = float(np.mean(y_valid))
            mean_pred = float(np.mean(p_valid))
            apr = mean_actual / mean_pred if mean_pred > 0 else 0.0
            ece = ShadowComparisonFramework._compute_ece(p_valid, y_valid, n_bins=10)
            return apr, float(ece)

        bl_apr, bl_ece = _apr_ece(bl_p_col)
        sh_apr, sh_ece = _apr_ece(sh_p_col)

        return bl_apr, sh_apr, bl_ece, sh_ece

    def _add_segment_columns(self, horse_work: pd.DataFrame) -> pd.DataFrame:
        """horse_diff にセグメント列を追加."""
        bl_p_col = f"{self.baseline_name}_p_win_final"

        # --- popularity_band ---
        if "popularity" in horse_work.columns:
            horse_work["popularity_band"] = pd.cut(
                pd.to_numeric(horse_work["popularity"], errors="coerce"),
                bins=POPULARITY_BAND_EDGES,
                labels=POPULARITY_BAND_NAMES,
                right=True,
            ).astype(str)
        else:
            # popularity が horse_diff に無い場合はセグメント計算をスキップ
            # missing_inputs は既に _detect_missing_inputs で記録済み
            pass

        # --- probability_rank_band (常に計算可能) ---
        if bl_p_col in horse_work.columns:
            p_vals = pd.to_numeric(horse_work[bl_p_col], errors="coerce")
            # レース内で降順ランク (rank 1 = 最高確率)
            horse_work["_prob_rank"] = p_vals.groupby(
                horse_work["race_id"], observed=True
            ).rank(ascending=False, method="min")
            horse_work["probability_rank_band"] = pd.cut(
                horse_work["_prob_rank"],
                bins=PROB_RANK_BAND_EDGES,
                labels=PROB_RANK_BAND_NAMES,
                right=True,
            ).astype(str)
            horse_work = horse_work.drop(columns=["_prob_rank"])

        # --- odds_band ---
        if "closing_win_odds" in horse_work.columns:
            horse_work["odds_band"] = pd.cut(
                pd.to_numeric(horse_work["closing_win_odds"], errors="coerce"),
                bins=ODDS_BAND_EDGES,
                labels=ODDS_BAND_NAMES,
                right=True,
            ).astype(str)
        elif "baseline_tanodds" not in horse_work.columns and "tanodds" not in horse_work.columns:
            # race_diff から baseline_tanodds をマージして試みる
            if (
                not self.race_diff.empty
                and "race_id" in self.race_diff.columns
                and "baseline_tanodds" in self.race_diff.columns
            ):
                odds_lookup = self.race_diff[["race_id", "baseline_tanodds"]].drop_duplicates(
                    subset=["race_id"]
                )
                # horse_diff の各馬にレースのオッズを付与 (近似)
                horse_work = horse_work.merge(
                    odds_lookup.rename(columns={"baseline_tanodds": "closing_win_odds"}),
                    on="race_id",
                    how="left",
                )
                if "closing_win_odds" in horse_work.columns:
                    horse_work["odds_band"] = pd.cut(
                        pd.to_numeric(horse_work["closing_win_odds"], errors="coerce"),
                        bins=ODDS_BAND_EDGES,
                        labels=ODDS_BAND_NAMES,
                        right=True,
                    ).astype(str)
            if "closing_win_odds" not in self.horse_diff.columns:
                if "closing_win_odds" not in self.missing_inputs:
                    self.missing_inputs.append("closing_win_odds")

        # --- surface ---
        if "surface" in horse_work.columns:
            pass  # 既に存在
        elif not self.race_diff.empty and "surface" in self.race_diff.columns:
            surface_lookup = self.race_diff[["race_id", "surface"]].drop_duplicates(
                subset=["race_id"]
            )
            horse_work = horse_work.merge(surface_lookup, on="race_id", how="left")
        else:
            if "surface" not in self.missing_inputs:
                self.missing_inputs.append("surface")

        # --- selected_changed ---
        if not self.race_diff.empty and "selected_changed" in self.race_diff.columns:
            sc_lookup = self.race_diff[["race_id", "selected_changed"]].drop_duplicates(
                subset=["race_id"]
            )
            horse_work = horse_work.merge(sc_lookup, on="race_id", how="left")
            horse_work["selected_changed"] = horse_work["selected_changed"].map(
                {True: "changed", False: "unchanged"}
            )

        return horse_work

    def _step3_calibration_by_segment(self) -> CalibrationResult:
        """セグメント別に actual/predicted 比率と ECE を比較 (DIAG-03)."""
        segments: list[SegmentCalibration] = []

        if self.horse_diff.empty:
            return CalibrationResult(segments=segments)

        horse_work = self.horse_diff.copy()
        horse_work = self._add_segment_columns(horse_work)

        segment_cols = [
            "popularity_band",
            "probability_rank_band",
            "odds_band",
            "surface",
            "selected_changed",
        ]

        for seg_col in segment_cols:
            if seg_col not in horse_work.columns:
                continue

            for seg_val, seg_df in horse_work.groupby(seg_col, observed=True):
                seg_val_str = str(seg_val)
                # NaN や "nan" はスキップ
                if seg_val_str in ("nan", "None", ""):
                    continue

                n_samples = len(seg_df)
                if n_samples == 0:
                    continue

                bl_apr, sh_apr, bl_ece, sh_ece = self._compute_segment_apr_ece(seg_df)

                segments.append(SegmentCalibration(
                    segment_name=seg_col,
                    segment_value=seg_val_str,
                    n_samples=n_samples,
                    actual_predicted_ratio_baseline=bl_apr,
                    actual_predicted_ratio_shadow=sh_apr,
                    ece_baseline=bl_ece,
                    ece_shadow=sh_ece,
                    delta_apr=sh_apr - bl_apr,
                    delta_ece=sh_ece - bl_ece,
                ))

        return CalibrationResult(segments=segments)


# ---------------------------------------------------------------------------
# Output: save_diagnosis_results (D-04)
# ---------------------------------------------------------------------------


def _result_to_dict(result: ShadowDiagnosisResult) -> dict[str, Any]:
    """ShadowDiagnosisResult を JSON 可能な dict に変換."""
    return {
        "generated_at": result.generated_at,
        "variant_names": result.variant_names,
        "missing_inputs": result.missing_inputs,
        "step1_probability_quality": {
            "baseline": {
                "brier": result.step1.baseline_brier,
                "logloss": result.step1.baseline_logloss,
                "ece": result.step1.baseline_ece,
                "actual_predicted_ratio": result.step1.baseline_apr,
                "n_horses": result.step1.baseline_n_horses,
            },
            "shadow": {
                "brier": result.step1.shadow_brier,
                "logloss": result.step1.shadow_logloss,
                "ece": result.step1.shadow_ece,
                "actual_predicted_ratio": result.step1.shadow_apr,
                "n_horses": result.step1.shadow_n_horses,
            },
            "delta": {
                "brier": result.step1.delta_brier,
                "logloss": result.step1.delta_logloss,
                "ece": result.step1.delta_ece,
                "actual_predicted_ratio": result.step1.delta_apr,
            },
        },
        "step2_selection_pattern": {
            "changed": {
                "roi": result.step2.changed.roi,
                "hit_rate": result.step2.changed.hit_rate,
                "avg_odds": result.step2.changed.avg_odds,
                "actual_predicted_ratio": result.step2.changed.actual_predicted_ratio,
                "bet_count": result.step2.changed.bet_count,
                "n_races": result.step2.changed.n_races,
            },
            "unchanged": {
                "roi": result.step2.unchanged.roi,
                "hit_rate": result.step2.unchanged.hit_rate,
                "avg_odds": result.step2.unchanged.avg_odds,
                "actual_predicted_ratio": result.step2.unchanged.actual_predicted_ratio,
                "bet_count": result.step2.unchanged.bet_count,
                "n_races": result.step2.unchanged.n_races,
            },
            "delta": {
                "roi": result.step2.delta_roi,
                "hit_rate": result.step2.delta_hit_rate,
            },
            "n_changed_races": result.step2.n_changed_races,
            "n_unchanged_races": result.step2.n_unchanged_races,
        },
        "step3_calibration": {
            "segments": [
                {
                    "segment_name": seg.segment_name,
                    "segment_value": seg.segment_value,
                    "n_samples": seg.n_samples,
                    "baseline_apr": seg.actual_predicted_ratio_baseline,
                    "shadow_apr": seg.actual_predicted_ratio_shadow,
                    "baseline_ece": seg.ece_baseline,
                    "shadow_ece": seg.ece_shadow,
                    "delta_apr": seg.delta_apr,
                    "delta_ece": seg.delta_ece,
                }
                for seg in result.step3.segments
            ],
        },
        "recommendations": [],
    }


def save_diagnosis_results(
    diagnosis_result: ShadowDiagnosisResult,
    output_dir: Path,
) -> dict[str, Path]:
    """診断結果を JSON + Markdown で出力 (D-04).

    Args:
        diagnosis_result: ShadowDiagnosis.run() の戻り値.
        output_dir: 出力ディレクトリ.

    Returns:
        出力ファイルパスの dict {"result_json": Path, "summary_md": Path}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON 出力
    result_dict = _result_to_dict(diagnosis_result)
    result_json_path = output_dir / "shadow_diagnosis_result.json"
    result_json_path.write_text(
        json.dumps(result_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Markdown 要約出力
    summary_path = output_dir / "shadow_diagnosis_summary.md"
    summary_lines = _build_summary_md(diagnosis_result)
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    return {"result_json": result_json_path, "summary_md": summary_path}


def _build_summary_md(result: ShadowDiagnosisResult) -> list[str]:
    """ShadowDiagnosisResult から Markdown 要約を生成."""
    def _md_row(label: str, bl: str, sh: str, delta: str) -> str:
        return f"| {label} | {bl} | {sh} | {delta} |"

    lines: list[str] = [
        "# Phase 43: Shadow Diagnosis Summary",
        "",
        f"**Generated:** {result.generated_at}",
        f"**Variants:** {', '.join(result.variant_names)}",
        "",
    ]

    # セクション1: Probability Quality
    s1 = result.step1
    lines += [
        "## 1. Probability Quality",
        "",
        "| Metric | Baseline | Shadow | Delta |",
        "|--------|----------|--------|-------|",
        (
            _md_row("Brier", f"{s1.baseline_brier:.4f}",
                    f"{s1.shadow_brier:.4f}", f"{s1.delta_brier:+.4f}")
        ),
        (
            _md_row("Logloss", f"{s1.baseline_logloss:.4f}",
                    f"{s1.shadow_logloss:.4f}", f"{s1.delta_logloss:+.4f}")
        ),
        _md_row("ECE", f"{s1.baseline_ece:.4f}", f"{s1.shadow_ece:.4f}", f"{s1.delta_ece:+.4f}"),
        _md_row("APR", f"{s1.baseline_apr:.4f}", f"{s1.shadow_apr:.4f}", f"{s1.delta_apr:+.4f}"),
        "",
    ]

    # セクション2: Selection Pattern
    s2 = result.step2
    lines += [
        "## 2. Selection Pattern",
        "",
        f"**Changed races:** {s2.n_changed_races}"
        f" | **Unchanged races:** {s2.n_unchanged_races}",
        "",
        "| Group | ROI | Hit Rate | Avg Odds | APR | Bet Count |",
        "|-------|-----|----------|----------|-----|-----------|",
        (
            f"| Changed | {s2.changed.roi:.4f} | {s2.changed.hit_rate:.4f}"
            f" | {s2.changed.avg_odds:.2f}"
            f" | {s2.changed.actual_predicted_ratio:.4f}"
            f" | {s2.changed.bet_count} |"
        ),
        (
            f"| Unchanged | {s2.unchanged.roi:.4f} | {s2.unchanged.hit_rate:.4f}"
            f" | {s2.unchanged.avg_odds:.2f}"
            f" | {s2.unchanged.actual_predicted_ratio:.4f}"
            f" | {s2.unchanged.bet_count} |"
        ),
        "",
        (
            f"**Delta ROI:** {s2.delta_roi:+.4f} | "
            f"**Delta Hit Rate:** {s2.delta_hit_rate:+.4f}"
        ),
        "",
    ]

    # セクション3: Top Calibration Gaps (|delta_apr|+|delta_ece| 降順上位5件)
    sorted_segments = sorted(
        result.step3.segments,
        key=lambda s: abs(s.delta_apr) + abs(s.delta_ece),
        reverse=True,
    )
    lines += [
        "## 3. Top Calibration Gaps",
        "",
        "| Segment | Value | N | BL APR | SH APR | D APR | BL ECE | SH ECE | D ECE |",
        "|---------|-------|---|--------|--------|-------|--------|--------|-------|",
    ]
    for seg in sorted_segments[:5]:
        lines.append(
            f"| {seg.segment_name} | {seg.segment_value} | {seg.n_samples}"
            f" | {seg.actual_predicted_ratio_baseline:.4f}"
            f" | {seg.actual_predicted_ratio_shadow:.4f}"
            f" | {seg.delta_apr:+.4f}"
            f" | {seg.ece_baseline:.4f}"
            f" | {seg.ece_shadow:.4f}"
            f" | {seg.delta_ece:+.4f} |"
        )
    lines.append("")

    # セクション4: Missing Inputs
    lines += [
        "## 4. Missing Inputs",
        "",
    ]
    if result.missing_inputs:
        for inp in result.missing_inputs:
            lines.append(f"- {inp}")
    else:
        lines.append("None")
    lines.append("")

    # セクション5: Recommendations (Phase 44で活用)
    lines += [
        "## 5. Recommendations for Phase 44/45",
        "",
        "(To be populated by Phase 44 analysis)",
        "",
    ]

    return lines


# ---------------------------------------------------------------------------
# Output: ShadowDiagnosisReportGenerator (D-04 HTML)
# ---------------------------------------------------------------------------


class ShadowDiagnosisReportGenerator:
    """Shadow Diagnosis HTML レポート生成 (D-04).

    Phase 41 ShadowComparisonReportGenerator パターンに従う。
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.template_dir = Path(__file__).parent / "templates"

    def generate(self, diagnosis_result: ShadowDiagnosisResult) -> Path:
        """HTML レポートを生成し、ファイルパスを返す."""
        env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True,
        )
        template = env.get_template("shadow_diagnosis_report.html")

        # context 構築
        context: dict[str, Any] = {
            "generated_at": diagnosis_result.generated_at,
            "variant_names": diagnosis_result.variant_names,
            "missing_inputs": diagnosis_result.missing_inputs,
            # Step 1
            "step1": diagnosis_result.step1,
            # Step 2
            "step2": diagnosis_result.step2,
            # Step 3 segments
            "step3_segments": diagnosis_result.step3.segments,
        }

        html = template.render(**context)

        outpath = self.output_dir / "shadow_diagnosis_report.html"
        outpath.write_text(html, encoding="utf-8")
        return outpath
