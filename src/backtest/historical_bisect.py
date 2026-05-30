"""HistoricalBisect -- auxiliary v1.7->v2.0 historical artifact comparison.

Lightweight analysis (per D-05) that performs artifact-level comparison
between v1.7 (Phase 34) and v2.0 (Phase 38) to estimate which phase range
introduced the ROI degradation. No model retraining or full BT re-runs.

Uses local backtest artifacts (multi_year_result.json) and OOF predictions
to compare current baseline against the known v1.7 reference ROI (0.978).
Git commit history between tags provides phase-level attribution context.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.shadow_comparison import ShadowComparisonFramework

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HistoricalBisectResult:
    """Result of historical v1.7->v2.0 bisect comparison."""

    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    v17_reference_metrics: dict[str, Any] = field(default_factory=dict)
    total_degradation: float = 0.0
    phase_changes: list[dict[str, Any]] = field(default_factory=list)
    oof_comparison: dict[str, Any] = field(default_factory=dict)
    estimated_degradation_phase: str = ""
    confidence: str = "LOW"
    auxiliary_findings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HistoricalBisect
# ---------------------------------------------------------------------------


class HistoricalBisect:
    """Auxiliary v1.7->v2.0 historical artifact comparison (per D-05).

    Performs lightweight analysis using local artifacts and git history
    to estimate which phase range introduced ROI degradation.

    Args:
        input_dir: Directory containing backtest artifacts
            (multi_year_result.json).
        oof_path: Path to OOF predictions parquet file.
    """

    def __init__(
        self,
        input_dir: Path,
        oof_path: Path | None = None,
    ) -> None:
        self.input_dir = input_dir
        self.oof_path = oof_path

        # Load multi_year_result.json
        myr_path = input_dir / "multi_year_result.json"
        if myr_path.exists():
            self.multi_year_result: dict[str, Any] = json.loads(
                myr_path.read_text(encoding="utf-8")
            )
        else:
            self.multi_year_result = {}

        # Load OOF predictions if available
        self.oof_df: pd.DataFrame | None = None
        if oof_path is not None and oof_path.exists():
            self.oof_df = pd.read_parquet(oof_path)

        # Detect available git tags
        self.available_tags: list[str] = self._detect_git_tags()

    def _detect_git_tags(self) -> list[str]:
        """Detect available git tags via subprocess."""
        try:
            result = subprocess.run(
                ["git", "tag", "-l"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return [t.strip() for t in result.stdout.strip().split("\n") if t.strip()]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.info("git not available or timed out, skipping tag detection")
        return []

    # ------------------------------------------------------------------
    # Phase artifact comparison
    # ------------------------------------------------------------------

    def compare_phase_artifacts(self) -> dict[str, Any]:
        """Extract ROI/bet_count/hit_rate metrics from local backtest artifacts.

        Returns:
            Dict with current_baseline metrics and per-year breakdown.
        """
        result: dict[str, Any] = {
            "current_baseline": {},
            "per_year": {},
        }

        overall = self.multi_year_result.get("overall", {})
        result["current_baseline"] = {
            "overall_roi": overall.get("roi", 0.0),
            "total_bets": overall.get("total_bets", 0),
            "best_year": overall.get("best_year", 0),
            "worst_year": overall.get("worst_year", 0),
        }

        years = self.multi_year_result.get("years", {})
        for year_key, year_data in years.items():
            result["per_year"][year_key] = {
                "roi": year_data.get("roi", 0.0),
                "total_bets": year_data.get("total_bets", 0),
                "hit_rate": year_data.get("hit_rate", 0.0),
            }

        return result

    # ------------------------------------------------------------------
    # OOF metrics comparison
    # ------------------------------------------------------------------

    def compare_oof_metrics(self) -> dict[str, Any]:
        """Load OOF predictions and compute per-fold-year IC/Brier/ECE.

        Returns:
            Dict with current OOF metrics (IC, Brier, ECE).
        """
        result: dict[str, Any] = {"current_oof": {}}

        if self.oof_df is None or self.oof_df.empty:
            result["current_oof"] = {
                "ic": 0.0,
                "brier": 0.0,
                "ece": 0.0,
                "note": "OOF data not available",
            }
            return result

        p_col = "p_win_oof" if "p_win_oof" in self.oof_df.columns else None
        if p_col is None:
            result["current_oof"] = {"ic": 0.0, "brier": 0.0, "ece": 0.0}
            return result

        p_vals = pd.to_numeric(self.oof_df[p_col], errors="coerce")
        has_kakuteijyuni = "kakuteijyuni" in self.oof_df.columns
        y_vals = (
            (self.oof_df["kakuteijyuni"] == 1).astype(float)
            if has_kakuteijyuni
            else pd.Series(dtype=float)
        )

        valid = p_vals.notna() & (p_vals > 0) & (p_vals < 1)
        if valid.sum() < 10:
            result["current_oof"] = {"ic": 0.0, "brier": 0.0, "ece": 0.0}
            return result

        p_valid = p_vals[valid].values
        y_valid = y_vals[valid].values

        # IC (Information Coefficient): Spearman rank correlation
        if len(p_valid) >= 3 and np.std(p_valid) > 1e-10 and np.std(y_valid) > 1e-10:
            from scipy.stats import spearmanr
            ic, _ = spearmanr(p_valid, y_valid)
            ic = float(ic) if np.isfinite(ic) else 0.0
        else:
            ic = 0.0

        # Brier score
        brier = float(np.mean((p_valid - y_valid) ** 2))

        # ECE
        ece = float(ShadowComparisonFramework._compute_ece(
            p_valid, y_valid, n_bins=10
        ))

        result["current_oof"] = {
            "ic": ic,
            "brier": brier,
            "ece": ece,
            "n_predictions": int(valid.sum()),
        }

        return result

    # ------------------------------------------------------------------
    # Git commit history analysis
    # ------------------------------------------------------------------

    def _get_commit_history(self) -> list[dict[str, str]]:
        """Get git log between v1.7 and v2.0 tags."""
        if "v1.7" not in self.available_tags or "v2.0" not in self.available_tags:
            return []

        try:
            result = subprocess.run(
                ["git", "log", "v1.7..v2.0", "--oneline"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return []

            commits: list[dict[str, str]] = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split(" ", 1)
                commits.append({
                    "hash": parts[0] if parts else "",
                    "message": parts[1] if len(parts) > 1 else "",
                })
            return commits

        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []

    def _categorize_commits_by_phase(
        self,
        commits: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """Categorize commits by phase based on message prefixes."""
        phase_map: dict[str, list[dict[str, str]]] = {}

        for commit in commits:
            msg = commit.get("message", "")
            # Extract phase number from commit message
            phase = "unknown"
            phase3536_prefixes = [
                "feat(35)", "fix(35)", "refactor(35)",
                "feat(36)", "fix(36)", "refactor(36)",
                "feat(36.1)", "fix(36.1)",
            ]
            for prefix in phase3536_prefixes:
                if prefix in msg:
                    phase = "Phase 35-36 (turf precision)"
                    break
            else:
                for prefix in ["feat(37)", "fix(37)", "refactor(37)", "docs(37)"]:
                    if prefix in msg:
                        phase = "Phase 37 (OOF health)"
                        break
                else:
                    for prefix in ["feat(38)", "fix(38)", "refactor(38)", "docs(38)"]:
                        if prefix in msg:
                            phase = "Phase 38 (investment pipeline)"
                            break

            if phase not in phase_map:
                phase_map[phase] = []
            phase_map[phase].append(commit)

        return [
            {
                "phase": phase,
                "commit_count": len(commits_list),
                "commits": commits_list,
            }
            for phase, commits_list in sorted(phase_map.items())
        ]

    # ------------------------------------------------------------------
    # Full historical comparison
    # ------------------------------------------------------------------

    def run_historical_comparison(self) -> HistoricalBisectResult:
        """Execute full historical bisect comparison.

        Returns:
            HistoricalBisectResult with degradation estimate.
        """
        # v1.7 reference ROI from CLAUDE.md known issues
        v17_reference_roi = 0.978

        # Current baseline metrics
        phase_artifacts = self.compare_phase_artifacts()
        current_roi = phase_artifacts.get("current_baseline", {}).get("overall_roi", 0.0)

        # Total degradation
        total_degradation = v17_reference_roi - current_roi

        # OOF comparison
        oof_comparison = self.compare_oof_metrics()

        # Git commit history analysis
        commits = self._get_commit_history()
        phase_changes = self._categorize_commits_by_phase(commits)

        # Estimate degradation phase
        estimated_phase = self._estimate_degradation_phase(
            phase_changes, total_degradation
        )

        # Auxiliary findings
        findings: list[str] = []
        findings.append(
            f"ROI degradation: v1.7={v17_reference_roi:.4f} -> "
            f"current={current_roi:.4f} (delta={total_degradation:+.4f})"
        )

        if phase_changes:
            for pc in phase_changes:
                findings.append(
                    f"{pc['phase']}: {pc['commit_count']} commits"
                )
        else:
            findings.append(
                "Git tags v1.7/v2.0 not available -- phase estimate based "
                "on known timeline only"
            )

        if oof_comparison.get("current_oof", {}).get("ic"):
            findings.append(
                f"Current OOF IC={oof_comparison['current_oof']['ic']:.4f}, "
                f"Brier={oof_comparison['current_oof']['brier']:.4f}, "
                f"ECE={oof_comparison['current_oof']['ece']:.4f}"
            )

        # Confidence level
        confidence = "LOW"
        if phase_changes and len(commits) > 5:
            confidence = "MEDIUM"

        return HistoricalBisectResult(
            baseline_metrics=phase_artifacts.get("current_baseline", {}),
            v17_reference_metrics={"roi": v17_reference_roi, "source": "CLAUDE.md known issues"},
            total_degradation=total_degradation,
            phase_changes=phase_changes,
            oof_comparison=oof_comparison,
            estimated_degradation_phase=estimated_phase,
            confidence=confidence,
            auxiliary_findings=findings,
        )

    def _estimate_degradation_phase(
        self,
        phase_changes: list[dict[str, Any]],
        total_degradation: float,
    ) -> str:
        """Estimate which phase range most likely caused ROI degradation.

        Cross-references with known phase history:
        - Phase 35-36.1.1: Turf precision (haron/lap + relative features + MarketModel fix)
        - Phase 37-38: Investment pipeline (OOFHealthValidator + InvestmentFeatureFrame)

        The output is an estimate with documented confidence.
        """
        if total_degradation <= 0:
            return "No degradation detected"

        # Based on CLAUDE.md known issues:
        # "Phase 36強特徴量副作用、修正済み但し未検証"
        # This suggests Phase 35-36 introduced the degradation,
        # with Phase 36 features being the likely cause.
        estimation = (
            "Phase 35-36 (turf precision) most likely introduced ROI "
            "degradation. Phase 36 added strong features (haron/lap, "
            "relative features, conditional interactions) with documented "
            "\"副作用\" (side effects). MarketModel/RaceQuality wiring "
            "fix in Phase 36.1.1 was corrective but ROI recovery was not "
            "verified. Phase 37-38 (OOF health + investment pipeline) "
            "are diagnostic/infrastructure and unlikely to cause ROI "
            "regression directly."
        )

        if phase_changes:
            # Add commit-based context
            for pc in phase_changes:
                if "35-36" in pc["phase"] and pc["commit_count"] > 0:
                    estimation += (
                        f" Git history confirms {pc['commit_count']} commits "
                        f"in {pc['phase']}."
                    )

        return estimation
