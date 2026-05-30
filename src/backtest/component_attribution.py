"""ComponentAttribution -- post-hoc component attribution for DeploymentGate FAILs.

Reads Phase 41/43/42 artifacts (shadow_horse_diff, shadow_race_diff,
diagnosis_result, deployment_gate_result) and trained model files to attribute
each DeploymentGate FAIL (ECE degradation, APR deviation, bet_count loss, OBF impact)
to a specific component (MAWC / Ranker / OddsBandFilter / Selection logic).

Implements D-02 sequential analysis: ECE -> APR -> bet_count -> OBF.
Implements D-03 coefficient analysis: MAWC (51-dim LogisticRegression) + Ranker (Ridge).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backtest.shadow_comparison import ShadowComparisonFramework
from backtest.shadow_diagnosis import (
    ODDS_BAND_EDGES,
    ODDS_BAND_NAMES,
    POPULARITY_BAND_EDGES,
    POPULARITY_BAND_NAMES,
    PROB_RANK_BAND_EDGES,
    PROB_RANK_BAND_NAMES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoefficientAnalysisResult:
    """Result of MAWC + Ranker coefficient analysis."""

    mawc_coef_analysis: dict[str, Any] = field(default_factory=dict)
    ranker_coef_analysis: dict[str, Any] = field(default_factory=dict)
    segment_contribution_comparison: dict[str, Any] = field(default_factory=dict)
    upstream_anomaly_check: str = ""


@dataclass(frozen=True)
class ComponentAttributionResult:
    """Complete result of 4-step sequential attribution."""

    ece_attribution: dict[str, Any] = field(default_factory=dict)
    apr_attribution: dict[str, Any] = field(default_factory=dict)
    bet_count_attribution: dict[str, Any] = field(default_factory=dict)
    coefficient_analysis: CoefficientAnalysisResult | None = None
    upstream_anomaly_check: str = ""
    recommendations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ComponentAttribution
# ---------------------------------------------------------------------------


class ComponentAttribution:
    """Post-hoc component attribution engine for DeploymentGate FAILs.

    Reads Phase 41/43/42 artifacts and trained model files to attribute each
    DeploymentGate FAIL to a specific component. No model retraining or
    BacktestEngine re-runs -- purely analytical.

    Args:
        input_dir: Directory containing Phase 41/43/42 shadow artifacts.
        models_dir: Directory containing trained model artifacts.
            Defaults to data/models-backtest.
    """

    def __init__(
        self,
        input_dir: Path,
        models_dir: Path | None = None,
    ) -> None:
        self.input_dir = input_dir
        self.models_dir = models_dir or Path("data/models-backtest")

        # Load Phase 41 artifacts
        self.horse_diff: pd.DataFrame = pd.read_parquet(
            input_dir / "shadow_horse_diff.parquet"
        )
        self.race_diff: pd.DataFrame = pd.read_parquet(
            input_dir / "shadow_race_diff.parquet"
        )

        # Load Phase 43 diagnosis result
        diag_path = input_dir / "diagnosis" / "shadow_diagnosis_result.json"
        if diag_path.exists():
            self.diagnosis: dict[str, Any] = json.loads(
                diag_path.read_text(encoding="utf-8")
            )
        else:
            self.diagnosis = {}

        # Load Phase 42 gate result
        gate_path = input_dir / "gates" / "deployment_gate_result.json"
        if gate_path.exists():
            self.gate_result: dict[str, Any] = json.loads(
                gate_path.read_text(encoding="utf-8")
            )
        else:
            self.gate_result = {}

        # Resolve variant names from column prefixes
        self.baseline_name, self.shadow_name = self._resolve_variant_names()

    def _resolve_variant_names(self) -> tuple[str, str]:
        """Resolve baseline and shadow variant names from horse_diff columns."""
        cols = self.horse_diff.columns.tolist()
        p_win_cols = [c for c in cols if c.endswith("_p_win_final")]
        if len(p_win_cols) >= 2:
            prefixes = [c.replace("_p_win_final", "") for c in p_win_cols]
            # baseline is typically first alphabetically
            return prefixes[0], prefixes[1]
        return "baseline", "shadow"

    # ------------------------------------------------------------------
    # Step 1: ECE Degradation Attribution (D-02 Step 1)
    # ------------------------------------------------------------------

    def attribute_ece_degradation(self) -> dict[str, Any]:
        """Attribute ECE degradation to MAWC direct effect or selection population.

        Segments horses by odds_band / popularity_band / probability_rank_band
        and computes per-segment ECE with baseline vs shadow probabilities.

        Returns:
            Dict with 'segments' (list of per-segment results) and
            'attribution' (list of attribution strings).
        """
        horse_work = self.horse_diff.copy()
        bl_p_col = f"{self.baseline_name}_p_win_final"
        sh_p_col = f"{self.shadow_name}_p_win_final"

        # Add segment columns
        horse_work = self._add_segment_columns(horse_work)

        segments: list[dict[str, Any]] = []
        attributions: list[str] = []

        # Analyze per segment type
        for seg_col, band_names in [
            ("odds_band", ODDS_BAND_NAMES),
            ("popularity_band", POPULARITY_BAND_NAMES),
            ("probability_rank_band", PROB_RANK_BAND_NAMES),
        ]:
            if seg_col not in horse_work.columns:
                continue

            for seg_val in band_names:
                seg_df = horse_work[horse_work[seg_col] == seg_val]
                if seg_df.empty:
                    continue

                bl_ece = self._compute_ece_for_col(seg_df, bl_p_col)
                sh_ece = self._compute_ece_for_col(seg_df, sh_p_col)
                bl_apr = self._compute_apr_for_col(seg_df, bl_p_col)
                sh_apr = self._compute_apr_for_col(seg_df, sh_p_col)

                p_win_shift = float(
                    (seg_df[sh_p_col] - seg_df[bl_p_col]).mean()
                )

                segments.append({
                    "segment_name": seg_col,
                    "segment_value": seg_val,
                    "n_samples": len(seg_df),
                    "baseline_ece": bl_ece,
                    "shadow_ece": sh_ece,
                    "delta_ece": sh_ece - bl_ece,
                    "baseline_apr": bl_apr,
                    "shadow_apr": sh_apr,
                    "mean_p_win_shift": p_win_shift,
                })

                # Attribution logic
                if sh_ece > bl_ece and abs(p_win_shift) > 0.005:
                    attributions.append(
                        f"{seg_col}={seg_val}: MAWC direct effect "
                        f"(ECE {bl_ece:.4f}->{sh_ece:.4f}, "
                        f"p_win shift={p_win_shift:+.4f})"
                    )
                elif sh_ece > bl_ece:
                    attributions.append(
                        f"{seg_col}={seg_val}: selection population effect "
                        f"(ECE {bl_ece:.4f}->{sh_ece:.4f})"
                    )

        if not attributions:
            attributions.append("No significant ECE degradation detected")

        return {"segments": segments, "attribution": attributions}

    # ------------------------------------------------------------------
    # Step 2: APR Deviation Attribution (D-02 Step 2)
    # ------------------------------------------------------------------

    def attribute_apr_deviation(self) -> dict[str, Any]:
        """Attribute APR deviation to MAWC probability level or Ranker selection bias.

        Separates all-horse APR from selected-horse APR to determine whether
        the over/under-prediction is from MAWC calibrating probabilities or
        from Ranker selecting a biased subpopulation.

        Returns:
            Dict with all_horse_apr, selected_horse_apr, and attribution.
        """
        bl_p_col = f"{self.baseline_name}_p_win_final"
        sh_p_col = f"{self.shadow_name}_p_win_final"
        bl_sel_col = f"{self.baseline_name}_selected"
        sh_sel_col = f"{self.shadow_name}_selected"

        # All-horse APR
        all_bl_apr = self._compute_apr_for_col(self.horse_diff, bl_p_col)
        all_sh_apr = self._compute_apr_for_col(self.horse_diff, sh_p_col)

        # Selected-horse APR (baseline selected vs shadow selected)
        sel_bl_apr = 0.0
        sel_sh_apr = 0.0

        if bl_sel_col in self.horse_diff.columns:
            bl_selected = self.horse_diff[self.horse_diff[bl_sel_col] == True]  # noqa: E712
            sel_bl_apr = self._compute_apr_for_col(bl_selected, bl_p_col)

        if sh_sel_col in self.horse_diff.columns:
            sh_selected = self.horse_diff[self.horse_diff[sh_sel_col] == True]  # noqa: E712
            sel_sh_apr = self._compute_apr_for_col(sh_selected, sh_p_col)

        all_delta = all_sh_apr - all_bl_apr
        sel_delta = sel_sh_apr - sel_bl_apr

        # Attribution logic
        attributions: list[str] = []
        if abs(all_delta) > abs(sel_delta):
            attributions.append(
                f"MAWC probability level issue: all-horse APR delta "
                f"({all_delta:+.4f}) exceeds selected-only ({sel_delta:+.4f})"
            )
        else:
            attributions.append(
                f"Ranker selection bias: selected-only APR delta "
                f"({sel_delta:+.4f}) exceeds all-horse ({all_delta:+.4f})"
            )

        return {
            "all_horse_apr": {
                "baseline_apr": all_bl_apr,
                "shadow_apr": all_sh_apr,
                "delta_apr": all_delta,
            },
            "selected_horse_apr": {
                "baseline_apr": sel_bl_apr,
                "shadow_apr": sel_sh_apr,
                "delta_apr": sel_delta,
            },
            "attribution": attributions,
        }

    # ------------------------------------------------------------------
    # Step 3: Bet Count Loss Attribution (D-02 Step 3 + D-04)
    # ------------------------------------------------------------------

    def attribute_bet_count_loss(self) -> dict[str, Any]:
        """Attribute bet_count loss to Ranker exclusion, selection changes, OBF.

        Decomposes bet_count gap into:
        (a) Ranker exclusion: races where selection changed
        (b) Selection stack changes in unchanged races
        (c) OBF pass-through rate changes (per D-04)

        Returns:
            Dict with bet counts, Ranker exclusion details, and OBF analysis.
        """
        # Count bets from race_diff (stake > 0)
        baseline_bet_count = 0
        shadow_bet_count = 0

        if not self.race_diff.empty:
            bl_stake = self._resolve_col(
                self.race_diff, self.baseline_name, "stake"
            )
            sh_stake = self._resolve_col(
                self.race_diff, self.shadow_name, "stake"
            )
            if bl_stake is not None:
                baseline_bet_count = int(
                    (pd.to_numeric(self.race_diff[bl_stake], errors="coerce").fillna(0) > 0)
                    .sum()
                )
            if sh_stake is not None:
                shadow_bet_count = int(
                    (pd.to_numeric(self.race_diff[sh_stake], errors="coerce").fillna(0) > 0)
                    .sum()
                )

        # If no data in race_diff, use gate_result
        if baseline_bet_count == 0 and shadow_bet_count == 0 and self.gate_result:
            bl_metrics = self.gate_result.get("baseline_metrics", {})
            sh_metrics = self.gate_result.get("shadow_metrics", {})
            for year_key in bl_metrics:
                baseline_bet_count += bl_metrics[year_key].get("bet_count", 0)
            for year_key in sh_metrics:
                shadow_bet_count += sh_metrics[year_key].get("bet_count", 0)

        gap = baseline_bet_count - shadow_bet_count

        # Ranker exclusion: count changed races
        selection_changed_count = 0
        changed_baseline_bets = 0
        changed_shadow_bets = 0

        if "selected_changed" in self.race_diff.columns:
            changed_races = self.race_diff[self.race_diff["selected_changed"] == True]  # noqa: E712
            selection_changed_count = len(changed_races)

            # Count bets in changed races
            if bl_stake is not None:
                changed_baseline_bets = int(
                    (pd.to_numeric(changed_races[bl_stake], errors="coerce").fillna(0) > 0)
                    .sum()
                )
            if sh_stake is not None:
                changed_shadow_bets = int(
                    (pd.to_numeric(changed_races[sh_stake], errors="coerce").fillna(0) > 0)
                    .sum()
                )

        # OBF analysis (per D-04)
        obf_analysis = self._analyze_obf_impact()

        return {
            "baseline_bet_count": baseline_bet_count,
            "shadow_bet_count": shadow_bet_count,
            "gap": gap,
            "ranker_exclusion": {
                "changed_baseline_bets": changed_baseline_bets,
                "changed_shadow_bets": changed_shadow_bets,
                "excluded_by_ranker": changed_baseline_bets - changed_shadow_bets,
            },
            "selection_changed_count": selection_changed_count,
            "obf_analysis": obf_analysis,
        }

    def _analyze_obf_impact(self) -> dict[str, Any]:
        """Analyze OddsBandFilter impact on bet_count (D-04).

        Compares excluded_bands and pass rates between baseline and shadow.
        Per D-04, OBF analysis is integrated into bet_count step.
        """
        # Since we're doing post-hoc analysis without re-running OBF,
        # we analyze the stake distribution to infer OBF impact
        bl_odds_col = self._resolve_col(
            self.race_diff, self.baseline_name, "tanodds"
        )
        sh_odds_col = self._resolve_col(
            self.race_diff, self.shadow_name, "tanodds"
        )

        obf_analysis: dict[str, Any] = {
            "method": "post_hoc_stake_analysis",
            "note": (
                "training_bet_history ROI overstatement is documented as "
                "in-sample calibration risk but not causally linked without ablation"
            ),
        }

        if bl_odds_col is not None and sh_odds_col is not None:
            bl_odds = pd.to_numeric(
                self.race_diff[bl_odds_col], errors="coerce"
            ).dropna()
            sh_odds = pd.to_numeric(
                self.race_diff[sh_odds_col], errors="coerce"
            ).dropna()

            obf_analysis["baseline_odds_distribution"] = {
                "mean": float(bl_odds.mean()) if len(bl_odds) > 0 else 0.0,
                "median": float(bl_odds.median()) if len(bl_odds) > 0 else 0.0,
            }
            obf_analysis["shadow_odds_distribution"] = {
                "mean": float(sh_odds.mean()) if len(sh_odds) > 0 else 0.0,
                "median": float(sh_odds.median()) if len(sh_odds) > 0 else 0.0,
            }

        return obf_analysis

    # ------------------------------------------------------------------
    # Step 4: Coefficient Analysis (D-03 / BISECT-02)
    # ------------------------------------------------------------------

    def analyze_mawc_coefficients(
        self,
        year: int | None = None,
        surface: str = "turf",
    ) -> dict[str, Any]:
        """Analyze MAWC LogisticRegression coefficients (51-dim).

        Loads the trained MAWC calibrator from joblib, extracts coef_ and
        feature_names, identifies top contributing features and per-segment
        effective contributions.

        Args:
            year: Model year to analyze. If None, tries 2024 then 2025.
            surface: Surface type (turf or dirt).

        Returns:
            Dict with feature_coefficients, top_features, segment_contributions.
        """
        if year is None:
            year = 2024

        model_path = (
            self.models_dir / str(year)
            / f"market_aware_win_calibrator_{surface}.joblib"
        )

        if not model_path.exists():
            return {
                "error": f"MAWC model not found: {model_path}",
                "feature_coefficients": [],
                "top_features": [],
                "segment_contributions": {},
            }

        state: dict[str, Any] = joblib.load(model_path)
        calibrator = state.get("calibrator")
        feature_names: list[str] = state.get("feature_names", [])
        training_summary: dict[str, Any] = state.get("training_summary", {})

        if calibrator is None or not feature_names:
            return {
                "error": "MAWC state missing calibrator or feature_names",
                "feature_coefficients": [],
                "top_features": [],
                "segment_contributions": {},
            }

        coef = calibrator.coef_[0]  # shape (51,)

        # Per-feature coefficients
        feature_coefs: list[dict[str, Any]] = [
            {
                "feature": name,
                "coefficient": float(coef[i]),
                "abs_coefficient": float(abs(coef[i])),
            }
            for i, name in enumerate(feature_names)
        ]

        # Top features by absolute coefficient
        sorted_coefs = sorted(feature_coefs, key=lambda x: x["abs_coefficient"], reverse=True)
        top_features = sorted_coefs[:10]

        # Per-segment effective contribution
        # For key segments, compute: coef[main] + sum(coef[interaction_with_main * indicator])
        segment_contributions = self._compute_segment_contributions(
            coef, feature_names
        )

        return {
            "feature_coefficients": feature_coefs,
            "top_features": top_features,
            "segment_contributions": segment_contributions,
            "training_summary": training_summary,
            "beta_market_contribution": training_summary.get(
                "beta_market_contribution", 0.0
            ),
        }

    def _compute_segment_contributions(
        self,
        coef: np.ndarray,
        feature_names: list[str],
    ) -> dict[str, Any]:
        """Compute per-segment effective MAWC contribution.

        For each segment type (odds_band, pop_bucket, p_rank), computes:
        effective = coef[logit_market] + coef[logit_market_x_segment]
        for the relevant segment indicator = 1.
        """
        contributions: dict[str, Any] = {}

        logit_market_idx = (
            feature_names.index("logit_market")
            if "logit_market" in feature_names else None
        )
        logit_model_idx = (
            feature_names.index("logit_model")
            if "logit_model" in feature_names else None
        )

        if logit_market_idx is None:
            return contributions

        # Find interaction terms
        market_interaction_indices: dict[str, int] = {}
        model_interaction_indices: dict[str, int] = {}
        for i, name in enumerate(feature_names):
            if name.startswith("logit_market_x_"):
                segment_key = name.replace("logit_market_x_", "")
                market_interaction_indices[segment_key] = i
            elif name.startswith("logit_model_x_"):
                segment_key = name.replace("logit_model_x_", "")
                model_interaction_indices[segment_key] = i

        # For each segment, compute effective contribution when that segment is active
        for seg_key, mkt_idx in market_interaction_indices.items():
            mdl_idx = model_interaction_indices.get(seg_key)
            effective_market = float(coef[logit_market_idx] + coef[mkt_idx])
            effective_model = (
                float(coef[logit_model_idx] + coef[mdl_idx])
                if mdl_idx is not None and logit_model_idx is not None
                else 0.0
            )
            contributions[seg_key] = {
                "effective_market_contribution": effective_market,
                "effective_model_contribution": effective_model,
                "market_interaction_coef": float(coef[mkt_idx]),
            }

        return contributions

    def analyze_ranker_coefficients(
        self,
        year: int | None = None,
    ) -> dict[str, Any]:
        """Analyze Ranker Ridge coefficients (relevance + value scorers).

        Args:
            year: Model year to analyze. If None, tries 2024.

        Returns:
            Dict with relevance and value coefficient analysis.
        """
        if year is None:
            year = 2024

        model_path = (
            self.models_dir / str(year)
            / "win_race_level_ranker_turf.joblib"
        )

        if not model_path.exists():
            return {
                "error": f"Ranker model not found: {model_path}",
                "relevance_coefficients": [],
                "value_coefficients": [],
                "relevance_top_features": [],
                "value_top_features": [],
            }

        state: dict[str, Any] = joblib.load(model_path)

        rel_scorer = state.get("relevance_scorer_turf")
        val_scorer = state.get("value_scorer_turf")
        rel_features: list[str] = state.get("relevance_feature_names", [])
        val_features: list[str] = state.get("value_feature_names", [])

        result: dict[str, Any] = {}

        # Relevance coefficients
        if rel_scorer is not None and rel_features:
            rel_coef = rel_scorer.coef_[0] if rel_scorer.coef_.ndim > 1 else rel_scorer.coef_
            rel_coefs = [
                {
                    "feature": name,
                    "coefficient": float(rel_coef[i]),
                    "abs_coefficient": float(abs(rel_coef[i])),
                }
                for i, name in enumerate(rel_features)
                if i < len(rel_coef)
            ]
            result["relevance_coefficients"] = rel_coefs
            result["relevance_top_features"] = sorted(
                rel_coefs, key=lambda x: x["abs_coefficient"], reverse=True
            )[:5]
        else:
            result["relevance_coefficients"] = []
            result["relevance_top_features"] = []

        # Value coefficients
        if val_scorer is not None and val_features:
            val_coef = val_scorer.coef_[0] if val_scorer.coef_.ndim > 1 else val_scorer.coef_
            val_coefs = [
                {
                    "feature": name,
                    "coefficient": float(val_coef[i]),
                    "abs_coefficient": float(abs(val_coef[i])),
                }
                for i, name in enumerate(val_features)
                if i < len(val_coef)
            ]
            result["value_coefficients"] = val_coefs
            result["value_top_features"] = sorted(
                val_coefs, key=lambda x: x["abs_coefficient"], reverse=True
            )[:5]
        else:
            result["value_coefficients"] = []
            result["value_top_features"] = []

        return result

    # ------------------------------------------------------------------
    # Step 4b: Segment Coefficient Contribution Comparison
    # ------------------------------------------------------------------

    def analyze_segment_coefficient_contribution(
        self,
        years: list[int] | None = None,
    ) -> dict[str, Any]:
        """Compare MAWC/Ranker coefficient contributions across race groups.

        Splits horse_diff into changed/dropped/retained race groups and
        compares per-group coefficient contribution distributions.

        Args:
            years: Model years to analyze. Defaults to [2024].

        Returns:
            Dict with changed/dropped/retained group analysis.
        """
        if years is None:
            years = [2024]

        # Split races into groups
        changed_ids: set[str] = set()
        if "selected_changed" in self.race_diff.columns:
            changed_ids = set(
                self.race_diff[self.race_diff["selected_changed"] == True]["race_id"]  # noqa: E712
            )

        all_race_ids = set(self.race_diff["race_id"].unique())
        unchanged_ids = all_race_ids - changed_ids

        # Further split unchanged into dropped and retained
        bl_sel_col = f"{self.baseline_name}_selected"
        sh_sel_col = f"{self.shadow_name}_selected"

        dropped_ids: set[str] = set()
        retained_ids: set[str] = set()

        if bl_sel_col in self.horse_diff.columns and sh_sel_col in self.horse_diff.columns:
            for race_id in unchanged_ids:
                race_horses = self.horse_diff[self.horse_diff["race_id"] == race_id]
                bl_selected = race_horses[race_horses[bl_sel_col] == True]  # noqa: E712
                sh_selected = race_horses[race_horses[sh_sel_col] == True]  # noqa: E712

                if len(bl_selected) > 0 and len(sh_selected) == 0:
                    dropped_ids.add(race_id)
                else:
                    retained_ids.add(race_id)
        else:
            retained_ids = unchanged_ids

        # Compute per-group MAWC contribution (p_win delta)
        bl_p_col = f"{self.baseline_name}_p_win_final"
        sh_p_col = f"{self.shadow_name}_p_win_final"

        def _group_pwin_stats(ids: set[str]) -> dict[str, Any]:
            if not ids:
                return {"mean_p_win_delta": 0.0, "n_races": 0}
            group = self.horse_diff[self.horse_diff["race_id"].isin(ids)]
            if group.empty or bl_p_col not in group.columns or sh_p_col not in group.columns:
                return {"mean_p_win_delta": 0.0, "n_races": len(ids)}
            delta = (group[sh_p_col] - group[bl_p_col]).dropna()
            return {
                "mean_p_win_delta": float(delta.mean()) if len(delta) > 0 else 0.0,
                "std_p_win_delta": float(delta.std()) if len(delta) > 1 else 0.0,
                "n_races": len(ids),
            }

        result = {
            "changed": _group_pwin_stats(changed_ids),
            "dropped": _group_pwin_stats(dropped_ids),
            "retained": _group_pwin_stats(retained_ids),
        }

        return result

    # ------------------------------------------------------------------
    # Conditional Upstream SHAP/gain Check (D-03 clause 4)
    # ------------------------------------------------------------------

    def _check_upstream_anomaly(
        self,
        mawc_result: dict[str, Any],
        ranker_result: dict[str, Any],
    ) -> str:
        """Check if coefficient analysis detects upstream anomalies.

        Per D-03 clause 4: if analyze_mawc_coefficients() or
        analyze_ranker_coefficients() detects anomalies in the input features
        themselves (e.g., unexpectedly extreme coefficients suggesting skewed
        upstream p_win values), flag for targeted gain analysis.

        Returns:
            Description of findings or "no anomalies detected" message.
        """
        anomalies: list[str] = []

        # Check MAWC: if beta_market_contribution is extreme (> 0.95)
        beta = mawc_result.get("beta_market_contribution", 0.0)
        if beta > 0.95:
            anomalies.append(
                f"MAWC beta_market_contribution={beta:.4f} is extremely high "
                f"(>0.95), suggesting logit(p_market) dominance that may "
                f"indicate upstream ability/win_hit model producing skewed p_win"
            )

        # Check MAWC: if any single coefficient is extremely large
        top_features = mawc_result.get("top_features", [])
        for feat in top_features:
            if feat.get("abs_coefficient", 0) > 1.5:
                anomalies.append(
                    f"MAWC coefficient {feat['feature']}={feat['coefficient']:.4f} "
                    f"is extreme (|coef|>1.5), may indicate upstream anomaly"
                )

        # Check Ranker: if relevance is heavily weighted on single feature
        rel_top = ranker_result.get("relevance_top_features", [])
        if rel_top:
            max_rel_weight = rel_top[0].get("abs_coefficient", 0)
            if max_rel_weight > 1.0:
                anomalies.append(
                    f"Ranker relevance dominated by {rel_top[0]['feature']} "
                    f"(weight={max_rel_weight:.4f}>1.0), "
                    f"may indicate upstream p_win anomaly"
                )

        if anomalies:
            return (
                "Anomalies detected -- targeted gain analysis recommended for "
                "upstream models: " + "; ".join(anomalies)
            )
        return "no anomalies detected, no SHAP/gain analysis needed"

    # ------------------------------------------------------------------
    # Full Attribution Sequence
    # ------------------------------------------------------------------

    def run_full_attribution(
        self,
        years: list[int] | None = None,
    ) -> ComponentAttributionResult:
        """Execute ECE -> APR -> bet_count -> OBF sequence + coefficient analysis.

        Per D-02, strict sequential order is maintained.

        Args:
            years: Model years for coefficient analysis. Defaults to [2024].

        Returns:
            Complete ComponentAttributionResult.
        """
        if years is None:
            years = [2024]

        # Step 1: ECE attribution
        ece_result = self.attribute_ece_degradation()

        # Step 2: APR attribution
        apr_result = self.attribute_apr_deviation()

        # Step 3: Bet count + OBF attribution
        bet_count_result = self.attribute_bet_count_loss()

        # Step 4: Coefficient analysis
        mawc_result = self.analyze_mawc_coefficients(year=years[0])
        ranker_result = self.analyze_ranker_coefficients(year=years[0])
        segment_contrib = self.analyze_segment_coefficient_contribution(years=years)

        # Conditional upstream anomaly check
        upstream_check = self._check_upstream_anomaly(mawc_result, ranker_result)

        # Build recommendations from analysis
        recommendations = self._build_recommendations(
            ece_result, apr_result, bet_count_result, mawc_result, ranker_result
        )

        coef_analysis = CoefficientAnalysisResult(
            mawc_coef_analysis=mawc_result,
            ranker_coef_analysis=ranker_result,
            segment_contribution_comparison=segment_contrib,
            upstream_anomaly_check=upstream_check,
        )

        return ComponentAttributionResult(
            ece_attribution=ece_result,
            apr_attribution=apr_result,
            bet_count_attribution=bet_count_result,
            coefficient_analysis=coef_analysis,
            upstream_anomaly_check=upstream_check,
            recommendations=recommendations,
        )

    def _build_recommendations(
        self,
        ece_result: dict[str, Any],
        apr_result: dict[str, Any],
        bet_count_result: dict[str, Any],
        mawc_result: dict[str, Any],
        ranker_result: dict[str, Any],
    ) -> list[str]:
        """Build actionable recommendations for Phase 45."""
        recs: list[str] = []

        # ECE recommendation
        ece_segs = ece_result.get("segments", [])
        worst_ece_seg = None
        worst_ece_delta = 0.0
        for seg in ece_segs:
            if seg.get("delta_ece", 0) > worst_ece_delta:
                worst_ece_delta = seg["delta_ece"]
                worst_ece_seg = seg

        if worst_ece_seg and worst_ece_delta > 0.01:
            recs.append(
                f"MAWC segment fix: {worst_ece_seg['segment_name']}="
                f"{worst_ece_seg['segment_value']} has worst ECE delta "
                f"({worst_ece_delta:+.4f}). Reduce logit_market weight for "
                f"this segment."
            )

        # Bet count recommendation
        ranker_excl = bet_count_result.get("ranker_exclusion", {})
        excluded = ranker_excl.get("excluded_by_ranker", 0)
        if excluded > 0:
            recs.append(
                f"Ranker exclusion fix: {excluded} bets excluded by Ranker "
                f"in changed races. Consider relaxing investment_score threshold."
            )

        # MAWC coefficient recommendation
        beta = mawc_result.get("beta_market_contribution", 0.0)
        if beta > 0.80:
            recs.append(
                f"MAWC market dominance: beta_market_contribution={beta:.4f}. "
                f"Consider increasing L2 regularization (decreasing C) or "
                f"capping logit_market coefficient."
            )

        if not recs:
            recs.append("No specific component fix recommended")

        return recs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_col(
        df: pd.DataFrame,
        variant_name: str,
        metric: str,
    ) -> str | None:
        """Resolve variant-prefixed column name in DataFrame."""
        candidates = [
            f"{variant_name}_{metric}",
            f"shadow_{metric}",
        ]
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def _add_segment_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add segment columns (odds_band, popularity_band, probability_rank_band)."""
        bl_p_col = f"{self.baseline_name}_p_win_final"

        if "closing_win_odds" in df.columns:
            df["odds_band"] = pd.cut(
                pd.to_numeric(df["closing_win_odds"], errors="coerce"),
                bins=ODDS_BAND_EDGES,
                labels=ODDS_BAND_NAMES,
                right=True,
            ).astype(str)

        if "popularity" in df.columns:
            df["popularity_band"] = pd.cut(
                pd.to_numeric(df["popularity"], errors="coerce"),
                bins=POPULARITY_BAND_EDGES,
                labels=POPULARITY_BAND_NAMES,
                right=True,
            ).astype(str)

        if bl_p_col in df.columns:
            p_vals = pd.to_numeric(df[bl_p_col], errors="coerce")
            df["_prob_rank"] = p_vals.groupby(
                df["race_id"], observed=True
            ).rank(ascending=False, method="min")
            df["probability_rank_band"] = pd.cut(
                df["_prob_rank"],
                bins=PROB_RANK_BAND_EDGES,
                labels=PROB_RANK_BAND_NAMES,
                right=True,
            ).astype(str)
            df = df.drop(columns=["_prob_rank"])

        return df

    def _compute_ece_for_col(
        self,
        df: pd.DataFrame,
        p_col: str,
    ) -> float:
        """Compute ECE for a specific probability column."""
        if df.empty or p_col not in df.columns:
            return 0.0
        if "kakuteijyuni" not in df.columns:
            return 0.0

        p_vals = pd.to_numeric(df[p_col], errors="coerce")
        y_vals = (df["kakuteijyuni"] == 1).astype(float)

        valid = p_vals.notna() & (p_vals > 0) & (p_vals < 1)
        if valid.sum() == 0:
            return 0.0

        return float(ShadowComparisonFramework._compute_ece(
            p_vals[valid].values, y_vals[valid].values, n_bins=10
        ))

    @staticmethod
    def _compute_apr_for_col(
        df: pd.DataFrame,
        p_col: str,
    ) -> float:
        """Compute actual/predicted ratio for a probability column."""
        if df.empty or p_col not in df.columns:
            return 0.0
        if "kakuteijyuni" not in df.columns:
            return 0.0

        p_vals = pd.to_numeric(df[p_col], errors="coerce")
        y_vals = (df["kakuteijyuni"] == 1).astype(float)

        valid = p_vals.notna() & (p_vals > 0) & (p_vals < 1)
        if valid.sum() == 0:
            return 0.0

        mean_actual = float(y_vals[valid].mean())
        mean_pred = float(p_vals[valid].mean())
        return mean_actual / mean_pred if mean_pred > 0 else 0.0
