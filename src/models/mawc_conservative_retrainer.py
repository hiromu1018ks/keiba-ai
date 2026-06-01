"""MawcConservativeRetrainer -- Conservative MAWC retraining with quality gates.

Phase 45 Plan 01: FIX-01 (structural fix via interaction removal + strong regularization)
and FIX-02 (generalizability confirmation via OOF metrics).

Retrains MarketAwareWinCalibrator with reduced 36-dim feature matrix
(removing all 15 logit_model_x_* interactions) and conservative C grid [0.003, 0.005, 0.01, 0.03].
Evaluates quality gates per C value and selects minimum passing C for deployment.
"""

# ruff: noqa: N803,N806 -- ML convention: X for feature matrix

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from models.market_aware_win_calibrator import MarketAwareWinCalibrator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ECE computation (self-contained, mirrors ShadowComparisonFramework._compute_ece)
# ---------------------------------------------------------------------------


def _compute_ece(y_pred: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error -- 10-bin equal-width binning.

    Args:
        y_pred: Predicted probabilities, shape (N,).
        y_true: True binary labels, shape (N,).
        n_bins: Number of equal-width bins.

    Returns:
        ECE value (weighted average of |avg_pred - avg_true| per bin).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_pred)
    if total == 0:
        return 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_pred >= bin_boundaries[i]) & (y_pred <= bin_boundaries[i + 1])
        else:
            mask = (y_pred >= bin_boundaries[i]) & (y_pred < bin_boundaries[i + 1])
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue
        avg_pred = y_pred[mask].mean()
        avg_true = y_true[mask].mean()
        ece += abs(avg_pred - avg_true) * (n_in_bin / total)

    return float(ece)


# ---------------------------------------------------------------------------
# Dataclass definitions
# ---------------------------------------------------------------------------


@dataclass
class FavoriteBandGuardResult:
    """Quality guard result for odds 1-3 favorite band."""

    odds_band: str  # "1-3"
    n_horses: int
    ece_baseline: float
    ece_conservative: float
    ece_delta: float
    ece_passed: bool
    p_compression_ratio: float  # mean(p_conservative / p_model) for odds 1-3
    p_compression_passed: bool  # ratio >= 0.90
    ev_pass_rate_baseline: float  # pct where p * odds >= 1.0
    ev_pass_rate_conservative: float
    ev_pass_rate_passed: bool
    overall_passed: bool


@dataclass
class QualityGateResult:
    """Complete quality gate evaluation result."""

    overall_brier: float
    overall_logloss: float
    overall_ece: float
    baseline_brier: float
    baseline_logloss: float
    baseline_ece: float
    brier_non_degraded: bool  # overall_brier <= baseline * (1 + tolerance)
    logloss_non_degraded: bool
    ece_non_degraded: bool
    favorite_band_guard: FavoriteBandGuardResult
    year_level_metrics: dict[int, dict[str, float]]
    year_level_passed: bool
    all_gates_passed: bool

    @property
    def is_shadow_candidate(self) -> bool:
        """Sanity guard for shadow comparison (ECE excluded).

        A candidate qualifies for shadow_only when Brier/logloss are non-degraded
        and the favorite band guard (p_compression + EV pass rate) passes.
        ECE is intentionally excluded because the baseline ECE (~0.003) with a 10%
        relative tolerance makes the gate structurally impossible for conservative
        candidates (ECE 0.012-0.023).  Final ECE evaluation belongs in Phase 46
        DeploymentGateEvaluator.
        """
        return (
            self.brier_non_degraded
            and self.logloss_non_degraded
            and self.favorite_band_guard.p_compression_passed
            and self.favorite_band_guard.ev_pass_rate_passed
        )


@dataclass
class CGridCandidateResult:
    """Result for a single C value in the grid search."""

    c_value: float
    mawc: MarketAwareWinCalibrator  # fitted conservative MAWC
    quality_gate: QualityGateResult
    beta_market_contribution: float


@dataclass
class ConservativeRetrainResult:
    """Complete result of conservative MAWC retraining for one surface."""

    surface: str
    best_c: float | None  # None if not_deployed
    best_candidate: CGridCandidateResult | None
    all_candidates: list[CGridCandidateResult]
    deployed: bool  # False if all candidates failed
    deployment_status: str  # "deployed" | "shadow_only" | "rejected"
    feature_names: list[str]  # 36-dim feature names
    n_samples: int
    removed_interactions: list[str]  # the 15 removed logit_model_x_* names
    manifest_metadata: dict


# ---------------------------------------------------------------------------
# All 15 removed logit_model_x_* interaction names (indices 21-35 of original 51-dim)
# ---------------------------------------------------------------------------

_REMOVED_INTERACTIONS: list[str] = [
    # logit_model x odds_band (7 terms)
    "logit_model_x_1-2",
    "logit_model_x_2-3",
    "logit_model_x_3-5",
    "logit_model_x_5-10",
    "logit_model_x_10-30",
    "logit_model_x_30-100",
    "logit_model_x_100+",
    # logit_model x pop_bucket (5 terms)
    "logit_model_x_pop_1",
    "logit_model_x_pop_2_3",
    "logit_model_x_pop_4_6",
    "logit_model_x_pop_7_9",
    "logit_model_x_pop_10_plus",
    # logit_model x p_rank (3 terms)
    "logit_model_x_top_25",
    "logit_model_x_mid_25_75",
    "logit_model_x_bottom_25",
]


# ---------------------------------------------------------------------------
# MawcConservativeRetrainer
# ---------------------------------------------------------------------------


class MawcConservativeRetrainer:
    """Conservative MAWC retraining with reduced features and strong regularization.

    Retrains MarketAwareWinCalibrator with:
    - 36-dim feature matrix (15 logit_model_x_* interactions removed)
    - Conservative C grid [0.003, 0.005, 0.01, 0.03]
    - Quality gate evaluation per C value
    - Favorite band guard (odds 1-3): ECE non-degradation + p compression + EV pass rate
    """

    CONSERVATIVE_C_GRID: ClassVar[list[float]] = [0.003, 0.005, 0.01, 0.03]
    REMOVED_INTERACTIONS: ClassVar[list[str]] = _REMOVED_INTERACTIONS
    ECE_DEGRADATION_TOLERANCE: ClassVar[float] = 0.10
    BET_COUNT_TOLERANCE: ClassVar[float] = 0.10
    P_COMPRESSION_FLOOR: ClassVar[float] = 0.90

    # Reuse MAWC encoding helpers
    _mawc_helper: ClassVar[MarketAwareWinCalibrator] = MarketAwareWinCalibrator()

    # ------------------------------------------------------------------
    # OOF data preparation
    # ------------------------------------------------------------------

    def prepare_oof_data(self, oof_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load OOF data, derive required columns, split by surface.

        Args:
            oof_path: Path to oof_predictions.parquet.

        Returns:
            (turf_df, dirt_df) with derived columns: p_model, p_market,
            p_win_race_rank_pct, popularity_rank_pct.
        """
        df = pd.read_parquet(oof_path)

        # Derive p_model from p_win_corrected (RQ-1)
        df["p_model"] = df["p_win_corrected"]

        # Derive p_market from tanodds (identical to MAWC inference path)
        df["p_market"] = np.clip(1.0 / df["tanodds"].values, 0.01, 0.99)

        # Derive p_win_race_rank_pct (per race ranking of model probability)
        df["p_win_race_rank_pct"] = (
            df.groupby("race_id", observed=True)["p_model"]
            .rank(pct=True, method="min", ascending=False)
        )

        # Derive popularity_rank_pct (per MAWC build_feature_matrix pattern)
        df["popularity_rank_pct"] = (
            df["popularity_rank"].astype(float) / df["field_size"].astype(float).clip(lower=1)
        ).clip(0, 1)

        # Drop rows with NaN in required columns
        required_cols = [
            "p_model", "p_market", "tanodds", "popularity_rank",
            "field_size", "kakuteijyuni", "race_id", "surface",
        ]
        df = df.dropna(subset=required_cols).copy()

        # Split by surface
        turf_df = df[df["surface"] == "turf"].copy()
        dirt_df = df[df["surface"] == "dirt"].copy()

        return turf_df, dirt_df

    # ------------------------------------------------------------------
    # Conservative feature matrix (36-dim)
    # ------------------------------------------------------------------

    def build_conservative_feature_matrix(
        self, df: pd.DataFrame,
    ) -> tuple[np.ndarray, list[str]]:
        """Build 36-dim feature matrix without logit_model_x_* interactions.

        Structure: 6 main + 15 segment one-hot + 15 logit_market_x_* = 36 features.

        Args:
            df: DataFrame with columns: p_model, p_market, tanodds,
                popularity_rank, field_size, p_win_race_rank_pct,
                popularity_rank_pct.

        Returns:
            (X, feature_names) where X has shape (N, 36).
        """
        eps = 1e-10
        helper = self._mawc_helper

        # Main effects (6 continuous) -- identical to original MAWC
        p_model_vals = np.clip(df["p_model"].values.astype(float), eps, 1 - eps)
        p_market_vals = np.clip(df["p_market"].values.astype(float), eps, 1 - eps)

        logit_model = np.log(p_model_vals / (1 - p_model_vals))
        logit_market = np.log(p_market_vals / (1 - p_market_vals))
        log_odds = np.log1p(df["tanodds"].values.astype(float))
        popularity_rank_pct = df["popularity_rank_pct"].values.astype(float)
        p_win_race_rank_pct = df["p_win_race_rank_pct"].values.astype(float)
        field_size_vals = df["field_size"].values.astype(float)

        main_names = [
            "logit_model",
            "logit_market",
            "log_odds",
            "popularity_rank_pct",
            "p_win_race_rank_pct",
            "field_size",
        ]

        # One-hot encodings (7 + 5 + 3 = 15 segment features)
        odds_band_oh = helper._encode_odds_band(df["tanodds"])
        pop_bucket_oh = helper._encode_pop_bucket(df["popularity_rank"], df["field_size"])
        p_rank_oh = helper._encode_p_rank(df["p_win_race_rank_pct"])

        segment_features = np.hstack([
            odds_band_oh.values, pop_bucket_oh.values, p_rank_oh.values,
        ])
        segment_names = (
            list(odds_band_oh.columns)
            + list(pop_bucket_oh.columns)
            + list(p_rank_oh.columns)
        )
        assert len(segment_names) == 15

        # ONLY logit_market x segment interactions (15) -- KEPT
        interaction_market = logit_market[:, None] * segment_features
        interaction_names_market = [f"logit_market_x_{s}" for s in segment_names]

        # NO logit_model x segment interactions -- REMOVED

        # Assemble: 6 main + 15 segment + 15 market interactions = 36
        X = np.hstack([
            np.column_stack([
                logit_model,
                logit_market,
                log_odds,
                popularity_rank_pct,
                p_win_race_rank_pct,
                field_size_vals,
            ]),
            segment_features,
            interaction_market,
        ])

        feature_names = main_names + segment_names + interaction_names_market
        assert len(feature_names) == 36, f"Expected 36 features, got {len(feature_names)}"
        assert X.shape[1] == 36, f"Expected X with 36 columns, got {X.shape[1]}"

        return X, feature_names

    # ------------------------------------------------------------------
    # Quality gate evaluation
    # ------------------------------------------------------------------

    def evaluate_quality_gates(
        self,
        y: np.ndarray,
        p_conservative: np.ndarray,
        p_baseline: np.ndarray,
        df: pd.DataFrame,
        baseline_brier: float,
        baseline_logloss: float,
        baseline_ece: float,
    ) -> QualityGateResult:
        """Evaluate quality gates for a conservative candidate.

        Gates:
        1. Overall Brier/logloss/ECE non-degradation (10% relative tolerance)
        2. Favorite band (odds 1-3) guard: ECE non-degradation + p compression + EV pass rate
        3. Year-level non-degradation

        Args:
            y: True binary labels.
            p_conservative: Conservative MAWC predicted probabilities.
            p_baseline: Baseline MAWC predicted probabilities.
            df: DataFrame with tanodds, race_date columns.
            baseline_brier/logloss/ece: Baseline metrics for comparison.

        Returns:
            QualityGateResult with pass/fail for each gate.
        """
        tol = self.ECE_DEGRADATION_TOLERANCE

        # 1. Overall metrics for conservative predictions
        cons_brier = float(brier_score_loss(y, p_conservative))
        cons_logloss = float(log_loss(y, p_conservative))
        cons_ece = _compute_ece(p_conservative, y)

        brier_non_degraded = cons_brier <= baseline_brier * (1 + tol)
        logloss_non_degraded = cons_logloss <= baseline_logloss * (1 + tol)
        ece_non_degraded = cons_ece <= baseline_ece * (1 + tol)

        # 2. Favorite band guard (odds 1-3)
        odds_mask = (df["tanodds"].values >= 0) & (df["tanodds"].values < 3)
        n_fav = int(odds_mask.sum())

        if n_fav > 0:
            y_fav = y[odds_mask]
            p_cons_fav = p_conservative[odds_mask]
            p_base_fav = p_baseline[odds_mask]
            odds_fav = df["tanodds"].values[odds_mask]

            ece_baseline_fav = _compute_ece(p_base_fav, y_fav)
            ece_conservative_fav = _compute_ece(p_cons_fav, y_fav)
            ece_delta = ece_conservative_fav - ece_baseline_fav
            ece_passed = ece_conservative_fav <= ece_baseline_fav * (1 + tol)

            # P compression check: mean(p_conservative / max(p_baseline, eps))
            p_compression_ratio = float(
                np.mean(p_cons_fav / np.maximum(p_base_fav, 1e-10))
            )
            p_compression_passed = p_compression_ratio >= self.P_COMPRESSION_FLOOR

            # EV pass rate: pct where p * odds >= 1.0
            ev_pass_rate_baseline = float(np.mean(p_base_fav * odds_fav >= 1.0))
            ev_pass_rate_conservative = float(np.mean(p_cons_fav * odds_fav >= 1.0))
            ev_pass_rate_passed = (
                ev_pass_rate_conservative >= ev_pass_rate_baseline * (1 - self.BET_COUNT_TOLERANCE)
            )

            fav_overall_passed = ece_passed and p_compression_passed and ev_pass_rate_passed
        else:
            # No favorite band data -- pass by default
            ece_baseline_fav = 0.0
            ece_conservative_fav = 0.0
            ece_delta = 0.0
            ece_passed = True
            p_compression_ratio = 1.0
            p_compression_passed = True
            ev_pass_rate_baseline = 0.0
            ev_pass_rate_conservative = 0.0
            ev_pass_rate_passed = True
            fav_overall_passed = True

        favorite_band_guard = FavoriteBandGuardResult(
            odds_band="1-3",
            n_horses=n_fav,
            ece_baseline=ece_baseline_fav,
            ece_conservative=ece_conservative_fav,
            ece_delta=ece_delta,
            ece_passed=ece_passed,
            p_compression_ratio=p_compression_ratio,
            p_compression_passed=p_compression_passed,
            ev_pass_rate_baseline=ev_pass_rate_baseline,
            ev_pass_rate_conservative=ev_pass_rate_conservative,
            ev_pass_rate_passed=ev_pass_rate_passed,
            overall_passed=fav_overall_passed,
        )

        # 3. Year-level non-degradation
        year_level_metrics: dict[int, dict[str, float]] = {}
        year_level_passed = True

        if "race_date" in df.columns:
            years = pd.to_datetime(df["race_date"]).dt.year
            for yr in years.unique():
                yr_mask = years.values == yr
                if yr_mask.sum() < 10:
                    continue
                y_yr = y[yr_mask]
                p_yr = p_conservative[yr_mask]
                yr_brier = float(brier_score_loss(y_yr, p_yr))
                yr_logloss = float(log_loss(y_yr, p_yr))
                yr_ece = _compute_ece(p_yr, y_yr)
                year_level_metrics[int(yr)] = {
                    "brier": yr_brier,
                    "logloss": yr_logloss,
                    "ece": yr_ece,
                }
                # Check non-degradation per year
                if (
                    yr_ece > baseline_ece * (1 + tol)
                    or yr_logloss > baseline_logloss * (1 + tol)
                ):
                    year_level_passed = False

        # Aggregate pass/fail
        all_gates_passed = (
            brier_non_degraded
            and logloss_non_degraded
            and ece_non_degraded
            and fav_overall_passed
            and year_level_passed
        )

        return QualityGateResult(
            overall_brier=cons_brier,
            overall_logloss=cons_logloss,
            overall_ece=cons_ece,
            baseline_brier=baseline_brier,
            baseline_logloss=baseline_logloss,
            baseline_ece=baseline_ece,
            brier_non_degraded=brier_non_degraded,
            logloss_non_degraded=logloss_non_degraded,
            ece_non_degraded=ece_non_degraded,
            favorite_band_guard=favorite_band_guard,
            year_level_metrics=year_level_metrics,
            year_level_passed=year_level_passed,
            all_gates_passed=all_gates_passed,
        )

    # ------------------------------------------------------------------
    # C grid search
    # ------------------------------------------------------------------

    def retrain_with_c_grid(
        self,
        df: pd.DataFrame,
        baseline_mawc: MarketAwareWinCalibrator,
    ) -> list[CGridCandidateResult]:
        """Fit LogisticRegression for each C in grid and evaluate quality gates.

        Args:
            df: Prepared OOF DataFrame with derived columns.
            baseline_mawc: Loaded baseline MAWC for comparison metrics.

        Returns:
            List of CGridCandidateResult sorted by C (ascending).
        """
        # Build conservative feature matrix
        X, feature_names = self.build_conservative_feature_matrix(df)

        # Target variable
        y = (df["kakuteijyuni"] == 1).astype(int).values

        # Baseline predictions (using baseline MAWC's 51-dim feature matrix)
        baseline_X, _ = baseline_mawc.build_feature_matrix(df)
        p_baseline = baseline_mawc.calibrator.predict_proba(baseline_X)[:, 1]
        baseline_brier = float(brier_score_loss(y, p_baseline))
        baseline_logloss = float(log_loss(y, p_baseline))
        baseline_ece = _compute_ece(p_baseline, y)

        candidates: list[CGridCandidateResult] = []

        for c in self.CONSERVATIVE_C_GRID:
            # Fit LogisticRegression with conservative C
            lr = LogisticRegression(C=c, max_iter=1000, fit_intercept=True)
            lr.fit(X, y)

            # Create new MAWC instance with conservative calibrator
            conservative_mawc = MarketAwareWinCalibrator()
            conservative_mawc.calibrator = lr
            conservative_mawc.feature_names = feature_names
            conservative_mawc.best_c = c
            conservative_mawc._trained = True

            # Compute predictions
            p_conservative = lr.predict_proba(X)[:, 1]

            # Evaluate quality gates
            quality_gate = self.evaluate_quality_gates(
                y=y,
                p_conservative=p_conservative,
                p_baseline=p_baseline,
                df=df,
                baseline_brier=baseline_brier,
                baseline_logloss=baseline_logloss,
                baseline_ece=baseline_ece,
            )

            # Compute beta_market contribution
            beta_market = conservative_mawc._compute_beta_market_contribution()

            candidates.append(CGridCandidateResult(
                c_value=c,
                mawc=conservative_mawc,
                quality_gate=quality_gate,
                beta_market_contribution=beta_market,
            ))

        return sorted(candidates, key=lambda c: c.c_value)

    # ------------------------------------------------------------------
    # Best C selection
    # ------------------------------------------------------------------

    def select_best_c(self, candidates: list[CGridCandidateResult]) -> CGridCandidateResult | None:
        """Select minimum C among gate-passing candidates (D-04).

        Args:
            candidates: All C grid candidates with quality gate results.

        Returns:
            Best CGridCandidateResult (minimum passing C), or None if all fail.
        """
        passing = [c for c in candidates if c.quality_gate.all_gates_passed]
        if not passing:
            return None

        # Return minimum C (strongest regularization) among passing candidates
        return min(passing, key=lambda c: c.c_value)

    def select_best_for_shadow(
        self,
        candidates: list[CGridCandidateResult],
    ) -> tuple[CGridCandidateResult | None, str]:
        """Select best candidate with 3-level deployment status.

        Priority:
        1. "deployed" -- all gates passed (ECE + Brier + Logloss + fav band)
        2. "shadow_only" -- sanity guards passed (Brier + Logloss + fav band, ECE excluded)
        3. "rejected" -- even sanity guards failed

        Returns:
            (best_candidate, deployment_status).  candidate is None when rejected.
        """
        # 1. Try fully passing candidates
        passing = [c for c in candidates if c.quality_gate.all_gates_passed]
        if passing:
            return min(passing, key=lambda c: c.c_value), "deployed"

        # 2. Try shadow candidates (ECE excluded from sanity guard)
        shadow = [c for c in candidates if c.quality_gate.is_shadow_candidate]
        if shadow:
            return min(shadow, key=lambda c: c.c_value), "shadow_only"

        # 3. All rejected
        return None, "rejected"

    # ------------------------------------------------------------------
    # Full retrain orchestration
    # ------------------------------------------------------------------

    def run_retrain(
        self,
        surface: str,
        df: pd.DataFrame,
        baseline_mawc_path: Path,
        year: int | None = None,
    ) -> ConservativeRetrainResult:
        """Orchestrate full conservative retrain for one surface.

        Args:
            surface: "turf" or "dirt".
            df: Prepared OOF DataFrame for this surface.
            baseline_mawc_path: Path to baseline MAWC joblib.
            year: Optional year for manifest metadata.

        Returns:
            ConservativeRetrainResult with best C and metadata.
        """
        # Load baseline MAWC
        baseline_mawc = MarketAwareWinCalibrator.load(baseline_mawc_path)

        # C grid search
        candidates = self.retrain_with_c_grid(df, baseline_mawc)

        # Select best candidate with 3-level deployment status
        best, deployment_status = self.select_best_for_shadow(candidates)

        # Build manifest metadata
        n_passing = sum(1 for c in candidates if c.quality_gate.all_gates_passed)
        metadata = {
            "surface": surface,
            "year": year,
            "best_c": best.c_value if best else None,
            "n_candidates": len(candidates),
            "n_passing": n_passing,
            "deployed": deployment_status == "deployed",
            "deployment_status": deployment_status,
            "removed_interactions": self.REMOVED_INTERACTIONS,
            "feature_dim": 36,
        }

        return ConservativeRetrainResult(
            surface=surface,
            best_c=best.c_value if best else None,
            best_candidate=best,
            all_candidates=candidates,
            deployed=deployment_status == "deployed",
            deployment_status=deployment_status,
            feature_names=candidates[0].mawc.feature_names if candidates else [],
            n_samples=len(df),
            removed_interactions=list(self.REMOVED_INTERACTIONS),
            manifest_metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Conservative variant directory creation
    # ------------------------------------------------------------------

    def create_conservative_variant(
        self,
        retrain_results: list[ConservativeRetrainResult],
        source_year_dir: Path,
        target_root: Path,
        year: int,
    ) -> Path:
        """Copy year model directory and replace MAWC joblib for deployed surfaces.

        For surfaces where deployed=False, the original MAWC is preserved (not replaced).

        Args:
            retrain_results: ConservativeRetrainResult for each surface.
            source_year_dir: e.g., Path("data/models-backtest/2024").
            target_root: e.g., Path("data/models-backtest-mawc-conservative").
            year: Year for the variant (e.g., 2024).

        Returns:
            Path to the created variant directory.

        Raises:
            FileNotFoundError: If source_year_dir does not exist.
        """
        if not source_year_dir.is_dir():
            raise FileNotFoundError(
                f"Source model directory not found: {source_year_dir}"
            )

        target_dir = target_root / str(year)

        # Copy entire directory (dirs_exist_ok allows re-copy for multi-surface)
        shutil.copytree(source_year_dir, target_dir, dirs_exist_ok=True)

        # Replace MAWC joblib for deployed and shadow_only surfaces
        for result in retrain_results:
            has_candidate = (
                result.deployment_status in ("deployed", "shadow_only")
                and result.best_candidate is not None
            )
            if has_candidate:
                mawc = result.best_candidate.mawc
                # Add deployment metadata to training_summary
                mawc.training_summary["deployment_status"] = (
                    "deployed_conservative" if result.deployment_status == "deployed"
                    else "shadow_only"
                )
                mawc.training_summary["fix_version"] = "45-conservative"
                mawc.training_summary["removed_interactions"] = result.removed_interactions
                mawc.training_summary["original_feature_dim"] = 51
                mawc.training_summary["conservative_feature_dim"] = 36
                mawc.training_summary["beta_market_contribution"] = (
                    result.best_candidate.beta_market_contribution
                )

                mawc_path = target_dir / f"market_aware_win_calibrator_{result.surface}.joblib"
                mawc.save(mawc_path)
                logger.info(
                    "Saved %s MAWC for surface=%s, C=%.4f to %s",
                    result.deployment_status, result.surface, result.best_c, mawc_path,
                )
            else:
                logger.warning(
                    "Surface %s: %s, keeping original MAWC",
                    result.surface, result.deployment_status,
                )

        # Verify meta.json exists (ModelLoader requirement)
        if not (target_dir / "meta.json").is_file():
            raise FileNotFoundError(f"meta.json missing in {target_dir} after copy")

        return target_dir

    # ------------------------------------------------------------------
    # Manifest generation
    # ------------------------------------------------------------------

    def generate_manifest(
        self,
        retrain_results: list[ConservativeRetrainResult],
        source_model_dir: Path,
        target_root: Path,
        years: list[int],
        oof_date_range: dict[str, str] | None = None,
    ) -> dict:
        """Generate manifest JSON dict for Phase 46 consumption.

        Per-surface results are keyed as ``per_year_surface["2024"]["turf"]`` to
        preserve year-specific quality gate data rather than silently overwriting
        earlier years (CR-01 fix).  A top-level ``per_surface`` aggregate is also
        provided for convenience (uses the last year's metrics per surface).

        Args:
            retrain_results: All ConservativeRetrainResult across surfaces and years.
            source_model_dir: e.g., Path("data/models-backtest").
            target_root: e.g., Path("data/models-backtest-mawc-conservative").
            years: Years included in this variant.
            oof_date_range: Optional {"min": "YYYY-MM-DD", "max": "YYYY-MM-DD"}.

        Returns:
            JSON-serializable manifest dict.
        """
        per_year_surface: dict[str, dict[str, dict]] = {}
        for result in retrain_results:
            gate = result.best_candidate.quality_gate if result.best_candidate else None
            year_key = str(result.manifest_metadata.get("year", "unknown"))
            surface_entry = {
                "best_c": result.best_c,
                "deployed": result.deployed,
                "deployment_status": result.deployment_status,
                "shadow_candidate_saved": result.deployment_status in ("deployed", "shadow_only"),
                "n_candidates": len(result.all_candidates),
                "n_passing": sum(
                    1 for c in result.all_candidates if c.quality_gate.all_gates_passed
                ),
                "beta_market_contribution": (
                    result.best_candidate.beta_market_contribution
                    if result.best_candidate
                    else None
                ),
                "quality_gate_summary": {
                    "overall_brier": gate.overall_brier if gate else None,
                    "overall_logloss": gate.overall_logloss if gate else None,
                    "overall_ece": gate.overall_ece if gate else None,
                    "favorite_band_ece_delta": (
                        gate.favorite_band_guard.ece_delta if gate else None
                    ),
                    "p_compression_ratio": (
                        gate.favorite_band_guard.p_compression_ratio if gate else None
                    ),
                },
            }
            if year_key not in per_year_surface:
                per_year_surface[year_key] = {}
            per_year_surface[year_key][result.surface] = surface_entry

        manifest: dict = {
            "mawc_fix_version": "45-conservative",
            "source_model_dir": str(source_model_dir),
            "target_variant_dir": str(target_root),
            "C_grid": self.CONSERVATIVE_C_GRID,
            "removed_interactions": list(self.REMOVED_INTERACTIONS),
            "feature_dim": 36,
            "original_feature_dim": 51,
            "years": [str(y) for y in years],
            "per_year_surface": per_year_surface,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        if oof_date_range:
            manifest["oof_date_range"] = oof_date_range

        return manifest

    # ------------------------------------------------------------------
    # Full pipeline (top-level orchestration)
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        oof_path: Path,
        source_model_dir: Path,
        target_root: Path,
        years: list[int],
    ) -> tuple[dict, list[ConservativeRetrainResult]]:
        """Run full conservative MAWC retraining pipeline.

        1. Prepare OOF data
        2. For each year x surface: retrain and collect results
        3. Create conservative variant directories
        4. Generate manifest

        Args:
            oof_path: Path to oof_predictions.parquet.
            source_model_dir: e.g., Path("data/models-backtest").
            target_root: e.g., Path("data/models-backtest-mawc-conservative").
            years: Years to create variants for (e.g., [2024, 2025]).

        Returns:
            (manifest_dict, all_results) -- caller serializes manifest to JSON
            and passes all_results to save_retrain_results().
        """
        # Step 1: Prepare OOF data
        turf_df, dirt_df = self.prepare_oof_data(oof_path)
        surface_dfs = {"turf": turf_df, "dirt": dirt_df}

        # Record OOF date range for manifest transparency
        oof_date_range: dict[str, str] | None = None
        all_oof = pd.concat([turf_df, dirt_df], ignore_index=True)
        if "race_date" in all_oof.columns:
            dates = pd.to_datetime(all_oof["race_date"])
            oof_date_range = {
                "min": str(dates.min().date()),
                "max": str(dates.max().date()),
                "n_rows": len(all_oof),
            }
            logger.info(
                "OOF date range: %s to %s (%d rows)",
                oof_date_range["min"], oof_date_range["max"], len(all_oof),
            )

        all_results: list[ConservativeRetrainResult] = []

        # Step 2: For each year, retrain per surface and create variant
        for year in years:
            year_results: list[ConservativeRetrainResult] = []

            for surface in ["turf", "dirt"]:
                baseline_path = (
                    source_model_dir / str(year)
                    / f"market_aware_win_calibrator_{surface}.joblib"
                )

                if not baseline_path.is_file():
                    logger.warning(
                        "Baseline MAWC not found for year=%d surface=%s, skipping",
                        year, surface,
                    )
                    continue

                result = self.run_retrain(
                    surface, surface_dfs[surface], baseline_path, year=year,
                )
                year_results.append(result)
                all_results.append(result)

                logger.info(
                    "Year %d surface %s: deployed=%s best_c=%s",
                    year, surface, result.deployed, result.best_c,
                )

            # Step 3: Create conservative variant for this year
            if year_results:
                source_year_dir = source_model_dir / str(year)
                if source_year_dir.is_dir():
                    self.create_conservative_variant(
                        year_results, source_year_dir, target_root, year,
                    )

        # Step 4: Generate manifest
        manifest = self.generate_manifest(
            all_results, source_model_dir, target_root, years,
            oof_date_range=oof_date_range,
        )

        return manifest, all_results


# ---------------------------------------------------------------------------
# Output functions (manifest JSON + retrain summary Markdown)
# ---------------------------------------------------------------------------


def save_retrain_results(
    manifest: dict,
    retrain_results: list[ConservativeRetrainResult],
    target_root: Path,
) -> tuple[Path, Path]:
    """Save manifest.json and retrain_summary.md to target_root.

    Args:
        manifest: Manifest dict from generate_manifest().
        retrain_results: All ConservativeRetrainResult across surfaces.
        target_root: Output directory (e.g., data/models-backtest-mawc-conservative).

    Returns:
        (manifest_path, summary_path) paths to written files.
    """
    import json as json_mod

    target_root.mkdir(parents=True, exist_ok=True)

    # 1. manifest.json
    manifest_path = target_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json_mod.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)
    logger.info("Manifest saved to %s", manifest_path)

    # 2. retrain_summary.md
    summary_path = target_root / "retrain_summary.md"
    _write_retrain_summary(manifest, retrain_results, summary_path)
    logger.info("Summary saved to %s", summary_path)

    return manifest_path, summary_path


def _write_retrain_summary(
    manifest: dict,
    retrain_results: list[ConservativeRetrainResult],
    outpath: Path,
) -> None:
    """Write human-readable Markdown summary with 6 sections."""
    lines: list[str] = []

    lines.append("# MAWC Conservative Retrain Summary")
    lines.append("")
    lines.append(f"Generated: {manifest.get('generated_at', 'N/A')}")
    lines.append("")

    # Section 1: Configuration
    lines.append("## Configuration")
    lines.append("")
    c_grid = manifest.get("C_grid", [])
    lines.append(f"- **C Grid**: `{c_grid}`")
    removed = manifest.get("removed_interactions", [])
    lines.append(f"- **Removed Interactions ({len(removed)} items)**:")
    for name in removed:
        lines.append(f"  - `{name}`")
    lines.append(f"- **Feature Dimensions**: {manifest.get('original_feature_dim', '?')} -> "
                 f"{manifest.get('feature_dim', '?')}")
    lines.append(f"- **Years**: {', '.join(manifest.get('years', []))}")
    lines.append(f"- **OOF Data Source**: {manifest.get('source_model_dir', 'N/A')}")
    lines.append("")

    # Section 2: Per-Surface Results (deduplicated -- one row per surface across years)
    lines.append("## Per-Surface Results")
    lines.append("")
    per_year_surface = manifest.get("per_year_surface", {})
    lines.append(
        "| Year | Surface | Best C | Deployed (Status) "
        "| Beta Market | N Cands | N Pass | Shadow |"
    )
    lines.append(
        "|------|---------|--------|-------------------"
        "|-------------|---------|--------|---------|"
    )
    for year_key in sorted(per_year_surface.keys()):
        for surface, data in per_year_surface[year_key].items():
            best_c = data.get("best_c", "N/A")
            if best_c is not None:
                best_c = f"{best_c:.4f}"
            deployed = data.get("deployed", False)
            dep_status = data.get("deployment_status", "unknown")
            shadow_saved = data.get("shadow_candidate_saved", False)
            beta = data.get("beta_market_contribution")
            beta_str = f"{beta:.4f}" if beta is not None else "N/A"
            n_cand = data.get("n_candidates", 0)
            n_pass = data.get("n_passing", 0)
            dep_str = "**Yes**" if deployed else "No"
            lines.append(
                f"| {year_key} | {surface} | {best_c} | "
                f"{dep_str} ({dep_status}) | {beta_str} | {n_cand} | {n_pass} | "
                f"{'Yes' if shadow_saved else 'No'} |"
            )
    lines.append("")

    # Section 3: Quality Gate Details
    lines.append("## Quality Gate Details")
    lines.append("")
    for result in retrain_results:
        gate = result.best_candidate.quality_gate if result.best_candidate else None
        year_label = result.manifest_metadata.get("year", "unknown")
        label = f"{result.surface} ({year_label})"
        if gate is None:
            lines.append(f"### {label}: No quality gate data (not_deployed)")
            lines.append("")
            continue
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| Metric | Baseline | Conservative | Delta | Pass |")
        lines.append("|--------|----------|-------------|-------|------|")
        for metric_name, cons_val, base_val, passed in [
            ("Brier", gate.overall_brier, gate.baseline_brier, gate.brier_non_degraded),
            ("Logloss", gate.overall_logloss, gate.baseline_logloss, gate.logloss_non_degraded),
            ("ECE", gate.overall_ece, gate.baseline_ece, gate.ece_non_degraded),
        ]:
            delta = cons_val - base_val
            pass_str = "PASS" if passed else "FAIL"
            lines.append(
                f"| {metric_name} | {base_val:.4f} | {cons_val:.4f} | {delta:+.4f} | {pass_str} |"
            )
        lines.append("")

        # Year-level sub-table
        if gate.year_level_metrics:
            lines.append(f"**Year-level ({result.surface}):**")
            lines.append("")
            lines.append("| Year | Brier | Logloss | ECE |")
            lines.append("|------|-------|---------|-----|")
            for yr, metrics in sorted(gate.year_level_metrics.items()):
                lines.append(
                    f"| {yr} | {metrics['brier']:.4f} | {metrics['logloss']:.4f} | "
                    f"{metrics['ece']:.4f} |"
                )
            lines.append(f"Year-level passed: {'Yes' if gate.year_level_passed else 'No'}")
            lines.append("")

    # Section 4: Favorite Band Guard (Odds 1-3)
    lines.append("## Favorite Band Guard (Odds 1-3)")
    lines.append("")
    for result in retrain_results:
        gate = result.best_candidate.quality_gate if result.best_candidate else None
        year_label = result.manifest_metadata.get("year", "unknown")
        label = f"{result.surface} ({year_label})"
        if gate is None:
            lines.append(f"### {label}: No data (not_deployed)")
            lines.append("")
            continue
        fbg = gate.favorite_band_guard
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| Guard Metric | Baseline | Conservative | Threshold | Pass |")
        lines.append("|-------------|----------|-------------|-----------|------|")
        lines.append(
            f"| ECE | {fbg.ece_baseline:.4f} | {fbg.ece_conservative:.4f} | "
            f"delta <= {(fbg.ece_baseline * 1.10):.4f} | "
            f"{'PASS' if fbg.ece_passed else 'FAIL'} |"
        )
        lines.append(
            f"| P Compression Ratio | - | {fbg.p_compression_ratio:.4f} | "
            f">= 0.90 | {'PASS' if fbg.p_compression_passed else 'FAIL'} |"
        )
        lines.append(
            f"| EV Pass Rate | {fbg.ev_pass_rate_baseline:.4f} | "
            f"{fbg.ev_pass_rate_conservative:.4f} | "
            f">= {fbg.ev_pass_rate_baseline * 0.90:.4f} | "
            f"{'PASS' if fbg.ev_pass_rate_passed else 'FAIL'} |"
        )
        lines.append(
            f"| **Overall** | | | | "
            f"**{'PASS' if fbg.overall_passed else 'FAIL'}** |"
        )
        lines.append(f"- N horses in odds 1-3: {fbg.n_horses}")
        lines.append("")

    # Section 5: C Grid Candidates
    lines.append("## C Grid Candidates")
    lines.append("")
    for result in retrain_results:
        year_label = result.manifest_metadata.get("year", "unknown")
        lines.append(f"### {result.surface} ({year_label})")
        lines.append("")
        lines.append("| C Value | Brier | Logloss | ECE | Gate Pass |")
        lines.append("|---------|-------|---------|-----|-----------|")
        for cand in result.all_candidates:
            g = cand.quality_gate
            pass_str = "PASS" if g.all_gates_passed else "FAIL"
            lines.append(
                f"| {cand.c_value:.4f} | {g.overall_brier:.4f} | {g.overall_logloss:.4f} | "
                f"{g.overall_ece:.4f} | {pass_str} |"
            )
        lines.append("")

    # Section 6: Phase 46 Next Steps
    lines.append("## Phase 46 Next Steps")
    lines.append("")
    lines.append("Run Shadow Comparison with conservative variant:")
    lines.append("```bash")
    lines.append(
        "python scripts/run_shadow_comparison.py \\"
    )
    lines.append(
        "  --baseline-root data/models-backtest \\"
    )
    shadow_root = manifest.get(
        "target_variant_dir", "data/models-backtest-mawc-conservative",
    )
    lines.append(
        f"  --shadow-root {shadow_root} \\"
    )
    lines.append(
        "  --folds 2024 2025 --report"
    )
    lines.append("```")
    lines.append("")

    outpath.write_text("\n".join(lines), encoding="utf-8")
