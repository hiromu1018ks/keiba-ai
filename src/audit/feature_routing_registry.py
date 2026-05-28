"""Feature Routing Audit Registry — single source of truth for SAF-01.

Defines forbidden feature sets (calibrator 50, ranker 28), critical and
advisory target models, and a run_feature_audit() function that dynamically
imports each model and checks for forbidden feature intersections.

The registry is the shared core consumed by both CI tests (fail-fast) and
the audit CLI script (JSON + Markdown reports).

Per RESEARCH Pitfall 3: FORBIDDEN_CALIBRATOR_FEATURES contains the DERIVED
output features from MarketAwareWinCalibrator.build_feature_matrix(), NOT the
raw inputs. ``field_size`` is excluded from the forbidden set because it is a
raw input passed through with the same column name and is legitimately used by
other models (MarketModel, RaceQualityScreener). Total forbidden = 50 (not 51).
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

REGISTRY_VERSION: str = "1.0"

# ---------------------------------------------------------------------------
# AuditTarget — describes a model to audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditTarget:
    """A model class to audit for forbidden feature intersections."""

    model_class_name: str
    model_module: str
    feature_cols_attr: str  # e.g. "FEATURE_COLS", "RELEVANCE_FEATURES"


# ---------------------------------------------------------------------------
# FORBIDDEN_CALIBRATOR_FEATURES — 50 derived output features from
# MAWC.build_feature_matrix(). Excludes ``field_size`` which is a raw input
# legitimately shared by other models (Pitfall 3).
# ---------------------------------------------------------------------------

# 5 derived main effects (field_size excluded — raw input)
_CALIBRATOR_MAIN: frozenset[str] = frozenset({
    "logit_model",
    "logit_market",
    "log_odds",
    "popularity_rank_pct",
    "p_win_race_rank_pct",
})

# 7 odds band one-hot
_CALIBRATOR_ODDS_BAND: frozenset[str] = frozenset({
    "1-2", "2-3", "3-5", "5-10", "10-30", "30-100", "100+",
})

# 5 popularity bucket one-hot
_CALIBRATOR_POP_BUCKET: frozenset[str] = frozenset({
    "pop_1", "pop_2_3", "pop_4_6", "pop_7_9", "pop_10_plus",
})

# 3 p_rank one-hot
_CALIBRATOR_P_RANK: frozenset[str] = frozenset({
    "top_25", "mid_25_75", "bottom_25",
})

# Segment names for interactions (7 + 5 + 3 = 15)
_SEGMENT_NAMES: tuple[str, ...] = (
    "1-2", "2-3", "3-5", "5-10", "10-30", "30-100", "100+",
    "pop_1", "pop_2_3", "pop_4_6", "pop_7_9", "pop_10_plus",
    "top_25", "mid_25_75", "bottom_25",
)

# 15 logit_model x segment interactions
_CALIBRATOR_INTERACTIONS_MODEL: frozenset[str] = frozenset(
    f"logit_model_x_{s}" for s in _SEGMENT_NAMES
)

# 15 logit_market x segment interactions
_CALIBRATOR_INTERACTIONS_MARKET: frozenset[str] = frozenset(
    f"logit_market_x_{s}" for s in _SEGMENT_NAMES
)

FORBIDDEN_CALIBRATOR_FEATURES: frozenset[str] = (
    _CALIBRATOR_MAIN
    | _CALIBRATOR_ODDS_BAND
    | _CALIBRATOR_POP_BUCKET
    | _CALIBRATOR_P_RANK
    | _CALIBRATOR_INTERACTIONS_MODEL
    | _CALIBRATOR_INTERACTIONS_MARKET
)

# Raw input features excluded from the forbidden set (Pitfall 3).
# These are passed through build_feature_matrix() with the same column name
# and are legitimately used by other models.
CALIBRATOR_EXCLUDED_RAW_INPUTS: frozenset[str] = frozenset({"field_size"})

# ---------------------------------------------------------------------------
# FORBIDDEN_RANKER_FEATURES — union of RLR RELEVANCE + VALUE + DERIVED_VALUE
# 15 + 13 + 2 = 30 listed, but if_p_win_race_rank and if_n_horses appear
# in both RELEVANCE and VALUE, so 28 unique features.
# ---------------------------------------------------------------------------

_RANKER_RELEVANCE: frozenset[str] = frozenset({
    "if_p_win_final",
    "if_p_win_race_rank",
    "if_p_ability_win",
    "rel_p_ability_win_rank",
    "if_norm_finish_avg",
    "if_closing_index",
    "if_weighted_recent_form",
    "if_jockey_wr",
    "if_trainer_wr",
    "if_blood_surface_wr",
    "if_class_level",
    "if_surface",
    "if_distance_bin",
    "if_grade_code",
    "if_n_horses",
})

_RANKER_VALUE: frozenset[str] = frozenset({
    "if_logit_gap",
    "if_edge_win",
    "if_ev_calibrated",
    "if_odds_log",
    "if_odds_band_id",
    "if_odds_drop_60_10",
    "if_odds_drop_30_10",
    "if_overround",
    "if_market_entropy",
    "if_conformal_width",
    "if_ev_uncertainty_ratio",
    "if_p_win_race_rank",
    "if_n_horses",
})

_RANKER_DERIVED: frozenset[str] = frozenset({
    "if_odds_rank",
    "if_abs_logit_gap",
})

FORBIDDEN_RANKER_FEATURES: frozenset[str] = (
    _RANKER_RELEVANCE | _RANKER_VALUE | _RANKER_DERIVED
)

# ---------------------------------------------------------------------------
# CRITICAL_TARGET_MODELS — fail-fast (CI must pass)
# ---------------------------------------------------------------------------

CRITICAL_TARGET_MODELS: tuple[AuditTarget, ...] = (
    AuditTarget("MarketModel", "models.market_model", "FEATURE_COLS"),
    AuditTarget("RaceQualityScreener", "models.race_quality_screener", "FEATURE_COLS"),
)

# ---------------------------------------------------------------------------
# ADVISORY_TARGET_MODELS — warning/report only (not fail-fast)
# ---------------------------------------------------------------------------

ADVISORY_TARGET_MODELS: tuple[AuditTarget, ...] = (
    AuditTarget("EVCorrectionModel", "models.ev_correction_model", "FEATURE_COLS"),
    AuditTarget("PlaceEVCorrectionModel", "models.ev_correction_model", "FEATURE_COLS"),
    AuditTarget("ConformalEVModel", "models.conformal_ev_model", "FEATURE_COLS"),
    AuditTarget("RegimeDetector", "models.regime_detector", "FEATURE_COLS"),
    AuditTarget("PlaceAbilityModel", "models.place_ability_model", "FEATURE_COLS"),
    AuditTarget("AbilityModel", "models.stage1_ability_model", "FEATURE_COLS"),
    AuditTarget("WinTwoStageModel", "models.two_stage_return_model", "FEATURE_COLS"),
)


# ---------------------------------------------------------------------------
# run_feature_audit — shared core for tests and audit script
# ---------------------------------------------------------------------------


def _get_model_features(target: AuditTarget) -> list[str] | None:
    """Dynamically import a model class and read its feature_cols_attr."""
    try:
        module = importlib.import_module(target.model_module)
        cls = getattr(module, target.model_class_name, None)
        if cls is None:
            logger.warning("Class %s not found in %s", target.model_class_name, target.model_module)
            return None
        return list(getattr(cls, target.feature_cols_attr, []))
    except Exception as exc:
        logger.warning(
            "Failed to import %s.%s: %s",
            target.model_module, target.model_class_name, exc,
        )
        return None


def _check_model(target: AuditTarget, *, critical: bool) -> dict[str, object]:
    """Check a single model against forbidden feature sets.

    Returns a dict with:
      - model_name: str
      - model_class_name: str
      - status: "PASS" | "FAIL" | "WARN" | "ERROR"
      - forbidden_intersections: list[str]
      - warning_intersections: list[str]
      - checked_feature_count: int
    """
    features = _get_model_features(target)
    if features is None:
        return {
            "model_name": target.model_class_name,
            "model_class_name": target.model_class_name,
            "status": "ERROR",
            "forbidden_intersections": [],
            "warning_intersections": [],
            "checked_feature_count": 0,
        }

    feature_set = set(features)
    calibrator_intersection = sorted(feature_set & FORBIDDEN_CALIBRATOR_FEATURES)
    ranker_intersection = sorted(feature_set & FORBIDDEN_RANKER_FEATURES)
    all_forbidden = sorted(set(calibrator_intersection + ranker_intersection))

    if critical:
        # Critical targets: any intersection = FAIL
        status = "FAIL" if all_forbidden else "PASS"
        return {
            "model_name": target.model_class_name,
            "model_class_name": target.model_class_name,
            "status": status,
            "forbidden_intersections": all_forbidden,
            "warning_intersections": [],
            "checked_feature_count": len(features),
        }
    else:
        # Advisory targets: intersection = WARN, no intersection = PASS
        status = "WARN" if all_forbidden else "PASS"
        return {
            "model_name": target.model_class_name,
            "model_class_name": target.model_class_name,
            "status": status,
            "forbidden_intersections": [],
            "warning_intersections": all_forbidden,
            "checked_feature_count": len(features),
        }


def run_feature_audit() -> dict[str, object]:
    """Run feature routing audit on all critical and advisory target models.

    Returns:
        dict with keys:
          - registry_version: str
          - critical_models: list[dict] — each with model_name, status,
            forbidden_intersections, warning_intersections, checked_feature_count
          - advisory_models: list[dict] — same structure
          - overall_status: "PASS" | "FAIL"
    """
    critical_results = [_check_model(t, critical=True) for t in CRITICAL_TARGET_MODELS]
    advisory_results = [_check_model(t, critical=False) for t in ADVISORY_TARGET_MODELS]

    overall_status = "PASS"
    for r in critical_results:
        if r["status"] == "FAIL":
            overall_status = "FAIL"
            break

    return {
        "registry_version": REGISTRY_VERSION,
        "critical_models": critical_results,
        "advisory_models": advisory_results,
        "overall_status": overall_status,
    }
