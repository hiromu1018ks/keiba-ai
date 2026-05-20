"""Gain per Depth diagnostic module.

Extracts LightGBM tree structures via trees_to_dataframe(), classifies each split
feature into Market/Fundamental/Categorical, and aggregates gain contributions by
tree depth. Provides Market Dominance Ratio and Fundamental Activation Depth
metrics to validate the implicit Two-Stage hypothesis.

GPD-01: depth-gain aggregation
GPD-03: StackedEnsemble Booster access
GPD-04: Two-Stage hypothesis validation metrics
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from domain.models import TrainedModelsV5

logger = logging.getLogger("models.gpd_diagnostics")

# ---------------------------------------------------------------------------
# D-01/D-03: Feature category map -- single source of truth
# ---------------------------------------------------------------------------

FEATURE_CATEGORY_MAP: dict[str, str] = {
    # --- Market (41): odds-derived, market structure, market cross-consistency ---
    "abs_log_error_win": "market",
    "deviation_rank": "market",
    "deviation_zscore": "market",
    "dm_confidence_range": "market",
    "dm_time_margin_to_fav": "market",
    "dm_time_rank": "market",
    "dm_time_zscore": "market",
    "e_return_place_pred": "market",
    "e_return_win_pred": "market",
    "entropy_ema": "market",
    "fukuoddslow": "market",
    "implied_prob_hhi": "market",
    "market_entropy": "market",
    "odds": "market",
    "odds_acceleration": "market",
    "odds_direction_consistency": "market",
    "odds_drop_rate_30_10": "market",
    "odds_drop_rate_60_10": "market",
    "odds_gap_fav12": "market",
    "odds_popularity_gap": "market",
    "odds_skewness": "market",
    "odds_to_ability_ratio": "market",
    "odds_velocity": "market",
    "odds_volatility": "market",
    "overround": "market",
    "overround_ema": "market",
    "p_minus_e_gap": "market",
    "p_x_e_interaction": "market",
    "popularity_change_30_10": "market",
    "popularity_rank": "market",
    "popularity_rank_fallback_used": "market",
    "race_mean_fuku_odds": "market",
    "race_std_fuku_odds": "market",
    "rl_favorite_in_wide_top1": "market",
    "rl_market_consistency": "market",
    "rl_trio_odds_ratio": "market",
    "rl_trio_overlap": "market",
    "rl_wide_harville_ratio": "market",
    "rl_log_odds_entropy": "market",
    "rl_odds_dispersion": "market",
    "rl_top3_odds_gap": "market",
    "rl_top1_odds": "market",
    "rl_favorite_rank_gap": "market",
    "rl_n_horses": "market",
    "signed_log_error_win": "market",
    "tanninki": "market",
    "tanodds": "market",

    # --- Fundamental (119): past performance, bloodline, form, pace, course ---
    "actual_pace_fit": "fundamental",
    "bataijyu": "fundamental",
    "blinker_change": "fundamental",
    "blinker_on": "fundamental",
    "blood_condition_wr": "fundamental",
    "blood_distance_wr": "fundamental",
    "blood_prize_log": "fundamental",
    "blood_surface_wr": "fundamental",
    "blood_surface_wr_x_condition": "fundamental",
    "blood_surface_wr_race_rank": "fundamental",
    "blood_total_wr": "fundamental",
    "blood_total_wr_race_rank": "fundamental",
    "bms_distance_wr": "fundamental",
    "bms_surface_wr": "fundamental",
    "bms_wr": "fundamental",
    "breeder_strength": "fundamental",
    "class_adj_formetric": "fundamental",
    "class_demotions": "fundamental",
    "class_drop_bounce": "fundamental",
    "class_level_std": "fundamental",
    "class_max_level": "fundamental",
    "class_move": "fundamental",
    "class_net_change": "fundamental",
    "class_promotions": "fundamental",
    "closing_index_avg": "fundamental",
    "closing_index_avg_race_rank": "fundamental",
    "closing_pace_wr": "fundamental",
    "closing_speed_ratio_avg": "fundamental",
    "closing_speed_ratio_avg_race_rank": "fundamental",
    "closing_speed_ratio_zscore": "fundamental",
    "closing_speed_ratio_trend": "fundamental",
    "cond_change_avg_pos": "fundamental",
    "cond_change_exp_count": "fundamental",
    "cond_change_win_rate": "fundamental",
    "course_distance_wr": "fundamental",
    "course_record_time": "fundamental",
    "course_wr": "fundamental",
    "dam_prize_log": "fundamental",
    "dam_surface_wr": "fundamental",
    "dam_wr": "fundamental",
    "days_since_last_race": "fundamental",
    "difficulty_score": "fundamental",
    "dist_change_avg_pos": "fundamental",
    "dist_change_exp_count": "fundamental",
    "dist_change_win_rate": "fundamental",
    "distance_change": "fundamental",
    "draw_ratio": "fundamental",
    "field_size": "fundamental",
    "form_consistency": "fundamental",
    "form_peak_flag": "fundamental",
    "form_trend": "fundamental",
    "form_trend_race_rank": "fundamental",
    "freshness_score": "fundamental",
    "front_pace_wr": "fundamental",
    "haron_x_distance": "fundamental",
    "haron_zscore_trend": "fundamental",
    "harontime_late_trend": "fundamental",
    "harontime_last3f_avg": "fundamental",
    "harontime_last3f_avg_race_rank": "fundamental",
    "harontime_last3f_zscore": "fundamental",
    "harontime_last3f_trend": "fundamental",
    "harontimel5_avg": "fundamental",
    "harontimel5_avg_race_rank": "fundamental",
    "harontimel5_zscore": "fundamental",
    "haron_race_gap_avg": "fundamental",
    "haron_race_gap_zscore": "fundamental",
    "haron_race_gap_trend": "fundamental",
    "pace_adj_finish_avg": "fundamental",
    "is_nar_transfer": "fundamental",
    "jockey_prize_log": "fundamental",
    "jockey_wr_distance": "fundamental",
    "jockey_wr_overall": "fundamental",
    "jockey_wr_venue": "fundamental",
    "jt_combo_place_rate": "fundamental",
    "jt_combo_prize_log": "fundamental",
    "jt_combo_starts": "fundamental",
    "jt_combo_wr": "fundamental",
    "jyuni1c_avg": "fundamental",
    "jyuni1c_avg_race_rank": "fundamental",
    "jyuni4c_avg": "fundamental",
    "nar_recent_ratio": "fundamental",
    "norm_finish_logit_avg": "fundamental",
    "norm_finish_logit_avg_race_rank": "fundamental",
    "p_ability_place": "fundamental",
    "p_ability_win": "fundamental",
    "pace_aptitude": "fundamental",
    "pace_ratio_avg": "fundamental",
    "pace_ratio_zscore": "fundamental",
    "pace_ratio_trend": "fundamental",
    "pace_early_avg": "fundamental",
    "pace_mid_avg": "fundamental",
    "pace_late_avg": "fundamental",
    "pace_closing_power": "fundamental",
    "pace_corner_stability": "fundamental",
    "pace_position_consistency": "fundamental",
    "pace_pressure": "fundamental",
    "pace_pressure_x_closing_index": "fundamental",
    "pace_scenario_fit": "fundamental",
    "position_improvement_rate": "fundamental",
    "rel_blood_quality_rank": "fundamental",
    "rel_closing_index_rank": "fundamental",
    "rel_fuku_odds_zscore": "fundamental",
    "rel_haron_vs_mean": "fundamental",
    "rel_norm_finish_zscore": "fundamental",
    "rel_odds_ability_deviation": "fundamental",
    "rel_p_ability_win_rank": "fundamental",
    "rel_p_ability_win_zscore": "fundamental",
    "rel_popularity_rank_zscore": "fundamental",
    "rel_sire_quality_rank": "fundamental",
    "rel_timediff_rank": "fundamental",
    "rel_weight_zscore": "fundamental",
    "rest_category": "fundamental",
    "sire_distance_wr": "fundamental",
    "sire_prize_avg": "fundamental",
    "sire_surface_wr": "fundamental",
    "sire_wr": "fundamental",
    "sire_wr_x_distance": "fundamental",
    "surf_change_avg_pos": "fundamental",
    "surf_change_exp_count": "fundamental",
    "surf_change_win_rate": "fundamental",
    "surface_change": "fundamental",
    "time_improvement_rate": "fundamental",
    "timediff_avg": "fundamental",
    "timediff_avg_race_rank": "fundamental",
    "track_condition_delta": "fundamental",
    "trainer_prize_log": "fundamental",
    "trainer_wr_distance": "fundamental",
    "trainer_wr_overall": "fundamental",
    "trainer_wr_venue": "fundamental",
    "v_recovery_duration": "fundamental",
    "v_recovery_flag": "fundamental",
    "weight_absolute": "fundamental",
    "weight_change_ratio": "fundamental",
    "weight_change_zone": "fundamental",
    "weight_diff_from_mean": "fundamental",
    "weight_x_class": "fundamental",
    "weight_x_distance": "fundamental",
    "weight_zscore": "fundamental",
    "win_dominance": "fundamental",
    "weighted_recent_form_finish": "fundamental",
    "weighted_recent_form_time": "fundamental",
    "zogen_sa": "fundamental",

    # --- Categorical (19): race conditions, categorical IDs, target encoding ---
    "blood_keito_cd": "categorical",
    "blood_keito_x_surface": "categorical",
    "distance_bin": "categorical",
    "frame_number": "categorical",
    "grade_code": "categorical",
    "grade_x_form_trend": "fundamental",
    "grade_x_blood_prize_log": "fundamental",
    "distance_x_closing_index": "fundamental",
    "grade_code_x_distance_bin": "categorical",
    "kyakusitu_x_distance": "categorical",
    "kyakusitu_x_surface": "categorical",
    "kyakusitukubun_cd": "categorical",
    "kyori": "categorical",
    "surface": "categorical",
    "surface_x_distance_bin": "categorical",
    "surface_x_past_perf": "categorical",
    "surface_track_interaction": "categorical",
    "te_blood_keito_cd": "categorical",
    "te_chokyosicode": "categorical",
    "te_kisyucode": "categorical",
    "trackcd": "categorical",
    "track_condition_code": "categorical",
}

# Tier classification for each booster name pattern
_PRIMARY_PATTERNS: tuple[str, ...] = (
    "stage1_",
    "win_hit_",
    "win_ret_",
    "market_",
    "ensemble_lgbm_",
)


def _get_tier(name: str) -> str:
    """Return 'primary' or 'detailed' tier label for a booster name."""
    for prefix in _PRIMARY_PATTERNS:
        if name.startswith(prefix):
            return "primary"
    return "detailed"


# ---------------------------------------------------------------------------
# D-04/D-06: Booster extraction
# ---------------------------------------------------------------------------


def _is_booster(obj: object) -> bool:
    """Check if obj is an lgb.Booster or a duck-type equivalent.

    Uses isinstance for real Boosters and hasattr fallback for test mocks.
    """
    if isinstance(obj, lgb.Booster):
        return True
    # Duck-type check: if it has trees_to_dataframe it's Booster-compatible
    return hasattr(obj, "trees_to_dataframe") and hasattr(obj, "feature_importance")


def _extract_boosters(models: TrainedModelsV5) -> dict[str, lgb.Booster]:
    """Extract all LightGBM Boosters from TrainedModelsV5 with descriptive names.

    Iterates over all SubmodelSet entries, extracting primary and detailed tier
    Boosters. Handles StackedEnsemble unwrapping and optional models.
    """
    boosters: dict[str, lgb.Booster] = {}

    for surface, sub in models.submodels.items():
        # Stage1 AbilityModel: per-surface boosters in dict
        for key, booster in sub.stage1.models.items():
            if _is_booster(booster):
                boosters[f"stage1_{key}"] = booster

        # Win TwoStage: hit_model (may be StackedEnsemble) + return_model
        hit_model = sub.win.hit_model
        if _is_booster(hit_model):
            boosters[f"win_hit_{surface}"] = hit_model
        elif hasattr(hit_model, "lgbm_model") and hit_model.lgbm_model is not None:
            boosters[f"ensemble_lgbm_{surface}"] = hit_model.lgbm_model

        if _is_booster(sub.win.return_model):
            boosters[f"win_ret_{surface}"] = sub.win.return_model

        # Market Model
        if sub.market.model is not None:
            model = sub.market.model
            if _is_booster(model):
                boosters[f"market_{surface}"] = model
            else:
                logger.warning(
                    "market_%s is not a Booster (type=%s), skipping",
                    surface, type(model).__name__,
                )

        # EV Correction: P + E models
        if sub.ev_corrector.p_correction_model is not None:
            model = sub.ev_corrector.p_correction_model
            if _is_booster(model):
                boosters[f"ev_corr_p_{surface}"] = model
            else:
                logger.warning(
                    "ev_corr_p_%s is not a Booster (type=%s), skipping",
                    surface, type(model).__name__,
                )
        if sub.ev_corrector.e_correction_model is not None:
            model = sub.ev_corrector.e_correction_model
            if _is_booster(model):
                boosters[f"ev_corr_e_{surface}"] = model
            else:
                logger.warning(
                    "ev_corr_e_%s is not a Booster (type=%s), skipping",
                    surface, type(model).__name__,
                )

        # Place (optional)
        if sub.place is not None:
            place_hit = sub.place.hit_model
            if _is_booster(place_hit):
                boosters[f"place_hit_{surface}"] = place_hit
            elif hasattr(place_hit, "lgbm_model") and place_hit.lgbm_model is not None:
                boosters[f"place_ensemble_lgbm_{surface}"] = place_hit.lgbm_model
            if _is_booster(sub.place.return_model):
                boosters[f"place_ret_{surface}"] = sub.place.return_model

        # Place EV Correction (optional)
        if sub.place_ev_corrector is not None:
            if sub.place_ev_corrector.p_correction_model is not None:
                model = sub.place_ev_corrector.p_correction_model
                if _is_booster(model):
                    boosters[f"place_ev_corr_p_{surface}"] = model
                else:
                    logger.warning(
                        "place_ev_corr_p_%s is not a Booster (type=%s), skipping",
                        surface, type(model).__name__,
                    )
            if sub.place_ev_corrector.e_correction_model is not None:
                model = sub.place_ev_corrector.e_correction_model
                if _is_booster(model):
                    boosters[f"place_ev_corr_e_{surface}"] = model
                else:
                    logger.warning(
                        "place_ev_corr_e_%s is not a Booster (type=%s), skipping",
                        surface, type(model).__name__,
                    )

        # Wide (optional)
        if sub.wide is not None:
            if sub.wide.hit_model is not None:
                model = sub.wide.hit_model
                if _is_booster(model):
                    boosters[f"wide_hit_{surface}"] = model
                else:
                    logger.warning(
                        "wide_hit_%s is not a Booster (type=%s), skipping",
                        surface, type(model).__name__,
                    )
            if sub.wide.return_model is not None:
                model = sub.wide.return_model
                if _is_booster(model):
                    boosters[f"wide_ret_{surface}"] = model
                else:
                    logger.warning(
                        "wide_ret_%s is not a Booster (type=%s), skipping",
                        surface, type(model).__name__,
                    )

        # ConformalEV / CQR (optional)
        if sub.conformal_ev_model is not None:
            if sub.conformal_ev_model.q_low_model is not None:
                model = sub.conformal_ev_model.q_low_model
                if _is_booster(model):
                    boosters[f"cqr_q_low_{surface}"] = model
                else:
                    logger.warning(
                        "cqr_q_low_%s is not a Booster (type=%s), skipping",
                        surface, type(model).__name__,
                    )
            if sub.conformal_ev_model.q_high_model is not None:
                model = sub.conformal_ev_model.q_high_model
                if _is_booster(model):
                    boosters[f"cqr_q_high_{surface}"] = model
                else:
                    logger.warning(
                        "cqr_q_high_%s is not a Booster (type=%s), skipping",
                        surface, type(model).__name__,
                    )

        # PlaceAbilityModel (optional, uses LGBMClassifier -> .booster_)
        if sub.place_ability is not None and hasattr(sub.place_ability, "_model"):
            inner_model = sub.place_ability._model
            if inner_model is not None and hasattr(inner_model, "booster_"):
                booster = inner_model.booster_
                if booster is not None:
                    boosters[f"place_ability_{surface}"] = booster

    return boosters


# ---------------------------------------------------------------------------
# GPD-01: Depth-gain computation
# ---------------------------------------------------------------------------


def _compute_depth_gains(booster: lgb.Booster) -> dict:
    """Compute gain by depth and category for a single Booster.

    Calls booster.trees_to_dataframe(), filters leaf nodes, maps features to
    categories via FEATURE_CATEGORY_MAP, and groups gain by (depth, category).

    Returns:
        dict with keys: depths, categories, gains, num_trees, max_depth, total_gain
    """
    tree_df: pd.DataFrame = booster.trees_to_dataframe()

    # Pitfall 3: filter leaf nodes (split_feature is None)
    split_nodes = tree_df[tree_df["split_feature"].notna()].copy()

    # Pitfall 7: fill NaN split_gain with 0
    split_nodes["split_gain"] = split_nodes["split_gain"].fillna(0.0)

    # Map split_feature to category
    split_nodes["category"] = split_nodes["split_feature"].map(
        FEATURE_CATEGORY_MAP
    ).fillna("fundamental")  # unknown features default to fundamental

    # Group by (node_depth, category) and sum split_gain
    grouped = (
        split_nodes
        .groupby(["node_depth", "category"], observed=True)["split_gain"]
        .sum()
        .reset_index()
    )

    # Compute summary statistics
    num_trees = int(tree_df["tree_index"].nunique())
    max_depth = int(tree_df["node_depth"].max())
    total_gain = float(split_nodes["split_gain"].sum())

    return {
        "depths": grouped["node_depth"].astype(int).tolist(),
        "categories": grouped["category"].tolist(),
        "gains": grouped["split_gain"].tolist(),
        "num_trees": num_trees,
        "max_depth": max_depth,
        "total_gain": total_gain,
    }


# ---------------------------------------------------------------------------
# D-11: Market Dominance Ratio
# ---------------------------------------------------------------------------


def _compute_market_dominance_ratio(depth_gains: dict) -> float | None:
    """Compute Market Dominance Ratio.

    MDR = (Market gain share at depth 1-3) - (Market gain share at depth 4+)

    Returns None if total gain at either depth range is zero.
    """
    shallow_market = 0.0
    shallow_total = 0.0
    deep_market = 0.0
    deep_total = 0.0

    for i, depth in enumerate(depth_gains["depths"]):
        gain = depth_gains["gains"][i]
        cat = depth_gains["categories"][i]
        if depth <= 3:
            shallow_total += gain
            if cat == "market":
                shallow_market += gain
        else:
            deep_total += gain
            if cat == "market":
                deep_market += gain

    if shallow_total <= 0.0 or deep_total <= 0.0:
        return None

    shallow_share = shallow_market / shallow_total
    deep_share = deep_market / deep_total

    return float(shallow_share - deep_share)


# ---------------------------------------------------------------------------
# D-11: Fundamental Activation Depth
# ---------------------------------------------------------------------------


def _compute_fundamental_activation_depth(depth_gains: dict) -> int | None:
    """Compute Fundamental Activation Depth.

    FAD = min(depth D where Fundamental gain share > Market gain share).
    Returns None if Market dominates at all depths.
    """
    # Aggregate gains per depth by category
    depth_market: dict[int, float] = {}
    depth_fundamental: dict[int, float] = {}

    for i, depth in enumerate(depth_gains["depths"]):
        gain = depth_gains["gains"][i]
        cat = depth_gains["categories"][i]
        if cat == "market":
            depth_market[depth] = depth_market.get(depth, 0.0) + gain
        elif cat == "fundamental":
            depth_fundamental[depth] = depth_fundamental.get(depth, 0.0) + gain

    all_depths = sorted(set(depth_market.keys()) | set(depth_fundamental.keys()))

    for d in all_depths:
        m = depth_market.get(d, 0.0)
        f = depth_fundamental.get(d, 0.0)
        if f > m:
            return int(d)

    return None


# ---------------------------------------------------------------------------
# D-07: JSON output helper
# ---------------------------------------------------------------------------


def _json_default(obj: object) -> object:
    """JSON non-serializable type fallback."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# D-07: Main orchestrator
# ---------------------------------------------------------------------------


def compute_gpd_diagnostics(
    models: TrainedModelsV5,
    output_dir: Path | None = None,
) -> dict:
    """Compute Gain per Depth diagnostics for all LightGBM Boosters.

    Args:
        models: TrainedModelsV5 containing all submodels.
        output_dir: If provided, write JSON report to this directory.

    Returns:
        dict with 'models' (per-model diagnostics) and 'metadata'.
    """
    boosters = _extract_boosters(models)

    result: dict = {
        "models": {},
        "metadata": {
            "num_boosters_analyzed": len(boosters),
            "feature_category_counts": {
                "market": sum(1 for v in FEATURE_CATEGORY_MAP.values() if v == "market"),
                "fundamental": sum(
                    1 for v in FEATURE_CATEGORY_MAP.values() if v == "fundamental"
                ),
                "categorical": sum(
                    1 for v in FEATURE_CATEGORY_MAP.values() if v == "categorical"
                ),
            },
        },
    }

    failed_names: list[str] = []
    for name, booster in boosters.items():
        try:
            depth_gains = _compute_depth_gains(booster)
        except Exception:
            logger.warning("Failed to compute depth gains for %s", name, exc_info=True)
            failed_names.append(name)
            continue

        mdr = _compute_market_dominance_ratio(depth_gains)
        fad = _compute_fundamental_activation_depth(depth_gains)

        result["models"][name] = {
            "tier": _get_tier(name),
            "depth_gains": depth_gains,
            "market_dominance_ratio": mdr,
            "fundamental_activation_depth": fad,
        }

    if failed_names:
        result["metadata"]["failed_boosters"] = failed_names
        result["metadata"]["num_failed"] = len(failed_names)

    # Write JSON if output_dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "gpd_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=_json_default)
        logger.info("GPD report written to %s", json_path)

    return result


# ---------------------------------------------------------------------------
# D-07/D-12: Console summary
# ---------------------------------------------------------------------------


def console_summary(result: dict) -> None:
    """Output formatted GPD diagnostic summary to log.

    No PASS/FAIL judgment per D-12 -- outputs metrics for human interpretation.
    """
    logger.info("=" * 60)
    logger.info("Gain per Depth Diagnostics")
    logger.info("=" * 60)

    meta = result.get("metadata", {})
    logger.info(
        "Boosters analyzed: %d  |  Features: market=%d fundamental=%d categorical=%d",
        meta.get("num_boosters_analyzed", 0),
        meta.get("feature_category_counts", {}).get("market", 0),
        meta.get("feature_category_counts", {}).get("fundamental", 0),
        meta.get("feature_category_counts", {}).get("categorical", 0),
    )
    logger.info("-" * 60)

    for model_name, model_data in result.get("models", {}).items():
        tier = model_data.get("tier", "?")
        mdr = model_data.get("market_dominance_ratio")
        fad = model_data.get("fundamental_activation_depth")
        depth_gains = model_data.get("depth_gains", {})

        logger.info("  [%s] %s:", tier.upper(), model_name)
        mdr_str = f"{mdr:.4f}" if mdr is not None else "N/A"
        fad_str = str(fad) if fad is not None else "N/A"
        logger.info("    Market Dominance Ratio: %s", mdr_str)
        logger.info("    Fundamental Activation Depth: %s", fad_str)
        logger.info(
            "    Trees: %d  Max Depth: %d  Total Gain: %.1f",
            depth_gains.get("num_trees", 0),
            depth_gains.get("max_depth", 0),
            depth_gains.get("total_gain", 0.0),
        )

        # Show top features at shallow depths (1-3)
        shallow_gains: dict[str, float] = {}
        for i, depth in enumerate(depth_gains.get("depths", [])):
            if depth <= 3:
                cat = depth_gains["categories"][i]
                gain = depth_gains["gains"][i]
                shallow_gains[cat] = shallow_gains.get(cat, 0.0) + gain

        if shallow_gains:
            sorted_gains = sorted(shallow_gains.items(), key=lambda x: x[1], reverse=True)
            top_str = ", ".join(f"{cat}={gain:.1f}" for cat, gain in sorted_gains[:3])
            logger.info("    Shallow (depth 1-3) gain: %s", top_str)

    logger.info("=" * 60)
