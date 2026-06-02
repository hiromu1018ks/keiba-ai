"""Create a selective-interaction MAWC shadow variant.

This is a lightweight v2.3 pre-shadow step. It trains only the
MarketAwareWinCalibrator logistic layer from the persisted win-selection OOF
artifact and copies the existing model directories, replacing only MAWC joblib
files.

It does not run training, backtest, walk-forward validation, or shadow comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from models.market_aware_win_calibrator import MarketAwareWinCalibrator  # noqa: E402
from utils.wf_splits import walk_forward_race_splits  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


DEFAULT_KEPT_MODEL_INTERACTIONS: list[str] = [
    "logit_model_x_100+",
    "logit_model_x_top_25",
    "logit_model_x_pop_1",
    "logit_model_x_pop_4_6",
    "logit_model_x_2-3",
    "logit_model_x_10-30",
]

ALL_MODEL_INTERACTIONS: list[str] = [
    "logit_model_x_1-2",
    "logit_model_x_2-3",
    "logit_model_x_3-5",
    "logit_model_x_5-10",
    "logit_model_x_10-30",
    "logit_model_x_30-100",
    "logit_model_x_100+",
    "logit_model_x_pop_1",
    "logit_model_x_pop_2_3",
    "logit_model_x_pop_4_6",
    "logit_model_x_pop_7_9",
    "logit_model_x_pop_10_plus",
    "logit_model_x_top_25",
    "logit_model_x_mid_25_75",
    "logit_model_x_bottom_25",
]

KEPT_MODEL_INTERACTIONS: list[str] = list(DEFAULT_KEPT_MODEL_INTERACTIONS)
REMOVED_MODEL_INTERACTIONS: list[str] = [
    name for name in ALL_MODEL_INTERACTIONS if name not in KEPT_MODEL_INTERACTIONS
]

C_GRID: list[float] = [0.03, 0.1, 0.3, 1.0, 3.0]


def compute_ece(y_pred: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Compute expected calibration error with equal-width bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_pred)
    if total == 0:
        return 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_pred >= edges[i]) & (y_pred <= edges[i + 1])
        else:
            mask = (y_pred >= edges[i]) & (y_pred < edges[i + 1])
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        ece += abs(float(y_pred[mask].mean()) - float(y_true[mask].mean())) * n_bin / total
    return float(ece)


def prepare_oof_data(oof_path: Path) -> pd.DataFrame:
    """Load win-selection OOF and derive MAWC training columns."""
    df = pd.read_parquet(oof_path)
    if "p_win_oof" not in df.columns:
        raise ValueError(f"{oof_path} must contain p_win_oof")

    df = df.copy()
    df["p_model"] = pd.to_numeric(df["p_win_oof"], errors="coerce")
    df["p_market"] = np.clip(1.0 / pd.to_numeric(df["tanodds"], errors="coerce"), 0.01, 0.99)
    df["field_size"] = df.groupby("race_id", observed=True)["race_id"].transform("size")
    df["popularity_rank"] = (
        df.groupby("race_id", observed=True)["tanodds"]
        .rank(method="min", ascending=True)
        .astype(float)
    )
    df["p_win_race_rank_pct"] = (
        df.groupby("race_id", observed=True)["p_model"]
        .rank(pct=True, method="min", ascending=False)
    )
    df["race_date"] = pd.to_datetime(df["race_date"])

    required = [
        "p_model",
        "p_market",
        "tanodds",
        "popularity_rank",
        "field_size",
        "kakuteijyuni",
        "race_id",
        "surface",
        "race_date",
    ]
    return df.dropna(subset=required).reset_index(drop=True)


def build_selective_feature_matrix(
    df: pd.DataFrame,
    helper: MarketAwareWinCalibrator,
) -> tuple[np.ndarray, list[str]]:
    """Build 42-dim MAWC features by removing 9 unstable model interactions."""
    x_full, names_full = helper.build_feature_matrix(df)
    keep_indices: list[int] = []
    keep_names: list[str] = []
    for idx, name in enumerate(names_full):
        if name.startswith("logit_model_x_") and name not in KEPT_MODEL_INTERACTIONS:
            continue
        keep_indices.append(idx)
        keep_names.append(name)
    return x_full[:, keep_indices], keep_names


def select_c_by_logloss(
    df: pd.DataFrame,
    x_matrix: np.ndarray,
    y: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    """Select C with baseline MAWC logic: lowest WF logloss, tie smaller C."""
    splits = walk_forward_race_splits(df.reset_index(drop=True), n_splits=5)
    if len(splits) < 2:
        return 1.0, {"warning": "insufficient_splits", "best_c": 1.0, "c_grid_results": {}}

    results: dict[float, dict[str, float]] = {}
    best_c = C_GRID[0]
    best_logloss = float("inf")
    for c_value in C_GRID:
        fold_loglosses: list[float] = []
        fold_briers: list[float] = []
        fold_eces: list[float] = []
        for train_idx, val_idx in splits:
            lr = LogisticRegression(C=c_value, max_iter=1000, fit_intercept=True)
            lr.fit(x_matrix[train_idx], y[train_idx])
            p_val = lr.predict_proba(x_matrix[val_idx])[:, 1]
            fold_loglosses.append(float(log_loss(y[val_idx], p_val)))
            fold_briers.append(float(brier_score_loss(y[val_idx], p_val)))
            fold_eces.append(compute_ece(p_val, y[val_idx]))

        mean_logloss = float(np.mean(fold_loglosses))
        results[c_value] = {
            "mean_logloss": mean_logloss,
            "mean_brier": float(np.mean(fold_briers)),
            "mean_ece": float(np.mean(fold_eces)),
        }
        if mean_logloss < best_logloss or (
            np.isclose(mean_logloss, best_logloss) and c_value < best_c
        ):
            best_logloss = mean_logloss
            best_c = c_value

    return best_c, {
        "best_c": best_c,
        "best_logloss": best_logloss,
        "c_grid_results": {str(c): result for c, result in results.items()},
    }


def train_selective_mawc(
    df: pd.DataFrame,
    *,
    year: int,
    surface: str,
) -> tuple[MarketAwareWinCalibrator, dict[str, Any]]:
    """Train one selective MAWC for a year/surface."""
    helper = MarketAwareWinCalibrator()
    x_matrix, feature_names = build_selective_feature_matrix(df, helper)
    y = (df["kakuteijyuni"] == 1).astype(int).values
    best_c, c_selection = select_c_by_logloss(df, x_matrix, y)

    lr = LogisticRegression(C=best_c, max_iter=1000, fit_intercept=True)
    lr.fit(x_matrix, y)
    p_pred = lr.predict_proba(x_matrix)[:, 1]

    coef = lr.coef_[0]
    idx_model = feature_names.index("logit_model")
    idx_market = feature_names.index("logit_market")
    abs_model = abs(float(coef[idx_model]))
    abs_market = abs(float(coef[idx_market]))
    beta_market = abs_market / (abs_model + abs_market) if abs_model + abs_market > 0 else 0.0
    feature_dim = int(x_matrix.shape[1])

    mawc = MarketAwareWinCalibrator(
        calibrator=lr,
        feature_names=feature_names,
        best_c=best_c,
        c_selection_results=c_selection,
        training_summary={
            "best_c": best_c,
            "n_samples": int(len(y)),
            "n_features": feature_dim,
            "n_positive": int(y.sum()),
            "deployment_status": "shadow_only",
            "fix_version": f"v2.3-selective-{feature_dim}",
            "original_feature_dim": 51,
            "selective_feature_dim": feature_dim,
            "kept_model_interactions": KEPT_MODEL_INTERACTIONS,
            "removed_model_interactions": REMOVED_MODEL_INTERACTIONS,
            "beta_market_contribution": beta_market,
            "coef_logit_model": float(coef[idx_model]),
            "coef_logit_market": float(coef[idx_market]),
            "in_sample_brier": float(brier_score_loss(y, p_pred)),
            "in_sample_logloss": float(log_loss(y, p_pred)),
            "in_sample_ece": compute_ece(p_pred, y),
            "source_oof": "data/oof/win_selection_oof.parquet",
            "p_model_source": "p_win_oof",
            "year": year,
            "surface": surface,
        },
        _trained=True,
    )
    summary = {
        "year": year,
        "surface": surface,
        "best_c": best_c,
        "n_rows": int(len(y)),
        "n_features": feature_dim,
        "beta_market": beta_market,
        "coef_logit_model": float(coef[idx_model]),
        "coef_logit_market": float(coef[idx_market]),
        "in_sample_brier": float(brier_score_loss(y, p_pred)),
        "in_sample_logloss": float(log_loss(y, p_pred)),
        "in_sample_ece": compute_ece(p_pred, y),
        "c_selection": c_selection,
    }
    return mawc, summary


def read_train_date_range(source_year_dir: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Read train_start/train_end from a model meta.json if present."""
    meta_path = source_year_dir / "meta.json"
    if not meta_path.is_file():
        return None, None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    start = pd.to_datetime(meta.get("train_start")) if meta.get("train_start") else None
    end = pd.to_datetime(meta.get("train_end")) if meta.get("train_end") else None
    return start, end


def copy_and_replace_year(
    *,
    source_model_dir: Path,
    target_root: Path,
    year: int,
    trained: dict[str, MarketAwareWinCalibrator],
) -> None:
    """Copy one year model directory and replace MAWC artifacts."""
    source_year_dir = source_model_dir / str(year)
    if not source_year_dir.is_dir():
        raise FileNotFoundError(f"Source year model directory not found: {source_year_dir}")

    target_year_dir = target_root / str(year)
    shutil.copytree(source_year_dir, target_year_dir, dirs_exist_ok=True)
    for surface, mawc in trained.items():
        mawc_path = target_year_dir / f"market_aware_win_calibrator_{surface}.joblib"
        mawc.save(mawc_path)


def write_summary(target_root: Path, manifest: dict[str, Any]) -> Path:
    """Write a compact Markdown summary."""
    lines = [
        f"# MAWC Selective {manifest['feature_dim']}-dim Variant",
        "",
        f"Generated: {manifest['generated_at']}",
        "",
        "## Configuration",
        "",
        "- p_model source: `p_win_oof` from `data/oof/win_selection_oof.parquet`",
        "- C grid: `[0.03, 0.1, 0.3, 1.0, 3.0]`",
        "- C selection: baseline-compatible WF logloss minimum",
        f"- Feature dim: 51 -> {manifest['feature_dim']}",
        "",
        "Kept `logit_model_x_*` interactions:",
    ]
    for name in KEPT_MODEL_INTERACTIONS:
        lines.append(f"- `{name}`")
    lines.extend(["", "Removed `logit_model_x_*` interactions:"])
    for name in REMOVED_MODEL_INTERACTIONS:
        lines.append(f"- `{name}`")

    lines.extend([
        "",
        "## Per Year / Surface",
        "",
        "| Year | Surface | Rows | Best C | beta_market | coef_model | coef_market | ECE |",
        "|------|---------|------|--------|-------------|------------|-------------|-----|",
    ])
    for year_key, surfaces in manifest["per_year_surface"].items():
        for surface, data in surfaces.items():
            lines.append(
                f"| {year_key} | {surface} | {data['n_rows']} | {data['best_c']:.4f} | "
                f"{data['beta_market']:.4f} | {data['coef_logit_model']:+.4f} | "
                f"{data['coef_logit_market']:+.4f} | {data['in_sample_ece']:.6f} |"
            )

    lines.extend([
        "",
        "## Notes",
        "",
        "- This only creates a shadow variant directory.",
        "- It does not run shadow comparison or backtesting.",
        "- The 2024 fold uses available OOF rows within the source model train_end;",
        "  the OOF artifact starts in 2022, so older train years are not reconstructed.",
    ])
    path = target_root / "selective_retrain_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Create selective 42-dim MAWC shadow variant")
    parser.add_argument(
        "--oof-path",
        type=Path,
        default=Path("data/oof/win_selection_oof.parquet"),
    )
    parser.add_argument(
        "--source-model-dir",
        type=Path,
        default=Path("data/models-backtest"),
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("data/models-backtest-mawc-selective"),
    )
    parser.add_argument("--years", type=str, default="2024,2025")
    parser.add_argument(
        "--kept-interactions",
        type=str,
        default=",".join(DEFAULT_KEPT_MODEL_INTERACTIONS),
        help=(
            "Comma-separated logit_model_x_* interactions to keep. "
            "Use full names, e.g. logit_model_x_100+."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Remove target root before writing")
    return parser


def configure_kept_interactions(value: str) -> None:
    """Set global kept/removed interaction lists from a CLI value."""
    global KEPT_MODEL_INTERACTIONS, REMOVED_MODEL_INTERACTIONS

    kept = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(kept) - set(ALL_MODEL_INTERACTIONS))
    if unknown:
        raise ValueError(f"Unknown logit_model interaction(s): {unknown}")
    KEPT_MODEL_INTERACTIONS = kept
    REMOVED_MODEL_INTERACTIONS = [
        name for name in ALL_MODEL_INTERACTIONS if name not in KEPT_MODEL_INTERACTIONS
    ]


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    configure_kept_interactions(args.kept_interactions)
    years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
    if args.target_root.exists() and args.force:
        shutil.rmtree(args.target_root)
    args.target_root.mkdir(parents=True, exist_ok=True)

    df_all = prepare_oof_data(args.oof_path)
    manifest: dict[str, Any] = {
        "mawc_fix_version": f"v2.3-selective-{36 + len(KEPT_MODEL_INTERACTIONS)}",
        "source_model_dir": str(args.source_model_dir),
        "target_variant_dir": str(args.target_root),
        "source_oof": str(args.oof_path),
        "p_model_source": "p_win_oof",
        "C_grid": C_GRID,
        "feature_dim": 36 + len(KEPT_MODEL_INTERACTIONS),
        "original_feature_dim": 51,
        "kept_model_interactions": KEPT_MODEL_INTERACTIONS,
        "removed_model_interactions": REMOVED_MODEL_INTERACTIONS,
        "years": [str(y) for y in years],
        "per_year_surface": {},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    for year in years:
        source_year_dir = args.source_model_dir / str(year)
        train_start, train_end = read_train_date_range(source_year_dir)
        df_year = df_all.copy()
        if train_start is not None:
            df_year = df_year[df_year["race_date"] >= train_start]
        if train_end is not None:
            df_year = df_year[df_year["race_date"] <= train_end]

        logger.info(
            "Training selective MAWC year=%s rows=%d train_start=%s train_end=%s",
            year,
            len(df_year),
            train_start.date() if train_start is not None else None,
            train_end.date() if train_end is not None else None,
        )

        trained: dict[str, MarketAwareWinCalibrator] = {}
        manifest["per_year_surface"][str(year)] = {}
        for surface in ["turf", "dirt"]:
            df_surface = (
                df_year[df_year["surface"] == surface]
                .copy()
                .sort_values(["race_date", "race_id", "umaban"])
                .reset_index(drop=True)
            )
            if len(df_surface) < 500:
                raise ValueError(f"Insufficient OOF rows for year={year} surface={surface}")
            mawc, summary = train_selective_mawc(df_surface, year=year, surface=surface)
            trained[surface] = mawc
            manifest["per_year_surface"][str(year)][surface] = summary
            logger.info(
                "year=%s surface=%s C=%.4f beta=%.4f coef_market=%+.4f ECE=%.6f",
                year,
                surface,
                summary["best_c"],
                summary["beta_market"],
                summary["coef_logit_market"],
                summary["in_sample_ece"],
            )

        copy_and_replace_year(
            source_model_dir=args.source_model_dir,
            target_root=args.target_root,
            year=year,
            trained=trained,
        )

    manifest_path = args.target_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path = write_summary(args.target_root, manifest)

    print()
    print("MAWC Selective Variant Complete")
    print("=" * 60)
    print(f"Target:   {args.target_root}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary:  {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
