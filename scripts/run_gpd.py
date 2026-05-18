"""Gain per Depth Diagnostic CLI script.

学習済みモデルを読み込み、GPD診断を実行してJSON + PNGチャートを出力する。

使い方:
  python scripts/run_gpd.py --models-dir data/models --output-dir data/gpd
  python scripts/run_gpd.py --models-dir data/models --ensemble
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Windows cp932 環境での文字化け回避
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Headless matplotlib backend -- must be set before pyplot import
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from db.model_loader import ModelLoader  # noqa: E402
from models.gpd_diagnostics import (  # noqa: E402
    compute_gpd_diagnostics,
    console_summary,
)

# Category color scheme (D-09)
_CATEGORY_COLORS: dict[str, str] = {
    "market": "#2196F3",
    "fundamental": "#4CAF50",
    "categorical": "#FF9800",
}


def build_parser() -> argparse.ArgumentParser:
    """Build argparse parser for run_gpd CLI."""
    parser = argparse.ArgumentParser(
        description="Gain per Depth Diagnostic -- model feature analysis by tree depth",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("data/models"),
        help="Directory containing trained model files (default: data/models)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/gpd"),
        help="Output directory for JSON + PNG files (default: data/gpd)",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Enable StackedEnsemble model loading",
    )
    return parser


def plot_gpd_charts(result: dict, output_dir: Path) -> list[Path]:
    """Generate per-model GPD depth-by-category charts.

    For each model in result["models"], creates a PNG with:
    - Top subplot: stacked bar (Market/Fundamental/Categorical gain by depth)
    - Bottom subplot: cumulative gain line per category

    Args:
        result: GPD diagnostic result dict from compute_gpd_diagnostics().
        output_dir: Directory to save PNG files.

    Returns:
        List of saved PNG file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []

    categories_order = ["market", "fundamental", "categorical"]

    for model_name, model_data in result.get("models", {}).items():
        depth_gains = model_data.get("depth_gains", {})
        tier = model_data.get("tier", "?")
        mdr = model_data.get("market_dominance_ratio")
        fad = model_data.get("fundamental_activation_depth")

        depths_list = depth_gains.get("depths", [])
        categories_list = depth_gains.get("categories", [])
        gains_list = depth_gains.get("gains", [])

        if not depths_list:
            continue

        # Aggregate gains by (depth, category)
        depth_set = sorted(set(depths_list))
        gain_matrix: dict[str, list[float]] = {
            cat: [0.0] * len(depth_set) for cat in categories_order
        }
        depth_idx_map = {d: i for i, d in enumerate(depth_set)}

        for i, depth in enumerate(depths_list):
            cat = categories_list[i]
            gain = gains_list[i]
            if cat in gain_matrix and depth in depth_idx_map:
                gain_matrix[cat][depth_idx_map[depth]] += gain

        depth_arr = np.array(depth_set)
        x = np.arange(len(depth_arr))

        # --- Create figure with 2 subplots ---
        fig, (ax_bar, ax_cum) = plt.subplots(
            2,
            1,
            figsize=(max(10, len(depth_set) * 0.8), 10),
            sharex=True,
        )
        fig.suptitle(
            f"GPD: {model_name} (tier={tier})",
            fontsize=14,
            fontweight="bold",
        )

        # --- Top: Stacked bar ---
        bottoms = np.zeros(len(depth_arr))
        for cat in categories_order:
            values = np.array(gain_matrix[cat])
            ax_bar.bar(
                x,
                values,
                bottom=bottoms,
                label=cat.capitalize(),
                color=_CATEGORY_COLORS[cat],
                edgecolor="white",
                linewidth=0.5,
            )
            bottoms += values

        ax_bar.set_ylabel("Gain Contribution")
        ax_bar.legend(loc="upper right", fontsize=9)
        ax_bar.set_title("Gain by Depth (stacked by category)", fontsize=11)

        # Annotate MDR and FAD on bar chart
        info_parts: list[str] = []
        if mdr is not None:
            info_parts.append(f"MDR={mdr:.4f}")
        if fad is not None:
            info_parts.append(f"FAD={fad}")
        if info_parts:
            ax_bar.text(
                0.02,
                0.95,
                "  ".join(info_parts),
                transform=ax_bar.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightyellow", "alpha": 0.8},
            )

        # --- Bottom: Cumulative gain line ---
        total_gain_sum = float(sum(bottoms))
        for cat in categories_order:
            values = np.array(gain_matrix[cat])
            cumulative = np.cumsum(values)
            # Normalize to percentage of total
            cumulative_pct = (
                (cumulative / total_gain_sum) * 100.0 if total_gain_sum > 0 else cumulative
            )
            ax_cum.plot(
                x,
                cumulative_pct,
                label=f"{cat.capitalize()} cumulative",
                color=_CATEGORY_COLORS[cat],
                linestyle="--",
                linewidth=2.0,
            )

        ax_cum.set_xlabel("Tree Depth")
        ax_cum.set_ylabel("Cumulative Gain (%)")
        ax_cum.legend(loc="upper left", fontsize=9)
        ax_cum.set_title("Cumulative Gain by Category", fontsize=11)

        # Set x-ticks to depth values
        ax_cum.set_xticks(x)
        ax_cum.set_xticklabels([str(d) for d in depth_arr])

        fig.tight_layout(rect=[0, 0, 1, 0.95])

        # Save
        png_path = output_dir / f"gpd_{model_name}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        saved_paths.append(png_path)
        logger.info("Saved chart: %s", png_path)

    return saved_paths


def main() -> None:
    """CLI entry point: load models, run diagnostics, generate charts."""
    parser = build_parser()
    args = parser.parse_args()

    logger.info("Loading models from %s", args.models_dir)
    loader = ModelLoader()
    trained_models, _model_info = loader.load_from_dir(
        args.models_dir,
        use_ensemble_override=args.ensemble,
    )

    logger.info("Running GPD diagnostics...")
    result = compute_gpd_diagnostics(trained_models, output_dir=args.output_dir)

    console_summary(result)

    logger.info("Generating GPD charts...")
    png_paths = plot_gpd_charts(result, output_dir=args.output_dir)

    # Summary
    n_models = len(result.get("models", {}))
    json_path = args.output_dir / "gpd_report.json"
    logger.info(
        "GPD diagnostics complete: %d models analyzed, %d charts saved to %s",
        n_models,
        len(png_paths),
        args.output_dir,
    )
    logger.info("JSON report: %s", json_path)
    for p in png_paths:
        logger.info("  Chart: %s", p)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Unhandled exception: %s", exc)
        sys.exit(1)
