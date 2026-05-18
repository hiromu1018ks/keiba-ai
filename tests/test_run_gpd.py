"""Tests for run_gpd.py CLI script and matplotlib visualization.

TDD RED phase: Failing tests for CLI argument parsing, PNG generation,
per-model PNG output, stacked bar edge cases, output directory creation,
and integration with compute_gpd_diagnostics.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib

# Use Agg backend for headless testing
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gpd_result() -> dict:
    """Create a mock GPD result dict with controlled depth/category gain data."""
    return {
        "models": {
            "stage1_turf": {
                "tier": "primary",
                "depth_gains": {
                    "depths": [1, 2, 3, 4, 5],
                    "categories": ["market", "market", "fundamental", "fundamental", "categorical"],
                    "gains": [100.0, 80.0, 60.0, 40.0, 20.0],
                    "num_trees": 10,
                    "max_depth": 5,
                    "total_gain": 300.0,
                },
                "market_dominance_ratio": 0.35,
                "fundamental_activation_depth": 3,
            },
            "win_hit_turf": {
                "tier": "primary",
                "depth_gains": {
                    "depths": [1, 2, 3],
                    "categories": ["market", "fundamental", "categorical"],
                    "gains": [50.0, 30.0, 10.0],
                    "num_trees": 5,
                    "max_depth": 3,
                    "total_gain": 90.0,
                },
                "market_dominance_ratio": 0.2,
                "fundamental_activation_depth": 2,
            },
        },
        "metadata": {
            "num_boosters_analyzed": 2,
            "feature_category_counts": {
                "market": 41,
                "fundamental": 119,
                "categorical": 19,
            },
        },
    }


# ---------------------------------------------------------------------------
# Test 1: CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLIArgumentParsing:
    """Verify argparse accepts --models-dir, --output-dir, --ensemble with defaults."""

    def test_default_values(self) -> None:
        """Default --models-dir=data/models, --output-dir=data/gpd, --ensemble=False."""
        from scripts.run_gpd import build_parser

        parser = build_parser()
        args = parser.parse_args([])

        assert args.models_dir == Path("data/models")
        assert args.output_dir == Path("data/gpd")
        assert args.ensemble is False

    def test_custom_values(self) -> None:
        """Custom paths and --ensemble flag parsed correctly."""
        from scripts.run_gpd import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "--models-dir",
                "/tmp/models",
                "--output-dir",
                "/tmp/gpd_out",
                "--ensemble",
            ]
        )

        assert args.models_dir == Path("/tmp/models")
        assert args.output_dir == Path("/tmp/gpd_out")
        assert args.ensemble is True

    def test_path_type_conversion(self) -> None:
        """Arguments are converted to Path objects."""
        from scripts.run_gpd import build_parser

        parser = build_parser()
        args = parser.parse_args(["--models-dir", "some/dir"])

        assert isinstance(args.models_dir, Path)
        assert isinstance(args.output_dir, Path)


# ---------------------------------------------------------------------------
# Test 2: PNG generation
# ---------------------------------------------------------------------------


class TestPNGGeneration:
    """Verify PNG file is created with non-zero size."""

    def test_png_created(self, tmp_path: Path) -> None:
        """PNG file is created at expected path with non-zero size."""
        from scripts.run_gpd import plot_gpd_charts

        result = _make_gpd_result()
        png_paths = plot_gpd_charts(result, output_dir=tmp_path)

        assert len(png_paths) > 0
        for png_path in png_paths:
            assert png_path.exists()
            assert png_path.stat().st_size > 0
            assert png_path.suffix == ".png"

    def test_png_content_is_valid_image(self, tmp_path: Path) -> None:
        """Generated PNG starts with PNG magic bytes."""
        from scripts.run_gpd import plot_gpd_charts

        result = _make_gpd_result()
        png_paths = plot_gpd_charts(result, output_dir=tmp_path)

        for png_path in png_paths:
            with open(png_path, "rb") as f:
                header = f.read(8)
            # PNG magic bytes: 89 50 4E 47 0D 0A 1A 0A
            assert header[:4] == b"\x89PNG"


# ---------------------------------------------------------------------------
# Test 3: PNG per model
# ---------------------------------------------------------------------------


class TestPNGPerModel:
    """Verify one PNG per model with correct naming convention."""

    def test_one_png_per_model(self, tmp_path: Path) -> None:
        """One PNG is generated for each model in the result dict."""
        from scripts.run_gpd import plot_gpd_charts

        result = _make_gpd_result()
        png_paths = plot_gpd_charts(result, output_dir=tmp_path)

        expected_models = list(result["models"].keys())
        assert len(png_paths) == len(expected_models)

    def test_png_naming_convention(self, tmp_path: Path) -> None:
        """PNG files follow gpd_{model_name}.png naming convention."""
        from scripts.run_gpd import plot_gpd_charts

        result = _make_gpd_result()
        png_paths = plot_gpd_charts(result, output_dir=tmp_path)

        expected_names = {f"gpd_{name}.png" for name in result["models"]}
        actual_names = {p.name for p in png_paths}
        assert actual_names == expected_names


# ---------------------------------------------------------------------------
# Test 4: Stacked bar edge cases
# ---------------------------------------------------------------------------


class TestStackedBarEdgeCases:
    """Verify chart function handles edge cases."""

    def test_single_depth(self, tmp_path: Path) -> None:
        """Chart handles a model with only one depth level."""
        from scripts.run_gpd import plot_gpd_charts

        result: dict = {
            "models": {
                "single_depth_model": {
                    "tier": "detailed",
                    "depth_gains": {
                        "depths": [1],
                        "categories": ["market"],
                        "gains": [50.0],
                        "num_trees": 1,
                        "max_depth": 1,
                        "total_gain": 50.0,
                    },
                    "market_dominance_ratio": None,
                    "fundamental_activation_depth": None,
                },
            },
            "metadata": {"num_boosters_analyzed": 1, "feature_category_counts": {}},
        }

        png_paths = plot_gpd_charts(result, output_dir=tmp_path)
        assert len(png_paths) == 1
        assert png_paths[0].exists()

    def test_single_category(self, tmp_path: Path) -> None:
        """Chart handles a model with only one category."""
        from scripts.run_gpd import plot_gpd_charts

        result: dict = {
            "models": {
                "single_cat_model": {
                    "tier": "primary",
                    "depth_gains": {
                        "depths": [1, 2, 3],
                        "categories": ["fundamental", "fundamental", "fundamental"],
                        "gains": [10.0, 20.0, 30.0],
                        "num_trees": 3,
                        "max_depth": 3,
                        "total_gain": 60.0,
                    },
                    "market_dominance_ratio": None,
                    "fundamental_activation_depth": None,
                },
            },
            "metadata": {"num_boosters_analyzed": 1, "feature_category_counts": {}},
        }

        png_paths = plot_gpd_charts(result, output_dir=tmp_path)
        assert len(png_paths) == 1
        assert png_paths[0].exists()

    def test_zero_gains(self, tmp_path: Path) -> None:
        """Chart handles a model where all gains are zero."""
        from scripts.run_gpd import plot_gpd_charts

        result: dict = {
            "models": {
                "zero_gain_model": {
                    "tier": "detailed",
                    "depth_gains": {
                        "depths": [1, 2, 3],
                        "categories": ["market", "fundamental", "categorical"],
                        "gains": [0.0, 0.0, 0.0],
                        "num_trees": 3,
                        "max_depth": 3,
                        "total_gain": 0.0,
                    },
                    "market_dominance_ratio": None,
                    "fundamental_activation_depth": None,
                },
            },
            "metadata": {"num_boosters_analyzed": 1, "feature_category_counts": {}},
        }

        png_paths = plot_gpd_charts(result, output_dir=tmp_path)
        assert len(png_paths) == 1
        assert png_paths[0].exists()

    def test_large_depth_count(self, tmp_path: Path) -> None:
        """Chart handles a model with >20 depth levels."""
        from scripts.run_gpd import plot_gpd_charts

        n = 25
        depths = list(range(1, n + 1))
        categories = ["market", "fundamental", "categorical"] * 9  # 27 > 25
        categories = categories[:n]
        gains = [float(i) for i in range(1, n + 1)]

        result: dict = {
            "models": {
                "deep_model": {
                    "tier": "primary",
                    "depth_gains": {
                        "depths": depths,
                        "categories": categories,
                        "gains": gains,
                        "num_trees": 50,
                        "max_depth": n,
                        "total_gain": sum(gains),
                    },
                    "market_dominance_ratio": 0.1,
                    "fundamental_activation_depth": 5,
                },
            },
            "metadata": {"num_boosters_analyzed": 1, "feature_category_counts": {}},
        }

        png_paths = plot_gpd_charts(result, output_dir=tmp_path)
        assert len(png_paths) == 1
        assert png_paths[0].exists()


# ---------------------------------------------------------------------------
# Test 5: Output directory creation
# ---------------------------------------------------------------------------


class TestOutputDirectoryCreation:
    """Verify output_dir is created if it does not exist."""

    def test_nested_output_dir_created(self, tmp_path: Path) -> None:
        """Non-existent nested output directory is created automatically."""
        from scripts.run_gpd import plot_gpd_charts

        nested_dir = tmp_path / "nested" / "deep" / "output"
        assert not nested_dir.exists()

        result = _make_gpd_result()
        plot_gpd_charts(result, output_dir=nested_dir)

        assert nested_dir.exists()
        assert nested_dir.is_dir()


# ---------------------------------------------------------------------------
# Test 6: Integration with compute_gpd_diagnostics
# ---------------------------------------------------------------------------


class TestIntegrationWithGpdDiagnostics:
    """Verify CLI integrates with compute_gpd_diagnostics end-to-end."""

    def test_main_calls_compute_and_plots(self, tmp_path: Path) -> None:
        """main() loads models, runs diagnostics, plots charts."""
        from scripts.run_gpd import main

        mock_result = _make_gpd_result()
        mock_models = MagicMock()
        mock_info = MagicMock()

        with (
            patch("scripts.run_gpd.ModelLoader") as mock_loader,
            patch(
                "scripts.run_gpd.compute_gpd_diagnostics",
                return_value=mock_result,
            ) as mock_compute,
            patch("scripts.run_gpd.console_summary") as mock_console,
            patch(
                "scripts.run_gpd.plot_gpd_charts",
                return_value=[],
            ) as mock_plot,
            patch(
                "sys.argv",
                [
                    "run_gpd.py",
                    "--models-dir",
                    str(tmp_path / "models"),
                    "--output-dir",
                    str(tmp_path / "gpd"),
                    "--ensemble",
                ],
            ),
        ):
            mock_loader.return_value.load_from_dir.return_value = (mock_models, mock_info)
            main()

            # Verify ModelLoader was called with correct args
            mock_loader.return_value.load_from_dir.assert_called_once()
            call_kwargs = mock_loader.return_value.load_from_dir.call_args
            assert call_kwargs.kwargs.get("use_ensemble_override") is True

            # Verify compute_gpd_diagnostics was called
            mock_compute.assert_called_once()

            # Verify console_summary was called
            mock_console.assert_called_once_with(mock_result)

            # Verify plot_gpd_charts was called
            mock_plot.assert_called_once()

    def test_main_default_args(self, tmp_path: Path) -> None:
        """main() works with default arguments (no --ensemble)."""
        from scripts.run_gpd import main

        mock_result = _make_gpd_result()
        mock_models = MagicMock()
        mock_info = MagicMock()

        with (
            patch("scripts.run_gpd.ModelLoader") as mock_loader,
            patch(
                "scripts.run_gpd.compute_gpd_diagnostics",
                return_value=mock_result,
            ),
            patch("scripts.run_gpd.console_summary"),
            patch("scripts.run_gpd.plot_gpd_charts", return_value=[]),
            patch(
                "sys.argv",
                [
                    "run_gpd.py",
                    "--models-dir",
                    str(tmp_path / "models"),
                    "--output-dir",
                    str(tmp_path / "gpd"),
                ],
            ),
        ):
            mock_loader.return_value.load_from_dir.return_value = (mock_models, mock_info)
            main()

            call_kwargs = mock_loader.return_value.load_from_dir.call_args
            assert call_kwargs.kwargs.get("use_ensemble_override") is False
