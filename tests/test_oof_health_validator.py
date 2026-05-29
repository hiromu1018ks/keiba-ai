"""OOFHealthValidator unit tests -- OOF-01~08, XCT-05, XCT-08."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from validation.oof_health_validator import (
    OOF_PREDICTIONS_PROFILE,
    WIN_SELECTION_OOF_PROFILE,
    OOFHealthProfile,
    OOFHealthValidator,
    ValidationResult,
    load_validated_oof,
)


def _make_valid_oof_df(
    n_races: int = 20,
    n_folds: int = 3,
    include_score: bool = False,
) -> pd.DataFrame:
    """Build a valid OOF artifact DataFrame for testing."""
    rng = np.random.default_rng(42)
    horses_per_race = 8
    n_rows = n_races * horses_per_race

    race_ids = [f"R{i:04d}" for i in range(n_races) for _ in range(horses_per_race)]
    fold_assignments = [i % n_folds for i in range(n_races) for _ in range(horses_per_race)]

    df = pd.DataFrame(
        {
            "race_id": race_ids,
            "race_date": pd.date_range("2020-01-01", periods=n_rows, freq="6h"),
            "is_oof": np.ones(n_rows, dtype=bool),
            "oof_artifact_version": np.ones(n_rows, dtype=int),
            "ability_oof_fold": fold_assignments,
            "umaban": [j + 1 for _ in range(n_races) for j in range(horses_per_race)],
        }
    )

    if include_score:
        df["p_ability_win"] = rng.uniform(0.01, 0.3, n_rows)
        df["confirmed_odds"] = rng.uniform(2.0, 50.0, n_rows)
        df["tanodds"] = df["confirmed_odds"]
        df["kakuteijyuni"] = rng.integers(1, horses_per_race + 1, n_rows)

    return df


def _make_win_selection_oof_df(
    n_races: int = 20,
    n_folds: int = 3,
) -> pd.DataFrame:
    """Build a valid win_selection_oof artifact DataFrame."""
    rng = np.random.default_rng(42)
    horses_per_race = 8
    n_rows = n_races * horses_per_race

    race_ids = [f"R{i:04d}" for i in range(n_races) for _ in range(horses_per_race)]
    fold_assignments = [i % n_folds for i in range(n_races) for _ in range(horses_per_race)]
    confirmed_odds = rng.uniform(2.0, 50.0, n_rows)

    df = pd.DataFrame(
        {
            "race_id": race_ids,
            "race_date": pd.date_range("2020-01-01", periods=n_rows, freq="6h"),
            "is_oof": np.ones(n_rows, dtype=bool),
            "oof_artifact_version": np.ones(n_rows, dtype=int),
            "kakuteijyuni": rng.integers(1, horses_per_race + 1, n_rows),
            "win_selection_oof_fold": fold_assignments,
            "win_market_selection_score": rng.uniform(0.01, 0.3, n_rows),
            "win_return_unit": rng.uniform(0.0, 1.0, n_rows),
            "win_return": rng.uniform(0.0, 100.0, n_rows),
            "confirmed_odds": confirmed_odds,
            "tanodds": confirmed_odds,
            "umaban": [j + 1 for _ in range(n_races) for j in range(horses_per_race)],
        }
    )

    return df


class TestOOFHealthProfile:
    """Tests for OOFHealthProfile dataclass."""

    def test_profile_is_frozen(self) -> None:
        profile = OOFHealthProfile(
            artifact_name="test",
            required_columns=("race_id",),
            fold_col="fold",
        )
        with pytest.raises(AttributeError):
            profile.artifact_name = "changed"  # type: ignore[misc]

    def test_default_values(self) -> None:
        profile = OOFHealthProfile(
            artifact_name="test",
            required_columns=("race_id",),
            fold_col="fold",
        )
        assert profile.score_col is None
        assert profile.return_cols == ()
        assert profile.max_top1_hit_rate == 0.35
        assert profile.max_top1_roi == 2.0
        assert profile.min_fold_count == 3
        assert profile.row_coverage_threshold == 0.70
        assert profile.enable_train_valid_overlap is False
        assert profile.strict_schema is False


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_result_is_frozen(self) -> None:
        result = ValidationResult(status="PASS", failures=[], warnings=[], metrics={})
        with pytest.raises(AttributeError):
            result.status = "FAIL"  # type: ignore[misc]


class TestOOF01Empty:
    """OOF-01: empty DataFrame raises ValueError."""

    def test_empty_df_raises_value_error(self) -> None:
        df = pd.DataFrame()
        profile = OOFHealthProfile(
            artifact_name="test",
            required_columns=("race_id",),
            fold_col="fold",
        )
        validator = OOFHealthValidator()
        with pytest.raises(ValueError, match="OOF-01"):
            validator.validate(df, profile)


class TestOOF04RowCoverage:
    """OOF-04: row coverage below threshold."""

    def test_low_coverage_returns_fail(self) -> None:
        df = _make_valid_oof_df(n_races=20, n_folds=3)
        profile = OOFHealthProfile(
            artifact_name="test",
            required_columns=("race_id",),
            fold_col="ability_oof_fold",
        )
        validator = OOFHealthValidator()
        result = validator.validate(df, profile, expected_row_count=10000)
        assert result["status"] == "FAIL"
        assert any("OOF-04" in f for f in result["failures"])

    def test_sufficient_coverage_passes(self) -> None:
        df = _make_valid_oof_df(n_races=20, n_folds=3)
        profile = OOFHealthProfile(
            artifact_name="test",
            required_columns=("race_id",),
            fold_col="ability_oof_fold",
        )
        validator = OOFHealthValidator()
        result = validator.validate(df, profile, expected_row_count=len(df))
        assert result["status"] == "PASS"

    def test_no_expected_row_count_skips_check(self) -> None:
        df = _make_valid_oof_df(n_races=20, n_folds=3)
        profile = OOFHealthProfile(
            artifact_name="test",
            required_columns=("race_id",),
            fold_col="ability_oof_fold",
        )
        validator = OOFHealthValidator()
        result = validator.validate(df, profile)
        assert result["status"] == "PASS"


class TestOOF05MinFoldCount:
    """OOF-05: fold count below minimum."""

    def test_fold_count_below_minimum(self) -> None:
        df = _make_valid_oof_df(n_races=20, n_folds=2)
        profile = OOFHealthProfile(
            artifact_name="test",
            required_columns=("race_id",),
            fold_col="ability_oof_fold",
            min_fold_count=3,
        )
        validator = OOFHealthValidator()
        result = validator.validate(df, profile)
        assert result["status"] == "FAIL"
        assert any("OOF-05" in f for f in result["failures"])

    def test_fold_count_at_minimum(self) -> None:
        df = _make_valid_oof_df(n_races=20, n_folds=3)
        profile = OOFHealthProfile(
            artifact_name="test",
            required_columns=("race_id",),
            fold_col="ability_oof_fold",
            min_fold_count=3,
        )
        validator = OOFHealthValidator()
        result = validator.validate(df, profile)
        assert result["status"] == "PASS"


class TestOOF06SameRaceMultipleFold:
    """OOF-06: same race_id in multiple folds."""

    def test_race_in_multiple_folds(self) -> None:
        df = _make_valid_oof_df(n_races=20, n_folds=3)
        # Force first race into two folds
        race0_mask = df["race_id"] == "R0000"
        df.loc[race0_mask, "ability_oof_fold"] = np.where(
            df.loc[race0_mask, "umaban"] <= 4, 0, 1
        )
        profile = OOFHealthProfile(
            artifact_name="test",
            required_columns=("race_id",),
            fold_col="ability_oof_fold",
        )
        validator = OOFHealthValidator()
        result = validator.validate(df, profile)
        assert result["status"] == "FAIL"
        assert any("OOF-06" in f for f in result["failures"])

    def test_all_races_unique_fold(self) -> None:
        df = _make_valid_oof_df(n_races=20, n_folds=3)
        profile = OOFHealthProfile(
            artifact_name="test",
            required_columns=("race_id",),
            fold_col="ability_oof_fold",
        )
        validator = OOFHealthValidator()
        result = validator.validate(df, profile)
        assert result["status"] == "PASS"


class TestOOF07RequiredColumns:
    """OOF-07: missing required columns or fold_col."""

    def test_missing_required_column(self) -> None:
        df = _make_valid_oof_df(n_races=20, n_folds=3)
        df = df.drop(columns=["race_id"])
        profile = OOFHealthProfile(
            artifact_name="test",
            required_columns=("race_id", "race_date"),
            fold_col="ability_oof_fold",
        )
        validator = OOFHealthValidator()
        with pytest.raises(ValueError, match="OOF-07"):
            validator.validate(df, profile)

    def test_missing_fold_col(self) -> None:
        df = _make_valid_oof_df(n_races=20, n_folds=3)
        df = df.drop(columns=["ability_oof_fold"])
        profile = OOFHealthProfile(
            artifact_name="test",
            required_columns=("race_id",),
            fold_col="ability_oof_fold",
        )
        validator = OOFHealthValidator()
        with pytest.raises(ValueError, match="OOF-07"):
            validator.validate(df, profile)


class TestOOF03Top1Anomaly:
    """OOF-03: top1 hit rate/ROI anomaly (profile-dependent)."""

    def test_high_hit_rate_fails(self) -> None:
        """When top1 hit rate exceeds threshold, validation fails."""
        df = _make_win_selection_oof_df(n_races=50, n_folds=3)
        # Force top1 hit rate = 100% by setting kakuteijyuni=1 for top scorer
        for race_id in df["race_id"].unique():
            race_mask = df["race_id"] == race_id
            race_df = df.loc[race_mask]
            top_idx = race_df["win_market_selection_score"].idxmax()
            df.loc[df.index.isin(race_df.index) & (df.index != top_idx), "kakuteijyuni"] = 2
            df.loc[top_idx, "kakuteijyuni"] = 1
            df.loc[top_idx, "win_return"] = 100.0
            df.loc[top_idx, "win_return_unit"] = 100.0
            df.loc[top_idx, "confirmed_odds"] = 50.0

        profile = OOFHealthProfile(
            artifact_name="test",
            required_columns=(
                "race_id",
                "race_date",
                "is_oof",
                "oof_artifact_version",
                "kakuteijyuni",
            ),
            fold_col="win_selection_oof_fold",
            score_col="win_market_selection_score",
            return_cols=("win_return_unit", "win_return", "confirmed_odds", "tanodds"),
            max_top1_hit_rate=0.35,
            max_top1_roi=2.0,
            min_guard_races=30,
        )
        validator = OOFHealthValidator()
        result = validator.validate(df, profile)
        assert result["status"] == "FAIL"
        assert any("OOF-03" in f for f in result["failures"])

    def test_profile_without_score_col_skips_check(self) -> None:
        """Profile without score_col skips OOF-03 check."""
        df = _make_valid_oof_df(n_races=20, n_folds=3)
        profile = OOFHealthProfile(
            artifact_name="test",
            required_columns=("race_id",),
            fold_col="ability_oof_fold",
            score_col=None,
        )
        validator = OOFHealthValidator()
        result = validator.validate(df, profile)
        assert result["status"] == "PASS"

    def test_missing_score_col_raises_value_error(self) -> None:
        """Profile defines score_col but df doesn't have it -- fail-fast."""
        df = _make_valid_oof_df(n_races=20, n_folds=3, include_score=False)
        profile = OOFHealthProfile(
            artifact_name="test",
            required_columns=("race_id",),
            fold_col="ability_oof_fold",
            score_col="win_market_selection_score",
            return_cols=("confirmed_odds",),
        )
        validator = OOFHealthValidator()
        with pytest.raises(ValueError, match="score_col|OOF-03"):
            validator.validate(df, profile)

    def test_raw_odds_are_counted_only_for_winners(self) -> None:
        """confirmed_odds/tanoddsは的中時だけ払戻として扱う."""
        n_races = 50
        horses_per_race = 4
        rows = []
        for race_idx in range(n_races):
            race_id = f"R{race_idx:04d}"
            top_is_winner = race_idx < 10
            for horse_idx in range(horses_per_race):
                is_top = horse_idx == 0
                rows.append(
                    {
                        "race_id": race_id,
                        "race_date": pd.Timestamp("2020-01-01"),
                        "is_oof": True,
                        "oof_artifact_version": 1,
                        "ability_oof_fold": race_idx % 3,
                        "umaban": horse_idx + 1,
                        "p_ability_win": 0.9 if is_top else 0.1,
                        "kakuteijyuni": 1 if (is_top and top_is_winner) else 2,
                        "confirmed_odds": 3.0 if (is_top and top_is_winner) else 100.0,
                        "tanodds": 3.0 if (is_top and top_is_winner) else 100.0,
                    }
                )
        df = pd.DataFrame(rows)
        profile = OOFHealthProfile(
            artifact_name="test",
            required_columns=(
                "race_id",
                "race_date",
                "is_oof",
                "oof_artifact_version",
                "kakuteijyuni",
            ),
            fold_col="ability_oof_fold",
            score_col="p_ability_win",
            return_cols=("confirmed_odds", "tanodds"),
            max_top1_hit_rate=0.35,
            max_top1_roi=2.0,
            min_guard_races=30,
        )
        result = OOFHealthValidator().validate(df, profile)
        assert result["status"] == "PASS"
        assert result["top1_hit_rate"] == pytest.approx(0.2)
        assert result["top1_roi"] == pytest.approx(0.6)


class TestOOF02TrainValidOverlap:
    """OOF-02: train/valid overlap check (profile-dependent, D-04)."""

    def test_enabled_without_metadata_raises(self) -> None:
        """D-04 fail-fast: enabled but no split_metadata raises ValueError."""
        df = _make_valid_oof_df(n_races=20, n_folds=3)
        profile = OOFHealthProfile(
            artifact_name="test",
            required_columns=("race_id",),
            fold_col="ability_oof_fold",
            enable_train_valid_overlap=True,
        )
        validator = OOFHealthValidator()
        with pytest.raises(ValueError, match="D-04"):
            validator.validate(df, profile)

    def test_disabled_skips_check(self) -> None:
        """Profile with enable_train_valid_overlap=False skips OOF-02."""
        df = _make_valid_oof_df(n_races=20, n_folds=3)
        profile = OOFHealthProfile(
            artifact_name="test",
            required_columns=("race_id",),
            fold_col="ability_oof_fold",
            enable_train_valid_overlap=False,
        )
        validator = OOFHealthValidator()
        result = validator.validate(df, profile)
        assert result["status"] == "PASS"

    def test_with_metadata_no_overlap_passes(self) -> None:
        """With metadata and no overlap, check passes."""
        df = _make_valid_oof_df(n_races=20, n_folds=3)
        all_races = df["race_id"].unique().tolist()
        split_metadata = {
            "train_race_ids": all_races[:10],
            "valid_race_ids": all_races[10:],
        }
        profile = OOFHealthProfile(
            artifact_name="test",
            required_columns=("race_id",),
            fold_col="ability_oof_fold",
            enable_train_valid_overlap=True,
        )
        validator = OOFHealthValidator()
        result = validator.validate(df, profile, split_metadata=split_metadata)
        assert result["status"] == "PASS"

    def test_with_metadata_overlap_fails(self) -> None:
        """With metadata and overlapping race_ids, validation fails."""
        df = _make_valid_oof_df(n_races=20, n_folds=3)
        all_races = df["race_id"].unique().tolist()
        # Overlap: same race in both train and valid
        split_metadata = {
            "train_race_ids": all_races[:15],
            "valid_race_ids": all_races[:5],  # overlap with train
        }
        profile = OOFHealthProfile(
            artifact_name="test",
            required_columns=("race_id",),
            fold_col="ability_oof_fold",
            enable_train_valid_overlap=True,
        )
        validator = OOFHealthValidator()
        result = validator.validate(df, profile, split_metadata=split_metadata)
        assert result["status"] == "FAIL"
        assert any("OOF-02" in f for f in result["failures"])


class TestXCT05DeterministicManifest:
    """XCT-05: deterministic JSON manifest output."""

    def test_generate_manifest_deterministic(self) -> None:
        """Two calls produce byte-identical JSON."""
        df = _make_valid_oof_df(n_races=20, n_folds=3)
        profile = OOFHealthProfile(
            artifact_name="test",
            required_columns=("race_id", "race_date", "is_oof", "oof_artifact_version"),
            fold_col="ability_oof_fold",
        )
        validator = OOFHealthValidator()
        fixed_time = "2025-01-01T00:00:00+00:00"
        with patch("validation.oof_health_validator.datetime") as mock_dt:
            mock_dt.now.return_value.isoformat.return_value = fixed_time
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            m1 = validator.generate_manifest(df, profile, artifact_hash="abc123")
            m2 = validator.generate_manifest(df, profile, artifact_hash="abc123")

        json1 = json.dumps(m1, sort_keys=True, indent=2, ensure_ascii=False)
        json2 = json.dumps(m2, sort_keys=True, indent=2, ensure_ascii=False)
        assert json1 == json2

    def test_schema_hash_order_independent(self) -> None:
        """Same columns in different order produce identical schema_hash."""
        df1 = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        df2 = pd.DataFrame({"c": [3], "a": [1], "b": [2]})

        hash1, _ = OOFHealthValidator._compute_schema_hashes(df1)
        hash2, _ = OOFHealthValidator._compute_schema_hashes(df2)
        assert hash1 == hash2


class TestXCT08ManifestFields:
    """XCT-08: manifest must contain required provenance fields."""

    def test_manifest_has_required_fields(self) -> None:
        df = _make_valid_oof_df(n_races=20, n_folds=3)
        profile = OOFHealthProfile(
            artifact_name="test",
            required_columns=("race_id", "race_date", "is_oof", "oof_artifact_version"),
            fold_col="ability_oof_fold",
        )
        validator = OOFHealthValidator()
        manifest = validator.generate_manifest(
            df, profile, artifact_hash="abc123", train_date_range=("2020-01-01", "2024-12-31")
        )

        assert "artifact_version" in manifest
        assert "schema_hash" in manifest
        assert "source_oof_manifest_path" in manifest
        assert "train_date_range" in manifest
        assert manifest["train_date_range"] == ("2020-01-01", "2024-12-31")


class TestManifestContent:
    """Tests for generate_manifest() content completeness."""

    def test_manifest_contains_d10_fields(self) -> None:
        df = _make_valid_oof_df(n_races=20, n_folds=3)
        profile = OOFHealthProfile(
            artifact_name="test_artifact",
            required_columns=("race_id", "race_date", "is_oof", "oof_artifact_version"),
            fold_col="ability_oof_fold",
        )
        validator = OOFHealthValidator()
        manifest = validator.generate_manifest(df, profile, artifact_hash="deadbeef")

        # Core D-10 fields
        assert manifest["artifact_name"] == "test_artifact"
        assert manifest["artifact_hash"] == "deadbeef"
        assert manifest["schema_hash"] != ""
        assert manifest["schema_dtype_hash"] != ""
        assert manifest["row_count"] == len(df)
        assert manifest["fold_count"] == 3
        assert manifest["fold_col"] == "ability_oof_fold"
        assert "generated_at" in manifest
        assert manifest["validator_version"] == OOFHealthValidator.VALIDATOR_VERSION
        assert manifest["status"] in ("PASS", "FAIL")
        assert isinstance(manifest["failures"], list)
        assert isinstance(manifest["warnings"], list)

    def test_manifest_race_and_horse_count(self) -> None:
        df = _make_valid_oof_df(n_races=20, n_folds=3)
        profile = OOFHealthProfile(
            artifact_name="test",
            required_columns=("race_id",),
            fold_col="ability_oof_fold",
        )
        validator = OOFHealthValidator()
        manifest = validator.generate_manifest(df, profile, artifact_hash="abc")

        assert manifest["race_count"] == 20
        assert "horse_count" in manifest  # "umaban" is in df


class TestConsumerSide:
    """load_validated_oof() consumer-side tests."""

    def test_load_raises_on_fail_status(self) -> None:
        """Manifest with FAIL status raises ValueError."""
        index_data = {"test": "data/oof/manifests/test.health.json"}
        manifest_data = {
            "status": "FAIL",
            "artifact_path": "data/oof/test.parquet",
            "artifact_hash": "abc",
            "failures": ["some failure"],
        }
        # json.load called twice: once for index, once for manifest
        with patch("builtins.open", MagicMock()):
            with patch(
                "validation.oof_health_validator.json.load",
                side_effect=[index_data, manifest_data],
            ):
                with pytest.raises(ValueError, match="FAIL"):
                    load_validated_oof("test")

    def test_load_raises_on_hash_mismatch(self) -> None:
        """Manifest with mismatched artifact_hash raises ValueError."""
        index_data = {"test": "data/oof/manifests/test.health.json"}
        manifest_data = {
            "status": "PASS",
            "artifact_path": "data/oof/test.parquet",
            "artifact_hash": "expected_hash",
        }
        with patch("builtins.open", MagicMock()):
            with patch(
                "validation.oof_health_validator.json.load",
                side_effect=[index_data, manifest_data],
            ):
                # Mock artifact_path.read_bytes to return data with different hash
                with patch("validation.oof_health_validator.Path") as mock_path:
                    mock_artifact = MagicMock()
                    mock_artifact.read_bytes.return_value = b"some data"
                    # Path("data/oof/test.parquet") -> mock_artifact
                    mock_path.return_value = mock_artifact
                    with pytest.raises(ValueError, match="artifact_hash"):
                        load_validated_oof("test")


class TestConcreteProfiles:
    """Tests for the two concrete profile instances."""

    def test_oof_predictions_profile(self) -> None:
        assert OOF_PREDICTIONS_PROFILE.artifact_name == "oof_predictions"
        assert OOF_PREDICTIONS_PROFILE.fold_col == "ability_oof_fold"
        assert OOF_PREDICTIONS_PROFILE.score_col == "p_ability_win"
        assert "race_id" in OOF_PREDICTIONS_PROFILE.required_columns
        assert "confirmed_odds" in OOF_PREDICTIONS_PROFILE.return_cols

    def test_oof_predictions_profile_does_not_use_in_sample_win_prediction(self) -> None:
        """oof_predictionsの健全性チェックでin-sample p_win_predを使わない."""
        assert OOF_PREDICTIONS_PROFILE.score_col != "p_win_pred"

    def test_win_selection_oof_profile(self) -> None:
        assert WIN_SELECTION_OOF_PROFILE.artifact_name == "win_selection_oof"
        assert WIN_SELECTION_OOF_PROFILE.fold_col == "win_selection_oof_fold"
        assert WIN_SELECTION_OOF_PROFILE.score_col == "win_market_selection_score"
        assert "kakuteijyuni" in WIN_SELECTION_OOF_PROFILE.required_columns
