"""OOF artifact health validation infrastructure.

Provides OOFHealthValidator for validating all OOF artifacts before they
are consumed by downstream components (backtest, calibration, rankers).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class OOFHealthProfile:
    """Per-artifact validation configuration.

    Each OOF artifact type gets its own profile defining which checks
    to run and with what thresholds.
    """

    artifact_name: str
    required_columns: tuple[str, ...]
    fold_col: str
    score_col: str | None = None
    return_cols: tuple[str, ...] = ()
    max_top1_hit_rate: float = 0.35
    max_top1_roi: float = 2.0
    min_fold_count: int = 3
    min_guard_races: int = 30
    row_coverage_threshold: float = 0.70
    enable_train_valid_overlap: bool = False
    manifest_path: str = ""
    strict_schema: bool = False


@dataclass(frozen=True)
class ValidationResult:
    """Immutable validation result."""

    status: str
    failures: list[str]
    warnings: list[str]
    metrics: dict[str, Any]


# Concrete profiles for the two known artifact types
OOF_PREDICTIONS_PROFILE = OOFHealthProfile(
    artifact_name="oof_predictions",
    required_columns=("race_id", "race_date", "is_oof", "oof_artifact_version"),
    fold_col="ability_oof_fold",
    score_col="p_win_oof",
    return_cols=("confirmed_odds", "tanodds"),
    manifest_path="data/oof/manifests/oof_predictions.health.json",
)

WIN_SELECTION_OOF_PROFILE = OOFHealthProfile(
    artifact_name="win_selection_oof",
    required_columns=(
        "race_id", "race_date", "is_oof", "oof_artifact_version", "kakuteijyuni",
    ),
    fold_col="win_selection_oof_fold",
    score_col="win_market_selection_score",
    return_cols=("win_return_unit", "win_return", "confirmed_odds", "tanodds"),
    max_top1_hit_rate=0.35,
    max_top1_roi=2.0,
    min_guard_races=30,
    manifest_path="data/oof/manifests/win_selection_oof.health.json",
)


class OOFHealthValidator:
    """Validates OOF artifacts across all health checks (OOF-01~08)."""

    VALIDATOR_VERSION = "1.0.0"

    def validate(
        self,
        df: pd.DataFrame,
        profile: OOFHealthProfile,
        *,
        train_date_range: tuple[str, str] | None = None,
        expected_row_count: int | None = None,
        split_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run all applicable OOF health checks and return result dict."""
        failures: list[str] = []
        warnings: list[str] = []
        metrics: dict[str, Any] = {}

        # OOF-01: empty check (always-on, D-03)
        if df.empty:
            raise ValueError("OOF artifact is empty (OOF-01)")

        # OOF-07: required columns + fold_col (always-on, D-03)
        required = set(profile.required_columns) | {profile.fold_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)} (OOF-07)"
            )

        # OOF-05: minimum fold count (always-on, D-03)
        fold_count = int(df[profile.fold_col].nunique())
        metrics["fold_count"] = fold_count
        if fold_count < profile.min_fold_count:
            failures.append(
                f"Fold count {fold_count} < minimum {profile.min_fold_count} (OOF-05)"
            )

        # OOF-06: same race in multiple folds (always-on, D-03)
        race_fold_counts = df.groupby("race_id")[profile.fold_col].nunique()
        multi_fold_races = int((race_fold_counts > 1).sum())
        metrics["same_race_multiple_fold_count"] = multi_fold_races
        if multi_fold_races > 0:
            failures.append(
                f"{multi_fold_races} races appear in multiple folds (OOF-06)"
            )

        # OOF-04: row coverage (always-on, D-03)
        if expected_row_count is not None and expected_row_count > 0:
            coverage = len(df) / expected_row_count
            metrics["row_coverage_ratio"] = coverage
            if coverage < profile.row_coverage_threshold:
                failures.append(
                    f"Row coverage {coverage:.1%} < "
                    f"{profile.row_coverage_threshold:.0%} (OOF-04)"
                )

        # OOF-03: top1 hit rate / ROI anomaly (profile-dependent, D-04)
        if profile.score_col is not None and profile.return_cols:
            if profile.score_col not in df.columns:
                raise ValueError(
                    f"score_col '{profile.score_col}' not in DataFrame (OOF-03)"
                )
            missing_return = [
                c for c in profile.return_cols if c not in df.columns
            ]
            if missing_return:
                raise ValueError(
                    f"return_cols missing from DataFrame: {sorted(missing_return)} "
                    f"(OOF-03)"
                )
            self._check_top1_anomaly(df, profile, failures, metrics)

        # OOF-02: train/valid overlap (profile-dependent, D-04)
        if profile.enable_train_valid_overlap:
            if split_metadata is None:
                raise ValueError(
                    "OOF-02 check requires split_metadata but none provided "
                    "(D-04 fail-fast)"
                )
            self._check_train_valid_overlap(
                df, split_metadata, failures, metrics
            )

        status = "PASS" if not failures else "FAIL"
        return {
            "status": status,
            "failures": failures,
            "warnings": warnings,
            **metrics,
        }

    def _check_top1_anomaly(
        self,
        df: pd.DataFrame,
        profile: OOFHealthProfile,
        failures: list[str],
        metrics: dict[str, Any],
    ) -> None:
        """OOF-03: check top1 hit rate and ROI for anomalies."""
        # Per-race top1: highest score_col horse
        top1_rows = df.loc[
            df.groupby("race_id")[profile.score_col].idxmax()
        ]

        n_races = len(top1_rows)

        # Need kakuteijyuni for hit rate
        if "kakuteijyuni" not in top1_rows.columns:
            return

        hit_rate = float((top1_rows["kakuteijyuni"] == 1).sum() / n_races)
        metrics["top1_hit_rate"] = hit_rate
        metrics["top1_score_col"] = profile.score_col

        # ROI calculation using first return_col as return_unit
        return_col = profile.return_cols[0] if profile.return_cols else None
        odds_col = "confirmed_odds" if "confirmed_odds" in top1_rows.columns else None

        roi = 0.0
        if return_col and return_col in top1_rows.columns:
            total_return = top1_rows[return_col].sum()
            total_bets = n_races
            roi = float(total_return / total_bets) if total_bets > 0 else 0.0
            metrics["top1_roi"] = roi
            metrics["return_col_used"] = return_col

        if n_races >= profile.min_guard_races:
            if hit_rate > profile.max_top1_hit_rate:
                failures.append(
                    f"Top1 hit rate {hit_rate:.1%} > "
                    f"{profile.max_top1_hit_rate:.0%} (OOF-03)"
                )
            if roi > profile.max_top1_roi:
                failures.append(
                    f"Top1 ROI {roi:.2f} > {profile.max_top1_roi:.1f} (OOF-03)"
                )

    def _check_train_valid_overlap(
        self,
        df: pd.DataFrame,
        split_metadata: dict[str, Any],
        failures: list[str],
        metrics: dict[str, Any],
    ) -> None:
        """OOF-02: check train/valid race_id overlap."""
        train_ids = set(split_metadata.get("train_race_ids", []))
        valid_ids = set(split_metadata.get("valid_race_ids", []))
        overlap = train_ids & valid_ids
        metrics["train_valid_overlap_count"] = len(overlap)
        if overlap:
            failures.append(
                f"{len(overlap)} race_ids overlap between train and valid (OOF-02)"
            )

    def generate_manifest(
        self,
        df: pd.DataFrame,
        profile: OOFHealthProfile,
        artifact_hash: str,
        *,
        train_date_range: tuple[str, str] | None = None,
        split_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a D-10 health manifest dict for an OOF artifact."""
        schema_hash, schema_dtype_hash = self._compute_schema_hashes(df)

        # Run validation to get current status
        result = self.validate(
            df,
            profile,
            train_date_range=train_date_range,
            split_metadata=split_metadata,
        )

        fold_col = profile.fold_col
        fold_counts = df[fold_col].value_counts().to_dict()
        fold_race_counts = (
            df.groupby(fold_col)["race_id"].nunique().to_dict()
        )

        # Fold row/race counts with string keys for JSON
        fold_row_counts = {str(k): int(v) for k, v in fold_counts.items()}
        fold_race_counts_str = {str(k): int(v) for k, v in fold_race_counts.items()}

        manifest: dict[str, Any] = {
            "artifact_hash": artifact_hash,
            "artifact_name": profile.artifact_name,
            "artifact_path": f"data/oof/{profile.artifact_name}.parquet",
            "artifact_version": int(
                df["oof_artifact_version"].iloc[0]
                if "oof_artifact_version" in df.columns
                else 0
            ),
            "date_max": str(df["race_date"].max()) if "race_date" in df.columns else "",
            "date_min": str(df["race_date"].min()) if "race_date" in df.columns else "",
            "failures": result["failures"],
            "fold_col": profile.fold_col,
            "fold_count": result.get("fold_count", 0),
            "fold_race_counts": fold_race_counts_str,
            "fold_race_id_uniqueness": result.get("same_race_multiple_fold_count", 0) == 0,
            "fold_row_counts": fold_row_counts,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "race_count": int(df["race_id"].nunique()) if "race_id" in df.columns else 0,
            "row_count": len(df),
            "schema_dtype_hash": schema_dtype_hash,
            "schema_hash": schema_hash,
            "same_race_multiple_fold_count": result.get(
                "same_race_multiple_fold_count", 0
            ),
            "source_code_version": "",
            "source_model_hash": "",
            "source_oof_manifest_path": "",
            "status": result["status"],
            "top1_hit_rate": result.get("top1_hit_rate"),
            "top1_roi": result.get("top1_roi"),
            "top1_score_col": result.get("top1_score_col"),
            "train_date_range": train_date_range,
            "train_valid_overlap_count": result.get("train_valid_overlap_count", 0),
            "validator_version": self.VALIDATOR_VERSION,
            "warnings": result["warnings"],
        }

        # Optional fields
        if "umaban" in df.columns:
            manifest["horse_count"] = int(df["umaban"].nunique())
        else:
            manifest["horse_count"] = None

        if "expected_row_count" in result:
            manifest["expected_row_count"] = result["expected_row_count"]

        if "row_coverage_ratio" in result:
            manifest["row_coverage_ratio"] = result["row_coverage_ratio"]

        if "return_col_used" in result:
            manifest["return_col_used"] = result["return_col_used"]

        return manifest

    @staticmethod
    def _compute_schema_hashes(df: pd.DataFrame) -> tuple[str, str]:
        """D-11: compute schema_hash and schema_dtype_hash."""
        cols_sorted = sorted(df.columns.tolist())
        schema_hash = hashlib.sha256(
            json.dumps(cols_sorted).encode()
        ).hexdigest()

        dtype_pairs = sorted(f"{col}:{df[col].dtype}" for col in df.columns)
        schema_dtype_hash = hashlib.sha256(
            json.dumps(dtype_pairs).encode()
        ).hexdigest()

        return schema_hash, schema_dtype_hash


def _update_index(
    artifact_name: str, manifest_path: str, artifact_hash: str
) -> None:
    """Update data/oof/manifests/index.json with artifact entry."""
    index_path = Path("data/oof/manifests/index.json")
    index_path.parent.mkdir(parents=True, exist_ok=True)

    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            index_data = json.load(f)
    else:
        index_data = {}

    index_data[artifact_name] = manifest_path

    index_path.write_text(
        json.dumps(index_data, sort_keys=True, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_validated_oof(
    artifact_name: str,
    *,
    force_revalidate: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """D-14: consumer-side OOF loading with manifest verification."""
    # Read index to find manifest path
    index_path = Path("data/oof/manifests/index.json")
    with open(index_path, encoding="utf-8") as f:
        index_data = json.load(f)

    if artifact_name not in index_data:
        raise ValueError(f"Artifact '{artifact_name}' not found in index")

    manifest_path = Path(index_data[artifact_name])
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest["status"] != "PASS":
        raise ValueError(
            f"OOF manifest status is {manifest['status']}: "
            f"{manifest.get('failures', [])}"
        )

    artifact_path = Path(manifest["artifact_path"])
    actual_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if actual_hash != manifest["artifact_hash"]:
        raise ValueError(
            f"artifact_hash mismatch: expected {manifest['artifact_hash']}, "
            f"got {actual_hash}"
        )

    df = pd.read_parquet(artifact_path)

    if force_revalidate:
        # Determine profile from artifact_name
        if artifact_name == "oof_predictions":
            profile = OOF_PREDICTIONS_PROFILE
        elif artifact_name == "win_selection_oof":
            profile = WIN_SELECTION_OOF_PROFILE
        else:
            raise ValueError(f"Unknown artifact: {artifact_name}")

        validator = OOFHealthValidator()
        result = validator.validate(df, profile)
        if result["status"] != "PASS":
            raise ValueError(
                f"Force revalidation failed: {result['failures']}"
            )

    return df, manifest
