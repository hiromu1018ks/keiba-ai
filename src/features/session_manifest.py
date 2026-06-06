"""PT セッションマニフェスト (D-06, D-09).

SessionManifest は PT 実行の監査記録を保持する。
MLflow run ID、学習期間、コードハッシュ (git SHA)、FeatureManifest ハッシュ、
PFP 検証結果を記録し、session_manifest.json にアトミック書き込みする。

get_code_version() は git の状態 (commit SHA, dirty 検出) を取得する。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def get_code_version() -> dict[str, Any]:
    """Git コミット SHA と dirty 状態を取得 (D-06).

    Returns:
        {
            "commit_sha": str,
            "git_dirty": bool,
            "dirty_diff_hash": str | None,
            "untracked_files": list[str],
        }

    Raises:
        RuntimeError: git が利用不可、または git リポジトリ内ではない場合。
    """
    # git コマンドの可用性を事前チェック (WR-03)
    git_path = shutil.which("git")
    if git_path is None:
        raise RuntimeError("git command not found on PATH")

    # コミット SHA
    try:
        result = subprocess.run(
            [git_path, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        commit_sha = result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(f"git rev-parse HEAD failed: {e}") from e

    # Dirty 状態 (src/, scripts/, config/)
    try:
        status_result = subprocess.run(
            [git_path, "status", "--porcelain", "--", "src/", "scripts/", "config/"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(f"git status failed: {e}") from e

    status_lines = [line for line in status_result.stdout.strip().splitlines() if line.strip()]
    git_dirty = len(status_lines) > 0

    # Untracked files
    untracked_files = [
        line[3:] for line in status_lines if line.startswith("?? ")
    ]

    # Dirty diff hash
    dirty_diff_hash: str | None = None
    if git_dirty:
        try:
            diff_result = subprocess.run(
                [git_path, "diff", "src/", "scripts/", "config/"],
                capture_output=True,
                text=True,
                check=True,
            )
            diff_output = diff_result.stdout
            if diff_output:
                dirty_diff_hash = hashlib.sha256(diff_output.encode("utf-8")).hexdigest()
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Failed to compute dirty diff hash")

    return {
        "commit_sha": commit_sha,
        "git_dirty": git_dirty,
        "dirty_diff_hash": dirty_diff_hash,
        "untracked_files": untracked_files,
    }


def compute_obf_config_hash(roi_threshold: float) -> str:
    """OddsBandFilter 設定の SHA256 hash を算出 (D-08).

    BANDS 境界 + roi_threshold を結合して hash 化する。
    PT ではキャリブレーションを行わないため、この hash で設定不変性を担保する。
    """
    from betting.odds_band_filter import OddsBandFilter

    parts = ",".join(f"{lo}-{hi}" for lo, hi in OddsBandFilter.BANDS)
    raw = f"{parts}|{roi_threshold}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class SessionManifest:
    """PT 実行記録 (D-09).

    セッションの開始時に生成し、実行中に各種 ID/ハッシュを記録、
    終了時に status を設定して session_manifest.json に書き込む。
    """

    session_id: str
    prediction_date: str
    code_version: dict[str, Any] = field(default_factory=dict)
    model_run_id: str = ""
    training_start: str = ""
    training_end: str = ""
    manifest_hash: str = ""
    pfp_result: dict[str, Any] = field(default_factory=dict)
    status: str = "started"
    exit_code: int = 0
    # STR-01: Strategy alignment fields
    betting_target: str = ""
    betting_mode: str = ""
    strategy_manifest_path: str = ""
    strategy_manifest_sha256: str = ""
    # D-08: OddsBandFilter metadata
    odds_band_filter_metadata: dict[str, Any] = field(default_factory=dict)
    # LIV-03: Live data metadata
    live_data: dict[str, Any] = field(default_factory=dict)

    def set_code_version(self, version: dict[str, Any]) -> None:
        """Git SHA と dirty 状態を記録."""
        self.code_version = version

    def set_model_identity(
        self,
        run_id: str,
        training_start: str,
        training_end: str,
        manifest_hash: str,
    ) -> None:
        """MLflow run ID、学習期間、FeatureManifest ハッシュを記録."""
        self.model_run_id = run_id
        self.training_start = training_start
        self.training_end = training_end
        self.manifest_hash = manifest_hash

    def set_pfp_result(self, result: dict[str, Any]) -> None:
        """PFP 検証結果を記録."""
        self.pfp_result = result

    def set_strategy_params(
        self,
        betting_target: str,
        betting_mode: str,
        strategy_manifest_path: str,
        strategy_manifest_sha256: str,
    ) -> None:
        """STR-01: 戦略パラメータを一括設定."""
        self.betting_target = betting_target
        self.betting_mode = betting_mode
        self.strategy_manifest_path = strategy_manifest_path
        self.strategy_manifest_sha256 = strategy_manifest_sha256

    def set_obf_metadata(
        self,
        calibration_data_end_date: str,
        roi_threshold: float,
        excluded_bands: set[str],
        config_hash: str,
    ) -> None:
        """D-08: OddsBandFilter の4メタデータを記録."""
        self.odds_band_filter_metadata = {
            "calibration_data_end_date": calibration_data_end_date,
            "roi_threshold": roi_threshold,
            "excluded_bands": sorted(excluded_bands),
            "config_hash": config_hash,
        }

    def set_live_data(
        self,
        source: str,
        measured_at: str,
        fetched_at: str,
        html_hash: str,
        venue_codes: list[str],
    ) -> None:
        """LIV-03: ライブトラック条件取得メタデータを記録."""
        self.live_data = {
            "source": source,
            "measured_at": measured_at,
            "fetched_at": fetched_at,
            "html_hash": html_hash,
            "venue_codes": venue_codes,
        }

    def set_status(self, status: str, exit_code: int = 0) -> None:
        """完了ステータスと終了コードを記録."""
        self.status = status
        self.exit_code = exit_code

    def to_dict(self) -> dict[str, Any]:
        """完全なマニフェストをシリアライズ可能な dict で返す."""
        return {
            "session_id": self.session_id,
            "prediction_date": self.prediction_date,
            "code_version": self.code_version,
            "model_run_id": self.model_run_id,
            "training_start": self.training_start,
            "training_end": self.training_end,
            "manifest_hash": self.manifest_hash,
            "pfp_result": self.pfp_result,
            "status": self.status,
            "exit_code": self.exit_code,
            "betting_target": self.betting_target,
            "betting_mode": self.betting_mode,
            "strategy_manifest_path": self.strategy_manifest_path,
            "strategy_manifest_sha256": self.strategy_manifest_sha256,
            "odds_band_filter_metadata": self.odds_band_filter_metadata,
            "live_data": self.live_data,
        }

    @property
    def is_dirty(self) -> bool:
        """コードが dirty 状態かどうか."""
        return bool(self.code_version.get("git_dirty", False))


def write_session_manifest(manifest: SessionManifest, path: Path) -> None:
    """SessionManifest を JSON ファイルにアトミック書き込み (D-09).

    アトミック性は tempfile.NamedTemporaryFile + os.replace で担保。

    Args:
        manifest: 書き込む SessionManifest。
        path: 書き込み先パス。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(manifest.to_dict(), indent=2, default=str, ensure_ascii=False)

    # アトミック書き込み: temp file → os.replace
    fd, tmp_path = tempfile.mkstemp(
        suffix=".json",
        prefix=".session_manifest_",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp_path, str(path))
    except BaseException:
        # クリーンアップ: temp file が残っていれば削除
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    logger.info("Session manifest written: %s", path)
