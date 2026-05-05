"""パラメータ凍結プロトコル (Rule 7)

OOS 期間中のモデル不变性を保証する。
freeze() でスナップショットを取得し、verify() で変更を検出。
frozen_period() コンテキストマネージャで OOS 期間を定義。
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from domain.models import TrainedModelsV5

logger = logging.getLogger(__name__)


class ParameterFreezeProtocol:
    """パラメータ凍結プロトコル

    Rule 7: out-of-sample期間ではパラメータ変更を一切行わない。

    使い方:
        protocol = ParameterFreezeProtocol(models)
        protocol.freeze()
        # ... OOS evaluation ...
        result = protocol.verify()  # {"passed": bool, "message": str}

    またはコンテキストマネージャ:
        with protocol.frozen_period():
            # ... OOS evaluation ...
            # 終了時に自動 verify → 変更があれば RuntimeError
    """

    def __init__(self, models: TrainedModelsV5) -> None:
        self.models = models
        self._snapshot: bytes | None = None
        self._frozen = False

    def freeze(self) -> None:
        """現在のモデル状態のスナップショットを取得

        pickle シリアライズのハッシュで状態を記録。
        """
        self._snapshot = self._serialize(self.models)
        self._frozen = True
        logger.info("Parameters frozen (Rule 7)")

    def verify(self) -> dict[str, Any]:
        """モデル状態が凍結時から変更されていないことを検証

        Returns:
            {"passed": bool, "message": str}
        """
        if not self._frozen:
            return {
                "passed": False,
                "message": "freeze() が呼ばれていません",
            }

        current = self._serialize(self.models)
        if current == self._snapshot:
            return {
                "passed": True,
                "message": "Parameters unchanged (Rule 7 OK)",
            }
        return {
            "passed": False,
            "message": "Parameters changed during frozen period (Rule 7 VIOLATION)",
        }

    @contextmanager
    def frozen_period(self) -> Iterator[None]:
        """OOS 期間のコンテキストマネージャ

        終了時に自動 verify。変更があれば RuntimeError を送出。
        """
        self.freeze()
        try:
            yield
        finally:
            result = self.verify()
            if not result["passed"]:
                raise RuntimeError(result["message"])
            self._frozen = False
            logger.info("Frozen period ended, parameters verified OK")

    @staticmethod
    def _serialize(obj: Any) -> bytes:
        """オブジェクトをシリアライズしてハッシュ化"""
        try:
            data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            return hashlib.sha256(data).digest()
        except Exception:
            # pickleはAttributeError, RuntimeError等も発生しうる。
            # pickle不可なオブジェクトはreprのハッシュを使用
            return hashlib.sha256(repr(obj).encode()).digest()


def save_strategy_manifest(
    params: dict[str, Any],
    path: Path,
) -> str:
    """戦略パラメータをJSON manifestとして保存 + SHA256ハッシュ返却

    D-13: Optuna最適化完了後のパラメータ保存用。
    JSON形式で人間可読 + SHA256で改ざん検知。

    Args:
        params: 戦略パラメータdict (Optuna best_params等)
        path: manifest保存先パス

    Returns:
        SHA256ハッシュ文字列
    """
    data = json.dumps(params, sort_keys=True, indent=2)
    sha256 = hashlib.sha256(data.encode()).hexdigest()
    manifest = {"params": params, "sha256": sha256}
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"Strategy manifest saved: {path} (sha256={sha256[:8]}...)")
    return sha256


def verify_strategy_manifest(path: Path) -> dict[str, Any]:
    """JSON manifestのSHA256照合

    D-14: テスト期間バックテスト実行前にmanifestを読み込みハッシュ照合。
    不一致時はValueError送出。

    Args:
        path: manifestファイルパス

    Returns:
        検証済みパラメータdict

    Raises:
        ValueError: SHA256ハッシュ不一致
        FileNotFoundError: manifestファイル不在
    """
    if not path.exists():
        raise FileNotFoundError(f"Strategy manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = manifest["sha256"]
    actual = hashlib.sha256(
        json.dumps(manifest["params"], sort_keys=True, indent=2).encode()
    ).hexdigest()
    if actual != expected:
        raise ValueError(
            f"Strategy manifest hash mismatch: expected={expected[:8]}... "
            f"actual={actual[:8]}... "
            f"Parameters may have been tampered with."
        )
    logger.info(f"Strategy manifest verified: {path} (sha256={actual[:8]}...)")
    return manifest["params"]


def load_and_freeze_strategy(
    params: dict[str, Any],
    manifest_path: Path,
) -> None:
    """パラメータを保存し、manifestが既に存在する場合は検証

    D-14: Optuna完了後の自動生成用。
    manifestが存在しない場合は新規保存。
    manifestが存在する場合はSHA256照合を実行。

    Args:
        params: 保存するパラメータ
        manifest_path: manifestパス
    """
    if manifest_path.exists():
        saved = verify_strategy_manifest(manifest_path)
        if saved != params:
            logger.warning(
                "Strategy params differ from manifest. "
                "Optuna may have produced different results."
            )
    else:
        save_strategy_manifest(params, manifest_path)
