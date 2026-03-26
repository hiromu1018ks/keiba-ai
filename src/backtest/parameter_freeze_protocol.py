"""パラメータ凍結プロトコル (Rule 7)

OOS 期間中のモデル不变性を保証する。
freeze() でスナップショットを取得し、verify() で変更を検出。
frozen_period() コンテキストマネージャで OOS 期間を定義。
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from collections.abc import Iterator
from contextlib import contextmanager
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
        except (pickle.PicklingError, TypeError):
            # pickle 不可なオブジェクトは repr のハッシュを使用
            return hashlib.sha256(repr(obj).encode()).digest()
