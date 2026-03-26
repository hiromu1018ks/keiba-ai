# src/monitoring/notifier.py
"""通知インタフェース (F-3a)

Slack / ログ / 複合 の通知バックエンド。
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class NotifierProtocol(Protocol):
    def send(self, message: str, level: str = "info") -> bool: ...


class LoggingNotifier:
    """ログ出力のみの通知（開発/テスト用）"""

    def send(self, message: str, level: str = "info") -> bool:
        """ログにメッセージを出力

        Args:
            message: 通知メッセージ
            level: "info", "warning", "critical"

        Returns:
            常に True
        """
        log_fn = {
            "info": logger.info,
            "warning": logger.warning,
            "critical": logger.critical,
        }.get(level, logger.info)
        log_fn(f"[NOTIFY/{level.upper()}] {message}")
        return True


class CompositeNotifier:
    """複数の通知バックエンドに配信する。

    1つでも成功すれば True。全て失敗すれば False。
    """

    def __init__(self, notifiers: list[NotifierProtocol]) -> None:
        self._notifiers = notifiers

    def send(self, message: str, level: str = "info") -> bool:
        """全通知先にメッセージを配信

        Returns:
            1つでも成功すれば True
        """
        any_success = False
        for notifier in self._notifiers:
            try:
                if notifier.send(message, level=level):
                    any_success = True
            except Exception:
                logger.exception("Notifier failed")
        return any_success
