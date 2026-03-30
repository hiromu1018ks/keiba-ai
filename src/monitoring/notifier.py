# src/monitoring/notifier.py
"""通知インタフェース (F-3a)

Slack / ログ / 複合 の通知バックエンド。
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any, Protocol, runtime_checkable

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


class SlackNotifier:
    """Slack Incoming Webhook 通知 (NotifierProtocol 準拠 + Paper Trading専用メソッド)"""

    def __init__(self, webhook_url: str) -> None:
        self._webhook_url = webhook_url

    def send(self, message: str, level: str = "info") -> bool:
        """NotifierProtocol 準拠: 汎用メッセージ送信"""
        payload = {"text": f"[{level.upper()}] {message}"}
        return self._post(payload)

    def send_prediction(self, bets: list[dict[str, Any]], date: str) -> bool:
        """ベット推薦を Slack に通知"""
        if not bets:
            return True
        lines = [f"*Paper Trading 予測 — {date}*\n"]
        for b in bets[:10]:  # 最大10件
            lines.append(
                f"  #{b['umaban']} {b.get('horse_name', '?')} "
                f"オッズ={b['odds']:.1f} EV={b['ev']:.2f}"
            )
        if len(bets) > 10:
            lines.append(f"  ...他 {len(bets) - 10} 件")
        return self._post({"text": "\n".join(lines)})

    def send_daily_result(self, summary: dict[str, Any]) -> bool:
        """日次サマリーを通知"""
        lines = [
            f"*Paper Trading サマリー — {summary['date']}*",
            f"  ベット数: {summary['n_bets']} / 的中: {summary['n_wins']}",
            f"  日次ROI: {summary['daily_roi']:.1%}",
            f"  累積ROI: {summary['cumulative_roi']:.1%}",
            f"  Max DD: {summary['max_dd']:.1%}",
            f"  資金: ¥{summary['bankroll']:,.0f}",
        ]
        return self._post({"text": "\n".join(lines)})

    def _post(self, payload: dict[str, Any]) -> bool:
        """Slack Webhook に POST"""
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self._webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return bool(resp.status == 200)
        except Exception:
            logger.exception("Slack通知失敗")
            return False
