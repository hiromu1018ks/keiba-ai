# tests/test_notifier.py
"""src/monitoring/notifier.py のテスト"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestLoggingNotifier:
    def test_send_logs_info_message(self, caplog: pytest.LogCaptureFixture) -> None:
        """INFO レベルメッセージをログ出力"""
        from monitoring.notifier import LoggingNotifier

        caplog.set_level("INFO")
        notifier = LoggingNotifier()
        result = notifier.send("テストメッセージ", level="info")

        assert result is True
        assert "テストメッセージ" in caplog.text

    def test_send_logs_warning_message(self, caplog: pytest.LogCaptureFixture) -> None:
        """WARNING レベルメッセージをログ出力"""
        from monitoring.notifier import LoggingNotifier

        notifier = LoggingNotifier()
        notifier.send("警告メッセージ", level="warning")

        assert "警告メッセージ" in caplog.text

    def test_send_logs_critical_message(self, caplog: pytest.LogCaptureFixture) -> None:
        """CRITICAL レベルメッセージをログ出力"""
        from monitoring.notifier import LoggingNotifier

        notifier = LoggingNotifier()
        notifier.send("緊急メッセージ", level="critical")

        assert "緊急メッセージ" in caplog.text


class TestCompositeNotifier:
    def test_send_dispatches_to_all_notifiers(self) -> None:
        """全ての通知先に配信する"""
        mock1 = MagicMock()
        mock1.send.return_value = True
        mock2 = MagicMock()
        mock2.send.return_value = True

        from monitoring.notifier import CompositeNotifier

        notifier = CompositeNotifier([mock1, mock2])
        notifier.send("テスト", level="info")

        mock1.send.assert_called_once_with("テスト", level="info")
        mock2.send.assert_called_once_with("テスト", level="info")

    def test_send_returns_true_if_any_succeeds(self) -> None:
        """1つでも成功すれば True"""
        mock_fail = MagicMock()
        mock_fail.send.return_value = False
        mock_ok = MagicMock()
        mock_ok.send.return_value = True

        from monitoring.notifier import CompositeNotifier

        notifier = CompositeNotifier([mock_fail, mock_ok])
        result = notifier.send("テスト", level="info")

        assert result is True

    def test_send_returns_false_if_all_fail(self) -> None:
        """全て失敗すれば False"""
        mock1 = MagicMock()
        mock1.send.return_value = False
        mock2 = MagicMock()
        mock2.send.return_value = False

        from monitoring.notifier import CompositeNotifier

        notifier = CompositeNotifier([mock1, mock2])
        result = notifier.send("テスト", level="info")

        assert result is False
