"""SlackNotifier のテスト"""

from unittest.mock import patch


class TestSlackNotifier:
    def test_send_calls_webhook(self) -> None:
        from monitoring.notifier import SlackNotifier

        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        with patch.object(notifier, "_post") as mock_post:
            mock_post.return_value = True
            result = notifier.send("test message", level="info")
            assert result is True
            mock_post.assert_called_once()

    def test_send_prediction_formats_bets(self) -> None:
        from monitoring.notifier import SlackNotifier

        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        with patch.object(notifier, "_post") as mock_post:
            mock_post.return_value = True
            bets = [
                {
                    "race_id": "2026040510010101",
                    "umaban": 3,
                    "horse_name": "テスト馬",
                    "odds": 2.4,
                    "ev": 1.5,
                    "stake": 100.0,
                },
            ]
            notifier.send_prediction(bets=bets, date="2026-04-05")
            payload = mock_post.call_args[0][0]
            assert "テスト馬" in payload["text"]

    def test_send_daily_result(self) -> None:
        from monitoring.notifier import SlackNotifier

        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        with patch.object(notifier, "_post") as mock_post:
            mock_post.return_value = True
            summary = {
                "date": "2026-04-05",
                "n_bets": 5,
                "n_wins": 2,
                "daily_roi": 1.20,
                "cumulative_roi": 1.10,
                "max_dd": 0.03,
                "bankroll": 101500.0,
            }
            notifier.send_daily_result(summary=summary)
            payload = mock_post.call_args[0][0]
            assert "Paper Trading サマリー" in payload["text"]
            assert "ベット数: 5" in payload["text"]

    def test_send_returns_false_on_error(self) -> None:
        from monitoring.notifier import SlackNotifier

        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        with patch("urllib.request.urlopen", side_effect=Exception("network error")):
            result = notifier.send("test", level="warning")
            assert result is False

    def test_notifier_protocol_compliance(self) -> None:
        from monitoring.notifier import NotifierProtocol, SlackNotifier

        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        assert isinstance(notifier, NotifierProtocol)
