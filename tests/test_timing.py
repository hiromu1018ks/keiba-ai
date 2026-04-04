"""タイミング計測ユーティリティのテスト。"""

from __future__ import annotations

import logging
import time

from utils.timing import TimingContext, timed


def test_timed_context_manager_logs_elapsed(caplog):
    """TimingContext logs elapsed time with [TIMING] prefix"""
    with caplog.at_level(logging.INFO):
        with TimingContext("my_step"):
            time.sleep(0.01)

    assert any("[TIMING] my_step:" in r.message for r in caplog.records)
    timing_records = [r for r in caplog.records if "[TIMING]" in r.message]
    assert len(timing_records) == 1
    elapsed = float(timing_records[0].message.split(":")[-1].strip().rstrip("s"))
    assert elapsed >= 0.01


def test_timed_decorator_logs_elapsed(caplog):
    """@timed decorator logs function execution time"""

    @timed("decorated_func")
    def slow_func():
        time.sleep(0.01)
        return 42

    with caplog.at_level(logging.INFO):
        result = slow_func()

    assert result == 42
    assert any("[TIMING] decorated_func:" in r.message for r in caplog.records)
