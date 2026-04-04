"""軽量タイミング計測ユーティリティ。"""

import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


class TimingContext:
    """with文でコードブロックの実行時間を計測するコンテキストマネージャ。"""

    def __init__(self, step_name: str) -> None:
        self._step_name = step_name
        self._start: float = 0.0

    def __enter__(self) -> "TimingContext":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed = time.perf_counter() - self._start
        logger.info("[TIMING] %s: %.3fs", self._step_name, elapsed)


def timed(step_name: str) -> Callable[..., Any]:
    """関数の実行時間を計測するデコレータ。"""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with TimingContext(step_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator
