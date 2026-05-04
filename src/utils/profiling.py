"""pyinstrumentベースのプロファイリングユーティリティ。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ProfileContext:
    """pyinstrumentプロファイリングコンテキストマネージャー。

    --profileフラグ指定時のみプロファイリングを実行。
    未指定時はno-op (オーバーヘッドなし)。

    Usage:
        with ProfileContext(enabled=args.profile, label='backtest'):
            run_backtest(...)
    """

    def __init__(self, enabled: bool = False, label: str = "profile") -> None:
        self._enabled = enabled
        self._label = label
        self._profiler: Any = None  # pyinstrument.Profiler or None

    def __enter__(self) -> "ProfileContext":
        if self._enabled:
            try:
                from pyinstrument import Profiler

                self._profiler = Profiler()
                self._profiler.start()
            except ImportError:
                logger.warning(
                    "pyinstrument is not installed. "
                    "Install with: pip install pyinstrument",
                )
                self._enabled = False
        return self

    def __exit__(self, *args: Any) -> None:
        if self._profiler is not None:
            self._profiler.stop()
            # テキスト出力 -> stdout
            text_output = self._profiler.output_text(unicode=True, color=True)
            print(text_output)
            # HTML出力 -> ファイル
            output_dir = Path("data/profiles")
            output_dir.mkdir(parents=True, exist_ok=True)
            html_path = output_dir / f"{self._label}.html"
            self._profiler.write_html(str(html_path))
            logger.info("Profile saved: %s", html_path)
