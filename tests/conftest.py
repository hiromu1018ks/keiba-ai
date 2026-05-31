"""Test configuration for keiba-ai tests.

Adds scripts/ to sys.path so scripts modules are importable in tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add scripts/ to sys.path for script module imports
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
