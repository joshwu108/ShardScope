"""Shared test setup.

``examples/`` holds runnable scripts rather than an installed package, so it
is put on ``sys.path`` here. The scripts are written to do no work at import
time (see examples/train_single.py) precisely so tests can drive a handful of
steps in-process instead of spawning subprocesses.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"

if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
