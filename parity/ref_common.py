"""Shared paths for optional PyTorch/VMAS reference generators."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def vmas_root() -> Path:
    raw = os.environ.get("VMAS_ROOT")
    if not raw:
        raise RuntimeError(
            "VMAS_ROOT is required for PyTorch reference generation; "
            "point it at VectorizedMultiAgentSimulator"
        )
    root = Path(raw).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def output_path(name: str) -> Path:
    return Path(__file__).resolve().with_name(name)
