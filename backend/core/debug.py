from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import DEBUG


def ensure_debug_dir(debug_dir: Path | None) -> Path | None:
    if debug_dir is None:
        return None
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def write_debug_file(debug_dir: Path | None, name: str, payload: Any) -> None:
    if not DEBUG or debug_dir is None:
        return
    ensure_debug_dir(debug_dir)
    path = debug_dir / name
    path.write_text(json.dumps(payload, indent=2))
