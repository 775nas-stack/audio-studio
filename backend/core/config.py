"""Runtime configuration flags for the offline engine."""

from __future__ import annotations

import os


def _env_flag(name: str, default: str = "true") -> bool:
    value = os.getenv(name, default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


DEBUG = _env_flag("AUDIO_STUDIO_DEBUG", "true")
