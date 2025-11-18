from __future__ import annotations

import logging
from pathlib import Path

from .pitch_pipeline import extract_unified_pitch
from .types import PitchTrack

LOGGER = logging.getLogger(__name__)


def extract_pitch(audio: np.ndarray, sr: int, engine: str | None = None, debug_dir: Path | None = None) -> PitchTrack:
    """Run the routing pipeline with optional engine preference."""

    track = extract_unified_pitch(audio, sr, requested_engine=engine, debug_dir=debug_dir)
    LOGGER.info("Unified pitch extraction complete using %s", track.engine)
    return track
