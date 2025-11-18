from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .pitch_pipeline import PitchPipelineResult, extract_unified_pitch

LOGGER = logging.getLogger(__name__)


def extract_pitch(
    audio: np.ndarray,
    sr: int,
    engine: str | None = None,
    debug_dir: Path | None = None,
) -> PitchPipelineResult:
    """Run the routing pipeline with optional engine preference."""

    result = extract_unified_pitch(audio, sr, requested_engine=engine, debug_dir=debug_dir)
    LOGGER.info("Unified pitch extraction complete using %s", result.track.engine)
    return result
