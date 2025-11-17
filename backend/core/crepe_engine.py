from __future__ import annotations

import logging

import numpy as np

from ..vendor import crepe
from .types import PitchTrack

LOGGER = logging.getLogger(__name__)


def run_crepe(audio: np.ndarray, sr: int) -> PitchTrack:
    """Run the official CREPE model and collect the raw contour."""

    try:
        time, frequency, confidence, _ = crepe.predict(
            audio,
            sr=sr,
            model_capacity="tiny",
            step_size=10,
            verbose=0,
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("CREPE inference failed: %s", exc)
        time = np.array([], dtype=float)
        frequency = np.array([], dtype=float)
        confidence = np.array([], dtype=float)

    return PitchTrack(time=time, frequency=frequency, confidence=confidence, engine="crepe")
