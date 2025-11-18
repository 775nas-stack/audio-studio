"""Librosa PYIN backup engine."""

from __future__ import annotations

import logging

import librosa
import numpy as np

from .types import PitchTrack
from .utils import FRAME_HOP, time_axis_for_frames

LOGGER = logging.getLogger(__name__)

FMIN = librosa.note_to_hz("C2")
FMAX = librosa.note_to_hz("C6")
FRAME_LENGTH = 1024


def run_pyin(audio: np.ndarray, sr: int) -> PitchTrack:
    """Run PYIN on CPU as a last-resort monophonic estimator."""

    try:
        f0, voiced_flag, prob = librosa.pyin(
            audio,
            fmin=FMIN,
            fmax=FMAX,
            sr=sr,
            frame_length=FRAME_LENGTH,
            hop_length=FRAME_HOP,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.exception("PYIN inference failed: %s", exc)
        return PitchTrack(
            time=np.array([], dtype=float),
            frequency=np.array([], dtype=float),
            confidence=np.array([], dtype=float),
            engine="pyin",
        )

    if prob is None:
        prob = voiced_flag.astype(float)

    time = time_axis_for_frames(len(f0))
    voiced = voiced_flag.astype(bool)
    freq = np.where(voiced, f0, np.nan)
    conf = np.where(voiced, prob, 0.0)

    return PitchTrack(time=time, frequency=freq, confidence=conf, engine="pyin")


__all__ = ["run_pyin"]
