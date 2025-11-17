from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .crepe_engine import run_crepe
from .pyin_engine import run_pyin
from .types import NoMelodyError, PitchTrack

LOGGER = logging.getLogger(__name__)


@dataclass
class TrackScore:
    track: PitchTrack
    finite_frames: int
    median_confidence: float
    median_jump: float

    @property
    def score_tuple(self) -> Tuple[float, float, float]:
        return (
            self.finite_frames,
            self.median_confidence,
            -self.median_jump,
        )


def _score_track(track: PitchTrack) -> TrackScore:
    finite_mask = track.finite_mask()
    finite_frames = int(finite_mask.sum())
    if finite_frames == 0:
        return TrackScore(track, 0, 0.0, float("inf"))

    usable_freq = track.frequency[finite_mask]
    usable_conf = track.confidence[finite_mask]

    if len(usable_freq) > 1:
        jumps = np.abs(np.diff(usable_freq))
        median_jump = float(np.median(jumps))
    else:
        median_jump = 0.0

    median_conf = float(np.median(usable_conf))
    return TrackScore(track, finite_frames, median_conf, median_jump)


def extract_pitch(audio: np.ndarray, sr: int) -> PitchTrack:
    """Run both engines and select the most trustworthy contour."""

    crepe_track = run_crepe(audio, sr)
    pyin_track = run_pyin(audio, sr)

    scores = [_score_track(crepe_track), _score_track(pyin_track)]
    best = max(scores, key=lambda s: s.score_tuple)

    if best.finite_frames == 0:
        raise NoMelodyError("No stable monophonic melody detected.")

    LOGGER.info(
        "Pitch engine selected: %s (frames=%d, median_conf=%.3f, median_jump=%.2f)",
        best.track.engine,
        best.finite_frames,
        best.median_confidence,
        best.median_jump,
    )

    return best.track
