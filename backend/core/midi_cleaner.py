"""Helpers for preparing smoothed pitch tracks for MIDI export."""

from __future__ import annotations

import numpy as np

from .types import NoMelodyError, PitchTrack
from .utils import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    frames_for_duration,
    iter_boolean_runs,
    median_frame_step,
)

MIN_SEGMENT_DURATION = 0.12  # seconds
MAX_SILENCE_GAP = 0.06  # seconds


def _copy_track_arrays(track: PitchTrack) -> tuple[np.ndarray, np.ndarray]:
    freq = track.frequency.astype(float).copy()
    conf = track.confidence.astype(float).copy()
    return freq, conf


def _prune_short_segments(mask: np.ndarray, freq: np.ndarray, conf: np.ndarray, min_frames: int) -> None:
    for start, end in iter_boolean_runs(mask):
        if end - start < min_frames:
            freq[start:end] = np.nan
            conf[start:end] = 0.0
            mask[start:end] = False


def _bridge_small_gaps(mask: np.ndarray, freq: np.ndarray, conf: np.ndarray, max_gap_frames: int) -> None:
    for start, end in iter_boolean_runs(~mask):
        length = end - start
        if length == 0 or length > max_gap_frames:
            continue
        left_idx = start - 1
        right_idx = end
        if left_idx < 0 or right_idx >= freq.size:
            continue
        left = freq[left_idx]
        right = freq[right_idx]
        if not (np.isfinite(left) and np.isfinite(right)):
            continue
        interp = np.linspace(left, right, length + 2, dtype=float)[1:-1]
        freq[start:end] = interp
        conf[start:end] = float((conf[left_idx] + conf[right_idx]) / 2.0)
        mask[start:end] = True


def clean_track_for_midi(track: PitchTrack) -> PitchTrack:
    """Remove noisy blips and bridge very small gaps before segmentation."""

    freq, conf = _copy_track_arrays(track)
    if freq.size == 0:
        raise NoMelodyError("No stable monophonic melody detected.")

    frame_step = median_frame_step(track.time)
    min_frames = frames_for_duration(MIN_SEGMENT_DURATION, frame_step)
    gap_frames = frames_for_duration(MAX_SILENCE_GAP, frame_step)

    voiced = np.isfinite(freq) & (conf >= DEFAULT_CONFIDENCE_THRESHOLD)
    freq[~voiced] = np.nan
    conf[~voiced] = 0.0

    _prune_short_segments(voiced, freq, conf, min_frames)
    _bridge_small_gaps(voiced, freq, conf, gap_frames)

    cleaned = PitchTrack(
        time=track.time,
        frequency=freq,
        confidence=conf,
        engine=track.engine,
        activation=track.activation,
        loudness=track.loudness,
        sources=track.sources,
    )

    if cleaned.finite_count() < 4:
        raise NoMelodyError("No stable monophonic melody detected.")

    return cleaned


__all__ = ["clean_track_for_midi"]
