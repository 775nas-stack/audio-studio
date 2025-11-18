"""Shared helper utilities for Offline Audio Studio."""

from __future__ import annotations

from typing import Iterator

import numpy as np

TARGET_SAMPLE_RATE = 16_000
FRAME_HOP = 320  # 20 ms at 16 kHz
FRAME_DURATION = FRAME_HOP / TARGET_SAMPLE_RATE
FMIN = 55.0
FMAX = 1_100.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.45


def time_axis_for_frames(num_frames: int, frame_duration: float = FRAME_DURATION) -> np.ndarray:
    """Return a regularly spaced time axis for a frame sequence."""

    return np.arange(num_frames, dtype=float) * frame_duration


def median_frame_step(time_axis: np.ndarray) -> float:
    """Compute the representative frame duration from a time array."""

    if time_axis.size < 2:
        return FRAME_DURATION
    diffs = np.diff(time_axis)
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return FRAME_DURATION
    return float(np.median(positive))


def frames_for_duration(duration: float, frame_duration: float | None = None) -> int:
    """Convert a duration in seconds into an integer frame count."""

    step = FRAME_DURATION if frame_duration is None else frame_duration
    if step <= 0:
        return 1
    return max(1, int(round(duration / step)))


def iter_boolean_runs(mask: np.ndarray, value: bool = True) -> Iterator[tuple[int, int]]:
    """Yield (start, end) index pairs for contiguous runs of a boolean mask."""

    if mask.ndim != 1:
        raise ValueError("Mask must be 1-D for iter_boolean_runs")
    start: int | None = None
    target = bool(value)
    for idx, flag in enumerate(mask):
        if bool(flag) == target:
            if start is None:
                start = idx
        elif start is not None:
            yield start, idx
            start = None
    if start is not None:
        yield start, mask.size


__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "FMAX",
    "FMIN",
    "FRAME_DURATION",
    "FRAME_HOP",
    "TARGET_SAMPLE_RATE",
    "frames_for_duration",
    "iter_boolean_runs",
    "median_frame_step",
    "time_axis_for_frames",
]
