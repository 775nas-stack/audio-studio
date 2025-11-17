from __future__ import annotations

import numpy as np

from .smoothing_advanced import smooth_track_advanced
from .types import NoMelodyError, PitchTrack


def _interpolate_small_gaps(freq: np.ndarray, max_gap_frames: int = 3) -> np.ndarray:
    data = freq.copy()
    n = len(data)
    i = 0
    while i < n:
        if np.isnan(data[i]):
            start = i
            while i < n and np.isnan(data[i]):
                i += 1
            end = i
            gap = end - start
            if (
                gap > 0
                and gap <= max_gap_frames
                and start > 0
                and end < n
                and np.isfinite(data[start - 1])
                and np.isfinite(data[end])
            ):
                left = data[start - 1]
                right = data[end]
                for offset, idx in enumerate(range(start, end), start=1):
                    ratio = offset / (gap + 1)
                    data[idx] = (1 - ratio) * left + ratio * right
        else:
            i += 1
    return data


def _rolling_operation(values: np.ndarray, fn) -> np.ndarray:
    result = values.copy()
    for idx in range(len(values)):
        window = values[max(0, idx - 1) : min(len(values), idx + 2)]
        valid = window[np.isfinite(window)]
        if len(valid) == 0:
            continue
        result[idx] = fn(valid)
    return result


def smooth_track_basic(track: PitchTrack) -> PitchTrack:
    freq = track.frequency.copy()
    conf = track.confidence.copy()

    freq[conf < 0.05] = np.nan
    freq = _interpolate_small_gaps(freq)
    freq = _rolling_operation(freq, np.median)
    freq = _rolling_operation(freq, np.mean)

    smoothed = PitchTrack(time=track.time, frequency=freq, confidence=conf, engine=track.engine)

    if smoothed.finite_count() < 10:
        raise NoMelodyError("No stable monophonic melody detected.")

    return smoothed


def smooth_track(track: PitchTrack, mode: str = "advanced") -> PitchTrack:
    if mode == "basic":
        return smooth_track_basic(track)
    return smooth_track_advanced(track)
