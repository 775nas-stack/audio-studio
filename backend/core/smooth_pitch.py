"""Pitch post-processing utilities."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


MEDIAN_WINDOW = 11
AVERAGE_WINDOW = 5
FRAME_RATE_THRESHOLD = 1000  # Hz; above this we assume a hop based analysis
DEFAULT_HOP_LENGTH = 160  # samples, TorchCREPE default
GAP_LIMIT_MS = 120


def _to_float_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Pitch and confidence arrays must be one-dimensional.")
    return array.copy()


def _nan_rolling_filter(data: np.ndarray, window: int, reducer) -> np.ndarray:
    if window <= 1 or data.size == 0:
        return data.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    result = data.copy()
    for idx in range(data.size):
        start = max(0, idx - half)
        end = min(data.size, idx + half + 1)
        window_vals = data[start:end]
        valid = window_vals[np.isfinite(window_vals)]
        if valid.size:
            result[idx] = reducer(valid)
        else:
            result[idx] = np.nan
    return result


def _remove_octave_jumps(freq: np.ndarray) -> np.ndarray:
    cleaned = freq.copy()
    finite_idx = np.where(np.isfinite(cleaned) & (cleaned > 0))[0]
    if finite_idx.size <= 1:
        return cleaned
    prev_value = cleaned[finite_idx[0]]
    for idx in finite_idx[1:]:
        current = cleaned[idx]
        if current <= 0 or not np.isfinite(current):
            continue
        ratio = current / prev_value if prev_value > 0 else np.nan
        if not np.isfinite(ratio) or ratio <= 0:
            prev_value = current
            continue
        diff = 12.0 * np.log2(ratio)
        while diff > 12.0:
            current /= 2.0
            ratio = current / prev_value
            diff = 12.0 * np.log2(ratio)
        while diff < -12.0:
            current *= 2.0
            ratio = current / prev_value
            diff = 12.0 * np.log2(ratio)
        cleaned[idx] = current
        prev_value = current
    return cleaned


def _interpolate_short_gaps(
    freq: np.ndarray, conf: np.ndarray, max_gap_frames: int
) -> Tuple[np.ndarray, np.ndarray]:
    if max_gap_frames <= 0 or freq.size == 0:
        return freq, conf
    interpolated_freq = freq.copy()
    interpolated_conf = conf.copy()
    n = freq.size
    idx = 0
    while idx < n:
        if np.isnan(interpolated_freq[idx]):
            start = idx
            while idx < n and np.isnan(interpolated_freq[idx]):
                idx += 1
            end = idx
            gap = end - start
            if (
                gap > 0
                and gap <= max_gap_frames
                and start > 0
                and end < n
                and np.isfinite(interpolated_freq[start - 1])
                and np.isfinite(interpolated_freq[end])
            ):
                left_freq = interpolated_freq[start - 1]
                right_freq = interpolated_freq[end]
                left_conf = interpolated_conf[start - 1]
                right_conf = interpolated_conf[end]
                for offset, pos in enumerate(range(start, end), start=1):
                    ratio = offset / (gap + 1)
                    interpolated_freq[pos] = (1 - ratio) * left_freq + ratio * right_freq
                    interpolated_conf[pos] = (1 - ratio) * left_conf + ratio * right_conf
        else:
            idx += 1
    return interpolated_freq, interpolated_conf


def _effective_frame_rate(sr: float) -> float:
    if sr <= 0:
        return 1.0
    if sr <= FRAME_RATE_THRESHOLD:
        return sr
    return sr / DEFAULT_HOP_LENGTH


def smooth_pitch(
    pitch: Iterable[float], confidence: Iterable[float], sr: float
) -> tuple[np.ndarray, np.ndarray]:
    """Clean raw pitch tracks.

    Parameters
    ----------
    pitch
        Iterable of pitch estimates in Hertz. Non-positive or non-finite values
        are treated as gaps.
    confidence
        Iterable of confidence scores aligned with ``pitch``.
    sr
        Analysis rate. When values greater than ``FRAME_RATE_THRESHOLD`` are
        supplied, the function assumes the value is an audio sample rate and
        converts it to a frame rate using ``DEFAULT_HOP_LENGTH``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Cleaned pitch and confidence arrays.
    """

    freq = _to_float_array(pitch)
    conf = _to_float_array(confidence)
    if freq.size != conf.size:
        raise ValueError("Pitch and confidence must have the same length.")

    freq[~np.isfinite(freq) | (freq <= 0)] = np.nan
    conf[~np.isfinite(conf)] = 0.0

    median_filtered = _nan_rolling_filter(freq, MEDIAN_WINDOW, np.median)
    averaged = _nan_rolling_filter(median_filtered, AVERAGE_WINDOW, np.mean)
    dejumped = _remove_octave_jumps(averaged)

    frame_rate = _effective_frame_rate(float(sr))
    max_gap_frames = int(round((GAP_LIMIT_MS / 1000.0) * frame_rate))
    dejumped, conf = _interpolate_short_gaps(dejumped, conf, max_gap_frames)

    conf[~np.isfinite(dejumped)] = 0.0
    return dejumped, conf
