"""Pitch smoothing utilities for the offline studio pipeline."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def _median_filter(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1 or len(values) < 2:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    filtered = np.empty_like(values)
    for idx in range(len(values)):
        filtered[idx] = np.median(padded[idx : idx + window])
    return filtered


def _suppress_jumps(values: np.ndarray, cents_threshold: float = 200.0) -> np.ndarray:
    if len(values) == 0:
        return values
    stabilized = values.copy()
    last = stabilized[0]
    for i in range(1, len(values)):
        current = stabilized[i]
        if last <= 0 or current <= 0:
            last = current
            continue
        cents = 1200.0 * np.log2(current / last)
        if abs(cents) > cents_threshold:
            stabilized[i] = last
        else:
            last = current
    return stabilized


def _trim_silence(track: Dict[str, List[float]], min_confidence: float = 0.5) -> Dict[str, List[float]]:
    time = np.array(track["time"], dtype=float)
    freq = np.array(track["frequency"], dtype=float)
    conf = np.array(track["confidence"], dtype=float)
    mask = conf >= min_confidence
    return {
        "time": time[mask].astype(float).tolist(),
        "frequency": freq[mask].astype(float).tolist(),
        "confidence": conf[mask].astype(float).tolist(),
        "sr": track.get("sr", 16_000),
    }


def smooth_pitch_track(track: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """Apply median filtering, jump suppression and silence trimming."""

    trimmed = _trim_silence(track)
    freq = np.array(trimmed["frequency"], dtype=float)
    if freq.size == 0:
        return trimmed

    median_filtered = _median_filter(freq, window=7)
    stabilized = _suppress_jumps(median_filtered, cents_threshold=150.0)

    return {
        "time": trimmed["time"],
        "frequency": stabilized.astype(float).tolist(),
        "confidence": trimmed["confidence"],
        "sr": trimmed.get("sr", 16_000),
    }
