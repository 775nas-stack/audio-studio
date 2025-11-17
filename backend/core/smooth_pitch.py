"""Lightweight smoothing for monophonic pitch tracks."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

MIN_CONFIDENCE = 0.05
MAX_GAP_FRAMES = 3
MEDIAN_WINDOW = 5
MEAN_WINDOW = 3


def _to_array(values, dtype=float) -> np.ndarray:
    return np.asarray(values if values is not None else [], dtype=dtype)


def _interpolate_short_gaps(time: np.ndarray, freq: np.ndarray) -> np.ndarray:
    if time.size == 0:
        return freq
    filled = freq.copy()
    isnan = ~np.isfinite(filled)
    idx = 0
    total = len(filled)
    while idx < total:
        if not isnan[idx]:
            idx += 1
            continue
        start = idx
        while idx < total and isnan[idx]:
            idx += 1
        end = idx
        gap = end - start
        prev_idx = start - 1
        next_idx = end
        if (
            gap <= MAX_GAP_FRAMES
            and prev_idx >= 0
            and next_idx < total
            and np.isfinite(filled[prev_idx])
            and np.isfinite(filled[next_idx])
        ):
            filled[start:end] = np.interp(
                time[start:end],
                [time[prev_idx], time[next_idx]],
                [filled[prev_idx], filled[next_idx]],
            )
            isnan[start:end] = False
    return filled


def _median_filter(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    filtered = np.empty_like(values)
    for idx in range(values.size):
        window_vals = padded[idx : idx + window]
        finite = window_vals[np.isfinite(window_vals)]
        filtered[idx] = np.median(finite) if finite.size else np.nan
    return filtered


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    smoothed = np.empty_like(values)
    for idx in range(values.size):
        window_vals = padded[idx : idx + window]
        finite = window_vals[np.isfinite(window_vals)]
        smoothed[idx] = np.mean(finite) if finite.size else np.nan
    return smoothed


def _sanitize_arrays(track: Dict[str, List[float]]) -> Dict[str, np.ndarray]:
    time = _to_array(track.get("time"))
    freq = _to_array(track.get("frequency"))
    conf = _to_array(track.get("confidence"))
    if time.size == 0 or freq.size == 0:
        return {"time": np.array([]), "freq": np.array([]), "conf": np.array([])}
    length = min(time.size, freq.size)
    time = time[:length]
    freq = freq[:length]
    if conf.size < length:
        if conf.size == 0:
            conf = np.ones(length, dtype=float)
        else:
            conf = np.pad(conf, (0, length - conf.size), mode="edge")
    conf = conf[:length]
    order = np.argsort(time)
    return {
        "time": time[order],
        "freq": freq[order],
        "conf": conf[order],
    }


def smooth_pitch_track(track: Dict[str, List[float]]) -> Dict[str, List[float]]:
    arrays = _sanitize_arrays(track)
    time = arrays["time"]
    freq = arrays["freq"].astype(float)
    conf = arrays["conf"].astype(float)

    if time.size == 0:
        return {
            "time": track.get("time", []),
            "frequency": track.get("frequency", []),
            "confidence": track.get("confidence", []),
            "sr": track.get("sr", 16_000),
            "source": track.get("source", "unknown"),
        }

    voiced = (conf >= MIN_CONFIDENCE) & np.isfinite(freq) & (freq > 0)
    freq = freq.copy()
    freq[~voiced] = np.nan

    freq = _interpolate_short_gaps(time, freq)
    freq = _median_filter(freq, MEDIAN_WINDOW)
    freq = _moving_average(freq, MEAN_WINDOW)

    voiced_out = np.isfinite(freq) & (freq > 0)
    conf_out = conf.copy()
    conf_out[~voiced_out] = 0.0

    print(
        f"[smooth_pitch] Frames in={len(time)} voiced_in={int(np.count_nonzero(voiced))} "
        f"voiced_out={int(np.count_nonzero(voiced_out))}"
    )

    return {
        "time": time.astype(float).tolist(),
        "frequency": np.nan_to_num(freq, nan=0.0).astype(float).tolist(),
        "confidence": conf_out.astype(float).tolist(),
        "sr": track.get("sr", 16_000),
        "source": track.get("source", "unknown"),
    }
