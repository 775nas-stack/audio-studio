"""Pitch smoothing utilities for the offline studio pipeline."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


MIN_FREQ = 65.0
MAX_FREQ = 1500.0
MAX_GAP_SECONDS = 0.12
MEDIAN_WINDOW = 5
MEAN_WINDOW = 5
JUMP_LIMIT_CENTS = 250.0


def _median_filter(values: np.ndarray, window: int = MEDIAN_WINDOW) -> np.ndarray:
    if window <= 1 or values.size < 2:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    filtered = np.empty_like(values)
    for idx in range(values.size):
        filtered[idx] = np.median(padded[idx : idx + window])
    return filtered


def _moving_average(values: np.ndarray, window: int = MEAN_WINDOW) -> np.ndarray:
    if window <= 1 or values.size < 2:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="same")
    return smoothed.astype(values.dtype)


def _limit_jumps(values: np.ndarray, cents_threshold: float = JUMP_LIMIT_CENTS) -> np.ndarray:
    if values.size == 0:
        return values
    stabilized = values.copy()
    last = stabilized[0]
    for idx in range(1, values.size):
        current = stabilized[idx]
        if last <= 0 or current <= 0:
            last = current
            continue
        cents = 1200.0 * np.log2(current / last)
        if abs(cents) > cents_threshold:
            stabilized[idx] = last
        else:
            last = current
    return stabilized


def _estimate_frame_duration(time: np.ndarray) -> float:
    if time.size < 2:
        return 0.0
    diffs = np.diff(time)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.0
    return float(np.median(diffs))


def _fill_short_gaps(
    time: np.ndarray, freq: np.ndarray, voiced: np.ndarray, max_gap_seconds: float
) -> np.ndarray:
    if time.size == 0:
        return freq
    frame_duration = _estimate_frame_duration(time)
    if frame_duration <= 0:
        frame_duration = max_gap_seconds / 2 if max_gap_seconds > 0 else 0.01
    max_gap_frames = max(1, int(round(max_gap_seconds / frame_duration)))

    filled = freq.copy()
    idx = 0
    total = freq.size
    while idx < total:
        if voiced[idx]:
            idx += 1
            continue
        start = idx
        while idx < total and not voiced[idx]:
            idx += 1
        end = idx
        gap = end - start
        prev_idx = start - 1
        next_idx = end
        if (
            gap <= max_gap_frames
            and prev_idx >= 0
            and next_idx < total
            and voiced[prev_idx]
            and voiced[next_idx]
        ):
            filled[start:end] = np.interp(
                time[start:end],
                [time[prev_idx], time[next_idx]],
                [filled[prev_idx], filled[next_idx]],
            )
            voiced[start:end] = True
    return filled


def _prepare_arrays(track: Dict[str, List[float]]) -> Dict[str, np.ndarray]:
    time = np.asarray(track.get("time", []), dtype=float)
    freq = np.asarray(track.get("frequency", []), dtype=float)
    conf = np.asarray(track.get("confidence", []), dtype=float)
    if time.size == 0 or freq.size == 0:
        return {"time": np.array([]), "freq": np.array([]), "conf": np.array([])}

    length = min(time.size, freq.size)
    time = time[:length]
    freq = freq[:length]
    freq = np.nan_to_num(freq, nan=0.0, posinf=0.0, neginf=0.0)

    if conf.size < length:
        if conf.size == 0:
            conf = np.ones(length, dtype=float)
        else:
            conf = np.pad(conf, (0, length - conf.size), mode="edge")
    else:
        conf = conf[:length]

    order = np.argsort(time)
    time = time[order]
    freq = freq[order]
    conf = conf[order]

    voiced_flag = track.get("voiced_flag")
    if voiced_flag is not None:
        voiced = np.asarray(voiced_flag, dtype=bool)
        if voiced.size >= length:
            voiced = voiced[:length]
        else:
            voiced = np.pad(voiced, (0, length - voiced.size), constant_values=False)
        voiced = voiced[order]
    else:
        voiced = freq > 0

    return {"time": time, "freq": freq, "conf": conf, "voiced": voiced}


def smooth_pitch_track(track: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """Repair humming pitch tracks by filling gaps and suppressing jitter."""

    arrays = _prepare_arrays(track)
    time = arrays["time"]
    freq = arrays["freq"]
    conf = arrays["conf"]

    if time.size == 0 or freq.size == 0:
        print("[smooth_pitch] Empty track received; returning original structure")
        return {
            "time": track.get("time", []),
            "frequency": track.get("frequency", []),
            "confidence": track.get("confidence", []),
            "sr": track.get("sr", 16_000),
            "voiced_flag": track.get("voiced_flag", []),
        }

    voiced = arrays["voiced"].astype(bool)
    original_voiced = np.count_nonzero(freq > 0)
    repaired = _fill_short_gaps(time, freq, voiced, MAX_GAP_SECONDS)
    repaired = _median_filter(repaired)
    repaired = _moving_average(repaired)
    repaired = _limit_jumps(repaired)
    repaired = np.clip(repaired, a_min=0.0, a_max=MAX_FREQ)
    repaired[repaired < MIN_FREQ] = 0.0

    has_voiced = np.count_nonzero(repaired > 0) > 0
    if not has_voiced and original_voiced > 0:
        fallback = float(np.median(freq[freq > 0]))
        fallback = float(np.clip(fallback, MIN_FREQ, MAX_FREQ))
        repaired = np.full_like(repaired, fallback, dtype=float)
        voiced = np.ones_like(repaired, dtype=bool)
        print("[smooth_pitch] Applied fallback constant pitch due to unstable input")
    else:
        voiced = repaired > 0

    conf = np.nan_to_num(conf, nan=0.0)
    conf[~voiced] = 0.0

    print(
        f"[smooth_pitch] Frames: {time.size}, voiced_in={original_voiced}, voiced_out={np.count_nonzero(voiced)}"
    )

    return {
        "time": time.astype(float).tolist(),
        "frequency": repaired.astype(float).tolist(),
        "confidence": conf.astype(float).tolist(),
        "sr": track.get("sr", 16_000),
        "voiced_flag": voiced.astype(bool).tolist(),
    }
