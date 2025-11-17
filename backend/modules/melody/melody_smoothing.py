"""Post processing utilities for cleaning the raw CREPE melody track."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


class MelodySmoothingError(RuntimeError):
    """Raised when smoothing cannot be completed."""


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _median_filter(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    smoothed = np.zeros_like(values)
    for idx in range(values.size):
        window_values = padded[idx : idx + window]
        voiced = window_values[window_values > 0]
        if voiced.size:
            smoothed[idx] = float(np.median(voiced))
        else:
            smoothed[idx] = 0.0
    return smoothed


def _filter_jumps(freqs: np.ndarray, max_jump_cents: float) -> np.ndarray:
    if freqs.size == 0:
        return freqs
    filtered = freqs.copy()
    last_valid = None
    for idx, value in enumerate(filtered):
        if value <= 0:
            continue
        if last_valid is None:
            last_valid = value
            continue
        cents = abs(1200.0 * np.log2(value / last_valid))
        if cents > max_jump_cents:
            filtered[idx] = 0.0
        else:
            last_valid = value
    return filtered


def _hold_stable(freqs: np.ndarray, hold_frames: int) -> np.ndarray:
    if hold_frames <= 0 or freqs.size == 0:
        return freqs
    held = freqs.copy()
    last_value = 0.0
    gap = 0
    for idx, value in enumerate(held):
        if value > 0:
            last_value = value
            gap = 0
            continue
        if last_value > 0 and gap < hold_frames:
            held[idx] = last_value
            gap += 1
        else:
            last_value = 0.0
            gap += 1
    return held


def _trim_silence(times: np.ndarray, freqs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    voiced_idx = np.where(freqs > 0)[0]
    if voiced_idx.size == 0:
        raise MelodySmoothingError("All frames were removed during smoothing.")
    start = voiced_idx[0]
    end = voiced_idx[-1] + 1
    return times[start:end], freqs[start:end]


def smooth_melody(
    raw_track: Dict[str, List[float]],
    project_dir: str,
    median_window: int = 5,
    max_jump_cents: float = 400.0,
    hold_frames: int = 3,
) -> Dict[str, List[float]]:
    """Apply median smoothing, jump filtering and stable pitch holding."""

    if not raw_track:
        raise MelodySmoothingError("Missing raw melody data for smoothing.")

    times = np.asarray(raw_track.get("time", []), dtype=float)
    freqs = np.asarray(raw_track.get("frequency", []), dtype=float)
    if times.size == 0 or freqs.size == 0:
        raise MelodySmoothingError("Raw melody data is empty.")
    if times.size != freqs.size:
        raise MelodySmoothingError("Time and frequency arrays have mismatched sizes.")

    smoothed = _median_filter(freqs, median_window)
    smoothed = _filter_jumps(smoothed, max_jump_cents)
    smoothed = _hold_stable(smoothed, hold_frames)
    times_trimmed, freqs_trimmed = _trim_silence(times, smoothed)

    result = {
        "time": times_trimmed.astype(float).tolist(),
        "frequency": freqs_trimmed.astype(float).tolist(),
        "sr": raw_track.get("sr", 16_000),
    }

    project_path = Path(project_dir)
    _ensure_directory(project_path)
    smooth_path = project_path / "melody_smooth.json"
    with smooth_path.open("w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2)

    return result


__all__ = ["smooth_melody", "MelodySmoothingError"]
