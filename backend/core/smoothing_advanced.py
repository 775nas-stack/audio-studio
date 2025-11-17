from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

import librosa

from .types import NoMelodyError, PitchTrack

VOICE_THRESHOLD = 0.15
MEDIAN_WINDOW = 7
SAVGOL_WINDOW = 11
SAVGOL_ORDER = 3
OUTLIER_RATIO = 0.1
JUMP_LIMIT = 6.0  # semitones


def _nanmedian_filter(data: np.ndarray, window: int) -> np.ndarray:
    result = data.copy()
    half = window // 2
    for idx in range(len(data)):
        start = max(0, idx - half)
        end = min(len(data), idx + half + 1)
        window_vals = data[start:end]
        valid = window_vals[np.isfinite(window_vals)]
        if valid.size:
            result[idx] = np.median(valid)
    return result


def _fill_missing(data: np.ndarray) -> np.ndarray:
    filled = data.copy()
    idx = np.arange(len(filled))
    valid = np.where(np.isfinite(filled))[0]
    if valid.size == 0:
        return filled
    filled = np.interp(idx, valid, filled[valid])
    return filled


def _trim_outliers(data: np.ndarray, ratio: float) -> np.ndarray:
    result = data.copy()
    valid = np.isfinite(result)
    if valid.sum() < 4:
        return result
    values = np.sort(result[valid])
    trim = max(1, int(len(values) * ratio))
    low = values[trim]
    high = values[-trim - 1]
    result[result < low] = low
    result[result > high] = high
    return result


def _octave_correct(freq: np.ndarray) -> np.ndarray:
    corrected = freq.copy()
    mask = np.isfinite(corrected) & (corrected > 0)
    if mask.sum() < 2:
        return corrected
    midi = librosa.hz_to_midi(corrected[mask])
    prev = midi[0]
    for idx in range(1, len(midi)):
        value = midi[idx]
        while value - prev > JUMP_LIMIT:
            value -= 12
        while prev - value > JUMP_LIMIT:
            value += 12
        midi[idx] = value
        prev = value
    corrected_masked = librosa.midi_to_hz(midi)
    corrected_values = corrected.copy()
    corrected_values[mask] = corrected_masked
    return corrected_values


def smooth_track_advanced(track: PitchTrack) -> PitchTrack:
    freq = track.frequency.astype(float).copy()
    conf = track.confidence.astype(float).copy()

    voiced = conf >= VOICE_THRESHOLD
    freq[~voiced] = np.nan
    freq = _trim_outliers(freq, OUTLIER_RATIO)

    if np.isfinite(freq).sum() < 8:
        raise NoMelodyError("No stable monophonic melody detected.")

    filled = _fill_missing(freq)
    median_smoothed = _nanmedian_filter(filled, MEDIAN_WINDOW)

    base = len(filled)
    window = min(SAVGOL_WINDOW, base - (1 - base % 2))
    if window < SAVGOL_ORDER + 2:
        window = SAVGOL_ORDER + 3
    if window % 2 == 0:
        window += 1
    if window > base:
        window = base if base % 2 == 1 else max(1, base - 1)
    poly_smoothed = savgol_filter(filled, window_length=window, polyorder=min(SAVGOL_ORDER, window - 1))

    median_smoothed = _octave_correct(median_smoothed)
    poly_smoothed = _octave_correct(poly_smoothed)

    weight = np.clip((conf - VOICE_THRESHOLD) / max(1e-6, 1 - VOICE_THRESHOLD), 0.0, 1.0)
    blended = weight * poly_smoothed + (1.0 - weight) * median_smoothed
    blended[~voiced] = np.nan

    refined = PitchTrack(
        time=track.time,
        frequency=blended,
        confidence=conf,
        engine=track.engine,
        activation=track.activation,
        loudness=track.loudness,
        sources=track.sources,
    )

    if refined.finite_count() < 8:
        raise NoMelodyError("No stable monophonic melody detected.")

    return refined
