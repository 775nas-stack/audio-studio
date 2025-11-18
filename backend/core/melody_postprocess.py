"""Professional-grade melody post-processing helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Mapping, Sequence

import librosa
import numpy as np

try:  # Optional dependency (already used elsewhere in the project)
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover - scipy is part of the default env but keep safe
    savgol_filter = None  # type: ignore

from .types import NoMelodyError, PitchTrack

LOGGER = logging.getLogger(__name__)

PitchableFrameClassifier = Callable[[np.ndarray], np.ndarray]


@dataclass
class MelodyNote:
    """Structured note representation returned by the post-processor."""

    start_time: float
    end_time: float
    midi: float
    frequency: float
    cents_offset: float
    confidence: float
    frames: int

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_payload(self) -> dict[str, float]:
        return {
            "start_time": float(self.start_time),
            "end_time": float(self.end_time),
            "midi": float(self.midi),
            "frequency": float(self.frequency),
            "cents_offset": float(self.cents_offset),
            "confidence": float(self.confidence),
            "frames": int(self.frames),
        }

    @staticmethod
    def from_payload(payload: Mapping[str, float]) -> "MelodyNote":
        return MelodyNote(
            start_time=float(payload["start_time"]),
            end_time=float(payload["end_time"]),
            midi=float(payload["midi"]),
            frequency=float(payload["frequency"]),
            cents_offset=float(payload.get("cents_offset", 0.0)),
            confidence=float(payload.get("confidence", 0.0)),
            frames=int(payload.get("frames", 0)),
        )


@dataclass
class MelodyPostProcessConfig:
    """Configuration switches for the professional melody logic."""

    confidence_threshold: float = 0.5
    energy_threshold: float = 0.01
    classifier_min_probability: float = 0.5

    median_window: int = 5
    mean_window: int = 7
    bilateral_window: int = 7
    bilateral_cents: float = 35.0
    savgol_window: int = 9
    savgol_order: int = 2

    apply_bilateral: bool = True
    apply_savgol: bool = True

    pitch_drift_cents: float = 60.0
    stability_frames: int = 5

    min_note_duration: float = 0.15  # seconds
    merge_gap: float = 0.15  # seconds
    pitch_merge_cents: float = 30.0
    octave_jump_limit: float = 7.5  # semitones


DEFAULT_CONFIG = MelodyPostProcessConfig()


def postprocess_pitch_track(
    track: PitchTrack,
    energy: np.ndarray | None = None,
    classifier: PitchableFrameClassifier | None = None,
    config: MelodyPostProcessConfig | None = None,
) -> list[MelodyNote]:
    """Transform a raw `PitchTrack` into musically meaningful note events."""

    if config is None:
        config = DEFAULT_CONFIG

    freq = track.frequency.astype(float).copy()
    conf = track.confidence.astype(float).copy()
    time = track.time.astype(float).copy()
    if energy is None and track.loudness is not None:
        energy = track.loudness.astype(float)
    elif energy is None:
        # Loudness occasionally missing; derive RMS per frame from activation if needed.
        energy = np.abs(track.activation) if track.activation is not None else np.full_like(freq, 1.0)

    valid_mask = _build_pitchable_mask(freq, conf, energy, classifier, config)
    if not np.any(valid_mask):
        raise NoMelodyError("No confident voiced frames after gating.")

    freq[~valid_mask] = np.nan
    freq = _smooth_frequency(freq, config)
    freq = _correct_octave_jumps(freq, config.octave_jump_limit)

    quantized = _quantize_with_hysteresis(freq, config)
    if not np.isfinite(quantized).any():
        raise NoMelodyError("Quantization removed all frames.")

    notes = _segment_notes(time, freq, conf, quantized, config)
    if not notes:
        raise NoMelodyError("Unable to form stable note events.")

    return notes


# ---------------------------------------------------------------------------
# Frame gating
# ---------------------------------------------------------------------------

def _build_pitchable_mask(
    freq: np.ndarray,
    conf: np.ndarray,
    energy: np.ndarray,
    classifier: PitchableFrameClassifier | None,
    config: MelodyPostProcessConfig,
) -> np.ndarray:
    mask = np.ones_like(freq, dtype=bool)
    mask &= np.isfinite(freq)
    mask &= conf >= config.confidence_threshold
    mask &= energy >= config.energy_threshold

    if classifier is not None:
        features = np.stack([freq, conf, energy], axis=1)
        features[~np.isfinite(features)] = 0.0
        classifier_mask = classifier(features)
        if classifier_mask.dtype != bool:
            classifier_mask = classifier_mask >= config.classifier_min_probability
        mask &= classifier_mask.astype(bool)
    return mask


# ---------------------------------------------------------------------------
# Smoothing pipeline
# ---------------------------------------------------------------------------

def _smooth_frequency(freq: np.ndarray, config: MelodyPostProcessConfig) -> np.ndarray:
    result = freq.copy()
    result = _nanmedian_filter(result, config.median_window)
    result = _nanmean_filter(result, config.mean_window)
    if config.apply_bilateral:
        result = _bilateral_filter(result, config.bilateral_window, config.bilateral_cents)
    if config.apply_savgol and savgol_filter is not None:
        result = _apply_savgol(result, config.savgol_window, config.savgol_order)
    return result


def _nanmedian_filter(data: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window) | 1)  # ensure odd window
    half = window // 2
    result = data.copy()
    for idx in range(len(data)):
        start = max(0, idx - half)
        end = min(len(data), idx + half + 1)
        values = data[start:end]
        values = values[np.isfinite(values)]
        if values.size:
            result[idx] = np.median(values)
    return result


def _nanmean_filter(data: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    result = np.copy(data)
    kernel = np.ones(window)
    finite = np.isfinite(data).astype(float)
    filled = np.where(finite, data, 0.0)
    sums = np.convolve(filled, kernel, mode="same")
    counts = np.convolve(finite, kernel, mode="same")
    counts[counts == 0] = 1.0
    result = sums / counts
    result[~np.isfinite(data)] = np.nan
    return result


def _bilateral_filter(data: np.ndarray, window: int, cents: float) -> np.ndarray:
    if window <= 1:
        return data
    result = data.copy()
    half = window // 2
    sigma = max(1e-6, cents)
    midi_vals = librosa.hz_to_midi(np.clip(data, 1e-6, None))
    for idx in range(len(data)):
        start = max(0, idx - half)
        end = min(len(data), idx + half + 1)
        neigh = data[start:end]
        mask = np.isfinite(neigh)
        if not mask.any():
            continue
        center = midi_vals[idx] if np.isfinite(data[idx]) else np.nanmedian(midi_vals[mask])
        neigh_midi = midi_vals[start:end][mask]
        weight = np.exp(-0.5 * ((neigh_midi - center) / sigma) ** 2)
        if weight.size:
            result[idx] = np.average(neigh[mask], weights=weight)
    return result


def _apply_savgol(data: np.ndarray, window: int, order: int) -> np.ndarray:
    valid = np.isfinite(data)
    idxs = np.flatnonzero(valid)
    if idxs.size < max(5, order + 2):
        return data
    filled = np.interp(np.arange(len(data)), idxs, data[valid])
    win = max(order + 2, window)
    if win % 2 == 0:
        win += 1
    if win >= len(filled):
        win = len(filled) - 1 if len(filled) % 2 == 0 else len(filled)
    filtered = savgol_filter(filled, window_length=max(order + 2, win), polyorder=order)
    result = data.copy()
    result[valid] = filtered[valid]
    return result


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def _correct_octave_jumps(freq: np.ndarray, limit: float) -> np.ndarray:
    result = freq.copy()
    mask = np.isfinite(result) & (result > 0)
    if mask.sum() < 2:
        return result
    midi = librosa.hz_to_midi(result[mask])
    prev = midi[0]
    for idx in range(1, len(midi)):
        value = midi[idx]
        while value - prev > limit:
            value -= 12
        while prev - value > limit:
            value += 12
        midi[idx] = value
        prev = value
    result[mask] = librosa.midi_to_hz(midi)
    return result


def _quantize_with_hysteresis(freq: np.ndarray, config: MelodyPostProcessConfig) -> np.ndarray:
    midi = np.full_like(freq, np.nan, dtype=float)
    mask = np.isfinite(freq) & (freq > 0)
    midi[mask] = librosa.hz_to_midi(freq[mask])
    quantized = np.full_like(midi, np.nan)

    current = np.nan
    candidate = np.nan
    hold = 0
    drift_limit = config.pitch_drift_cents / 100.0

    for idx, value in enumerate(midi):
        if not np.isfinite(value):
            hold = 0
            candidate = np.nan
            continue
        snapped = np.round(value)
        if not np.isfinite(current):
            current = snapped
            quantized[idx] = snapped
            continue
        diff = abs(value - current)
        if diff <= drift_limit:
            quantized[idx] = current
            candidate = np.nan
            hold = 0
            continue

        # Candidate switch
        if not np.isfinite(candidate) or abs(snapped - candidate) > 1e-3:
            candidate = snapped
            hold = 1
        else:
            hold += 1

        if hold >= config.stability_frames:
            current = candidate
            quantized[idx] = current
            hold = 0
        else:
            quantized[idx] = current
    return quantized


# ---------------------------------------------------------------------------
# Note segmentation
# ---------------------------------------------------------------------------

def _segment_notes(
    time: np.ndarray,
    freq: np.ndarray,
    confidence: np.ndarray,
    quantized: np.ndarray,
    config: MelodyPostProcessConfig,
) -> list[MelodyNote]:
    notes: List[MelodyNote] = []
    idx = 0
    n = len(time)
    while idx < n:
        if not np.isfinite(quantized[idx]):
            idx += 1
            continue
        start = idx
        current_note = quantized[idx]
        while idx < n and np.isfinite(quantized[idx]) and abs(quantized[idx] - current_note) < 0.25:
            idx += 1
        end = idx
        note = _build_note(start, end, time, freq, confidence, quantized)
        if note:
            notes.append(note)
    notes = _merge_short_gaps(notes, config.merge_gap)
    notes = _merge_pitch_neighbors(notes, config.pitch_merge_cents)
    notes = [note for note in notes if note.duration >= config.min_note_duration]
    return notes


def _build_note(
    start: int,
    end: int,
    time: np.ndarray,
    freq: np.ndarray,
    confidence: np.ndarray,
    quantized: np.ndarray,
) -> MelodyNote | None:
    if end - start <= 0:
        return None
    start_time = time[start]
    end_time = time[end - 1]
    duration = max(1e-9, end_time - start_time)
    frame_slice = slice(start, end)
    q = quantized[frame_slice]
    valid = np.isfinite(q)
    if not valid.any():
        return None
    midi_value = float(np.nanmedian(q[valid]))
    hz_values = freq[frame_slice]
    hz_values = hz_values[np.isfinite(hz_values)]
    hz = float(np.nanmedian(hz_values)) if hz_values.size else float(librosa.midi_to_hz(midi_value))
    confidence_mean = float(np.nanmean(confidence[frame_slice]))
    cents_offset = float((np.nanmedian(librosa.hz_to_midi(np.clip(hz_values, 1e-6, None))) - midi_value) * 100) if hz_values.size else 0.0
    return MelodyNote(
        start_time=start_time,
        end_time=end_time,
        midi=midi_value,
        frequency=hz,
        cents_offset=cents_offset,
        confidence=confidence_mean,
        frames=end - start,
    )


def _merge_short_gaps(notes: Sequence[MelodyNote], max_gap: float) -> list[MelodyNote]:
    if not notes:
        return []
    merged: list[MelodyNote] = [notes[0]]
    for note in notes[1:]:
        prev = merged[-1]
        gap = note.start_time - prev.end_time
        if gap <= max_gap:
            combined = _combine_notes(prev, note)
            merged[-1] = combined
        else:
            merged.append(note)
    return merged


def _merge_pitch_neighbors(notes: Sequence[MelodyNote], cents: float) -> list[MelodyNote]:
    if not notes:
        return []
    merged: list[MelodyNote] = [notes[0]]
    tolerance = cents / 100.0
    for note in notes[1:]:
        prev = merged[-1]
        if abs(prev.midi - note.midi) <= tolerance:
            merged[-1] = _combine_notes(prev, note)
        else:
            merged.append(note)
    return merged


def _combine_notes(a: MelodyNote, b: MelodyNote) -> MelodyNote:
    total_frames = a.frames + b.frames
    if total_frames <= 0:
        total_frames = 1
    weight_a = a.frames / total_frames
    weight_b = b.frames / total_frames
    return MelodyNote(
        start_time=a.start_time,
        end_time=b.end_time,
        midi=a.midi * weight_a + b.midi * weight_b,
        frequency=a.frequency * weight_a + b.frequency * weight_b,
        cents_offset=a.cents_offset * weight_a + b.cents_offset * weight_b,
        confidence=a.confidence * weight_a + b.confidence * weight_b,
        frames=total_frames,
    )


__all__ = [
    "MelodyNote",
    "MelodyPostProcessConfig",
    "PitchableFrameClassifier",
    "postprocess_pitch_track",
]
