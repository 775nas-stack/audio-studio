"""Utilities for converting smoothed melody tracks into MIDI files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pretty_midi

MIN_ALLOWED_FREQ = 65.0
MAX_ALLOWED_FREQ = 1000.0
HUMMING_MAX_ALLOWED_FREQ = 1500.0
MAX_INTERP_GAP_SECONDS = 0.1
HUMMING_MAX_INTERP_GAP_SECONDS = 0.05
MIN_NOTE_DURATION_SECONDS = 0.08
HUMMING_MIN_NOTE_DURATION_SECONDS = 0.03


def _hz_to_midi(freq: float) -> int:
    return int(round(pretty_midi.hz_to_note_number(max(freq, 1e-6))))


def _quantize_time(value: float, quantum: float) -> float:
    if quantum <= 0:
        return value
    return round(value / quantum) * quantum


def _estimate_frame_duration(time: np.ndarray) -> float:
    if time.size < 2:
        return 0.0
    diffs = np.diff(time)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.0
    return float(np.median(diffs))


def _interpolate_small_gaps(
    time: np.ndarray,
    freq: np.ndarray,
    mask: np.ndarray,
    max_gap: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if time.size == 0:
        return freq, mask

    filled = freq.copy()
    valid = mask.copy()
    frame_duration = _estimate_frame_duration(time)
    if frame_duration <= 0:
        frame_duration = max_gap

    idx = 0
    total = len(freq)
    while idx < total:
        if valid[idx]:
            idx += 1
            continue
        start = idx
        while idx < total and not valid[idx]:
            idx += 1
        end = idx

        gap_frames = end - start
        prev_idx = start - 1
        next_idx = end
        approx_duration = gap_frames * frame_duration

        if (
            approx_duration <= max_gap
            and prev_idx >= 0
            and next_idx < total
            and valid[prev_idx]
            and valid[next_idx]
        ):
            start_time = time[prev_idx]
            end_time = time[next_idx]
            if end_time > start_time:
                filled[start:end] = np.interp(
                    time[start:end],
                    [start_time, end_time],
                    [filled[prev_idx], filled[next_idx]],
                )
                valid[start:end] = True

    return filled, valid


def _clean_pitch_track(
    track: Dict[str, Sequence[float]],
    *,
    humming_mode: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    time = np.asarray(track.get("time", []), dtype=float)
    freq = np.asarray(track.get("frequency", []), dtype=float)
    if time.size == 0 or freq.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    length = min(time.size, freq.size)
    time = time[:length]
    freq = np.nan_to_num(freq[:length], nan=0.0, posinf=0.0, neginf=0.0)

    mask = np.ones(length, dtype=bool)
    mask &= np.isfinite(time)
    mask &= np.isfinite(freq)
    mask &= freq > 0

    voiced_flag = track.get("voiced_flag")
    if voiced_flag is not None:
        voiced_arr = np.asarray(voiced_flag, dtype=bool)
        if voiced_arr.size >= length:
            mask &= voiced_arr[:length]

    max_gap = HUMMING_MAX_INTERP_GAP_SECONDS if humming_mode else MAX_INTERP_GAP_SECONDS
    freq, mask = _interpolate_small_gaps(time, freq, mask, max_gap)
    freq = np.clip(freq, a_min=0.0, a_max=None)

    mask &= np.isfinite(freq)
    mask &= freq >= MIN_ALLOWED_FREQ
    max_freq = HUMMING_MAX_ALLOWED_FREQ if humming_mode else MAX_ALLOWED_FREQ
    mask &= freq <= max_freq

    if not np.any(mask):
        return np.array([], dtype=float), np.array([], dtype=float)

    return time[mask], freq[mask]


def _merge_midi_frames(
    time: np.ndarray, midi_pitch: np.ndarray, frame_duration: float
) -> List[Tuple[float, float, int]]:
    if midi_pitch.size == 0:
        return []

    segments: List[Tuple[float, float, int]] = []
    current_pitch = int(midi_pitch[0])
    start_time = float(time[0])

    for idx in range(1, midi_pitch.size):
        if int(midi_pitch[idx]) != current_pitch:
            end_time = float(time[idx])
            if end_time <= start_time:
                end_time = start_time + max(frame_duration, 1e-3)
            segments.append((start_time, end_time, current_pitch))
            start_time = float(time[idx])
            current_pitch = int(midi_pitch[idx])

    end_time = float(time[-1]) + frame_duration
    if end_time <= start_time:
        end_time = start_time + max(frame_duration, 1e-3)
    segments.append((start_time, end_time, current_pitch))
    return segments


def _filter_short_segments(
    segments: Iterable[Tuple[float, float, int]], min_duration: float
) -> List[Tuple[float, float, int]]:
    filtered: List[Tuple[float, float, int]] = []
    for start, end, pitch in segments:
        duration = end - start
        if duration < min_duration or duration <= 0:
            continue
        filtered.append((start, end, pitch))
    return filtered


def build_midi_from_track(
    track: Dict[str, List[float]],
    output_path: Path,
    tempo: float = 120.0,
    *,
    humming_mode: bool = False,
) -> Path:
    """Convert a smoothed melody track into a quantized PrettyMIDI file."""

    time, freq = _clean_pitch_track(track, humming_mode=humming_mode)
    if time.size == 0 or freq.size == 0:
        raise ValueError("No stable melody detected")

    midi_pitch = np.array([_hz_to_midi(hz) for hz in freq], dtype=int)
    frame_duration = _estimate_frame_duration(time)
    segments = _merge_midi_frames(time, midi_pitch, frame_duration)
    min_duration = (
        HUMMING_MIN_NOTE_DURATION_SECONDS if humming_mode else MIN_NOTE_DURATION_SECONDS
    )
    segments = _filter_short_segments(segments, min_duration)

    if not segments and humming_mode and time.size > 0:
        start = float(time[0])
        end = float(time[-1]) + max(frame_duration, min_duration, 1e-3)
        fallback_pitch = int(np.clip(int(round(np.median(midi_pitch))), 0, 127))
        segments = [(start, end, fallback_pitch)]

    if not segments:
        raise ValueError("No stable melody detected")

    sixteenth = 60.0 / tempo / 4.0

    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=0, name="Melody")

    for start, end, midi_note in segments:
        start_q = _quantize_time(start, sixteenth)
        end_q = _quantize_time(end, sixteenth)
        if end_q <= start_q:
            end_q = start_q + max(sixteenth, 1e-3)
        note = pretty_midi.Note(
            velocity=100,
            pitch=int(midi_note),
            start=start_q,
            end=end_q,
        )
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_path))
    return output_path


def melody_to_midi(
    track: Dict[str, List[float]],
    output_path: Path,
    tempo: float = 120.0,
    *,
    humming_mode: bool = False,
) -> Path:
    """Backward-compatible wrapper for callers relying on the old name."""

    return build_midi_from_track(
        track,
        output_path,
        tempo=tempo,
        humming_mode=humming_mode,
    )
