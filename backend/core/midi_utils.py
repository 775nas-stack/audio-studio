"""Utilities for converting smoothed melody tracks into MIDI files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pretty_midi


def _hz_to_midi(freq: float) -> int:
    return int(round(pretty_midi.hz_to_note_number(max(freq, 1e-6))))


def _quantize_time(value: float, quantum: float) -> float:
    if quantum <= 0:
        return value
    return round(value / quantum) * quantum


def _segment_notes(time: np.ndarray, freq: np.ndarray, max_cents: float = 50.0) -> List[Tuple[float, float, float]]:
    if len(freq) == 0:
        return []

    segments: List[Tuple[float, float, float]] = []
    start_idx = 0
    current = freq[0]

    for idx in range(1, len(freq)):
        if current <= 0 or freq[idx] <= 0:
            continue
        cents = 1200.0 * np.log2(freq[idx] / current)
        if abs(cents) > max_cents:
            segments.append((time[start_idx], time[idx - 1], current))
            start_idx = idx
            current = freq[idx]
        else:
            current = (current + freq[idx]) / 2.0

    segments.append((time[start_idx], time[-1], current))
    return segments


def _dedupe_short_segments(segments: Iterable[Tuple[float, float, float]], min_duration: float) -> List[Tuple[float, float, float]]:
    filtered: List[Tuple[float, float, float]] = []
    for start, end, freq in segments:
        if end <= start:
            continue
        if (end - start) < min_duration:
            continue
        filtered.append((start, end, freq))
    return filtered


def melody_to_midi(track: Dict[str, List[float]], output_path: Path, tempo: float = 120.0) -> Path:
    """Convert a smoothed melody track into a quantized PrettyMIDI file."""

    time = np.array(track["time"], dtype=float)
    freq = np.array(track["frequency"], dtype=float)
    if time.size == 0 or freq.size == 0:
        raise ValueError("No melody frames available for MIDI export")

    sixteenth = 60.0 / tempo / 4.0
    segments = _segment_notes(time, freq)
    segments = _dedupe_short_segments(segments, min_duration=sixteenth)

    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=0, name="Melody")

    for start, end, hz in segments:
        start_q = _quantize_time(start, sixteenth)
        end_q = _quantize_time(end, sixteenth)
        if end_q <= start_q:
            end_q = start_q + sixteenth
        note_number = _hz_to_midi(hz)
        note = pretty_midi.Note(velocity=100, pitch=note_number, start=start_q, end=end_q)
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_path))
    return output_path
