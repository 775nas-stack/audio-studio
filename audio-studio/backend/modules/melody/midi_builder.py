"""Utilities for turning a smoothed melody into a MIDI file."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pretty_midi


class MelodyMidiError(RuntimeError):
    """Raised when MIDI conversion fails."""


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _hz_to_midi(freq: float) -> float:
    if freq <= 0:
        return float("nan")
    return 69.0 + 12.0 * np.log2(freq / 440.0)


def _segment_notes(
    times: np.ndarray,
    freqs: np.ndarray,
    cents_threshold: float,
) -> List[Tuple[float, float, float]]:
    notes: List[Tuple[float, float, float]] = []
    if times.size == 0 or freqs.size == 0:
        return notes

    frame_step = np.median(np.diff(times)) if times.size > 1 else 0.01
    frame_step = max(frame_step, 1e-3)

    current_pitch = float("nan")
    start_time = None

    def close_note(end_idx: int) -> None:
        nonlocal current_pitch, start_time
        if start_time is None or np.isnan(current_pitch):
            return
        end_time = times[end_idx] if end_idx < times.size else times[-1] + frame_step
        if end_time <= start_time:
            end_time = start_time + frame_step
        notes.append((start_time, end_time, current_pitch))
        start_time = None
        current_pitch = float("nan")

    for idx, freq in enumerate(freqs):
        midi_pitch = _hz_to_midi(freq)
        if np.isnan(midi_pitch):
            close_note(idx)
            continue
        if start_time is None:
            start_time = times[idx]
            current_pitch = midi_pitch
            continue
        cents_change = abs(midi_pitch - current_pitch) * 100.0
        if cents_change >= cents_threshold:
            close_note(idx)
            start_time = times[idx]
            current_pitch = midi_pitch
        else:
            current_pitch = (current_pitch + midi_pitch) / 2.0

    close_note(times.size)
    return notes


def _quantize_duration(duration: float, step: float) -> float:
    if duration <= 0:
        return step
    return max(step, round(duration / step) * step)


def build_midi(
    smooth_track: Dict[str, List[float]],
    project_dir: str,
    velocity: int = 80,
    cents_threshold: float = 50.0,
    quantize_step: float = 0.01,
) -> str:
    """Convert the smoothed melody track to ``melody.mid`` using PrettyMIDI."""

    times = np.asarray(smooth_track.get("time", []), dtype=float)
    freqs = np.asarray(smooth_track.get("frequency", []), dtype=float)
    if times.size == 0 or freqs.size == 0:
        raise MelodyMidiError("Smoothed melody track is empty. Cannot build MIDI.")
    if times.size != freqs.size:
        raise MelodyMidiError("Time and frequency arrays have mismatched sizes.")

    notes = _segment_notes(times, freqs, cents_threshold)
    if not notes:
        raise MelodyMidiError("No valid notes detected in smoothed melody track.")

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, is_drum=False, name="Lead")
    for start, end, pitch in notes:
        duration = _quantize_duration(end - start, quantize_step)
        midi_note = pretty_midi.Note(
            velocity=int(np.clip(velocity, 1, 127)),
            pitch=int(np.clip(round(pitch), 0, 127)),
            start=float(start),
            end=float(start + duration),
        )
        instrument.notes.append(midi_note)
    midi.instruments.append(instrument)

    project_path = Path(project_dir)
    _ensure_directory(project_path)
    midi_path = project_path / "melody.mid"
    midi.write(str(midi_path))
    return str(midi_path)


__all__ = ["build_midi", "MelodyMidiError"]
