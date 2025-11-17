"""Convert smoothed monophonic pitch tracks to MIDI."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pretty_midi

try:
    import librosa
except Exception as exc:  # pragma: no cover
    raise RuntimeError("librosa is required for MIDI conversion") from exc


MIN_CONFIDENCE = 0.05
MAX_INTERVAL_SEMITONES = 24
MAX_GAP_SECONDS = 0.06
MIN_NOTE_DURATION = 0.1
MERGE_GAP_SECONDS = 0.04
DEFAULT_VELOCITY = 80


class NoMelodyError(RuntimeError):
    """Raised when no usable melody notes can be created."""


def _to_array(values, dtype=float) -> np.ndarray:
    return np.asarray(values if values is not None else [], dtype=dtype)


def _prepare_arrays(track: Dict[str, Sequence[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = _to_array(track.get("time"))
    freq = _to_array(track.get("frequency"))
    conf = _to_array(track.get("confidence"))
    if time.size == 0 or freq.size == 0:
        return np.array([]), np.array([]), np.array([])
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
    return time[order], freq[order], conf[order]


def _estimate_frame_duration(time: np.ndarray) -> float:
    if time.size < 2:
        return 0.01
    diffs = np.diff(time)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.01
    return float(np.median(diffs))


def _clean_voiced_mask(freq: np.ndarray, conf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    voiced = np.isfinite(freq) & (freq > 0) & (conf >= MIN_CONFIDENCE)
    midi = np.zeros_like(freq, dtype=int)
    if not np.any(voiced):
        return midi, voiced
    midi[voiced] = np.round(librosa.hz_to_midi(freq[voiced])).astype(int)
    # Drop unrealistic jumps.
    for idx in range(1, len(midi)):
        if not voiced[idx] or not voiced[idx - 1]:
            continue
        if abs(int(midi[idx]) - int(midi[idx - 1])) > MAX_INTERVAL_SEMITONES:
            voiced[idx] = False
    return midi, voiced


def _segment_notes(
    time: np.ndarray,
    midi: np.ndarray,
    voiced: np.ndarray,
    frame_duration: float,
) -> List[Tuple[float, float, int]]:
    segments: List[Tuple[float, float, int]] = []
    idx = 0
    total = len(time)
    while idx < total:
        if not voiced[idx]:
            idx += 1
            continue
        note = int(midi[idx])
        start = float(time[idx])
        end = start + frame_duration
        last_time = time[idx]
        j = idx + 1
        while j < total and voiced[j] and midi[j] == note:
            gap = float(time[j] - last_time)
            if gap > MAX_GAP_SECONDS:
                break
            end = float(time[j]) + frame_duration
            last_time = time[j]
            j += 1
        segments.append((start, end, note))
        idx = j
    return segments


def _merge_segments(segments: List[Tuple[float, float, int]]) -> List[Tuple[float, float, int]]:
    if not segments:
        return []
    merged: List[Tuple[float, float, int]] = [segments[0]]
    for start, end, note in segments[1:]:
        prev_start, prev_end, prev_note = merged[-1]
        gap = start - prev_end
        if gap <= MERGE_GAP_SECONDS and note == prev_note:
            merged[-1] = (prev_start, max(prev_end, end), prev_note)
        else:
            merged.append((start, end, note))
    return merged


def _filter_short_segments(segments: List[Tuple[float, float, int]]) -> List[Tuple[float, float, int]]:
    filtered: List[Tuple[float, float, int]] = []
    for start, end, note in segments:
        if (end - start) >= MIN_NOTE_DURATION:
            filtered.append((start, end, note))
    return filtered


def build_midi_from_track(
    track: Dict[str, List[float]],
    output_path: Path,
    tempo: float = 120.0,
) -> Tuple[Path, int]:
    """Convert a smoothed pitch contour into a PrettyMIDI file."""

    time, freq, conf = _prepare_arrays(track)
    print(f"[midi_builder] Candidate frames: {len(time)}")
    if time.size == 0:
        raise NoMelodyError("Smoothed track contained no frames")

    midi, voiced = _clean_voiced_mask(freq, conf)
    if midi.size == 0 or not np.any(voiced):
        raise NoMelodyError("No stable monophonic melody could be extracted from this audio.")

    frame_duration = _estimate_frame_duration(time)
    segments = _segment_notes(time, midi, voiced, frame_duration)
    segments = _merge_segments(segments)
    segments = _filter_short_segments(segments)

    if not segments:
        raise NoMelodyError("No stable monophonic melody could be extracted from this audio.")

    midi_obj = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=0, name="Melody")
    for start, end, note in segments:
        instrument.notes.append(
            pretty_midi.Note(
                velocity=DEFAULT_VELOCITY,
                pitch=int(np.clip(note, 0, 127)),
                start=float(start),
                end=float(end),
            )
        )
    midi_obj.instruments.append(instrument)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi_obj.write(str(output_path))
    coverage = (
        f"{instrument.notes[0].start:.2f}s→{instrument.notes[-1].end:.2f}s"
        if instrument.notes
        else "0.00s"
    )
    print(f"[midi_builder] Wrote {len(instrument.notes)} notes covering {coverage}")
    return output_path, len(instrument.notes)


def melody_to_midi(
    track: Dict[str, List[float]],
    output_path: Path,
    tempo: float = 120.0,
) -> Tuple[Path, int]:
    return build_midi_from_track(track, output_path, tempo=tempo)
