from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

CONSECUTIVE_ZERO_END = 3
MIN_NOTE_DURATION = 0.07  # seconds
MAX_MIDI = 127


@dataclass
class MidiNote:
    pitch: int
    start: float
    end: float
    velocity: int

    @property
    def duration(self) -> float:
        return self.end - self.start


def _validate_inputs(pitch: Sequence[float], confidence: Sequence[float], time: Sequence[float]) -> None:
    if not (len(pitch) == len(confidence) == len(time)):
        raise ValueError("pitch, confidence, and time arrays must share the same length")


def _frame_duration(time: np.ndarray) -> float:
    if time.size < 2:
        return 0.01
    diffs = np.diff(time)
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return 0.01
    return float(np.median(positive))


def _hz_to_midi(pitches: np.ndarray) -> int:
    valid = pitches[pitches > 0]
    if valid.size == 0:
        return 0
    midi_values = 69.0 + 12.0 * np.log2(valid / 440.0)
    midi = int(np.clip(np.rint(np.nanmedian(midi_values)), 0, MAX_MIDI))
    return midi


def _frames_to_note(
    frames: List[int],
    pitch: np.ndarray,
    confidence: np.ndarray,
    time: np.ndarray,
    frame_duration: float,
) -> MidiNote:
    start_time = float(time[frames[0]])
    end_time = float(time[frames[-1]] + frame_duration)
    note_pitch = _hz_to_midi(pitch[frames])
    velocity_value = confidence[frames]
    velocity = int(np.clip(float(np.mean(velocity_value)) * 90.0, 0, MAX_MIDI))
    return MidiNote(pitch=note_pitch, start=start_time, end=end_time, velocity=velocity)


def _merge_short_notes(notes: List[MidiNote]) -> List[MidiNote]:
    if not notes:
        return []

    merged: List[MidiNote] = []
    idx = 0
    total = len(notes)

    while idx < total:
        note = notes[idx]
        if note.duration < MIN_NOTE_DURATION:
            if merged:
                previous = merged[-1]
                previous.end = max(previous.end, note.end)
                previous.velocity = int(round((previous.velocity + note.velocity) / 2))
                idx += 1
                continue
            if idx + 1 < total:
                next_note = notes[idx + 1]
                next_note.start = min(note.start, next_note.start)
                next_note.velocity = int(round((next_note.velocity + note.velocity) / 2))
                idx += 1
                continue
        merged.append(note)
        idx += 1

    return merged


def segment_melody(
    pitch: Sequence[float],
    confidence: Sequence[float],
    time: Sequence[float],
) -> List[MidiNote]:
    _validate_inputs(pitch, confidence, time)

    pitch_arr = np.asarray(pitch, dtype=float)
    confidence_arr = np.asarray(confidence, dtype=float)
    time_arr = np.asarray(time, dtype=float)

    if pitch_arr.size == 0:
        return []

    frame_duration = _frame_duration(time_arr)
    active_frames: List[int] = []
    zero_run = 0
    notes: List[MidiNote] = []

    for idx, value in enumerate(pitch_arr):
        if value > 0:
            active_frames.append(idx)
            zero_run = 0
            continue

        zero_run += 1
        if zero_run >= CONSECUTIVE_ZERO_END and active_frames:
            notes.append(
                _frames_to_note(
                    frames=active_frames,
                    pitch=pitch_arr,
                    confidence=confidence_arr,
                    time=time_arr,
                    frame_duration=frame_duration,
                )
            )
            active_frames = []

    if active_frames:
        notes.append(
            _frames_to_note(
                frames=active_frames,
                pitch=pitch_arr,
                confidence=confidence_arr,
                time=time_arr,
                frame_duration=frame_duration,
            )
        )

    return _merge_short_notes(notes)


__all__ = ["MidiNote", "segment_melody"]
