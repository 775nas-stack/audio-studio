from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import librosa
import numpy as np
import pretty_midi

from .types import NoMelodyError, PitchTrack

FRAME_DURATION = 0.01
MAX_JUMP = 24  # semitones
GAP_THRESHOLD = 0.04
MIN_NOTE = 0.08
MERGE_GAP = 0.05


@dataclass
class NoteEvent:
    pitch: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def _filter_jumps(times: np.ndarray, midi_int: np.ndarray):
    if len(midi_int) == 0:
        return times, midi_int
    keep = [0]
    prev_pitch = midi_int[0]
    for idx in range(1, len(midi_int)):
        if abs(midi_int[idx] - prev_pitch) > MAX_JUMP:
            continue
        keep.append(idx)
        prev_pitch = midi_int[idx]
    keep = np.array(keep, dtype=int)
    return times[keep], midi_int[keep]


def _segment_notes(frame_times: np.ndarray, midi_int: np.ndarray) -> List[NoteEvent]:
    events: List[NoteEvent] = []
    if len(frame_times) == 0:
        return events

    current_pitch = int(midi_int[0])
    start_time = frame_times[0]
    prev_time = start_time

    for frame_time, pitch in zip(frame_times[1:], midi_int[1:]):
        gap = frame_time - prev_time
        if gap > GAP_THRESHOLD or pitch != current_pitch:
            events.append(NoteEvent(current_pitch, start_time, prev_time + FRAME_DURATION))
            current_pitch = int(pitch)
            start_time = frame_time
        prev_time = frame_time

    events.append(NoteEvent(current_pitch, start_time, prev_time + FRAME_DURATION))
    return events


def _remove_micro_notes(events: List[NoteEvent]) -> List[NoteEvent]:
    return [evt for evt in events if evt.duration >= MIN_NOTE]


def _merge_adjacent(events: List[NoteEvent]) -> List[NoteEvent]:
    if not events:
        return events
    merged: List[NoteEvent] = [events[0]]
    for evt in events[1:]:
        last = merged[-1]
        if evt.pitch == last.pitch and evt.start - last.end <= MERGE_GAP:
            merged[-1] = NoteEvent(last.pitch, last.start, evt.end)
        else:
            merged.append(evt)
    return merged


def build_midi(track: PitchTrack, destination: Path) -> Path:
    mask = track.finite_mask()
    if mask.sum() == 0:
        raise NoMelodyError("No stable monophonic melody detected.")

    freq = track.frequency[mask]
    time = track.time[mask]

    midi = librosa.hz_to_midi(freq)
    midi_int = np.rint(midi).astype(int)

    time, midi_int = _filter_jumps(time, midi_int)

    if len(midi_int) == 0:
        raise NoMelodyError("No stable monophonic melody detected.")

    note_candidates = _segment_notes(time, midi_int)
    note_candidates = _remove_micro_notes(note_candidates)
    note_candidates = _merge_adjacent(note_candidates)

    if not note_candidates:
        raise NoMelodyError("No stable monophonic melody detected.")

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    for note in note_candidates:
        instrument.notes.append(
            pretty_midi.Note(pitch=note.pitch, velocity=100, start=note.start, end=note.end)
        )
    pm.instruments.append(instrument)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(destination))
    return destination
