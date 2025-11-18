from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import librosa
import numpy as np
import pretty_midi

from .types import NoMelodyError, PitchTrack

MIN_NOTE_DURATION = 0.12
MIN_GAP = 0.04
MAX_OCTAVE_JUMP = 6
PITCH_BEND_THRESHOLD = 0.3  # semitones
PITCH_BEND_RANGE = 2.0  # semitones


@dataclass
class NoteEvent:
    pitch: int
    start: float
    end: float
    velocity: int
    source: str


@dataclass
class MidiExportResult:
    path: Path
    notes: List[NoteEvent]
    pitch_bends: List[pretty_midi.PitchBend]
    segmentation: List[dict]


def _frame_step(track: PitchTrack) -> float:
    if track.time.size < 2:
        return 0.01
    diffs = np.diff(track.time)
    return float(np.median(diffs))


def _dynamic_threshold(confidence: np.ndarray) -> float:
    voiced = confidence[confidence > 0]
    if voiced.size == 0:
        return 0.2
    percentile = np.percentile(voiced, 20)
    return float(max(0.2, percentile))


def _estimate_velocity(track: PitchTrack, frames: np.ndarray) -> int:
    if track.loudness is None:
        return 100
    values = track.loudness[frames]
    if values.size == 0:
        return 100
    weight = float(np.mean(values))
    return int(np.clip(20 + weight * 100, 20, 120))


def _apply_octave_correction(pitch: float, previous: int | None) -> int:
    midi_pitch = float(np.round(pitch))
    if previous is None:
        return int(midi_pitch)
    while midi_pitch - previous > MAX_OCTAVE_JUMP:
        midi_pitch -= 12
    while previous - midi_pitch > MAX_OCTAVE_JUMP:
        midi_pitch += 12
    return int(np.clip(midi_pitch, 0, 127))


def _pitch_bend_events(track: PitchTrack, frames: np.ndarray, target_pitch: int) -> List[pretty_midi.PitchBend]:
    if frames.size == 0:
        return []
    midi_curve = librosa.hz_to_midi(track.frequency[frames])
    if np.all(np.isnan(midi_curve)):
        return []
    diffs = midi_curve - target_pitch
    if np.nanmax(np.abs(diffs)) < PITCH_BEND_THRESHOLD:
        return []
    bends: List[pretty_midi.PitchBend] = []
    for idx, frame in enumerate(frames):
        delta = diffs[idx]
        if np.isnan(delta):
            continue

        normalized = float(np.clip(delta / PITCH_BEND_RANGE, -1.0, 1.0))
        scaled_value = int(round(normalized * 8192))
        value = int(np.clip(scaled_value, -8192, 8191))

        time_value = float(track.time[frame])
        if not np.isfinite(time_value):
            continue

        bends.append(
            pretty_midi.PitchBend(
                pitch=value,
                time=time_value,
            )
        )
    return bends


def _state_machine(track: PitchTrack) -> List[NoteEvent]:
    step = _frame_step(track)
    threshold = _dynamic_threshold(track.confidence)
    notes: List[NoteEvent] = []
    frames: List[int] = []
    last_state = "silence"

    for idx, (freq, conf) in enumerate(zip(track.frequency, track.confidence)):
        voiced = np.isfinite(freq) and conf >= threshold
        if voiced:
            frames.append(idx)
            if last_state == "silence":
                last_state = "onset"
            elif last_state == "onset" and len(frames) * step >= 0.03:
                last_state = "sustain"
        else:
            if frames:
                start = track.time[frames[0]]
                end = track.time[frames[-1]] + step
                if end - start >= MIN_NOTE_DURATION:
                    midi_vals = librosa.hz_to_midi(track.frequency[frames])
                    midi_vals = midi_vals[np.isfinite(midi_vals)]
                    if midi_vals.size:
                        pitch = np.median(midi_vals)
                        pitch = _apply_octave_correction(pitch, notes[-1].pitch if notes else None)
                        velocity = _estimate_velocity(track, np.array(frames))
                        source = str(track.sources[frames[0]]) if track.sources is not None else track.engine
                        notes.append(NoteEvent(pitch=pitch, start=start, end=end, velocity=velocity, source=source))
                frames = []
                last_state = "silence"
            else:
                last_state = "silence"

    if frames:
        start = track.time[frames[0]]
        end = track.time[frames[-1]] + step
        if end - start >= MIN_NOTE_DURATION:
            midi_vals = librosa.hz_to_midi(track.frequency[frames])
            midi_vals = midi_vals[np.isfinite(midi_vals)]
            if midi_vals.size:
                pitch = np.median(midi_vals)
                pitch = _apply_octave_correction(pitch, notes[-1].pitch if notes else None)
                velocity = _estimate_velocity(track, np.array(frames))
                source = str(track.sources[frames[0]]) if track.sources is not None else track.engine
                notes.append(NoteEvent(pitch=pitch, start=start, end=end, velocity=velocity, source=source))

    # Merge close notes with same pitch
    merged: List[NoteEvent] = []
    for note in notes:
        if merged and note.pitch == merged[-1].pitch and note.start - merged[-1].end <= MIN_GAP:
            merged[-1] = NoteEvent(
                pitch=note.pitch,
                start=merged[-1].start,
                end=note.end,
                velocity=max(note.velocity, merged[-1].velocity),
                source=note.source,
            )
        else:
            merged.append(note)

    return merged


def build_midi_advanced(
    track: PitchTrack, destination: Path, note_events: Sequence[NoteEvent] | None = None
) -> MidiExportResult:
    if track.finite_count() == 0:
        raise NoMelodyError("No stable monophonic melody detected.")

    if note_events is None:
        notes = _state_machine(track)
    else:
        notes = list(note_events)
    if not notes:
        raise NoMelodyError("No stable monophonic melody detected.")

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    pitch_bends: List[pretty_midi.PitchBend] = []
    segmentation: List[dict] = []

    for note in notes:
        midi_note = pretty_midi.Note(pitch=note.pitch, velocity=note.velocity, start=note.start, end=note.end)
        instrument.notes.append(midi_note)
        frames = np.where((track.time >= note.start) & (track.time <= note.end))[0]
        bends = _pitch_bend_events(track, frames, note.pitch)
        pitch_bends.extend(bends)
        segmentation.append(
            {
                "pitch": note.pitch,
                "start": note.start,
                "end": note.end,
                "velocity": note.velocity,
                "source": note.source,
                "bend_points": [b.time for b in bends],
            }
        )

    instrument.pitch_bends.extend(pitch_bends)
    pm.instruments.append(instrument)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(destination))

    return MidiExportResult(path=destination, notes=notes, pitch_bends=pitch_bends, segmentation=segmentation)
