from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .debug import write_debug_file
from .melody_postprocess import MelodyNote, postprocess_pitch_track
from .midi_advanced import MidiExportResult, NoteEvent, build_midi_advanced
from .midi_cleaner import clean_track_for_midi
from .types import PitchTrack


def build_midi(
    track: PitchTrack,
    destination: Path,
    debug_dir: Path | None = None,
    notes: Sequence[MelodyNote] | None = None,
) -> MidiExportResult:
    cleaned = clean_track_for_midi(track)
    note_sequence = list(notes) if notes is not None else postprocess_pitch_track(cleaned)
    write_debug_file(debug_dir, "midi_input_track.json", cleaned.to_payload(include_activation=True))
    write_debug_file(debug_dir, "melody_notes_input.json", [note.to_payload() for note in note_sequence])
    note_events = _notes_to_events(cleaned, note_sequence)
    result = build_midi_advanced(cleaned, destination, note_events=note_events)

    notes_payload = [
        {"pitch": note.pitch, "start": note.start, "end": note.end, "velocity": note.velocity, "source": note.source}
        for note in result.notes
    ]
    write_debug_file(debug_dir, "segmentation.json", result.segmentation)
    write_debug_file(debug_dir, "final_notes.json", notes_payload)

    return result


def _notes_to_events(track: PitchTrack, notes: Sequence[MelodyNote]) -> list[NoteEvent]:
    events: list[NoteEvent] = []
    for note in notes:
        pitch = int(np.clip(round(note.midi), 0, 127))
        start = float(note.start_time)
        end = float(note.end_time)
        velocity = int(np.clip(20 + note.confidence * 100, 20, 120))
        events.append(
            NoteEvent(
                pitch=pitch,
                start=start,
                end=end,
                velocity=velocity,
                source=track.engine,
            )
        )
    return events


__all__ = ["build_midi", "MidiExportResult", "NoteEvent"]
