from __future__ import annotations

from pathlib import Path

from .debug import write_debug_file
from .midi_advanced import MidiExportResult, NoteEvent, build_midi_advanced
from .midi_cleaner import clean_track_for_midi
from .types import PitchTrack


def build_midi(track: PitchTrack, destination: Path, debug_dir: Path | None = None) -> MidiExportResult:
    cleaned = clean_track_for_midi(track)
    write_debug_file(debug_dir, "midi_input_track.json", cleaned.to_payload(include_activation=True))
    result = build_midi_advanced(cleaned, destination)

    notes_payload = [
        {"pitch": note.pitch, "start": note.start, "end": note.end, "velocity": note.velocity, "source": note.source}
        for note in result.notes
    ]
    write_debug_file(debug_dir, "segmentation.json", result.segmentation)
    write_debug_file(debug_dir, "final_notes.json", notes_payload)

    return result


__all__ = ["build_midi", "MidiExportResult", "NoteEvent"]
